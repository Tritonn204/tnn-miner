#include "pearl_mining.hpp"
#include "pearl_share_audit.hpp"
#include <tnn_hip/common/gpu_algo.hpp>
#include <tnn_hip/common/gpu_submit_queue.hpp>
#include <tnn_hip/crypto/pearl/pearl_native.hpp>
#include <algo_definitions.h>
#include <tnn-common.hpp>
#include "tnn_hip_common_embedded.hpp"
#include "pearl_embedded_headers.hpp"
#include "pearl_iris_qualified_headers.hpp"
#include "pearl_iris_qualified_kernel.hpp"
#include "pearl_prepare_source.hpp"
#include <tnn_hip/common/gpu_rtc.hpp>

#include <bit>
#include <chrono>
#include <random>
#include <set>
#include <thread>
#include <deque>
#include <condition_variable>
#include <functional>

namespace tnn::pearl {
namespace {

using Clock = std::chrono::steady_clock;
static_assert(std::endian::native == std::endian::little);

void checked(oroError_t error, const char *operation) {
    if (error != oroSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + tnn_error_string(error));
    }
}

// Host mirror of the qualified 40/8/24-byte device result ABI.
struct DeviceWinner {
    uint32_t row, col, digest[8];
};
struct DeviceState {
    uint32_t total_hits, overflow;
};
struct DeviceResults {
    DeviceWinner *winners;
    DeviceState *state;
    uint32_t capacity;
};
static_assert(sizeof(DeviceWinner) == 40 && sizeof(DeviceState) == 8 &&
              sizeof(DeviceResults) == 24);
static_assert(offsetof(DeviceResults, state) == 8 && offsetof(DeviceResults, capacity) == 16);

struct Buffer {
    uint8_t *allocation = nullptr;
    size_t size = 0;
    static constexpr size_t guard_size = 256;

    Buffer() = default;
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    ~Buffer() {
        if (allocation)
            (void)oroFree(allocation);
    }

    void allocate(size_t bytes) {
        size = bytes;
        checked(
            oro_safe_malloc(reinterpret_cast<oroDeviceptr *>(&allocation), bytes + 2 * guard_size),
            "Pearl allocation");
        checked(oro_safe_memset(allocation, 0x5a, bytes + 2 * guard_size), "Pearl guards");
    }
    uint8_t *data() const {
        return allocation + guard_size;
    }
    void upload(const void *input, size_t bytes) {
        if (bytes != size)
            throw std::invalid_argument("Pearl upload length");
        checked(oro_safe_memcpy(data(), input, bytes, oroMemcpyHostToDevice), "Pearl upload");
    }
    void check_guards() const {
        std::array<uint8_t, guard_size> before, after;
        checked(oro_safe_memcpy(before.data(), allocation, guard_size, oroMemcpyDeviceToHost),
                "Pearl prefix guard");
        checked(oro_safe_memcpy(after.data(), data() + size, guard_size, oroMemcpyDeviceToHost),
                "Pearl suffix guard");
        for (size_t i = 0; i < guard_size; ++i) {
            if (before[i] != 0x5a || after[i] != 0x5a)
                throw std::runtime_error("Pearl device guard corruption");
        }
    }
};

struct PreparedMatrix {
    Buffer descriptor, base, tree, seed, dense, pairs, sample_rows, sampled;
    std::vector<size_t> offsets;
    uint32_t rows = 0, k = 0;

    void allocate(uint32_t row_count, uint32_t depth) {
        rows = row_count;
        k = depth;
        size_t bytes = 0;
        for (size_t count = size_t(rows) * k / 1024;; count = (count + 1) / 2) {
            offsets.push_back(bytes);
            bytes += count * 32;
            if (count == 1)
                break;
        }
        descriptor.allocate(sizeof(native::BaseOutput));
        base.allocate(size_t(rows) * k);
        tree.allocate(bytes);
        seed.allocate(32);
        dense.allocate(size_t(rows) * 128);
        pairs.allocate(size_t(k) * 8);
        sample_rows.allocate(64 * sizeof(uint32_t));
        sampled.allocate(64 * size_t(k));
    }
};

struct Preparation {
    oroModule_t module = nullptr; // Owned by RTCCompiler's module cache.
    KernelMap kernels;
    Buffer job_key;
    Buffer b_operand;
    PreparedMatrix b;
    std::array<PreparedMatrix, 2> a;
    native::Identity base_identity;
    bool ready = false;
    std::array<bool, 2> a_ready{};
    template <class... Args>
    void launch(const char *name, uint32_t count, oroStream_t stream, Args... args) {
        void *arguments[] = {&args...};
        checked(oroModuleLaunchKernel(kernels.at(name), (count + 127) / 128, 1, 1, 128, 1, 1, 0,
                                      stream, arguments, nullptr),
                name);
    }

    void initialize(native::Shape shape, int device) {
        auto compiled = RTCCompiler::instance().compile_from_source(
            std::string(
                hip_pearl_prepare_source::SRC_TNN_HIP_CRYPTO_PEARL_PEARL_PREPARE_HIP_SOURCE),
            "pearl_prepare.hip", "pearl_prepare_base", {"-O3", "-std=c++20"}, device);
        module = compiled.module;
        for (const auto *name :
             {"pearl_prepare_base", "pearl_prepare_leaves", "pearl_prepare_parents",
              "pearl_prepare_seed", "pearl_prepare_dense", "pearl_prepare_sparse",
              "pearl_prepare_materialize", "pearl_prepare_gather", "pearl_prepare_zero_fused",
              "pearl_prepare_incremental_seed"}) {
            oroFunction_t function;
            checked(oroModuleGetFunction(&function, module, name), "Preparation entry point");
            kernels[name] = function;
        }
        job_key.allocate(32);
        b.allocate(shape.n, shape.k);
        b_operand.allocate(size_t(shape.n) * shape.k);
        for (auto &slot : a)
            slot.allocate(shape.m, shape.k);
    }

    void matrix(PreparedMatrix &matrix, Buffer &operand, const native::BaseOutput &output,
                uint8_t *seed_left, bool is_a, oroStream_t stream, bool rebuild = true,
                uint64_t nonce = 0, uint64_t nonce_high = 0) {
        uint32_t bytes = matrix.rows * matrix.k;
        if (rebuild) {
            if (is_a) {
                checked(oroMemsetAsync(matrix.base.data(), 0, bytes, stream),
                        "Initialize zero base");
            } else {
                matrix.descriptor.upload(&output, sizeof(output));
                launch("pearl_prepare_base", bytes / 64, stream, matrix.descriptor.data(),
                       matrix.base.data(), bytes);
            }
            uint32_t count = bytes / 1024;
            launch("pearl_prepare_leaves", count, stream, matrix.base.data(), job_key.data(),
                   matrix.tree.data(), count);
            for (size_t level = 1; level < matrix.offsets.size(); ++level) {
                launch("pearl_prepare_parents", (count + 1) / 2, stream,
                       matrix.tree.data() + matrix.offsets[level - 1], job_key.data(),
                       matrix.tree.data() + matrix.offsets[level], count);
                count = (count + 1) / 2;
            }
        }
        if (is_a) {
            launch("pearl_prepare_incremental_seed", 1, stream, matrix.base.data(), job_key.data(),
                   matrix.tree.data(), bytes / 1024, 0u, nonce, nonce_high, seed_left,
                   matrix.seed.data(), base_identity.cert_version, matrix.rows);
            launch("pearl_prepare_sparse", matrix.k / 8, stream, matrix.seed.data(),
                   matrix.pairs.data(), matrix.k, 1u);
            launch("pearl_prepare_zero_fused", matrix.rows * 4, stream, matrix.base.data(),
                   matrix.seed.data(), matrix.pairs.data(), operand.data(), matrix.rows, matrix.k);
            return;
        }
        launch("pearl_prepare_seed", 1, stream, seed_left,
               matrix.tree.data() + matrix.offsets.back(), matrix.seed.data(),
               base_identity.cert_version, matrix.rows, 0u);
        launch("pearl_prepare_dense", matrix.rows * 4, stream, matrix.seed.data(),
               matrix.dense.data(), matrix.rows, 0u);
        launch("pearl_prepare_sparse", matrix.k / 8, stream, matrix.seed.data(),
               matrix.pairs.data(), matrix.k, uint32_t(is_a));
        launch("pearl_prepare_materialize", bytes, stream, matrix.base.data(), matrix.dense.data(),
               matrix.pairs.data(), operand.data(), matrix.rows, matrix.k, 0u);
    }

    void verify(const PreparedMatrix &matrix, const Buffer &operand,
                const native::MatrixTree &reference, const native::Digest &seed,
                const std::vector<int8_t> &materialized) {
        const auto compare = [](const uint8_t *source, const void *expected, size_t size) {
            std::vector<uint8_t> bytes(size);
            checked(oro_safe_memcpy(bytes.data(), source, size, oroMemcpyDeviceToHost),
                    "Read preparation check");
            if (std::memcmp(bytes.data(), expected, size))
                throw std::runtime_error("GPU preparation differs from native reference");
        };
        compare(matrix.base.data(), reference.bytes().data(), reference.bytes().size());
        for (size_t level = 0; level < reference.layers().size(); ++level) {
            compare(matrix.tree.data() + matrix.offsets[level], reference.layers()[level].data(),
                    reference.layers()[level].size() * 32);
        }
        compare(matrix.seed.data(), seed.data(), 32);
        compare(operand.data(), materialized.data(), materialized.size());
        for (const auto *buffer : {&matrix.descriptor, &matrix.base, &matrix.tree, &matrix.seed,
                                   &matrix.dense, &matrix.pairs})
            buffer->check_guards();
    }
};

struct State {
    oroStream_t compute_stream = nullptr;
    oroEvent_t readback_done = nullptr;
    uint8_t *host_transfer = nullptr;
    native::Digest uploaded_target{};
    bool target_valid = false;

    ~State() {
        // Drain before freeing staging or any member-owned device buffers,
        // including exception paths during initialization or submission.
        if (compute_stream)
            (void)oroStreamSynchronize(compute_stream);
        if (readback_done)
            (void)oroEventDestroy(readback_done);
        if (host_transfer)
            (void)oroHostFree(host_transfer);
        if (compute_stream)
            (void)oroStreamDestroy(compute_stream);
    }
    explicit State(const ExecutionOptions &options)
        : shape(options.shape), capacity(options.winner_capacity),
          validation(options.mode == ExecutionMode::Validation) {
    }
    const native::Shape shape;
    const uint32_t capacity;
    const bool validation;
    int device = 0;
    unsigned launches = 0;
    unsigned completed = 0;
    double prepare_host_ms = 0, completion_host_ms = 0, collect_host_ms = 0;
    Clock::time_point first_work{}, last_work{};
    native::Digest session_seed{};
    std::string previous_job;
    std::shared_ptr<const native::Attempt> attempt;
    native::Identity current_identity;
    std::array<native::Identity, 2> logged_target_identity;
    // User and dev jobs retain separate dependency caches across fee switches.
    std::array<std::unique_ptr<Preparation>, 2> preparation;
    unsigned slot = 0;
    Buffer alternate_a;
    Buffer a, d, target, winners, result;
};

struct Evidence : GPUBatchEvidence {
    std::shared_ptr<const native::Attempt> attempt;
    std::vector<native::Winner> winners;
    native::Identity identity;
    native::Shape shape;
    struct Samples {
        std::vector<uint32_t> a_rows, b_rows;
        std::vector<uint8_t> a_bytes, b_bytes;
    };
    std::vector<uint8_t> a_tree, b_tree;
    std::vector<Samples> samples;
};

bool prepare(const KernelLaunchContext &context) {
    auto &state = *static_cast<State *>(context.algo_data);
    const auto shape = state.shape;
    const auto *job = context.job_snapshot;
    if (!job || job->work_template.size() != 76 || job->connection_generation == 0) {
        throw std::invalid_argument("Pearl requires a complete atomic job snapshot");
    }
    const auto preparation_started = Clock::now();
    if (state.first_work == Clock::time_point{})
        state.first_work = preparation_started;
    native::Identity identity;
    identity.job_id = job->job_id_str;
    identity.connection_generation = job->connection_generation;
    identity.attempt_id = state.launches;
    identity.device_id = state.device;
    identity.is_dev = job->is_dev;
    std::copy(job->work_template.begin(), job->work_template.end(), identity.header.begin());
    identity.wire_target_le = job->raw_target;
    identity.target_le = native::jackpot_target(identity.wire_target_le, shape.k);
    auto &logged_target = state.logged_target_identity[identity.is_dev];
    if (logged_target.job_id != identity.job_id ||
        logged_target.wire_target_le != identity.wire_target_le ||
        logged_target.connection_generation != identity.connection_generation) {
        TNN_LOG_DEBUG("[PEARL-TARGET] job=%s dev=%d K=%u rank=128 work_per_jackpot=%llu (wire "
                      "target scaled once)\n",
                      identity.job_id.c_str(), int(identity.is_dev), shape.k,
                      static_cast<unsigned long long>(native::jackpot_work(shape.k)));
        logged_target = identity;
    }
    identity.cert_version = job->pearl_cert_version;
    state.current_identity = identity;
    {
        auto &preparation = *state.preparation[identity.is_dev];
        const bool new_job =
            !preparation.ready || preparation.base_identity.header != identity.header ||
            preparation.base_identity.cert_version != identity.cert_version ||
            preparation.base_identity.connection_generation != identity.connection_generation ||
            preparation.base_identity.is_dev != identity.is_dev;
        if (new_job) {
            preparation.base_identity = identity;
            preparation.a_ready.fill(false);
            const auto key = native::job_key(identity.header, shape);
            preparation.job_key.upload(key.data(), key.size());
        }
        auto base_identity = preparation.base_identity;
        base_identity.attempt_id = 0;
        state.slot = state.launches % 2;
        auto &operand = state.slot ? state.alternate_a : state.a;
        const bool rebuild_a = !preparation.a_ready[state.slot];
        uint64_t nonce_high = 0;
        std::memcpy(&nonce_high, state.session_seed.data(), sizeof(nonce_high));
        if (new_job)
            preparation.matrix(preparation.b, preparation.b_operand,
                               native::base_output(state.session_seed, base_identity, shape, false),
                               preparation.job_key.data(), false, context.stream);
        preparation.matrix(preparation.a[state.slot], operand,
                           native::base_output(state.session_seed, base_identity, shape, true),
                           preparation.b.seed.data(), true, context.stream, rebuild_a,
                           uint64_t(identity.attempt_id) + 1, nonce_high);
        preparation.a_ready[state.slot] = true;
        preparation.ready = true;
        if (state.validation) {
            auto base_a = std::vector<uint8_t>(size_t(shape.m) * shape.k, 0);
            {
                const uint64_t nonce = uint64_t(identity.attempt_id) + 1;
                for (unsigned i = 0; i < 19; ++i) {
                    unsigned value = 0;
                    for (unsigned j = 0; j < 7 && i * 7 + j < 128; ++j) {
                        const unsigned bit = i * 7 + j;
                        value |= unsigned((bit < 64 ? nonce >> bit : nonce_high >> (bit - 64)) & 1)
                                 << j;
                    }
                    base_a[i] = uint8_t(int(value) - 64);
                }
            }
            state.attempt = std::make_shared<native::Attempt>(
                identity, shape, std::move(base_a),
                native::fresh_base(state.session_seed, base_identity, shape, false));
        }
        if (!state.target_valid || state.uploaded_target != identity.target_le) {
            std::memcpy(state.host_transfer, identity.target_le.data(), 32);
            checked(oroMemcpyAsync(state.target.data(), state.host_transfer, 32,
                                   oroMemcpyHostToDevice, context.stream),
                    "Enqueue Pearl target");
            state.uploaded_target = identity.target_le;
            state.target_valid = true;
        }
        checked(oroMemsetAsync(state.result.data(), 0, sizeof(DeviceState), context.stream),
                "Enqueue Pearl reset");
    }
    return true;
}

bool execute(const KernelMap &kernels, const KernelLaunchContext &context) {
    auto &state = *static_cast<State *>(context.algo_data);
    const auto shape = state.shape;
    const auto capacity = state.capacity;
    if (context.block_size != 256 || context.batch_size != 1) {
        throw std::runtime_error("Pearl launch contract");
    }
    auto a = state.slot ? state.alternate_a.data() : state.a.data();
    auto b = state.preparation[state.current_identity.is_dev]->b_operand.data();
    auto d = state.d.data();
    auto key = state.preparation[state.current_identity.is_dev]->a[state.slot].seed.data();
    auto target = state.target.data();
    unsigned m = shape.m, n = shape.n, k = shape.k;
    unsigned lda = m, ldb = k, ldd = m;
    uint32_t *diagnostic = nullptr;
    DeviceResults results{reinterpret_cast<DeviceWinner *>(state.winners.data()),
                          reinterpret_cast<DeviceState *>(state.result.data()), capacity};
    void *arguments[] = {&a,   &b,   &d,   &m,      &n,       &k,         &lda,
                         &ldb, &ldd, &key, &target, &results, &diagnostic};
    ++state.launches;
    state.previous_job = state.current_identity.job_id;
    checked(oroModuleLaunchKernel(kernels.at("pearl_iris_fused"), (shape.m / 128) * (shape.n / 256),
                                  1, 1, 256, 1, 1, 0, context.stream, arguments, nullptr),
            "Launch Pearl fused");
    return true;
}

void collect(const KernelLaunchContext &context, BatchResult &batch) {
    const auto collect_begin = Clock::now();
    auto &state = *static_cast<State *>(context.algo_data);
    const auto shape = state.shape;
    const auto capacity = state.capacity;
    if (context.elapsed_ms > 250)
        throw std::runtime_error("Pearl kernel exceeded 250 ms; stopped");
    if (context.completion_host_ms > 250)
        throw std::runtime_error("Pearl GPU completion exceeded 250 ms; stopped");
    if (state.validation) {
        auto &preparation = *state.preparation[state.current_identity.is_dev];
        auto &operand = state.slot ? state.alternate_a : state.a;
        preparation.verify(preparation.b, preparation.b_operand, state.attempt->bt_tree,
                           state.attempt->b_seed, state.attempt->bt);
        preparation.verify(preparation.a[state.slot], operand, state.attempt->a_tree,
                           state.attempt->a_seed, state.attempt->a);
        TNN_LOG_DEBUG("[PEARL-PREP-VERIFIED] all bytes/levels/seeds match after GEMM\n");
    }
    if (state.validation || state.launches % 128 == 0) {
        for (const auto *buffer : {&state.a, &state.d, &state.target,
                                   &state.winners, &state.result})
            buffer->check_guards();
        state.alternate_a.check_guards();
    }
    DeviceState result{};
    std::memcpy(&result, state.host_transfer + 32, sizeof(result));
    if (result.overflow || result.total_hits > capacity) {
        throw std::runtime_error("Pearl winner capacity exceeded; increase pool difficulty (no "
                                 "target tightening or silent candidate drops)");
    }
    std::vector<DeviceWinner> winners(result.total_hits);
    if (!winners.empty()) {
        checked(oro_safe_memcpy(winners.data(), state.winners.data(),
                                winners.size() * sizeof(DeviceWinner), oroMemcpyDeviceToHost),
                "Read Pearl winners");
    }
    auto evidence = std::make_shared<Evidence>();
    evidence->attempt = state.attempt;
    evidence->identity = state.current_identity;
    evidence->shape = shape;
    std::set<std::pair<uint32_t, uint32_t>> positions;
    for (const auto &source : winners) {
        if (source.row >= shape.m - 1 || source.row % 2 || source.col >= shape.n - 238 ||
            (source.col & 238) || !positions.emplace(source.row, source.col).second) {
            throw std::runtime_error("Invalid/duplicate Pearl winner coordinates");
        }
        native::Winner winner{source.row, source.col, {}};
        std::memcpy(winner.digest.data(), source.digest, 32);
        if (!native::meets_target(winner.digest, state.current_identity.target_le)) {
            throw std::runtime_error("Pearl GPU returned a winner above target");
        }
        evidence->winners.push_back(winner);
    }
    if (!evidence->winners.empty()) {
        auto &preparation = *state.preparation[state.current_identity.is_dev];
        auto &a = preparation.a[state.slot];
        auto &b = preparation.b;
        const auto copy = [](const uint8_t *device, size_t size) {
            std::vector<uint8_t> bytes(size);
            checked(oro_safe_memcpy(bytes.data(), device, size, oroMemcpyDeviceToHost),
                    "Copy owned proof evidence");
            return bytes;
        };
        evidence->a_tree = copy(a.tree.data(), a.tree.size);
        evidence->b_tree = copy(b.tree.data(), b.tree.size);
        const size_t count = evidence->winners.size();
        for (size_t index = 0; index < count; ++index) {
            const auto &winner = evidence->winners[index];
            Evidence::Samples samples;
            samples.a_rows = {winner.row, winner.row + 1};
            for (uint32_t col = 0; col < 64; ++col)
                samples.b_rows.push_back(winner.col + col / 8 * 32 + col % 8 * 2);
            const auto gather = [&](PreparedMatrix &matrix, const std::vector<uint32_t> &rows) {
                checked(oro_safe_memcpy(matrix.sample_rows.data(), rows.data(), rows.size() * 4,
                                        oroMemcpyHostToDevice),
                        "Upload sample rows");
                preparation.launch("pearl_prepare_gather", uint32_t(rows.size()) * shape.k,
                                   context.stream, matrix.base.data(), matrix.sample_rows.data(),
                                   matrix.sampled.data(), uint32_t(rows.size()), shape.k);
                checked(oroStreamSynchronize(context.stream), "Complete sample gather");
                return copy(matrix.sampled.data(), rows.size() * shape.k);
            };
            samples.a_bytes = gather(a, samples.a_rows);
            samples.b_bytes = gather(b, samples.b_rows);
            evidence->samples.push_back(std::move(samples));
        }
    }
    batch.count = 1;
    ++state.completed;
    state.prepare_host_ms += context.prepare_host_ms;
    state.completion_host_ms += context.completion_host_ms;
    state.collect_host_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - collect_begin).count();
    state.last_work = Clock::now();
    batch.work_multiplier = shape.macs();
    batch.evidence = std::move(evidence);
    if (state.validation || state.launches % 128 == 0 || result.total_hits)
        TNN_LOG_DEBUG(
            "[PEARL-WALL-BATCH] launch=%u job=%s winners=%u completion_ms=%.3f MACs=%llu\n",
            state.launches, state.previous_job.c_str(), result.total_hits,
            context.completion_host_ms, static_cast<unsigned long long>(shape.macs()));
}

} // namespace

AlgoConfig pearl_gpu_config(ExecutionOptions options) {
    options.shape.validate();
    if (!options.winner_capacity)
        throw std::invalid_argument("Pearl winner capacity is zero");
    AlgoConfig config{};
    config.rate_unit = RateUnit::MultiplyAccumulates;
    config.name = "pearl";
    config.algo_id = ALGO_PEARL_POUW;
    config.source_path = "src/tnn_hip/crypto/iris/gemm/qualified/rtc.hip";
    config.source =
        hip_pearl_iris_qualified_source::SRC_TNN_HIP_CRYPTO_IRIS_GEMM_QUALIFIED_RTC_HIP_SOURCE;
    config.kernel_names = {"pearl_iris_fused", "pearl_iris_raw", "pearl_iris_diagnostic"};
    config.rtc_headers =
        build_rtc_headers(hip_embedded::COMMON_HEADERS, hip_embedded::PEARL_HEADERS,
                          hip_embedded::PEARL_IRIS_QUALIFIED_HEADERS);
    // HIPRTC virtual includes need the same suffix aliases used by the
    // compile-only qualification (api.hpp, hiprtc_types.hip.h, etc.).
    const size_t header_count = config.rtc_headers.size();
    for (size_t i = 0; i < header_count; ++i) {
        const auto header = config.rtc_headers[i];
        for (size_t slash = header.name.find('/'); slash != std::string_view::npos;
             slash = header.name.find('/', slash + 1)) {
            config.rtc_headers.push_back({header.name.substr(slash + 1), header.source});
        }
    }
    config.template_size = 76;
    config.hash_size = 32;
    config.nonce_size = 0;
    config.scratch_per_hash = 0;
    config.allocate_output_buffer = false;
    config.allocate_scratch_buffer = false;
    config.owns_work_and_result_buffers = true;
    config.host_timing_only = true;
    config.preferred_block_size = 256;
    config.enable_autotune = false;
    config.enable_reg_tuning = false;
    config.skip_cached_tune_validation = true;
    TuningResult fixed{};
    fixed.block_size = 256;
    fixed.num_blocks = (options.shape.m / 128) * (options.shape.n / 256);
    fixed.batch_size = 1;
    fixed.valid = true;
    config.fixed_launch = fixed;
    config.pre_tune_fn = [options](const KernelMap &, const oroDeviceProp_t &props, int device,
                                   void **output) {
        if (!tnn_is_amd_device(device) || parse_gfx_number(props.gcnArchName) != 1100) {
            throw std::runtime_error("Pearl mining currently supports one AMD gfx1100 GPU");
        }
        auto state = std::make_unique<State>(options);
        {
            checked(oroStreamCreateWithFlags(&state->compute_stream, oroStreamNonBlocking),
                    "Pearl compute stream");
            checked(oroEventCreateWithFlags(&state->readback_done, oroEventBlockingSync),
                    "Pearl readback event");
            checked(oroHostMalloc(reinterpret_cast<void **>(&state->host_transfer),
                                  32 + sizeof(DeviceState), 0),
                    "Pearl pinned staging");
        }
        const auto shape = state->shape;
        const auto capacity = state->capacity;
        state->device = device;
        std::random_device random;
        for (auto &byte : state->session_seed)
            byte = uint8_t(random());
        // Full-size buffers stay below 800 MiB, including both channel caches.
        state->a.allocate(size_t(shape.m) * shape.k);
        state->d.allocate(size_t(shape.m) * shape.n * 4);
        state->target.allocate(32);
        state->winners.allocate(capacity * sizeof(DeviceWinner));
        state->result.allocate(sizeof(DeviceState));
        {
            state->alternate_a.allocate(size_t(shape.m) * shape.k);
            for (auto &channel : state->preparation) {
                channel = std::make_unique<Preparation>();
                channel->initialize(shape, device);
            }
        }
        *output = state.release();
        return true;
    };
    config.algo_data_cleanup_fn = [](void *pointer) {
        auto *state = static_cast<State *>(pointer);
        if (state->completed) {
            const double seconds =
                std::chrono::duration<double>(state->last_work - state->first_work).count();
            const double macs = double(state->completed) * state->shape.macs();
            TNN_LOG_DEBUG(
                "[PEARL-WALL-SUMMARY] completed=%u active_wall_s=%.6f active_wall_TMACs=%.3f "
                "prepare_host_ms=%.3f completion_host_ms=%.3f collect_host_ms=%.3f\n",
                state->completed, seconds, macs / (seconds * 1e12), state->prepare_host_ms,
                state->completion_host_ms, state->collect_host_ms);
        }
        delete state;
    };
    config.prepare_batch_fn = prepare;
    config.execute_fn = execute;
    config.collect_batch_fn = collect;
    {
        config.batch_stream_fn = [](void *pointer) {
            return static_cast<State *>(pointer)->compute_stream;
        };
        config.finish_batch_fn = [](const KernelLaunchContext &context) {
            auto &state = *static_cast<State *>(context.algo_data);
            checked(oroMemcpyAsync(state.host_transfer + 32, state.result.data(),
                                   sizeof(DeviceState), oroMemcpyDeviceToHost, context.stream),
                    "Enqueue Pearl result header");
            checked(oroEventRecord(state.readback_done, context.stream), "Record Pearl readback");
            checked(oroEventSynchronize(state.readback_done), "Complete Pearl readback");
        };
    }
    return config;
}

static std::vector<GPUSubmitEntry> build_one(const std::shared_ptr<const Evidence> &evidence,
                                             int device, const JobSnapshot &job,
                                             const native::Winner &winner) {
    const auto &attempt = *evidence->attempt;
    if (attempt.identity.device_id != device || attempt.identity.job_id != job.job_id_str ||
        attempt.identity.connection_generation != job.connection_generation ||
        attempt.identity.is_dev != job.is_dev ||
        attempt.identity.wire_target_le != job.raw_target ||
        attempt.identity.target_le != native::jackpot_target(job.raw_target, attempt.shape.k) ||
        attempt.identity.cert_version != job.pearl_cert_version ||
        !std::equal(attempt.identity.header.begin(), attempt.identity.header.end(),
                    job.work_template.begin(), job.work_template.end())) {
        throw std::runtime_error("Pearl completed batch/job binding mismatch");
    }
    const auto proof_started = Clock::now();
    auto proof = attempt.proof(winner);
    ++share_audit.proofs_built;
    TNN_LOG_DEBUG("[PEARL-PROOF] attempt=%llu proof_ms=%.3f additional_winners=%llu\n",
                  static_cast<unsigned long long>(attempt.identity.attempt_id),
                  std::chrono::duration<double, std::milli>(Clock::now() - proof_started).count(),
                  static_cast<unsigned long long>(evidence->winners.size() - 1));
    constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve((proof.size() + 2) / 3 * 4);
    for (size_t i = 0; i < proof.size(); i += 3) {
        uint32_t value = uint32_t(proof[i]) << 16;
        if (i + 1 < proof.size())
            value |= uint32_t(proof[i + 1]) << 8;
        if (i + 2 < proof.size())
            value |= proof[i + 2];
        encoded += alphabet[(value >> 18) & 63];
        encoded += alphabet[(value >> 12) & 63];
        encoded += i + 1 < proof.size() ? alphabet[(value >> 6) & 63] : '=';
        encoded += i + 2 < proof.size() ? alphabet[value & 63] : '=';
    }
    auto hex = [](auto bytes) {
        constexpr char digits[] = "0123456789abcdef";
        std::string result;
        for (uint8_t byte : bytes) {
            result += digits[byte >> 4];
            result += digits[byte & 15];
        }
        return result;
    };
    auto target_be = job.raw_target;
    std::reverse(target_be.begin(), target_be.end());
    boost::json::object binding{{"job_id", job.job_id_str},
                                {"header", hex(job.work_template)},
                                {"target", hex(target_be)},
                                {"height", job.block_height},
                                {"connection_generation", job.connection_generation},
                                {"cert_version", job.pearl_cert_version}};
    boost::json::object payload{
        {"method", "mining.submit"},
        {"_pearl_job", std::move(binding)},
        {"_pearl_evidence", "attempt-" + std::to_string(attempt.identity.attempt_id) + "-" +
                                std::to_string(winner.row) + "-" + std::to_string(winner.col)},
        {"_pearl_digest", hex(winner.digest)},
        {"_pearl_device", device},
        {"params",
         boost::json::object{{"job_id", job.job_id_str}, {"plain_proof", std::move(encoded)}}}};
    return {{std::move(payload), job.is_dev, job.job_id}};
}

namespace {
class ProofWorker {
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<std::function<void()>> queue;
    std::thread worker;
    bool stopping = true;
    std::exception_ptr failure;

  public:
    void start() {
        std::lock_guard lock(mutex);
        stopping = false;
        failure = nullptr;
        worker = std::thread([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock lock(mutex);
                    wake.wait(lock, [&] { return stopping || !queue.empty(); });
                    if (stopping)
                        return;
                    task = std::move(queue.front());
                    queue.pop_front();
                    wake.notify_all();
                }
                try {
                    task();
                } catch (const std::exception &error) {
                    std::lock_guard lock(mutex);
                    failure = std::current_exception();
                    worker_failed = true;
                    fflush(stdout);
                    TNN_LOG_ERROR("\n[PEARL-ERROR] proof worker: %s; stopping\n", error.what());
                    wake.notify_all();
                    return;
                }
            }
        });
    }
    void push(std::function<void()> task) {
        std::unique_lock lock(mutex);
        if (queue.size() == 8)
            TNN_LOG_DEBUG("[PEARL-PROOF-BACKPRESSURE] queue full\n");
        while (queue.size() >= 8 && !stopping && !failure && !ABORT_MINER) {
            wake.wait_for(lock, std::chrono::milliseconds(10));
        }
        if (failure)
            std::rethrow_exception(failure);
        if (stopping || ABORT_MINER)
            return;
        queue.push_back(std::move(task));
        wake.notify_all();
    }
    bool submit(GPUSubmitEntry entry) {
        auto &submissions = GPUSubmitQueue::instance();
        std::unique_lock lock(mutex);
        if (submissions.pending() >= 8)
            TNN_LOG_DEBUG("[PEARL-SUBMIT-BACKPRESSURE] queue full\n");
        while (!stopping && !ABORT_MINER && submissions.pending() >= 8) {
            wake.wait_for(lock, std::chrono::milliseconds(10));
        }
        if (stopping || ABORT_MINER) {
            TNN_LOG_DEBUG("[PEARL-PROOF-CANCEL] shutdown cancels active batch\n");
            return false;
        }
        // Pearl has one producer; the drain thread can only reduce this bound.
        submissions.push(std::move(entry));
        return true;
    }
    void stop() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
            if (!queue.empty())
                TNN_LOG_DEBUG("[PEARL-PROOF-CANCEL] shutdown cancels %zu pending batches\n",
                              queue.size());
            queue.clear();
            wake.notify_all();
        }
        if (worker.joinable())
            worker.join();
    }
    ~ProofWorker() {
        stop();
    }
};
ProofWorker proofs;
} // namespace

void pearl_start_proofs() {
    worker_failed = false;
    proofs.start();
}
void pearl_stop_proofs() {
    proofs.stop();
}

size_t pearl_validate_batch(const BatchResult &batch, bool verify_all_positions) {
    const auto evidence = std::dynamic_pointer_cast<const Evidence>(batch.evidence);
    if (!evidence || !evidence->attempt)
        throw std::runtime_error("Missing validation reference");
    const auto &reference = *evidence->attempt;
    std::set<std::pair<uint32_t, uint32_t>> found;
    for (const auto &winner : evidence->winners) {
        if (winner.digest != reference.winner_digest(winner.row, winner.col))
            throw std::runtime_error("Pearl GPU jackpot differs from CPU reference");
        found.emplace(winner.row, winner.col);
    }
    if (verify_all_positions) {
        for (uint32_t row = 0; row < reference.shape.m; row += 2) {
            for (uint32_t col = 0; col + 238 < reference.shape.n; ++col) {
                if (col & 238)
                    continue;
                const bool expected = native::meets_target(reference.winner_digest(row, col),
                                                           reference.identity.target_le);
                if (expected != found.contains({row, col}))
                    throw std::runtime_error("Pearl candidate coverage mismatch");
            }
        }
    }
    // Check both ends of the captured batch through the same snapshot importer
    // used by the proof worker, without starting threads or a network connection.
    if (!evidence->winners.empty()) {
        for (size_t index : {size_t(0), evidence->winners.size() - 1}) {
            const auto &samples = evidence->samples.at(index);
            native::Attempt imported(evidence->identity, evidence->shape,
                                     native::MatrixTree::from_snapshot(
                                         evidence->shape.m, evidence->shape.k, samples.a_rows,
                                         samples.a_bytes, evidence->a_tree, reference.job_key),
                                     native::MatrixTree::from_snapshot(
                                         evidence->shape.n, evidence->shape.k, samples.b_rows,
                                         samples.b_bytes, evidence->b_tree, reference.job_key));
            const auto &winner = evidence->winners[index];
            if (imported.proof(winner) != reference.proof(winner))
                throw std::runtime_error("Pearl captured proof differs from CPU reference");
        }
    }
    return found.size();
}

std::vector<GPUSubmitEntry> pearl_build_batch(const BatchResult &batch, int device,
                                              const JobSnapshot &job) {
    auto evidence = std::dynamic_pointer_cast<const Evidence>(batch.evidence);
    if (!evidence || evidence->winners.empty())
        return {};
    struct PendingWinners {
        size_t remaining;
        explicit PendingWinners(size_t count) : remaining(count) {
            share_audit.discovered += count;
        }
        ~PendingWinners() {
            share_audit.cancelled += remaining;
        }
    };
    auto pending = std::make_shared<PendingWinners>(evidence->winners.size());
    proofs.push([evidence, device, job, pending] {
        try {
            auto owned = std::make_shared<Evidence>(*evidence);
            const auto key = native::job_key(owned->identity.header, owned->shape);
            for (size_t index = 0; index < owned->samples.size(); ++index) {
                auto &samples = owned->samples[index];
                owned->attempt = std::make_shared<native::Attempt>(
                    owned->identity, owned->shape,
                    native::MatrixTree::from_snapshot(owned->shape.m, owned->shape.k,
                                                      samples.a_rows, samples.a_bytes,
                                                      owned->a_tree, key),
                    native::MatrixTree::from_snapshot(owned->shape.n, owned->shape.k,
                                                      samples.b_rows, samples.b_bytes,
                                                      owned->b_tree, key));
                const auto &winner = owned->winners[index];
                for (auto &entry : build_one(owned, device, job, winner)) {
                    if (!proofs.submit(std::move(entry)))
                        return;
                    --pending->remaining;
                }
            }
        } catch (...) {
            share_audit.failed += pending->remaining;
            pending->remaining = 0;
            throw;
        }
    });
    return {};
}

} // namespace tnn::pearl
