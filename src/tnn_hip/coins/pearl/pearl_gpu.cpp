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
#include "pearl_rdna4_headers.hpp"
#include "pearl_rdna4_kernel.hpp"
#include "pearl_prepare_source.hpp"
#include "pearl_batch_prepare_source.hpp"
#include "iris_embedded_headers.hpp"
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

// Host mirrors of the kernel's winner record and per-attempt counters.
struct DeviceWinner {
    uint32_t row, col, digest[8];
};
struct DeviceState {
    uint32_t total_hits, overflow;
};
static_assert(sizeof(DeviceWinner) == 40 && sizeof(DeviceState) == 8);

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
    Buffer base, tree, seed, dense, pairs, sample_rows, sampled;
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
    Buffer template_tree;
    std::vector<uint8_t> template_tree_cpu;
    std::vector<size_t> template_offsets;
    native::Identity base_identity;
    bool ready = false;
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
                hip_pearl_prepare_source::SRC_TNN_HIP_CRYPTO_PEARL_PEARL_PREPARE_HIP_SOURCE) +
                std::string(hip_pearl_batch_prepare_source::
                                SRC_TNN_HIP_CRYPTO_PEARL_PEARL_BATCH_PREPARE_HIP_SOURCE),
            "pearl_prepare.hip", "pearl_prepare_base",
            {"-O3", "-std=c++20",
             "-DPEARL_BASE_VALUE=" + std::to_string(native::mining_base_value)}, device);
        module = compiled.module;
        for (const auto *name :
             {"pearl_prepare_leaves", "pearl_prepare_parents",
              "pearl_prepare_seed", "pearl_prepare_dense", "pearl_prepare_sparse",
              "pearl_prepare_materialize", "pearl_prepare_gather", "pearl_batch_paths",
              "pearl_batch_sparse", "pearl_batch_materialize"}) {
            oroFunction_t function;
            checked(oroModuleGetFunction(&function, module, name), "Preparation entry point");
            kernels[name] = function;
        }
        job_key.allocate(32);
        b.allocate(shape.n, shape.k);
        b_operand.allocate(size_t(shape.n) * shape.k);
        size_t tree_bytes = 0;
        for (size_t nodes = size_t(shape.m) * shape.k / 1024;; nodes = (nodes + 1) / 2) {
            tree_bytes += nodes * 32;
            if (nodes == 1)
                break;
        }
        template_tree.allocate(tree_bytes);
    }

    // B is job-stable. Unlike fresh A, it is built once when the channel's
    // immutable job identity changes, not once per GEMM or batch.
    void prepare_b(oroStream_t stream) {
        const uint32_t bytes = b.rows * b.k;

        // The base is constant, not the GEMM operand: the job-bound seed and
        // required noise below still change with each immutable job identity.
        checked(oroMemsetAsync(b.base.data(), native::mining_base_value, bytes, stream),
                "Initialize Pearl B base");

        uint32_t count = bytes / 1024;
        launch("pearl_prepare_leaves", count, stream, b.base.data(), job_key.data(), b.tree.data(),
               count);

        for (size_t level = 1; level < b.offsets.size(); ++level) {
            launch("pearl_prepare_parents", (count + 1) / 2, stream,
                   b.tree.data() + b.offsets[level - 1], job_key.data(),
                   b.tree.data() + b.offsets[level], count);
            count = (count + 1) / 2;
        }

        launch("pearl_prepare_seed", 1, stream, job_key.data(), b.tree.data() + b.offsets.back(),
               b.seed.data(), base_identity.cert_version, b.rows, 0u);

        launch("pearl_prepare_dense", b.rows * 4, stream, b.seed.data(), b.dense.data(), b.rows,
               0u);
        launch("pearl_prepare_sparse", b.k / 8, stream, b.seed.data(), b.pairs.data(), b.k, 0u);

        launch("pearl_prepare_materialize", bytes, stream, b.base.data(), b.dense.data(),
               b.pairs.data(), b_operand.data(), b.rows, b.k, 0u);
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
        for (const auto *buffer : {&matrix.base, &matrix.tree, &matrix.seed,
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
          validation(options.mode == ExecutionMode::Validation), batch_size(options.batch_size) {
    }
    const native::Shape shape;
    const uint32_t capacity;
    const bool validation;
    const unsigned batch_size;
    int device = 0;
    uint64_t launches = 0;
    uint64_t completed = 0;
    uint64_t first_nonce = 0, nonce_high = 0;
    double prepare_host_ms = 0, completion_host_ms = 0, collect_host_ms = 0;
    Clock::time_point first_work{}, last_work{};
    native::Digest session_seed{};
    native::Identity current_identity;
    // User and dev jobs retain separate dependency caches across fee switches.
    std::array<std::unique_ptr<Preparation>, 2> preparation;
    Buffer a, target, winners, result, patches, paths, seeds, pairs;
};

struct Evidence : GPUBatchEvidence {
    std::vector<std::shared_ptr<const Evidence>> children;
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

#include "pearl_batch.inc"
#include "pearl_rdna4_check.inc"

} // namespace

AlgoConfig pearl_gpu_config(ExecutionOptions options) {
    const bool rdna4 = options.backend == Backend::ExperimentalGfx1201;
    if (options.recipe >= 8 || (!rdna4 && options.recipe != 0))
        throw std::invalid_argument("Invalid Pearl backend recipe");
    if (rdna4 && options.mode == ExecutionMode::Mining)
        throw std::invalid_argument("gfx1201 is offline-test-only until hardware qualification");
    if (options.test_workload != TestWorkload::FreshFused &&
        (!rdna4 || options.mode != ExecutionMode::Benchmark))
        throw std::invalid_argument("Prepared-operand controls are experimental benchmarks only");
    options.shape.layout = native::CandidateLayout::native_4x32;
    options.shape.validate();
    if (!options.batch_size || options.batch_size > 32)
        throw std::invalid_argument("Pearl batch must contain 1 through 32 attempts");
    if (!options.winner_capacity)
        throw std::invalid_argument("Pearl winner capacity is zero");
    AlgoConfig config{};
    config.rate_unit = RateUnit::MultiplyAccumulates;
    config.name = "pearl";
    config.algo_id = ALGO_PEARL_POUW;
    config.source_path = "src/tnn_hip/crypto/pearl/native128/rtc.hip";
    config.source =
        hip_pearl_iris_qualified_source::SRC_TNN_HIP_CRYPTO_PEARL_NATIVE128_RTC_HIP_SOURCE;
    config.kernel_names = {"pearl_iris_fused", "pearl_iris_raw", "pearl_iris_diagnostic"};
    config.rtc_headers =
        build_rtc_headers(hip_embedded::COMMON_HEADERS, hip_embedded::IRIS_HEADERS,
                          hip_embedded::PEARL_HEADERS, hip_embedded::PEARL_IRIS_QUALIFIED_HEADERS);
    if (rdna4) {
        config.name = "pearl-gfx1201-r" + std::to_string(options.recipe);
        config.source_path = "src/tnn_hip/crypto/pearl/rdna4/rtc.hip";
        config.source = hip_pearl_rdna4_source::SRC_TNN_HIP_CRYPTO_PEARL_RDNA4_RTC_HIP_SOURCE;
        config.rtc_headers = build_rtc_headers(hip_embedded::COMMON_HEADERS,
            hip_embedded::PEARL_HEADERS, hip_embedded::PEARL_RDNA4_HEADERS);
        config.source_transform_fn = [recipe = options.recipe](const std::string& source, int) {
            return "#define PEARL_GFX12_RECIPE " + std::to_string(recipe) + "\n" + source;
        };
    }
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
    config.preferred_block_size = 128;
    config.enable_autotune = false;
    config.enable_reg_tuning = false;
    config.skip_cached_tune_validation = true;
    TuningResult fixed{};
    fixed.block_size = 128;
    fixed.num_blocks = (options.shape.m / 128) * (options.shape.n / 128);
    fixed.batch_size = options.batch_size;
    fixed.valid = true;
    config.fixed_launch = fixed;
    config.pre_tune_fn = [options](const KernelMap &, const oroDeviceProp_t &props, int device,
                                   void **output) {
        const int required = options.backend == Backend::ExperimentalGfx1201 ? 1201 : 1100;
        if (!tnn_is_amd_device(device) || parse_gfx_number(props.gcnArchName) != required) {
            throw std::runtime_error("Pearl backend/device mismatch; refusing launch");
        }
        size_t free_bytes = 0, total_bytes = 0;
        checked(oroMemGetInfo(&free_bytes, &total_bytes), "Pearl allocation budget");
        const uint64_t reserve = std::max<uint64_t>(512ull << 20, total_bytes / 10);
        if (free_bytes <= reserve || native::allocation_budget(options.shape,
                options.batch_size, options.winner_capacity) > free_bytes - reserve)
            throw std::runtime_error("Pearl shape exceeds the available device-memory budget");
        auto state = std::make_unique<State>(options);
        {
            checked(oroStreamCreateWithFlags(&state->compute_stream, oroStreamNonBlocking),
                    "Pearl compute stream");
            checked(oroEventCreateWithFlags(&state->readback_done, oroEventBlockingSync),
                    "Pearl readback event");
            checked(oroHostMalloc(reinterpret_cast<void **>(&state->host_transfer),
                                  32 + options.batch_size * sizeof(DeviceState), 0),
                    "Pearl pinned staging");
        }
        const auto shape = state->shape;
        const auto capacity = state->capacity;
        state->device = device;
        std::random_device random;
        for (auto &byte : state->session_seed)
            byte = uint8_t(random());
        state->a.allocate(size_t(options.batch_size) * shape.m * shape.k);
        unsigned levels = 1;
        for (unsigned nodes = shape.m * shape.k / 1024; nodes > 1; nodes = (nodes + 1) / 2)
            ++levels;
        state->patches.allocate(options.batch_size * 32);
        state->paths.allocate(size_t(options.batch_size) * levels * 32);
        state->seeds.allocate(options.batch_size * 32);
        state->pairs.allocate(size_t(options.batch_size) * shape.k * 8);
        state->target.allocate(32);
        state->winners.allocate(size_t(options.batch_size) * capacity * sizeof(DeviceWinner));
        state->result.allocate(options.batch_size * sizeof(DeviceState));
        {
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
                "[PEARL-WALL-SUMMARY] completed=%llu active_wall_s=%.6f active_wall_TMACs=%.3f "
                "prepare_host_ms=%.3f completion_host_ms=%.3f collect_host_ms=%.3f\n",
                static_cast<unsigned long long>(state->completed), seconds, macs / (seconds * 1e12),
                state->prepare_host_ms, state->completion_host_ms, state->collect_host_ms);
        }
        delete state;
    };
    config.prepare_batch_fn = prepare;
    config.execute_fn = execute;
    if (rdna4 && options.mode == ExecutionMode::Validation) {
        config.execute_fn = [](const KernelMap& kernels, const KernelLaunchContext& context) {
            check_rdna4(kernels, context);
            return execute(kernels, context);
        };
    }
    if (options.test_workload != TestWorkload::FreshFused) {
        config.prepare_batch_fn = [ready = false](const KernelLaunchContext& context) mutable {
            if (!ready) { ready = prepare(context); return ready; }
            auto& state = *static_cast<State*>(context.algo_data);
            checked(oroMemsetAsync(state.result.data(), 0, state.result.size, context.stream),
                    "Reset prepared-operand control");
            return true;
        };
    }
    if (options.test_workload == TestWorkload::PreparedRaw) {
        config.execute_fn = [](const KernelMap& kernels, const KernelLaunchContext& context) {
            auto& state = *static_cast<State*>(context.algo_data);
            unsigned m = state.shape.m, n = state.shape.n, k = state.shape.k;
            auto* b = state.preparation[state.current_identity.is_dev]->b_operand.data();
            int32_t* d = nullptr;
            for (unsigned slot = 0; slot < state.batch_size; ++slot) {
                auto* a = state.a.data() + size_t(slot) * m * k;
                void* args[] = {&a, &b, &d, &m, &n, &k, &m, &k, &m};
                checked(oroModuleLaunchKernel(kernels.at("pearl_iris_raw"),
                    m / 128 * (n / 128), 1, 1, 128, 1, 1, 0, context.stream, args, nullptr),
                    "Launch D-free raw control");
                ++state.launches;
            }
            return true;
        };
    }
    config.collect_batch_fn = collect;
    {
        config.batch_stream_fn = [](void *pointer) {
            return static_cast<State *>(pointer)->compute_stream;
        };
        config.finish_batch_fn = [](const KernelLaunchContext &context) {
            auto &state = *static_cast<State *>(context.algo_data);
            checked(oroMemcpyAsync(state.host_transfer + 32, state.result.data(), state.result.size,
                                   oroMemcpyDeviceToHost, context.stream),
                    "Enqueue Pearl result header");
            checked(oroEventRecord(state.readback_done, context.stream), "Record Pearl readback");
            checked(oroEventSynchronize(state.readback_done), "Complete Pearl readback");
        };
    }
    if (options.mode == ExecutionMode::Mining)
        pearl_configure_tuning(config, options);
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
        attempt.identity.target_le !=
            native::jackpot_target(job.raw_target, attempt.shape.k, attempt.shape.layout) ||
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
    if (evidence && !evidence->children.empty()) {
        size_t found = 0;
        for (const auto &child : evidence->children) {
            auto member = batch;
            member.count = 1;
            member.evidence = child;
            found += pearl_validate_batch(member, verify_all_positions);
        }
        return found;
    }
    if (evidence && !evidence->attempt && !verify_all_positions) {
        // Tune qualification checks real captured winners without materializing
        // entire matrices on the CPU. The snapshot importer authenticates the
        // Merkle paths; winner_digest regenerates the prescribed noise.
        const auto key = native::job_key(evidence->identity.header, evidence->shape);
        for (size_t i = 0; i < evidence->winners.size(); ++i) {
            const auto& samples = evidence->samples.at(i);
            native::Attempt reference(evidence->identity, evidence->shape,
                native::MatrixTree::from_snapshot(evidence->shape.m, evidence->shape.k,
                    samples.a_rows, samples.a_bytes, evidence->a_tree, key),
                native::MatrixTree::from_snapshot(evidence->shape.n, evidence->shape.k,
                    samples.b_rows, samples.b_bytes, evidence->b_tree, key));
            const auto& winner = evidence->winners[i];
            if (winner.digest != reference.winner_digest(winner.row, winner.col))
                throw std::runtime_error("Pearl tune winner differs from CPU reference");
            (void)reference.proof(winner);
        }
        return evidence->winners.size();
    }
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
        for (uint32_t row = 0; row + 96 < reference.shape.m; ++row) {
            if (row % 128 >= 32)
                continue;
            for (uint32_t col = 0; col + 59 < reference.shape.n; ++col) {
                if (col % 64 != 0 && col % 64 != 4)
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
    if (evidence && !evidence->children.empty()) {
        for (const auto &child : evidence->children) {
            auto member = batch;
            member.count = 1;
            member.evidence = child;
            pearl_build_batch(member, device, job);
        }
        return {};
    }
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
