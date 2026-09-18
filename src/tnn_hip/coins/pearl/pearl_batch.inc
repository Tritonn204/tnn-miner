// Production batch-32 adapter. GPU buffers are reusable; submitted evidence is not.
bool prepare(const KernelLaunchContext &context) {
    auto &state = *static_cast<State *>(context.algo_data);
    const auto shape = state.shape;
    const auto *job = context.job_snapshot;

    // A batch belongs to one immutable Stratum snapshot. Never recover its
    // target, certificate version, or proof identity from a later live job.
    if (!job || job->work_template.size() != 76 || !job->connection_generation)
        throw std::invalid_argument("Pearl requires a complete atomic job snapshot");
    if (context.batch_size != state.batch_size)
        throw std::invalid_argument("Pearl batch size does not match its owned buffers");
    if (job->pearl_cert_version != 2 && job->pearl_cert_version != 3)
        throw std::invalid_argument("Unsupported Pearl certificate version");
    if (state.first_work == Clock::time_point{})
        state.first_work = Clock::now();
    if (state.launches > UINT64_MAX - state.batch_size - 1)
        throw std::runtime_error("Pearl attempt counter exhausted");

    native::Identity identity;
    identity.job_id = job->job_id_str;
    identity.connection_generation = job->connection_generation;
    identity.attempt_id = state.launches;
    identity.device_id = state.device;
    identity.is_dev = job->is_dev;
    identity.cert_version = job->pearl_cert_version;
    std::copy(job->work_template.begin(), job->work_template.end(), identity.header.begin());
    identity.wire_target_le = job->raw_target;
    identity.target_le = native::jackpot_target(job->raw_target, shape.k, shape.layout);
    state.current_identity = identity;

    // User and dev work have independent job-stable B and base-template caches.
    auto &prep = *state.preparation[identity.is_dev];
    const bool new_job =
        !prep.ready || prep.base_identity.header != identity.header ||
        prep.base_identity.cert_version != identity.cert_version ||
        prep.base_identity.connection_generation != identity.connection_generation ||
        prep.base_identity.is_dev != identity.is_dev;
    if (new_job) {
        prep.base_identity = identity;
        prep.base_identity.attempt_id = 0;

        const auto key = native::job_key(identity.header, shape);
        prep.job_key.upload(key.data(), key.size());

        // Cache every level once. Each fresh attempt replaces only the
        // leftmost path, so proof collection never rebuilds the whole tree.
        native::MatrixTree base_template(
            std::vector<uint8_t>(size_t(shape.m) * shape.k, native::mining_base_value),
            shape.m, shape.k, key);
        prep.template_tree_cpu.clear();
        prep.template_offsets.clear();

        for (const auto &level : base_template.layers()) {
            prep.template_offsets.push_back(prep.template_tree_cpu.size());
            for (const auto &digest : level)
                prep.template_tree_cpu.insert(prep.template_tree_cpu.end(), digest.begin(), digest.end());
        }

        prep.template_tree.upload(prep.template_tree_cpu.data(), prep.template_tree_cpu.size());

        prep.prepare_b(context.stream);
        prep.ready = true;
    }

    // Only the first leaf changes between attempts. Compute each candidate's
    // Merkle path and seed, then materialize all A operands in one launch.
    state.first_nonce = state.launches + 1;
    std::memcpy(&state.nonce_high, state.session_seed.data(), sizeof(state.nonce_high));
    const unsigned count = state.batch_size, levels = unsigned(prep.template_offsets.size());

    prep.launch("pearl_batch_paths", count, context.stream, prep.job_key.data(),
                prep.template_tree.data(), prep.b.seed.data(), state.patches.data(), state.paths.data(),
                state.seeds.data(), shape.m * shape.k / 1024, levels, count, state.first_nonce,
                state.nonce_high, identity.cert_version, shape.m);

    prep.launch("pearl_batch_sparse", count * (shape.k / 8), context.stream, state.seeds.data(),
                state.pairs.data(), shape.k, count);

    auto patches = state.patches.data(), seeds = state.seeds.data();
    auto pairs = state.pairs.data(), a = state.a.data();
    unsigned rows = shape.m, k = shape.k, first = 0;
    void *args[] = {&patches, &seeds, &pairs, &a, &rows, &k, &first};
    checked(oroModuleLaunchKernel(prep.kernels.at("pearl_batch_materialize"), rows / 32, count, 1,
                                  128, 1, 1, 0, context.stream, args, nullptr),
            "Prepare Pearl A batch");

    // The pinned upload remains owned until the batch's completion event.
    if (!state.target_valid || state.uploaded_target != identity.target_le) {
        std::memcpy(state.host_transfer, identity.target_le.data(), 32);
        checked(oroMemcpyAsync(state.target.data(), state.host_transfer, 32, oroMemcpyHostToDevice,
                               context.stream),
                "Enqueue Pearl target");
        state.uploaded_target = identity.target_le;
        state.target_valid = true;
    }
    checked(oroMemsetAsync(state.result.data(), 0, state.result.size, context.stream),
            "Reset Pearl batch results");

    return true;
}

bool execute(const KernelMap &kernels, const KernelLaunchContext &context) {
    auto &state = *static_cast<State *>(context.algo_data);
    if (context.block_size != 128 || context.batch_size != state.batch_size)
        throw std::runtime_error("Pearl Iris launch contract");

    unsigned m = state.shape.m, n = state.shape.n, k = state.shape.k;
    auto capacity = state.capacity;
    auto b = state.preparation[state.current_identity.is_dev]->b_operand.data();
    auto target = state.target.data();
    int32_t *d = nullptr;
    uint32_t *diagnostic = nullptr;

    // No completion waits or diagnostic stores between these launches.
    // Every GEMM owns a different A, seed, result counter, and winner region.
    for (unsigned slot = 0; slot < state.batch_size; ++slot) {
        auto a = state.a.data() + size_t(slot) * m * k;
        auto key = state.seeds.data() + slot * 32;
        auto winners = state.winners.data() + size_t(slot) * capacity * sizeof(DeviceWinner);
        auto result = state.result.data() + slot * sizeof(DeviceState);
        void *args[] = {&a, &b,   &d,      &m,       &n,      &k,        &m,         &k,
                        &m, &key, &target, &winners, &result, &capacity, &diagnostic};
        checked(oroModuleLaunchKernel(kernels.at("pearl_iris_fused"), (m / 128) * (n / 128), 1, 1,
                                      128, 1, 1, 0, context.stream, args, nullptr),
                "Launch Pearl Iris batch member");
        ++state.launches;
    }

    return true;
}

void collect(const KernelLaunchContext &context, BatchResult &batch) {
    const auto began = Clock::now();
    auto &state = *static_cast<State *>(context.algo_data);
    const auto shape = state.shape;
    auto &prep = *state.preparation[state.current_identity.is_dev];

    // Large qualified shapes contain up to 32 individually bounded launches.
    // Their aggregate time can exceed a second without a long-running kernel.
    if (context.elapsed_ms > 5000 || context.completion_host_ms > 5000)
        throw std::runtime_error("Pearl batch completion exceeded 5000 ms; stopped");

    auto root = std::make_shared<Evidence>();
    const auto copy = [](const uint8_t *device, size_t bytes) {
        std::vector<uint8_t> result(bytes);
        checked(oro_safe_memcpy(result.data(), device, bytes, oroMemcpyDeviceToHost),
                "Copy owned Pearl proof evidence");
        return result;
    };
    std::vector<uint8_t> b_tree;

    // The common completion hook has already copied all counters. Empty
    // attempts require no tree/operand downloads in normal mining mode.
    for (unsigned slot = 0; slot < state.batch_size; ++slot) {
        DeviceState status{};
        std::memcpy(&status, state.host_transfer + 32 + slot * sizeof(status), sizeof(status));
        if (status.overflow || status.total_hits > state.capacity)
            throw std::runtime_error("Pearl winner capacity exceeded; no candidates discarded");
        if (!status.total_hits && !state.validation)
            continue;

        auto evidence = std::make_shared<Evidence>();
        evidence->identity = state.current_identity;
        evidence->identity.attempt_id += slot;
        evidence->shape = shape;

        // Copy-on-win evidence is independent of the next GPU batch. The
        // The template tree's first node at every level is patched from this attempt.
        const auto patch = copy(state.patches.data() + slot * 32, 32);
        evidence->a_tree = prep.template_tree_cpu;
        const auto path = copy(state.paths.data() + size_t(slot) * prep.template_offsets.size() * 32,
                               prep.template_offsets.size() * 32);
        for (unsigned level = 0; level < prep.template_offsets.size(); ++level)
            std::memcpy(evidence->a_tree.data() + prep.template_offsets[level],
                        path.data() + level * 32, 32);
        if (b_tree.empty())
            b_tree = copy(prep.b.tree.data(), prep.b.tree.size);
        evidence->b_tree = b_tree;

        // Expensive CPU reconstruction is exclusively an offline-test feature.
        if (state.validation) {
            std::vector<uint8_t> base(size_t(shape.m) * shape.k, native::mining_base_value);
            const uint64_t low = state.first_nonce + slot;
            for (unsigned i = 0; i < 19; ++i) {
                unsigned value = 0;
                for (unsigned j = 0; j < 7; ++j) {
                    const unsigned bit = i * 7 + j;
                    if (bit < 128)
                        value |= unsigned(((bit < 64 ? low : state.nonce_high) >> (bit % 64)) & 1)
                                 << j;
                }
                base[i] = uint8_t(int(value) - 64);
            }
            if (!std::equal(base.begin(), base.begin() + 19, patch.begin()))
                throw std::runtime_error("Pearl batch patch mismatch");
            evidence->attempt = std::make_shared<native::Attempt>(
                evidence->identity, shape, std::move(base),
                std::vector<uint8_t>(size_t(shape.n) * shape.k, native::mining_base_value));
            const auto &reference = *evidence->attempt;
            const auto operand =
                copy(state.a.data() + size_t(slot) * shape.m * shape.k, size_t(shape.m) * shape.k);
            if (std::memcmp(operand.data(), reference.a.data(), operand.size()) ||
                copy(state.seeds.data() + slot * 32, 32) !=
                    std::vector<uint8_t>(reference.a_seed.begin(), reference.a_seed.end()))
                throw std::runtime_error("Pearl batch operand/seed mismatch");
            prep.verify(prep.b, prep.b_operand, reference.bt_tree, reference.b_seed, reference.bt);
        }
        std::vector<DeviceWinner> winners(status.total_hits);
        if (!winners.empty())
            checked(oro_safe_memcpy(winners.data(),
                                    state.winners.data() +
                                        size_t(slot) * state.capacity * sizeof(DeviceWinner),
                                    winners.size() * sizeof(DeviceWinner), oroMemcpyDeviceToHost),
                    "Read Pearl winners");

        std::set<std::pair<uint32_t, uint32_t>> positions;
        for (const auto &source : winners) {
            if (source.row >= shape.m - 96 || source.row % 128 >= 32 ||
                source.col >= shape.n - 59 || (source.col % 64 != 0 && source.col % 64 != 4) ||
                !positions.emplace(source.row, source.col).second)
                throw std::runtime_error("Invalid/duplicate Pearl winner coordinates");
            native::Winner winner{source.row, source.col, {}};
            std::memcpy(winner.digest.data(), source.digest, 32);
            if (!native::meets_target(winner.digest, evidence->identity.target_le))
                throw std::runtime_error("Pearl GPU winner above target");
            evidence->winners.push_back(winner);

            // A's base is constant except its nonce patch; only selected B rows
            // need a device gather. Both snapshots outlive reusable buffers.
            Evidence::Samples samples;
            samples.a_rows = native::candidate_rows(shape, winner.row);
            samples.b_rows = native::candidate_columns(shape, winner.col);
            samples.a_bytes.resize(samples.a_rows.size() * shape.k, native::mining_base_value);
            for (unsigned row = 0; row < samples.a_rows.size(); ++row)
                if (samples.a_rows[row] == 0)
                    std::copy_n(patch.begin(), 19, samples.a_bytes.begin() + size_t(row) * shape.k);
            checked(oro_safe_memcpy(prep.b.sample_rows.data(), samples.b_rows.data(),
                                    samples.b_rows.size() * 4, oroMemcpyHostToDevice),
                    "Upload Pearl B sample rows");
            prep.launch("pearl_prepare_gather", unsigned(samples.b_rows.size()) * shape.k,
                        context.stream, prep.b.base.data(), prep.b.sample_rows.data(),
                        prep.b.sampled.data(), unsigned(samples.b_rows.size()), shape.k);
            checked(oroStreamSynchronize(context.stream), "Complete Pearl proof gather");
            samples.b_bytes = copy(prep.b.sampled.data(), samples.b_rows.size() * shape.k);
            evidence->samples.push_back(std::move(samples));
        }
        root->children.push_back(std::move(evidence));
    }

    if (state.validation) {
        for (const auto *buffer : {&state.a, &state.patches, &state.paths, &state.seeds,
                                   &state.pairs, &state.target, &state.winners, &state.result})
            buffer->check_guards();
    }

    // The miner accounts GEMMs here, with MACs per GEMM as its multiplier.
    batch.count = state.batch_size;
    batch.work_multiplier = shape.macs();
    batch.evidence = std::move(root);
    state.completed += batch.count;
    state.prepare_host_ms += context.prepare_host_ms;
    state.completion_host_ms += context.completion_host_ms;
    state.collect_host_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - began).count();
    state.last_work = Clock::now();
}
