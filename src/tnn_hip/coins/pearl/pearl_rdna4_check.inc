// Offline-only full-D and checkpoint oracle. Uses the exact prepared operands
// that the production adapter will feed to the fused kernel next.
void check_rdna4_operands(const KernelMap& kernels, const KernelLaunchContext& context,
                          unsigned pattern, unsigned test_depth = 0) {
    auto& state = *static_cast<State*>(context.algo_data);
    unsigned m = state.shape.m, n = state.shape.n, k = state.shape.k;
    if (m > 384 || n > 768 || (n > 384 && k != 2048))
        throw std::invalid_argument("Exhaustive RDNA4 oracle is restricted to small matrices");
    checked(oroStreamSynchronize(context.stream), "Finish diagnostic preparation");
    auto* a = state.a.data();
    auto* b = state.preparation[state.current_identity.is_dev]->b_operand.data();
    auto* key = state.seeds.data();
    auto* target = state.target.data();
    std::vector<int8_t> original_a(size_t(m) * k), original_b(size_t(n) * k);
    native::Digest host_key{};
    checked(oro_safe_memcpy(original_a.data(), a, original_a.size(), oroMemcpyDeviceToHost), "Read oracle A");
    checked(oro_safe_memcpy(original_b.data(), b, original_b.size(), oroMemcpyDeviceToHost), "Read oracle B");
    checked(oro_safe_memcpy(host_key.data(), key, 32, oroMemcpyDeviceToHost), "Read oracle key");
    if (test_depth) {
        validation::require(test_depth <= k && test_depth % 128 == 0, "Invalid test-only depth");
        m = 128; n = 256; k = test_depth;
    }
    std::vector<int8_t> host_a(size_t(m) * k), host_b(size_t(n) * k);
    for (unsigned kk = 0; kk < k; ++kk)
        std::copy_n(original_a.data() + size_t(kk) * state.shape.m, m, host_a.data() + size_t(kk) * m);
    for (unsigned col = 0; col < n; ++col)
        std::copy_n(original_b.data() + size_t(col) * state.shape.k, k, host_b.data() + size_t(col) * k);
    if (pattern) {
        auto sample = [pattern](size_t i, unsigned salt) -> int8_t {
            if (pattern == 1) return 0;
            if (pattern == 2) return salt == 17 ? -128 : 127;
            if (pattern == 3) return int8_t(int((i * salt + i / 251) % 256) - 128);
            uint32_t bits = uint32_t(i) ^ (salt * 0x9e3779b9u);
            bits ^= bits >> 16; bits *= 0x7feb352du;
            bits ^= bits >> 15; bits *= 0x846ca68bu;
            bits ^= bits >> 16;
            return int8_t(int(bits & 255) - 128);
        };
        for (size_t i = 0; i < host_a.size(); ++i)
            host_a[i] = pattern == 4 ? int8_t(i == m * 129 + 17 ? -127 : 0) :
                        pattern == 5 ? int8_t(int((i % m) * 7 + i / m * 3) % 127 - 63) : sample(i, 17);
        for (size_t i = 0; i < host_b.size(); ++i)
            host_b[i] = pattern == 4 ? int8_t(i == k * 31 + 129 ? 113 : 0) :
                        pattern == 5 ? int8_t(int((i / k) * 11 + i % k * 5) % 127 - 63) : sample(i, 13);
    }
    if (pattern || test_depth) {
        checked(oro_safe_memcpy(a, host_a.data(), host_a.size(), oroMemcpyHostToDevice), "Write oracle A");
        checked(oro_safe_memcpy(b, host_b.data(), host_b.size(), oroMemcpyHostToDevice), "Write oracle B");
    }

    const unsigned positions = m * n / 128;
    Buffer output, diagnostics, winners, counters;
    output.allocate(size_t(m) * n * 4);
    diagnostics.allocate(size_t(positions) * 24 * 4);
    winners.allocate(size_t(positions) * sizeof(DeviceWinner));
    counters.allocate(sizeof(DeviceState));
    // Guard initialization uses the default stream; the validation stream is
    // nonblocking and must not race those fills when overwriting the payload.
    checked(oroStreamSynchronize(nullptr), "Publish oracle allocation guards");
    auto* d = output.data();
    void* raw_args[] = {&a, &b, &d, &m, &n,
                        &k, &m,
                        &k, &m};
    const char* raw_symbol = state.backend == Backend::Rdna3 ? "pearl_iris_validation_raw" : "pearl_iris_raw";
    checked(oroModuleLaunchKernel(kernels.at(raw_symbol), m / 128 * (n / 128), 1, 1,
                                  128, 1, 1, 0, context.stream, raw_args, nullptr), "RDNA4 raw check");
    checked(oroMemsetAsync(counters.data(), 0, counters.size, context.stream), "Clear diagnostic counters");
    auto* diag = diagnostics.data();
    auto* win = winners.data();
    auto* count = counters.data();
    unsigned capacity = positions;
    void* args[] = {&a, &b, &d, &m, &n,
                    &k, &m, &k,
                    &m, &key, &target, &win, &count, &capacity, &diag};
    checked(oroModuleLaunchKernel(kernels.at("pearl_iris_diagnostic"), m / 128 * (n / 128), 1, 1,
                                  128, 1, 1, 0, context.stream, args, nullptr), "RDNA4 checkpoint check");
    checked(oroStreamSynchronize(context.stream), "Finish RDNA4 diagnostics");

    std::vector<int64_t> expected(size_t(m) * n);
    std::vector<int32_t> actual(expected.size());
    std::vector<uint32_t> words(size_t(positions) * 24), reference(words.size());
    checked(oro_safe_memcpy(actual.data(), d, output.size, oroMemcpyDeviceToHost), "Read raw D");
    checked(oro_safe_memcpy(words.data(), diag, diagnostics.size, oroMemcpyDeviceToHost), "Read transcripts");
    words = validation::normalize_diagnostics(words, m, n, state.backend == Backend::Rdna3);
    for (unsigned base = 0; base < k; base += 128) {
        // Independent mathematical accumulation: no WMMA fragment maps here.
        for (unsigned col = 0; col < n; ++col)
            for (unsigned kk = base; kk < base + 128; ++kk)
                for (unsigned row = 0; row < m; ++row)
                    expected[size_t(col) * m + row] +=
                        int(host_a[size_t(kk) * m + row]) * int(host_b[size_t(col) * k + kk]);
        for (unsigned tile = 0; tile < m / 128 * (n / 128); ++tile)
            for (unsigned tid = 0; tid < 128; ++tid) {
                const unsigned row = tile / (n / 128) * 128 + (tid / 32) % 2 * 16 + tid % 16;
                const unsigned col = tile % (n / 128) * 128 + (tid / 64) * 64 + (tid % 32) / 16 * 4;
                uint32_t value = 0;
                for (unsigned r = 0; r < 4; ++r)
                    for (unsigned c = 0; c < 32; ++c)
                        value ^= uint32_t(expected[size_t(col + c / 4 * 8 + c % 4) * m + row + r * 32]);
                auto& slot = reference[(tile * 128 + tid) * 24 + (base / 128) % 16];
                slot = std::rotl(slot, 13) ^ value;
            }
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        validation::require(expected[i] >= INT32_MIN && expected[i] <= INT32_MAX,
                            "RDNA4 oracle accumulator out of range");
        if (actual[i] != expected[i])
            throw std::runtime_error("Pearl full-D mismatch: index=" + std::to_string(i) +
                                     " actual=" + std::to_string(actual[i]) +
                                     " expected=" + std::to_string(expected[i]) +
                                     " pattern=" + std::to_string(pattern));
    }
    for (unsigned position = 0; position < positions; ++position) {
        auto* transcript = reference.data() + position * 24;
        const auto digest = native::keyed_hash(
            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(transcript), 64), host_key);
        std::memcpy(transcript + 16, digest.data(), 32);
    }
    if (words != reference) throw std::runtime_error("RDNA4 checkpoint/digest mismatch");

    // Validate both emitted paths against the complete CPU candidate universe.
    // A valid sampled proof cannot detect a candidate silently dropped by a kernel.
    validation::DigestWords original_target{};
    checked(oro_safe_memcpy(original_target.data(), target, 32, oroMemcpyDeviceToHost), "Read oracle target");
    std::vector<validation::Record> candidates;
    for (unsigned position = 0; position < positions; ++position) {
        const unsigned tile = position / 128, tid = position % 128;
        validation::Record record{
            tile / (n / 128) * 128 + (tid / 32) % 2 * 16 + tid % 16,
            tile % (n / 128) * 128 + (tid / 64) * 64 + (tid % 32) / 16 * 4};
        std::copy_n(reference.data() + position * 24 + 16, 8, record.begin() + 2);
        candidates.push_back(record);
    }
    auto verify = [&](const validation::DigestWords& bound, unsigned limit) {
        std::vector<validation::Record> expected_winners;
        for (const auto& candidate : candidates) {
            validation::DigestWords digest{};
            std::copy_n(candidate.begin() + 2, 8, digest.begin());
            if (validation::meets_target(digest, bound)) expected_winners.push_back(candidate);
        }
        DeviceState result{};
        checked(oro_safe_memcpy(&result, count, sizeof(result), oroMemcpyDeviceToHost), "Read oracle counters");
        const size_t stored = std::min<size_t>(expected_winners.size(), limit);
        std::vector<DeviceWinner> found(stored);
        if (stored) checked(oro_safe_memcpy(found.data(), win, stored * sizeof(DeviceWinner), oroMemcpyDeviceToHost), "Read oracle winners");
        validation::check_winners<DeviceWinner>(expected_winners, found, result.total_hits, result.overflow, limit);
        winners.check_guards();
        counters.check_guards();
    };
    verify(original_target, capacity);

    Buffer test_target;
    test_target.allocate(32);
    auto* isolated_target = test_target.data();
    void* lean_args[] = {&a, &b, &d, &m, &n, &k, &m, &k, &m,
                        &key, &isolated_target, &win, &count, &capacity, &diag};
    std::vector<validation::DigestWords> bounds{original_target};
    if (pattern && !test_depth) {
        validation::DigestWords maximum{}, pivot{};
        maximum.fill(UINT32_MAX);
        std::copy_n(candidates.front().begin() + 2, 8, pivot.begin());
        bounds.insert(bounds.end(), {validation::DigestWords{}, maximum, pivot});
        auto below = pivot, above = pivot;
        for (auto& word : below) if (word-- != 0) break;
        for (auto& word : above) if (++word != 0) break;
        if (pivot != validation::DigestWords{}) bounds.push_back(below);
        if (pivot != maximum) bounds.push_back(above);
    }
    for (const auto& bound : bounds) {
        checked(oro_safe_memcpy(isolated_target, bound.data(), 32, oroMemcpyHostToDevice), "Write isolated oracle target");
        std::vector<unsigned> capacities{positions};
        if (pattern && !test_depth) {
            unsigned hits = 0;
            for (const auto& candidate : candidates) {
                validation::DigestWords digest{};
                std::copy_n(candidate.begin() + 2, 8, digest.begin());
                hits += validation::meets_target(digest, bound);
            }
            capacities.insert(capacities.end(), {0u, 1u, hits, hits ? hits - 1 : 0u});
            std::sort(capacities.begin(), capacities.end());
            capacities.erase(std::unique(capacities.begin(), capacities.end()), capacities.end());
        }
        for (unsigned limit : capacities) {
            capacity = limit;
            for (const char* symbol : {"pearl_iris_diagnostic", "pearl_iris_fused"}) {
                checked(oroMemsetAsync(count, 0, counters.size, context.stream), "Clear oracle counters");
                checked(oroMemsetAsync(win, 0xa5, winners.size, context.stream), "Poison oracle winners");
                checked(oroModuleLaunchKernel(kernels.at(symbol), m / 128 * (n / 128), 1, 1,
                                             128, 1, 1, 0, context.stream, lean_args, nullptr), "Winner coverage check");
                checked(oroStreamSynchronize(context.stream), "Finish winner coverage check");
                verify(bound, limit);
                const size_t tail_size = (positions - limit) * sizeof(DeviceWinner);
                std::vector<uint8_t> tail(tail_size);
                if (tail_size) {
                    checked(oro_safe_memcpy(tail.data(), win + limit * sizeof(DeviceWinner), tail_size,
                                           oroMemcpyDeviceToHost), "Read winner capacity guard");
                    validation::require(std::all_of(tail.begin(), tail.end(), [](auto x) { return x == 0xa5; }),
                                        "Write beyond winner capacity");
                }
            }
        }
    }
    test_target.check_guards();
    for (auto* buffer : {&output, &diagnostics, &winners, &counters}) buffer->check_guards();
    std::vector<int8_t> after_a(host_a.size()), after_b(host_b.size());
    checked(oro_safe_memcpy(after_a.data(), a, after_a.size(), oroMemcpyDeviceToHost), "Recheck A");
    checked(oro_safe_memcpy(after_b.data(), b, after_b.size(), oroMemcpyDeviceToHost), "Recheck B");
    if (host_a != after_a || host_b != after_b)
        throw std::runtime_error("RDNA4 changed an input operand");
    if (pattern || test_depth) {
        checked(oro_safe_memcpy(a, original_a.data(), original_a.size(), oroMemcpyHostToDevice), "Restore A");
        checked(oro_safe_memcpy(b, original_b.data(), original_b.size(), oroMemcpyHostToDevice), "Restore B");
    }
}

void check_rdna4_suite(const KernelMap& checked_kernels, const KernelLaunchContext& context) {
    auto& state = *static_cast<State*>(context.algo_data);
    check_rdna4_operands(checked_kernels, context, 0);
    if (state.launches == 0 && state.shape.k == 8192 && state.current_identity.cert_version == 3) {
        // These are kernel-only depths, never accepted by production shape validation.
        // Prefix runs expose transcript wrap boundaries without altering GPU ISA.
        for (unsigned checkpoints : {15u, 16u, 17u, 31u, 32u, 33u}) {
            if (state.backend == Backend::Rdna3 && checkpoints < 16) continue;
            check_rdna4_operands(checked_kernels, context, 6, checkpoints * 128);
        }
    }
    if (state.launches == 0 && state.shape.k == 2048 && state.shape.m <= 256 && state.shape.n <= 384) {
        for (unsigned pattern = 1; pattern <= 6; ++pattern)
            check_rdna4_operands(checked_kernels, context, pattern);
        TNN_LOG_INFO("[PEARL-ORACLE] Full D, transcripts, digests, guards and signed patterns passed\n");
    }
}
