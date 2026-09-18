// Offline-only full-D and checkpoint oracle. Uses the exact prepared operands
// that the production adapter will feed to the fused kernel next.
void check_rdna4_operands(const KernelMap& kernels, const KernelLaunchContext& context,
                          unsigned pattern) {
    auto& state = *static_cast<State*>(context.algo_data);
    unsigned m = state.shape.m, n = state.shape.n, k = state.shape.k;
    if (m > 384 || n > 384)
        throw std::invalid_argument("Exhaustive RDNA4 oracle is restricted to small matrices");
    checked(oroStreamSynchronize(context.stream), "Finish diagnostic preparation");
    auto* a = state.a.data();
    auto* b = state.preparation[state.current_identity.is_dev]->b_operand.data();
    auto* key = state.seeds.data();
    auto* target = state.target.data();
    std::vector<int8_t> host_a(size_t(m) * k), host_b(size_t(n) * k);
    native::Digest host_key{};
    checked(oro_safe_memcpy(host_a.data(), a, host_a.size(), oroMemcpyDeviceToHost), "Read oracle A");
    checked(oro_safe_memcpy(host_b.data(), b, host_b.size(), oroMemcpyDeviceToHost), "Read oracle B");
    checked(oro_safe_memcpy(host_key.data(), key, 32, oroMemcpyDeviceToHost), "Read oracle key");
    const auto original_a = host_a, original_b = host_b;
    if (pattern) {
        for (size_t i = 0; i < host_a.size(); ++i)
            host_a[i] = pattern == 1 ? 0 : pattern == 2 ? -128 : int8_t(int((i * 17 + i / 251) % 256) - 128);
        for (size_t i = 0; i < host_b.size(); ++i)
            host_b[i] = pattern == 1 ? 0 : pattern == 2 ? 127 : int8_t(int((i * 13 + i / 257) % 256) - 128);
        checked(oro_safe_memcpy(a, host_a.data(), host_a.size(), oroMemcpyHostToDevice), "Write oracle A");
        checked(oro_safe_memcpy(b, host_b.data(), host_b.size(), oroMemcpyHostToDevice), "Write oracle B");
    }

    const unsigned positions = m * n / 128;
    Buffer output, diagnostics, winners, counters;
    output.allocate(size_t(m) * n * 4);
    diagnostics.allocate(size_t(positions) * 24 * 4);
    winners.allocate(size_t(positions) * sizeof(DeviceWinner));
    counters.allocate(sizeof(DeviceState));
    auto* d = output.data();
    void* raw_args[] = {&a, &b, &d, &m, &n,
                        &k, &m,
                        &k, &m};
    checked(oroModuleLaunchKernel(kernels.at("pearl_iris_raw"), m / 128 * (n / 128), 1, 1,
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

    std::vector<int32_t> expected(size_t(m) * n), actual(expected.size());
    std::vector<uint32_t> words(size_t(positions) * 24), reference(words.size());
    checked(oro_safe_memcpy(actual.data(), d, output.size, oroMemcpyDeviceToHost), "Read raw D");
    checked(oro_safe_memcpy(words.data(), diag, diagnostics.size, oroMemcpyDeviceToHost), "Read transcripts");
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
    if (actual != expected) throw std::runtime_error("RDNA4 full-D mismatch");
    for (unsigned position = 0; position < positions; ++position) {
        auto* transcript = reference.data() + position * 24;
        const auto digest = native::keyed_hash(
            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(transcript), 64), host_key);
        std::memcpy(transcript + 16, digest.data(), 32);
    }
    if (words != reference) throw std::runtime_error("RDNA4 checkpoint/digest mismatch");
    for (auto* buffer : {&output, &diagnostics, &winners, &counters}) buffer->check_guards();
    std::vector<int8_t> after_a(host_a.size()), after_b(host_b.size());
    checked(oro_safe_memcpy(after_a.data(), a, after_a.size(), oroMemcpyDeviceToHost), "Recheck A");
    checked(oro_safe_memcpy(after_b.data(), b, after_b.size(), oroMemcpyDeviceToHost), "Recheck B");
    if (host_a != after_a || host_b != after_b)
        throw std::runtime_error("RDNA4 changed an input operand");
    if (pattern) {
        checked(oro_safe_memcpy(a, original_a.data(), original_a.size(), oroMemcpyHostToDevice), "Restore A");
        checked(oro_safe_memcpy(b, original_b.data(), original_b.size(), oroMemcpyHostToDevice), "Restore B");
    }
}

void check_rdna4(const KernelMap& kernels, const KernelLaunchContext& context) {
    auto& state = *static_cast<State*>(context.algo_data);
    check_rdna4_operands(kernels, context, 0);
    if (state.launches == 0 && state.shape.k == 2048 && state.shape.m <= 256) {
        for (unsigned pattern = 1; pattern <= 3; ++pattern)
            check_rdna4_operands(kernels, context, pattern);
        TNN_LOG_INFO("[PEARL-GFX12] Full D, transcripts, digests, guards and signed patterns passed\n");
    }
}
