#include <tnn_hip/crypto/pearl/pearl_native.hpp>
#include <cassert>
#include <cstring>
#include <iostream>

// Execute the exact preparation source on CPU, without any GPU library.
#define PEARL_PREP_CPU
#define __device__
#define __global__
struct Index { uint32_t x; };
Index blockIdx{}, threadIdx{}, blockDim{128};
#include "../../src/tnn_hip/crypto/pearl/pearl_prepare.hip"

template<class Function, class... Args>
void launch(uint32_t threads, Function function, Args... args) {
    for (uint32_t index = 0; index < threads; ++index) {
        blockIdx.x = index / 128;
        threadIdx.x = index % 128;
        function(args...);
    }
}

int main() {
    namespace native = tnn::pearl::native;
    // Upstream api/seed.rs pinned V3 vector: job=11, A=aa, B=bb, m=192, n=320.
    native::Digest vector_key{}, vector_a{}, vector_b{}, vector_b_seed{}, vector_a_seed{};
    vector_key.fill(0x11);
    vector_a.fill(0xaa);
    vector_b.fill(0xbb);
    launch(1, pearl_prepare_seed, vector_key.data(), vector_b.data(), vector_b_seed.data(), 3u, 320u, 0u);
    launch(1, pearl_prepare_seed, vector_b_seed.data(), vector_a.data(), vector_a_seed.data(), 3u, 192u, 1u);
    const auto matches_hex = [](const native::Digest& digest, const char* text) {
        constexpr char digits[] = "0123456789abcdef";
        for (unsigned i = 0; i < 32; ++i)
            if (text[i * 2] != digits[digest[i] >> 4] || text[i * 2 + 1] != digits[digest[i] & 15]) return false;
        return true;
    };
    assert(matches_hex(vector_b_seed, "60ed9b73c5a9599b200b6cd563e7f0d5d9a67d2402d85fd4ef966c580080d0e5"));
    assert(matches_hex(vector_a_seed, "301784168005ec833ab0aa60006f7fe7faaa95307d8c1fc6819b2ffdd717eccf"));
    native::Digest entropy{};
    native::Identity identity;
    identity.job_id = "preparation-contract";
    identity.connection_generation = 1;
    for (uint32_t version : {2u, 3u}) for (native::Shape shape : {native::Shape{256, 256, 2048}, native::Shape{384, 512, 4096}}) {
        // Largest wire target that can safely be scaled for this configuration.
        identity.wire_target_le.fill(255);
        identity.wire_target_le[31] = 0;
        identity.wire_target_le[30] = 0;
        identity.wire_target_le[29] = shape.k == 2048 ? 63 : 31;
        identity.target_le = native::jackpot_target(identity.wire_target_le, shape.k);
        assert(native::jackpot_work(shape.k) == uint64_t(128) * shape.k);
        assert(native::meets_target(identity.target_le, identity.target_le));
        auto above = identity.target_le;
        for (auto& byte : above) { if (++byte) break; }
        assert(!native::meets_target(above, identity.target_le));
        auto invalid = identity;
        invalid.target_le = invalid.wire_target_le; // Missing adjustment.
        bool invalid_rejected = false;
        try { native::Attempt bad(invalid, shape, {}, {}, false); }
        catch (const std::exception& error) {
            invalid_rejected = std::string(error.what()) == "Pearl jackpot target/configuration mismatch";
        }
        assert(invalid_rejected);
        identity.cert_version = version;
        native::Attempt reference(identity, shape,
            native::fresh_base(entropy, identity, shape, true),
            native::fresh_base(entropy, identity, shape, false));
        native::Attempt proof_only(identity, shape, reference.a_tree.bytes(), reference.bt_tree.bytes(), false);
        native::Winner winner{0, 0, reference.winner_digest(0, 0)};
        assert(proof_only.a.empty() && proof_only.bt.empty());
        assert(proof_only.proof(winner) == reference.proof(winner));
        auto snapshot = [&](const native::MatrixTree& tree, uint32_t rows, std::vector<uint32_t> selected) {
            std::vector<uint8_t> bytes, levels;
            for (auto index : selected) {
                auto row = tree.row(index);
                bytes.insert(bytes.end(), row.begin(), row.end());
            }
            for (const auto& level : tree.layers()) {
                const auto* first = reinterpret_cast<const uint8_t*>(level.data());
                levels.insert(levels.end(), first, first + level.size() * 32);
            }
            auto corrupted = bytes;
            corrupted[0] ^= 1;
            bool rejected = false;
            try { native::MatrixTree::from_snapshot(rows, shape.k, selected, corrupted, levels, reference.job_key); }
            catch (const std::exception&) { rejected = true; }
            assert(rejected);
            auto corrupted_levels = levels;
            corrupted_levels.back() ^= 1;
            rejected = false;
            try { native::MatrixTree::from_snapshot(rows, shape.k, selected, bytes, corrupted_levels, reference.job_key); }
            catch (const std::exception&) { rejected = true; }
            assert(rejected);
            return native::MatrixTree::from_snapshot(rows, shape.k, selected, bytes, levels, reference.job_key);
        };
        std::vector<uint32_t> columns;
        for (uint32_t i = 0; i < 64; ++i) columns.push_back(i / 8 * 32 + i % 8 * 2);
        native::Attempt imported(identity, shape, snapshot(reference.a_tree, shape.m, {0, 1}),
            snapshot(reference.bt_tree, shape.n, columns));
        assert(imported.proof(winner) == reference.proof(winner));
        // Independent full-tree reconstruction checks nonce encoding and every
        // updated ancestor, including a nonzero chunk counter at the last leaf.
        for (bool zero : {false, true}) {
            auto base = zero ? std::vector<uint8_t>(reference.a_tree.bytes().size(), 0) : reference.a_tree.bytes();
            for (uint32_t leaf : {0u, uint32_t(base.size() / 1024 - 1)}) {
                native::MatrixTree original(base, shape.m, shape.k, reference.job_key);
                std::vector<uint8_t> flat;
                for (const auto& layer : original.layers()) {
                    auto begin = reinterpret_cast<const uint8_t*>(layer.data());
                    flat.insert(flat.end(), begin, begin + layer.size() * 32);
                }
                for (uint64_t nonce : {1ull, 127ull, 128ull, 0xffffffffffffffffull}) {
                    constexpr uint64_t high = 0x123456789abcdef0ull;
                    auto expected = base;
                    for (unsigned i = 0; i < 19; ++i) {
                        unsigned value = 0;
                        for (unsigned bit = 0; bit < 7 && i * 7 + bit < 128; ++bit) {
                            unsigned at = i * 7 + bit;
                            value += unsigned((at < 64 ? nonce >> at : high >> (at - 64)) & 1) << bit;
                        }
                        expected[leaf * 1024 + i] = uint8_t(int(value) - 64);
                    }
                    launch(1, pearl_prepare_incremental, base.data(), reference.job_key.data(), flat.data(),
                        uint32_t(base.size() / 1024), leaf, nonce, high);
                    assert(base == expected);
                    native::Digest fused_seed{}, separate_seed{};
                    launch(1, pearl_prepare_incremental_seed, base.data(), reference.job_key.data(), flat.data(),
                        uint32_t(base.size() / 1024), leaf, nonce, high, reference.b_seed.data(), fused_seed.data(), identity.cert_version, shape.m);
                    launch(1, pearl_prepare_seed, reference.b_seed.data(), flat.data() + flat.size() - 32,
                        separate_seed.data(), identity.cert_version, shape.m, 1u);
                    assert(fused_seed == separate_seed);
                    native::MatrixTree rebuilt(expected, shape.m, shape.k, reference.job_key);
                    size_t offset = 0;
                    for (const auto& layer : rebuilt.layers()) {
                        assert(std::memcmp(flat.data() + offset, layer.data(), layer.size() * 32) == 0);
                        offset += layer.size() * 32;
                    }
                }
                if (zero && leaf == 0) {
                    native::Attempt changed(identity, shape, base, reference.bt_tree.bytes());
                    std::vector<int8_t> dense(shape.m * 128);
                    std::vector<uint32_t> pairs(shape.k * 2), packed(base.size() / 4);
                    launch(shape.m * 4, pearl_prepare_dense, changed.a_seed.data(),
                        reinterpret_cast<uint8_t*>(dense.data()), shape.m, 2u);
                    launch(shape.k / 8, pearl_prepare_sparse, changed.a_seed.data(), pairs.data(), shape.k, 1u);
                    launch(base.size() / 4, pearl_prepare_zero_operand, reinterpret_cast<int8_t*>(base.data()),
                        dense.data(), pairs.data(), packed.data(), shape.m, shape.k);
                    assert(std::memcmp(packed.data(), changed.a.data(), base.size()) == 0);
                    assert(changed.winner_digest(0, 0) != winner.digest);
                }
            }
        }
        native::Digest b_seed{};
        for (bool a : {false, true}) {
            const auto rows = a ? shape.m : shape.n;
            const auto descriptor = native::base_output(entropy, identity, shape, a);
            pearl_prepare::Output device_descriptor;
            std::memcpy(&device_descriptor, &descriptor, sizeof(descriptor));
            std::vector<uint8_t> base(size_t(rows) * shape.k);
            launch(base.size() / 64, pearl_prepare_base, &device_descriptor, base.data(), uint32_t(base.size()));
            const auto& tree = a ? reference.a_tree : reference.bt_tree;
            assert(base == tree.bytes());
            uint32_t count = base.size() / 1024;
            std::vector<uint8_t> children(count * 32);
            launch(count, pearl_prepare_leaves, base.data(), reference.job_key.data(), children.data(), count);
            size_t level = 0;
            for (;;) {
                assert(std::memcmp(children.data(), tree.layers()[level].data(), children.size()) == 0);
                if (count == 1) break;
                std::vector<uint8_t> parents(((count + 1) / 2) * 32);
                launch((count + 1) / 2, pearl_prepare_parents, children.data(), reference.job_key.data(), parents.data(), count);
                children = std::move(parents);
                count = (count + 1) / 2;
                ++level;
            }
            native::Digest seed{};
            launch(1, pearl_prepare_seed, a ? b_seed.data() : reference.job_key.data(), children.data(), seed.data(),
                identity.cert_version, a ? shape.m : shape.n, uint32_t(a));
            assert(seed == (a ? reference.a_seed : reference.b_seed));
            if (!a) b_seed = seed;
            std::vector<int8_t> dense(rows * 128), operand(base.size());
            std::vector<uint32_t> pairs(shape.k * 2);
            launch(rows * 4, pearl_prepare_dense, seed.data(), reinterpret_cast<uint8_t*>(dense.data()), rows, uint32_t(a));
            launch(shape.k / 8, pearl_prepare_sparse, seed.data(), pairs.data(), shape.k, uint32_t(a));
            launch(base.size(), pearl_prepare_materialize, reinterpret_cast<int8_t*>(base.data()), dense.data(),
                pairs.data(), operand.data(), rows, shape.k, uint32_t(a));
            assert(operand == (a ? reference.a : reference.bt));
        }
        ++identity.attempt_id;
    }
    std::cout << "PEARL_PREP_CPU_PASS base, every_merkle_level, seeds, full_noise; gpu_launches=0\n";
}
