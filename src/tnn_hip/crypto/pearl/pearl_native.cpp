// Reference-compatible algorithms; see PEARL_REFERENCE_LICENSE.txt.
#include "pearl_native.hpp"
#include "pearl_target.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <set>
#include <stdexcept>

extern "C" {
#include "blake3.h"
#include "blake3_impl.h"
}

namespace tnn::pearl::native {
namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

uint32_t read32(const uint8_t* bytes) {
    return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
           (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
}

void write32(uint8_t* out, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) {
        out[i] = uint8_t(value >> (8 * i));
    }
}

void append64(std::vector<uint8_t>& out, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) {
        out.push_back(uint8_t(value >> (8 * i)));
    }
}

Digest compress(const Digest& key, std::span<const uint8_t> input,
                uint64_t counter, uint8_t flags) {
    require(!input.empty() && input.size() % 64 == 0, "BLAKE3 block shape");
    uint32_t cv[8];
    for (unsigned i = 0; i < 8; ++i) {
        cv[i] = read32(key.data() + 4 * i);
    }
    for (size_t offset = 0; offset < input.size(); offset += 64) {
        uint8_t block_flags = flags;
        if (!(flags & PARENT)) {
            if (offset == 0) block_flags |= CHUNK_START;
            if (offset + 64 == input.size()) block_flags |= CHUNK_END;
        }
        blake3_compress_in_place(cv, input.data() + offset, 64, counter, block_flags);
    }
    Digest result;
    for (unsigned i = 0; i < 8; ++i) {
        write32(result.data() + 4 * i, cv[i]);
    }
    return result;
}

Digest join_hash(const Digest& left, const Digest& right) {
    std::array<uint8_t, 64> bytes;
    std::copy(left.begin(), left.end(), bytes.begin());
    std::copy(right.begin(), right.end(), bytes.begin() + 32);
    return hash(bytes);
}

Digest make_job_key(const Header& header, Shape shape) {
    shape.validate();
    auto config = configuration(shape.k);
    std::vector<uint8_t> bytes(header.begin(), header.end());
    bytes.insert(bytes.end(), config.begin(), config.end());
    return hash(bytes);
}

Identity checked_identity(Identity identity, Shape shape) {
    require(identity.cert_version >= 1 && identity.cert_version <= 3, "Unsupported certificate version");
    require(!identity.job_id.empty() && identity.connection_generation != 0 &&
            identity.device_id >= 0, "Missing attempt identity");
    require(std::any_of(identity.target_le.begin(), identity.target_le.end(),
                        [](uint8_t value) { return value != 0; }), "Zero share target");
    require(identity.target_le == jackpot_target(identity.wire_target_le, shape.k),
            "Pearl jackpot target/configuration mismatch");
    return identity;
}

Digest bound_root(const Digest& root, uint32_t dimension, bool is_a, uint32_t version) {
    if (version != 3) return root;
    static const Digest salt_a = [] {
        constexpr std::string_view context = "pearl/cert-v3/noise-seed/A";
        return hash({reinterpret_cast<const uint8_t*>(context.data()), context.size()});
    }();
    static const Digest salt_b = [] {
        constexpr std::string_view context = "pearl/cert-v3/noise-seed/B";
        return hash({reinterpret_cast<const uint8_t*>(context.data()), context.size()});
    }();
    std::array<uint8_t, 64> message{};
    std::copy(root.begin(), root.end(), message.begin());
    write32(message.data() + 32, dimension);
    return keyed_hash(message, is_a ? salt_a : salt_b);
}

Digest random_hash(uint32_t index, const Digest& key, bool is_a, bool sparse) {
    std::array<uint8_t, 64> message{};
    write32(message.data() + (sparse ? 4 : 0), index + 1);
    std::memcpy(message.data() + 32, is_a ? "A_tensor" : "B_tensor", 8);
    return keyed_hash(message, key);
}

std::vector<int8_t> materialize(const MatrixTree& tree, uint32_t rows, uint32_t k,
                              const Digest& seed, bool is_a, std::span<const uint32_t> selected = {}) {
    std::vector<std::array<uint32_t, 2>> pairs(k);
    for (uint32_t block = 0; block < k / 8; ++block) {
        auto random = random_hash(block, seed, is_a, true);
        for (unsigned j = 0; j < 8; ++j) {
            uint32_t word = read32(random.data() + 4 * j);
            uint32_t first = word & 127;
            uint32_t second = first ^ (1 + uint32_t((uint64_t(127) * word) >> 32));
            pairs[block * 8 + j] = {first, second};
        }
    }

    const uint32_t output_rows = selected.empty() ? rows : uint32_t(selected.size());
    std::vector<int8_t> result(size_t(output_rows) * k);
    for (uint32_t output_row = 0; output_row < output_rows; ++output_row) {
        const uint32_t row = selected.empty() ? output_row : selected[output_row];
        const auto base_row = tree.row(row);
        std::array<int, 128> dense;
        for (unsigned block = 0; block < 4; ++block) {
            auto random = random_hash(row * 4 + block, seed, is_a, false);
            for (unsigned j = 0; j < 32; ++j) {
                dense[block * 32 + j] = int(random[j] & 63) - 32;
            }
        }
        for (uint32_t column = 0; column < k; ++column) {
            int base = int(std::bit_cast<int8_t>(base_row[column]));
            auto pair = pairs[column];
            int value = base + dense[pair[0]] - dense[pair[1]];
            require(value >= -128 && value <= 127, "Noise overflow; clipping is forbidden");
            size_t index = is_a ? size_t(column) * output_rows + output_row : size_t(output_row) * k + column;
            result[index] = int8_t(value);
        }
    }
    return result;
}

void check_position(Shape shape, uint32_t row, uint32_t col) {
    require(row < shape.m - 1 && row % 2 == 0, "Invalid jackpot row");
    require(col < shape.n - 238 && (col & 238) == 0, "Invalid jackpot column");
}

} // namespace

void Shape::validate() const {
    require(m >= 256 && m <= 8192 && m % 128 == 0 &&
            n >= 256 && n <= 8192 && n % 256 == 0 &&
            (k == 2048 || k == 4096), "Unqualified Pearl shape");
}

uint64_t Shape::macs() const {
    validate();
    return uint64_t(m) * n * k;
}

Digest hash(std::span<const uint8_t> bytes) {
    blake3_hasher state;
    blake3_hasher_init(&state);
    blake3_hasher_update(&state, bytes.data(), bytes.size());
    Digest result;
    blake3_hasher_finalize(&state, result.data(), result.size());
    return result;
}

Digest keyed_hash(std::span<const uint8_t> bytes, const Digest& key) {
    blake3_hasher state;
    blake3_hasher_init_keyed(&state, key.data());
    blake3_hasher_update(&state, bytes.data(), bytes.size());
    Digest result;
    blake3_hasher_finalize(&state, result.data(), result.size());
    return result;
}

bool meets_target(const Digest& digest, const Digest& target_le) {
    for (int i = 31; i >= 0; --i) {
        if (digest[i] != target_le[i]) return digest[i] < target_le[i];
    }
    return true;
}

std::array<uint8_t, 52> configuration(uint32_t k) {
    require(k == 2048 || k == 4096, "Unqualified Pearl K");
    std::array<uint8_t, 52> result{};
    write32(result.data(), k);
    result[4] = 128;
    result[9] = 1;
    result[14] = 1;
    result[15] = 7;
    result[16] = 1;
    result[17] = 7;
    return result;
}

uint64_t jackpot_work(uint32_t k) {
    const auto config = configuration(k);
    const uint32_t rank = uint32_t(config[4]) | (uint32_t(config[5]) << 8);
    uint32_t rows = 1, columns = 1;
    for (unsigned i = 0; i < 3; ++i) {
        rows *= uint32_t(config[9 + 2 * i]) + 1;
        columns *= uint32_t(config[15 + 2 * i]) + 1;
    }
    return tnn::pearl::jackpot_work(rows, columns, k, rank);
}

Digest jackpot_target(const Digest& wire_target, uint32_t k) {
    return scale_jackpot_target(wire_target, jackpot_work(k));
}

MatrixTree::MatrixTree(std::vector<uint8_t> bytes, uint32_t rows, uint32_t k, const Digest& key)
    : rows_(rows), k_(k), bytes_(std::move(bytes)) {
    require(rows >= 256 && rows <= 8192 && rows % 128 == 0 &&
            (k == 2048 || k == 4096), "Unqualified matrix shape");
    require(bytes_.size() == size_t(rows) * k, "Base matrix length");
    for (uint8_t value : bytes_) {
        int signed_value = std::bit_cast<int8_t>(value);
        require(signed_value >= -64 && signed_value <= 63, "Base matrix int7 range");
    }
    size_t count = bytes_.size() / 1024;
    std::vector<Digest> leaves;
    leaves.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        leaves.push_back(compress(key, std::span(bytes_).subspan(i * 1024, 1024), i, KEYED_HASH));
    }
    layers_.push_back(std::move(leaves));
    while (layers_.back().size() > 1) {
        const auto& previous = layers_.back();
        std::vector<Digest> next;
        next.reserve((previous.size() + 1) / 2);
        for (size_t i = 0; i < previous.size(); i += 2) {
            if (i + 1 == previous.size()) {
                next.push_back(previous[i]);
                continue;
            }
            std::array<uint8_t, 64> block;
            std::copy(previous[i].begin(), previous[i].end(), block.begin());
            std::copy(previous[i + 1].begin(), previous[i + 1].end(), block.begin() + 32);
            uint8_t flags = KEYED_HASH | PARENT;
            if (previous.size() == 2) flags |= ROOT;
            next.push_back(compress(key, block, 0, flags));
        }
        layers_.push_back(std::move(next));
    }
    require(root() == keyed_hash(bytes_, key), "Merkle root/hash mismatch");
}

void MatrixTree::append_proof(std::vector<uint8_t>& out, std::span<const uint32_t> rows) const {
    require(!rows.empty() && rows.size() <= 64, "Proof row count");
    std::set<uint64_t> leaves;
    uint32_t previous = 0;
    for (size_t at = 0; at < rows.size(); ++at) {
        uint32_t row = rows[at];
        require(row < rows_ && (at == 0 || row > previous), "Proof row order/bounds");
        previous = row;
        for (uint64_t index = uint64_t(row) * k_ / 1024;
             index < uint64_t(row + 1) * k_ / 1024; ++index) {
            leaves.insert(index);
        }
    }
    // bincode 1.x Vec<Vec<u8>>: outer length, then a length for each chunk.
    append64(out, leaves.size());
    for (uint64_t leaf : leaves) {
        append64(out, 1024);
        const auto base_row = row(uint32_t(leaf * 1024 / k_));
        auto first = base_row.begin() + (leaf * 1024 % k_);
        out.insert(out.end(), first, first + 1024);
    }
    append64(out, leaves.size());
    for (uint64_t leaf : leaves) append64(out, leaf);
    append64(out, layers_.front().size());
    out.insert(out.end(), root().begin(), root().end());

    std::vector<Digest> siblings;
    for (size_t level = 0; level + 1 < layers_.size(); ++level) {
        std::set<uint64_t> parents;
        for (uint64_t index : leaves) {
            uint64_t sibling = index ^ 1;
            if (sibling < layers_[level].size() && !leaves.contains(sibling)) {
                siblings.push_back(layers_[level][sibling]);
            }
            parents.insert(index / 2);
        }
        leaves = std::move(parents);
    }
    append64(out, siblings.size());
    for (const auto& sibling : siblings) out.insert(out.end(), sibling.begin(), sibling.end());
    append64(out, rows.size());
    for (uint32_t row : rows) append64(out, row);
}

Attempt::Attempt(Identity id, Shape dimensions, std::vector<uint8_t> base_a,
                 std::vector<uint8_t> base_bt, bool materialize_operands)
    : identity(checked_identity(std::move(id), dimensions)), shape(dimensions), job_key(make_job_key(identity.header, shape)),
      a_tree(std::move(base_a), shape.m, shape.k, job_key),
      bt_tree(std::move(base_bt), shape.n, shape.k, job_key),
      b_seed(join_hash(job_key, bound_root(bt_tree.root(), shape.n, false, identity.cert_version))),
      a_seed(join_hash(b_seed, bound_root(a_tree.root(), shape.m, true, identity.cert_version))),
      a(materialize_operands ? materialize(a_tree, shape.m, shape.k, a_seed, true) : std::vector<int8_t>{}),
      bt(materialize_operands ? materialize(bt_tree, shape.n, shape.k, b_seed, false) : std::vector<int8_t>{}) {}

std::span<const uint8_t> MatrixTree::row(uint32_t index) const {
    require(index < rows_, "Matrix row bounds");
    size_t offset = index;
    if (!selected_.empty()) {
        auto found = std::lower_bound(selected_.begin(), selected_.end(), index);
        require(found != selected_.end() && *found == index, "Missing sampled matrix row");
        offset = size_t(found - selected_.begin());
    }
    return std::span(bytes_).subspan(offset * k_, k_);
}

MatrixTree MatrixTree::from_snapshot(uint32_t rows, uint32_t k, std::vector<uint32_t> selected,
    std::vector<uint8_t> bytes, std::vector<uint8_t> tree, const Digest& key) {
    require(rows >= 256 && rows <= 8192 && rows % 128 == 0 && (k == 2048 || k == 4096), "Invalid snapshot shape");
    require(!selected.empty() && selected.size() <= 64 && std::is_sorted(selected.begin(), selected.end()), "Invalid sampled rows");
    require(std::adjacent_find(selected.begin(), selected.end()) == selected.end() && selected.back() < rows, "Duplicate/out-of-range sample");
    require(bytes.size() == selected.size() * k, "Sample byte count");
    MatrixTree result;
    result.rows_ = rows;
    result.k_ = k;
    result.selected_ = std::move(selected);
    result.bytes_ = std::move(bytes);
    size_t offset = 0;
    for (size_t count = size_t(rows) * k / 1024;; count = (count + 1) / 2) {
        require(offset + count * 32 <= tree.size(), "Truncated tree snapshot");
        auto& layer = result.layers_.emplace_back(count);
        std::memcpy(layer.data(), tree.data() + offset, count * 32);
        offset += count * 32;
        if (count == 1) break;
    }
    require(offset == tree.size(), "Trailing tree snapshot");
    std::set<size_t> indices;
    for (auto selected_row : result.selected_) {
        const auto row_bytes = result.row(selected_row);
        for (auto byte : row_bytes) require(std::bit_cast<int8_t>(byte) >= -64 && std::bit_cast<int8_t>(byte) <= 63, "Sample int7 range");
        for (size_t chunk = 0; chunk < k / 1024; ++chunk) {
            size_t index = size_t(selected_row) * (k / 1024) + chunk;
            require(compress(key, row_bytes.subspan(chunk * 1024, 1024), index, KEYED_HASH) == result.layers_[0][index], "Snapshot leaf mismatch");
            indices.insert(index);
        }
    }
    for (size_t level = 0; level + 1 < result.layers_.size(); ++level) {
        std::set<size_t> parents;
        const auto& children = result.layers_[level];
        for (auto index : indices) parents.insert(index / 2);
        for (auto index : parents) {
            Digest expected = children[index * 2];
            if (index * 2 + 1 < children.size()) {
                std::array<uint8_t, 64> block;
                std::memcpy(block.data(), children[index * 2].data(), 32);
                std::memcpy(block.data() + 32, children[index * 2 + 1].data(), 32);
                expected = compress(key, block, 0, KEYED_HASH | PARENT | (children.size() == 2 ? ROOT : 0));
            }
            require(expected == result.layers_[level + 1][index], "Snapshot Merkle path mismatch");
        }
        indices = std::move(parents);
    }
    return result;
}

Attempt::Attempt(Identity id, Shape dimensions, MatrixTree base_a, MatrixTree base_b)
    : identity(checked_identity(std::move(id), dimensions)), shape(dimensions), job_key(make_job_key(identity.header, shape)),
      a_tree(std::move(base_a)), bt_tree(std::move(base_b)),
      b_seed(join_hash(job_key, bound_root(bt_tree.root(), shape.n, false, identity.cert_version))),
      a_seed(join_hash(b_seed, bound_root(a_tree.root(), shape.m, true, identity.cert_version))) {}

Digest Attempt::winner_digest(uint32_t row, uint32_t col) const {
    check_position(shape, row, col);
    std::array<uint32_t, 2> selected_rows{row, row + 1};
    std::array<uint32_t, 64> selected_cols{};
    for (unsigned i = 0; i < 64; ++i) selected_cols[i] = col + (i / 8) * 32 + (i % 8) * 2;
    auto sampled_a = a.empty() ? materialize(a_tree, shape.m, shape.k, a_seed, true, selected_rows) : std::vector<int8_t>{};
    auto sampled_b = bt.empty() ? materialize(bt_tree, shape.n, shape.k, b_seed, false, selected_cols) : std::vector<int8_t>{};
    std::array<std::array<int32_t, 64>, 2> accum{};
    std::array<uint32_t, 16> transcript{};
    for (uint32_t checkpoint = 0; checkpoint < shape.k / 128; ++checkpoint) {
        uint32_t reduced = 0;
        for (unsigned u = 0; u < 2; ++u) {
            for (unsigned v = 0; v < 64; ++v) {
                uint32_t column = col + (v / 8) * 32 + (v % 8) * 2;
                for (uint32_t kk = checkpoint * 128; kk < (checkpoint + 1) * 128; ++kk) {
                    const int av = a.empty() ? sampled_a[size_t(kk) * 2 + u] : a[size_t(kk) * shape.m + row + u];
                    const int bv = bt.empty() ? sampled_b[size_t(v) * shape.k + kk] : bt[size_t(column) * shape.k + kk];
                    accum[u][v] += av * bv;
                }
                reduced ^= uint32_t(accum[u][v]);
            }
        }
        auto& slot = transcript[checkpoint % 16];
        slot = std::rotl(slot, 13) ^ reduced;
    }
    std::array<uint8_t, 64> bytes;
    for (unsigned i = 0; i < 16; ++i) write32(bytes.data() + 4 * i, transcript[i]);
    return keyed_hash(bytes, a_seed);
}

std::vector<uint8_t> Attempt::proof(const Winner& winner) const {
    require(winner.digest == winner_digest(winner.row, winner.col), "Winner digest mismatch");
    require(meets_target(winner.digest, identity.target_le), "Winner above share target");
    std::vector<uint8_t> out;
    append64(out, shape.m);
    append64(out, shape.n);
    append64(out, shape.k);
    append64(out, 128);
    std::array<uint32_t, 2> rows{winner.row, winner.row + 1};
    std::array<uint32_t, 64> cols;
    for (unsigned i = 0; i < 64; ++i) cols[i] = winner.col + (i / 8) * 32 + (i % 8) * 2;
    a_tree.append_proof(out, rows);
    bt_tree.append_proof(out, cols);
    require(out.size() <= 1'875'000, "Proof exceeds uncompressed pool limit");
    return out;
}

Digest job_key(const Header& header, Shape shape) { return make_job_key(header, shape); }

static blake3_hasher base_hasher(const Digest& session_seed, const Identity& identity,
                                 Shape shape, bool is_a) {
    shape.validate();
    std::vector<uint8_t> domain(identity.header.begin(), identity.header.end());
    auto config = configuration(shape.k);
    domain.insert(domain.end(), config.begin(), config.end());
    append64(domain, shape.m);
    append64(domain, shape.n);
    append64(domain, identity.connection_generation);
    append64(domain, identity.job_id.size());
    domain.insert(domain.end(), identity.job_id.begin(), identity.job_id.end());
    append64(domain, uint64_t(identity.device_id));
    append64(domain, identity.is_dev);
    append64(domain, is_a ? identity.attempt_id : 0);
    domain.push_back(is_a ? 'A' : 'B');
    require(domain.size() <= 1024, "Base derivation domain exceeds one BLAKE3 chunk");
    blake3_hasher state;
    blake3_hasher_init_keyed(&state, session_seed.data());
    blake3_hasher_update(&state, domain.data(), domain.size());
    return state;
}

BaseOutput base_output(const Digest& session_seed, const Identity& identity,
                       Shape shape, bool is_a) {
    const auto state = base_hasher(session_seed, identity, shape, is_a);
    BaseOutput output{};
    std::copy_n(state.chunk.cv, 8, output.cv.begin());
    std::copy_n(state.chunk.buf, state.chunk.buf_len, output.block.begin());
    output.length = state.chunk.buf_len;
    output.flags = KEYED_HASH | CHUNK_END | ROOT;
    if (state.chunk.blocks_compressed == 0) output.flags |= CHUNK_START;
    return output;
}

std::vector<uint8_t> fresh_base(const Digest& session_seed, const Identity& identity,
                              Shape shape, bool is_a) {
    auto state = base_hasher(session_seed, identity, shape, is_a);
    std::vector<uint8_t> result(size_t(is_a ? shape.m : shape.n) * shape.k);
    blake3_hasher_finalize(&state, result.data(), result.size());
    for (auto& value : result) value = uint8_t(int(value & 127) - 64);
    return result;
}

} // namespace tnn::pearl::native
