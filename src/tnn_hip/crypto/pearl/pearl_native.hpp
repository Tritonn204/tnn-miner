#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace tnn::pearl::native {

using Digest = std::array<uint8_t, 32>;
using Header = std::array<uint8_t, 76>;

// Mining base signal only. Commitment-derived rank-128 noise is still mandatory.
inline constexpr uint8_t mining_base_value = 32;

enum class CandidateLayout { legacy_2x64, native_4x32 };

struct Shape {
    uint32_t m = 256;
    uint32_t n = 256;
    uint32_t k = 2048;
    CandidateLayout layout = CandidateLayout::legacy_2x64;

    void validate() const;
    uint64_t macs() const;
};

// Copied with the owned attempt, never recovered from mutable global job state.
struct Identity {
    std::string job_id;
    uint64_t connection_generation = 0;
    uint64_t attempt_id = 0;
    int device_id = 0;
    bool is_dev = false;
    Header header{};
    Digest wire_target_le{}; // Original pool target, used for job binding.
    Digest target_le{};      // Configuration-adjusted jackpot bound.
    uint32_t cert_version = 2; // Explicit legacy fixtures; live jobs always supply their version.
};

struct Winner {
    uint32_t row = 0;
    uint32_t col = 0;
    Digest digest{};
};

Digest hash(std::span<const uint8_t> bytes);
Digest keyed_hash(std::span<const uint8_t> bytes, const Digest& key);
// Final BLAKE3 output block descriptor: small CPU work, parallel GPU XOF.
struct BaseOutput {
    std::array<uint32_t, 8> cv;
    std::array<uint8_t, 64> block;
    uint32_t length;
    uint32_t flags;
};
static_assert(sizeof(BaseOutput) == 104);
BaseOutput base_output(const Digest&, const Identity&, Shape, bool is_a);
Digest job_key(const Header&, Shape);
bool meets_target(const Digest& digest, const Digest& target_le);
std::array<uint8_t, 52> configuration(
    uint32_t k, CandidateLayout layout = CandidateLayout::legacy_2x64);
uint64_t jackpot_work(uint32_t k, CandidateLayout layout = CandidateLayout::legacy_2x64);
Digest jackpot_target(const Digest& wire_target, uint32_t k,
                      CandidateLayout layout = CandidateLayout::legacy_2x64);
std::vector<uint32_t> candidate_rows(Shape shape, uint32_t origin);
std::vector<uint32_t> candidate_columns(Shape shape, uint32_t origin);

// Base matrix bytes are row-major A and row-major B-transpose.
// The tree owns its bytes so proof construction cannot outlive the input.
class MatrixTree {
public:
    MatrixTree(std::vector<uint8_t> bytes, uint32_t rows, uint32_t k, const Digest& key);

    const Digest& root() const { return layers_.back().front(); }
    const std::vector<uint8_t>& bytes() const { return bytes_; }
    const std::vector<std::vector<Digest>>& layers() const { return layers_; }
    std::span<const uint8_t> row(uint32_t index) const;
    static MatrixTree from_snapshot(uint32_t rows, uint32_t k, std::vector<uint32_t> selected,
        std::vector<uint8_t> bytes, std::vector<uint8_t> tree, const Digest& key);
    void append_proof(std::vector<uint8_t>& out, std::span<const uint32_t> rows) const;

private:
    MatrixTree() = default;
    uint32_t rows_;
    uint32_t k_;
    std::vector<uint8_t> bytes_;
    std::vector<std::vector<Digest>> layers_;
    std::vector<uint32_t> selected_;
};

class Attempt {
public:
    // Imported bases support differential tests; the miner uses fresh_bases().
    Attempt(Identity identity, Shape shape, std::vector<uint8_t> a,
            std::vector<uint8_t> bt, bool materialize_operands = true);
    Attempt(Identity identity, Shape shape, MatrixTree a, MatrixTree bt);

    const Identity identity;
    const Shape shape;
    const Digest job_key;
    const MatrixTree a_tree;
    const MatrixTree bt_tree;
    const Digest b_seed;
    const Digest a_seed;

    // Exact qualified GEMM layouts: A column-major, B-transpose row-major.
    const std::vector<int8_t> a;
    const std::vector<int8_t> bt;

    Digest winner_digest(uint32_t row, uint32_t col) const;
    std::vector<uint8_t> proof(const Winner& winner) const;
};

// Caller supplies a session-unique seed and monotonically increasing attempt ID.
// B is stable within a job; A changes for every attempt. No repeated short pattern.
std::vector<uint8_t> fresh_base(const Digest& session_seed, const Identity& identity,
                              Shape shape, bool is_a);

} // namespace tnn::pearl::native
