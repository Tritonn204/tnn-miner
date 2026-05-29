#pragma once
// ============================================================================
// QHash CPU Reference — QubitCoin Proof-of-Work Hash
// ============================================================================
// Matches the official miner at github.com/super-quantum/qubitcoin-miner
// (algo/qhash/qhash.c + algo/qhash/qhash-custatevec.c)
//
// Pipeline:
//   1. SHA-256(80-byte block header) → 32 bytes
//   2. Split bytes → 64 nibbles (4-bit each, 0–15)
//   3. Quantum circuit: 16 qubits, 2 layers of {RY·RZ per qubit, CNOT chain}
//      Angles: θ = -nibble · π/16  (matches official custatevec impl)
//   4. Measure ⟨Z_q⟩ for each qubit q=0..15
//   5. Encode as int16_t fixed-point (15 fractional bits), little-endian
//   6. Append to SHA buffer → final SHA-256 → 32-byte hash
// ============================================================================

#include <openssl/sha.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <complex>

// ── Algorithm constants (from qhash-gate.h) ───────────────────────────
static constexpr int QHASH_NUM_QUBITS     = 16;
static constexpr int QHASH_NUM_LAYERS     = 2;
static constexpr int QHASH_INPUT_SIZE     = 80;   // block header bytes
static constexpr int QHASH_SHA256_BYTES   = 32;
static constexpr int QHASH_NUM_NIBBLES    = 2 * QHASH_SHA256_BYTES;  // 64
static constexpr int QHASH_STATE_DIM      = 1 << QHASH_NUM_QUBITS;   // 65536
static constexpr int QHASH_FRACTION_BITS  = 15;

using qhash_complex = std::complex<double>;

// ── Precomputed trig table: cos(k·π/32), sin(k·π/32) for k=0..15 ────
// θ = -k · π/16, so θ/2 = -k · π/32. Only need cos(θ/2) and sin(θ/2).
// Since cos is even and sin is odd, cos(-θ/2)=cos(θ/2), sin(-θ/2)=-sin(θ/2).
// So RY uses [c, -s; s, c] and RZ uses exp(±iθ/2) with c=cos(kπ/32), s=sin(kπ/32).
struct QHashTrig {
    double cos_half[16];   // cos(k · π/32)
    double sin_half[16];   // sin(k · π/32)

    QHashTrig() {
        for (int k = 0; k < 16; ++k) {
            double angle = k * M_PI / 32.0;
            cos_half[k] = std::cos(angle);
            sin_half[k] = std::sin(angle);
        }
    }
};
static const QHashTrig qhash_trig;

// ── Split 32 bytes into 64 nibbles ────────────────────────────────────
inline void qhash_split_nibbles(const uint8_t input[QHASH_SHA256_BYTES],
                                uint8_t nibbles[QHASH_NUM_NIBBLES]) {
    for (int i = 0; i < QHASH_SHA256_BYTES; ++i) {
        nibbles[2 * i]     = (input[i] >> 4) & 0xF;
        nibbles[2 * i + 1] = input[i] & 0xF;
    }
}

// ── Run the full quantum circuit on a state vector ────────────────────
// data: 64 nibbles (values 0-15) driving rotation angles
// sv:   state vector array of DIM complex doubles (DIM = 65536)
inline void qhash_run_circuit(const uint8_t data[QHASH_NUM_NIBBLES],
                              qhash_complex sv[QHASH_STATE_DIM]) {
    constexpr int DIM    = QHASH_STATE_DIM;
    constexpr int NQ     = QHASH_NUM_QUBITS;
    constexpr int NL     = QHASH_NUM_LAYERS;
    constexpr int HALF   = DIM >> 1;

    // Initialize |0⟩ state
    sv[0] = qhash_complex(1.0, 0.0);
    for (int i = 1; i < DIM; ++i)
        sv[i] = qhash_complex(0.0, 0.0);

    for (int l = 0; l < NL; ++l) {

        // ── Fused RY(θ)·RZ(φ) on each qubit ───────────────────────────
        for (int q = 0; q < NQ; ++q) {
            int k_ry = data[(2 * l * NQ + q)       % QHASH_NUM_NIBBLES];
            int k_rz = data[((2 * l + 1) * NQ + q) % QHASH_NUM_NIBBLES];

            const double cry = qhash_trig.cos_half[k_ry];
            const double sry = qhash_trig.sin_half[k_ry];  // sin(θ/2), θ = -k·π/16
            const double crz = qhash_trig.cos_half[k_rz];
            const double srz = qhash_trig.sin_half[k_rz];

            const int qmask = 1 << q;

            for (int t = 0; t < HALF; ++t) {
                // Map t → (lo, hi) where lo has bit q=0, hi has bit q=1
                int lo = ((t >> q) << (q + 1)) | (t & (qmask - 1));
                int hi = lo | qmask;

                double ar = sv[lo].real(), ai = sv[lo].imag();
                double br = sv[hi].real(), bi = sv[hi].imag();

                // RY(θ) with θ = -k·π/16:
                //   cos(θ/2) = cry,  sin(θ/2) = -sry
                //   matrix = [[cry, sry], [-sry, cry]]
                double ya =  cry * ar + sry * br;
                double yb =  cry * ai + sry * bi;
                double yc = -sry * ar + cry * br;
                double yd = -sry * ai + cry * bi;

                // RZ(φ) with φ = -k·π/16:
                //   |0⟩ → e^{+i·kπ/32} =  crz + i·srz
                //   |1⟩ → e^{-i·kπ/32} =  crz - i·srz
                sv[lo] = qhash_complex(crz * ya - srz * yb,
                                       crz * yb + srz * ya);
                sv[hi] = qhash_complex(crz * yc + srz * yd,
                                       crz * yd - srz * yc);
            }
        }

        // ── CNOT chain: control=q → target=q+1 ────────────────────────
        for (int q = 0; q < NQ - 1; ++q) {
            int ctrl_mask = 1 << q;
            int tgt_mask  = 1 << (q + 1);

            for (int i = 0; i < DIM; ++i) {
                // Swap amplitudes of |ctrl=1, tgt=0⟩ and |ctrl=1, tgt=1⟩
                if ((i & ctrl_mask) && !(i & tgt_mask)) {
                    int j = i | tgt_mask;
                    if (i < j) {  // swap each pair once
                        qhash_complex tmp = sv[i];
                        sv[i] = sv[j];
                        sv[j] = tmp;
                    }
                }
            }
        }
    }
}

// ── Measure ⟨Z_q⟩ = Σ |α_i|² · (-1)^{bit_q} for q=0..15 ──────────────
inline void qhash_measure_expectations(const qhash_complex sv[QHASH_STATE_DIM],
                                       double expectations[QHASH_NUM_QUBITS]) {
    constexpr int DIM = QHASH_STATE_DIM;

    for (int q = 0; q < QHASH_NUM_QUBITS; ++q) {
        int mask = 1 << q;
        double acc = 0.0;
        for (int i = 0; i < DIM; ++i) {
            double p = sv[i].real() * sv[i].real() + sv[i].imag() * sv[i].imag();
            acc += (i & mask) ? -p : p;
        }
        expectations[q] = acc;
    }
}

// ── Fixed-point conversion (int16_t, 15 fractional bits) ──────────────
inline int16_t qhash_to_fixed(double x) {
    constexpr int32_t mult = 1 << QHASH_FRACTION_BITS;  // 32768
    return (x >= 0.0) ? static_cast<int16_t>(x * mult + 0.5)
                      : static_cast<int16_t>(x * mult - 0.5);
}

// ── Full QHash computation: input[80] → output[32] ────────────────────
// workspace must point to QHashWorkspace allocated by the caller
// (65536 complex<double> = 1 MiB, reused across calls for performance)

struct QHashWorkspace {
    qhash_complex sv[QHASH_STATE_DIM];  // state vector, 1 MiB
};

inline void qhash_compute(const uint8_t input[QHASH_INPUT_SIZE],
                          uint8_t output[QHASH_SHA256_BYTES],
                          QHashWorkspace* ws) {
    // 1. First SHA-256 of the 80-byte block header
    uint8_t buf[QHASH_SHA256_BYTES + QHASH_NUM_QUBITS * sizeof(int16_t)]; // 64 bytes
    {
        SHA256_CTX ctx;
        SHA256_Init(&ctx);
        SHA256_Update(&ctx, input, QHASH_INPUT_SIZE);
        SHA256_Final(buf, &ctx);
    }

    // 2. Split into nibbles
    uint8_t nibbles[QHASH_NUM_NIBBLES];  // 64
    qhash_split_nibbles(buf, nibbles);

    // 3. Quantum circuit simulation
    qhash_run_circuit(nibbles, ws->sv);

    // 4. Measure expectations
    double expectations[QHASH_NUM_QUBITS];  // 16
    qhash_measure_expectations(ws->sv, expectations);

    // 5. Encode as int16_t fixed-point (little-endian), appended to buf
    for (int i = 0; i < QHASH_NUM_QUBITS; ++i) {
        int16_t fixed = qhash_to_fixed(expectations[i]);
        size_t off = QHASH_SHA256_BYTES + i * sizeof(int16_t);
        buf[off]     = fixed & 0xFF;
        buf[off + 1] = (fixed >> 8) & 0xFF;
    }

    // 6. Final SHA-256 of the 64-byte buffer
    {
        SHA256_CTX ctx;
        SHA256_Init(&ctx);
        SHA256_Update(&ctx, buf, sizeof(buf));
        SHA256_Final(output, &ctx);
    }
}
