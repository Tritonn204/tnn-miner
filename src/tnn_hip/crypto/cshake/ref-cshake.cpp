/** libkeccak-tiny
 *
 * A single-file implementation of SHA-3 and SHAKE.
 *
 * Implementor: David Leon Gil
 * License: CC0, attribution kindly requested. Blame taken too,
 * but not liability.
 */
// #include <libkeccak/libkeccak.h>
// #include <libkeccak/common.h>

#include "common_ref.h"

#include "keccak-tiny.h"
#include "cshake256.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <crypto/cshake/cshake.h>

#define rotate_ref(x, n, w, wmod) ((((x) >> ((w) - ((n) % (w)))) | ((x) << ((n) % (w)))) & (wmod))


/**
 * Rotate a 64-bit word
 * 
 * @param   x:int_fast64_t  The value to rotate
 * @param   n:long          Rotation steps, may not be zero
 * @return   :int_fast64_t  The value rotated
 */
#define rotate64_ref(x, n) ((int_fast64_t)(((uint64_t)(x) >> (64L - (n))) | ((uint64_t)(x) << (n))))

#define LIST_5_REF X(0) X(1) X(2) X(3) X(4)

/**
 * X-macro-enabled listing of all intergers in [0, 7]
 */
#define LIST_8_REF LIST_5_REF X(5) X(6) X(7)

/**
 * X-macro-enabled listing of all intergers in [0, 24]
 */
#define LIST_25_REF LIST_8_REF X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15)\
                X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23) X(24)



#define X(N) (N % 5) * 5 + N / 5,
/**
 * The order the lanes should be read when absorbing or squeezing,
 * it transposes the lanes in the sponge
 */
static const long int LANE_TRANSPOSE_MAP[] = { LIST_25_REF };
#undef X

static const uint_fast64_t RC_REF[] = {
	0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
	0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
	0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
	0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
	0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
	0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

int libkeccak_state_initialise_ref(struct libkeccak_state *state, const struct libkeccak_spec *spec)
{
  long int x;

  state->r = spec->bitrate;
  state->n = spec->output;
  state->c = spec->capacity;
  state->b = state->r + state->c;
  state->w = x = state->b / 25;
  state->l = 0;

  if (x & 0xF0L)
  {
    state->l |= 4;
    x >>= 4;
  }
  if (x & 0x0CL)
  {
    state->l |= 2;
    x >>= 2;
  }
  if (x & 0x02L)
  {
    state->l |= 1;
  }

  state->nr = 12 + (state->l << 1);
  state->wmod = (state->w == 64) ? ~0LL : (int64_t)((1ULL << state->w) - 1);
  for (x = 0; x < 25; x++)
    state->S[x] = 0;
  state->mptr = 0;
  state->mlen = (size_t)(state->r * state->b) >> 2;

  // Explicitly cast malloc in C++
  state->M = (unsigned char *)malloc(state->mlen * sizeof(unsigned char));
  return state->M == nullptr ? -1 : 0; // Use nullptr instead of NULL in C++
}

static void
libkeccak_f_round(struct libkeccak_state *state, int_fast64_t rc)
{
	int_fast64_t *A = state->S;
	int_fast64_t B[25];
	int_fast64_t C[5];
	int_fast64_t da, db, dc, dd, de;
	int_fast64_t wmod = state->wmod;
	long int w = state->w;

	/* θ step (step 1 of 3). */
#define X(N) C[N] = A[N * 5] ^ A[N * 5 + 1] ^ A[N * 5 + 2] ^ A[N * 5 + 3] ^ A[N * 5 + 4];
	LIST_5_REF
#undef X

	/* θ step (step 2 of 3). */
	da = C[4] ^ rotate_ref(C[1], 1, w, wmod);
	dd = C[2] ^ rotate_ref(C[4], 1, w, wmod);
	db = C[0] ^ rotate_ref(C[2], 1, w, wmod);
	de = C[3] ^ rotate_ref(C[0], 1, w, wmod);
	dc = C[1] ^ rotate_ref(C[3], 1, w, wmod);

	/* ρ and π steps, with last two part of θ. */
#define X(bi, ai, dv, r) B[bi] = rotate_ref(A[ai] ^ dv, r, w, wmod)
	B[0] = A[0] ^ da;   X( 1, 15, dd, 28);  X( 2,  5, db,  1);  X( 3, 20, de, 27);  X( 4, 10, dc, 62);
	X( 5,  6, db, 44);  X( 6, 21, de, 20);  X( 7, 11, dc,  6);  X( 8,  1, da, 36);  X( 9, 16, dd, 55);
	X(10, 12, dc, 43);  X(11,  2, da,  3);  X(12, 17, dd, 25);  X(13,  7, db, 10);  X(14, 22, de, 39);
	X(15, 18, dd, 21);  X(16,  8, db, 45);  X(17, 23, de,  8);  X(18, 13, dc, 15);  X(19,  3, da, 41);
	X(20, 24, de, 14);  X(21, 14, dc, 61);  X(22,  4, da, 18);  X(23, 19, dd, 56);  X(24,  9, db,  2);
#undef X

	/* ξ step. */
#define X(N) A[N] = B[N] ^ ((~(B[(N + 5) % 25])) & B[(N + 10) % 25]);
	LIST_25_REF
#undef X

	/* ι step. */
	A[0] ^= rc;
}

static void
libkeccak_f_round64(struct libkeccak_state *state, int_fast64_t rc)
{
	int_fast64_t *A = state->S;
	int_fast64_t B[25];
	int_fast64_t C[5];
	int_fast64_t da, db, dc, dd, de;

	/* θ step (step 1 of 3). */
#define X(N) C[N] = A[N * 5] ^ A[N * 5 + 1] ^ A[N * 5 + 2] ^ A[N * 5 + 3] ^ A[N * 5 + 4];
	LIST_5_REF
#undef X

	/* θ step (step 2 of 3). */
	da = C[4] ^ rotate64_ref(C[1], 1);
	dd = C[2] ^ rotate64_ref(C[4], 1);
	db = C[0] ^ rotate64_ref(C[2], 1);
	de = C[3] ^ rotate64_ref(C[0], 1);
	dc = C[1] ^ rotate64_ref(C[3], 1);

	/* ρ and π steps, with last two part of θ. */
#define X(bi, ai, dv, r) B[bi] = rotate64_ref(A[ai] ^ dv, r)
	B[0] = A[0] ^ da;   X( 1, 15, dd, 28);  X( 2,  5, db,  1);  X( 3, 20, de, 27);  X( 4, 10, dc, 62);
	X( 5,  6, db, 44);  X( 6, 21, de, 20);  X( 7, 11, dc,  6);  X( 8,  1, da, 36);  X( 9, 16, dd, 55);
	X(10, 12, dc, 43);  X(11,  2, da,  3);  X(12, 17, dd, 25);  X(13,  7, db, 10);  X(14, 22, de, 39);
	X(15, 18, dd, 21);  X(16,  8, db, 45);  X(17, 23, de,  8);  X(18, 13, dc, 15);  X(19,  3, da, 41);
	X(20, 24, de, 14);  X(21, 14, dc, 61);  X(22,  4, da, 18);  X(23, 19, dd, 56);  X(24,  9, db,  2);
#undef X

	/* ξ step. */
#define X(N) A[N] = B[N] ^ ((~(B[(N + 5) % 25])) & B[(N + 10) % 25]);
	LIST_25_REF
#undef X

	/* ι step. */
	A[0] ^= rc;
}

static inline int_fast64_t
libkeccak_to_lane(const unsigned char *message, size_t msglen,
                  long int rr, long int ww, size_t off)
{
	long int n = (long)((msglen < (size_t)rr ? msglen : (size_t)rr) - off);
	int_fast64_t rc = 0;
	message += off;
	while (ww--) {
		rc <<= 8;
		rc |= __builtin_expect(ww < n, 1) ? (int_fast64_t)(unsigned char)(message[ww]) : 0L;
	}
	return rc;
}


/**
 * 64-bit lane version of `libkeccak_to_lane`
 * 
 * @param   message  The message
 * @param   msglen   The length of the message
 * @param   rr       Bitrate in bytes
 * @param   off      The offset in the message
 * @return           The lane
 */
LIBKECCAK_GCC_ONLY(__attribute__((__nonnull__, __nothrow__, __pure__, __hot__, __warn_unused_result__, __gnu_inline__)))
static inline int_fast64_t
libkeccak_to_lane64(const unsigned char *message, size_t msglen, long int rr, size_t off)
{
	long int n = (long)((msglen < (size_t)rr ? msglen : (size_t)rr) - off);
	int_fast64_t rc = 0;
	message += off;
#define X(N) if (__builtin_expect(N < n, 1)) rc |= (int_fast64_t)(unsigned char)(message[N]) << (N * 8);\
             else  return rc;
	LIST_8_REF
#undef X
	return rc;
}

static inline void
libkeccak_f(struct libkeccak_state *state)
{
	long int i = 0;
	long int nr = state->nr;
	long int wmod = state->wmod;
	if (nr == 24) {
		for (; i < nr; i++)
			libkeccak_f_round64(state, (int_fast64_t)(RC_REF[i]));
	} else {
		for (; i < nr; i++)
			libkeccak_f_round(state, (int_fast64_t)(RC_REF[i] & (uint_fast64_t)wmod));
	}
}

static void
libkeccak_absorption_phase_ref(struct libkeccak_state *state,
                           const unsigned char *message, size_t len)
{
	long int rr = state->r >> 3;
	long int ww = state->w >> 3;
	long int n = (long)len / rr;
	if (__builtin_expect(ww >= 8, 1)) { /* ww > 8 is impossible, it is just for optimisation possibilities. */
		while (n--) {
#define X(N) state->S[N] ^= libkeccak_to_lane64(message, len, rr, (size_t)(LANE_TRANSPOSE_MAP[N] * 8));
			LIST_25_REF
#undef X
			libkeccak_f(state);
			message += (size_t)rr;
			len -= (size_t)rr;
		}
	} else {
		while (n--) {
#define X(N) state->S[N] ^= libkeccak_to_lane(message, len, rr, ww, (size_t)(LANE_TRANSPOSE_MAP[N] * ww));
			LIST_25_REF
#undef X
			libkeccak_f(state);
			message += (size_t)rr;
			len -= (size_t)rr;
		}
	}
}

void
libkeccak_zerocopy_update_ref(struct libkeccak_state *state, const uint8_t *msg, size_t msglen)
{
	libkeccak_absorption_phase_ref(state, msg, msglen);
}

static size_t
encode_left(struct libkeccak_state *state, uint8_t *buf, size_t byterate, size_t value, size_t off)
{
    size_t x, n, j, i = off;

    for (x = value, n = 0; x; x >>= 8)
        n += 1;
    if (!n)
        n = 1;
    buf[i++] = static_cast<uint8_t>(n);  // Explicit cast to uint8_t in C++
    if (i == byterate) {
        libkeccak_zerocopy_update_ref(state, buf, byterate);
        i = 0;
    }

    for (j = 0; j < n;) {
        buf[i++] = static_cast<uint8_t>(value >> ((n - ++j) << 3));  // Explicit cast to uint8_t
        if (i == byterate) {
            libkeccak_zerocopy_update_ref(state, buf, byterate);
            i = 0;
        }
    }

    return i;
}

static size_t
encode_left_shifted(struct libkeccak_state *state, uint8_t *buf, size_t byterate, size_t value, size_t off, size_t bitoff)
{
    size_t x, n, j, i = off;
    uint16_t v;

    for (x = value, n = 0; x; x >>= 8)
        n += 1;
    if (!n)
        n = 1;
    v = static_cast<uint16_t>((n & 255UL) << bitoff);  // Explicit cast to uint16_t
    buf[i++] |= static_cast<uint8_t>(v);               // Explicit cast to uint8_t
    if (i == byterate) {
        libkeccak_zerocopy_update_ref(state, buf, byterate);
        i = 0;
    }
    buf[i] = static_cast<uint8_t>(n >> 8);  // Explicit cast

    for (j = 0; j < n;) {
        v = static_cast<uint16_t>(((value >> ((n - ++j) << 3)) & 255UL) << bitoff);  // Explicit cast to uint16_t
        buf[i++] |= static_cast<uint8_t>(v);                                          // Explicit cast to uint8_t
        if (i == byterate) {
            libkeccak_zerocopy_update_ref(state, buf, byterate);
            i = 0;
        }
        buf[i] = static_cast<uint8_t>(v >> 8);  // Explicit cast
    }

    return i;
}

static size_t
feed_text(struct libkeccak_state *state, uint8_t *buf, const uint8_t *text, size_t bytes, size_t bits, const char *suffix, size_t off, size_t byterate, size_t *bitoffp)
{
    size_t n, bitoff;

    if (off) {
        n = bytes < byterate - off ? bytes : byterate - off;
        memcpy(&buf[off], text, n);
        off += n;
        if (off == byterate) {
            libkeccak_zerocopy_update_ref(state, buf, byterate);
            off = 0;
        }
        text = &text[n];
        bytes -= n;
    }
    if (bytes) {
        n = bytes;
        n -= bytes %= byterate;
        libkeccak_zerocopy_update_ref(state, text, n);
        text = &text[n];
    }
    memcpy(&buf[off], text, bytes + !!bits);
    off += bytes;
    bitoff = bits;
    if (!bitoff)
        buf[off] = 0;
    for (; *suffix; suffix++) {
        if (*suffix == '1')
            buf[off] |= static_cast<uint8_t>(1 << bitoff);  // Explicit cast to uint8_t
        if (++bitoff == 8) {
            if (++off == byterate) {
                libkeccak_zerocopy_update_ref(state, buf, byterate);
                off = 0;
            }
            bitoff = 0;
            buf[off] = 0;
        }
    }

    *bitoffp = bitoff;
    return off;
}

static size_t
feed_text_shifted(struct libkeccak_state *state, uint8_t *buf, const uint8_t *text, size_t bytes, size_t bits, const char *suffix, size_t off, size_t byterate, size_t *bitoffp)
{
    size_t i, bitoff = *bitoffp;
    uint16_t v;

    for (i = 0; i < bytes; i++) {
        v = static_cast<uint16_t>(static_cast<uint16_t>(text[i]) << bitoff);  // Explicit cast to uint16_t
        buf[off] |= static_cast<uint8_t>(v);  // Explicit cast to uint8_t
        if (++off == byterate) {
            libkeccak_zerocopy_update_ref(state, buf, byterate);
            off = 0;
        }
        buf[off] = static_cast<uint8_t>(v >> 8);  // Explicit cast
    }
    if (bits) {
        v = static_cast<uint16_t>(static_cast<uint16_t>(text[bytes]) << bitoff);  // Explicit cast to uint16_t
        buf[off] |= static_cast<uint8_t>(v);  // Explicit cast to uint8_t
        bitoff += bits;
        if (bitoff >= 8) {
            if (++off == byterate) {
                libkeccak_zerocopy_update_ref(state, buf, byterate);
                off = 0;
            }
            bitoff &= 7;
            buf[off] = static_cast<uint8_t>(v >> 8);  // Explicit cast
        }
    }
    if (!bitoff)
        buf[off] = 0;
    for (; *suffix; suffix++) {
        if (*suffix == '1')
            buf[off] |= static_cast<uint8_t>(1 << bitoff);  // Explicit cast to uint8_t
        if (++bitoff == 8) {
            if (++off == byterate) {
                libkeccak_zerocopy_update_ref(state, buf, byterate);
                off = 0;
            }
            bitoff = 0;
            buf[off] = 0;
        }
    }

    *bitoffp = bitoff;
    return off;
}


void libkeccak_cshake_initialise_ref(struct libkeccak_state *state, // Remove restrict
                                 const uint8_t *n_text, size_t n_len, size_t n_bits, const char *n_suffix,
                                 uint8_t *s_text, size_t s_len, size_t s_bits, const char *s_suffix)
{
  size_t off = 0, bitoff = 0;
  size_t byterate = static_cast<size_t>(state->r) >> 3;

  if (!n_suffix)
    n_suffix = "";
  if (!s_suffix)
    s_suffix = "";

  if (!n_len && !s_len && !n_bits && !s_bits && !*n_suffix && !*s_suffix)
    return;

  // Adjust lengths for bits
  n_len += n_bits >> 3;
  s_len += s_bits >> 3;
  n_bits &= 7;
  s_bits &= 7;

  // Function calls with adjusted parameters
  off = encode_left(state, static_cast<uint8_t *>(state->M), byterate, byterate, off); // Explicit cast for state->M
  off = encode_left(state, static_cast<uint8_t *>(state->M), byterate, (n_len << 3) + n_bits + strlen(n_suffix), off);
  off = feed_text(state, static_cast<uint8_t *>(state->M), n_text, n_len, n_bits, n_suffix, off, byterate, &bitoff);

  if (!bitoff)
  {
    off = encode_left(state, static_cast<uint8_t *>(state->M), byterate, (s_len << 3) + s_bits + strlen(s_suffix), off);
    off = feed_text(state, static_cast<uint8_t *>(state->M), s_text, s_len, s_bits, s_suffix, off, byterate, &bitoff);
  }
  else
  {
    off = encode_left_shifted(state, static_cast<uint8_t *>(state->M), byterate, (s_len << 3) + s_bits + strlen(s_suffix), off, bitoff);
    off = feed_text_shifted(state, static_cast<uint8_t *>(state->M), s_text, s_len, s_bits, s_suffix, off, byterate, &bitoff);
  }

  if (bitoff)
    off++;
  if (off)
  {
    memset(&state->M[off], 0, byterate - off);
    libkeccak_zerocopy_update_ref(state, static_cast<uint8_t *>(state->M), byterate);
  }
}

void libkeccak_state_wipe_message_ref(volatile struct libkeccak_state *state)
{
  volatile unsigned char *M = state->M;
  size_t i;

  for (i = 0; i < state->mptr; i++)
    M[i] = 0;
}

void
libkeccak_state_wipe_sponge_ref(volatile struct libkeccak_state *state)
{
	volatile int64_t *S = state->S;
	size_t i;

	for (i = 0; i < 25; i++)
		S[i] = 0;
}


void
libkeccak_state_wipe_ref(volatile struct libkeccak_state *state)
{
	libkeccak_state_wipe_message_ref(state);
	libkeccak_state_wipe_sponge_ref(state);
}

int libkeccak_update_ref(struct libkeccak_state *state, const uint8_t *msg, size_t msglen)
{
  size_t len;
  unsigned char *new_mem; // Explicitly declare the type of the 'new' variable

  // Check if the current message pointer plus new message length exceeds the allocated message length
  if (__builtin_expect(state->mptr + msglen > state->mlen, 0))
  {
    // Update the message length to accommodate the new message
    state->mlen += msglen;

    // Allocate new memory
    new_mem = (unsigned char *)malloc(state->mlen * sizeof(char)); // Explicitly cast to unsigned char*
    if (!new_mem)
    {                        // Check if memory allocation failed
      state->mlen -= msglen; // Roll back the message length change
      return -1;             // Return error code
    }

    // Wipe the current message state
    libkeccak_state_wipe_message_ref(state);

    // Free the old message memory
    free(state->M);

    // Set the new memory as the current message memory
    state->M = new_mem;
  }

  // Copy the new message data to the state message buffer
  memcpy(state->M + state->mptr, msg, msglen * sizeof(char));
  state->mptr += msglen; // Update the message pointer

  // Calculate how much of the message buffer can be processed
  len = state->mptr;
  len -= state->mptr % (size_t)(state->r >> 3); // Align to the rate (r)
  state->mptr -= len;                           // Reduce the message pointer by the amount we can process

  // Absorb the message into the Keccak state
  libkeccak_absorption_phase_ref(state, state->M, len);

  // Move the remaining message data to the front of the buffer
  memmove(state->M, state->M + len, state->mptr * sizeof(char));

  return 0;
}

static inline size_t
libkeccak_pad10star1_ref(size_t r, unsigned char *msg, size_t msglen, size_t bits)
{
	size_t nrf = msglen - !!bits;
	size_t len = (nrf << 3) | bits;
	size_t ll = len % r;
	unsigned char b = (unsigned char)(bits ? (msg[nrf] | (1 << bits)) : 1);

	if (r - 8 <= ll && ll <= r - 2) {
		msg[nrf] = (unsigned char)(b ^ 0x80);
		msglen = nrf + 1;
	} else {
		len = ++nrf << 3;
		len = (len - (len % r) + (r - 8)) >> 3;
		msglen = len + 1;

		msg[nrf - 1] = b;
		__builtin_memset(&msg[nrf], 0, (len - nrf) * sizeof(char));
		msg[len] = (unsigned char)0x80;
	}
	return msglen;
}

static void
libkeccak_squeezing_phase_ref(struct libkeccak_state *state, long int rr,
                          long int nn, long int ww, unsigned char *hashsum)
{
	int_fast64_t v;
	long int ni = rr / ww;
	long int olen = state->n;
	long int i, j = 0;
	long int k;
	while (olen > 0) {
		for (i = 0; i < ni && j < nn; i++) {
			v = state->S[LANE_TRANSPOSE_MAP[i]];
			for (k = 0; k++ < ww && j++ < nn; v >>= 8)
				*hashsum++ = (unsigned char)v;
		}
		olen -= state->r;
		if (olen > 0)
			libkeccak_f(state);
	}
	if (state->n & 7)
		hashsum[-1] &= (unsigned char)((1 << (state->n & 7)) - 1);
}

int
libkeccak_digest(struct libkeccak_state *state, const uint8_t *msg_, size_t msglen,
                 size_t bits, const char *suffix, uint8_t *hashsum)
{
	const unsigned char *msg = msg_;
	unsigned char *new_mem;
	long int rr = state->r >> 3;
	size_t suffix_len = suffix ? __builtin_strlen(suffix) : 0;
	size_t ext;
	long int i;

	if (!msg) {
		msglen = 0;
		bits = 0;
	} else {
		msglen += bits >> 3;
		bits &= 7;
	}

	ext = msglen + ((bits + suffix_len + 7) >> 3) + (size_t)rr;
	if (__builtin_expect(state->mptr + ext > state->mlen, 0)) {
		state->mlen += ext;
		new_mem = (unsigned char*)malloc(state->mlen * sizeof(char));
		if (!new_mem) {
			state->mlen -= ext;
			return -1;
		}
		libkeccak_state_wipe_message_ref(state);
		free(state->M);
		state->M = new_mem;
	}

	if (msglen)
		__builtin_memcpy(state->M + state->mptr, msg, msglen * sizeof(char));
	state->mptr += msglen;

	if (bits)
		state->M[state->mptr] = msg[msglen] & (unsigned char)((1 << bits) - 1);
	if (__builtin_expect(!!suffix_len, 1)) {
		if (!bits)
			state->M[state->mptr] = 0;
		while (suffix_len--) {
			state->M[state->mptr] |= (unsigned char)((*suffix++ & 1) << bits++);
			if (bits == 8) {
				bits = 0;
				state->M[++(state->mptr)] = 0;
			}
		}
	}
	if (bits)
		state->mptr++;

	state->mptr = libkeccak_pad10star1_ref((size_t)state->r, state->M, state->mptr, bits);
	libkeccak_absorption_phase_ref(state, state->M, state->mptr);

	if (hashsum) {
		libkeccak_squeezing_phase_ref(state, rr, (state->n + 7) >> 3, state->w >> 3, hashsum);
	} else {
		for (i = (state->n - 1) / state->r; i--;)
			libkeccak_f(state);
	}

	return 0;
}

inline void
libkeccak_state_destroy_ref(volatile struct libkeccak_state *state)
{
	if (state) {
		libkeccak_state_wipe_ref(state);
		free(state->M);
		state->M = NULL;
	}
}

void cShake256_ref(const uint8_t *msg, size_t msg_len, const char* custom, uint8_t *digest, size_t output_length) {
  struct libkeccak_spec spec;
  libkeccak_spec_shake(&spec, 256, output_length);

  struct libkeccak_state state;
  libkeccak_state_initialise_ref(&state, &spec);
  libkeccak_cshake_initialise_ref(&state, NULL, 0, 0, NULL,
                                (uint8_t*)custom, strlen(custom), 0, NULL);
  
  libkeccak_update_ref(&state, msg, msg_len);
  libkeccak_digest(&state, NULL, 0, 0, "00", digest);
  libkeccak_state_destroy_ref(&state);
}

void test_cshake256_comparison()
{
  // Example input data
  uint8_t input[] = {0xDE, 0xAD, 0xBE, 0xEF}; // Input data
  size_t inputLen = sizeof(input);

  // Customization string used in both cases
  const char *customString = "ProofOfWorkHash";

  // Output buffers for both implementations (32 bytes = 256 bits)
  uint8_t customOutput[32] = {0}; // For custom implementation
  uint8_t keccakOutput[32] = {0}; // For libkeccak

  // Output length (32 bytes = 256 bits)
  size_t outputLen = 32;

  // Customization string as byte array (for custom implementation)
  uint8_t S[] = "ProofOfWorkHash"; // Same as the custom string
  size_t sLen = strlen((char *)S); // Length of the custom string

  // Run the custom cshake256 implementation with 32 bytes output
  cShake256_ref(input, inputLen, customString, customOutput, outputLen * 8);

  // Run the libkeccak cshake256 implementation (N is nil, use custom string)
  cshake256_nil_function_name(input, inputLen, customString, keccakOutput, outputLen * 8); // Output length in bits for libkeccak

  // Compare both outputs byte by byte
  if (memcmp(customOutput, keccakOutput, outputLen) == 0)
  {
    printf("Test passed: Outputs match!\n");
  }
  else
  {
    printf("Test failed: Outputs do not match!\n");

    // Print both outputs for comparison
    printf("Custom cSHAKE256 output:\n");
    for (size_t i = 0; i < outputLen; i++)
    {
      printf("%02x", customOutput[i]);
    }
    printf("\n");

    printf("libkeccak cSHAKE256 output:\n");
    for (size_t i = 0; i < outputLen; i++)
    {
      printf("%02x", keccakOutput[i]);
    }
    printf("\n");
  }
}