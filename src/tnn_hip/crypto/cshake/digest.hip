#include "common.hip.h"

#define rotate(x, n, w, wmod) ((((x) >> ((w) - ((n) % (w)))) | ((x) << ((n) % (w)))) & (wmod))


/**
 * Rotate a 64-bit word
 * 
 * @param   x:int_fast64_t  The value to rotate
 * @param   n:long          Rotation steps, may not be zero
 * @return   :int_fast64_t  The value rotated
 */
#define rotate64(x, n) ((int_fast64_t)(((uint64_t)(x) >> (64L - (n))) | ((uint64_t)(x) << (n))))

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

__device__ static void
hipkeccak_f_round(struct hipkeccak_state *state, int_fast64_t rc)
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
	da = C[4] ^ rotate(C[1], 1, w, wmod);
	dd = C[2] ^ rotate(C[4], 1, w, wmod);
	db = C[0] ^ rotate(C[2], 1, w, wmod);
	de = C[3] ^ rotate(C[0], 1, w, wmod);
	dc = C[1] ^ rotate(C[3], 1, w, wmod);

	/* ρ and π steps, with last two part of θ. */
#define X(bi, ai, dv, r) B[bi] = rotate(A[ai] ^ dv, r, w, wmod)
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

__device__ static void
hipkeccak_f_round64(struct hipkeccak_state *state, int_fast64_t rc)
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
	da = C[4] ^ rotate64(C[1], 1);
	dd = C[2] ^ rotate64(C[4], 1);
	db = C[0] ^ rotate64(C[2], 1);
	de = C[3] ^ rotate64(C[0], 1);
	dc = C[1] ^ rotate64(C[3], 1);

	/* ρ and π steps, with last two part of θ. */
#define X(bi, ai, dv, r) B[bi] = rotate64(A[ai] ^ dv, r)
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

__device__ static inline void
hipkeccak_f(struct hipkeccak_state *state)
{
	long int i = 0;
	long int nr = state->nr;
	long int wmod = state->wmod;
	if (nr == 24) {
		for (; i < nr; i++)
			hipkeccak_f_round64(state, (int_fast64_t)(RC_REF[i]));
	} else {
		for (; i < nr; i++)
			hipkeccak_f_round(state, (int_fast64_t)(RC_REF[i] & (uint_fast64_t)wmod));
	}
}

__device__ static inline int_fast64_t
hipkeccak_to_lane(const unsigned char *message, size_t msglen,
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
 * 64-bit lane version of `hipkeccak_to_lane`
 * 
 * @param   message  The message
 * @param   msglen   The length of the message
 * @param   rr       Bitrate in bytes
 * @param   off      The offset in the message
 * @return           The lane
 */
__device__ static inline int_fast64_t
hipkeccak_to_lane64(const unsigned char *message, size_t msglen, long int rr, size_t off)
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

__device__ static inline size_t
hipkeccak_pad10star1(size_t r, unsigned char *msg, size_t msglen, size_t bits)
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

__device__ static void
hipkeccak_absorption_phase(struct hipkeccak_state *state,
                           const unsigned char *message, size_t len)
{
	long int rr = state->r >> 3;
	long int ww = state->w >> 3;
	long int n = (long)len / rr;
	if (__builtin_expect(ww >= 8, 1)) { /* ww > 8 is impossible, it is just for optimisation possibilities. */
		while (n--) {
#define X(N) state->S[N] ^= hipkeccak_to_lane64(message, len, rr, (size_t)(LANE_TRANSPOSE_MAP[N] * 8));
			LIST_25_REF
#undef X
			hipkeccak_f(state);
			message += (size_t)rr;
			len -= (size_t)rr;
		}
	} else {
		while (n--) {
#define X(N) state->S[N] ^= hipkeccak_to_lane(message, len, rr, ww, (size_t)(LANE_TRANSPOSE_MAP[N] * ww));
			LIST_25_REF
#undef X
			hipkeccak_f(state);
			message += (size_t)rr;
			len -= (size_t)rr;
		}
	}
}

__device__ static void
hipkeccak_squeezing_phase(struct hipkeccak_state *state, long int rr,
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
			hipkeccak_f(state);
	}
	if (state->n & 7)
		hashsum[-1] &= (unsigned char)((1 << (state->n & 7)) - 1);
}

__device__ void
hipkeccak_zerocopy_update(struct hipkeccak_state *state, const uint8_t *msg, size_t msglen)
{
	hipkeccak_absorption_phase(state, msg, msglen);
}

__device__ int hipkeccak_update(struct hipkeccak_state *state, const uint8_t *msg, size_t msglen)
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
    hipkeccak_state_wipe_message(state);

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
  hipkeccak_absorption_phase(state, state->M, len);

  // Move the remaining message data to the front of the buffer
  memcpy(state->M, state->M + len, state->mptr * sizeof(char));

  return 0;
}

__device__ int
hipkeccak_digest(struct hipkeccak_state *state, const uint8_t *msg_, size_t msglen,
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
		hipkeccak_state_wipe_message(state);
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

	state->mptr = hipkeccak_pad10star1((size_t)state->r, state->M, state->mptr, bits);
	hipkeccak_absorption_phase(state, state->M, state->mptr);

	if (hashsum) {
		hipkeccak_squeezing_phase(state, rr, (state->n + 7) >> 3, state->w >> 3, hashsum);
	} else {
		for (i = (state->n - 1) / state->r; i--;)
			hipkeccak_f(state);
	}

	return 0;
}