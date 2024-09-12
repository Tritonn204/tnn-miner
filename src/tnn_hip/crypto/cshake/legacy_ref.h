/* See LICENSE file for copyright and license details. */


typedef struct libkeccak_spec libkeccak_spec_t;

typedef struct libkeccak_generalised_spec libkeccak_generalised_spec_t;

typedef struct libkeccak_state libkeccak_state_t;

typedef struct libkeccak_hmac_state libkeccak_hmac_state_t;

static inline size_t
libkeccak_hmac_unmarshal_skip(const void *data)
{
	return libkeccak_hmac_unmarshal(NULL, data);
}

static inline size_t
libkeccak_state_unmarshal_skip(const void *data)
{
	return libkeccak_state_unmarshal(NULL, data);
}

static inline size_t
libkeccak_hmac_marshal_size(const struct libkeccak_hmac_state *state)
{
	return libkeccak_hmac_marshal(state, NULL);
}

static inline size_t
libkeccak_state_marshal_size(const struct libkeccak_state *state)
{
	return libkeccak_state_marshal(state, NULL);
}
