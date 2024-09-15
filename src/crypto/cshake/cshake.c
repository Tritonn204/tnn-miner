#include <libkeccak/libkeccak.h>

// Function to perform cSHAKE256 hashing
void cshake256_nil_function_name(const uint8_t *msg, size_t msg_len, const char* custom, uint8_t *digest, size_t output_length) {
    struct libkeccak_state state;
    libkeccak_kas_state_initialise(&state, NULL);

    libkeccak_cshake_initialise(&state, NULL, 0, 0, NULL,
                                custom, strlen(custom), 0, NULL);

    libkeccak_fast_update(&state, msg, msg_len);
    libkeccak_fast_digest(&state, NULL, 0, 0, "00", digest);
    libkeccak_state_destroy(&state);
}