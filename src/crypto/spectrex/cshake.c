#include <libkeccak/libkeccak.h>

// Function to perform cSHAKE256 hashing
void cshake256_nil_function_name(const uint8_t *msg, size_t msg_len, const char* custom, uint8_t *digest, size_t output_length) {
    struct libkeccak_spec spec;
    libkeccak_spec_shake(&spec, 256, output_length);

    struct libkeccak_state state;
    libkeccak_state_initialise(&state, &spec);

    libkeccak_cshake_initialise(&state, NULL, 0, 0, NULL,
                                custom, strlen(custom), 0, NULL);

    libkeccak_update(&state, msg, msg_len);
    libkeccak_digest(&state, NULL, 0, 0, libkeccak_cshake_suffix(0, 1), digest);
    libkeccak_state_destroy(&state);
}