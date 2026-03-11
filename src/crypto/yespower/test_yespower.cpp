// yespower_test.cpp
#include <iomanip>
#include <sstream>
#include <chrono>
#include <cstring>
#include <cstdio>

#include "yespower.h"
#include "yespower_algo.h"

static void bytes_to_hex(const uint8_t* data, size_t len, char* output) {
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        output[i * 2] = hex[data[i] >> 4];
        output[i * 2 + 1] = hex[data[i] & 0xF];
    }
    output[len * 2] = '\0';
}

static bool hex_to_bytes(const char* hex, uint8_t* out, size_t out_len) {
    for (size_t i = 0; i < out_len; i++) {
        unsigned int byte;
        if (sscanf(hex + i * 2, "%02x", &byte) != 1)
            return false;
        out[i] = (uint8_t)byte;
    }
    return true;
}

YespowerTestResult testYespower(const uint8_t* input, size_t input_len,
                               const yespower_params_t* params) {
    YespowerTestResult result = {0};
    yespower_binary_t ref_result, fmv_result;
    yespower_local_t ref_local, fmv_local;

    yespower_init_local(&ref_local);
    yespower_init_local(&fmv_local);

    auto start = std::chrono::high_resolution_clock::now();
    yespower_ref(&ref_local, input, input_len, params, &ref_result);
    auto ref_end = std::chrono::high_resolution_clock::now();

    result.ref_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        ref_end - start).count();

    start = std::chrono::high_resolution_clock::now();
    yespower(&fmv_local, input, input_len, params, &fmv_result);
    auto fmv_end = std::chrono::high_resolution_clock::now();

    result.matches = (memcmp(&ref_result, &fmv_result, sizeof(yespower_binary_t)) == 0);
    bytes_to_hex(ref_result.uc, sizeof(ref_result.uc), result.ref_hash);
    bytes_to_hex(fmv_result.uc, sizeof(fmv_result.uc), result.fmv_hash);

    result.fmv_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        fmv_end - start).count();

    yespower_free_local(&ref_local);
    yespower_free_local(&fmv_local);

    return result;
}

void runYespowerTests(std::vector<YespowerTestResult>& results) {
    std::vector<std::vector<uint8_t>> inputs = {
        {},
        {'p', 'a', 's', 's', 'w', 'o', 'r', 'd'},
        {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
         0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff}
    };

    std::vector<yespower_params_t> params = {
        {YESPOWER_0_5, 2048, 8, nullptr, 0},
        {YESPOWER_1_0, 2048, 8, nullptr, 0},
        {YESPOWER_1_0, 2048, 8, (const uint8_t*)"Test", 4}
    };

    for (const auto& param : params) {
        for (const auto& input : inputs) {
            results.push_back(testYespower(input.data(), input.size(), &param));
        }
    }
}

// Reference test vectors from upstream yespower (cpuminer-opt/algo/yespower/TESTS-OK)
// Input for all: src[i] = i * 3, 80 bytes
struct KnownVector {
    const char *name;
    yespower_params_t params;
    const char *expected_hex; // 32 bytes = 64 hex chars
};

static const KnownVector known_vectors[] = {
    // v0.5 variants
    {"YescryptR8 (MTBC)",
     {YESPOWER_0_5, 2048, 8, (const uint8_t*)"Client Key", 10},
     "a59fec4c4fdda16e3b1405adda66d525b68e7cadfcfe6ac066c7ad118cd80590"},
    {"Yescrypt (YSC)",
     {YESPOWER_0_5, 2048, 8, nullptr, 0},
     "5ecbd8e8d7c90baed4bbf8916a1225dcc3c65f5c9165bae81cdde3cffad128e8"},
    {"YescryptR16 (GOLD)",
     {YESPOWER_0_5, 4096, 16, (const uint8_t*)"Client Key", 10},
     "927e72d0ded3d80475473f40f1743c67289d453d5242d4f55af4e325e06699c5"},
    {"Yescrypt v0.5 N=4096 r=24 Jagaricoin",
     {YESPOWER_0_5, 4096, 24, (const uint8_t*)"Jagaricoin", 10},
     "0e1366973211e7fea8ad9d81989c84a254d968c9d333dd8ff099324f38611e04"},
    {"YescryptR32 (LuckyPepe/LPEPE)",
     {YESPOWER_0_5, 4096, 32, (const uint8_t*)"WaviBanana", 10},
     "3ae05abb3c5cf6f75415a92554c98d50e38ec9552cfa78373616f480b24e559f"},

    // v1.0 variants
    {"Yespower v1.0 N=2048 r=8 (base)",
     {YESPOWER_1_0, 2048, 8, nullptr, 0},
     "69e0e895b3df7aeeb837d71fe199e9d34f7ec46ecbca7a2c4308e51857ae9b46"},
    {"YespowerR16 (YTN)",
     {YESPOWER_1_0, 4096, 16, nullptr, 0},
     "33fb8f063824a4a020f63dca535f5ca66ab5576468c75d1ccaac7542f76495ac"},
    {"Yespower v1.0 N=2048 r=32 (base for ADVC)",
     {YESPOWER_1_0, 2048, 32, nullptr, 0},
     "d5efb813cd263e9b34540130233cbbc6a921fbff3431e5ec1a1abde2aea6ff4d"},
    {"Yespower v1.0 N=1024 r=32 +pers",
     {YESPOWER_1_0, 1024, 32, (const uint8_t*)"personality test", 16},
     "1f0269acf565c49adc0ef9b8f26ab3808cdc38394a254fddeedcc3aacff6ad9d"},

    // Coin-specific params (same input, verifies param init)
    {"YespowerADVC",
     {YESPOWER_1_0, 2048, 32, (const uint8_t*)"Let the quest begin", 19},
     nullptr},  // no upstream vector, just ref vs opt match
    {"YespowerMGPC",
     {YESPOWER_1_0, 2048, 32, (const uint8_t*)"Magpies are birds of the Corvidae family.", 42},
     nullptr},
    {"YespowerURX",
     {YESPOWER_1_0, 2048, 32, (const uint8_t*)"UraniumX", 8},
     nullptr},
    {"YespowerLTNCG",
     {YESPOWER_1_0, 2048, 32, (const uint8_t*)"LTNCGYES", 8},
     nullptr},
    {"YespowerTIDE",
     {YESPOWER_1_0, 2048, 8, nullptr, 0},
     "69e0e895b3df7aeeb837d71fe199e9d34f7ec46ecbca7a2c4308e51857ae9b46"}, // same as base v1.0 N=2048 r=8
};

static const char* strip_spaces(const char* hex, char* buf, size_t buflen) {
    size_t j = 0;
    for (size_t i = 0; hex[i] && j < buflen - 1; i++) {
        if (hex[i] != ' ')
            buf[j++] = hex[i];
    }
    buf[j] = '\0';
    return buf;
}

int runYespowerKnownVectorTests() {
    // Build standard 80-byte input: src[i] = i * 3
    uint8_t src[80];
    for (int i = 0; i < 80; i++)
        src[i] = (uint8_t)(i * 3);

    yespower_local_t local;
    yespower_init_local(&local);

    int total = 0, passed = 0, failed = 0;
    int num_vectors = sizeof(known_vectors) / sizeof(known_vectors[0]);

    printf("Running %d yespower known-vector tests...\n\n", num_vectors);

    for (int t = 0; t < num_vectors; t++) {
        const KnownVector& tv = known_vectors[t];
        total++;

        yespower_binary_t result;
        if (yespower(&local, src, 80, &tv.params, &result) != 0) {
            printf("  FAIL  %s — yespower() returned error\n", tv.name);
            failed++;
            continue;
        }

        char result_hex[65];
        bytes_to_hex(result.uc, 32, result_hex);

        if (tv.expected_hex) {
            // Compare against known vector (strip spaces from reference)
            char expected_clean[65];
            strip_spaces(tv.expected_hex, expected_clean, sizeof(expected_clean));

            if (strcmp(result_hex, expected_clean) == 0) {
                printf("  PASS  %s\n", tv.name);
                passed++;
            } else {
                printf("  FAIL  %s\n", tv.name);
                printf("        expected: %s\n", expected_clean);
                printf("        got:      %s\n", result_hex);
                failed++;
            }
        } else {
            // No known vector — verify ref vs opt match
            yespower_local_t ref_local;
            yespower_init_local(&ref_local);
            yespower_binary_t ref_result;
            yespower_ref(&ref_local, src, 80, &tv.params, &ref_result);
            yespower_free_local(&ref_local);

            char ref_hex[65];
            bytes_to_hex(ref_result.uc, 32, ref_hex);

            if (strcmp(result_hex, ref_hex) == 0) {
                printf("  PASS  %s (ref vs opt match)\n", tv.name);
                passed++;
            } else {
                printf("  FAIL  %s (ref vs opt MISMATCH)\n", tv.name);
                printf("        ref: %s\n", ref_hex);
                printf("        opt: %s\n", result_hex);
                failed++;
            }
        }
    }

    yespower_free_local(&local);

    printf("\n%d/%d passed", passed, total);
    if (failed > 0)
        printf(", %d FAILED", failed);
    printf("\n");

    return failed;
}
