// yespower_test.cpp
#include <iomanip>
#include <sstream>
#include <chrono>
#include <cstring>

#include "yespower.h"

static void bytes_to_hex(const uint8_t* data, size_t len, char* output) {
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        output[i * 2] = hex[data[i] >> 4];
        output[i * 2 + 1] = hex[data[i] & 0xF];
    }
    output[len * 2] = '\0';
}

YespowerTestResult testYespower(const uint8_t* input, size_t input_len,
                               const yespower_params_t* params) {
    YespowerTestResult result = {0};  // Initialize to zero
    yespower_binary_t ref_result, fmv_result;
    yespower_local_t ref_local, fmv_local;
    
    // Initialize contexts
    yespower_init_local(&ref_local);
    yespower_init_local(&fmv_local);
    
    // Time reference implementation
    auto start = std::chrono::high_resolution_clock::now();
    yespower_ref(&ref_local, input, input_len, params, &ref_result);
    auto ref_end = std::chrono::high_resolution_clock::now();
    
    result.ref_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        ref_end - start).count();

    // Time FMV implementation
    start = std::chrono::high_resolution_clock::now();
    yespower(&fmv_local, input, input_len, params, &fmv_result);
    auto fmv_end = std::chrono::high_resolution_clock::now();
    
    // Record results
    result.matches = (memcmp(&ref_result, &fmv_result, sizeof(yespower_binary_t)) == 0);
    bytes_to_hex(ref_result.uc, sizeof(ref_result.uc), result.ref_hash);
    bytes_to_hex(fmv_result.uc, sizeof(fmv_result.uc), result.fmv_hash);
    
    result.fmv_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        fmv_end - start).count();
    
    // Cleanup
    yespower_free_local(&ref_local);
    yespower_free_local(&fmv_local);
    
    return result;
}

void runYespowerTests(std::vector<YespowerTestResult>& results) {
    // Your original C++ implementation stays the same
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