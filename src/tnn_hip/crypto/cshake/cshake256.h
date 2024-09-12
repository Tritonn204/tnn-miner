#pragma once

void cshake256_single(const char *message, int message_len, const char *custom_str, uint8_t custom_len, unsigned char *output, int output_len);
extern "C" void test_cshake256();