#pragma once

// KawPow (ProgPoW 0.9.4 / Ravencoin) CPU reference test against known vectors
int test_kawpow_hip();

// GPU test: compile kernel, generate DAG, compare GPU vs CPU hashes
int kawpow_gpu_test();

// GPU benchmark: measure kernel throughput
void kawpow_bench(int block_height = -1);

// Dump generated program body (non-GLC + GLC + full kernel) for a given block
void kawpow_dump_program(int block_number);
