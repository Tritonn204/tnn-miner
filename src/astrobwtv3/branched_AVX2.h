#include "astrobwtv3.h"

#if defined(__AVX2__)

namespace astro_branched_zOp {
  // arithmetic operations

  void __attribute__((noinline)) a0(__m256i &in) {
    in = _mm256_add_epi8(in, in);
  }

  void __attribute__((noinline)) p0(__m256i &in) {
    in = _mm256_xor_si256(in, popcnt256_epi8(in));
  }

  void __attribute__((noinline)) r0(__m256i &in) {
    in = _mm256_rolv_epi8(in, in);
  }

  void __attribute__((noinline)) r1(__m256i &in) {
    in = _mm256_rol_epi8(in, 1);
  }

  void __attribute__((noinline)) r2(__m256i &in) {
    in = _mm256_rol_epi8(in, 2);
  }

  void __attribute__((noinline)) r3(__m256i &in) {
    in = _mm256_rol_epi8(in, 3);
  }

  void __attribute__((noinline)) r5(__m256i &in) {
    in = _mm256_rol_epi8(in, 5);
  }

  void __attribute__((noinline)) sl03(__m256i &in) {
    in = _mm256_sllv_epi8(in,_mm256_and_si256(in,vec_3));
  }

  void __attribute__((noinline)) sr03(__m256i &in) {
    in = _mm256_srlv_epi8(in,_mm256_and_si256(in,vec_3));
  }

  void __attribute__((noinline)) m0(__m256i &in) {
    in = _mm256_mul_epi8(in, in);
  }

  void __attribute__((noinline)) xp2(__m256i &in, byte p2) {
    in = _mm256_xor_si256(in, _mm256_set1_epi8(p2));
  }

  void __attribute__((noinline)) addp2(__m256i &in, byte p2) {
    in = _mm256_and_si256(in, _mm256_set1_epi8(p2));
  }

  void __attribute__((noinline)) x0_r2(__m256i &in) {
    in = _mm256_xor_si256(in, _mm256_rol_epi8(in, 2));
  }

  void __attribute__((noinline)) x0_r4(__m256i &in) {
    in = _mm256_xor_si256(in, _mm256_rol_epi8(in, 4));
  }

  void __attribute__((noinline)) subx97(__m256i &in) {
    in = _mm256_sub_epi8(in, _mm256_xor_si256(in, _mm256_set1_epi8(97)));
  }

  // Data manipulation

  void __attribute__((noinline)) notData(__m256i &in) {
    in = _mm256_xor_si256(in, _mm256_set1_epi64x(-1LL));
  }

  void __attribute__((noinline)) revData(__m256i &in) {
    in = _mm256_reverse_epi8(in);
  }
  

  // utility

  void __attribute__((noinline)) blendStep(workerData &worker, __m256i &data, __m256i &old) {
    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2 - worker.pos1));
  }

  void __attribute__((noinline)) storeStep(workerData &worker, __m256i &data, __m256i &old) {
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  // op codes

  void op0(workerData &worker, __m256i &data, __m256i &old) {
    p0(data);
    r5(data);
    m0(data);
    r0(data);
    blendStep(worker, data, old);
    storeStep(worker, data, old);

    if ((worker.pos2 - worker.pos1) % 2 == 1) {
      worker.t1 = worker.chunk[worker.pos1];
      worker.t2 = worker.chunk[worker.pos2];
      worker.chunk[worker.pos1] = reverse8(worker.t2);
      worker.chunk[worker.pos2] = reverse8(worker.t1);
    }
  }

  void op1(workerData &worker, __m256i &data, __m256i &old) {
    __m256i shift = _mm256_and_si256(data, vec_3);
    data = _mm256_sllv_epi8(data, shift);
    r1(data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.prev_chunk[worker.pos2]));
    a0(data);
    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op2(workerData &worker, __m256i &data, __m256i &old) {
    p0(data);
    revData(data);
    __m256i shift = _mm256_and_si256(data, vec_3);
    data = _mm256_sllv_epi8(data, shift);
    p0(data);
    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op3(workerData &worker, __m256i &data, __m256i &old) {
    data = _mm256_rolv_epi8(data, _mm256_add_epi8(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    r1(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op4(workerData &worker, __m256i &data, __m256i &old) {
    notData(data);
    sr03(data);
    r0(data);
    subx97(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op5(workerData &worker, __m256i &data, __m256i &old) {
    p0(data);
    xp2(data, worker.chunk[worker.pos2]);
    sl03(data);
    sr03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op6(workerData &worker, __m256i &data, __m256i &old) {
    sl03(data);
    r3(data);
    notData(data);

    __m256i x = _mm256_xor_si256(data,_mm256_set1_epi8(97));
    data = _mm256_sub_epi8(data, x);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op7(workerData &worker, __m256i &data, __m256i &old) {
    a0(data);
    r0(data);

    p0(data);
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op8(workerData &worker, __m256i &data, __m256i &old) {
    notData(data);
    r2(data);
    sl03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op9(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    x0_r4(data);
    sr03(data);
    x0_r2(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op10(workerData &worker, __m256i &data, __m256i &old) {
    notData(data);
    m0(data);
    r3(data);
    m0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op11(workerData &worker, __m256i &data, __m256i &old) {
    r3(data);
    r3(data);
    data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
    r0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op12(workerData &worker, __m256i &data, __m256i &old) {
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    m0(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op13(workerData &worker, __m256i &data, __m256i &old) {
    r1(data);
    data = _mm256_xor_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
    sr03(data);
    r5(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op14(workerData &worker, __m256i &data, __m256i &old) {
    sr03(data);
    sl03(data);
    m0(data);
    sl03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op15(workerData &worker, __m256i &data, __m256i &old) {
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data,2));
    sl03(data);
    addp2(data, worker.chunk[worker.pos2]);
    subx97(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op16(workerData &worker, __m256i &data, __m256i &old) {
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data,4));
    m0(data);
    r1(data);
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op17(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    m0(data);
    r5(data);
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op18(workerData &worker, __m256i &data, __m256i &old) {
    x0_r4(data);
    r1(data);
    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op19(workerData &worker, __m256i &data, __m256i &old) {
    subx97(data);
    r5(data);
    sl03(data);
    a0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op20(workerData &worker, __m256i &data, __m256i &old) {
    addp2(data, worker.chunk[worker.pos2]);
    xp2(data, worker.chunk[worker.pos2]);
    revData(data);
    x0_r2(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op21(workerData &worker, __m256i &data, __m256i &old) {
    r1(data);
    xp2(data, worker.chunk[worker.pos2]);
    a0(data);
    data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op22(workerData &worker, __m256i &data, __m256i &old) {
    sl03(data);
    revData(data);
    m0(data);
    r1(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op23(workerData &worker, __m256i &data, __m256i &old) {
    data = _mm256_rol_epi8(data, 4);
    data = _mm256_xor_si256(data,popcnt256_epi8(data));
    data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op24(workerData &worker, __m256i &data, __m256i &old) {
    a0(data);
    sr03(data);
    x0_r4(data);
    r5(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op25(workerData &worker, __m256i &data, __m256i &old) {
    
    #pragma GCC unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      if (worker.prev_chunk[i] == 0) {
        worker.chunk[i] = 0x00 - 0x61;
        continue;
      }
      // INSERT_RANDOM_CODE_START
      worker.chunk[i] = worker.prev_chunk[i] ^ (byte)bitTable[worker.prev_chunk[i]];             // ones count bits
      worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
      worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
      worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                        // INSERT_RANDOM_CODE_END
    }
  }

  void op26(workerData &worker, __m256i &data, __m256i &old) {
    m0(data);
    data = _mm256_xor_si256(data,popcnt256_epi8(data));
    a0(data);
    revData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op27(workerData &worker, __m256i &data, __m256i &old) {
    r5(data);
    data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
    x0_r4(data);
    r5(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op28(workerData &worker, __m256i &data, __m256i &old) {
    sl03(data);
    a0(data);
    a0(data);
    r5(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op29(workerData &worker, __m256i &data, __m256i &old) {
    m0(data);
    xp2(data, worker.chunk[worker.pos2]);
    sr03(data);
    a0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op30(workerData &worker, __m256i &data, __m256i &old) {
    addp2(data, worker.chunk[worker.pos2]);
    x0_r4(data);
    r5(data);
    sl03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op31(workerData &worker, __m256i &data, __m256i &old) {
    notData(data);
    x0_r2(data);
    sl03(data);
    m0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op32(workerData &worker, __m256i &data, __m256i &old) {
    x0_r2(data);
    revData(data);
    r3(data);
    x0_r2(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op33(workerData &worker, __m256i &data, __m256i &old) {
    r0(data);
    x0_r4(data);
    revData(data);
    m0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op34(workerData &worker, __m256i &data, __m256i &old) {
    subx97(data);
    sl03(data);
    sl03(data);
    subx97(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op35(workerData &worker, __m256i &data, __m256i &old) {
    a0(data);
    notData(data);
    r1(data);
    xp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op36(workerData &worker, __m256i &data, __m256i &old) {
    p0(data);
    r1(data);
    x0_r2(data);
    r1(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op37(workerData &worker, __m256i &data, __m256i &old) {
    r0(data);
    sr03(data);
    sr03(data);
    m0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op38(workerData &worker, __m256i &data, __m256i &old) {
    
    #pragma GCC unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      if (worker.prev_chunk[i] == 0) {
        worker.chunk[i] = 0;
        continue;
      } 
      // INSERT_RANDOM_CODE_START
      worker.chunk[i] = worker.prev_chunk[i] >> (worker.prev_chunk[i] & 3);    // shift right
      worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
      worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
      worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                        // INSERT_RANDOM_CODE_END
    }
  }

  void op39(workerData &worker, __m256i &data, __m256i &old) {
    x0_r2(data);
    xp2(data, worker.chunk[worker.pos2]);
    sr03(data);
    addp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op40(workerData &worker, __m256i &data, __m256i &old) {
    r0(data);
    xp2(data, worker.chunk[worker.pos2]);
    p0(data);
    xp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op41(workerData &worker, __m256i &data, __m256i &old) {
    r5(data);
    subx97(data);
    r3(data);
    x0_r4(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op42(workerData &worker, __m256i &data, __m256i &old) {
    data = _mm256_rol_epi8(data, 4);
    x0_r2(data);
    r0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op43(workerData &worker, __m256i &data, __m256i &old) {
    addp2(data, worker.chunk[worker.pos2]);
    a0(data);
    addp2(data, worker.chunk[worker.pos2]);
    subx97(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op44(workerData &worker, __m256i &data, __m256i &old) {
    
    #pragma GCC unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      if (worker.prev_chunk[i] == 0) {
        worker.chunk[i] = 0;
        continue;
      }
      // INSERT_RANDOM_CODE_START
      worker.chunk[i] = worker.prev_chunk[i] ^ (byte)bitTable[worker.prev_chunk[i]];             // ones count bits
      worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
      worker.chunk[i] = rl8(worker.chunk[i], 3);                // rotate  bits by 3
      worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                        // INSERT_RANDOM_CODE_END
    }
  }

  void op45(workerData &worker, __m256i &data, __m256i &old) {
    r2(data);
    addp2(data, worker.chunk[worker.pos2]);
    p0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op46(workerData &worker, __m256i &data, __m256i &old) {
    p0(data);
    a0(data);
    r5(data);
    x0_r4(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op47(workerData &worker, __m256i &data, __m256i &old) {
    r5(data);
    addp2(data, worker.chunk[worker.pos2]);
    r5(data);
    sl03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op48(workerData &worker, __m256i &data, __m256i &old) {
    r0(data);
    r5(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op49(workerData &worker, __m256i &data, __m256i &old) {
    p0(data);
    a0(data);
    revData(data);
    x0_r4(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op50(workerData &worker, __m256i &data, __m256i &old) {
    revData(data);
    r3(data);
    a0(data);
    r1(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op51(workerData &worker, __m256i &data, __m256i &old) {
    #pragma GCC unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      if (worker.prev_chunk[i] + worker.chunk[worker.pos2] == 0) {
        worker.chunk[i] = 0;
        continue;
      }
      // INSERT_RANDOM_CODE_START
      worker.chunk[i] = worker.prev_chunk[i] ^ worker.chunk[worker.pos2];     // XOR
      worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
      worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
      worker.chunk[i] = rl8(worker.chunk[i], 5);  // rotate  bits by 5
                                                          // INSERT_RANDOM_CODE_END
    }
  }

  void op52(workerData &worker, __m256i &data, __m256i &old) {
    r0(data);
    sr03(data);
    notData(data);
    p0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op53(workerData &worker, __m256i &data, __m256i &old) {
    
    #pragma GCC unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.chunk[i] = worker.prev_chunk[i]*2;                 // +
      worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
      worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
      worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
    }
  }

  void op54(workerData &worker, __m256i &data, __m256i &old) {
    revData(data);
    xp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op55(workerData &worker, __m256i &data, __m256i &old) {
    
    #pragma GCC unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      if (worker.prev_chunk[i] == 0) {
        worker.chunk[i] = 0;
        continue;
      }
      // INSERT_RANDOM_CODE_START
      worker.chunk[i] = reverse8(worker.prev_chunk[i]);      // reverse bits
      worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
      worker.chunk[i] ^= rl8(worker.chunk[i], 4); // rotate  bits by 4
      worker.chunk[i] = rl8(worker.chunk[i], 1);  // rotate  bits by 1
                                                          // INSERT_RANDOM_CODE_END
    }
  }

  void op56(workerData &worker, __m256i &data, __m256i &old) {
    x0_r2(data);
    m0(data);
    notData(data);
    r1(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op57(workerData &worker, __m256i &data, __m256i &old) {
    r0(data);
    revData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op58(workerData &worker, __m256i &data, __m256i &old) {
    revData(data);
    x0_r2(data);
    addp2(data, worker.chunk[worker.pos2]);
    a0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op59(workerData &worker, __m256i &data, __m256i &old) {
    r1(data);
    m0(data);
    r0(data);
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op60(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    notData(data);
    m0(data);
    r3(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op61(workerData &worker, __m256i &data, __m256i &old) {
    r5(data);
    sl03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op62(workerData &worker, __m256i &data, __m256i &old) {
    addp2(data, worker.chunk[worker.pos2]);
    notData(data);
    x0_r2(data);
    a0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op63(workerData &worker, __m256i &data, __m256i &old) {
    r5(data);
    p0(data);
    subx97(data);
    a0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op64(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    revData(data);
    x0_r4(data);
    m0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op65(workerData &worker, __m256i &data, __m256i &old) {
    x0_r2(data);
    m0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op66(workerData &worker, __m256i &data, __m256i &old) {
    x0_r2(data);
    revData(data);
    x0_r4(data);
    r1(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op67(workerData &worker, __m256i &data, __m256i &old) {
    r1(data);
    p0(data);
    x0_r2(data);
    r5(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op68(workerData &worker, __m256i &data, __m256i &old) {
    addp2(data, worker.chunk[worker.pos2]);
    notData(data);
    x0_r4(data);
    xp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op69(workerData &worker, __m256i &data, __m256i &old) {
    a0(data);
    m0(data);
    revData(data);
    sr03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op70(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    m0(data);
    sr03(data);
    x0_r4(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op71(workerData &worker, __m256i &data, __m256i &old) {
    r5(data);
    notData(data);
    m0(data);
    sl03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op72(workerData &worker, __m256i &data, __m256i &old) {
    revData(data);
    p0(data);
    xp2(data, worker.chunk[worker.pos2]);
    sl03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op73(workerData &worker, __m256i &data, __m256i &old) {
    p0(data);
    revData(data);
    r5(data);
    subx97(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op74(workerData &worker, __m256i &data, __m256i &old) {
    m0(data);
    r3(data);
    revData(data);
    addp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op75(workerData &worker, __m256i &data, __m256i &old) {
    m0(data);
    p0(data);
    addp2(data, worker.chunk[worker.pos2]);
    x0_r4(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op76(workerData &worker, __m256i &data, __m256i &old) {
    r0(data);
    x0_r2(data);
    r5(data);
    sr03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op77(workerData &worker, __m256i &data, __m256i &old) {
    r3(data);
    a0(data);
    sl03(data);
    p0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op78(workerData &worker, __m256i &data, __m256i &old) {
    r0(data);
    revData(data);
    m0(data);
    subx97(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op79(workerData &worker, __m256i &data, __m256i &old) {
    x0_r4(data);
    x0_r2(data);
    a0(data);
    m0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op80(workerData &worker, __m256i &data, __m256i &old) {
    r0(data);
    sl03(data);
    a0(data);
    addp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op81(workerData &worker, __m256i &data, __m256i &old) {
    x0_r4(data);
    sl03(data);
    r0(data);
    p0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op82(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    sr03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op83(workerData &worker, __m256i &data, __m256i &old) {
    sl03(data);
    revData(data);
    r3(data);
    revData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op84(workerData &worker, __m256i &data, __m256i &old) {
    subx97(data);
    r1(data);
    sl03(data);
    a0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op85(workerData &worker, __m256i &data, __m256i &old) {
    sr03(data);
    xp2(data, worker.chunk[worker.pos2]);
    r0(data);
    sl03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op86(workerData &worker, __m256i &data, __m256i &old) {
    x0_r4(data);
    r0(data);
    x0_r4(data);
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op87(workerData &worker, __m256i &data, __m256i &old) {
    a0(data);
    r3(data);
    x0_r4(data);
    a0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op88(workerData &worker, __m256i &data, __m256i &old) {
    x0_r2(data);
    r1(data);
    m0(data);
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op89(workerData &worker, __m256i &data, __m256i &old) {
    a0(data);
    m0(data);
    notData(data);
    x0_r2(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op90(workerData &worker, __m256i &data, __m256i &old) {
    revData(data);
    r3(data);
    r3(data);
    sr03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op91(workerData &worker, __m256i &data, __m256i &old) {
    p0(data);
    addp2(data, worker.chunk[worker.pos2]);
    x0_r4(data);
    revData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op92(workerData &worker, __m256i &data, __m256i &old) {
    p0(data);
    notData(data);
    p0(data);
    addp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op93(workerData &worker, __m256i &data, __m256i &old) {
    x0_r2(data);
    m0(data);
    addp2(data, worker.chunk[worker.pos2]);
    a0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op94(workerData &worker, __m256i &data, __m256i &old) {
    r1(data);
    r0(data);
    addp2(data, worker.chunk[worker.pos2]);
    sl03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op95(workerData &worker, __m256i &data, __m256i &old) {
    r1(data);
    notData(data);
    r2(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op96(workerData &worker, __m256i &data, __m256i &old) {
    #pragma GCC unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[i]];
      if (worker.unchangedBytes[worker.reg_idx[worker.op]].test(worker.prev_chunk[i])) {
        worker.chunk[i] = worker.prev_chunk[i];
      } else {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.prev_chunk[i] ^ rl8(worker.prev_chunk[i], 2);   // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);   // rotate  bits by 2
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = rl8(worker.chunk[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
    }
  }

  void op97(workerData &worker, __m256i &data, __m256i &old) {
    r1(data);
    sl03(data);
    p0(data);
    sr03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op98(workerData &worker, __m256i &data, __m256i &old) {
    x0_r4(data);
    sl03(data);
    sr03(data);
    x0_r4(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op99(workerData &worker, __m256i &data, __m256i &old) {
    x0_r4(data);
    subx97(data);
    revData(data);
    sr03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op100(workerData &worker, __m256i &data, __m256i &old) {
    r0(data);
    sl03(data);
    revData(data);
    p0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op101(workerData &worker, __m256i &data, __m256i &old) {
    sr03(data);
    p0(data);
    sr03(data);
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op102(workerData &worker, __m256i &data, __m256i &old) {
    r3(data);
    subx97(data);
    a0(data);
    r3(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op103(workerData &worker, __m256i &data, __m256i &old) {
    r1(data);
    revData(data);
    xp2(data, worker.chunk[worker.pos2]);
    r0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op104(workerData &worker, __m256i &data, __m256i &old) {
    revData(data);
    p0(data);
    r5(data);
    a0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op105(workerData &worker, __m256i &data, __m256i &old) {
    sl03(data);
    r3(data);
    r0(data);
    x0_r2(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op106(workerData &worker, __m256i &data, __m256i &old) {
    revData(data);
    x0_r4(data);
    r1(data);
    m0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op107(workerData &worker, __m256i &data, __m256i &old) {
    sr03(data);
    x0_r2(data);
    r3(data);
    r3(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op108(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    notData(data);
    addp2(data, worker.chunk[worker.pos2]);
    x0_r2(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op109(workerData &worker, __m256i &data, __m256i &old) {
    m0(data);
    r0(data);
    xp2(data, worker.chunk[worker.pos2]);
    x0_r2(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op110(workerData &worker, __m256i &data, __m256i &old) {
    a0(data);
    x0_r2(data);
    x0_r2(data);
    sr03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op111(workerData &worker, __m256i &data, __m256i &old) {
    m0(data);
    revData(data);
    m0(data);
    sr03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op112(workerData &worker, __m256i &data, __m256i &old) {
    r3(data);
    notData(data);
    r5(data);
    subx97(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op113(workerData &worker, __m256i &data, __m256i &old) {
    r3(data);
    r3(data);
    p0(data);
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op114(workerData &worker, __m256i &data, __m256i &old) {
    r1(data);
    revData(data);
    r0(data);
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op115(workerData &worker, __m256i &data, __m256i &old) {
    r0(data);
    r5(data);
    addp2(data, worker.chunk[worker.pos2]);
    r3(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op116(workerData &worker, __m256i &data, __m256i &old) {
    addp2(data, worker.chunk[worker.pos2]);
    xp2(data, worker.chunk[worker.pos2]);
    p0(data);
    sl03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op117(workerData &worker, __m256i &data, __m256i &old) {
    sl03(data);
    r3(data);
    sl03(data);
    addp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op118(workerData &worker, __m256i &data, __m256i &old) {
    sr03(data);
    a0(data);
    sl03(data);
    r5(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op119(workerData &worker, __m256i &data, __m256i &old) {
    revData(data);
    x0_r2(data);
    notData(data);
    xp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op120(workerData &worker, __m256i &data, __m256i &old) {
    x0_r2(data);
    m0(data);
    xp2(data, worker.chunk[worker.pos2]);
    revData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op121(workerData &worker, __m256i &data, __m256i &old) {
    sr03(data);
    a0(data);
    p0(data);
    m0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op122(workerData &worker, __m256i &data, __m256i &old) {
    x0_r4(data);
    r0(data);
    r5(data);
    x0_r2(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op123(workerData &worker, __m256i &data, __m256i &old) {
    addp2(data, worker.chunk[worker.pos2]);
    notData(data);
    r3(data);
    r3(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op124(workerData &worker, __m256i &data, __m256i &old) {
    x0_r2(data);
    x0_r2(data);
    xp2(data, worker.chunk[worker.pos2]);
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op125(workerData &worker, __m256i &data, __m256i &old) {
    revData(data);
    x0_r2(data);
    a0(data);
    sr03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op126(workerData &worker, __m256i &data, __m256i &old) {
    r1(data);
    revData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op127(workerData &worker, __m256i &data, __m256i &old) {
    sl03(data);
    m0(data);
    addp2(data, worker.chunk[worker.pos2]);
    xp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op128(workerData &worker, __m256i &data, __m256i &old) {
    #pragma GCC unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[i]];
      if (worker.unchangedBytes[worker.reg_idx[worker.op]].test(worker.prev_chunk[i])) {
        worker.chunk[i] = worker.prev_chunk[i];
      } else {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = rl8(worker.prev_chunk[i], worker.prev_chunk[i]); // rotate  bits by random
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] ^= rl8(worker.chunk[i], 2);               // rotate  bits by 2
        worker.chunk[i] = rl8(worker.chunk[i], 5);                // rotate  bits by 5
                                                                          // INSERT_RANDOM_CODE_END
      }
    }
  }

  void op129(workerData &worker, __m256i &data, __m256i &old) {
    notData(data);
    p0(data);
    p0(data);
    sr03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op130(workerData &worker, __m256i &data, __m256i &old) {
    sr03(data);
    r0(data);
    r1(data);
    x0_r4(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op131(workerData &worker, __m256i &data, __m256i &old) {
    subx97(data);
    r1(data);
    p0(data);
    m0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op132(workerData &worker, __m256i &data, __m256i &old) {
    addp2(data, worker.chunk[worker.pos2]);
    revData(data);
    r5(data);
    x0_r2(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op133(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    r5(data);
    x0_r2(data);
    sl03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op134(workerData &worker, __m256i &data, __m256i &old) {
    notData(data);
    x0_r4(data);
    r1(data);
    addp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op135(workerData &worker, __m256i &data, __m256i &old) {
    sr03(data);
    x0_r2(data);
    a0(data);
    revData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op136(workerData &worker, __m256i &data, __m256i &old) {
    sr03(data);
    subx97(data);
    xp2(data, worker.chunk[worker.pos2]);
    r5(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op137(workerData &worker, __m256i &data, __m256i &old) {
    r5(data);
    sr03(data);
    revData(data);
    r0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op138(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    xp2(data, worker.chunk[worker.pos2]);
    a0(data);
    subx97(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op139(workerData &worker, __m256i &data, __m256i &old) {
    x0_r2(data);
    r3(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op140(workerData &worker, __m256i &data, __m256i &old) {
    r1(data);
    x0_r2(data);
    xp2(data, worker.chunk[worker.pos2]);
    r5(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op141(workerData &worker, __m256i &data, __m256i &old) {
    r1(data);
    subx97(data);
    p0(data);
    a0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op142(workerData &worker, __m256i &data, __m256i &old) {
    addp2(data, worker.chunk[worker.pos2]);
    r5(data);
    revData(data);
    x0_r2(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op143(workerData &worker, __m256i &data, __m256i &old) {
    addp2(data, worker.chunk[worker.pos2]);
    r3(data);
    sr03(data);
    sl03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op144(workerData &worker, __m256i &data, __m256i &old) {
    r0(data);
    sl03(data);
    notData(data);
    r0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op145(workerData &worker, __m256i &data, __m256i &old) {
    revData(data);
    x0_r4(data);
    x0_r2(data);
    x0_r4(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op146(workerData &worker, __m256i &data, __m256i &old) {
    addp2(data, worker.chunk[worker.pos2]);
    sl03(data);
    addp2(data, worker.chunk[worker.pos2]);
    p0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op147(workerData &worker, __m256i &data, __m256i &old) {
    notData(data);
    sl03(data);
    x0_r4(data);
    m0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op148(workerData &worker, __m256i &data, __m256i &old) {
    addp2(data, worker.chunk[worker.pos2]);
    r5(data);
    sl03(data);
    subx97(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op149(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    revData(data);
    subx97(data);
    a0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op150(workerData &worker, __m256i &data, __m256i &old) {
    sl03(data);
    sl03(data);
    sl03(data);
    addp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op151(workerData &worker, __m256i &data, __m256i &old) {
    a0(data);
    sl03(data);
    m0(data);
    sl03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op152(workerData &worker, __m256i &data, __m256i &old) {
    sr03(data);
    notData(data);
    sl03(data);
    x0_r2(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op153(workerData &worker, __m256i &data, __m256i &old) {
    data = _mm256_rol_epi8(data, 4);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op154(workerData &worker, __m256i &data, __m256i &old) {
    r5(data);
    notData(data);
    xp2(data, worker.chunk[worker.pos2]);
    p0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op155(workerData &worker, __m256i &data, __m256i &old) {
    subx97(data);
    xp2(data, worker.chunk[worker.pos2]);
    p0(data);
    xp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op156(workerData &worker, __m256i &data, __m256i &old) {
    sr03(data);
    sr03(data);
    data = _mm256_rol_epi8(data, 4);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op157(workerData &worker, __m256i &data, __m256i &old) {
    sr03(data);
    sl03(data);
    r0(data);
    r1(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op158(workerData &worker, __m256i &data, __m256i &old) {
    p0(data);
    r3(data);
    a0(data);
    r1(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op159(workerData &worker, __m256i &data, __m256i &old) {
    subx97(data);
    xp2(data, worker.chunk[worker.pos2]);
    r0(data);
    xp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op160(workerData &worker, __m256i &data, __m256i &old) {
    sr03(data);
    revData(data);
    data = _mm256_rol_epi8(data, 4);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op161(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    xp2(data, worker.chunk[worker.pos2]);
    r5(data);
    r0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op162(workerData &worker, __m256i &data, __m256i &old) {
    m0(data);
    revData(data);
    x0_r2(data);
    subx97(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op163(workerData &worker, __m256i &data, __m256i &old) {
    sl03(data);
    subx97(data);
    x0_r4(data);
    r1(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op164(workerData &worker, __m256i &data, __m256i &old) {
    m0(data);
    p0(data);
    subx97(data);
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op165(workerData &worker, __m256i &data, __m256i &old) {
    x0_r4(data);
    xp2(data, worker.chunk[worker.pos2]);
    sl03(data);
    a0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op166(workerData &worker, __m256i &data, __m256i &old) {
    r3(data);
    a0(data);
    x0_r2(data);
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op167(workerData &worker, __m256i &data, __m256i &old) {
    m0(data);
    sr03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op168(workerData &worker, __m256i &data, __m256i &old) {
    r0(data);
    addp2(data, worker.chunk[worker.pos2]);
    r0(data);
    r1(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op169(workerData &worker, __m256i &data, __m256i &old) {
    r1(data);
    sl03(data);
    x0_r4(data);
    addp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op170(workerData &worker, __m256i &data, __m256i &old) {
    subx97(data);
    revData(data);
    subx97(data);
    m0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op171(workerData &worker, __m256i &data, __m256i &old) {
    r3(data);
    subx97(data);
    p0(data);
    revData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op172(workerData &worker, __m256i &data, __m256i &old) {
    x0_r4(data);
    subx97(data);
    sl03(data);
    r1(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op173(workerData &worker, __m256i &data, __m256i &old) {
    notData(data);
    sl03(data);
    m0(data);
    a0(data);

  blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op174(workerData &worker, __m256i &data, __m256i &old) {
    notData(data);
    r0(data);
    p0(data);
    p0(data);

  blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op175(workerData &worker, __m256i &data, __m256i &old) {
    r3(data);
    subx97(data);
    m0(data);
    r5(data);

  blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op176(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    m0(data);
    xp2(data, worker.chunk[worker.pos2]);
    r5(data);

  blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op177(workerData &worker, __m256i &data, __m256i &old) {
    p0(data);
    x0_r2(data);
    x0_r2(data);
    addp2(data, worker.chunk[worker.pos2]);

  blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op178(workerData &worker, __m256i &data, __m256i &old) {
    addp2(data, worker.chunk[worker.pos2]);
    a0(data);
    notData(data);
    r1(data);

  blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op179(workerData &worker, __m256i &data, __m256i &old) {
    x0_r2(data);
    a0(data);
    sr03(data);
    revData(data);

  blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op180(workerData &worker, __m256i &data, __m256i &old) {
    sr03(data);
    x0_r4(data);
    xp2(data, worker.chunk[worker.pos2]);
    subx97(data);

  blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op181(workerData &worker, __m256i &data, __m256i &old) {
    notData(data);
    sl03(data);
    x0_r2(data);
    r5(data);

  blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op182(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    r3(data);
    r3(data);
    x0_r4(data);

  blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op183(workerData &worker, __m256i &data, __m256i &old) {
    a0(data);
    subx97(data);
    subx97(data);
    m0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op184(workerData &worker, __m256i &data, __m256i &old) {
    sl03(data);
    m0(data);
    r5(data);
    xp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op185(workerData &worker, __m256i &data, __m256i &old) {
    notData(data);
    x0_r4(data);
    r5(data);
    sr03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op186(workerData &worker, __m256i &data, __m256i &old) {
    x0_r2(data);
    x0_r4(data);
    subx97(data);
    sr03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op187(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    notData(data);
    a0(data);
    r3(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op188(workerData &worker, __m256i &data, __m256i &old) {
  #pragma GCC unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.chunk[i] ^= rl8(worker.prev_chunk[i], 4);   // rotate  bits by 4
      worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
      worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
      worker.chunk[i] ^= rl8(worker.chunk[i], 4);   // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
    }
  }

  void op189(workerData &worker, __m256i &data, __m256i &old) {
    r5(data);
    x0_r4(data);
    xp2(data, worker.chunk[worker.pos2]);
    subx97(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op190(workerData &worker, __m256i &data, __m256i &old) {
    r5(data);
    sr03(data);
    addp2(data, worker.chunk[worker.pos2]);
    x0_r2(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op191(workerData &worker, __m256i &data, __m256i &old) {
    a0(data);
    r3(data);
    r0(data);
    sr03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op192(workerData &worker, __m256i &data, __m256i &old) {
    a0(data);
    sl03(data);
    a0(data);
    m0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op193(workerData &worker, __m256i &data, __m256i &old) {
    addp2(data, worker.chunk[worker.pos2]);
    sl03(data);
    r0(data);
    r1(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op194(workerData &worker, __m256i &data, __m256i &old) {
    addp2(data, worker.chunk[worker.pos2]);
    r0(data);
    sl03(data);
    addp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op195(workerData &worker, __m256i &data, __m256i &old) {
    p0(data);
    x0_r2(data);
    xp2(data, worker.chunk[worker.pos2]);
    x0_r4(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op196(workerData &worker, __m256i &data, __m256i &old) {
    r3(data);
    revData(data);
    sl03(data);
    r1(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op197(workerData &worker, __m256i &data, __m256i &old) {
    x0_r4(data);
    r0(data);
    m0(data);
    m0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op198(workerData &worker, __m256i &data, __m256i &old) {
    sr03(data);
    sr03(data);
    revData(data);
    r1(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op199(workerData &worker, __m256i &data, __m256i &old) {
    notData(data);
    a0(data);
    m0(data);
    xp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op200(workerData &worker, __m256i &data, __m256i &old) {
    sr03(data);
    p0(data);
    revData(data);
    revData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op201(workerData &worker, __m256i &data, __m256i &old) {
    r3(data);
    x0_r2(data);
    x0_r4(data);
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op202(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    notData(data);
    r0(data);
    r5(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op203(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    addp2(data, worker.chunk[worker.pos2]);
    r1(data);
    r0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op204(workerData &worker, __m256i &data, __m256i &old) {
    r5(data);
    x0_r2(data);
    r0(data);
    xp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op205(workerData &worker, __m256i &data, __m256i &old) {
    p0(data);
    x0_r4(data);
    sl03(data);
    a0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op206(workerData &worker, __m256i &data, __m256i &old) {
    x0_r4(data);
    revData(data);
    revData(data);
    p0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op207(workerData &worker, __m256i &data, __m256i &old) {
    p0(data);
    p0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op208(workerData &worker, __m256i &data, __m256i &old) {
  #pragma GCC unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.chunk[i] = worker.prev_chunk[i]*2;                          // +
      worker.chunk[i] += worker.chunk[i];                          // +
      worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
      worker.chunk[i] = rl8(worker.chunk[i], 3);             // rotate  bits by 3
                                                                        // INSERT_RANDOM_CODE_END
    }
  }

  void op209(workerData &worker, __m256i &data, __m256i &old) {
    r5(data);
    revData(data);
    p0(data);
    subx97(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op210(workerData &worker, __m256i &data, __m256i &old) {
    x0_r2(data);
    r0(data);
    r5(data);
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op211(workerData &worker, __m256i &data, __m256i &old) {
    x0_r4(data);
    a0(data);
    subx97(data);
    r0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op212(workerData &worker, __m256i &data, __m256i &old) {
    r0(data);
    x0_r2(data);
    // xp2(data, worker.chunk[worker.pos2]);
    // xp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op213(workerData &worker, __m256i &data, __m256i &old) {
    a0(data);
    sl03(data);
    r3(data);
    subx97(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op214(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    subx97(data);
    sr03(data);
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op215(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    addp2(data, worker.chunk[worker.pos2]);
    sl03(data);
    m0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op216(workerData &worker, __m256i &data, __m256i &old) {
    r0(data);
    notData(data);
    subx97(data);
    addp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op217(workerData &worker, __m256i &data, __m256i &old) {
    r5(data);
    a0(data);
    r1(data);
    x0_r4(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op218(workerData &worker, __m256i &data, __m256i &old) {
    revData(data);
    notData(data);
    m0(data);
    subx97(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op219(workerData &worker, __m256i &data, __m256i &old) {
    x0_r4(data);
    r3(data);
    addp2(data, worker.chunk[worker.pos2]);
    revData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op220(workerData &worker, __m256i &data, __m256i &old) {
    r1(data);
    sl03(data);
    revData(data);
    sl03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op221(workerData &worker, __m256i &data, __m256i &old) {
    r5(data);
    xp2(data, worker.chunk[worker.pos2]);
    notData(data);
    revData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op222(workerData &worker, __m256i &data, __m256i &old) {
    sr03(data);
    sl03(data);
    xp2(data, worker.chunk[worker.pos2]);
    m0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op223(workerData &worker, __m256i &data, __m256i &old) {
    r3(data);
    xp2(data, worker.chunk[worker.pos2]);
    r0(data);
    subx97(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op224(workerData &worker, __m256i &data, __m256i &old) {
    x0_r2(data);
    data = _mm256_rol_epi8(data, 4);
    sl03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op225(workerData &worker, __m256i &data, __m256i &old) {
    notData(data);
    sr03(data);
    revData(data);
    r3(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op226(workerData &worker, __m256i &data, __m256i &old) {
    revData(data);
    subx97(data);
    m0(data);
    xp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op227(workerData &worker, __m256i &data, __m256i &old) {
    notData(data);
    sl03(data);
    subx97(data);
    addp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op228(workerData &worker, __m256i &data, __m256i &old) {
    a0(data);
    sr03(data);
    a0(data);
    p0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op229(workerData &worker, __m256i &data, __m256i &old) {
    r3(data);
    r0(data);
    x0_r2(data);
    p0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op230(workerData &worker, __m256i &data, __m256i &old) {
    m0(data);
    addp2(data, worker.chunk[worker.pos2]);
    r0(data);
    r0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op231(workerData &worker, __m256i &data, __m256i &old) {
    r3(data);
    sr03(data);
    xp2(data, worker.chunk[worker.pos2]);
    revData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op232(workerData &worker, __m256i &data, __m256i &old) {
    m0(data);
    m0(data);
    x0_r4(data);
    r5(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op233(workerData &worker, __m256i &data, __m256i &old) {
    r1(data);
    p0(data);
    r3(data);
    p0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op234(workerData &worker, __m256i &data, __m256i &old) {
    addp2(data, worker.chunk[worker.pos2]);
    m0(data);
    sr03(data);
    xp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op235(workerData &worker, __m256i &data, __m256i &old) {
    x0_r2(data);
    m0(data);
    r3(data);
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op236(workerData &worker, __m256i &data, __m256i &old) {
    xp2(data, worker.chunk[worker.pos2]);
    a0(data);
    addp2(data, worker.chunk[worker.pos2]);
    subx97(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op237(workerData &worker, __m256i &data, __m256i &old) {
    r5(data);
    sl03(data);
    x0_r2(data);
    r3(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op238(workerData &worker, __m256i &data, __m256i &old) {
    a0(data);
    a0(data);
    r3(data);
    subx97(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op239(workerData &worker, __m256i &data, __m256i &old) {
    r3(data);
    r3(data);
    m0(data);
    addp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op240(workerData &worker, __m256i &data, __m256i &old) {
    notData(data);
    a0(data);
    addp2(data, worker.chunk[worker.pos2]);
    sl03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op241(workerData &worker, __m256i &data, __m256i &old) {
    x0_r4(data);
    p0(data);
    xp2(data, worker.chunk[worker.pos2]);
    r1(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op242(workerData &worker, __m256i &data, __m256i &old) {
    a0(data);
    a0(data);
    subx97(data);
    xp2(data, worker.chunk[worker.pos2]);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op243(workerData &worker, __m256i &data, __m256i &old) {
    r5(data);
    x0_r2(data);
    p0(data);
    r1(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op244(workerData &worker, __m256i &data, __m256i &old) {
    notData(data);
    x0_r2(data);
    revData(data);
    r5(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op245(workerData &worker, __m256i &data, __m256i &old) {
    subx97(data);
    r5(data);
    x0_r2(data);
    sr03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op246(workerData &worker, __m256i &data, __m256i &old) {
    a0(data);
    r1(data);
    sr03(data);
    a0(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op247(workerData &worker, __m256i &data, __m256i &old) {
    r5(data);
    x0_r2(data);
    r5(data);
    notData(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op248(workerData &worker, __m256i &data, __m256i &old) {
    notData(data);
    subx97(data);
    p0(data);
    r5(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op249(workerData &worker, __m256i &data, __m256i &old) {
  #pragma GCC unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.chunk[i] = reverse8(worker.prev_chunk[i]);                    // reverse bits
      worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
      worker.chunk[i] ^= rl8(worker.chunk[i], 4);               // rotate  bits by 4
      worker.chunk[i] = rl8(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
                                                                            // INSERT_RANDOM_CODE_END
    }
  }

  void op250(workerData &worker, __m256i &data, __m256i &old) {
    addp2(data, worker.chunk[worker.pos2]);
    r0(data);
    p0(data);
    x0_r4(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op251(workerData &worker, __m256i &data, __m256i &old) {
    a0(data);
    p0(data);
    revData(data);
    x0_r2(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op252(workerData &worker, __m256i &data, __m256i &old) {
    revData(data);
    x0_r4(data);
    x0_r2(data);
    sl03(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op253(workerData &worker, __m256i &data, __m256i &old) {
    storeStep(worker, data, old);
  #pragma GCC unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
      worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
      worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
      worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
      // INSERT_RANDOM_CODE_END

      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = XXHash64::hash(worker.chunk, worker.pos2,0);
    }
  }

  void op254(workerData &worker, __m256i &data, __m256i &old) {
    RC4_set_key(&worker.key, 256, worker.prev_chunk);

    p0(data);
    r3(data);
    x0_r2(data);
    r3(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void op255(workerData &worker, __m256i &data, __m256i &old) {
    RC4_set_key(&worker.key, 256, worker.prev_chunk);

    p0(data);
    r3(data);
    x0_r2(data);
    r3(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void r_op0(workerData &worker, __m256i &data, __m256i &old) {
    byte newVal = worker.simpleLookup[worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2 - worker.pos1));
    storeStep(worker, data, old);

    if ((worker.pos2 - worker.pos1) % 2 == 1) {
      worker.t1 = worker.chunk[worker.pos1];
      worker.t2 = worker.chunk[worker.pos2];
      worker.chunk[worker.pos1] = reverse8(worker.t2);
      worker.chunk[worker.pos2] = reverse8(worker.t1);
      worker.isSame = false;
    }
  }

  void r_op1(workerData &worker, __m256i &data, __m256i &old) {
    byte newVal = worker.lookup3D[worker.branched_idx[worker.op] * 256 * 256 +
                                  worker.prev_chunk[worker.pos2] * 256 +
                                  worker.prev_chunk[worker.pos1]];

    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2 - worker.pos1));
    storeStep(worker, data, old);

    return;
  }

  void r_op2(workerData &worker, __m256i &data, __m256i &old) {
    byte newVal = worker.simpleLookup[worker.reg_idx[worker.op] * 256 + worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2 - worker.pos1));
    storeStep(worker, data, old);
  }

  void r_op253(workerData &worker, __m256i &data, __m256i &old) {
    storeStep(worker, data, old);
    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];
      if (worker.prev_chunk[worker.pos1] == newVal) {
        worker.prev_lhash = (worker.lhash * (worker.pos2-worker.pos1))+ worker.prev_lhash;
        worker.lhash = XXHash64::hash(worker.chunk, worker.pos2,0);
        return;
      }
    }
  #pragma GCC unroll 32
    for (int i = worker.pos1; i < worker.pos2; i++)
    {
      // INSERT_RANDOM_CODE_START
      worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
      worker.chunk[i] ^= rl8(worker.chunk[i], 2); // rotate  bits by 2
      worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
      worker.chunk[i] = rl8(worker.chunk[i], 3);  // rotate  bits by 3
      // INSERT_RANDOM_CODE_END

      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = XXHash64::hash(worker.chunk, worker.pos2,0);
    }
  }

  void r_op254(workerData &worker, __m256i &data, __m256i &old) {
    RC4_set_key(&worker.key, 256, worker.prev_chunk);

    p0(data);
    r3(data);
    x0_r2(data);
    r3(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  void r_op255(workerData &worker, __m256i &data, __m256i &old) {
    RC4_set_key(&worker.key, 256, worker.prev_chunk);

    p0(data);
    r3(data);
    x0_r2(data);
    r3(data);

    blendStep(worker, data, old);
    storeStep(worker, data, old);
  }

  typedef void (*OpFunc)(workerData &, __m256i &, __m256i &);

  alignas(32) OpFunc branchCompute[512] = {
    // standard versions
    op0, op1, op2, op3, op4, op5, op6, op7, op8, op9, op10,
    op11, op12, op13, op14, op15, op16, op17, op18, op19, op20,
    op21, op22, op23, op24, op25, op26, op27, op28, op29, op30,
    op31, op32, op33, op34, op35, op36, op37, op38, op39, op40,
    op41, op42, op43, op44, op45, op46, op47, op48, op49, op50,
    op51, op52, op53, op54, op55, op56, op57, op58, op59, op60,
    op61, op62, op63, op64, op65, op66, op67, op68, op69, op70,
    op71, op72, op73, op74, op75, op76, op77, op78, op79, op80,
    op81, op82, op83, op84, op85, op86, op87, op88, op89, op90,
    op91, op92, op93, op94, op95, op96, op97, op98, op99, op100,
    op101, op102, op103, op104, op105, op106, op107, op108, op109, op110,
    op111, op112, op113, op114, op115, op116, op117, op118, op119, op120,
    op121, op122, op123, op124, op125, op126, op127, op128, op129, op130,
    op131, op132, op133, op134, op135, op136, op137, op138, op139, op140,
    op141, op142, op143, op144, op145, op146, op147, op148, op149, op150,
    op151, op152, op153, op154, op155, op156, op157, op158, op159, op160,
    op161, op162, op163, op164, op165, op166, op167, op168, op169, op170,
    op171, op172, op173, op174, op175, op176, op177, op178, op179, op180,
    op181, op182, op183, op184, op185, op186, op187, op188, op189, op190,
    op191, op192, op193, op194, op195, op196, op197, op198, op199, op200,
    op201, op202, op203, op204, op205, op206, op207, op208, op209, op210,
    op211, op212, op213, op214, op215, op216, op217, op218, op219, op220,
    op221, op222, op223, op224, op225, op226, op227, op228, op229, op230,
    op231, op232, op233, op234, op235, op236, op237, op238, op239, op240,
    op241, op242, op243, op244, op245, op246, op247, op248, op249, op250,
    op251, op252, op253, op254, op255,
    // Repeated char versions
    r_op0, r_op1, r_op2, r_op1, r_op2, r_op1, r_op2, r_op2, r_op2, r_op1, r_op2,
    r_op1, r_op2, r_op1, r_op2, r_op1, r_op2, r_op1, r_op2, r_op2, r_op1,
    r_op1, r_op2, r_op1, r_op2, r_op2, r_op2, r_op1, r_op2, r_op1, r_op1,
    r_op2, r_op2, r_op2, r_op2, r_op1, r_op2, r_op2, r_op2, r_op1, r_op1,
    r_op2, r_op2, r_op1, r_op2, r_op1, r_op2, r_op1, r_op2, r_op2, r_op2,
    r_op1, r_op2, r_op2, r_op1, r_op2, r_op2, r_op2, r_op1, r_op2, r_op1,
    r_op2, r_op1, r_op2, r_op1, r_op2, r_op2, r_op2, r_op1, r_op2, r_op1,
    r_op2, r_op1, r_op2, r_op1, r_op1, r_op2, r_op2, r_op2, r_op2, r_op1,
    r_op2, r_op1, r_op2, r_op2, r_op1, r_op2, r_op2, r_op2, r_op2, r_op2,
    r_op1, r_op1, r_op1, r_op1, r_op2, r_op2, r_op2, r_op2, r_op2, r_op2,
    r_op2, r_op2, r_op1, r_op2, r_op2, r_op2, r_op2, r_op1, r_op1, r_op2,
    r_op2, r_op2, r_op2, r_op2, r_op1, r_op1, r_op1, r_op2, r_op1, r_op1,
    r_op2, r_op2, r_op1, r_op1, r_op2, r_op2, r_op1, r_op2, r_op2, r_op2,
    r_op2, r_op1, r_op1, r_op1, r_op2, r_op1, r_op2, r_op1, r_op2, r_op1,
    r_op2, r_op1, r_op1, r_op2, r_op2, r_op1, r_op2, r_op1, r_op1, r_op1,
    r_op2, r_op2, r_op2, r_op1, r_op1, r_op2, r_op2, r_op2, r_op1, r_op2,
    r_op1, r_op2, r_op2, r_op2, r_op1, r_op2, r_op2, r_op1, r_op1, r_op2,
    r_op2, r_op2, r_op2, r_op2, r_op2, r_op1, r_op1, r_op1, r_op2, r_op1,
    r_op2, r_op1, r_op2, r_op1, r_op2, r_op2, r_op1, r_op2, r_op1, r_op1,
    r_op2, r_op2, r_op1, r_op1, r_op1, r_op2, r_op2, r_op2, r_op1, r_op2,
    r_op2, r_op1, r_op1, r_op1, r_op2, r_op2, r_op2, r_op2, r_op2, r_op2,
    r_op2, r_op1, r_op2, r_op1, r_op1, r_op1, r_op2, r_op2, r_op1, r_op2,
    r_op1, r_op1, r_op1, r_op2, r_op2, r_op1, r_op1, r_op2, r_op2, r_op1,
    r_op1, r_op2, r_op2, r_op1, r_op2, r_op1, r_op2, r_op2, r_op1, r_op1,
    r_op1, r_op1, r_op2, r_op2, r_op2, r_op2, r_op2, r_op2, r_op2, r_op1,
    r_op2, r_op2, r_op253, r_op254, r_op255,
  };

      // switch (worker.op)
    // {
    // case 0:
    //   astro_branched_zOp::branchCompute[0](worker);
    //   break;
    // case 1:
    //   astro_branched_zOp::branchCompute[1](worker);
    //   break;
    // case 2:
    //   astro_branched_zOp::branchCompute[2](worker);
    //   break;
    // case 3:
    //   astro_branched_zOp::branchCompute[3](worker);
    //   break;
    // case 4:
    //   astro_branched_zOp::branchCompute[4](worker);
    //   break;
    // case 5:
    //   astro_branched_zOp::branchCompute[5](worker);
    //   break;
    // case 6:
    //   astro_branched_zOp::branchCompute[6](worker);
    //   break;
    // case 7:
    //   astro_branched_zOp::branchCompute[7](worker);
    //   break;
    // case 8:
    //   astro_branched_zOp::branchCompute[8](worker);
    //   break;
    // case 9:
    //   astro_branched_zOp::branchCompute[9](worker);
    //   break;
    // case 10:
    //   astro_branched_zOp::branchCompute[10](worker);
    //   break;
    // case 11:
    //   astro_branched_zOp::branchCompute[11](worker);
    //   break;
    // case 12:
    //   astro_branched_zOp::branchCompute[12](worker);
    //   break;
    // case 13:
    //   astro_branched_zOp::branchCompute[13](worker);
    //   break;
    // case 14:
    //   astro_branched_zOp::branchCompute[14](worker);
    //   break;
    // case 15:
    //   astro_branched_zOp::branchCompute[15](worker);
    //   break;
    // case 16:
    //   astro_branched_zOp::branchCompute[16](worker);
    //   break;
    // case 17:
    //   astro_branched_zOp::branchCompute[17](worker);
    //   break;
    // case 18:
    //   astro_branched_zOp::branchCompute[18](worker);
    //   break;
    // case 19:
    //   astro_branched_zOp::branchCompute[19](worker);
    //   break;
    // case 20:
    //   astro_branched_zOp::branchCompute[20](worker);
    //   break;
    // case 21:
    //   astro_branched_zOp::branchCompute[21](worker);
    //   break;
    // case 22:
    //   astro_branched_zOp::branchCompute[22](worker);
    //   break;
    // case 23:
    //   astro_branched_zOp::branchCompute[23](worker);
    //   break;
    // case 24:
    //   astro_branched_zOp::branchCompute[24](worker);
    //   break;
    // case 25:
    //   astro_branched_zOp::branchCompute[25](worker);
    //   break;
    // case 26:
    //   astro_branched_zOp::branchCompute[26](worker);
    //   break;
    // case 27:
    //   astro_branched_zOp::branchCompute[27](worker);
    //   break;
    // case 28:
    //   astro_branched_zOp::branchCompute[28](worker);
    //   break;
    // case 29:
    //   astro_branched_zOp::branchCompute[29](worker);
    //   break;
    // case 30:
    //   astro_branched_zOp::branchCompute[30](worker);
    //   break;
    // case 31:
    //   astro_branched_zOp::branchCompute[31](worker);
    //   break;
    // case 32:
    //   astro_branched_zOp::branchCompute[32](worker);
    //   break;
    // case 33:
    //   astro_branched_zOp::branchCompute[33](worker);
    //   break;
    // case 34:
    //   astro_branched_zOp::branchCompute[34](worker);
    //   break;
    // case 35:
    //   astro_branched_zOp::branchCompute[35](worker);
    //   break;
    // case 36:
    //   astro_branched_zOp::branchCompute[36](worker);
    //   break;
    // case 37:
    //   astro_branched_zOp::branchCompute[37](worker);
    //   break;
    // case 38:
    //   astro_branched_zOp::branchCompute[38](worker);
    //   break;
    // case 39:
    //   astro_branched_zOp::branchCompute[39](worker);
    //   break;
    // case 40:
    //   astro_branched_zOp::branchCompute[40](worker);
    //   break;
    // case 41:
    //   astro_branched_zOp::branchCompute[41](worker);
    //   break;
    // case 42:
    //   astro_branched_zOp::branchCompute[42](worker);
    //   break;
    // case 43:
    //   astro_branched_zOp::branchCompute[43](worker);
    //   break;
    // case 44:
    //   astro_branched_zOp::branchCompute[44](worker);
    //   break;
    // case 45:
    //   astro_branched_zOp::branchCompute[45](worker);
    //   break;
    // case 46:
    //   astro_branched_zOp::branchCompute[46](worker);
    //   break;
    // case 47:
    //   astro_branched_zOp::branchCompute[47](worker);
    //   break;
    // case 48:
    //   astro_branched_zOp::branchCompute[48](worker);
    //   break;
    // case 49:
    //   astro_branched_zOp::branchCompute[49](worker);
    //   break;
    // case 50:
    //   astro_branched_zOp::branchCompute[50](worker);
    //   break;
    // case 51:
    //   astro_branched_zOp::branchCompute[51](worker);
    //   break;
    // case 52:
    //   astro_branched_zOp::branchCompute[52](worker);
    //   break;
    // case 53:
    //   astro_branched_zOp::branchCompute[53](worker);
    //   break;
    // case 54:
    //   astro_branched_zOp::branchCompute[54](worker);
    //   break;
    // case 55:
    //   astro_branched_zOp::branchCompute[55](worker);
    //   break;
    // case 56:
    //   astro_branched_zOp::branchCompute[56](worker);
    //   break;
    // case 57:
    //   astro_branched_zOp::branchCompute[57](worker);
    //   break;
    // case 58:
    //   astro_branched_zOp::branchCompute[58](worker);
    //   break;
    // case 59:
    //   astro_branched_zOp::branchCompute[59](worker);
    //   break;
    // case 60:
    //   astro_branched_zOp::branchCompute[60](worker);
    //   break;
    // case 61:
    //   astro_branched_zOp::branchCompute[61](worker);
    //   break;
    // case 62:
    //   astro_branched_zOp::branchCompute[62](worker);
    //   break;
    // case 63:
    //   astro_branched_zOp::branchCompute[63](worker);
    //   break;
    // case 64:
    //   astro_branched_zOp::branchCompute[64](worker);
    //   break;
    // case 65:
    //   astro_branched_zOp::branchCompute[65](worker);
    //   break;
    // case 66:
    //   astro_branched_zOp::branchCompute[66](worker);
    //   break;
    // case 67:
    //   astro_branched_zOp::branchCompute[67](worker);
    //   break;
    // case 68:
    //   astro_branched_zOp::branchCompute[68](worker);
    //   break;
    // case 69:
    //   astro_branched_zOp::branchCompute[69](worker);
    //   break;
    // case 70:
    //   astro_branched_zOp::branchCompute[70](worker);
    //   break;
    // case 71:
    //   astro_branched_zOp::branchCompute[71](worker);
    //   break;
    // case 72:
    //   astro_branched_zOp::branchCompute[72](worker);
    //   break;
    // case 73:
    //   astro_branched_zOp::branchCompute[73](worker);
    //   break;
    // case 74:
    //   astro_branched_zOp::branchCompute[74](worker);
    //   break;
    // case 75:
    //   astro_branched_zOp::branchCompute[75](worker);
    //   break;
    // case 76:
    //   astro_branched_zOp::branchCompute[76](worker);
    //   break;
    // case 77:
    //   astro_branched_zOp::branchCompute[77](worker);
    //   break;
    // case 78:
    //   astro_branched_zOp::branchCompute[78](worker);
    //   break;
    // case 79:
    //   astro_branched_zOp::branchCompute[79](worker);
    //   break;
    // case 80:
    //   astro_branched_zOp::branchCompute[80](worker);
    //   break;
    // case 81:
    //   astro_branched_zOp::branchCompute[81](worker);
    //   break;
    // case 82:
    //   astro_branched_zOp::branchCompute[82](worker);
    //   break;
    // case 83:
    //   astro_branched_zOp::branchCompute[83](worker);
    //   break;
    // case 84:
    //   astro_branched_zOp::branchCompute[84](worker);
    //   break;
    // case 85:
    //   astro_branched_zOp::branchCompute[85](worker);
    //   break;
    // case 86:
    //   astro_branched_zOp::branchCompute[86](worker);
    //   break;
    // case 87:
    //   astro_branched_zOp::branchCompute[87](worker);
    //   break;
    // case 88:
    //   astro_branched_zOp::branchCompute[88](worker);
    //   break;
    // case 89:
    //   astro_branched_zOp::branchCompute[89](worker);
    //   break;
    // case 90:
    //   astro_branched_zOp::branchCompute[90](worker);
    //   break;
    // case 91:
    //   astro_branched_zOp::branchCompute[91](worker);
    //   break;
    // case 92:
    //   astro_branched_zOp::branchCompute[92](worker);
    //   break;
    // case 93:
    //   astro_branched_zOp::branchCompute[93](worker);
    //   break;
    // case 94:
    //   astro_branched_zOp::branchCompute[94](worker);
    //   break;
    // case 95:
    //   astro_branched_zOp::branchCompute[95](worker);
    //   break;
    // case 96:
    //   astro_branched_zOp::branchCompute[96](worker);
    //   break;
    // case 97:
    //   astro_branched_zOp::branchCompute[97](worker);
    //   break;
    // case 98:
    //   astro_branched_zOp::branchCompute[98](worker);
    //   break;
    // case 99:
    //   astro_branched_zOp::branchCompute[99](worker);
    //   break;
    // case 100:
    //   astro_branched_zOp::branchCompute[100](worker);
    //   break;
    // case 101:
    //   astro_branched_zOp::branchCompute[101](worker);
    //   break;
    // case 102:
    //   astro_branched_zOp::branchCompute[102](worker);
    //   break;
    // case 103:
    //   astro_branched_zOp::branchCompute[103](worker);
    //   break;
    // case 104:
    //   astro_branched_zOp::branchCompute[104](worker);
    //   break;
    // case 105:
    //   astro_branched_zOp::branchCompute[105](worker);
    //   break;
    // case 106:
    //   astro_branched_zOp::branchCompute[106](worker);
    //   break;
    // case 107:
    //   astro_branched_zOp::branchCompute[107](worker);
    //   break;
    // case 108:
    //   astro_branched_zOp::branchCompute[108](worker);
    //   break;
    // case 109:
    //   astro_branched_zOp::branchCompute[109](worker);
    //   break;
    // case 110:
    //   astro_branched_zOp::branchCompute[110](worker);
    //   break;
    // case 111:
    //   astro_branched_zOp::branchCompute[111](worker);
    //   break;
    // case 112:
    //   astro_branched_zOp::branchCompute[112](worker);
    //   break;
    // case 113:
    //   astro_branched_zOp::branchCompute[113](worker);
    //   break;
    // case 114:
    //   astro_branched_zOp::branchCompute[114](worker);
    //   break;
    // case 115:
    //   astro_branched_zOp::branchCompute[115](worker);
    //   break;
    // case 116:
    //   astro_branched_zOp::branchCompute[116](worker);
    //   break;
    // case 117:
    //   astro_branched_zOp::branchCompute[117](worker);
    //   break;
    // case 118:
    //   astro_branched_zOp::branchCompute[118](worker);
    //   break;
    // case 119:
    //   astro_branched_zOp::branchCompute[119](worker);
    //   break;
    // case 120:
    //   astro_branched_zOp::branchCompute[120](worker);
    //   break;
    // case 121:
    //   astro_branched_zOp::branchCompute[121](worker);
    //   break;
    // case 122:
    //   astro_branched_zOp::branchCompute[122](worker);
    //   break;
    // case 123:
    //   astro_branched_zOp::branchCompute[123](worker);
    //   break;
    // case 124:
    //   astro_branched_zOp::branchCompute[124](worker);
    //   break;
    // case 125:
    //   astro_branched_zOp::branchCompute[125](worker);
    //   break;
    // case 126:
    //   astro_branched_zOp::branchCompute[126](worker);
    //   break;
    // case 127:
    //   astro_branched_zOp::branchCompute[127](worker);
    //   break;
    // case 128:
    //   astro_branched_zOp::branchCompute[128](worker);
    //   break;
    // case 129:
    //   astro_branched_zOp::branchCompute[129](worker);
    //   break;
    // case 130:
    //   astro_branched_zOp::branchCompute[130](worker);
    //   break;
    // case 131:
    //   astro_branched_zOp::branchCompute[131](worker);
    //   break;
    // case 132:
    //   astro_branched_zOp::branchCompute[132](worker);
    //   break;
    // case 133:
    //   astro_branched_zOp::branchCompute[133](worker);
    //   break;
    // case 134:
    //   astro_branched_zOp::branchCompute[134](worker);
    //   break;
    // case 135:
    //   astro_branched_zOp::branchCompute[135](worker);
    //   break;
    // case 136:
    //   astro_branched_zOp::branchCompute[136](worker);
    //   break;
    // case 137:
    //   astro_branched_zOp::branchCompute[137](worker);
    //   break;
    // case 138:
    //   astro_branched_zOp::branchCompute[138](worker);
    //   break;
    // case 139:
    //   astro_branched_zOp::branchCompute[139](worker);
    //   break;
    // case 140:
    //   astro_branched_zOp::branchCompute[140](worker);
    //   break;
    // case 141:
    //   astro_branched_zOp::branchCompute[141](worker);
    //   break;
    // case 142:
    //   astro_branched_zOp::branchCompute[142](worker);
    //   break;
    // case 143:
    //   astro_branched_zOp::branchCompute[143](worker);
    //   break;
    // case 144:
    //   astro_branched_zOp::branchCompute[144](worker);
    //   break;
    // case 145:
    //   astro_branched_zOp::branchCompute[145](worker);
    //   break;
    // case 146:
    //   astro_branched_zOp::branchCompute[146](worker);
    //   break;
    // case 147:
    //   astro_branched_zOp::branchCompute[147](worker);
    //   break;
    // case 148:
    //   astro_branched_zOp::branchCompute[148](worker);
    //   break;
    // case 149:
    //   astro_branched_zOp::branchCompute[149](worker);
    //   break;
    // case 150:
    //   astro_branched_zOp::branchCompute[150](worker);
    //   break;
    // case 151:
    //   astro_branched_zOp::branchCompute[151](worker);
    //   break;
    // case 152:
    //   astro_branched_zOp::branchCompute[152](worker);
    //   break;
    // case 153:
    //   astro_branched_zOp::branchCompute[153](worker);
    //   break;
    // case 154:
    //   astro_branched_zOp::branchCompute[154](worker);
    //   break;
    // case 155:
    //   astro_branched_zOp::branchCompute[155](worker);
    //   break;
    // case 156:
    //   astro_branched_zOp::branchCompute[156](worker);
    //   break;
    // case 157:
    //   astro_branched_zOp::branchCompute[157](worker);
    //   break;
    // case 158:
    //   astro_branched_zOp::branchCompute[158](worker);
    //   break;
    // case 159:
    //   astro_branched_zOp::branchCompute[159](worker);
    //   break;
    // case 160:
    //   astro_branched_zOp::branchCompute[160](worker);
    //   break;
    // case 161:
    //   astro_branched_zOp::branchCompute[161](worker);
    //   break;
    // case 162:
    //   astro_branched_zOp::branchCompute[162](worker);
    //   break;
    // case 163:
    //   astro_branched_zOp::branchCompute[163](worker);
    //   break;
    // case 164:
    //   astro_branched_zOp::branchCompute[164](worker);
    //   break;
    // case 165:
    //   astro_branched_zOp::branchCompute[165](worker);
    //   break;
    // case 166:
    //   astro_branched_zOp::branchCompute[166](worker);
    //   break;
    // case 167:
    //   astro_branched_zOp::branchCompute[167](worker);
    //   break;
    // case 168:
    //   astro_branched_zOp::branchCompute[168](worker);
    //   break;
    // case 169:
    //   astro_branched_zOp::branchCompute[169](worker);
    //   break;
    // case 170:
    //   astro_branched_zOp::branchCompute[170](worker);
    //   break;
    // case 171:
    //   astro_branched_zOp::branchCompute[171](worker);
    //   break;
    // case 172:
    //   astro_branched_zOp::branchCompute[172](worker);
    //   break;
    // case 173:
    //   astro_branched_zOp::branchCompute[173](worker);
    //   break;
    // case 174:
    //   astro_branched_zOp::branchCompute[174](worker);
    //   break;
    // case 175:
    //   astro_branched_zOp::branchCompute[175](worker);
    //   break;
    // case 176:
    //   astro_branched_zOp::branchCompute[176](worker);
    //   break;
    // case 177:
    //   astro_branched_zOp::branchCompute[177](worker);
    //   break;
    // case 178:
    //   astro_branched_zOp::branchCompute[178](worker);
    //   break;
    // case 179:
    //   astro_branched_zOp::branchCompute[179](worker);
    //   break;
    // case 180:
    //   astro_branched_zOp::branchCompute[180](worker);
    //   break;
    // case 181:
    //   astro_branched_zOp::branchCompute[181](worker);
    //   break;
    // case 182:
    //   astro_branched_zOp::branchCompute[182](worker);
    //   break;
    // case 183:
    //   astro_branched_zOp::branchCompute[183](worker);
    //   break;
    // case 184:
    //   astro_branched_zOp::branchCompute[184](worker);
    //   break;
    // case 185:
    //   astro_branched_zOp::branchCompute[185](worker);
    //   break;
    // case 186:
    //   astro_branched_zOp::branchCompute[186](worker);
    //   break;
    // case 187:
    //   astro_branched_zOp::branchCompute[187](worker);
    //   break;
    // case 188:
    //   astro_branched_zOp::branchCompute[188](worker);
    //   break;
    // case 189:
    //   astro_branched_zOp::branchCompute[189](worker);
    //   break;
    // case 190:
    //   astro_branched_zOp::branchCompute[190](worker);
    //   break;
    // case 191:
    //   astro_branched_zOp::branchCompute[191](worker);
    //   break;
    // case 192:
    //   astro_branched_zOp::branchCompute[192](worker);
    //   break;
    // case 193:
    //   astro_branched_zOp::branchCompute[193](worker);
    //   break;
    // case 194:
    //   astro_branched_zOp::branchCompute[194](worker);
    //   break;
    // case 195:
    //   astro_branched_zOp::branchCompute[195](worker);
    //   break;
    // case 196:
    //   astro_branched_zOp::branchCompute[196](worker);
    //   break;
    // case 197:
    //   astro_branched_zOp::branchCompute[197](worker);
    //   break;
    // case 198:
    //   astro_branched_zOp::branchCompute[198](worker);
    //   break;
    // case 199:
    //   astro_branched_zOp::branchCompute[199](worker);
    //   break;
    // case 200:
    //   astro_branched_zOp::branchCompute[200](worker);
    //   break;
    // case 201:
    //   astro_branched_zOp::branchCompute[201](worker);
    //   break;
    // case 202:
    //   astro_branched_zOp::branchCompute[202](worker);
    //   break;
    // case 203:
    //   astro_branched_zOp::branchCompute[203](worker);
    //   break;
    // case 204:
    //   astro_branched_zOp::branchCompute[204](worker);
    //   break;
    // case 205:
    //   astro_branched_zOp::branchCompute[205](worker);
    //   break;
    // case 206:
    //   astro_branched_zOp::branchCompute[206](worker);
    //   break;
    // case 207:
    //   astro_branched_zOp::branchCompute[207](worker);
    //   break;
    // case 208:
    //   astro_branched_zOp::branchCompute[208](worker);
    //   break;
    // case 209:
    //   astro_branched_zOp::branchCompute[209](worker);
    //   break;
    // case 210:
    //   astro_branched_zOp::branchCompute[210](worker);
    //   break;
    // case 211:
    //   astro_branched_zOp::branchCompute[211](worker);
    //   break;
    // case 212:
    //   astro_branched_zOp::branchCompute[212](worker);
    //   break;
    // case 213:
    //   astro_branched_zOp::branchCompute[213](worker);
    //   break;
    // case 214:
    //   astro_branched_zOp::branchCompute[214](worker);
    //   break;
    // case 215:
    //   astro_branched_zOp::branchCompute[215](worker);
    //   break;
    // case 216:
    //   astro_branched_zOp::branchCompute[216](worker);
    //   break;
    // case 217:
    //   astro_branched_zOp::branchCompute[217](worker);
    //   break;
    // case 218:
    //   astro_branched_zOp::branchCompute[218](worker);
    //   break;
    // case 219:
    //   astro_branched_zOp::branchCompute[219](worker);
    //   break;
    // case 220:
    //   astro_branched_zOp::branchCompute[220](worker);
    //   break;
    // case 221:
    //   astro_branched_zOp::branchCompute[221](worker);
    //   break;
    // case 222:
    //   astro_branched_zOp::branchCompute[222](worker);
    //   break;
    // case 223:
    //   astro_branched_zOp::branchCompute[223](worker);
    //   break;
    // case 224:
    //   astro_branched_zOp::branchCompute[224](worker);
    //   break;
    // case 225:
    //   astro_branched_zOp::branchCompute[225](worker);
    //   break;
    // case 226:
    //   astro_branched_zOp::branchCompute[226](worker);
    //   break;
    // case 227:
    //   astro_branched_zOp::branchCompute[227](worker);
    //   break;
    // case 228:
    //   astro_branched_zOp::branchCompute[228](worker);
    //   break;
    // case 229:
    //   astro_branched_zOp::branchCompute[229](worker);
    //   break;
    // case 230:
    //   astro_branched_zOp::branchCompute[230](worker);
    //   break;
    // case 231:
    //   astro_branched_zOp::branchCompute[231](worker);
    //   break;
    // case 232:
    //   astro_branched_zOp::branchCompute[232](worker);
    //   break;
    // case 233:
    //   astro_branched_zOp::branchCompute[233](worker);
    //   break;
    // case 234:
    //   astro_branched_zOp::branchCompute[234](worker);
    //   break;
    // case 235:
    //   astro_branched_zOp::branchCompute[235](worker);
    //   break;
    // case 236:
    //   astro_branched_zOp::branchCompute[236](worker);
    //   break;
    // case 237:
    //   astro_branched_zOp::branchCompute[237](worker);
    //   break;
    // case 238:
    //   astro_branched_zOp::branchCompute[238](worker);
    //   break;
    // case 239:
    //   astro_branched_zOp::branchCompute[239](worker);
    //   break;
    // case 240:
    //   astro_branched_zOp::branchCompute[240](worker);
    //   break;
    // case 241:
    //   astro_branched_zOp::branchCompute[241](worker);
    //   break;
    // case 242:
    //   astro_branched_zOp::branchCompute[242](worker);
    //   break;
    // case 243:
    //   astro_branched_zOp::branchCompute[243](worker);
    //   break;
    // case 244:
    //   astro_branched_zOp::branchCompute[244](worker);
    //   break;
    // case 245:
    //   astro_branched_zOp::branchCompute[245](worker);
    //   break;
    // case 246:
    //   astro_branched_zOp::branchCompute[246](worker);
    //   break;
    // case 247:
    //   astro_branched_zOp::branchCompute[247](worker);
    //   break;
    // case 248:
    //   astro_branched_zOp::branchCompute[248](worker);
    //   break;
    // case 249:
    //   astro_branched_zOp::branchCompute[249](worker);
    //   break;
    // case 250:
    //   astro_branched_zOp::branchCompute[250](worker);
    //   break;
    // case 251:
    //   astro_branched_zOp::branchCompute[251](worker);
    //   break;
    // case 252:
    //   astro_branched_zOp::branchCompute[252](worker);
    //   break;
    // case 253:
    //   astro_branched_zOp::branchCompute[253](worker);
    //   break;
    // case 254:
    //   astro_branched_zOp::branchCompute[254](worker);
    //   break;
    // case 255:
    //   astro_branched_zOp::branchCompute[255](worker);
    //   break;
    // default:
    //   // Handle default case if needed
    //   break;
    // }


  int branchComputeSize = 14;
}

#endif

