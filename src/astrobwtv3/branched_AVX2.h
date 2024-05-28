#include "astrobwtv3.h"

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

  void __attribute__((noinline)) blendStep(workerData &worker) {
    worker.data = _mm256_blendv_epi8(worker.old, worker.data, genMask(worker.pos2 - worker.pos1));
  }

  void __attribute__((noinline)) storeStep(workerData &worker) {
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], worker.data);
  }

  // op codes

  void op0(workerData &worker) {
    p0(worker.data);
    r5(worker.data);
    m0(worker.data);
    r0(worker.data);
    blendStep(worker);
    storeStep(worker);

    if ((worker.pos2 - worker.pos1) % 2 == 1) {
      worker.t1 = worker.chunk[worker.pos1];
      worker.t2 = worker.chunk[worker.pos2];
      worker.chunk[worker.pos1] = reverse8(worker.t2);
      worker.chunk[worker.pos2] = reverse8(worker.t1);
    }
  }

  void op1(workerData &worker) {
    __m256i shift = _mm256_and_si256(worker.data, vec_3);
    worker.data = _mm256_sllv_epi8(worker.data, shift);
    r1(worker.data);
    worker.data = _mm256_and_si256(worker.data, _mm256_set1_epi8(worker.prev_chunk[worker.pos2]));
    a0(worker.data);
    blendStep(worker);
    storeStep(worker);
  }

  void op2(workerData &worker) {
    p0(worker.data);
    revData(worker.data);
    __m256i shift = _mm256_and_si256(worker.data, vec_3);
    worker.data = _mm256_sllv_epi8(worker.data, shift);
    p0(worker.data);
    blendStep(worker);
    storeStep(worker);
  }

  void op3(workerData &worker) {
    worker.data = _mm256_rolv_epi8(worker.data, _mm256_add_epi8(worker.data, vec_3));
    worker.data = _mm256_xor_si256(worker.data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    r1(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op4(workerData &worker) {
    notData(worker.data);
    sr03(worker.data);
    r0(worker.data);
    subx97(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op5(workerData &worker) {
    p0(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    sl03(worker.data);
    sr03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op6(workerData &worker) {
    sl03(worker.data);
    r3(worker.data);
    notData(worker.data);

    __m256i x = _mm256_xor_si256(worker.data,_mm256_set1_epi8(97));
    worker.data = _mm256_sub_epi8(worker.data, x);

    blendStep(worker);
    storeStep(worker);
  }

  void op7(workerData &worker) {
    a0(worker.data);
    r0(worker.data);

    p0(worker.data);
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op8(workerData &worker) {
    notData(worker.data);
    r2(worker.data);
    sl03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op9(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    x0_r4(worker.data);
    sr03(worker.data);
    x0_r2(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op10(workerData &worker) {
    notData(worker.data);
    m0(worker.data);
    r3(worker.data);
    m0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op11(workerData &worker) {
    r3(worker.data);
    r3(worker.data);
    worker.data = _mm256_and_si256(worker.data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
    r0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op12(workerData &worker) {
    worker.data = _mm256_xor_si256(worker.data, _mm256_rol_epi8(worker.data, 2));
    m0(worker.data);
    worker.data = _mm256_xor_si256(worker.data, _mm256_rol_epi8(worker.data, 2));
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op13(workerData &worker) {
    r1(worker.data);
    worker.data = _mm256_xor_si256(worker.data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
    sr03(worker.data);
    r5(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op14(workerData &worker) {
    sr03(worker.data);
    sl03(worker.data);
    m0(worker.data);
    sl03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op15(workerData &worker) {
    worker.data = _mm256_xor_si256(worker.data, _mm256_rol_epi8(worker.data,2));
    sl03(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    subx97(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op16(workerData &worker) {
    worker.data = _mm256_xor_si256(worker.data, _mm256_rol_epi8(worker.data,4));
    m0(worker.data);
    r1(worker.data);
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op17(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    m0(worker.data);
    r5(worker.data);
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op18(workerData &worker) {
    x0_r4(worker.data);
    r1(worker.data);
    blendStep(worker);
    storeStep(worker);
  }

  void op19(workerData &worker) {
    subx97(worker.data);
    r5(worker.data);
    sl03(worker.data);
    a0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op20(workerData &worker) {
    addp2(worker.data, worker.chunk[worker.pos2]);
    xp2(worker.data, worker.chunk[worker.pos2]);
    revData(worker.data);
    x0_r2(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op21(workerData &worker) {
    r1(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    a0(worker.data);
    worker.data = _mm256_and_si256(worker.data,_mm256_set1_epi8(worker.chunk[worker.pos2]));

    blendStep(worker);
    storeStep(worker);
  }

  void op22(workerData &worker) {
    sl03(worker.data);
    revData(worker.data);
    m0(worker.data);
    r1(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op23(workerData &worker) {
    worker.data = _mm256_rol_epi8(worker.data, 4);
    worker.data = _mm256_xor_si256(worker.data,popcnt256_epi8(worker.data));
    worker.data = _mm256_and_si256(worker.data,_mm256_set1_epi8(worker.chunk[worker.pos2]));

    blendStep(worker);
    storeStep(worker);
  }

  void op24(workerData &worker) {
    a0(worker.data);
    sr03(worker.data);
    x0_r4(worker.data);
    r5(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op25(workerData &worker) {
    
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

  void op26(workerData &worker) {
    m0(worker.data);
    worker.data = _mm256_xor_si256(worker.data,popcnt256_epi8(worker.data));
    a0(worker.data);
    revData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op27(workerData &worker) {
    r5(worker.data);
    worker.data = _mm256_and_si256(worker.data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
    x0_r4(worker.data);
    r5(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op28(workerData &worker) {
    sl03(worker.data);
    a0(worker.data);
    a0(worker.data);
    r5(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op29(workerData &worker) {
    m0(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    sr03(worker.data);
    a0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op30(workerData &worker) {
    addp2(worker.data, worker.chunk[worker.pos2]);
    x0_r4(worker.data);
    r5(worker.data);
    sl03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op31(workerData &worker) {
    notData(worker.data);
    x0_r2(worker.data);
    sl03(worker.data);
    m0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op32(workerData &worker) {
    x0_r2(worker.data);
    revData(worker.data);
    r3(worker.data);
    x0_r2(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op33(workerData &worker) {
    r0(worker.data);
    x0_r4(worker.data);
    revData(worker.data);
    m0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op34(workerData &worker) {
    subx97(worker.data);
    sl03(worker.data);
    sl03(worker.data);
    subx97(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op35(workerData &worker) {
    a0(worker.data);
    notData(worker.data);
    r1(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op36(workerData &worker) {
    p0(worker.data);
    r1(worker.data);
    x0_r2(worker.data);
    r1(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op37(workerData &worker) {
    r0(worker.data);
    sr03(worker.data);
    sr03(worker.data);
    m0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op38(workerData &worker) {
    
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

  void op39(workerData &worker) {
    x0_r2(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    sr03(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op40(workerData &worker) {
    r0(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    p0(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op41(workerData &worker) {
    r5(worker.data);
    subx97(worker.data);
    r3(worker.data);
    x0_r4(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op42(workerData &worker) {
    worker.data = _mm256_rol_epi8(worker.data, 4);
    x0_r2(worker.data);
    r0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op43(workerData &worker) {
    addp2(worker.data, worker.chunk[worker.pos2]);
    a0(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    subx97(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op44(workerData &worker) {
    
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

  void op45(workerData &worker) {
    r2(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    p0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op46(workerData &worker) {
    p0(worker.data);
    a0(worker.data);
    r5(worker.data);
    x0_r4(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op47(workerData &worker) {
    r5(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    r5(worker.data);
    sl03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op48(workerData &worker) {
    r0(worker.data);
    r5(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op49(workerData &worker) {
    p0(worker.data);
    a0(worker.data);
    revData(worker.data);
    x0_r4(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op50(workerData &worker) {
    revData(worker.data);
    r3(worker.data);
    a0(worker.data);
    r1(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op51(workerData &worker) {
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

  void op52(workerData &worker) {
    r0(worker.data);
    sr03(worker.data);
    notData(worker.data);
    p0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op53(workerData &worker) {
    
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

  void op54(workerData &worker) {
    revData(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op55(workerData &worker) {
    
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

  void op56(workerData &worker) {
    x0_r2(worker.data);
    m0(worker.data);
    notData(worker.data);
    r1(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op57(workerData &worker) {
    r0(worker.data);
    revData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op58(workerData &worker) {
    revData(worker.data);
    x0_r2(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    a0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op59(workerData &worker) {
    r1(worker.data);
    m0(worker.data);
    r0(worker.data);
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op60(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    notData(worker.data);
    m0(worker.data);
    r3(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op61(workerData &worker) {
    r5(worker.data);
    sl03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op62(workerData &worker) {
    addp2(worker.data, worker.chunk[worker.pos2]);
    notData(worker.data);
    x0_r2(worker.data);
    a0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op63(workerData &worker) {
    r5(worker.data);
    p0(worker.data);
    subx97(worker.data);
    a0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op64(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    revData(worker.data);
    x0_r4(worker.data);
    m0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op65(workerData &worker) {
    x0_r2(worker.data);
    m0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op66(workerData &worker) {
    x0_r2(worker.data);
    revData(worker.data);
    x0_r4(worker.data);
    r1(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op67(workerData &worker) {
    r1(worker.data);
    p0(worker.data);
    x0_r2(worker.data);
    r5(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op68(workerData &worker) {
    addp2(worker.data, worker.chunk[worker.pos2]);
    notData(worker.data);
    x0_r4(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op69(workerData &worker) {
    a0(worker.data);
    m0(worker.data);
    revData(worker.data);
    sr03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op70(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    m0(worker.data);
    sr03(worker.data);
    x0_r4(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op71(workerData &worker) {
    r5(worker.data);
    notData(worker.data);
    m0(worker.data);
    sl03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op72(workerData &worker) {
    revData(worker.data);
    p0(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    sl03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op73(workerData &worker) {
    p0(worker.data);
    revData(worker.data);
    r5(worker.data);
    subx97(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op74(workerData &worker) {
    m0(worker.data);
    r3(worker.data);
    revData(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op75(workerData &worker) {
    m0(worker.data);
    p0(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    x0_r4(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op76(workerData &worker) {
    r0(worker.data);
    x0_r2(worker.data);
    r5(worker.data);
    sr03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op77(workerData &worker) {
    r3(worker.data);
    a0(worker.data);
    sl03(worker.data);
    p0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op78(workerData &worker) {
    r0(worker.data);
    revData(worker.data);
    m0(worker.data);
    subx97(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op79(workerData &worker) {
    x0_r4(worker.data);
    x0_r2(worker.data);
    a0(worker.data);
    m0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op80(workerData &worker) {
    r0(worker.data);
    sl03(worker.data);
    a0(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op81(workerData &worker) {
    x0_r4(worker.data);
    sl03(worker.data);
    r0(worker.data);
    p0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op82(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    sr03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op83(workerData &worker) {
    sl03(worker.data);
    revData(worker.data);
    r3(worker.data);
    revData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op84(workerData &worker) {
    subx97(worker.data);
    r1(worker.data);
    sl03(worker.data);
    a0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op85(workerData &worker) {
    sr03(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    r0(worker.data);
    sl03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op86(workerData &worker) {
    x0_r4(worker.data);
    r0(worker.data);
    x0_r4(worker.data);
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op87(workerData &worker) {
    a0(worker.data);
    r3(worker.data);
    x0_r4(worker.data);
    a0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op88(workerData &worker) {
    x0_r2(worker.data);
    r1(worker.data);
    m0(worker.data);
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op89(workerData &worker) {
    a0(worker.data);
    m0(worker.data);
    notData(worker.data);
    x0_r2(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op90(workerData &worker) {
    revData(worker.data);
    r3(worker.data);
    r3(worker.data);
    sr03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op91(workerData &worker) {
    p0(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    x0_r4(worker.data);
    revData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op92(workerData &worker) {
    p0(worker.data);
    notData(worker.data);
    p0(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op93(workerData &worker) {
    x0_r2(worker.data);
    m0(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    a0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op94(workerData &worker) {
    r1(worker.data);
    r0(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    sl03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op95(workerData &worker) {
    r1(worker.data);
    notData(worker.data);
    r2(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op96(workerData &worker) {
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

  void op97(workerData &worker) {
    r1(worker.data);
    sl03(worker.data);
    p0(worker.data);
    sr03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op98(workerData &worker) {
    x0_r4(worker.data);
    sl03(worker.data);
    sr03(worker.data);
    x0_r4(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op99(workerData &worker) {
    x0_r4(worker.data);
    subx97(worker.data);
    revData(worker.data);
    sr03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op100(workerData &worker) {
    r0(worker.data);
    sl03(worker.data);
    revData(worker.data);
    p0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op101(workerData &worker) {
    sr03(worker.data);
    p0(worker.data);
    sr03(worker.data);
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op102(workerData &worker) {
    r3(worker.data);
    subx97(worker.data);
    a0(worker.data);
    r3(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op103(workerData &worker) {
    r1(worker.data);
    revData(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    r0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op104(workerData &worker) {
    revData(worker.data);
    p0(worker.data);
    r5(worker.data);
    a0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op105(workerData &worker) {
    sl03(worker.data);
    r3(worker.data);
    r0(worker.data);
    x0_r2(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op106(workerData &worker) {
    revData(worker.data);
    x0_r4(worker.data);
    r1(worker.data);
    m0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op107(workerData &worker) {
    sr03(worker.data);
    x0_r2(worker.data);
    r3(worker.data);
    r3(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op108(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    notData(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    x0_r2(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op109(workerData &worker) {
    m0(worker.data);
    r0(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    x0_r2(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op110(workerData &worker) {
    a0(worker.data);
    x0_r2(worker.data);
    x0_r2(worker.data);
    sr03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op111(workerData &worker) {
    m0(worker.data);
    revData(worker.data);
    m0(worker.data);
    sr03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op112(workerData &worker) {
    r3(worker.data);
    notData(worker.data);
    r5(worker.data);
    subx97(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op113(workerData &worker) {
    r3(worker.data);
    r3(worker.data);
    p0(worker.data);
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op114(workerData &worker) {
    r1(worker.data);
    revData(worker.data);
    r0(worker.data);
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op115(workerData &worker) {
    r0(worker.data);
    r5(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    r3(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op116(workerData &worker) {
    addp2(worker.data, worker.chunk[worker.pos2]);
    xp2(worker.data, worker.chunk[worker.pos2]);
    p0(worker.data);
    sl03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op117(workerData &worker) {
    sl03(worker.data);
    r3(worker.data);
    sl03(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op118(workerData &worker) {
    sr03(worker.data);
    a0(worker.data);
    sl03(worker.data);
    r5(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op119(workerData &worker) {
    revData(worker.data);
    x0_r2(worker.data);
    notData(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op120(workerData &worker) {
    x0_r2(worker.data);
    m0(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    revData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op121(workerData &worker) {
    sr03(worker.data);
    a0(worker.data);
    p0(worker.data);
    m0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op122(workerData &worker) {
    x0_r4(worker.data);
    r0(worker.data);
    r5(worker.data);
    x0_r2(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op123(workerData &worker) {
    addp2(worker.data, worker.chunk[worker.pos2]);
    notData(worker.data);
    r3(worker.data);
    r3(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op124(workerData &worker) {
    x0_r2(worker.data);
    x0_r2(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op125(workerData &worker) {
    revData(worker.data);
    x0_r2(worker.data);
    a0(worker.data);
    sr03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op126(workerData &worker) {
    r1(worker.data);
    revData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op127(workerData &worker) {
    sl03(worker.data);
    m0(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    xp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op128(workerData &worker) {
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

  void op129(workerData &worker) {
    notData(worker.data);
    p0(worker.data);
    p0(worker.data);
    sr03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op130(workerData &worker) {
    sr03(worker.data);
    r0(worker.data);
    r1(worker.data);
    x0_r4(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op131(workerData &worker) {
    subx97(worker.data);
    r1(worker.data);
    p0(worker.data);
    m0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op132(workerData &worker) {
    addp2(worker.data, worker.chunk[worker.pos2]);
    revData(worker.data);
    r5(worker.data);
    x0_r2(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op133(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    r5(worker.data);
    x0_r2(worker.data);
    sl03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op134(workerData &worker) {
    notData(worker.data);
    x0_r4(worker.data);
    r1(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op135(workerData &worker) {
    sr03(worker.data);
    x0_r2(worker.data);
    a0(worker.data);
    revData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op136(workerData &worker) {
    sr03(worker.data);
    subx97(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    r5(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op137(workerData &worker) {
    r5(worker.data);
    sr03(worker.data);
    revData(worker.data);
    r0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op138(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    xp2(worker.data, worker.chunk[worker.pos2]);
    a0(worker.data);
    subx97(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op139(workerData &worker) {
    x0_r2(worker.data);
    r3(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op140(workerData &worker) {
    r1(worker.data);
    x0_r2(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    r5(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op141(workerData &worker) {
    r1(worker.data);
    subx97(worker.data);
    p0(worker.data);
    a0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op142(workerData &worker) {
    addp2(worker.data, worker.chunk[worker.pos2]);
    r5(worker.data);
    revData(worker.data);
    x0_r2(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op143(workerData &worker) {
    addp2(worker.data, worker.chunk[worker.pos2]);
    r3(worker.data);
    sr03(worker.data);
    sl03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op144(workerData &worker) {
    r0(worker.data);
    sl03(worker.data);
    notData(worker.data);
    r0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op145(workerData &worker) {
    revData(worker.data);
    x0_r4(worker.data);
    x0_r2(worker.data);
    x0_r4(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op146(workerData &worker) {
    addp2(worker.data, worker.chunk[worker.pos2]);
    sl03(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    p0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op147(workerData &worker) {
    notData(worker.data);
    sl03(worker.data);
    x0_r4(worker.data);
    m0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op148(workerData &worker) {
    addp2(worker.data, worker.chunk[worker.pos2]);
    r5(worker.data);
    sl03(worker.data);
    subx97(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op149(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    revData(worker.data);
    subx97(worker.data);
    a0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op150(workerData &worker) {
    sl03(worker.data);
    sl03(worker.data);
    sl03(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op151(workerData &worker) {
    a0(worker.data);
    sl03(worker.data);
    m0(worker.data);
    sl03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op152(workerData &worker) {
    sr03(worker.data);
    notData(worker.data);
    sl03(worker.data);
    x0_r2(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op153(workerData &worker) {
    worker.data = _mm256_rol_epi8(worker.data, 4);

    blendStep(worker);
    storeStep(worker);
  }

  void op154(workerData &worker) {
    r5(worker.data);
    notData(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    p0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op155(workerData &worker) {
    subx97(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    p0(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op156(workerData &worker) {
    sr03(worker.data);
    sr03(worker.data);
    worker.data = _mm256_rol_epi8(worker.data, 4);

    blendStep(worker);
    storeStep(worker);
  }

  void op157(workerData &worker) {
    sr03(worker.data);
    sl03(worker.data);
    r0(worker.data);
    r1(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op158(workerData &worker) {
    p0(worker.data);
    r3(worker.data);
    a0(worker.data);
    r1(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op159(workerData &worker) {
    subx97(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    r0(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op160(workerData &worker) {
    sr03(worker.data);
    revData(worker.data);
    worker.data = _mm256_rol_epi8(worker.data, 4);

    blendStep(worker);
    storeStep(worker);
  }

  void op161(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    xp2(worker.data, worker.chunk[worker.pos2]);
    r5(worker.data);
    r0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op162(workerData &worker) {
    m0(worker.data);
    revData(worker.data);
    x0_r2(worker.data);
    subx97(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op163(workerData &worker) {
    sl03(worker.data);
    subx97(worker.data);
    x0_r4(worker.data);
    r1(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op164(workerData &worker) {
    m0(worker.data);
    p0(worker.data);
    subx97(worker.data);
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op165(workerData &worker) {
    x0_r4(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    sl03(worker.data);
    a0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op166(workerData &worker) {
    r3(worker.data);
    a0(worker.data);
    x0_r2(worker.data);
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op167(workerData &worker) {
    m0(worker.data);
    sr03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op168(workerData &worker) {
    r0(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    r0(worker.data);
    r1(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op169(workerData &worker) {
    r1(worker.data);
    sl03(worker.data);
    x0_r4(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op170(workerData &worker) {
    subx97(worker.data);
    revData(worker.data);
    subx97(worker.data);
    m0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op171(workerData &worker) {
    r3(worker.data);
    subx97(worker.data);
    p0(worker.data);
    revData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op172(workerData &worker) {
    x0_r4(worker.data);
    subx97(worker.data);
    sl03(worker.data);
    r1(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op173(workerData &worker) {
    notData(worker.data);
    sl03(worker.data);
    m0(worker.data);
    a0(worker.data);

  blendStep(worker);
    storeStep(worker);
  }

  void op174(workerData &worker) {
    notData(worker.data);
    r0(worker.data);
    p0(worker.data);
    p0(worker.data);

  blendStep(worker);
    storeStep(worker);
  }

  void op175(workerData &worker) {
    r3(worker.data);
    subx97(worker.data);
    m0(worker.data);
    r5(worker.data);

  blendStep(worker);
    storeStep(worker);
  }

  void op176(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    m0(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    r5(worker.data);

  blendStep(worker);
    storeStep(worker);
  }

  void op177(workerData &worker) {
    p0(worker.data);
    x0_r2(worker.data);
    x0_r2(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);

  blendStep(worker);
    storeStep(worker);
  }

  void op178(workerData &worker) {
    addp2(worker.data, worker.chunk[worker.pos2]);
    a0(worker.data);
    notData(worker.data);
    r1(worker.data);

  blendStep(worker);
    storeStep(worker);
  }

  void op179(workerData &worker) {
    x0_r2(worker.data);
    a0(worker.data);
    sr03(worker.data);
    revData(worker.data);

  blendStep(worker);
    storeStep(worker);
  }

  void op180(workerData &worker) {
    sr03(worker.data);
    x0_r4(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    subx97(worker.data);

  blendStep(worker);
    storeStep(worker);
  }

  void op181(workerData &worker) {
    notData(worker.data);
    sl03(worker.data);
    x0_r2(worker.data);
    r5(worker.data);

  blendStep(worker);
    storeStep(worker);
  }

  void op182(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    r3(worker.data);
    r3(worker.data);
    x0_r4(worker.data);

  blendStep(worker);
    storeStep(worker);
  }

  void op183(workerData &worker) {
    a0(worker.data);
    subx97(worker.data);
    subx97(worker.data);
    m0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op184(workerData &worker) {
    sl03(worker.data);
    m0(worker.data);
    r5(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op185(workerData &worker) {
    notData(worker.data);
    x0_r4(worker.data);
    r5(worker.data);
    sr03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op186(workerData &worker) {
    x0_r2(worker.data);
    x0_r4(worker.data);
    subx97(worker.data);
    sr03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op187(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    notData(worker.data);
    a0(worker.data);
    r3(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op188(workerData &worker) {
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

  void op189(workerData &worker) {
    r5(worker.data);
    x0_r4(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    subx97(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op190(workerData &worker) {
    r5(worker.data);
    sr03(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    x0_r2(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op191(workerData &worker) {
    a0(worker.data);
    r3(worker.data);
    r0(worker.data);
    sr03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op192(workerData &worker) {
    a0(worker.data);
    sl03(worker.data);
    a0(worker.data);
    m0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op193(workerData &worker) {
    addp2(worker.data, worker.chunk[worker.pos2]);
    sl03(worker.data);
    r0(worker.data);
    r1(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op194(workerData &worker) {
    addp2(worker.data, worker.chunk[worker.pos2]);
    r0(worker.data);
    sl03(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op195(workerData &worker) {
    p0(worker.data);
    x0_r2(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    x0_r4(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op196(workerData &worker) {
    r3(worker.data);
    revData(worker.data);
    sl03(worker.data);
    r1(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op197(workerData &worker) {
    x0_r4(worker.data);
    r0(worker.data);
    m0(worker.data);
    m0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op198(workerData &worker) {
    sr03(worker.data);
    sr03(worker.data);
    revData(worker.data);
    r1(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op199(workerData &worker) {
    notData(worker.data);
    a0(worker.data);
    m0(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op200(workerData &worker) {
    sr03(worker.data);
    p0(worker.data);
    revData(worker.data);
    revData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op201(workerData &worker) {
    r3(worker.data);
    x0_r2(worker.data);
    x0_r4(worker.data);
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op202(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    notData(worker.data);
    r0(worker.data);
    r5(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op203(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    addp2(worker.data, worker.chunk[worker.pos2]);
    r1(worker.data);
    r0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op204(workerData &worker) {
    r5(worker.data);
    x0_r2(worker.data);
    r0(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op205(workerData &worker) {
    p0(worker.data);
    x0_r4(worker.data);
    sl03(worker.data);
    a0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op206(workerData &worker) {
    x0_r4(worker.data);
    revData(worker.data);
    revData(worker.data);
    p0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op207(workerData &worker) {
    p0(worker.data);
    p0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op208(workerData &worker) {
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

  void op209(workerData &worker) {
    r5(worker.data);
    revData(worker.data);
    p0(worker.data);
    subx97(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op210(workerData &worker) {
    x0_r2(worker.data);
    r0(worker.data);
    r5(worker.data);
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op211(workerData &worker) {
    x0_r4(worker.data);
    a0(worker.data);
    subx97(worker.data);
    r0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op212(workerData &worker) {
    r0(worker.data);
    x0_r2(worker.data);
    // xp2(worker.data, worker.chunk[worker.pos2]);
    // xp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op213(workerData &worker) {
    a0(worker.data);
    sl03(worker.data);
    r3(worker.data);
    subx97(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op214(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    subx97(worker.data);
    sr03(worker.data);
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op215(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    addp2(worker.data, worker.chunk[worker.pos2]);
    sl03(worker.data);
    m0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op216(workerData &worker) {
    r0(worker.data);
    notData(worker.data);
    subx97(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op217(workerData &worker) {
    r5(worker.data);
    a0(worker.data);
    r1(worker.data);
    x0_r4(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op218(workerData &worker) {
    revData(worker.data);
    notData(worker.data);
    m0(worker.data);
    subx97(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op219(workerData &worker) {
    x0_r4(worker.data);
    r3(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    revData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op220(workerData &worker) {
    r1(worker.data);
    sl03(worker.data);
    revData(worker.data);
    sl03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op221(workerData &worker) {
    r5(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    notData(worker.data);
    revData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op222(workerData &worker) {
    sr03(worker.data);
    sl03(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    m0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op223(workerData &worker) {
    r3(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    r0(worker.data);
    subx97(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op224(workerData &worker) {
    x0_r2(worker.data);
    worker.data = _mm256_rol_epi8(worker.data, 4);
    sl03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op225(workerData &worker) {
    notData(worker.data);
    sr03(worker.data);
    revData(worker.data);
    r3(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op226(workerData &worker) {
    revData(worker.data);
    subx97(worker.data);
    m0(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op227(workerData &worker) {
    notData(worker.data);
    sl03(worker.data);
    subx97(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op228(workerData &worker) {
    a0(worker.data);
    sr03(worker.data);
    a0(worker.data);
    p0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op229(workerData &worker) {
    r3(worker.data);
    r0(worker.data);
    x0_r2(worker.data);
    p0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op230(workerData &worker) {
    m0(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    r0(worker.data);
    r0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op231(workerData &worker) {
    r3(worker.data);
    sr03(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    revData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op232(workerData &worker) {
    m0(worker.data);
    m0(worker.data);
    x0_r4(worker.data);
    r5(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op233(workerData &worker) {
    r1(worker.data);
    p0(worker.data);
    r3(worker.data);
    p0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op234(workerData &worker) {
    addp2(worker.data, worker.chunk[worker.pos2]);
    m0(worker.data);
    sr03(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op235(workerData &worker) {
    x0_r2(worker.data);
    m0(worker.data);
    r3(worker.data);
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op236(workerData &worker) {
    xp2(worker.data, worker.chunk[worker.pos2]);
    a0(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    subx97(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op237(workerData &worker) {
    r5(worker.data);
    sl03(worker.data);
    x0_r2(worker.data);
    r3(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op238(workerData &worker) {
    a0(worker.data);
    a0(worker.data);
    r3(worker.data);
    subx97(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op239(workerData &worker) {
    r3(worker.data);
    r3(worker.data);
    m0(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op240(workerData &worker) {
    notData(worker.data);
    a0(worker.data);
    addp2(worker.data, worker.chunk[worker.pos2]);
    sl03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op241(workerData &worker) {
    x0_r4(worker.data);
    p0(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);
    r1(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op242(workerData &worker) {
    a0(worker.data);
    a0(worker.data);
    subx97(worker.data);
    xp2(worker.data, worker.chunk[worker.pos2]);

    blendStep(worker);
    storeStep(worker);
  }

  void op243(workerData &worker) {
    r5(worker.data);
    x0_r2(worker.data);
    p0(worker.data);
    r1(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op244(workerData &worker) {
    notData(worker.data);
    x0_r2(worker.data);
    revData(worker.data);
    r5(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op245(workerData &worker) {
    subx97(worker.data);
    r5(worker.data);
    x0_r2(worker.data);
    sr03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op246(workerData &worker) {
    a0(worker.data);
    r1(worker.data);
    sr03(worker.data);
    a0(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op247(workerData &worker) {
    r5(worker.data);
    x0_r2(worker.data);
    r5(worker.data);
    notData(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op248(workerData &worker) {
    notData(worker.data);
    subx97(worker.data);
    p0(worker.data);
    r5(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op249(workerData &worker) {
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

  void op250(workerData &worker) {
    addp2(worker.data, worker.chunk[worker.pos2]);
    r0(worker.data);
    p0(worker.data);
    x0_r4(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op251(workerData &worker) {
    a0(worker.data);
    p0(worker.data);
    revData(worker.data);
    x0_r2(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op252(workerData &worker) {
    revData(worker.data);
    x0_r4(worker.data);
    x0_r2(worker.data);
    sl03(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op253(workerData &worker) {
    storeStep(worker);
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

  void op254(workerData &worker) {
    RC4_set_key(&worker.key, 256, worker.prev_chunk);

    p0(worker.data);
    r3(worker.data);
    x0_r2(worker.data);
    r3(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void op255(workerData &worker) {
    RC4_set_key(&worker.key, 256, worker.prev_chunk);

    p0(worker.data);
    r3(worker.data);
    x0_r2(worker.data);
    r3(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void r_op0(workerData &worker) {
    byte newVal = worker.simpleLookup[worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    worker.data = _mm256_blendv_epi8(worker.data, newVec, genMask(worker.pos2 - worker.pos1));
    storeStep(worker);

    if ((worker.pos2 - worker.pos1) % 2 == 1) {
      worker.t1 = worker.chunk[worker.pos1];
      worker.t2 = worker.chunk[worker.pos2];
      worker.chunk[worker.pos1] = reverse8(worker.t2);
      worker.chunk[worker.pos2] = reverse8(worker.t1);
      worker.isSame = false;
    }
  }

  void r_op1(workerData &worker) {
    byte newVal = worker.lookup3D[worker.branched_idx[worker.op] * 256 * 256 +
                                  worker.prev_chunk[worker.pos2] * 256 +
                                  worker.prev_chunk[worker.pos1]];

    __m256i newVec = _mm256_set1_epi8(newVal);
    worker.data = _mm256_blendv_epi8(worker.data, newVec, genMask(worker.pos2 - worker.pos1));
    storeStep(worker);

    return;
  }

  void r_op2(workerData &worker) {
    byte newVal = worker.simpleLookup[worker.reg_idx[worker.op] * 256 + worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    worker.data = _mm256_blendv_epi8(worker.data, newVec, genMask(worker.pos2 - worker.pos1));
    storeStep(worker);
  }

  void r_op253(workerData &worker) {
    storeStep(worker);
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

  void r_op254(workerData &worker) {
    RC4_set_key(&worker.key, 256, worker.prev_chunk);

    p0(worker.data);
    r3(worker.data);
    x0_r2(worker.data);
    r3(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  void r_op255(workerData &worker) {
    RC4_set_key(&worker.key, 256, worker.prev_chunk);

    p0(worker.data);
    r3(worker.data);
    x0_r2(worker.data);
    r3(worker.data);

    blendStep(worker);
    storeStep(worker);
  }

  typedef void (*OpFunc)(workerData &);

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

  int branchComputeSize = 14;
}