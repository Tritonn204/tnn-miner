#include "astrobwtv3.h"

namespace astro_branched_zOp {
  void op0(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2 - worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

    if ((worker.pos2 - worker.pos1) % 2 == 1) {
      worker.t1 = worker.chunk[worker.pos1];
      worker.t2 = worker.chunk[worker.pos2];
      worker.chunk[worker.pos1] = reverse8(worker.t2);
      worker.chunk[worker.pos2] = reverse8(worker.t1);
    }
  }

  void op1(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i shift = _mm256_and_si256(data, vec_3);
    data = _mm256_sllv_epi8(data, shift);
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.prev_chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);
    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2 - worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op2(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_reverse_epi8(data);
    __m256i shift = _mm256_and_si256(data, vec_3);
    data = _mm256_sllv_epi8(data, shift);
    pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2 - worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op3(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rolv_epi8(data,_mm256_add_epi8(data,vec_3));
    data = _mm256_xor_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data,1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op4(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_srlv_epi8(data,_mm256_and_si256(data,vec_3));
    data = _mm256_rolv_epi8(data,data);
    data = _mm256_sub_epi8(data,_mm256_xor_si256(data,_mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op5(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data,pop);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
    data = _mm256_srlv_epi8(data,_mm256_and_si256(data,vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op6(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    __m256i x = _mm256_xor_si256(data,_mm256_set1_epi8(97));
    data = _mm256_sub_epi8(data,x);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op7(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);;
    data = _mm256_rolv_epi8(data, data);

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data,pop);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op8(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rol_epi8(data,2);
    data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op9(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data,4));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data,2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op10(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op11(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;





    data = _mm256_rol_epi8(data, 6);
    data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op12(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_xor_si256(data, _mm256_rol_epi8(data,2));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data,2));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op13(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_srlv_epi8(data,_mm256_and_si256(data,vec_3));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op14(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_srlv_epi8(data,_mm256_and_si256(data,vec_3));
    data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op15(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_xor_si256(data, _mm256_rol_epi8(data,2));
    data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sub_epi8(data,_mm256_xor_si256(data,_mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op16(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_xor_si256(data, _mm256_rol_epi8(data,4));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data,1);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op17(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data,5);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op18(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op19(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_sub_epi8(data,_mm256_xor_si256(data,_mm256_set1_epi8(97)));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
    data = _mm256_add_epi8(data, data);;;

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op20(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op21(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);
    data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op22(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_mul_epi8(data,data);
    data = _mm256_rol_epi8(data,1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op23(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_rol_epi8(data, 4);
    data = _mm256_xor_si256(data,popcnt256_epi8(data));
    data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op24(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_add_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op25(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






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
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data,popcnt256_epi8(data));
    data = _mm256_add_epi8(data, data);
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op27(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_rol_epi8(data, 5);
    data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op28(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_add_epi8(data, data);
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op29(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op30(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op31(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op32(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op33(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_reverse_epi8(data);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op34(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op35(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_add_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op36(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op37(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_rolv_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op38(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






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
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op40(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op41(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op42(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_rol_epi8(data, 4);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op43(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;





    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op44(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






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
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;





    data = _mm256_rol_epi8(data, 2);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, popcnt256_epi8(data));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op46(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op47(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;





    data = _mm256_rol_epi8(data, 5);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op48(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op49(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_add_epi8(data, data);
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op50(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op51(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;





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
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_rolv_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, popcnt256_epi8(data));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op53(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






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
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;





    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op55(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






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
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op57(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_rolv_epi8(data, data);
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op58(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;





    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op59(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_rol_epi8(data, 1);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op60(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;





    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);

    #ifdef _WIN32
      data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    #else
      data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    #endif
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op61(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op62(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;





    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op63(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op64(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;





    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op65(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op66(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op67(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;






    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op68(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op69(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_reverse_epi8(data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op70(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op71(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op72(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op73(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op74(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_reverse_epi8(data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op75(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op76(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op77(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_rol_epi8(data, 3);
    data = _mm256_add_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, popcnt256_epi8(data));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op78(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_rolv_epi8(data, data);
    data = _mm256_reverse_epi8(data);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op79(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_add_epi8(data, data);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op80(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_rolv_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op81(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, popcnt256_epi8(data));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op82(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op83(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op84(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op85(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op86(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op87(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op88(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op89(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_add_epi8(data, data);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op90(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 6);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op91(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op92(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op93(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op94(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_rol_epi8(data, 1);
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op95(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rol_epi8(data, 2);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
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
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_rol_epi8(data, 1);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op98(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op99(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_reverse_epi8(data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op100(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_rolv_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, popcnt256_epi8(data));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op101(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op102(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_rol_epi8(data, 3);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op103(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_rol_epi8(data, 1);
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op104(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op105(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op106(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op107(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 6);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op108(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op109(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_mul_epi8(data, data);
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op110(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_add_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op111(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_mul_epi8(data, data);
    data = _mm256_reverse_epi8(data);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op112(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op113(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_rol_epi8(data, 6);
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op114(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_rol_epi8(data, 1);
    data = _mm256_reverse_epi8(data);
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op115(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op116(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op117(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op118(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op119(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op120(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op121(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op122(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op123(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rol_epi8(data, 6);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op124(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op125(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_add_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op126(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_rol_epi8(data, 1);
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op127(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
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
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op130(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op131(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_rol_epi8(data, 1);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op132(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op133(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op134(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op135(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;



    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_add_epi8(data, data);
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op136(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op137(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op138(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op139(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op140(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op141(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op142(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op143(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op144(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op145(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op146(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op147(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op148(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op149(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_reverse_epi8(data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op150(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op151(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op152(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op153(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 4);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op154(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op155(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op156(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 4);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op157(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op158(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op159(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op160(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 4);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op161(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op162(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_mul_epi8(data, data);
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op163(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op164(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_mul_epi8(data, data);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op165(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op166(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_add_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op167(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_mul_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op168(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op169(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op170(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_reverse_epi8(data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op171(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op172(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op173(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_add_epi8(data, data);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op174(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rolv_epi8(data, data);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op175(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op176(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op177(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op178(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rol_epi8(data, 1);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op179(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_add_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op180(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op181(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 5);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op182(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 6);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op183(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op184(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op185(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op186(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op187(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
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
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op190(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op191(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op192(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op193(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op194(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op195(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op196(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_reverse_epi8(data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op197(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op198(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op199(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_add_epi8(data, data);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op200(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_reverse_epi8(data);
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op201(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op202(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op203(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op204(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op205(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op206(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_reverse_epi8(data);
    data = _mm256_reverse_epi8(data);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op207(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
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
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_reverse_epi8(data);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op210(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op211(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_add_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op212(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    // data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    // data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op213(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op214(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op215(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op216(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op217(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op218(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op219(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op220(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op221(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op222(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op223(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op224(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 4);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op225(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op226(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_reverse_epi8(data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op227(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op228(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op229(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op230(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_mul_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op231(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op232(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_mul_epi8(data, data);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op233(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 1);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_rol_epi8(data, 3);
    pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op234(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op235(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op236(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op237(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op238(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op239(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 6);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op240(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_add_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op241(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op242(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_add_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op243(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op244(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op245(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op246(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op247(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op248(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
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
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op251(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op252(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op253(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
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

  void op254(workerData &worker) {
    RC4_set_key(&worker.key, 256, worker.prev_chunk);

    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void op255(workerData &worker) {
    RC4_set_key(&worker.key, 256, worker.prev_chunk);

    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op0(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    byte newVal = worker.simpleLookup[worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2 - worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

    if ((worker.pos2 - worker.pos1) % 2 == 1) {
      worker.t1 = worker.chunk[worker.pos1];
      worker.t2 = worker.chunk[worker.pos2];
      worker.chunk[worker.pos1] = reverse8(worker.t2);
      worker.chunk[worker.pos2] = reverse8(worker.t1);
      worker.isSame = false;
    }
  }

  void r_op1(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    byte newVal = worker.lookup3D[worker.branched_idx[worker.op] * 256 * 256 +
                                  worker.prev_chunk[worker.pos2] * 256 +
                                  worker.prev_chunk[worker.pos1]];

    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2 - worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

    return;
  }

  void r_op2(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    byte newVal = worker.simpleLookup[worker.reg_idx[worker.op] * 256 + worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2 - worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op3(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op4(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op5(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op6(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op7(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op8(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op9(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op10(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op11(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op12(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
}

  void r_op13(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op14(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op15(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];


    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op16(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);

    byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
    __m256i newVec = _mm256_set1_epi8(newVal);
    data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op17(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];


      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data,5);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op18(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op19(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_sub_epi8(data,_mm256_xor_si256(data,_mm256_set1_epi8(97)));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
    data = _mm256_add_epi8(data, data);;;

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op20(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];


      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op21(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];


      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);
    data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op22(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_mul_epi8(data,data);
    data = _mm256_rol_epi8(data,1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op23(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];


      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 4);
    data = _mm256_xor_si256(data,popcnt256_epi8(data));
    data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op24(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_add_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op25(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

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

  void r_op26(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data,popcnt256_epi8(data));
    data = _mm256_add_epi8(data, data);
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op27(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];


      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op28(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_add_epi8(data, data);
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op29(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];


      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op30(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];


      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op31(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op32(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op33(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_reverse_epi8(data);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op34(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op35(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];


      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_add_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op36(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op37(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op38(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

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

  void r_op39(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];


      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op40(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];


      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op41(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op42(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 4);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op43(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op44(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

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

  void r_op45(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 2);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, popcnt256_epi8(data));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op46(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op47(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data,vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op48(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op49(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_add_epi8(data, data);
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op50(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op51(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

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

  void r_op52(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, popcnt256_epi8(data));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op53(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

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

  void r_op54(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op55(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

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

  void r_op56(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op57(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op58(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op59(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op60(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);

    #ifdef _WIN32
      data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    #else
      data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    #endif
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op61(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op62(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op63(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op64(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op65(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op66(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op67(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op68(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op69(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {

      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_add_epi8(data, data);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_reverse_epi8(data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op70(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op71(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op72(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op73(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op74(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_reverse_epi8(data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op75(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op76(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op77(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_add_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, popcnt256_epi8(data));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op78(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_reverse_epi8(data);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op79(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_add_epi8(data, data);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op80(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op81(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, popcnt256_epi8(data));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op82(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op83(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op84(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op85(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op86(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op87(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op88(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op89(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_add_epi8(data, data);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op90(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 6);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op91(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op92(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op93(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op94(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op95(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rol_epi8(data, 2);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op96(workerData &worker) {
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

  void r_op97(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op98(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op99(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_reverse_epi8(data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op100(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, popcnt256_epi8(data));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op101(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op102(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op103(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op104(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op105(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op106(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op107(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 6);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op108(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op109(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_mul_epi8(data, data);
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op110(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_add_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op111(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_mul_epi8(data, data);
    data = _mm256_reverse_epi8(data);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op112(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op113(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 6);
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op114(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_reverse_epi8(data);
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op115(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op116(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, popcnt256_epi8(data));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op117(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op118(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op119(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op120(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op121(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op122(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op123(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rol_epi8(data, 6);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op124(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op125(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_add_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op126(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op127(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op128(workerData &worker) {
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

  void r_op129(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op130(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op131(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_rol_epi8(data, 1);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op132(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op133(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op134(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op135(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_add_epi8(data, data);
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op136(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op137(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op138(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op139(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op140(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op141(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op142(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op143(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op144(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op145(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op146(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op147(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op148(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op149(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_reverse_epi8(data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op150(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op151(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op152(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op153(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 4);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op154(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op155(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op156(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 4);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op157(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op158(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op159(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op160(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 4);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op161(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op162(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_mul_epi8(data, data);
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op163(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op164(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_mul_epi8(data, data);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op165(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op166(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_add_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op167(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_mul_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op168(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op169(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op170(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_reverse_epi8(data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op171(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op172(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op173(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_add_epi8(data, data);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op174(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rolv_epi8(data, data);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op175(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op176(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op177(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op178(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rol_epi8(data, 1);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op179(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_add_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op180(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op181(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 5);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op182(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 6);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op183(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op184(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op185(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op186(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op187(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op188(workerData &worker) {
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

  void r_op189(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op190(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op191(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op192(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op193(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op194(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op195(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op196(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_reverse_epi8(data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op197(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op198(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op199(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_add_epi8(data, data);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op200(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_reverse_epi8(data);
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op201(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op202(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op203(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op204(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op205(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op206(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_reverse_epi8(data);
    data = _mm256_reverse_epi8(data);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op207(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op208(workerData &worker) {
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

  void r_op209(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_reverse_epi8(data);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op210(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op211(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_add_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op212(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    // data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    // data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op213(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op214(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op215(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op216(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op217(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op218(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op219(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op220(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op221(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op222(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op223(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op224(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 4);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op225(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op226(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_reverse_epi8(data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op227(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op228(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op229(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op230(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_mul_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op231(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op232(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_mul_epi8(data, data);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op233(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 1);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_rol_epi8(data, 3);
    pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op234(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op235(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op236(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op237(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op238(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op239(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 6);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op240(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_add_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op241(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op242(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_add_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op243(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op244(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op245(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op246(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op247(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op248(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op249(workerData &worker) {
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

  void r_op250(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op251(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op252(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op253(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
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

    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  void r_op255(workerData &worker) {
    RC4_set_key(&worker.key, 256, worker.prev_chunk);

    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
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
    r_op0, r_op1, r_op2, r_op3, r_op4, r_op5, r_op6, r_op7, r_op8, r_op9, r_op10,
    r_op11, r_op12, r_op13, r_op14, r_op15, r_op16, r_op17, r_op18, r_op19, r_op20,
    r_op21, r_op22, r_op23, r_op24, r_op25, r_op26, r_op27, r_op28, r_op29, r_op30,
    r_op31, r_op32, r_op33, r_op34, r_op35, r_op36, r_op37, r_op38, r_op39, r_op40,
    r_op41, r_op42, r_op43, r_op44, r_op45, r_op46, r_op47, r_op48, r_op49, r_op50,
    r_op51, r_op52, r_op53, r_op54, r_op55, r_op56, r_op57, r_op58, r_op59, r_op60,
    r_op61, r_op62, r_op63, r_op64, r_op65, r_op66, r_op67, r_op68, r_op69, r_op70,
    r_op71, r_op72, r_op73, r_op74, r_op75, r_op76, r_op77, r_op78, r_op79, r_op80,
    r_op81, r_op82, r_op83, r_op84, r_op85, r_op86, r_op87, r_op88, r_op89, r_op90,
    r_op91, r_op92, r_op93, r_op94, r_op95, r_op96, r_op97, r_op98, r_op99, r_op100,
    r_op101, r_op102, r_op103, r_op104, r_op105, r_op106, r_op107, r_op108, r_op109, r_op110,
    r_op111, r_op112, r_op113, r_op114, r_op115, r_op116, r_op117, r_op118, r_op119, r_op120,
    r_op121, r_op122, r_op123, r_op124, r_op125, r_op126, r_op127, r_op128, r_op129, r_op130,
    r_op131, r_op132, r_op133, r_op134, r_op135, r_op136, r_op137, r_op138, r_op139, r_op140,
    r_op141, r_op142, r_op143, r_op144, r_op145, r_op146, r_op147, r_op148, r_op149, r_op150,
    r_op151, r_op152, r_op153, r_op154, r_op155, r_op156, r_op157, r_op158, r_op159, r_op160,
    r_op161, r_op162, r_op163, r_op164, r_op165, r_op166, r_op167, r_op168, r_op169, r_op170,
    r_op171, r_op172, r_op173, r_op174, r_op175, r_op176, r_op177, r_op178, r_op179, r_op180,
    r_op181, r_op182, r_op183, r_op184, r_op185, r_op186, r_op187, r_op188, r_op189, r_op190,
    r_op191, r_op192, r_op193, r_op194, r_op195, r_op196, r_op197, r_op198, r_op199, r_op200,
    r_op201, r_op202, r_op203, r_op204, r_op205, r_op206, r_op207, r_op208, r_op209, r_op210,
    r_op211, r_op212, r_op213, r_op214, r_op215, r_op216, r_op217, r_op218, r_op219, r_op220,
    r_op221, r_op222, r_op223, r_op224, r_op225, r_op226, r_op227, r_op228, r_op229, r_op230,
    r_op231, r_op232, r_op233, r_op234, r_op235, r_op236, r_op237, r_op238, r_op239, r_op240,
    r_op241, r_op242, r_op243, r_op244, r_op245, r_op246, r_op247, r_op248, r_op249, r_op250,
    r_op251, r_op252, r_op253, r_op254, r_op255,
  };
  int branchComputeSize = 14;
}