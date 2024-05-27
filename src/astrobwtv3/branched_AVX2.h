#include "astrobwtv3.h"

namespace astro_branched_zOp {


  inline void op0(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    if (worker.isSame) {
      byte newVal = worker.simpleLookup[worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(old, newVec, genMask(worker.pos2 - worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      if ((worker.pos2 - worker.pos1) % 2 == 1) {
        worker.t1 = worker.chunk[worker.pos1];
        worker.t2 = worker.chunk[worker.pos2];
        worker.chunk[worker.pos1] = reverse8(worker.t2);
        worker.chunk[worker.pos2] = reverse8(worker.t1);
        worker.isSame = false;
      }
      return;
    }

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

  inline void op1(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op] * 256 * 256 +
                                    worker.prev_chunk[worker.pos2] * 256 +
                                    worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(old, newVec, genMask(worker.pos2 - worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    __m256i shift = _mm256_and_si256(data, vec_3);
    data = _mm256_sllv_epi8(data, shift);
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.prev_chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);
    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2 - worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op2(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    if (worker.isSame) {
      byte oldVal = worker.prev_chunk[worker.pos1];

      if (oldVal == 0 || oldVal == 145 || oldVal == 174 || oldVal == 220) {
        _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        return;
      }

      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op] * 256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(old, newVec, genMask(worker.pos2 - worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

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

  inline void op3(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];


      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rolv_epi8(data,_mm256_add_epi8(data,vec_3));
    data = _mm256_xor_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data,1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op4(workerData &worker) {
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
    data = _mm256_srlv_epi8(data,_mm256_and_si256(data,vec_3));
    data = _mm256_rolv_epi8(data,data);
    data = _mm256_sub_epi8(data,_mm256_xor_si256(data,_mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op5(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;


    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data,pop);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
    data = _mm256_srlv_epi8(data,_mm256_and_si256(data,vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op6(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {

      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      if (worker.prev_chunk[worker.pos1] == 97) {
        _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        return;
      }
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    __m256i x = _mm256_xor_si256(data,_mm256_set1_epi8(97));
    data = _mm256_sub_epi8(data,x);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op7(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {

      if (worker.prev_chunk[worker.pos1] == 84) {
        _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        return;
      }

      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_add_epi8(data, data);;
    data = _mm256_rolv_epi8(data, data);

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data,pop);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op8(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {

      if (worker.prev_chunk[worker.pos1] == 170) {
        _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        return;
      }

      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rol_epi8(data,2);
    data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op9(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte oldVal = worker.prev_chunk[worker.pos1];
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);

      if (oldVal == newVal) {
        _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        return;
      }

      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data,4));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data,vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data,2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op10(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {

      byte oldVal = worker.prev_chunk[worker.pos1];
      if (oldVal == 89 || oldVal == 201) {
        _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
        return;
      }

      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op11(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];

      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_rol_epi8(data, 6);
    data = _mm256_and_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op12(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data,2));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data,2));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op13(workerData &worker) {
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
    data = _mm256_xor_si256(data,_mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_srlv_epi8(data,_mm256_and_si256(data,vec_3));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op14(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_srlv_epi8(data,_mm256_and_si256(data,vec_3));
    data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op15(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {
      byte newVal = worker.lookup3D[worker.branched_idx[worker.op]*256*256 + worker.prev_chunk[worker.pos2]*256 + worker.prev_chunk[worker.pos1]];


      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data,2));
    data = _mm256_sllv_epi8(data,_mm256_and_si256(data,vec_3));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sub_epi8(data,_mm256_xor_si256(data,_mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op16(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;




    if (worker.isSame) {


      byte newVal = worker.simpleLookup[worker.reg_idx[worker.op]*256 + worker.prev_chunk[worker.pos1]];
      __m256i newVec = _mm256_set1_epi8(newVal);
      data = _mm256_blendv_epi8(data, newVec, genMask(worker.pos2-worker.pos1));
      _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);

      return;
    }

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data,4));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data,1);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op17(workerData &worker) {
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

  inline void op18(workerData &worker) {
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

  inline void op19(workerData &worker) {
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

  inline void op20(workerData &worker) {
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

  inline void op21(workerData &worker) {
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

  inline void op22(workerData &worker) {
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

  inline void op23(workerData &worker) {
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

  inline void op24(workerData &worker) {
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

  inline void op25(workerData &worker) {
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

  inline void op26(workerData &worker) {
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

  inline void op27(workerData &worker) {
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

  inline void op28(workerData &worker) {
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

  inline void op29(workerData &worker) {
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

inline void op30(workerData &worker) {
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

  inline void op31(workerData &worker) {
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

  inline void op32(workerData &worker) {
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

  inline void op33(workerData &worker) {
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

  inline void op34(workerData &worker) {
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

  inline void op35(workerData &worker) {
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

  inline void op36(workerData &worker) {
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

  inline void op37(workerData &worker) {
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

  inline void op38(workerData &worker) {
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

  inline void op39(workerData &worker) {
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

  inline void op40(workerData &worker) {
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

  inline void op41(workerData &worker) {
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

  inline void op42(workerData &worker) {
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

  inline void op43(workerData &worker) {
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

  inline void op44(workerData &worker) {
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

  inline void op45(workerData &worker) {
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

  inline void op46(workerData &worker) {
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

  inline void op47(workerData &worker) {
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

    inline void op48(workerData &worker) {
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

  inline void op49(workerData &worker) {
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

  inline void op50(workerData &worker) {
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

  inline void op51(workerData &worker) {
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

  inline void op52(workerData &worker) {
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

  inline void op53(workerData &worker) {
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

  inline void op54(workerData &worker) {
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

  inline void op55(workerData &worker) {
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

  inline void op56(workerData &worker) {
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

  inline void op57(workerData &worker) {
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

  inline void op58(workerData &worker) {
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

  inline void op59(workerData &worker) {
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

  inline void op60(workerData &worker) {
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

  inline void op61(workerData &worker) {
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

  inline void op62(workerData &worker) {
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

  inline void op63(workerData &worker) {
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

  inline void op64(workerData &worker) {
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

  inline void op65(workerData &worker) {
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

  inline void op66(workerData &worker) {
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

  inline void op67(workerData &worker) {
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

  inline void op68(workerData &worker) {
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

  inline void op69(workerData &worker) {
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

    inline void op70(workerData &worker) {
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

  inline void op71(workerData &worker) {
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

  inline void op72(workerData &worker) {
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

  inline void op73(workerData &worker) {
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

  inline void op74(workerData &worker) {
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

  inline void op75(workerData &worker) {
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

  inline void op76(workerData &worker) {
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

  inline void op77(workerData &worker) {
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

  inline void op78(workerData &worker) {
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

  inline void op79(workerData &worker) {
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

  inline void op80(workerData &worker) {
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

    inline void op81(workerData &worker) {
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

  inline void op82(workerData &worker) {
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

  inline void op83(workerData &worker) {
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

  inline void op84(workerData &worker) {
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

  inline void op85(workerData &worker) {
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

  inline void op86(workerData &worker) {
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

  inline void op87(workerData &worker) {
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

  inline void op88(workerData &worker) {
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

  inline void op89(workerData &worker) {
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

  inline void op90(workerData &worker) {
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

  inline void op91(workerData &worker) {
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

  inline void op92(workerData &worker) {
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

  inline void op93(workerData &worker) {
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

  inline void op94(workerData &worker) {
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

  inline void op95(workerData &worker) {
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

  inline void op96(workerData &worker) {
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

  inline void op97(workerData &worker) {
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

  inline void op98(workerData &worker) {
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

  inline void op99(workerData &worker) {
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

  inline void op100(workerData &worker) {
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

  inline void op101(workerData &worker) {
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

  inline void op102(workerData &worker) {
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

  inline void op103(workerData &worker) {
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

  inline void op104(workerData &worker) {
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

  inline void op105(workerData &worker) {
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

  inline void op106(workerData &worker) {
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

  inline void op107(workerData &worker) {
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

  inline void op108(workerData &worker) {
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

  inline void op109(workerData &worker) {
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

  inline void op110(workerData &worker) {
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

  inline void op111(workerData &worker) {
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

  inline void op112(workerData &worker) {
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

  inline void op113(workerData &worker) {
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

  inline void op114(workerData &worker) {
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

  inline void op115(workerData &worker) {
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

  inline void op116(workerData &worker) {
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

  inline void op117(workerData &worker) {
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

  inline void op118(workerData &worker) {
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

  inline void op119(workerData &worker) {
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

  inline void op120(workerData &worker) {
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

  inline void op121(workerData &worker) {
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

  inline void op122(workerData &worker) {
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

  inline void op123(workerData &worker) {
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

  inline void op124(workerData &worker) {
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

  inline void op125(workerData &worker) {
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

  inline void op126(workerData &worker) {
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

  inline void op127(workerData &worker) {
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

  inline void op128(workerData &worker) {
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

  inline void op129(workerData &worker) {
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

  inline void op130(workerData &worker) {
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

  inline void op131(workerData &worker) {
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

  inline void op132(workerData &worker) {
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

  inline void op133(workerData &worker) {
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

  inline void op134(workerData &worker) {
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

  inline void op135(workerData &worker) {
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

  inline void op136(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op137(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op138(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op139(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op140(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op141(workerData &worker) {
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

  inline void op142(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op143(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op144(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op145(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op146(workerData &worker) {
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

  inline void op147(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op148(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op149(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_reverse_epi8(data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op150(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op151(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op152(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op153(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 4);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op154(workerData &worker) {
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

  inline void op155(workerData &worker) {
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

  inline void op156(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 4);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op157(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op158(workerData &worker) {
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

  inline void op159(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op160(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 4);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op161(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op162(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_mul_epi8(data, data);
    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op163(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op164(workerData &worker) {
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

  inline void op165(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op166(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_add_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op167(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_mul_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op168(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op169(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op170(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_reverse_epi8(data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op171(workerData &worker) {
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

  inline void op172(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op173(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_add_epi8(data, data);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op174(workerData &worker) {
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

  inline void op175(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op176(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 5);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op177(workerData &worker) {
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

  inline void op178(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rol_epi8(data, 1);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op179(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_add_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op180(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op181(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 5);

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op182(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 6);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

  data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op183(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op184(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op185(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op186(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op187(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op188(workerData &worker) {
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

  inline void op189(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op190(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op191(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op192(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op193(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op194(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op195(workerData &worker) {
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

  inline void op196(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_reverse_epi8(data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op197(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op198(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 1);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op199(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_add_epi8(data, data);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op200(workerData &worker) {
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

  inline void op201(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op202(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op203(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op204(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op205(workerData &worker) {
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

  inline void op206(workerData &worker) {
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

  inline void op207(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    __m256i pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);
    pop = popcnt256_epi8(data);
    data = _mm256_xor_si256(data, pop);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op208(workerData &worker) {
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

  inline void op209(workerData &worker) {
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

  inline void op210(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op211(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_add_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op212(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    // data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    // data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op213(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op214(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op215(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op216(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rolv_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op217(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op218(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op219(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op220(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 1);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op221(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op222(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_mul_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op223(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op224(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 4);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op225(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op226(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_reverse_epi8(data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op227(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op228(workerData &worker) {
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

  inline void op229(workerData &worker) {
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

  inline void op230(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_mul_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_rolv_epi8(data, data);
    data = _mm256_rolv_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op231(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 3);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_reverse_epi8(data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op232(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_mul_epi8(data, data);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op233(workerData &worker) {
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

  inline void op234(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op235(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_mul_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op236(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_add_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op237(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 3);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op238(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 3);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op239(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 6);
    data = _mm256_mul_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op240(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_add_epi8(data, data);
    data = _mm256_and_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op241(workerData &worker) {
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

  inline void op242(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_add_epi8(data, data);
    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_xor_si256(data, _mm256_set1_epi8(worker.chunk[worker.pos2]));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op243(workerData &worker) {
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

  inline void op244(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_reverse_epi8(data);
    data = _mm256_rol_epi8(data, 5);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op245(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_sub_epi8(data, _mm256_xor_si256(data, _mm256_set1_epi8(97)));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op246(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_add_epi8(data, data);
    data = _mm256_rol_epi8(data, 1);
    data = _mm256_srlv_epi8(data, _mm256_and_si256(data, vec_3));
    data = _mm256_add_epi8(data, data);

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op247(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_rol_epi8(data, 5);
    data = _mm256_xor_si256(data, _mm256_set1_epi64x(-1LL));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op248(workerData &worker) {
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

  inline void op249(workerData &worker) {
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

  inline void op250(workerData &worker) {
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

  inline void op251(workerData &worker) {
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

  inline void op252(workerData &worker) {
    __m256i data = _mm256_loadu_si256((__m256i*)&worker.prev_chunk[worker.pos1]);
    __m256i old = data;

    data = _mm256_reverse_epi8(data);
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 4));
    data = _mm256_xor_si256(data, _mm256_rol_epi8(data, 2));
    data = _mm256_sllv_epi8(data, _mm256_and_si256(data, vec_3));

    data = _mm256_blendv_epi8(old, data, genMask(worker.pos2-worker.pos1));
    _mm256_storeu_si256((__m256i*)&worker.chunk[worker.pos1], data);
  }

  inline void op253(workerData &worker) {
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

  inline void op254(workerData &worker) {
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

  inline void op255(workerData &worker) {
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

  alignas(32) OpFunc branchCompute[256] = {
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
  };
  int branchComputeSize = 14;
}