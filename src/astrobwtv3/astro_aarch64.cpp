
#include "astro_aarch64.hpp"

using byte = unsigned char;
bool debugOpOrderAA = false;

inline uint8x16_t binary_not(uint8x16_t data) {
  //worker.chunk[i] = ~worker.chunk[i];
  // also maybe
  //const uint8x16_t ones = vdupq_n_u8(0xFF);
  // return vbicq_u8(data, ones);
  return vmvnq_u8(data);
}

inline uint8x16_t rotate_bits(uint8x16_t data, int rotation) {
  //worker.chunk[i] = std::rotl(worker.chunk[i], 3);
  //worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));
  rotation %= 8;
  // TODO: Find out how we can make clang tell us the different between ARMv8.2a (which compiles here) and ARMv8-a (which does not)
  //return vorrq_u8(vshlq_n_u8(data, rotation), vshrq_n_u8(data, 8 - rotation));
  auto rotation_amounts = vdupq_n_u8(rotation);
  return vorrq_u8(vshlq_u8(data, rotation_amounts), vshlq_u8(data, vsubq_u8(rotation_amounts, vdupq_n_u8(8))));
}

inline uint8x16_t rotate_and_xor(uint8x16_t left_side, int rotation) {
  //worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));
  //rotation = rotation % 8;
  //rotation %= 8;
  //uint8x16_t rotated = vorrq_u8(vshlq_n_u8(left_side, rotation), vshrq_n_u8(left_side, 8 - rotation));

  // Perform XOR with original data
  return veorq_u8(left_side, rotate_bits(left_side, rotation));
}


inline uint8x16_t add_with_self(uint8x16_t a) {
  //worker.chunk[i] += worker.chunk[i];
  return vaddq_u8(a, a);
}

inline uint8x16_t mul_with_self(uint8x16_t a) {
  //worker.chunk[i] *= worker.chunk[i];
  return vmulq_u8(a, a);
}

inline uint8x16_t and_vectors(uint8x16_t a, uint8x16_t b) {
  //worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2];
  // Perform XOR with original data
  return vandq_u8(a, b);
}

inline uint8x16_t xor_vectors(uint8x16_t a, uint8x16_t b) {
  //worker.chunk[i] ^= worker.chunk[worker.pos2];
  // Perform XOR with original data
  return veorq_u8(a, b);
}

inline uint8x16_t xor_with_bittable(uint8x16_t a) {
  //worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];
  //auto count = vcntq_u8(a);
  // Perform XOR with original data
  return veorq_u8(a, vcntq_u8(a));
}

inline uint8x16_t reverse_vector(uint8x16_t data) {
    return vrbitq_u8(data);
}

/*
uint8x16_t shift_left_by_int_with_and(uint8x16_t data, int andint) {
  //worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);
  // Note: This is signed!
  int8x16_t anded = vandq_s8(data, vdupq_n_u8(andint));
  return vshlq_u8(data, anded);
}
*/

inline uint8x16_t shift_left_by_int_with_and(uint8x16_t data, int andint) {
  //worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);
  // Note: This is signed!
  //int8x16_t anded = vandq_s8(data, vdupq_n_u8(andint));
  return vshlq_u8(data, vandq_s8(data, vdupq_n_u8(andint)));
}

/*
uint8x16_t shift_right_by_int_with_and(uint8x16_t data, int andint) {
  //worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);
  // Note: This is signed!
  int8x16_t anded = vandq_s8(data, vdupq_n_u8(andint));

  // We can negate and left-shift to effectively do a right-shift;
  int8x16_t negated = vqnegq_s8(anded);
  return vshlq_u8(data, negated);
}
*/

inline uint8x16_t shift_right_by_int_with_and(uint8x16_t data, int andint) {
  //worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);
  return vshlq_u8(data, vqnegq_s8(vandq_s8(data, vdupq_n_u8(andint))));
}

inline uint8x16_t subtract_xored(uint8x16_t data, int xor_value) {
  //worker.chunk[i] -= (worker.chunk[i] ^ 97);
  //auto xored = veorq_u8(data, vdupq_n_u8(xor_value));
  return vsubq_u8(data, veorq_u8(data, vdupq_n_u8(xor_value)));
}

inline uint8x16_t rotate_by_self(uint8x16_t data) {

  // see rotate_by_self
  //(worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8)));
  // Shift left by the remainder of each element divided by 8
  uint8x16_t rotation_amounts = vandq_u8(data, vdupq_n_u8(7));

  //for(int x = 0; x < 16; x++) {
  //  printf("mod: %02x\n", rotation_amounts[x]);
  //}

  //uint8x16_t shifted_left = vshlq_u8(data, rotation_amounts);


  //uint8x16_t right_shift_amounts = vsubq_u8(vandq_u8(data, vdupq_n_u8(7)), vdupq_n_u8(8));
  //uint8x16_t right_shift_amounts = vsubq_u8(rotation_amounts, vdupq_n_u8(8));

  // Perform the right shift using left shift with negative amounts
  //return vshlq_u8(data, right_shift_amounts);
  // Shift right by (8 - remainder) of each element


  // Combine the shifted results using bitwise OR
  //return vorrq_u8(shifted_left, vshlq_u8(data, right_shift_amounts));
  return vorrq_u8(vshlq_u8(data, rotation_amounts), vshlq_u8(data, vsubq_u8(rotation_amounts, vdupq_n_u8(8))));

  //worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8)));
  //worker.chunk[i] = std::rotl(worker.chunk[i], worker.chunk[i]);
  //return rotate_bits_by_vector(data);
}

void branchComputeCPU_aarch64(workerData &worker, bool isTest)
{
  //if (debugOpOrderAA) printf("cpu\n");
  
  while (true)
  {
    if(isTest) {

    } else {
      worker.tries++;
      if (debugOpOrderAA) printf("t: 0x%lx p: 0x%lx l: 0x%lx\n", worker.tries, worker.prev_lhash, worker.lhash);
      worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries;
      // __builtin_prefetch(&worker.random_switcher,0,3);
      // printf("%d worker.random_switcher %d %08jx\n", worker.tries, worker.random_switcher, worker.random_switcher);

      worker.op = static_cast<byte>(worker.random_switcher);
      //if (debugOpOrderAA) worker.opsA.push_back(worker.op);

      // printf("op: %d\n", worker.op);

      worker.pos1 = static_cast<byte>(worker.random_switcher >> 8);
      worker.pos2 = static_cast<byte>(worker.random_switcher >> 16);

      if (worker.pos1 > worker.pos2)
      {
        std::swap(worker.pos1, worker.pos2);
      }

      if (worker.pos2 - worker.pos1 > 32)
      {
        worker.pos2 = worker.pos1 + ((worker.pos2 - worker.pos1) & 0x1f);
      }

      worker.chunk = &worker.sData[(worker.tries - 1) * 256];
      if (debugOpOrderAA) printf("worker.op: %03d p1: %03d p2: %03d\n", worker.op, worker.pos1, worker.pos2);

      if (worker.tries == 1) {
        worker.prev_chunk = worker.chunk;
      } else {
        worker.prev_chunk = &worker.sData[(worker.tries - 2) * 256];
        /*
        if (debugOpOrderAA) {
          printf("tries: %03lu prev_chunk[0->%03d]: ", worker.tries, worker.pos2);
          for (int x = 0; x <= worker.pos2+16 && worker.pos2+16 < 256; x++) {
            printf("%02x", worker.prev_chunk[x]);
          }
          printf("\n");
        }

        __builtin_prefetch(worker.prev_chunk,0,3);
        __builtin_prefetch(worker.prev_chunk+64,0,3);
        __builtin_prefetch(worker.prev_chunk+128,0,3);
        __builtin_prefetch(worker.prev_chunk+192,0,3);

        // Calculate the start and end blocks
        int start_block = 0;
        int end_block = worker.pos1 / 16;
        if (debugOpOrderAA) printf("loopa: %03lu %03d < %03d\n", worker.tries, start_block, end_block);

        // Copy the blocks before worker.pos1
        for (int i = start_block; i < end_block; i++) {
            __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
            _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
        }

        if (debugOpOrderAA) printf("loopb: %03lu %03d < %03d\n", worker.tries, end_block * 16, worker.pos1);
        // Copy the remaining bytes before worker.pos1
        for (int i = end_block * 16; i < worker.pos1; i++) {
            worker.chunk[i] = worker.prev_chunk[i];
        }

        // Calculate the start and end blocks
        start_block = (worker.pos2 + 15) / 16;
        end_block = 16;
        if (debugOpOrderAA) printf("loopc: %03lu %03d < %03d\n", worker.tries, start_block, end_block);

        // Copy the blocks after worker.pos2
        for (int i = start_block; i < end_block; i++) {
            __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
            _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
        }

        if (debugOpOrderAA) printf("loopd: %03lu %03d < %03d\n", worker.tries, worker.pos2, start_block * 16);
        // Copy the remaining bytes after worker.pos2
        for (int i = worker.pos2; i < start_block * 16; i++) {
          worker.chunk[i] = worker.prev_chunk[i];
        }
        */
      }

      if (debugOpOrderAA) {
        printf("tries: %03lu chunk_before[  0->%03d]: ", worker.tries, worker.pos2);
        for (int x = 0; x <= worker.pos2+16 && worker.pos2+16 < 256; x++) {
          printf("%02x", worker.chunk[x]);
        }
        printf("\n");
      }

      //for (int x = worker.pos1; x < worker.pos2; x++) {
      //  worker.chunk[x] = worker.prev_chunk[x];
      //}
      
      //for (int x = 0; x < 256; x++) {
      //  worker.chunk[x] = worker.prev_chunk[x];
      //}
      memcpy(worker.chunk, worker.prev_chunk, 256);
      if (debugOpOrderAA) {
        printf("tries: %03lu  chunk_fixed[  0->%03d]: ", worker.tries, worker.pos2);
        for (int x = 0; x <= worker.pos2+16 && worker.pos2+16 < 256; x++) {
          //printf("%d \n", x);
          printf("%02x", worker.chunk[x]);
        }
        printf("\n");
      }
    }
    
    //printf("tries: %03d step_3[0->%-3d]: ", worker.tries, worker.pos2);
    //for (int x = 0; x < worker.pos2; x++) {
    //  printf("%02x", worker.step_3[x]);
    //}
    //printf("\n");

    //printf("%02d ", worker.op);
    //if(worker.tries > 100) {
    //  break;
    //}

    memcpy(worker.aarchFixup, &worker.chunk[worker.pos2], 16);
    switch (worker.op)
    {
    case 0:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));                // rotate  bits by 5
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random

        // INSERT_RANDOM_CODE_END
        worker.t1 = worker.chunk[worker.pos1];
        worker.t2 = worker.chunk[worker.pos2];
        worker.chunk[worker.pos1] = reverse8(worker.t2);
        worker.chunk[worker.pos2] = reverse8(worker.t1);
      }
      break;
    case 1:
      {
        worker.opt[worker.op] = true;
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
          data = shift_left_by_int_with_and(data, 3);
          //worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));
          data = rotate_bits(data, 1);
          //worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2];
          data = and_vectors(data, p2vec);
          //worker.chunk[i] += worker.chunk[i];
          data = add_with_self(data);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 2:
    {
        worker.opt[worker.op] = true;
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);
          //worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
          data = xor_with_bittable(data);
                      
          //worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
          data = reverse_vector(data);
          
          //worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
          data = shift_left_by_int_with_and(data, 3);
          
          //worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
          data = xor_with_bittable(data);
          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
    }
      break;
    case 3:
    {
        worker.opt[worker.op] = true;
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = std::rotl(worker.chunk[i], worker.chunk[i]); // rotate  bits by random
          data = rotate_by_self(data);

          //worker.chunk[i] = std::rotl(worker.chunk[i], 3);                // rotate  bits by 3
          data = rotate_bits(data, 3);

          //worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
          data = xor_vectors(data, p2vec);

          //worker.chunk[i] = std::rotl(worker.chunk[i], 1);                // rotate  bits by 1
          data = rotate_bits(data, 1);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
    }
      break;
    case 4:
    {
        worker.opt[worker.op] = true;
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);
          //worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
          data = binary_not(data);
          
          //worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
          data = shift_right_by_int_with_and(data, 3);

          //worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
          data = rotate_by_self(data);

          //worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
          data = subtract_xored(data, 97);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
    }
      break;
    case 5:
      {
        worker.opt[worker.op] = true;
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];
          data = xor_with_bittable(data);
          //worker.chunk[i] ^= worker.chunk[worker.pos2];
          data = xor_vectors(data, p2vec);
          //worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);
          data = shift_left_by_int_with_and(data, 3);
          //worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);
          data = shift_right_by_int_with_and(data, 3);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
    break;
    case 6:
      {
        worker.opt[worker.op] = true;
        //uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);
          data = shift_left_by_int_with_and(data, 3);
          //worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));
          data = rotate_bits(data, 3);
          //worker.chunk[i] = ~worker.chunk[i];
          data = binary_not(data);
          //worker.chunk[i] -= (worker.chunk[i] ^ 97);
          data = subtract_xored(data, 97);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 7:
      {
        worker.opt[worker.op] = true;
        memcpy(worker.aarchFixup, &worker.chunk[worker.pos2], 16);
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);

        for (int i = worker.pos1; i < worker.pos2; i += 16) {
            // Load 16 bytes (128 bits) of data from chunk
            uint8x16_t data = vld1q_u8(&worker.chunk[i]);

            //worker.chunk[i] += worker.chunk[i];
            data = add_with_self(data);

            //worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8)));
            data = rotate_by_self(data);


            //worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];
            data = xor_with_bittable(data);

            //worker.chunk[i] = ~worker.chunk[i];
            data = binary_not(data);
            vst1q_u8(&worker.chunk[i], data);

            //data = vmulq_u8(data, data);
            //vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 8:
      {
        worker.opt[worker.op] = true;
        memcpy(worker.aarchFixup, &worker.chunk[worker.pos2], 16);
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);

        for (int i = worker.pos1; i < worker.pos2; i += 16) {
            // Load 16 bytes (128 bits) of data from chunk
            uint8x16_t data = vld1q_u8(&worker.chunk[i]);

            //worker.chunk[i] = ~worker.chunk[i];
            data = binary_not(data);
        
            //worker.chunk[i] = (worker.chunk[i] << 2) | (worker.chunk[i] >> 6);
            data = rotate_bits(data, 2);

            //worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
            data = shift_left_by_int_with_and(data, 3);

            vst1q_u8(&worker.chunk[i], data);

            //data = vmulq_u8(data, data);
            //vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 9:
      {
        worker.opt[worker.op] = true;
        memcpy(worker.aarchFixup, &worker.chunk[worker.pos2], 16);
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);

        for (int i = worker.pos1; i < worker.pos2; i += 16) {
            // Load 16 bytes (128 bits) of data from chunk
            uint8x16_t data = vld1q_u8(&worker.chunk[i]);

            data = xor_vectors(data, p2vec);
            //vst1q_u8(&worker.chunk[i], data);

            data = rotate_and_xor(data, 4);
            //vst1q_u8(&worker.chunk[i], data);

            data = shift_right_by_int_with_and(data, 3);
            /*
            // store
            vst1q_u8(&worker.chunk[i], data);
            for(int x = i; x < i+16; x++) {
              worker.chunk[x] = worker.chunk[x] >> (worker.chunk[x] & 3); // shift right
            }
            // load 
            data = vld1q_u8(&worker.chunk[i]);
            */

            data = rotate_and_xor(data, 2);
            vst1q_u8(&worker.chunk[i], data);

            //data = vmulq_u8(data, data);
            //vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 10:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = ~worker.chunk[i];
          data = binary_not(data);
          //worker.chunk[i] *= worker.chunk[i];
          data = mul_with_self(data);
          //worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));
          data = rotate_bits(data, 3);
          //worker.chunk[i] *= worker.chunk[i];
          data = mul_with_self(data);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 11:
      {
        worker.opt[worker.op] = true;
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = (worker.chunk[i] << 6) | (worker.chunk[i] >> (8 - 6));
          data = rotate_bits(data, 6);
          // worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));
          //worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2];
          data = and_vectors(data, p2vec);
          //worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8)));
          data = rotate_by_self(data);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 12:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));
          data = rotate_and_xor(data, 2);
          //worker.chunk[i] *= worker.chunk[i];
          data = mul_with_self(data);
          //worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));
          data = rotate_and_xor(data, 2);
          //worker.chunk[i] = ~worker.chunk[i];
          data = binary_not(data);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 13:
      {
        worker.opt[worker.op] = true;
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));
          data = rotate_bits(data, 1);
          //worker.chunk[i] ^= worker.chunk[worker.pos2];
          data = xor_vectors(data, p2vec);
          //worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);
          data = shift_right_by_int_with_and(data, 3);
          //worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));
          data = rotate_bits(data, 5);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 14:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);
          data = shift_right_by_int_with_and(data, 3);
          //worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);
          data = shift_left_by_int_with_and(data, 3);
          //worker.chunk[i] *= worker.chunk[i];
          data = mul_with_self(data);
          //worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);
          data = shift_left_by_int_with_and(data, 3);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 15:
      {
        worker.opt[worker.op] = true;
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));               // rotate  bits by 2
          data = rotate_and_xor(data, 2);
          //worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);
          data = shift_left_by_int_with_and(data, 3);
          //worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2];
          data = and_vectors(data, p2vec);
          //worker.chunk[i] -= (worker.chunk[i] ^ 97);
          data = subtract_xored(data, 97);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 16:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] *= worker.chunk[i];
          data = mul_with_self(data);
          //worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));
          data = rotate_bits(data, 1);
          //worker.chunk[i] = ~worker.chunk[i];
          data = binary_not(data);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 17:
      {
        worker.opt[worker.op] = true;
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] ^= worker.chunk[worker.pos2];
          data = xor_vectors(data, p2vec);
          //worker.chunk[i] *= worker.chunk[i];
          data = mul_with_self(data);
          //worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));
          data = rotate_bits(data, 5);
          //worker.chunk[i] = ~worker.chunk[i];
          data = binary_not(data);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 18:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> 7);
          data = rotate_bits(data, 1);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 19:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] -= (worker.chunk[i] ^ 97);
          data = subtract_xored(data, 97);
          //worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));
          data = rotate_bits(data, 5);
          //worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);
          data = shift_left_by_int_with_and(data, 3);
          //worker.chunk[i] += worker.chunk[i];     
          data = add_with_self(data);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 20:
      {
        worker.opt[worker.op] = true;
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);
          //worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2];
          data = and_vectors(data, p2vec);

          //worker.chunk[i] ^= worker.chunk[worker.pos2];
          data = xor_vectors(data, p2vec);

          //worker.chunk[i] = reverse8(worker.chunk[i]);
          data = reverse_vector(data);

          //worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));
          data = rotate_and_xor(data, 2);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 21:
      {
        worker.opt[worker.op] = true;
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));
          data = rotate_bits(data, 1);

          //worker.chunk[i] ^= worker.chunk[worker.pos2];
          data = xor_vectors(data, p2vec);

          //worker.chunk[i] += worker.chunk[i];
          data = add_with_self(data);

          //worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2];
          data = and_vectors(data, p2vec);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 22:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);
          data = shift_left_by_int_with_and(data, 3);
          //worker.chunk[i] = reverse8(worker.chunk[i]);
          data = reverse_vector(data);
          //worker.chunk[i] *= worker.chunk[i];
          data = mul_with_self(data);
          //worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));   
          data = rotate_bits(data, 1);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 23:
      {
        worker.opt[worker.op] = true;
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_bits(data, 4);
          // worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));
          //worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];
          data = xor_with_bittable(data);
          //worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2];
          data = and_vectors(data, p2vec);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 24:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] += worker.chunk[i];
          data = add_with_self(data);
          //worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);
          data = shift_right_by_int_with_and(data, 3);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5)); 
          data = rotate_bits(data, 5);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 25:
      worker.opt[worker.op] = true;
      for (int i = worker.pos1; i < worker.pos2; i += 16) {
          // Load 16 bytes (128 bits) of data from chunk
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];
          data = xor_with_bittable(data);

          //worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));
          data = rotate_bits(data, 3);

          //worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8)));
          data = rotate_by_self(data);

          //worker.chunk[i] -= (worker.chunk[i] ^ 97);
          data = subtract_xored(data, 97);

          vst1q_u8(&worker.chunk[i], data);
      }
      memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      break;
    case 26:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] *= worker.chunk[i];
          data = mul_with_self(data);
          //worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];
          data = xor_with_bittable(data);
          //worker.chunk[i] += worker.chunk[i];
          data = add_with_self(data);
          //worker.chunk[i] = reverse8(worker.chunk[i]);
          data = reverse_vector(data);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 27:
      {
        worker.opt[worker.op] = true;
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));
          data = rotate_bits(data, 5);
          //worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2];
          data = and_vectors(data, p2vec);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));
          data = rotate_bits(data, 5);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 28:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);
          data = shift_left_by_int_with_and(data, 3);
          //worker.chunk[i] += worker.chunk[i];
          data = add_with_self(data);
          //worker.chunk[i] += worker.chunk[i];
          data = add_with_self(data);
          //worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));
          data = rotate_bits(data, 5);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 29:
      {
        worker.opt[worker.op] = true;
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] *= worker.chunk[i];
          data = mul_with_self(data);
          //worker.chunk[i] ^= worker.chunk[worker.pos2];
          data = xor_vectors(data, p2vec);
          //worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);
          data = shift_right_by_int_with_and(data, 3);
          //worker.chunk[i] += worker.chunk[i];  
          data = add_with_self(data);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 30:
      {
        worker.opt[worker.op] = true;
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2];
          data = and_vectors(data, p2vec);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));
          data = rotate_bits(data, 5);
          //worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);
          data = shift_left_by_int_with_and(data, 3);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 31:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = ~worker.chunk[i];
          data = binary_not(data);
          //worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));
          data = rotate_and_xor(data, 2);
          //worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);
          data = shift_left_by_int_with_and(data, 3);
          //worker.chunk[i] *= worker.chunk[i];   
          data = mul_with_self(data);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 32:
      {
        worker.opt[worker.op] = true;
        for (int i = worker.pos1; i < worker.pos2; i += 16) {
            uint8x16_t data = vld1q_u8(&worker.chunk[i]);
            data = rotate_and_xor(data, 2);
            data = reverse_vector(data);
            data = rotate_bits(data, 3);
            data = rotate_and_xor(data, 2);
            vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 33:
      worker.opt[worker.op] = true;
      memcpy(worker.aarchFixup, &worker.chunk[worker.pos2], 16);
      for (int i = worker.pos1; i < worker.pos2; i += 16) {
          // Load 16 bytes (128 bits) of data from chunk
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          data = rotate_by_self(data);
          
          //vst1q_u8(&worker.chunk[i], data);

          data = rotate_and_xor(data, 4);
          //vst1q_u8(&worker.chunk[i], data);

          data = reverse_vector(data);
          //vst1q_u8(&worker.chunk[i], data);

          data = vmulq_u8(data, data);
          vst1q_u8(&worker.chunk[i], data);
      }
      memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      break;
    case 34:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] -= (worker.chunk[i] ^ 97);
          data = subtract_xored(data, 97);
          //worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);
          data = shift_left_by_int_with_and(data, 3);
          //worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);
          data = shift_left_by_int_with_and(data, 3);
          //worker.chunk[i] -= (worker.chunk[i] ^ 97); 
          data = subtract_xored(data, 97);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 35:
      {
        worker.opt[worker.op] = true;
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] += worker.chunk[i];
          data = add_with_self(data);
          //worker.chunk[i] = ~worker.chunk[i];
          data = binary_not(data);
          //worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));
          data = rotate_bits(data, 1);
          //worker.chunk[i] ^= worker.chunk[worker.pos2];
          data = xor_vectors(data, p2vec);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 36:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];
          data = xor_with_bittable(data);
          //worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));
          data = rotate_bits(data, 1);
          //worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));
          data = rotate_and_xor(data, 2);
          //worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1)); 
          data = rotate_bits(data, 1);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 37:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8)));
          data = rotate_by_self(data);
          //worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);
          data = shift_right_by_int_with_and(data, 3);
          //worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);
          data = shift_right_by_int_with_and(data, 3);
          //worker.chunk[i] *= worker.chunk[i];        
          data = mul_with_self(data);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 38:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);
          data = shift_right_by_int_with_and(data, 3);
          //worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));
          data = rotate_bits(data, 3);
          //worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];
          data = xor_with_bittable(data);
          //worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8)));
          data = rotate_by_self(data);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 39:
      {
        worker.opt[worker.op] = true;
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));
          data = rotate_and_xor(data, 2);
          //worker.chunk[i] ^= worker.chunk[worker.pos2];
          data = xor_vectors(data, p2vec);
          //worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);
          data = shift_right_by_int_with_and(data, 3);
          //worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2];
          data = and_vectors(data, p2vec);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 40:
      {
        worker.opt[worker.op] = true;
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8)));
          data = rotate_by_self(data);
          //worker.chunk[i] ^= worker.chunk[worker.pos2];
          data = xor_vectors(data, p2vec);
          //worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];
          data = xor_with_bittable(data);
          //worker.chunk[i] ^= worker.chunk[worker.pos2];
          data = xor_vectors(data, p2vec);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 41:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));
          data = rotate_bits(data, 5);
          //worker.chunk[i] -= (worker.chunk[i] ^ 97);
          data = subtract_xored(data, 97);
          //worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));
          data = rotate_bits(data, 3);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 42:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4)); // rotate  bits by 1
        // worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));                // rotate  bits by 3
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));               // rotate  bits by 2
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 43:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 44:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];
          data = xor_with_bittable(data);
          //worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];
          data = xor_with_bittable(data);
          //worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));
          data = rotate_bits(data, 3);
          //worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); 
          data = rotate_by_self(data);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 45:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 2) | (worker.chunk[i] >> 6);
        // worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));                       // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 46:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] += worker.chunk[i];                 // +
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));    // rotate  bits by 5
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 47:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 48:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        // worker.chunk[i] = ~worker.chunk[i];                    // binary NOT operator
        // worker.chunk[i] = ~worker.chunk[i];                    // binary NOT operator
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5)); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 49:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];
          data = xor_with_bittable(data);
          //worker.chunk[i] += worker.chunk[i];
          data = add_with_self(data);
          //worker.chunk[i] = reverse8(worker.chunk[i]);
          data = reverse_vector(data);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 50:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);     // reverse bits
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3)); // rotate  bits by 3
        worker.chunk[i] += worker.chunk[i];              // +
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1)); // rotate  bits by 1
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 51:
      {
        worker.opt[worker.op] = true;
        uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] ^= worker.chunk[worker.pos2];
          data = xor_vectors(data, p2vec);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));
          data = rotate_bits(data, 5);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 52:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 53:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] += worker.chunk[i];
          data = add_with_self(data);
          //worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];
          data = xor_with_bittable(data);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 54:

#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);  // reverse bits
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
        // worker.chunk[i] = ~worker.chunk[i];    // binary NOT operator
        // worker.chunk[i] = ~worker.chunk[i];    // binary NOT operator
        // INSERT_RANDOM_CODE_END
      }

      break;
    case 55:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = reverse8(worker.chunk[i]);
          data = reverse_vector(data);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));
          data = rotate_bits(data, 1);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 56:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));  // rotate  bits by 1
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 57:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        //worker.chunk[i] = std::rotl(worker.chunk[i], 8); // no-op                // rotate  bits by 5
        // worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));                // rotate  bits by 3
        worker.chunk[i] = reverse8(worker.chunk[i]); // reverse bits
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 58:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));               // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] += worker.chunk[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 59:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));                // rotate  bits by 1
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 60:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
        worker.chunk[i] *= worker.chunk[i];              // *
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3)); // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 61:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));             // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        //worker.chunk[i] = std::rotl(worker.chunk[i], 8);             // rotate  bits by 3
        // worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));// rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 62:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));               // rotate  bits by 2
        worker.chunk[i] += worker.chunk[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 63:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));    // rotate  bits by 5
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] += worker.chunk[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 64:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4)); // rotate  bits by 4
        worker.chunk[i] *= worker.chunk[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 65:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        //worker.chunk[i] = std::rotl(worker.chunk[i], 8); // rotate  bits by 5
        // worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));             // rotate  bits by 3
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] *= worker.chunk[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 66:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));
          data = rotate_and_xor(data, 2);
          //worker.chunk[i] = reverse8(worker.chunk[i]);
          data = reverse_vector(data);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));
          data = rotate_bits(data, 1);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 67:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));    // rotate  bits by 1
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));   // rotate  bits by 2
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));    // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 68:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));               // rotate  bits by 4
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 69:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 70:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));            // rotate  bits by 4
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 71:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));             // rotate  bits by 5
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 72:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 73:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));    // rotate  bits by 5
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 74:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));                // rotate  bits by 3
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 75:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 76:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8)));
          data = rotate_by_self(data);
          //worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));
          data = rotate_and_xor(data, 2);
          //worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));
          data = rotate_bits(data, 5);
          //worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);
          data = shift_right_by_int_with_and(data, 3);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 77:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));             // rotate  bits by 3
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 78:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8)));
          data = rotate_by_self(data);
          //worker.chunk[i] = reverse8(worker.chunk[i]);
          data = reverse_vector(data);
          //worker.chunk[i] *= worker.chunk[i];
          data = mul_with_self(data);
          //worker.chunk[i] -= (worker.chunk[i] ^ 97);
          data = subtract_xored(data, 97);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 79:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4)); // rotate  bits by 4
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] += worker.chunk[i];               // +
        worker.chunk[i] *= worker.chunk[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 80:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 81:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));               // rotate  bits by 4
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 82:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
        // worker.chunk[i] = ~worker.chunk[i];        // binary NOT operator
        // worker.chunk[i] = ~worker.chunk[i];        // binary NOT operator
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 83:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        //worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));             // rotate  bits by 3
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        //worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 84:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));             // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 85:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 86:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8)));
          data = rotate_by_self(data);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] = ~worker.chunk[i];
          data = binary_not(data);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 87:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];               // +
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));  // rotate  bits by 3
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4)); // rotate  bits by 4
        worker.chunk[i] += worker.chunk[i];               // +
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 88:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));  // rotate  bits by 1
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 89:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];               // +
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 90:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);     // reverse bits
        worker.chunk[i] = (worker.chunk[i] << 6) | (worker.chunk[i] >> (8 - 6)); // rotate  bits by 5
        // worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));    // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 91:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));               // rotate  bits by 4
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 92:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 93:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));               // rotate  bits by 2
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] += worker.chunk[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 94:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));                // rotate  bits by 1
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 95:
    {
      worker.opt[worker.op] = true;
      for (int i = worker.pos1; i < worker.pos2; i+=16)
      {
        uint8x16_t vec = vld1q_u8(&worker.chunk[i]);

        // Shift the vector elements to the left by one position
        uint8x16_t shifted_left = vshlq_n_u8(vec, 1);
        uint8x16_t shifted_right = vshrq_n_u8(vec, 8-1);
        uint8x16_t rotated = vorrq_u8(shifted_left, shifted_right);
        //worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));  // rotate  bits by 1
      
        //worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
        uint8x16_t data = binary_not(rotated);vmvnq_u8(rotated);        
        
        uint8x16_t shifted_a = rotate_bits(data, 10);
        //worker.chunk[i] = std::rotl(worker.chunk[i], 10);

        vst1q_u8(&worker.chunk[i], shifted_a);
      }
      //memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, (worker.pos2-worker.pos1)%16);
      memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
    }
      break;
    case 96:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));   // rotate  bits by 2
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));   // rotate  bits by 2
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 97:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));             // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 98:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));            // rotate  bits by 4
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));            // rotate  bits by 4
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 99:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));            // rotate  bits by 4
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 100:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 101:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 102:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3)); // rotate  bits by 3
        worker.chunk[i] -= (worker.chunk[i] ^ 97);       // XOR and -
        worker.chunk[i] += worker.chunk[i];              // +
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3)); // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 103:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));                // rotate  bits by 1
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 104:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));    // rotate  bits by 5
        worker.chunk[i] += worker.chunk[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 105:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));                // rotate  bits by 3
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 106:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4)); // rotate  bits by 4
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));  // rotate  bits by 1
        worker.chunk[i] *= worker.chunk[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 107:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));            // rotate  bits by 2
        worker.chunk[i] = (worker.chunk[i] << 6) | (worker.chunk[i] >> (8 - 6));             // rotate  bits by 5
        // worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));             // rotate  bits by 1
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 108:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 109:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 110:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));            // rotate  bits by 2
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));            // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 111:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 112:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3)); // rotate  bits by 3
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5)); // rotate  bits by 5
        worker.chunk[i] -= (worker.chunk[i] ^ 97);       // XOR and -
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 113:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 6) | (worker.chunk[i] >> (8 - 6)); // rotate  bits by 5
        // worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));                           // rotate  bits by 1
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = ~worker.chunk[i];                 // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 114:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));                // rotate  bits by 1
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 115:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));                // rotate  bits by 3
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 116:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 117:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));                // rotate  bits by 3
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 118:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 119:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 120:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 121:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] *= worker.chunk[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 122:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8)));
          data = rotate_by_self(data);
          //worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));
          data = rotate_bits(data, 5);
          //worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));
          data = data = rotate_and_xor(data, 2);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 123:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = (worker.chunk[i] << 6) | (worker.chunk[i] >> (8 - 6));                // rotate  bits by 3
        // worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3)); // rotate  bits by 3
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 124:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 125:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));            // rotate  bits by 2
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 126:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> 7);
        // worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1)); // rotate  bits by 1
        // worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5)); // rotate  bits by 5
        worker.chunk[i] = reverse8(worker.chunk[i]); // reverse bits
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 127:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 128:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8)));
          data = rotate_by_self(data);
          //worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));
          data = rotate_and_xor(data, 2);
          //worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));
          data = rotate_and_xor(data, 2);
          //worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));
          data = rotate_bits(data, 5);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 129:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 130:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));                // rotate  bits by 1
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 131:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));    // rotate  bits by 1
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] *= worker.chunk[i];                 // *
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 132:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));                // rotate  bits by 5
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 133:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));             // rotate  bits by 5
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));            // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 134:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));               // rotate  bits by 4
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));                // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 135:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));            // rotate  bits by 2
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 136:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 137:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 138:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
        worker.chunk[i] += worker.chunk[i];           // +
        worker.chunk[i] -= (worker.chunk[i] ^ 97);    // XOR and -
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 139:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        //worker.chunk[i] = std::rotl(worker.chunk[i], 8); // rotate  bits by 5
        // worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));             // rotate  bits by 3
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));  // rotate  bits by 3
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 140:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));  // rotate  bits by 1
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 141:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));    // rotate  bits by 1
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] += worker.chunk[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 142:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));                // rotate  bits by 5
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 143:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));                // rotate  bits by 3
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 144:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 145:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = reverse8(worker.chunk[i]);
          data = reverse_vector(data);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));
          data = rotate_and_xor(data, 2);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 146:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 147:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));            // rotate  bits by 4
        worker.chunk[i] *= worker.chunk[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 148:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 149:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
        worker.chunk[i] = reverse8(worker.chunk[i]);  // reverse bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);    // XOR and -
        worker.chunk[i] += worker.chunk[i];           // +
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 150:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 151:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 152:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));            // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 153:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4)); // rotate  bits by 1
        // worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3)); // rotate  bits by 3
        // worker.chunk[i] = ~worker.chunk[i];     // binary NOT operator
        // worker.chunk[i] = ~worker.chunk[i];     // binary NOT operator
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 154:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));    // rotate  bits by 5
        worker.chunk[i] = ~worker.chunk[i];                 // binary NOT operator
        worker.chunk[i] ^= worker.chunk[worker.pos2];       // XOR
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 155:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] ^= worker.chunk[worker.pos2];       // XOR
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= worker.chunk[worker.pos2];       // XOR
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 156:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));             // rotate  bits by 3
        // worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));    // rotate  bits by 1
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 157:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 158:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));    // rotate  bits by 3
        worker.chunk[i] += worker.chunk[i];                 // +
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 159:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 160:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));             // rotate  bits by 1
        // worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));    // rotate  bits by 3
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 161:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));                // rotate  bits by 5
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 162:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] -= (worker.chunk[i] ^ 97);        // XOR and -
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 163:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));            // rotate  bits by 4
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 164:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                 // *
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] = ~worker.chunk[i];                 // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 165:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));            // rotate  bits by 4
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 166:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));  // rotate  bits by 3
        worker.chunk[i] += worker.chunk[i];               // +
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 167:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        // worker.chunk[i] = ~worker.chunk[i];        // binary NOT operator
        // worker.chunk[i] = ~worker.chunk[i];        // binary NOT operator
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 168:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 169:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));                // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));               // rotate  bits by 4
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 170:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);   // XOR and -
        worker.chunk[i] = reverse8(worker.chunk[i]); // reverse bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);   // XOR and -
        worker.chunk[i] *= worker.chunk[i];          // *
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 171:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));    // rotate  bits by 3
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 172:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));            // rotate  bits by 4
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 173:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 174:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 175:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3)); // rotate  bits by 3
        worker.chunk[i] -= (worker.chunk[i] ^ 97);       // XOR and -
        worker.chunk[i] *= worker.chunk[i];              // *
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5)); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 176:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] *= worker.chunk[i];              // *
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5)); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 177:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));               // rotate  bits by 2
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));               // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 178:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 179:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));            // rotate  bits by 2
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 180:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));            // rotate  bits by 4
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 181:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));            // rotate  bits by 2
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 182:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] = (worker.chunk[i] << 6) | (worker.chunk[i] >> (8 - 6)); // rotate  bits by 1
        // worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));         // rotate  bits by 5
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4)); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 183:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];        // +
        worker.chunk[i] -= (worker.chunk[i] ^ 97); // XOR and -
        worker.chunk[i] -= (worker.chunk[i] ^ 97); // XOR and -
        worker.chunk[i] *= worker.chunk[i];        // *
                                                     // INSERT_RANDOM_CODE_END
      }
      break;
    case 184:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] *= worker.chunk[i];                          // *
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));             // rotate  bits by 5
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 185:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));            // rotate  bits by 4
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));             // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 186:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));            // rotate  bits by 2
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));            // rotate  bits by 4
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 187:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
        worker.chunk[i] += worker.chunk[i];              // +
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3)); // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 188:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];
          data = xor_with_bittable(data);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      break;
    case 189:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));  // rotate  bits by 5
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4)); // rotate  bits by 4
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] -= (worker.chunk[i] ^ 97);        // XOR and -
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 190:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));                // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 191:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));                // rotate  bits by 3
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 192:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] *= worker.chunk[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 193:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 194:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 195:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));   // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];       // XOR
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 196:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));             // rotate  bits by 3
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 197:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));               // rotate  bits by 4
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] *= worker.chunk[i];                             // *
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 198:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 199:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];           // binary NOT operator
        worker.chunk[i] += worker.chunk[i];           // +
        worker.chunk[i] *= worker.chunk[i];           // *
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 200:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 201:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));  // rotate  bits by 3
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4)); // rotate  bits by 4
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 202:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));                // rotate  bits by 5
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 203:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));                // rotate  bits by 1
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 204:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));                // rotate  bits by 5
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));               // rotate  bits by 2
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 205:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));            // rotate  bits by 4
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 206:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));   // rotate  bits by 4
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 207:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        //worker.chunk[i] = std::rotl(worker.chunk[i], 8); // rotate  bits by 5
        // worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));                           // rotate  bits by 3
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 208:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));             // rotate  bits by 3
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 209:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));    // rotate  bits by 5
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 210:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));               // rotate  bits by 2
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));                // rotate  bits by 5
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 211:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));               // rotate  bits by 4
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 212:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));               // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 213:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));             // rotate  bits by 3
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 214:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 215:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] *= worker.chunk[i];                             // *
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 216:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 217:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));  // rotate  bits by 5
        worker.chunk[i] += worker.chunk[i];               // +
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));  // rotate  bits by 1
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4)); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 218:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]); // reverse bits
        worker.chunk[i] = ~worker.chunk[i];          // binary NOT operator
        worker.chunk[i] *= worker.chunk[i];          // *
        worker.chunk[i] -= (worker.chunk[i] ^ 97);   // XOR and -
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 219:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));               // rotate  bits by 4
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));                // rotate  bits by 3
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 220:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));             // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 221:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5)); // rotate  bits by 5
        worker.chunk[i] ^= worker.chunk[worker.pos2];    // XOR
        worker.chunk[i] = ~worker.chunk[i];              // binary NOT operator
        worker.chunk[i] = reverse8(worker.chunk[i]);     // reverse bits
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 222:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] *= worker.chunk[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 223:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));                // rotate  bits by 3
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 224:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] = (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));  // rotate  bits by 1
        // worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));             // rotate  bits by 3
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
      }
      break;
    case 225:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        worker.chunk[i] = ~worker.chunk[i];                          // binary NOT operator
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));             // rotate  bits by 3
      }
      break;
    case 226:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);  // reverse bits
        worker.chunk[i] -= (worker.chunk[i] ^ 97);    // XOR and -
        worker.chunk[i] *= worker.chunk[i];           // *
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 227:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 228:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 229:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));                // rotate  bits by 3
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));               // rotate  bits by 2
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 230:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 231:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));             // rotate  bits by 3
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] ^= worker.chunk[worker.pos2];                // XOR
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 232:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4)); // rotate  bits by 4
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 233:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));    // rotate  bits by 1
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));    // rotate  bits by 3
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 234:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3);    // shift right
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 235:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] *= worker.chunk[i];               // *
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));  // rotate  bits by 3
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 236:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= worker.chunk[worker.pos2];                   // XOR
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 237:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));             // rotate  bits by 5
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));            // rotate  bits by 2
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));             // rotate  bits by 3
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 238:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];              // +
        worker.chunk[i] += worker.chunk[i];              // +
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3)); // rotate  bits by 3
        worker.chunk[i] -= (worker.chunk[i] ^ 97);       // XOR and -
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 239:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 6) | (worker.chunk[i] >> (8 - 6)); // rotate  bits by 5
        // worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1)); // rotate  bits by 1
        worker.chunk[i] *= worker.chunk[i];                             // *
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 240:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                             // binary NOT operator
        worker.chunk[i] += worker.chunk[i];                             // +
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 241:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));   // rotate  bits by 4
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] ^= worker.chunk[worker.pos2];       // XOR
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 242:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];           // +
        worker.chunk[i] += worker.chunk[i];           // +
        worker.chunk[i] -= (worker.chunk[i] ^ 97);    // XOR and -
        worker.chunk[i] ^= worker.chunk[worker.pos2]; // XOR
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 243:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));    // rotate  bits by 5
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));   // rotate  bits by 2
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 244:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] = reverse8(worker.chunk[i]);      // reverse bits
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 245:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] -= (worker.chunk[i] ^ 97);                   // XOR and -
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));             // rotate  bits by 5
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));            // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 246:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                          // +
        worker.chunk[i] = (worker.chunk[i] << 1) | (worker.chunk[i] >> (8 - 1));             // rotate  bits by 1
        worker.chunk[i] = worker.chunk[i] >> (worker.chunk[i] & 3); // shift right
        worker.chunk[i] += worker.chunk[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 247:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));  // rotate  bits by 5
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));  // rotate  bits by 5
        worker.chunk[i] = ~worker.chunk[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 248:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = ~worker.chunk[i];                 // binary NOT operator
        worker.chunk[i] -= (worker.chunk[i] ^ 97);          // XOR and -
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = (worker.chunk[i] << 5) | (worker.chunk[i] >> (8 - 5));    // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 249:
      {
        worker.opt[worker.op] = true;
        // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
        for (int i = worker.pos1; i < worker.pos2; i+=16)
        {
          uint8x16_t data = vld1q_u8(&worker.chunk[i]);

          //worker.chunk[i] = reverse8(worker.chunk[i]);
          data = reverse_vector(data);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));
          data = rotate_and_xor(data, 4);
          //worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8)));
          data = rotate_by_self(data);

          vst1q_u8(&worker.chunk[i], data);
        }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
      }
      /*
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);                    // reverse bits
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));               // rotate  bits by 4
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));               // rotate  bits by 4
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      */
      break;
    case 250:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = worker.chunk[i] & worker.chunk[worker.pos2]; // AND
        worker.chunk[i] = (worker.chunk[i] << (worker.chunk[i] % 8)) | (worker.chunk[i] >> (8 - (worker.chunk[i] % 8))); // rotate  bits by random
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]];             // ones count bits
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 251:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] += worker.chunk[i];                 // +
        worker.chunk[i] ^= (byte)bitTable[worker.chunk[i]]; // ones count bits
        worker.chunk[i] = reverse8(worker.chunk[i]);        // reverse bits
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));   // rotate  bits by 2
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 252:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = reverse8(worker.chunk[i]);                 // reverse bits
        worker.chunk[i] ^= (worker.chunk[i] << 4) | (worker.chunk[i] >> (8 - 4));            // rotate  bits by 4
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));            // rotate  bits by 2
        worker.chunk[i] = worker.chunk[i] << (worker.chunk[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 253:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));  // rotate  bits by 3
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2)); // rotate  bits by 2
        worker.chunk[i] ^= worker.chunk[worker.pos2];     // XOR
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));  // rotate  bits by 3
        // INSERT_RANDOM_CODE_END

        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        worker.lhash = XXHash64::hash(worker.chunk, worker.pos2,0);
        //worker.lhash = XXH64(worker.chunk, worker.pos2, 0);
        //worker.lhash = XXH3_64bits(worker.chunk, worker.pos2);
      }
      break;
    case 254:
    case 255:
      RC4_set_key(&worker.key, 256,  worker.chunk);
// worker.chunk = highwayhash.Sum(worker.chunk[:], worker.chunk[:])
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.chunk[i] ^= static_cast<uint8_t>(std::bitset<8>(worker.chunk[i]).count()); // ones count bits
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));                                  // rotate  bits by 3
        worker.chunk[i] ^= (worker.chunk[i] << 2) | (worker.chunk[i] >> (8 - 2));                                 // rotate  bits by 2
        worker.chunk[i] = (worker.chunk[i] << 3) | (worker.chunk[i] >> (8 - 3));                                  // rotate  bits by 3
                                                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    default:
      break;
    }

    if(isTest) {
      break;
    }

    // if (op == 53) {
    //   std::cout << hexStr(worker.step_3, 256) << std::endl << std::endl;
    //   std::cout << hexStr(&worker.step_3[worker.pos1], 1) << std::endl;
    //   std::cout << hexStr(&worker.step_3[worker.pos2], 1) << std::endl;
    // }

    __builtin_prefetch(worker.chunk,0,3);
    // __builtin_prefetch(worker.chunk+64,0,3);
    // __builtin_prefetch(worker.chunk+128,0,3);
    __builtin_prefetch(worker.chunk+192,0,3);

    if (debugOpOrderAA) {
      printf("tries: %03lu  chunk_after[  0->%03d]: ", worker.tries, worker.pos2);
      for (int x = 0; x <= worker.pos2+16 && worker.pos2+16 < 256; x++) {
        printf("%02x", worker.chunk[x]);
      }
      printf("\n");
    }
    if (debugOpOrderAA && sus_op == worker.op) {
      break;
    }
    if (worker.op == sus_op && debugOpOrderAA) printf(" CPU: A: c[%02d] %02x c[%02d] %02x\n", worker.pos1, worker.chunk[worker.pos1], worker.pos2, worker.chunk[worker.pos2]);
    worker.A = (worker.chunk[worker.pos1] - worker.chunk[worker.pos2]);
    worker.A = (256 + (worker.A % 256)) % 256;

    if (worker.A < 0x10)
    { // 6.25 % probability
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);
      //worker.lhash = XXH64(worker.chunk, worker.pos2, 0);
      //worker.lhash = XXH3_64bits(worker.chunk, worker.pos2);

      // uint64_t test = XXHash64::hash(worker.step_3, worker.pos2, 0);
      if (worker.op == sus_op && debugOpOrderAA) printf(" CPU: A: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A < 0x20)
    { // 12.5 % probability
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);

      // uint64_t test = hash_64_fnv1a(worker.step_3, worker.pos2);
      if (worker.op == sus_op && debugOpOrderAA) printf(" CPU: B: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A < 0x30)
    { // 18.75 % probability
      worker.prev_lhash = worker.lhash + worker.prev_lhash;
      HH_ALIGNAS(16)
      const highwayhash::HH_U64 key2[2] = {worker.tries, worker.prev_lhash};
      worker.lhash = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2); // more deviations

      // uint64_t test = highwayhash::SipHash(key2, (char*)worker.step_3, worker.pos2); // more deviations
      if (worker.op == sus_op && debugOpOrderAA) printf(" CPU: C: new worker.lhash: %08jx\n", worker.lhash);
    }

    if (worker.A <= 0x40)
    { // 25% probablility
      // if (debugOpOrderAA && worker.op == sus_op) {
      //   printf("SIMD: D: RC4 key:\n");
      //   for (int i = 0; i < 256; i++) {
      //     printf("%d, ", worker.key.data[i]);
      //   }
      // }
      RC4(&worker.key, 256, worker.chunk, worker.chunk);
    }

    worker.chunk[255] = worker.chunk[255] ^ worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

    // if (debugOpOrderAA && worker.op == sus_op) {
    //   printf("SIMD op %d result:\n", worker.op);
    //   for (int i = 0; i < 256; i++) {
    //       printf("%02X ", worker.chunk[i]);
    //   } 
    //   printf("\n");
    // }

    // memcpy(&worker.sData[(worker.tries - 1) * 256], worker.step_3, 256);
    
    // std::copy(worker.step_3, worker.step_3 + 256, &worker.sData[(worker.tries - 1) * 256]);

    // memcpy(&worker->data.data()[(worker.tries - 1) * 256], worker.step_3, 256);

    // std::cout << hexStr(worker.step_3, 256) << std::endl;

    if (worker.tries > 260 + 16 || (worker.sData[(worker.tries-1)*256+255] >= 0xf0 && worker.tries > 260))
    {
      break;
    }
    if (debugOpOrderAA) printf("\n\n");
  }
  worker.data_len = static_cast<uint32_t>((worker.tries - 4) * 256 + (((static_cast<uint64_t>(worker.chunk[253]) << 8) | static_cast<uint64_t>(worker.chunk[254])) & 0x3ff));
}
