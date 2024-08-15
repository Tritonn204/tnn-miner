
#if defined(__aarch64__)

#include "astro_aarch64.hpp"

  #define MINPREFLEN 4


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

void branchComputeCPU_aarch64(workerData &worker, bool isTest, int wIndex)
{
  //if (debugOpOrderAA) printf("cpu\n");
  
  worker.templateIdx = 0;
  uint8_t chunkCount = 1;
  int firstChunk = 0;

  uint8_t lp1 = 0;
  uint8_t lp2 = 255;

  while (true)
  {
    if(isTest) {

    } else {
      worker.tries[wIndex]++;
      if (debugOpOrderAA) printf("t: 0x%hx p: 0x%lx l: 0x%lx\n", worker.tries[wIndex], worker.prev_lhash, worker.lhash);
      worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries[wIndex];
      // __builtin_prefetch(&worker.random_switcher,0,3);
      // printf("%d worker.random_switcher %d %08jx\n", worker.tries[wIndex], worker.random_switcher, worker.random_switcher);

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

      if (worker.tries[wIndex] > 0) {
        lp1 = std::min(lp1, worker.pos1);
        lp2 = std::max(lp2, worker.pos2);
      }

      worker.chunk = &worker.sData[(worker.tries[wIndex] - 1) * 256];
      if (debugOpOrderAA) printf("worker.op: %03d p1: %03d p2: %03d\n", worker.op, worker.pos1, worker.pos2);

      if (worker.tries[wIndex] == 1) {
        worker.prev_chunk = worker.chunk;
      } else {
        worker.prev_chunk = &worker.sData[(worker.tries[wIndex] - 2) * 256];
        /*
        if (debugOpOrderAA) {
          printf("tries: %03lu prev_chunk[0->%03d]: ", worker.tries[wIndex], worker.pos2);
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
        if (debugOpOrderAA) printf("loopa: %03lu %03d < %03d\n", worker.tries[wIndex], start_block, end_block);

        // Copy the blocks before worker.pos1
        for (int i = start_block; i < end_block; i++) {
            __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
            _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
        }

        if (debugOpOrderAA) printf("loopb: %03lu %03d < %03d\n", worker.tries[wIndex], end_block * 16, worker.pos1);
        // Copy the remaining bytes before worker.pos1
        for (int i = end_block * 16; i < worker.pos1; i++) {
            worker.chunk[i] = worker.prev_chunk[i];
        }

        // Calculate the start and end blocks
        start_block = (worker.pos2 + 15) / 16;
        end_block = 16;
        if (debugOpOrderAA) printf("loopc: %03lu %03d < %03d\n", worker.tries[wIndex], start_block, end_block);

        // Copy the blocks after worker.pos2
        for (int i = start_block; i < end_block; i++) {
            __m128i prev_data = _mm_loadu_si128((__m128i*)&worker.prev_chunk[i * 16]);
            _mm_storeu_si128((__m128i*)&worker.chunk[i * 16], prev_data);
        }

        if (debugOpOrderAA) printf("loopd: %03lu %03d < %03d\n", worker.tries[wIndex], worker.pos2, start_block * 16);
        // Copy the remaining bytes after worker.pos2
        for (int i = worker.pos2; i < start_block * 16; i++) {
          worker.chunk[i] = worker.prev_chunk[i];
        }
        */
      }

      if (debugOpOrderAA) {
        printf("tries: %03hu chunk_before[  0->%03d]: ", worker.tries[wIndex], worker.pos2);
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
        printf("tries: %03hu  chunk_fixed[  0->%03d]: ", worker.tries[wIndex], worker.pos2);
        for (int x = 0; x <= worker.pos2+16 && worker.pos2+16 < 256; x++) {
          //printf("%d \n", x);
          printf("%02x", worker.chunk[x]);
        }
        printf("\n");
      }
    }
    
    //printf("tries: %03d step_3[0->%-3d]: ", worker.tries[wIndex], worker.pos2);
    //for (int x = 0; x < worker.pos2; x++) {
    //  printf("%02x", worker.step_3[x]);
    //}
    //printf("\n");

    //printf("%02d ", worker.op);
    //if(worker.tries[wIndex] > 100) {
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_left_by_int_with_and(data, 3);
              data = rotate_bits(data, 1);
              data = and_vectors(data, p2vec);
              data = add_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 2:
        {
          worker.opt[worker.op] = true;
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_with_bittable(data);
              data = reverse_vector(data);
              data = shift_left_by_int_with_and(data, 3);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_by_self(data);
              data = rotate_bits(data, 3);
              data = xor_vectors(data, p2vec);
              data = rotate_bits(data, 1);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 4:
        {
          worker.opt[worker.op] = true;
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = binary_not(data);
              data = shift_right_by_int_with_and(data, 3);
              data = rotate_by_self(data);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_with_bittable(data);
              data = xor_vectors(data, p2vec);
              data = shift_left_by_int_with_and(data, 3);
              data = shift_right_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 6:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_left_by_int_with_and(data, 3);
              data = rotate_bits(data, 3);
              data = binary_not(data);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              // Load 16 bytes (128 bits) of data from chunk
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = rotate_by_self(data);
              data = xor_with_bittable(data);
              data = binary_not(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 8:
        {
          worker.opt[worker.op] = true;
          memcpy(worker.aarchFixup, &worker.chunk[worker.pos2], 16);
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              // Load 16 bytes (128 bits) of data from chunk
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = binary_not(data);
              data = rotate_bits(data, 2);
              data = shift_left_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 9:
        {
          worker.opt[worker.op] = true;
          memcpy(worker.aarchFixup, &worker.chunk[worker.pos2], 16);
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);

          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              // Load 16 bytes (128 bits) of data from chunk
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = rotate_and_xor(data, 4);
              data = shift_right_by_int_with_and(data, 3);
              data = rotate_and_xor(data, 2);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 10:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = binary_not(data);
              data = mul_with_self(data);
              data = rotate_bits(data, 3);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 6);
              data = and_vectors(data, p2vec);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 2);
              data = mul_with_self(data);
              data = rotate_and_xor(data, 2);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 1);
              data = xor_vectors(data, p2vec);
              data = shift_right_by_int_with_and(data, 3);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = shift_left_by_int_with_and(data, 3);
              data = mul_with_self(data);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 2);
              data = shift_left_by_int_with_and(data, 3);
              data = and_vectors(data, p2vec);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 4);
              data = mul_with_self(data);
              data = rotate_bits(data, 1);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = mul_with_self(data);
              data = rotate_bits(data, 5);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 4);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = subtract_xored(data, 97);
              data = rotate_bits(data, 5);
              data = shift_left_by_int_with_and(data, 3);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = and_vectors(data, p2vec);
              data = xor_vectors(data, p2vec);
              data = reverse_vector(data);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 1);
              data = xor_vectors(data, p2vec);
              data = add_with_self(data);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_left_by_int_with_and(data, 3);
              data = reverse_vector(data);
              data = mul_with_self(data);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 4);
              data = xor_with_bittable(data);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = shift_right_by_int_with_and(data, 3);
              data = rotate_and_xor(data, 4);
              data = rotate_bits(data, 5);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 25:
        worker.opt[worker.op] = true;
        for (int i = worker.pos1; i < worker.pos2; i += 16)
          {
            // Load 16 bytes (128 bits) of data from chunk
            uint8x16_t data = vld1q_u8(&worker.chunk[i]);
            data = xor_with_bittable(data);
            data = rotate_bits(data, 3);
            data = rotate_by_self(data);
            data = subtract_xored(data, 97);
            vst1q_u8(&worker.chunk[i], data);
          }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        break;
      case 26:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = mul_with_self(data);
              data = xor_with_bittable(data);
              data = add_with_self(data);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 5);
              data = and_vectors(data, p2vec);
              data = rotate_and_xor(data, 4);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_left_by_int_with_and(data, 3);
              data = add_with_self(data);
              data = add_with_self(data);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = mul_with_self(data);
              data = xor_vectors(data, p2vec);
              data = shift_right_by_int_with_and(data, 3);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = and_vectors(data, p2vec);
              data = rotate_and_xor(data, 4);
              data = rotate_bits(data, 5);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = binary_not(data);
              data = rotate_and_xor(data, 2);
              data = shift_left_by_int_with_and(data, 3);
              data = mul_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 32:
        {
          worker.opt[worker.op] = true;
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
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
        for (int i = worker.pos1; i < worker.pos2; i += 16)
          {
            // Load 16 bytes (128 bits) of data from chunk
            uint8x16_t data = vld1q_u8(&worker.chunk[i]);
            data = rotate_by_self(data);
            data = rotate_and_xor(data, 4);
            data = reverse_vector(data);
            data = vmulq_u8(data, data);
            vst1q_u8(&worker.chunk[i], data);
          }
        memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        break;
      case 34:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = subtract_xored(data, 97);
              data = shift_left_by_int_with_and(data, 3);
              data = shift_left_by_int_with_and(data, 3);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = binary_not(data);
              data = rotate_bits(data, 1);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_with_bittable(data);
              data = rotate_bits(data, 1);
              data = rotate_and_xor(data, 2);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_by_self(data);
              data = shift_right_by_int_with_and(data, 3);
              data = shift_right_by_int_with_and(data, 3);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = rotate_bits(data, 3);
              data = xor_with_bittable(data);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 2);
              data = xor_vectors(data, p2vec);
              data = shift_right_by_int_with_and(data, 3);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_by_self(data);
              data = xor_vectors(data, p2vec);
              data = xor_with_bittable(data);
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
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 5);
              data = subtract_xored(data, 97);
              data = rotate_bits(data, 3);
              data = rotate_and_xor(data, 4);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 42:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 4);
              data = rotate_and_xor(data, 2);
              data = rotate_by_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 43:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = and_vectors(data, p2vec);
              data = add_with_self(data);
              data = and_vectors(data, p2vec);
              data = subtract_xored(data, 97);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 44:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_with_bittable(data);
              data = xor_with_bittable(data);
              data = rotate_bits(data, 3);
              data = rotate_by_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 45:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 2);
              data = and_vectors(data, p2vec);
              data = xor_with_bittable(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 46:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_with_bittable(data);
              data = add_with_self(data);
              data = rotate_bits(data, 5);
              data = rotate_and_xor(data, 4);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 47:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 5);
              data = and_vectors(data, p2vec);
              data = rotate_bits(data, 5);
              data = shift_left_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 48:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_by_self(data);
              data = rotate_bits(data, 5);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 49:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_with_bittable(data);
              data = add_with_self(data);
              data = reverse_vector(data);
              data = rotate_and_xor(data, 4);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 50:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = reverse_vector(data);
              data = rotate_bits(data, 3);
              data = add_with_self(data);
              data = rotate_bits(data, 1);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 51:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = rotate_and_xor(data, 4);
              data = rotate_and_xor(data, 4);
              data = rotate_bits(data, 5);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 52:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_by_self(data);
              data = shift_right_by_int_with_and(data, 3);
              data = binary_not(data);
              data = xor_with_bittable(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 53:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = xor_with_bittable(data);
              data = rotate_and_xor(data, 4);
              data = rotate_and_xor(data, 4);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 54:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = reverse_vector(data);
              data = xor_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 55:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = reverse_vector(data);
              data = rotate_and_xor(data, 4);
              data = rotate_and_xor(data, 4);
              data = rotate_bits(data, 1);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 56:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 2);
              data = mul_with_self(data);
              data = binary_not(data);
              data = rotate_bits(data, 1);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 57:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_by_self(data);
              data = reverse_vector(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 58:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = reverse_vector(data);
              data = rotate_and_xor(data, 2);
              data = and_vectors(data, p2vec);
              data = add_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 59:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 1);
              data = mul_with_self(data);
              data = rotate_by_self(data);
              data = binary_not(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 60:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = binary_not(data);
              data = mul_with_self(data);
              data = rotate_bits(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 61:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 5);
              data = shift_left_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 62:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = and_vectors(data, p2vec);
              data = binary_not(data);
              data = rotate_and_xor(data, 2);
              data = add_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 63:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 5);
              data = xor_with_bittable(data);
              data = subtract_xored(data, 97);
              data = add_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 64:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = reverse_vector(data);
              data = rotate_and_xor(data, 4);
              data = mul_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 65:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 2);
              data = mul_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 66:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 2);
              data = reverse_vector(data);
              data = rotate_and_xor(data, 4);
              data = rotate_bits(data, 1);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 67:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 1);
              data = xor_with_bittable(data);
              data = rotate_and_xor(data, 2);
              data = rotate_bits(data, 5);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 68:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = and_vectors(data, p2vec);
              data = binary_not(data);
              data = rotate_and_xor(data, 4);
              data = xor_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 69:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = mul_with_self(data);
              data = reverse_vector(data);
              data = shift_right_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 70:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = mul_with_self(data);
              data = shift_right_by_int_with_and(data, 3);
              data = rotate_and_xor(data, 4);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 71:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 5);
              data = binary_not(data);
              data = mul_with_self(data);
              data = shift_left_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 72:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = reverse_vector(data);
              data = xor_with_bittable(data);
              data = xor_vectors(data, p2vec);
              data = shift_left_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 73:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_with_bittable(data);
              data = reverse_vector(data);
              data = rotate_bits(data, 5);
              data = subtract_xored(data, 97);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 74:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = mul_with_self(data);
              data = rotate_bits(data, 3);
              data = reverse_vector(data);
              data = and_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 75:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = mul_with_self(data);
              data = xor_with_bittable(data);
              data = and_vectors(data, p2vec);
              data = rotate_and_xor(data, 4);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 76:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_by_self(data);
              data = rotate_and_xor(data, 2);
              data = rotate_bits(data, 5);
              data = shift_right_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 77:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 3);
              data = add_with_self(data);
              data = shift_left_by_int_with_and(data, 3);
              data = xor_with_bittable(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 78:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_by_self(data);
              data = reverse_vector(data);
              data = mul_with_self(data);
              data = subtract_xored(data, 97);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 79:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 4);
              data = rotate_and_xor(data, 2);
              data = add_with_self(data);
              data = mul_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 80:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_by_self(data);
              data = shift_left_by_int_with_and(data, 3);
              data = add_with_self(data);
              data = and_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 81:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 4);
              data = shift_left_by_int_with_and(data, 3);
              data = rotate_by_self(data);
              data = xor_with_bittable(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 82:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = shift_right_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 83:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_left_by_int_with_and(data, 3);
              data = reverse_vector(data);
              data = rotate_bits(data, 3);
              data = reverse_vector(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 84:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = subtract_xored(data, 97);
              data = rotate_bits(data, 1);
              data = shift_left_by_int_with_and(data, 3);
              data = add_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 85:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = xor_vectors(data, p2vec);
              data = rotate_by_self(data);
              data = shift_left_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 86:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 4);
              data = rotate_by_self(data);
              data = rotate_and_xor(data, 4);
              data = binary_not(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 87:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = rotate_bits(data, 3);
              data = rotate_and_xor(data, 4);
              data = add_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 88:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 2);
              data = rotate_bits(data, 1);
              data = mul_with_self(data);
              data = binary_not(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 89:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = mul_with_self(data);
              data = binary_not(data);
              data = rotate_and_xor(data, 2);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 90:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = reverse_vector(data);
              data = rotate_bits(data, 6);
              data = shift_right_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 91:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_with_bittable(data);
              data = and_vectors(data, p2vec);
              data = rotate_and_xor(data, 4);
              data = reverse_vector(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 92:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_with_bittable(data);
              data = binary_not(data);
              data = xor_with_bittable(data);
              data = and_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 93:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 2);
              data = mul_with_self(data);
              data = and_vectors(data, p2vec);
              data = add_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 94:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 1);
              data = rotate_by_self(data);
              data = and_vectors(data, p2vec);
              data = shift_left_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 95:
        {
          worker.opt[worker.op] = true;
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t vec = vld1q_u8(&worker.chunk[i]);
              // Shift the vector elements to the left by one position
              uint8x16_t shifted_left = vshlq_n_u8(vec, 1);
              uint8x16_t shifted_right = vshrq_n_u8(vec, 8 - 1);
              uint8x16_t rotated = vorrq_u8(shifted_left, shifted_right);
              uint8x16_t data = binary_not(rotated);
              //vmvnq_u8(rotated);
              uint8x16_t shifted_a = rotate_bits(data, 10);
              vst1q_u8(&worker.chunk[i], shifted_a);
            }
          // memcpy(&worker.chunk[worker.pos2], worker.aarchFixup,
          // (worker.pos2-worker.pos1)%16);
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 96:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 2);
              data = rotate_and_xor(data, 2);
              data = xor_with_bittable(data);
              data = rotate_bits(data, 1);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 97:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 1);
              data = shift_left_by_int_with_and(data, 3);
              data = xor_with_bittable(data);
              data = shift_right_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 98:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 4);
              data = shift_left_by_int_with_and(data, 3);
              data = shift_right_by_int_with_and(data, 3);
              data = rotate_and_xor(data, 4);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 99:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 4);
              data = subtract_xored(data, 97);
              data = reverse_vector(data);
              data = shift_right_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 100:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_by_self(data);
              data = shift_left_by_int_with_and(data, 3);
              data = reverse_vector(data);
              data = xor_with_bittable(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 101:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = xor_with_bittable(data);
              data = shift_right_by_int_with_and(data, 3);
              data = binary_not(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 102:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 3);
              data = subtract_xored(data, 97);
              data = add_with_self(data);
              data = rotate_bits(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 103:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 1);
              data = reverse_vector(data);
              data = xor_vectors(data, p2vec);
              data = rotate_by_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 104:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = reverse_vector(data);
              data = xor_with_bittable(data);
              data = rotate_bits(data, 5);
              data = add_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 105:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_left_by_int_with_and(data, 3);
              data = rotate_bits(data, 3);
              data = rotate_by_self(data);
              data = rotate_and_xor(data, 2);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 106:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = reverse_vector(data);
              data = rotate_and_xor(data, 4);
              data = rotate_bits(data, 1);
              data = mul_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 107:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = rotate_and_xor(data, 2);
              data = rotate_bits(data, 6);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 108:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = binary_not(data);
              data = and_vectors(data, p2vec);
              data = rotate_and_xor(data, 2);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 109:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = mul_with_self(data);
              data = rotate_by_self(data);
              data = xor_vectors(data, p2vec);
              data = rotate_and_xor(data, 2);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 110:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = rotate_and_xor(data, 2);
              data = rotate_and_xor(data, 2);
              data = shift_right_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 111:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = mul_with_self(data);
              data = reverse_vector(data);
              data = mul_with_self(data);
              data = shift_right_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 112:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 3);
              data = binary_not(data);
              data = rotate_bits(data, 5);
              data = subtract_xored(data, 97);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 113:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 6);
              data = xor_with_bittable(data);
              data = binary_not(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 114:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 1);
              data = reverse_vector(data);
              data = rotate_by_self(data);
              data = binary_not(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 115:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_by_self(data);
              data = rotate_bits(data, 5);
              data = and_vectors(data, p2vec);
              data = rotate_bits(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 116:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = and_vectors(data, p2vec);
              data = xor_vectors(data, p2vec);
              data = xor_with_bittable(data);
              data = shift_left_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 117:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_left_by_int_with_and(data, 3);
              data = rotate_bits(data, 3);
              data = shift_left_by_int_with_and(data, 3);
              data = and_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 118:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = add_with_self(data);
              data = shift_left_by_int_with_and(data, 3);
              data = rotate_bits(data, 5);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 119:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = reverse_vector(data);
              data = rotate_and_xor(data, 2);
              data = binary_not(data);
              data = xor_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 120:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 2);
              data = mul_with_self(data);
              data = xor_vectors(data, p2vec);
              data = reverse_vector(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 121:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = add_with_self(data);
              data = xor_with_bittable(data);
              data = mul_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 122:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 4);
              data = rotate_by_self(data);
              data = rotate_bits(data, 5);
              data = data = rotate_and_xor(data, 2);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 123:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = and_vectors(data, p2vec);
              data = binary_not(data);
              data = rotate_bits(data, 6);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 124:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 2);
              data = rotate_and_xor(data, 2);
              data = xor_vectors(data, p2vec);
              data = binary_not(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 125:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = reverse_vector(data);
              data = rotate_and_xor(data, 2);
              data = add_with_self(data);
              data = shift_right_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 126:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 1);
              data = reverse_vector(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 127:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_left_by_int_with_and(data, 3);
              data = mul_with_self(data);
              data = and_vectors(data, p2vec);
              data = xor_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 128:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_by_self(data);
              data = rotate_and_xor(data, 2);
              data = rotate_and_xor(data, 2);
              data = rotate_bits(data, 5);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 129:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = binary_not(data);
              data = xor_with_bittable(data);
              data = xor_with_bittable(data);
              data = shift_right_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 130:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = rotate_by_self(data);
              data = rotate_bits(data, 1);
              data = rotate_and_xor(data, 4);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 131:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = subtract_xored(data, 97);
              data = rotate_bits(data, 1);
              data = xor_with_bittable(data);
              data = mul_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 132:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = and_vectors(data, p2vec);
              data = reverse_vector(data);
              data = rotate_bits(data, 5);
              data = rotate_and_xor(data, 2);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 133:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = rotate_bits(data, 5);
              data = rotate_and_xor(data, 2);
              data = shift_left_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 134:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = binary_not(data);
              data = rotate_and_xor(data, 4);
              data = rotate_bits(data, 1);
              data = and_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 135:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = rotate_and_xor(data, 2);
              data = add_with_self(data);
              data = reverse_vector(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 136:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = subtract_xored(data, 97);
              data = xor_vectors(data, p2vec);
              data = rotate_bits(data, 5);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 137:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 5);
              data = shift_right_by_int_with_and(data, 3);
              data = reverse_vector(data);
              data = rotate_by_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 138:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = xor_vectors(data, p2vec);
              data = add_with_self(data);
              data = subtract_xored(data, 97);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 139:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 2);
              data = rotate_bits(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 140:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 1);
              data = rotate_and_xor(data, 2);
              data = xor_vectors(data, p2vec);
              data = rotate_bits(data, 5);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 141:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 1);
              data = subtract_xored(data, 97);
              data = xor_with_bittable(data);
              data = add_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 142:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = and_vectors(data, p2vec);
              data = rotate_bits(data, 5);
              data = reverse_vector(data);
              data = rotate_and_xor(data, 2);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 143:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = and_vectors(data, p2vec);
              data = rotate_bits(data, 3);
              data = shift_right_by_int_with_and(data, 3);
              data = shift_left_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 144:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_by_self(data);
              data = shift_left_by_int_with_and(data, 3);
              data = binary_not(data);
              data = rotate_by_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 145:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = reverse_vector(data);
              data = rotate_and_xor(data, 4);
              data = rotate_and_xor(data, 2);
              data = rotate_and_xor(data, 4);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 146:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = and_vectors(data, p2vec);
              data = shift_left_by_int_with_and(data, 3);
              data = and_vectors(data, p2vec);
              data = xor_with_bittable(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 147:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = binary_not(data);
              data = shift_left_by_int_with_and(data, 3);
              data = rotate_and_xor(data, 4);
              data = mul_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 148:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = and_vectors(data, p2vec);
              data = rotate_bits(data, 5);
              data = shift_left_by_int_with_and(data, 3);
              data = subtract_xored(data, 97);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 149:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = reverse_vector(data);
              data = subtract_xored(data, 97);
              data = add_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 150:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_left_by_int_with_and(data, 3);
              data = shift_left_by_int_with_and(data, 3);
              data = shift_left_by_int_with_and(data, 3);
              data = and_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 151:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = shift_left_by_int_with_and(data, 3);
              data = mul_with_self(data);
              data = shift_left_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 152:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = binary_not(data);
              data = shift_left_by_int_with_and(data, 3);
              data = rotate_and_xor(data, 2);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 153:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 4);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 154:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 5);
              data = binary_not(data);
              data = xor_vectors(data, p2vec);
              data = xor_with_bittable(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 155:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = subtract_xored(data, 97);
              data = xor_vectors(data, p2vec);
              data = xor_with_bittable(data);
              data = xor_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 156:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = shift_right_by_int_with_and(data, 3);
              data = rotate_bits(data, 4);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 157:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = shift_left_by_int_with_and(data, 3);
              data = rotate_by_self(data);
              data = rotate_bits(data, 1);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 158:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_with_bittable(data);
              data = rotate_bits(data, 3);
              data = add_with_self(data);
              data = rotate_bits(data, 1);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 159:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = subtract_xored(data, 97);
              data = xor_vectors(data, p2vec);
              data = rotate_by_self(data);
              data = xor_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 160:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = reverse_vector(data);
              data = rotate_bits(data, 4);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 161:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = xor_vectors(data, p2vec);
              data = rotate_bits(data, 5);
              data = rotate_by_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 162:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = mul_with_self(data);
              data = reverse_vector(data);
              data = rotate_and_xor(data, 2);
              data = subtract_xored(data, 97);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 163:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_left_by_int_with_and(data, 3);
              data = subtract_xored(data, 97);
              data = rotate_and_xor(data, 4);
              data = rotate_bits(data, 1);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 164:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = mul_with_self(data);
              data = xor_with_bittable(data);
              data = subtract_xored(data, 97);
              data = binary_not(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 165:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 4);
              data = xor_vectors(data, p2vec);
              data = shift_left_by_int_with_and(data, 3);
              data = add_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 166:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 3);
              data = add_with_self(data);
              data = rotate_and_xor(data, 2);
              data = binary_not(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 167:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = mul_with_self(data);
              data = shift_right_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 168:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_by_self(data);
              data = and_vectors(data, p2vec);
              data = rotate_by_self(data);
              data = rotate_bits(data, 1);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 169:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 1);
              data = shift_left_by_int_with_and(data, 3);
              data = rotate_and_xor(data, 4);
              data = and_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 170:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = subtract_xored(data, 97);
              data = reverse_vector(data);
              data = subtract_xored(data, 97);
              data = mul_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 171:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 3);
              data = subtract_xored(data, 97);
              data = xor_with_bittable(data);
              data = reverse_vector(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 172:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 4);
              data = subtract_xored(data, 97);
              data = shift_left_by_int_with_and(data, 3);
              data = rotate_bits(data, 1);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 173:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = binary_not(data);
              data = shift_left_by_int_with_and(data, 3);
              data = mul_with_self(data);
              data = add_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 174:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = binary_not(data);
              data = rotate_by_self(data);
              data = xor_with_bittable(data);
              data = xor_with_bittable(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 175:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 3);
              data = subtract_xored(data, 97);
              data = mul_with_self(data);
              data = rotate_bits(data, 5);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 176:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = mul_with_self(data);
              data = xor_vectors(data, p2vec);
              data = rotate_bits(data, 5);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 177:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_with_bittable(data);
              data = rotate_and_xor(data, 2);
              data = rotate_and_xor(data, 2);
              data = and_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 178:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = and_vectors(data, p2vec);
              data = add_with_self(data);
              data = binary_not(data);
              data = rotate_bits(data, 1);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 179:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 2);
              data = add_with_self(data);
              data = shift_right_by_int_with_and(data, 3);
              data = reverse_vector(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 180:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = rotate_and_xor(data, 4);
              data = xor_vectors(data, p2vec);
              data = subtract_xored(data, 97);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 181:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = binary_not(data);
              data = shift_left_by_int_with_and(data, 3);
              data = rotate_and_xor(data, 2);
              data = rotate_bits(data, 5);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 182:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = rotate_bits(data, 6);
              data = rotate_and_xor(data, 4);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 183:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = subtract_xored(data, 97);
              data = subtract_xored(data, 97);
              data = mul_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 184:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_left_by_int_with_and(data, 3);
              data = mul_with_self(data);
              data = rotate_bits(data, 5);
              data = xor_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 185:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = binary_not(data);
              data = rotate_and_xor(data, 4);
              data = rotate_bits(data, 5);
              data = shift_right_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 186:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 2);
              data = rotate_and_xor(data, 4);
              data = subtract_xored(data, 97);
              data = shift_right_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 187:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = binary_not(data);
              data = add_with_self(data);
              data = rotate_bits(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 188:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 4);
              data = xor_with_bittable(data);
              data = rotate_and_xor(data, 4);
              data = rotate_and_xor(data, 4);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 189:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 5);
              data = rotate_and_xor(data, 4);
              data = xor_vectors(data, p2vec);
              data = subtract_xored(data, 97);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 190:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 5);
              data = shift_right_by_int_with_and(data, 3);
              data = and_vectors(data, p2vec);
              data = rotate_and_xor(data, 2);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 191:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = rotate_bits(data, 3);
              data = rotate_by_self(data);
              data = shift_right_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 192:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = shift_left_by_int_with_and(data, 3);
              data = add_with_self(data);
              data = mul_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 193:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = and_vectors(data, p2vec);
              data = shift_left_by_int_with_and(data, 3);
              data = rotate_by_self(data);
              data = rotate_bits(data, 1);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 194:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = and_vectors(data, p2vec);
              data = rotate_by_self(data);
              data = shift_left_by_int_with_and(data, 3);
              data = and_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 195:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_with_bittable(data);
              data = rotate_and_xor(data, 2);
              data = xor_vectors(data, p2vec);
              data = rotate_and_xor(data, 4);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 196:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 3);
              data = reverse_vector(data);
              data = shift_left_by_int_with_and(data, 3);
              data = rotate_bits(data, 1);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 197:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 4);
              data = rotate_by_self(data);
              data = mul_with_self(data);
              data = mul_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 198:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = shift_right_by_int_with_and(data, 3);
              data = reverse_vector(data);
              data = rotate_bits(data, 1);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 199:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = binary_not(data);
              data = add_with_self(data);
              data = mul_with_self(data);
              data = xor_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 200:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = xor_with_bittable(data);
              data = reverse_vector(data);
              data = reverse_vector(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 201:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 3);
              data = rotate_and_xor(data, 2);
              data = rotate_and_xor(data, 4);
              data = binary_not(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 202:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = binary_not(data);
              data = rotate_by_self(data);
              data = rotate_bits(data, 5);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 203:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = and_vectors(data, p2vec);
              data = rotate_bits(data, 1);
              data = rotate_by_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 204:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 5);
              data = rotate_and_xor(data, 2);
              data = rotate_by_self(data);
              data = xor_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 205:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_with_bittable(data);
              data = rotate_and_xor(data, 4);
              data = shift_left_by_int_with_and(data, 3);
              data = add_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 206:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 4);
              data = reverse_vector(data);
              data = reverse_vector(data);
              data = xor_with_bittable(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 207:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_with_bittable(data);
              data = xor_with_bittable(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 208:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = add_with_self(data);
              data = shift_right_by_int_with_and(data, 3);
              data = rotate_bits(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 209:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 5);
              data = reverse_vector(data);
              data = xor_with_bittable(data);
              data = subtract_xored(data, 97);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 210:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 2);
              data = rotate_by_self(data);
              data = rotate_bits(data, 5);
              data = binary_not(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 211:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 4);
              data = add_with_self(data);
              data = subtract_xored(data, 97);
              data = rotate_by_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 212:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_by_self(data);
              data = rotate_and_xor(data, 2);
              data = xor_vectors(data, p2vec);
              data = xor_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 213:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = shift_left_by_int_with_and(data, 3);
              data = rotate_bits(data, 3);
              data = subtract_xored(data, 97);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 214:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = subtract_xored(data, 97);
              data = shift_right_by_int_with_and(data, 3);
              data = binary_not(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 215:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = and_vectors(data, p2vec);
              data = shift_left_by_int_with_and(data, 3);
              data = mul_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 216:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_by_self(data);
              data = binary_not(data);
              data = subtract_xored(data, 97);
              data = and_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 217:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 5);
              data = add_with_self(data);
              data = rotate_bits(data, 1);
              data = rotate_and_xor(data, 4);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 218:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = reverse_vector(data);
              data = binary_not(data);
              data = mul_with_self(data);
              data = subtract_xored(data, 97);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 219:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 4);
              data = rotate_bits(data, 3);
              data = and_vectors(data, p2vec);
              data = reverse_vector(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 220:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 1);
              data = shift_left_by_int_with_and(data, 3);
              data = reverse_vector(data);
              data = shift_left_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 221:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 5);
              data = xor_vectors(data, p2vec);
              data = binary_not(data);
              data = reverse_vector(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 222:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = shift_right_by_int_with_and(data, 3);
              data = shift_left_by_int_with_and(data, 3);
              data = xor_vectors(data, p2vec);
              data = mul_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 223:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 3);
              data = xor_vectors(data, p2vec);
              data = rotate_by_self(data);
              data = subtract_xored(data, 97);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 224:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 2);
              data = rotate_bits(data, 4);
              data = shift_left_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 225:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = binary_not(data);
              data = shift_right_by_int_with_and(data, 3);
              data = reverse_vector(data);
              data = rotate_bits(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 226:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = reverse_vector(data);
              data = subtract_xored(data, 97);
              data = mul_with_self(data);
              data = xor_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 227:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = binary_not(data);
              data = shift_left_by_int_with_and(data, 3);
              data = subtract_xored(data, 97);
              data = and_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 228:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = shift_right_by_int_with_and(data, 3);
              data = add_with_self(data);
              data = xor_with_bittable(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 229:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 3);
              data = rotate_by_self(data);
              data = rotate_and_xor(data, 2);
              data = xor_with_bittable(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 230:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = mul_with_self(data);
              data = and_vectors(data, p2vec);
              data = rotate_by_self(data);
              data = rotate_by_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 231:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 3);
              data = shift_right_by_int_with_and(data, 3);
              data = xor_vectors(data, p2vec);
              data = reverse_vector(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 232:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = mul_with_self(data);
              data = mul_with_self(data);
              data = rotate_and_xor(data, 4);
              data = rotate_bits(data, 5);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 233:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 1);
              data = xor_with_bittable(data);
              data = rotate_bits(data, 3);
              data = xor_with_bittable(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 234:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = and_vectors(data, p2vec);
              data = mul_with_self(data);
              data = shift_right_by_int_with_and(data, 3);
              data = xor_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 235:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 2);
              data = mul_with_self(data);
              data = rotate_bits(data, 3);
              data = binary_not(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 236:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = xor_vectors(data, p2vec);
              data = add_with_self(data);
              data = and_vectors(data, p2vec);
              data = subtract_xored(data, 97);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 237:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 5);
              data = shift_left_by_int_with_and(data, 3);
              data = rotate_and_xor(data, 2);
              data = rotate_bits(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 238:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = add_with_self(data);
              data = rotate_bits(data, 3);
              data = subtract_xored(data, 97);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 239:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 6);
              data = mul_with_self(data);
              data = and_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 240:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = binary_not(data);
              data = add_with_self(data);
              data = and_vectors(data, p2vec);
              data = shift_left_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 241:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_and_xor(data, 4);
              data = xor_with_bittable(data);
              data = xor_vectors(data, p2vec);
              data = rotate_bits(data, 1);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 242:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = add_with_self(data);
              data = subtract_xored(data, 97);
              data = xor_vectors(data, p2vec);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 243:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 5);
              data = rotate_and_xor(data, 2);
              data = xor_with_bittable(data);
              data = rotate_bits(data, 1);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 244:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = binary_not(data);
              data = rotate_and_xor(data, 2);
              data = reverse_vector(data);
              data = rotate_bits(data, 5);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 245:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = subtract_xored(data, 97);
              data = rotate_bits(data, 5);
              data = rotate_and_xor(data, 2);
              data = shift_right_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 246:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = rotate_bits(data, 1);
              data = shift_right_by_int_with_and(data, 3);
              data = add_with_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 247:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = rotate_bits(data, 5);
              data = rotate_and_xor(data, 2);
              data = rotate_bits(data, 5);
              data = binary_not(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 248:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = binary_not(data);
              data = subtract_xored(data, 97);
              data = xor_with_bittable(data);
              data = rotate_bits(data, 5);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 249:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = reverse_vector(data);
              data = rotate_and_xor(data, 4);
              data = rotate_and_xor(data, 4);
              data = rotate_by_self(data);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 250:
        {
          worker.opt[worker.op] = true;
          uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = and_vectors(data, p2vec);
              data = rotate_by_self(data);
              data = xor_with_bittable(data);
              data = rotate_and_xor(data, 4);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 251:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = add_with_self(data);
              data = xor_with_bittable(data);
              data = reverse_vector(data);
              data = rotate_and_xor(data, 2);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
        }
        break;
      case 252:
        {
          worker.opt[worker.op] = true;
          // uint8x16_t p2vec = vdupq_n_u8(worker.chunk[worker.pos2]);
          for (int i = worker.pos1; i < worker.pos2; i += 16)
            {
              uint8x16_t data = vld1q_u8(&worker.chunk[i]);
              data = reverse_vector(data);
              data = rotate_and_xor(data, 4);
              data = rotate_and_xor(data, 2);
              data = shift_left_by_int_with_and(data, 3);
              vst1q_u8(&worker.chunk[i], data);
            }
          memcpy(&worker.chunk[worker.pos2], worker.aarchFixup, 16);
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
      RC4_set_key(&worker.key[wIndex], 256,  worker.chunk);
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
      printf("tries: %03hu  chunk_after[  0->%03d]: ", worker.tries[wIndex], worker.pos2);
      for (int x = 0; x <= worker.pos2+16 && worker.pos2+16 < 256; x++) {
        printf("%02x", worker.chunk[x]);
      }
      printf("\n");
    }
    if (debugOpOrderAA && sus_op == worker.op) {
      break;
    }
    if (worker.op == sus_op && debugOpOrderAA) printf(" CPU: A: c[%02d] %02x c[%02d] %02x\n", worker.pos1, worker.chunk[worker.pos1], worker.pos2, worker.chunk[worker.pos2]);
    
    uint8_t pushPos1 = lp1;
    uint8_t pushPos2 = lp2;

    if (worker.pos1 == worker.pos2) {
      pushPos1 = -1;
      pushPos2 = -1;
    }
    
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
      const highwayhash::HH_U64 key2[2] = {worker.tries[wIndex], worker.prev_lhash};
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
      RC4(&worker.key[wIndex], 256, worker.chunk,  worker.chunk);
      if (255 - pushPos2 < MINPREFLEN)
        pushPos2 = 255;
      if (pushPos1 < MINPREFLEN)
        pushPos1 = 0;


      if (pushPos1 == 255) pushPos1 = 0;
      
      worker.astroTemplate[worker.templateIdx] = templateMarker{
        (uint8_t)(chunkCount > 1 ? pushPos1 : 0),
        (uint8_t)(chunkCount > 1 ? pushPos2 : 255),
        (uint16_t)((firstChunk << 7) | chunkCount)
      };

      pushPos1 = 0;
      pushPos2 = 255;
      worker.templateIdx += (worker.tries[wIndex] > 1);
      firstChunk = worker.tries[wIndex]-1;
      lp1 = 255;
      lp2 = 0;
      chunkCount = 1;
    } else {
      chunkCount++;
    }

    if (255 - pushPos2 < MINPREFLEN)
      pushPos2 = 255;
    if (pushPos1 < MINPREFLEN)
      pushPos1 = 0;

    worker.chunk[255] = worker.chunk[255] ^ worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

    // if (debugOpOrderAA && worker.op == sus_op) {
    //   printf("SIMD op %d result:\n", worker.op);
    //   for (int i = 0; i < 256; i++) {
    //       printf("%02X ", worker.chunk[i]);
    //   } 
    //   printf("\n");
    // }

    // memcpy(&worker.sData[(worker.tries[wIndex] - 1) * 256], worker.step_3, 256);
    
    // std::copy(worker.step_3, worker.step_3 + 256, &worker.sData[(worker.tries[wIndex] - 1) * 256]);

    // memcpy(&worker->data.data()[(worker.tries[wIndex] - 1) * 256], worker.step_3, 256);

    // std::cout << hexStr(worker.step_3, 256) << std::endl;

    if (worker.tries[wIndex] > 260 + 16 || (worker.sData[(worker.tries[wIndex]-1)*256+255] >= 0xf0 && worker.tries[wIndex] > 260))
    {
      break;
    }
    if (debugOpOrderAA) printf("\n\n");
  }

  if (chunkCount > 0) {
    if (255 - lp2 < MINPREFLEN)
      lp2 = 255;
    if (lp1 < MINPREFLEN)
      lp1 = 0;
    worker.astroTemplate[worker.templateIdx] = templateMarker{
      (uint8_t)(chunkCount > 1 ? lp1 : 0),
      (uint8_t)(chunkCount > 1 ? lp2 : 255),
      (uint16_t)((firstChunk << 7) | chunkCount)
    };
    worker.templateIdx++;
  }

  worker.data_len = static_cast<uint32_t>((worker.tries[wIndex] - 4) * 256 + (((static_cast<uint64_t>(worker.chunk[253]) << 8) | static_cast<uint64_t>(worker.chunk[254])) & 0x3ff));
}

#endif
