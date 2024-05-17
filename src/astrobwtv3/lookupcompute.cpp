#include "lookupcompute.h"

#include <hugepages.h>
#include <lookup.h>
#include <algorithm>
#include <bit>
#include <bitset>
#include <cstdint>
#include <fstream>
#include <filesystem>

// Generate lookup tables for values computed formerly by branchedCompute

// @param workerData: The workerData context that will reference these tables
// @param lookup2D: a pointer to the 2D (non-deterministic) lookup table of 2-byte chunks
// @param lookup3D: a pointer to the 3D (deterministic) lookup table of 1-byte chunks

void lookupGen(workerData &worker, uint16_t *lookup2D, byte *lookup3D) {
  for (int op = 0; op < 256; op++) {
    for (int v2 = 0; v2 < 256; v2++) {
      for (int val = 0; val < 256; val++) {
        auto spot = std::find(worker.branchedOps, worker.branchedOps + branchedOps_size, op);
        if (spot == worker.branchedOps + branchedOps_size) continue;
        int pos = std::distance(worker.branchedOps, spot);
        worker.branched_idx[op] = pos;
        byte trueVal = (byte)val;
        branchResult(trueVal, op, v2);
        worker.lookup3D[pos*256*256 + v2*256 + val] = trueVal;
      }
    }
    int d = -1;
    for (int val = 0; val < 256*256; val++) {
      auto spot = std::find(worker.regularOps, worker.regularOps + regOps_size, op);
      if (spot == worker.regularOps + regOps_size) continue;
      int pos = std::distance(worker.regularOps, spot);
      worker.reg_idx[op] = pos;
      uint16_t trueVal = val;
      byte v1 = (byte)(trueVal >> 8);
      byte v2 = (byte)(trueVal & 0xFF);
      branchResult(v1, op, 0);
      branchResult(v2, op, 0);

      // printf("hex combo val: %04X\n", val);
      // printf("hex p1: %02X\n", val >> 8);
      // printf("hex p2: %02X\n", val & 0xFF);

      // printf("result p1: %02X\n", v1);
      // printf("result p2: %02X\n", v2);
      // printf("lookup entry: %04X\n", v2 | (uint16_t)v1 << 8);

      trueVal = v2 | (uint16_t)v1 << 8;
      worker.lookup2D[pos*256*256 + val] = trueVal;
    }
  }
  std::ofstream file1("2d.bin", std::ios::binary);
  if (file1.is_open()) {
      file1.write(reinterpret_cast<const char*>(worker.lookup2D), 256*256*regOps_size);
      file1.close();
  } else {
      std::cerr << "Unable to open file: " << "2d.bin" << std::endl;
  }
  std::ofstream file2("3d.bin", std::ios::binary);
  if (file2.is_open()) {
      file2.write(reinterpret_cast<const char*>(worker.lookup3D), 256*256*branchedOps_size);
      file2.close();
  } else {
      std::cerr << "Unable to open file: " << "3d.bin" << std::endl;
  }
  // worker.lookup3D = &lookup3D[0];
  // worker.lookup2D = &lookup2D[0];
}

void branchResult(byte &val, int op, byte v2) {
    switch (op)
    {
    case 0:


        // INSERT_RANDOM_CODE_START
        val ^= (byte)bitTable[val];             // ones count bits
        val = std::rotl(val, 5);                // rotate  bits by 5
        val *= val;                             // *
        val = std::rotl(val, val); // rotate  bits by random

        // // INSERT_RANDOM_CODE_END
        // worker.t1 = worker.step_3[worker.pos1];
        // worker.t2 = v2;
        // worker.step_3[worker.pos1] = reverse8(worker.t2);
        // v2 = reverse8(worker.t1);
      break;

    case 1:


        // INSERT_RANDOM_CODE_START
        val = val << (val & 3);    // shift left
        val = std::rotl(val, 1);                // rotate  bits by 1
        val = val & v2; // AND
        val += val;                             // +
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 2:


        // INSERT_RANDOM_CODE_START
        val ^= (byte)bitTable[val];          // ones count bits
        val = reverse8(val);                 // reverse bits
        val = val << (val & 3); // shift left
        val ^= (byte)bitTable[val];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 3:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, val); // rotate  bits by random
        val = std::rotl(val, 3);                // rotate  bits by 3
        val ^= v2;                   // XOR
        val = std::rotl(val, 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 4:


        // INSERT_RANDOM_CODE_START
        val = ~val;                             // binary NOT operator
        val = val >> (val & 3);    // shift right
        val = std::rotl(val, val); // rotate  bits by random
        val -= (val ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 5:
    {



        // INSERT_RANDOM_CODE_START
        val ^= (byte)bitTable[val];          // ones count bits
        val ^= v2;                // XOR
        val = val << (val & 3); // shift left
        val = val >> (val & 3); // shift right

        // INSERT_RANDOM_CODE_END
    }
    break;
    case 6:


        // INSERT_RANDOM_CODE_START
        val = val << (val & 3); // shift left
        val = std::rotl(val, 3);             // rotate  bits by 3
        val = ~val;                          // binary NOT operator
        val -= (val ^ 97);                   // XOR and -

        // INSERT_RANDOM_CODE_END
      break;

    case 7:


        // INSERT_RANDOM_CODE_START
        val += val;                             // +
        val = std::rotl(val, val); // rotate  bits by random
        val ^= (byte)bitTable[val];             // ones count bits
        val = ~val;                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 8:


        // INSERT_RANDOM_CODE_START
        val = ~val;               // binary NOT operator
        val = std::rotl(val, 10); // rotate  bits by 5
        // val = std::rotl(val, 5);// rotate  bits by 5
        val = val << (val & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 9:


        // INSERT_RANDOM_CODE_START
        val ^= v2;                // XOR
        val ^= std::rotl(val, 4);            // rotate  bits by 4
        val = val >> (val & 3); // shift right
        val ^= std::rotl(val, 2);            // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 10:


        // INSERT_RANDOM_CODE_START
        val = ~val;              // binary NOT operator
        val *= val;              // *
        val = std::rotl(val, 3); // rotate  bits by 3
        val *= val;              // *
                                                           // INSERT_RANDOM_CODE_END
      break;

    case 11:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 6); // rotate  bits by 1
        // val = std::rotl(val, 5);            // rotate  bits by 5
        val = val & v2; // AND
        val = std::rotl(val, val); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 12:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val *= val;               // *
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val = ~val;               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 13:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 1);             // rotate  bits by 1
        val ^= v2;                // XOR
        val = val >> (val & 3); // shift right
        val = std::rotl(val, 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 14:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3); // shift right
        val = val << (val & 3); // shift left
        val *= val;                          // *
        val = val << (val & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 15:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 2);               // rotate  bits by 2
        val = val << (val & 3);    // shift left
        val = val & v2; // AND
        val -= (val ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 16:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 4); // rotate  bits by 4
        val *= val;               // *
        val = std::rotl(val, 1);  // rotate  bits by 1
        val = ~val;               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 17:


        // INSERT_RANDOM_CODE_START
        val ^= v2;    // XOR
        val *= val;              // *
        val = std::rotl(val, 5); // rotate  bits by 5
        val = ~val;              // binary NOT operator
                                                           // INSERT_RANDOM_CODE_END
      break;

    case 18:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 4); // rotate  bits by 4
        val = std::rotl(val, 9);  // rotate  bits by 3
        // val = std::rotl(val, 1);             // rotate  bits by 1
        // val = std::rotl(val, 5);         // rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      break;

    case 19:


        // INSERT_RANDOM_CODE_START
        val -= (val ^ 97);                   // XOR and -
        val = std::rotl(val, 5);             // rotate  bits by 5
        val = val << (val & 3); // shift left
        val += val;                          // +
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 20:


        // INSERT_RANDOM_CODE_START
        val = val & v2; // AND
        val ^= v2;                   // XOR
        val = reverse8(val);                    // reverse bits
        val ^= std::rotl(val, 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 21:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 1);                // rotate  bits by 1
        val ^= v2;                   // XOR
        val += val;                             // +
        val = val & v2; // AND
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 22:


        // INSERT_RANDOM_CODE_START
        val = val << (val & 3); // shift left
        val = reverse8(val);                 // reverse bits
        val *= val;                          // *
        val = std::rotl(val, 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 23:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 4); // rotate  bits by 3
        // val = std::rotl(val, 1);                           // rotate  bits by 1
        val ^= (byte)bitTable[val];             // ones count bits
        val = val & v2; // AND
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 24:


        // INSERT_RANDOM_CODE_START
        val += val;                          // +
        val = val >> (val & 3); // shift right
        val ^= std::rotl(val, 4);            // rotate  bits by 4
        val = std::rotl(val, 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 25:


        // INSERT_RANDOM_CODE_START
        val ^= (byte)bitTable[val];             // ones count bits
        val = std::rotl(val, 3);                // rotate  bits by 3
        val = std::rotl(val, val); // rotate  bits by random
        val -= (val ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 26:


        // INSERT_RANDOM_CODE_START
        val *= val;                 // *
        val ^= (byte)bitTable[val]; // ones count bits
        val += val;                 // +
        val = reverse8(val);        // reverse bits
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 27:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 5);                // rotate  bits by 5
        val = val & v2; // AND
        val ^= std::rotl(val, 4);               // rotate  bits by 4
        val = std::rotl(val, 5);                // rotate  bits by 5
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 28:


        // INSERT_RANDOM_CODE_START
        val = val << (val & 3); // shift left
        val += val;                          // +
        val += val;                          // +
        val = std::rotl(val, 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 29:


        // INSERT_RANDOM_CODE_START
        val *= val;                          // *
        val ^= v2;                // XOR
        val = val >> (val & 3); // shift right
        val += val;                          // +
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 30:


        // INSERT_RANDOM_CODE_START
        val = val & v2; // AND
        val ^= std::rotl(val, 4);               // rotate  bits by 4
        val = std::rotl(val, 5);                // rotate  bits by 5
        val = val << (val & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 31:


        // INSERT_RANDOM_CODE_START
        val = ~val;                          // binary NOT operator
        val ^= std::rotl(val, 2);            // rotate  bits by 2
        val = val << (val & 3); // shift left
        val *= val;                          // *
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 32:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val = reverse8(val);      // reverse bits
        val = std::rotl(val, 3);  // rotate  bits by 3
        val ^= std::rotl(val, 2); // rotate  bits by 2
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 33:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, val); // rotate  bits by random
        val ^= std::rotl(val, 4);               // rotate  bits by 4
        val = reverse8(val);                    // reverse bits
        val *= val;                             // *
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 34:


        // INSERT_RANDOM_CODE_START
        val -= (val ^ 97);                   // XOR and -
        val = val << (val & 3); // shift left
        val = val << (val & 3); // shift left
        val -= (val ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 35:


        // INSERT_RANDOM_CODE_START
        val += val;              // +
        val = ~val;              // binary NOT operator
        val = std::rotl(val, 1); // rotate  bits by 1
        val ^= v2;    // XOR
                                                           // INSERT_RANDOM_CODE_END
      break;

    case 36:


        // INSERT_RANDOM_CODE_START
        val ^= (byte)bitTable[val]; // ones count bits
        val = std::rotl(val, 1);    // rotate  bits by 1
        val ^= std::rotl(val, 2);   // rotate  bits by 2
        val = std::rotl(val, 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 37:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, val); // rotate  bits by random
        val = val >> (val & 3);    // shift right
        val = val >> (val & 3);    // shift right
        val *= val;                             // *
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 38:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3);    // shift right
        val = std::rotl(val, 3);                // rotate  bits by 3
        val ^= (byte)bitTable[val];             // ones count bits
        val = std::rotl(val, val); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 39:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 2);               // rotate  bits by 2
        val ^= v2;                   // XOR
        val = val >> (val & 3);    // shift right
        val = val & v2; // AND
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 40:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, val); // rotate  bits by random
        val ^= v2;                   // XOR
        val ^= (byte)bitTable[val];             // ones count bits
        val ^= v2;                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 41:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 5);  // rotate  bits by 5
        val -= (val ^ 97);        // XOR and -
        val = std::rotl(val, 3);  // rotate  bits by 3
        val ^= std::rotl(val, 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 42:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 4); // rotate  bits by 1
        // val = std::rotl(val, 3);                // rotate  bits by 3
        val ^= std::rotl(val, 2);               // rotate  bits by 2
        val = std::rotl(val, val); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 43:


        // INSERT_RANDOM_CODE_START
        val = val & v2; // AND
        val += val;                             // +
        val = val & v2; // AND
        val -= (val ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 44:


        // INSERT_RANDOM_CODE_START
        val ^= (byte)bitTable[val];             // ones count bits
        val ^= (byte)bitTable[val];             // ones count bits
        val = std::rotl(val, 3);                // rotate  bits by 3
        val = std::rotl(val, val); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 45:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 10); // rotate  bits by 5
        // val = std::rotl(val, 5);                       // rotate  bits by 5
        val = val & v2; // AND
        val ^= (byte)bitTable[val];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 46:


        // INSERT_RANDOM_CODE_START
        val ^= (byte)bitTable[val]; // ones count bits
        val += val;                 // +
        val = std::rotl(val, 5);    // rotate  bits by 5
        val ^= std::rotl(val, 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 47:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 5);                // rotate  bits by 5
        val = val & v2; // AND
        val = std::rotl(val, 5);                // rotate  bits by 5
        val = val << (val & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 48:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, val); // rotate  bits by random
        // val = ~val;                    // binary NOT operator
        // val = ~val;                    // binary NOT operator
        val = std::rotl(val, 5); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      break;

    case 49:


        // INSERT_RANDOM_CODE_START
        val ^= (byte)bitTable[val]; // ones count bits
        val += val;                 // +
        val = reverse8(val);        // reverse bits
        val ^= std::rotl(val, 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 50:


        // INSERT_RANDOM_CODE_START
        val = reverse8(val);     // reverse bits
        val = std::rotl(val, 3); // rotate  bits by 3
        val += val;              // +
        val = std::rotl(val, 1); // rotate  bits by 1
                                                           // INSERT_RANDOM_CODE_END
      break;

    case 51:


        // INSERT_RANDOM_CODE_START
        val ^= v2;     // XOR
        val ^= std::rotl(val, 4); // rotate  bits by 4
        val ^= std::rotl(val, 4); // rotate  bits by 4
        val = std::rotl(val, 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 52:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, val); // rotate  bits by random
        val = val >> (val & 3);    // shift right
        val = ~val;                             // binary NOT operator
        val ^= (byte)bitTable[val];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 53:


        // INSERT_RANDOM_CODE_START
        val += val;                 // +
        val ^= (byte)bitTable[val]; // ones count bits
        val ^= std::rotl(val, 4);   // rotate  bits by 4
        val ^= std::rotl(val, 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 54:



        // INSERT_RANDOM_CODE_START
        val = reverse8(val);  // reverse bits
        val ^= v2; // XOR
        // val = ~val;    // binary NOT operator
        // val = ~val;    // binary NOT operator
        // INSERT_RANDOM_CODE_END

      break;
    case 55:


        // INSERT_RANDOM_CODE_START
        val = reverse8(val);      // reverse bits
        val ^= std::rotl(val, 4); // rotate  bits by 4
        val ^= std::rotl(val, 4); // rotate  bits by 4
        val = std::rotl(val, 1);  // rotate  bits by 1
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 56:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val *= val;               // *
        val = ~val;               // binary NOT operator
        val = std::rotl(val, 1);  // rotate  bits by 1
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 57:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, val); // rotate  bits by random
        val = std::rotl(val, 8);                // rotate  bits by 5
        // val = std::rotl(val, 3);                // rotate  bits by 3
        val = reverse8(val); // reverse bits
                                                       // INSERT_RANDOM_CODE_END
      break;

    case 58:


        // INSERT_RANDOM_CODE_START
        val = reverse8(val);                    // reverse bits
        val ^= std::rotl(val, 2);               // rotate  bits by 2
        val = val & v2; // AND
        val += val;                             // +
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 59:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 1);                // rotate  bits by 1
        val *= val;                             // *
        val = std::rotl(val, val); // rotate  bits by random
        val = ~val;                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 60:


        // INSERT_RANDOM_CODE_START
        val ^= v2;    // XOR
        val = ~val;              // binary NOT operator
        val *= val;              // *
        val = std::rotl(val, 3); // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
      break;

    case 61:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 5);             // rotate  bits by 5
        val = val << (val & 3); // shift left
        val = std::rotl(val, 8);             // rotate  bits by 3
        // val = std::rotl(val, 5);// rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      break;

    case 62:


        // INSERT_RANDOM_CODE_START
        val = val & v2; // AND
        val = ~val;                             // binary NOT operator
        val ^= std::rotl(val, 2);               // rotate  bits by 2
        val += val;                             // +
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 63:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 5);    // rotate  bits by 5
        val ^= (byte)bitTable[val]; // ones count bits
        val -= (val ^ 97);          // XOR and -
        val += val;                 // +
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 64:


        // INSERT_RANDOM_CODE_START
        val ^= v2;     // XOR
        val = reverse8(val);      // reverse bits
        val ^= std::rotl(val, 4); // rotate  bits by 4
        val *= val;               // *
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 65:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 8); // rotate  bits by 5
        // val = std::rotl(val, 3);             // rotate  bits by 3
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val *= val;               // *
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 66:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val = reverse8(val);      // reverse bits
        val ^= std::rotl(val, 4); // rotate  bits by 4
        val = std::rotl(val, 1);  // rotate  bits by 1
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 67:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 1);    // rotate  bits by 1
        val ^= (byte)bitTable[val]; // ones count bits
        val ^= std::rotl(val, 2);   // rotate  bits by 2
        val = std::rotl(val, 5);    // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 68:


        // INSERT_RANDOM_CODE_START
        val = val & v2; // AND
        val = ~val;                             // binary NOT operator
        val ^= std::rotl(val, 4);               // rotate  bits by 4
        val ^= v2;                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 69:


        // INSERT_RANDOM_CODE_START
        val += val;                          // +
        val *= val;                          // *
        val = reverse8(val);                 // reverse bits
        val = val >> (val & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 70:


        // INSERT_RANDOM_CODE_START
        val ^= v2;                // XOR
        val *= val;                          // *
        val = val >> (val & 3); // shift right
        val ^= std::rotl(val, 4);            // rotate  bits by 4
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 71:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 5);             // rotate  bits by 5
        val = ~val;                          // binary NOT operator
        val *= val;                          // *
        val = val << (val & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 72:


        // INSERT_RANDOM_CODE_START
        val = reverse8(val);                 // reverse bits
        val ^= (byte)bitTable[val];          // ones count bits
        val ^= v2;                // XOR
        val = val << (val & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 73:


        // INSERT_RANDOM_CODE_START
        val ^= (byte)bitTable[val]; // ones count bits
        val = reverse8(val);        // reverse bits
        val = std::rotl(val, 5);    // rotate  bits by 5
        val -= (val ^ 97);          // XOR and -
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 74:


        // INSERT_RANDOM_CODE_START
        val *= val;                             // *
        val = std::rotl(val, 3);                // rotate  bits by 3
        val = reverse8(val);                    // reverse bits
        val = val & v2; // AND
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 75:


        // INSERT_RANDOM_CODE_START
        val *= val;                             // *
        val ^= (byte)bitTable[val];             // ones count bits
        val = val & v2; // AND
        val ^= std::rotl(val, 4);               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 76:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, val); // rotate  bits by random
        val ^= std::rotl(val, 2);               // rotate  bits by 2
        val = std::rotl(val, 5);                // rotate  bits by 5
        val = val >> (val & 3);    // shift right
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 77:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 3);             // rotate  bits by 3
        val += val;                          // +
        val = val << (val & 3); // shift left
        val ^= (byte)bitTable[val];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 78:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, val); // rotate  bits by random
        val = reverse8(val);                    // reverse bits
        val *= val;                             // *
        val -= (val ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 79:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 4); // rotate  bits by 4
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val += val;               // +
        val *= val;               // *
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 80:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, val); // rotate  bits by random
        val = val << (val & 3);    // shift left
        val += val;                             // +
        val = val & v2; // AND
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 81:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 4);               // rotate  bits by 4
        val = val << (val & 3);    // shift left
        val = std::rotl(val, val); // rotate  bits by random
        val ^= (byte)bitTable[val];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 82:


        // INSERT_RANDOM_CODE_START
        val ^= v2; // XOR
        // val = ~val;        // binary NOT operator
        // val = ~val;        // binary NOT operator
        val = val >> (val & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 83:


        // INSERT_RANDOM_CODE_START
        val = val << (val & 3); // shift left
        val = reverse8(val);                 // reverse bits
        val = std::rotl(val, 3);             // rotate  bits by 3
        val = reverse8(val);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 84:


        // INSERT_RANDOM_CODE_START
        val -= (val ^ 97);                   // XOR and -
        val = std::rotl(val, 1);             // rotate  bits by 1
        val = val << (val & 3); // shift left
        val += val;                          // +
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 85:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3);    // shift right
        val ^= v2;                   // XOR
        val = std::rotl(val, val); // rotate  bits by random
        val = val << (val & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 86:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 4);               // rotate  bits by 4
        val = std::rotl(val, val); // rotate  bits by random
        val ^= std::rotl(val, 4);               // rotate  bits by 4
        val = ~val;                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 87:


        // INSERT_RANDOM_CODE_START
        val += val;               // +
        val = std::rotl(val, 3);  // rotate  bits by 3
        val ^= std::rotl(val, 4); // rotate  bits by 4
        val += val;               // +
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 88:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val = std::rotl(val, 1);  // rotate  bits by 1
        val *= val;               // *
        val = ~val;               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 89:


        // INSERT_RANDOM_CODE_START
        val += val;               // +
        val *= val;               // *
        val = ~val;               // binary NOT operator
        val ^= std::rotl(val, 2); // rotate  bits by 2
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 90:


        // INSERT_RANDOM_CODE_START
        val = reverse8(val);     // reverse bits
        val = std::rotl(val, 6); // rotate  bits by 5
        // val = std::rotl(val, 1);    // rotate  bits by 1
        val = val >> (val & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 91:


        // INSERT_RANDOM_CODE_START
        val ^= (byte)bitTable[val];             // ones count bits
        val = val & v2; // AND
        val ^= std::rotl(val, 4);               // rotate  bits by 4
        val = reverse8(val);                    // reverse bits
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 92:


        // INSERT_RANDOM_CODE_START
        val ^= (byte)bitTable[val];             // ones count bits
        val = ~val;                             // binary NOT operator
        val ^= (byte)bitTable[val];             // ones count bits
        val = val & v2; // AND
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 93:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 2);               // rotate  bits by 2
        val *= val;                             // *
        val = val & v2; // AND
        val += val;                             // +
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 94:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 1);                // rotate  bits by 1
        val = std::rotl(val, val); // rotate  bits by random
        val = val & v2; // AND
        val = val << (val & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 95:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 1);  // rotate  bits by 1
        val = ~val;               // binary NOT operator
        val = std::rotl(val, 10); // rotate  bits by 5
        // val = std::rotl(val, 5); // rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      break;

    case 96:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 2);   // rotate  bits by 2
        val ^= std::rotl(val, 2);   // rotate  bits by 2
        val ^= (byte)bitTable[val]; // ones count bits
        val = std::rotl(val, 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 97:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 1);             // rotate  bits by 1
        val = val << (val & 3); // shift left
        val ^= (byte)bitTable[val];          // ones count bits
        val = val >> (val & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 98:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 4);            // rotate  bits by 4
        val = val << (val & 3); // shift left
        val = val >> (val & 3); // shift right
        val ^= std::rotl(val, 4);            // rotate  bits by 4
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 99:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 4);            // rotate  bits by 4
        val -= (val ^ 97);                   // XOR and -
        val = reverse8(val);                 // reverse bits
        val = val >> (val & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 100:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, val); // rotate  bits by random
        val = val << (val & 3);    // shift left
        val = reverse8(val);                    // reverse bits
        val ^= (byte)bitTable[val];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 101:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3); // shift right
        val ^= (byte)bitTable[val];          // ones count bits
        val = val >> (val & 3); // shift right
        val = ~val;                          // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 102:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 3); // rotate  bits by 3
        val -= (val ^ 97);       // XOR and -
        val += val;              // +
        val = std::rotl(val, 3); // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
      break;

    case 103:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 1);                // rotate  bits by 1
        val = reverse8(val);                    // reverse bits
        val ^= v2;                   // XOR
        val = std::rotl(val, val); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 104:


        // INSERT_RANDOM_CODE_START
        val = reverse8(val);        // reverse bits
        val ^= (byte)bitTable[val]; // ones count bits
        val = std::rotl(val, 5);    // rotate  bits by 5
        val += val;                 // +
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 105:


        // INSERT_RANDOM_CODE_START
        val = val << (val & 3);    // shift left
        val = std::rotl(val, 3);                // rotate  bits by 3
        val = std::rotl(val, val); // rotate  bits by random
        val ^= std::rotl(val, 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 106:


        // INSERT_RANDOM_CODE_START
        val = reverse8(val);      // reverse bits
        val ^= std::rotl(val, 4); // rotate  bits by 4
        val = std::rotl(val, 1);  // rotate  bits by 1
        val *= val;               // *
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 107:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3); // shift right
        val ^= std::rotl(val, 2);            // rotate  bits by 2
        val = std::rotl(val, 6);             // rotate  bits by 5
        // val = std::rotl(val, 1);             // rotate  bits by 1
        // INSERT_RANDOM_CODE_END
      break;

    case 108:


        // INSERT_RANDOM_CODE_START
        val ^= v2;                   // XOR
        val = ~val;                             // binary NOT operator
        val = val & v2; // AND
        val ^= std::rotl(val, 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 109:


        // INSERT_RANDOM_CODE_START
        val *= val;                             // *
        val = std::rotl(val, val); // rotate  bits by random
        val ^= v2;                   // XOR
        val ^= std::rotl(val, 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 110:


        // INSERT_RANDOM_CODE_START
        val += val;                          // +
        val ^= std::rotl(val, 2);            // rotate  bits by 2
        val ^= std::rotl(val, 2);            // rotate  bits by 2
        val = val >> (val & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 111:


        // INSERT_RANDOM_CODE_START
        val *= val;                          // *
        val = reverse8(val);                 // reverse bits
        val *= val;                          // *
        val = val >> (val & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 112:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 3); // rotate  bits by 3
        val = ~val;              // binary NOT operator
        val = std::rotl(val, 5); // rotate  bits by 5
        val -= (val ^ 97);       // XOR and -
                                                           // INSERT_RANDOM_CODE_END
      break;

    case 113:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 6); // rotate  bits by 5
        // val = std::rotl(val, 1);                           // rotate  bits by 1
        val ^= (byte)bitTable[val]; // ones count bits
        val = ~val;                 // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 114:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 1);                // rotate  bits by 1
        val = reverse8(val);                    // reverse bits
        val = std::rotl(val, val); // rotate  bits by random
        val = ~val;                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 115:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, val); // rotate  bits by random
        val = std::rotl(val, 5);                // rotate  bits by 5
        val = val & v2; // AND
        val = std::rotl(val, 3);                // rotate  bits by 3
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 116:


        // INSERT_RANDOM_CODE_START
        val = val & v2; // AND
        val ^= v2;                   // XOR
        val ^= (byte)bitTable[val];             // ones count bits
        val = val << (val & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 117:


        // INSERT_RANDOM_CODE_START
        val = val << (val & 3);    // shift left
        val = std::rotl(val, 3);                // rotate  bits by 3
        val = val << (val & 3);    // shift left
        val = val & v2; // AND
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 118:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3); // shift right
        val += val;                          // +
        val = val << (val & 3); // shift left
        val = std::rotl(val, 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 119:


        // INSERT_RANDOM_CODE_START
        val = reverse8(val);      // reverse bits
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val = ~val;               // binary NOT operator
        val ^= v2;     // XOR
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 120:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val *= val;               // *
        val ^= v2;     // XOR
        val = reverse8(val);      // reverse bits
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 121:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3); // shift right
        val += val;                          // +
        val ^= (byte)bitTable[val];          // ones count bits
        val *= val;                          // *
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 122:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 4);               // rotate  bits by 4
        val = std::rotl(val, val); // rotate  bits by random
        val = std::rotl(val, 5);                // rotate  bits by 5
        val ^= std::rotl(val, 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 123:


        // INSERT_RANDOM_CODE_START
        val = val & v2; // AND
        val = ~val;                             // binary NOT operator
        val = std::rotl(val, 6);                // rotate  bits by 3
        // val = std::rotl(val, 3); // rotate  bits by 3
        // INSERT_RANDOM_CODE_END
      break;

    case 124:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val ^= v2;     // XOR
        val = ~val;               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 125:


        // INSERT_RANDOM_CODE_START
        val = reverse8(val);                 // reverse bits
        val ^= std::rotl(val, 2);            // rotate  bits by 2
        val += val;                          // +
        val = val >> (val & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 126:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 9); // rotate  bits by 3
        // val = std::rotl(val, 1); // rotate  bits by 1
        // val = std::rotl(val, 5); // rotate  bits by 5
        val = reverse8(val); // reverse bits
                                                       // INSERT_RANDOM_CODE_END
      break;

    case 127:


        // INSERT_RANDOM_CODE_START
        val = val << (val & 3);    // shift left
        val *= val;                             // *
        val = val & v2; // AND
        val ^= v2;                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 128:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, val); // rotate  bits by random
        val ^= std::rotl(val, 2);               // rotate  bits by 2
        val ^= std::rotl(val, 2);               // rotate  bits by 2
        val = std::rotl(val, 5);                // rotate  bits by 5
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 129:


        // INSERT_RANDOM_CODE_START
        val = ~val;                          // binary NOT operator
        val ^= (byte)bitTable[val];          // ones count bits
        val ^= (byte)bitTable[val];          // ones count bits
        val = val >> (val & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 130:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3);    // shift right
        val = std::rotl(val, val); // rotate  bits by random
        val = std::rotl(val, 1);                // rotate  bits by 1
        val ^= std::rotl(val, 4);               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 131:


        // INSERT_RANDOM_CODE_START
        val -= (val ^ 97);          // XOR and -
        val = std::rotl(val, 1);    // rotate  bits by 1
        val ^= (byte)bitTable[val]; // ones count bits
        val *= val;                 // *
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 132:


        // INSERT_RANDOM_CODE_START
        val = val & v2; // AND
        val = reverse8(val);                    // reverse bits
        val = std::rotl(val, 5);                // rotate  bits by 5
        val ^= std::rotl(val, 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 133:


        // INSERT_RANDOM_CODE_START
        val ^= v2;                // XOR
        val = std::rotl(val, 5);             // rotate  bits by 5
        val ^= std::rotl(val, 2);            // rotate  bits by 2
        val = val << (val & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 134:


        // INSERT_RANDOM_CODE_START
        val = ~val;                             // binary NOT operator
        val ^= std::rotl(val, 4);               // rotate  bits by 4
        val = std::rotl(val, 1);                // rotate  bits by 1
        val = val & v2; // AND
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 135:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3); // shift right
        val ^= std::rotl(val, 2);            // rotate  bits by 2
        val += val;                          // +
        val = reverse8(val);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 136:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3); // shift right
        val -= (val ^ 97);                   // XOR and -
        val ^= v2;                // XOR
        val = std::rotl(val, 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 137:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 5);                // rotate  bits by 5
        val = val >> (val & 3);    // shift right
        val = reverse8(val);                    // reverse bits
        val = std::rotl(val, val); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 138:


        // INSERT_RANDOM_CODE_START
        val ^= v2; // XOR
        val ^= v2; // XOR
        val += val;           // +
        val -= (val ^ 97);    // XOR and -
                                                        // INSERT_RANDOM_CODE_END
      break;

    case 139:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 8); // rotate  bits by 5
        // val = std::rotl(val, 3);             // rotate  bits by 3
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val = std::rotl(val, 3);  // rotate  bits by 3
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 140:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 1);  // rotate  bits by 1
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val ^= v2;     // XOR
        val = std::rotl(val, 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 141:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 1);    // rotate  bits by 1
        val -= (val ^ 97);          // XOR and -
        val ^= (byte)bitTable[val]; // ones count bits
        val += val;                 // +
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 142:


        // INSERT_RANDOM_CODE_START
        val = val & v2; // AND
        val = std::rotl(val, 5);                // rotate  bits by 5
        val = reverse8(val);                    // reverse bits
        val ^= std::rotl(val, 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 143:


        // INSERT_RANDOM_CODE_START
        val = val & v2; // AND
        val = std::rotl(val, 3);                // rotate  bits by 3
        val = val >> (val & 3);    // shift right
        val = val << (val & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 144:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, val); // rotate  bits by random
        val = val << (val & 3);    // shift left
        val = ~val;                             // binary NOT operator
        val = std::rotl(val, val); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 145:


        // INSERT_RANDOM_CODE_START
        val = reverse8(val);      // reverse bits
        val ^= std::rotl(val, 4); // rotate  bits by 4
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val ^= std::rotl(val, 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 146:


        // INSERT_RANDOM_CODE_START
        val = val & v2; // AND
        val = val << (val & 3);    // shift left
        val = val & v2; // AND
        val ^= (byte)bitTable[val];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 147:


        // INSERT_RANDOM_CODE_START
        val = ~val;                          // binary NOT operator
        val = val << (val & 3); // shift left
        val ^= std::rotl(val, 4);            // rotate  bits by 4
        val *= val;                          // *
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 148:


        // INSERT_RANDOM_CODE_START
        val = val & v2; // AND
        val = std::rotl(val, 5);                // rotate  bits by 5
        val = val << (val & 3);    // shift left
        val -= (val ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 149:


        // INSERT_RANDOM_CODE_START
        val ^= v2; // XOR
        val = reverse8(val);  // reverse bits
        val -= (val ^ 97);    // XOR and -
        val += val;           // +
                                                        // INSERT_RANDOM_CODE_END
      break;

    case 150:


        // INSERT_RANDOM_CODE_START
        val = val << (val & 3);    // shift left
        val = val << (val & 3);    // shift left
        val = val << (val & 3);    // shift left
        val = val & v2; // AND
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 151:


        // INSERT_RANDOM_CODE_START
        val += val;                          // +
        val = val << (val & 3); // shift left
        val *= val;                          // *
        val = val << (val & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 152:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3); // shift right
        val = ~val;                          // binary NOT operator
        val = val << (val & 3); // shift left
        val ^= std::rotl(val, 2);            // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 153:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 4); // rotate  bits by 1
        // val = std::rotl(val, 3); // rotate  bits by 3
        // val = ~val;     // binary NOT operator
        // val = ~val;     // binary NOT operator
        // INSERT_RANDOM_CODE_END
      break;

    case 154:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 5);    // rotate  bits by 5
        val = ~val;                 // binary NOT operator
        val ^= v2;       // XOR
        val ^= (byte)bitTable[val]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 155:


        // INSERT_RANDOM_CODE_START
        val -= (val ^ 97);          // XOR and -
        val ^= v2;       // XOR
        val ^= (byte)bitTable[val]; // ones count bits
        val ^= v2;       // XOR
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 156:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3); // shift right
        val = val >> (val & 3); // shift right
        val = std::rotl(val, 4);             // rotate  bits by 3
        // val = std::rotl(val, 1);    // rotate  bits by 1
        // INSERT_RANDOM_CODE_END
      break;

    case 157:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3);    // shift right
        val = val << (val & 3);    // shift left
        val = std::rotl(val, val); // rotate  bits by random
        val = std::rotl(val, 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 158:


        // INSERT_RANDOM_CODE_START
        val ^= (byte)bitTable[val]; // ones count bits
        val = std::rotl(val, 3);    // rotate  bits by 3
        val += val;                 // +
        val = std::rotl(val, 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 159:


        // INSERT_RANDOM_CODE_START
        val -= (val ^ 97);                      // XOR and -
        val ^= v2;                   // XOR
        val = std::rotl(val, val); // rotate  bits by random
        val ^= v2;                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 160:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3); // shift right
        val = reverse8(val);                 // reverse bits
        val = std::rotl(val, 4);             // rotate  bits by 1
        // val = std::rotl(val, 3);    // rotate  bits by 3
        // INSERT_RANDOM_CODE_END
      break;

    case 161:


        // INSERT_RANDOM_CODE_START
        val ^= v2;                   // XOR
        val ^= v2;                   // XOR
        val = std::rotl(val, 5);                // rotate  bits by 5
        val = std::rotl(val, val); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 162:


        // INSERT_RANDOM_CODE_START
        val *= val;               // *
        val = reverse8(val);      // reverse bits
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val -= (val ^ 97);        // XOR and -
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 163:


        // INSERT_RANDOM_CODE_START
        val = val << (val & 3); // shift left
        val -= (val ^ 97);                   // XOR and -
        val ^= std::rotl(val, 4);            // rotate  bits by 4
        val = std::rotl(val, 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 164:


        // INSERT_RANDOM_CODE_START
        val *= val;                 // *
        val ^= (byte)bitTable[val]; // ones count bits
        val -= (val ^ 97);          // XOR and -
        val = ~val;                 // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 165:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 4);            // rotate  bits by 4
        val ^= v2;                // XOR
        val = val << (val & 3); // shift left
        val += val;                          // +
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 166:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 3);  // rotate  bits by 3
        val += val;               // +
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val = ~val;               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 167:


        // INSERT_RANDOM_CODE_START
        // val = ~val;        // binary NOT operator
        // val = ~val;        // binary NOT operator
        val *= val;                          // *
        val = val >> (val & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 168:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, val); // rotate  bits by random
        val = val & v2; // AND
        val = std::rotl(val, val); // rotate  bits by random
        val = std::rotl(val, 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 169:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 1);                // rotate  bits by 1
        val = val << (val & 3);    // shift left
        val ^= std::rotl(val, 4);               // rotate  bits by 4
        val = val & v2; // AND
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 170:


        // INSERT_RANDOM_CODE_START
        val -= (val ^ 97);   // XOR and -
        val = reverse8(val); // reverse bits
        val -= (val ^ 97);   // XOR and -
        val *= val;          // *
                                                       // INSERT_RANDOM_CODE_END
      break;

    case 171:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 3);    // rotate  bits by 3
        val -= (val ^ 97);          // XOR and -
        val ^= (byte)bitTable[val]; // ones count bits
        val = reverse8(val);        // reverse bits
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 172:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 4);            // rotate  bits by 4
        val -= (val ^ 97);                   // XOR and -
        val = val << (val & 3); // shift left
        val = std::rotl(val, 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 173:


        // INSERT_RANDOM_CODE_START
        val = ~val;                          // binary NOT operator
        val = val << (val & 3); // shift left
        val *= val;                          // *
        val += val;                          // +
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 174:


        // INSERT_RANDOM_CODE_START
        val = ~val;                             // binary NOT operator
        val = std::rotl(val, val); // rotate  bits by random
        val ^= (byte)bitTable[val];             // ones count bits
        val ^= (byte)bitTable[val];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 175:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 3); // rotate  bits by 3
        val -= (val ^ 97);       // XOR and -
        val *= val;              // *
        val = std::rotl(val, 5); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      break;

    case 176:


        // INSERT_RANDOM_CODE_START
        val ^= v2;    // XOR
        val *= val;              // *
        val ^= v2;    // XOR
        val = std::rotl(val, 5); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      break;

    case 177:


        // INSERT_RANDOM_CODE_START
        val ^= (byte)bitTable[val];             // ones count bits
        val ^= std::rotl(val, 2);               // rotate  bits by 2
        val ^= std::rotl(val, 2);               // rotate  bits by 2
        val = val & v2; // AND
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 178:


        // INSERT_RANDOM_CODE_START
        val = val & v2; // AND
        val += val;                             // +
        val = ~val;                             // binary NOT operator
        val = std::rotl(val, 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 179:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 2);            // rotate  bits by 2
        val += val;                          // +
        val = val >> (val & 3); // shift right
        val = reverse8(val);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 180:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3); // shift right
        val ^= std::rotl(val, 4);            // rotate  bits by 4
        val ^= v2;                // XOR
        val -= (val ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 181:


        // INSERT_RANDOM_CODE_START
        val = ~val;                          // binary NOT operator
        val = val << (val & 3); // shift left
        val ^= std::rotl(val, 2);            // rotate  bits by 2
        val = std::rotl(val, 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 182:


        // INSERT_RANDOM_CODE_START
        val ^= v2;    // XOR
        val = std::rotl(val, 6); // rotate  bits by 1
        // val = std::rotl(val, 5);         // rotate  bits by 5
        val ^= std::rotl(val, 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 183:


        // INSERT_RANDOM_CODE_START
        val += val;        // +
        val -= (val ^ 97); // XOR and -
        val -= (val ^ 97); // XOR and -
        val *= val;        // *
                                                     // INSERT_RANDOM_CODE_END
      break;

    case 184:


        // INSERT_RANDOM_CODE_START
        val = val << (val & 3); // shift left
        val *= val;                          // *
        val = std::rotl(val, 5);             // rotate  bits by 5
        val ^= v2;                // XOR
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 185:


        // INSERT_RANDOM_CODE_START
        val = ~val;                          // binary NOT operator
        val ^= std::rotl(val, 4);            // rotate  bits by 4
        val = std::rotl(val, 5);             // rotate  bits by 5
        val = val >> (val & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 186:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 2);            // rotate  bits by 2
        val ^= std::rotl(val, 4);            // rotate  bits by 4
        val -= (val ^ 97);                   // XOR and -
        val = val >> (val & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 187:


        // INSERT_RANDOM_CODE_START
        val ^= v2;    // XOR
        val = ~val;              // binary NOT operator
        val += val;              // +
        val = std::rotl(val, 3); // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
      break;

    case 188:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 4);   // rotate  bits by 4
        val ^= (byte)bitTable[val]; // ones count bits
        val ^= std::rotl(val, 4);   // rotate  bits by 4
        val ^= std::rotl(val, 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 189:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 5);  // rotate  bits by 5
        val ^= std::rotl(val, 4); // rotate  bits by 4
        val ^= v2;     // XOR
        val -= (val ^ 97);        // XOR and -
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 190:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 5);                // rotate  bits by 5
        val = val >> (val & 3);    // shift right
        val = val & v2; // AND
        val ^= std::rotl(val, 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 191:


        // INSERT_RANDOM_CODE_START
        val += val;                             // +
        val = std::rotl(val, 3);                // rotate  bits by 3
        val = std::rotl(val, val); // rotate  bits by random
        val = val >> (val & 3);    // shift right
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 192:


        // INSERT_RANDOM_CODE_START
        val += val;                          // +
        val = val << (val & 3); // shift left
        val += val;                          // +
        val *= val;                          // *
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 193:


        // INSERT_RANDOM_CODE_START
        val = val & v2; // AND
        val = val << (val & 3);    // shift left
        val = std::rotl(val, val); // rotate  bits by random
        val = std::rotl(val, 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 194:


        // INSERT_RANDOM_CODE_START
        val = val & v2; // AND
        val = std::rotl(val, val); // rotate  bits by random
        val = val << (val & 3);    // shift left
        val = val & v2; // AND
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 195:


        // INSERT_RANDOM_CODE_START
        val ^= (byte)bitTable[val]; // ones count bits
        val ^= std::rotl(val, 2);   // rotate  bits by 2
        val ^= v2;       // XOR
        val ^= std::rotl(val, 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 196:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 3);             // rotate  bits by 3
        val = reverse8(val);                 // reverse bits
        val = val << (val & 3); // shift left
        val = std::rotl(val, 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 197:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 4);               // rotate  bits by 4
        val = std::rotl(val, val); // rotate  bits by random
        val *= val;                             // *
        val *= val;                             // *
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 198:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3); // shift right
        val = val >> (val & 3); // shift right
        val = reverse8(val);                 // reverse bits
        val = std::rotl(val, 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 199:


        // INSERT_RANDOM_CODE_START
        val = ~val;           // binary NOT operator
        val += val;           // +
        val *= val;           // *
        val ^= v2; // XOR
                                                        // INSERT_RANDOM_CODE_END
      break;

    case 200:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3); // shift right
        val ^= (byte)bitTable[val];          // ones count bits
        val = reverse8(val);                 // reverse bits
        val = reverse8(val);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 201:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 3);  // rotate  bits by 3
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val ^= std::rotl(val, 4); // rotate  bits by 4
        val = ~val;               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 202:


        // INSERT_RANDOM_CODE_START
        val ^= v2;                   // XOR
        val = ~val;                             // binary NOT operator
        val = std::rotl(val, val); // rotate  bits by random
        val = std::rotl(val, 5);                // rotate  bits by 5
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 203:


        // INSERT_RANDOM_CODE_START
        val ^= v2;                   // XOR
        val = val & v2; // AND
        val = std::rotl(val, 1);                // rotate  bits by 1
        val = std::rotl(val, val); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 204:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 5);                // rotate  bits by 5
        val ^= std::rotl(val, 2);               // rotate  bits by 2
        val = std::rotl(val, val); // rotate  bits by random
        val ^= v2;                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 205:


        // INSERT_RANDOM_CODE_START
        val ^= (byte)bitTable[val];          // ones count bits
        val ^= std::rotl(val, 4);            // rotate  bits by 4
        val = val << (val & 3); // shift left
        val += val;                          // +
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 206:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 4);   // rotate  bits by 4
        val = reverse8(val);        // reverse bits
        val = reverse8(val);        // reverse bits
        val ^= (byte)bitTable[val]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 207:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 8); // rotate  bits by 5
        // val = std::rotl(val, 3);                           // rotate  bits by 3
        val ^= (byte)bitTable[val]; // ones count bits
        val ^= (byte)bitTable[val]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 208:


        // INSERT_RANDOM_CODE_START
        val += val;                          // +
        val += val;                          // +
        val = val >> (val & 3); // shift right
        val = std::rotl(val, 3);             // rotate  bits by 3
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 209:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 5);    // rotate  bits by 5
        val = reverse8(val);        // reverse bits
        val ^= (byte)bitTable[val]; // ones count bits
        val -= (val ^ 97);          // XOR and -
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 210:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 2);               // rotate  bits by 2
        val = std::rotl(val, val); // rotate  bits by random
        val = std::rotl(val, 5);                // rotate  bits by 5
        val = ~val;                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 211:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 4);               // rotate  bits by 4
        val += val;                             // +
        val -= (val ^ 97);                      // XOR and -
        val = std::rotl(val, val); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 212:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, val); // rotate  bits by random
        val ^= std::rotl(val, 2);               // rotate  bits by 2
        val ^= v2;                   // XOR
        val ^= v2;                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 213:


        // INSERT_RANDOM_CODE_START
        val += val;                          // +
        val = val << (val & 3); // shift left
        val = std::rotl(val, 3);             // rotate  bits by 3
        val -= (val ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 214:


        // INSERT_RANDOM_CODE_START
        val ^= v2;                // XOR
        val -= (val ^ 97);                   // XOR and -
        val = val >> (val & 3); // shift right
        val = ~val;                          // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 215:


        // INSERT_RANDOM_CODE_START
        val ^= v2;                   // XOR
        val = val & v2; // AND
        val = val << (val & 3);    // shift left
        val *= val;                             // *
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 216:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, val); // rotate  bits by random
        val = ~val;                             // binary NOT operator
        val -= (val ^ 97);                      // XOR and -
        val = val & v2; // AND
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 217:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 5);  // rotate  bits by 5
        val += val;               // +
        val = std::rotl(val, 1);  // rotate  bits by 1
        val ^= std::rotl(val, 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 218:


        // INSERT_RANDOM_CODE_START
        val = reverse8(val); // reverse bits
        val = ~val;          // binary NOT operator
        val *= val;          // *
        val -= (val ^ 97);   // XOR and -
                                                       // INSERT_RANDOM_CODE_END
      break;

    case 219:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 4);               // rotate  bits by 4
        val = std::rotl(val, 3);                // rotate  bits by 3
        val = val & v2; // AND
        val = reverse8(val);                    // reverse bits
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 220:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 1);             // rotate  bits by 1
        val = val << (val & 3); // shift left
        val = reverse8(val);                 // reverse bits
        val = val << (val & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 221:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 5); // rotate  bits by 5
        val ^= v2;    // XOR
        val = ~val;              // binary NOT operator
        val = reverse8(val);     // reverse bits
                                                           // INSERT_RANDOM_CODE_END
      break;

    case 222:


        // INSERT_RANDOM_CODE_START
        val = val >> (val & 3); // shift right
        val = val << (val & 3); // shift left
        val ^= v2;                // XOR
        val *= val;                          // *
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 223:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 3);                // rotate  bits by 3
        val ^= v2;                   // XOR
        val = std::rotl(val, val); // rotate  bits by random
        val -= (val ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 224:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val = std::rotl(val, 4);  // rotate  bits by 1
        // val = std::rotl(val, 3);             // rotate  bits by 3
        val = val << (val & 3); // shift left
                                                                       //
      break;

    case 225:


        // INSERT_RANDOM_CODE_START
        val = ~val;                          // binary NOT operator
        val = val >> (val & 3); // shift right
        val = reverse8(val);                 // reverse bits
        val = std::rotl(val, 3);             // rotate  bits by 3
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 226:


        // INSERT_RANDOM_CODE_START
        val = reverse8(val);  // reverse bits
        val -= (val ^ 97);    // XOR and -
        val *= val;           // *
        val ^= v2; // XOR
                                                        // INSERT_RANDOM_CODE_END
      break;

    case 227:


        // INSERT_RANDOM_CODE_START
        val = ~val;                             // binary NOT operator
        val = val << (val & 3);    // shift left
        val -= (val ^ 97);                      // XOR and -
        val = val & v2; // AND
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 228:


        // INSERT_RANDOM_CODE_START
        val += val;                          // +
        val = val >> (val & 3); // shift right
        val += val;                          // +
        val ^= (byte)bitTable[val];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 229:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 3);                // rotate  bits by 3
        val = std::rotl(val, val); // rotate  bits by random
        val ^= std::rotl(val, 2);               // rotate  bits by 2
        val ^= (byte)bitTable[val];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 230:


        // INSERT_RANDOM_CODE_START
        val *= val;                             // *
        val = val & v2; // AND
        val = std::rotl(val, val); // rotate  bits by random
        val = std::rotl(val, val); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 231:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 3);             // rotate  bits by 3
        val = val >> (val & 3); // shift right
        val ^= v2;                // XOR
        val = reverse8(val);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 232:


        // INSERT_RANDOM_CODE_START
        val *= val;               // *
        val *= val;               // *
        val ^= std::rotl(val, 4); // rotate  bits by 4
        val = std::rotl(val, 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 233:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 1);    // rotate  bits by 1
        val ^= (byte)bitTable[val]; // ones count bits
        val = std::rotl(val, 3);    // rotate  bits by 3
        val ^= (byte)bitTable[val]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 234:


        // INSERT_RANDOM_CODE_START
        val = val & v2; // AND
        val *= val;                             // *
        val = val >> (val & 3);    // shift right
        val ^= v2;                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 235:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val *= val;               // *
        val = std::rotl(val, 3);  // rotate  bits by 3
        val = ~val;               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 236:


        // INSERT_RANDOM_CODE_START
        val ^= v2;                   // XOR
        val += val;                             // +
        val = val & v2; // AND
        val -= (val ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 237:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 5);             // rotate  bits by 5
        val = val << (val & 3); // shift left
        val ^= std::rotl(val, 2);            // rotate  bits by 2
        val = std::rotl(val, 3);             // rotate  bits by 3
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 238:


        // INSERT_RANDOM_CODE_START
        val += val;              // +
        val += val;              // +
        val = std::rotl(val, 3); // rotate  bits by 3
        val -= (val ^ 97);       // XOR and -
                                                           // INSERT_RANDOM_CODE_END
      break;

    case 239:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 6); // rotate  bits by 5
        // val = std::rotl(val, 1); // rotate  bits by 1
        val *= val;                             // *
        val = val & v2; // AND
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 240:


        // INSERT_RANDOM_CODE_START
        val = ~val;                             // binary NOT operator
        val += val;                             // +
        val = val & v2; // AND
        val = val << (val & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 241:


        // INSERT_RANDOM_CODE_START
        val ^= std::rotl(val, 4);   // rotate  bits by 4
        val ^= (byte)bitTable[val]; // ones count bits
        val ^= v2;       // XOR
        val = std::rotl(val, 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 242:


        // INSERT_RANDOM_CODE_START
        val += val;           // +
        val += val;           // +
        val -= (val ^ 97);    // XOR and -
        val ^= v2; // XOR
                                                        // INSERT_RANDOM_CODE_END
      break;

    case 243:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 5);    // rotate  bits by 5
        val ^= std::rotl(val, 2);   // rotate  bits by 2
        val ^= (byte)bitTable[val]; // ones count bits
        val = std::rotl(val, 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 244:


        // INSERT_RANDOM_CODE_START
        val = ~val;               // binary NOT operator
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val = reverse8(val);      // reverse bits
        val = std::rotl(val, 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 245:


        // INSERT_RANDOM_CODE_START
        val -= (val ^ 97);                   // XOR and -
        val = std::rotl(val, 5);             // rotate  bits by 5
        val ^= std::rotl(val, 2);            // rotate  bits by 2
        val = val >> (val & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 246:


        // INSERT_RANDOM_CODE_START
        val += val;                          // +
        val = std::rotl(val, 1);             // rotate  bits by 1
        val = val >> (val & 3); // shift right
        val += val;                          // +
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 247:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 5);  // rotate  bits by 5
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val = std::rotl(val, 5);  // rotate  bits by 5
        val = ~val;               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      break;

    case 248:


        // INSERT_RANDOM_CODE_START
        val = ~val;                 // binary NOT operator
        val -= (val ^ 97);          // XOR and -
        val ^= (byte)bitTable[val]; // ones count bits
        val = std::rotl(val, 5);    // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 249:


        // INSERT_RANDOM_CODE_START
        val = reverse8(val);                    // reverse bits
        val ^= std::rotl(val, 4);               // rotate  bits by 4
        val ^= std::rotl(val, 4);               // rotate  bits by 4
        val = std::rotl(val, val); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 250:


        // INSERT_RANDOM_CODE_START
        val = val & v2; // AND
        val = std::rotl(val, val); // rotate  bits by random
        val ^= (byte)bitTable[val];             // ones count bits
        val ^= std::rotl(val, 4);               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      break;

    case 251:


        // INSERT_RANDOM_CODE_START
        val += val;                 // +
        val ^= (byte)bitTable[val]; // ones count bits
        val = reverse8(val);        // reverse bits
        val ^= std::rotl(val, 2);   // rotate  bits by 2
                                                              // INSERT_RANDOM_CODE_END
      break;

    case 252:


        // INSERT_RANDOM_CODE_START
        val = reverse8(val);                 // reverse bits
        val ^= std::rotl(val, 4);            // rotate  bits by 4
        val ^= std::rotl(val, 2);            // rotate  bits by 2
        val = val << (val & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      break;

    case 253:


        // INSERT_RANDOM_CODE_START
        val = std::rotl(val, 3);  // rotate  bits by 3
        val ^= std::rotl(val, 2); // rotate  bits by 2
        val ^= v2;     // XOR
        val = std::rotl(val, 3);  // rotate  bits by 3
        // INSERT_RANDOM_CODE_END

        // worker.prev_lhash = worker.lhash + worker.prev_lhash;
        // worker.lhash = XXHash64::hash(worker.step_3, worker.pos2,0);
      break;

    case 254:
    case 255:
      // RC4_set_key(&worker.key, 256,  worker.step_3);
// worker.step_3 = highwayhash.Sum(worker.step_3[:], worker.step_3[:])


        // INSERT_RANDOM_CODE_START
        val ^= static_cast<uint8_t>(std::bitset<8>(val).count()); // ones count bits
        val = std::rotl(val, 3);                                  // rotate  bits by 3
        val ^= std::rotl(val, 2);                                 // rotate  bits by 2
        val = std::rotl(val, 3);                                  // rotate  bits by 3
                                                                                            // INSERT_RANDOM_CODE_END
      break;

    default:
      break;
    }
}