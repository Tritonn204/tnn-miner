#ifndef SA_FAST
#define SA_FAST

#include <stdint.h>
#include <assert.h>

#include <array>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#endif

#ifdef linux
#include <arpa/inet.h>
#endif

inline const uint32_t MAX_LENGTH = (256 * 384) - 1; // this is the maximum

struct ScratchData
{
  unsigned char *hash[32];
  std::array<byte, MAX_LENGTH+64> data;
  uint16_t *stage1_result[MAX_LENGTH + 1];
  unsigned char *stage1_result_bytes[MAX_LENGTH * 2];
  uint32_t indices[MAX_LENGTH + 1];
  uint32_t tmp_indices[MAX_LENGTH + 1];
  int32_t *sa[MAX_LENGTH];
  unsigned char *sa_bytes[MAX_LENGTH * 4];

  ScratchData()
  {
    *hash = new byte[32];
    *stage1_result = new uint16_t[MAX_LENGTH + 1];
    *stage1_result_bytes = new byte[MAX_LENGTH * 2];
    *sa = new int32_t[MAX_LENGTH];
    *sa_bytes = new byte[MAX_LENGTH + 4];
  }
};

void fix(unsigned char *v, uint32_t *indices, int i);
void sort_indices(uint32_t N, unsigned char *v, uint16_t *output, ScratchData &d);
void text_32_0alloc(unsigned char *text, int32_t *sa, int tLen, int sLen);

#endif