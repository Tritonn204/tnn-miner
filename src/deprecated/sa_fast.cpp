#include <iostream>
#include "sa_fast.h"

#include "sais2.h"

using byte = unsigned char;


void fix(byte *v, uint32_t *indices, int i) {
  uint32_t prev_t = indices[i];
  uint32_t t = indices[i+1];

  uint32_t data_a = htonl(v[(t & 0xffff)+2]);
  if (data_a <= htonl(v[(prev_t & 0xffff)+2])) {
    uint32_t t2 = prev_t;
    int j = i;
    while(true) {
      indices[j+2] = prev_t;
      j--;
      if (j < 0) break;
      prev_t = indices[j];
      if ((t^prev_t) <= 0xffff && data_a <= htonl(v[(prev_t & 0xffff)+2])) {
        continue;
      } else {
        break;
      }
    }
    indices[j+1] = t;
    t = t2;
  }
}

void sort_indices(uint32_t N, byte *v, uint16_t *output, ScratchData &d) {
  uint16_t byte_counters[2][256];
  uint16_t counters[2][256];

  v[N] = 0;
  v[N+1] = 0;

  uint32_t *indices = d.indices;
  uint32_t *tmp_indices = d.tmp_indices;

  for (int i = 0; i < N; i++) {
    byte_counters[1][v[i]]++;
  }
  *byte_counters[0] = *byte_counters[1];
  byte_counters[0][v[0]]--;

  counters[0][0] = (uint16_t)byte_counters[0][0];
  counters[1][0] = (uint16_t)byte_counters[1][0] - 1;

  uint16_t c0 = counters[0][0];
  uint16_t c1 = counters[1][0];

  for (int i = 0; i < 256; i++) {
    c0 += (uint16_t)byte_counters[0][i];
    c1 += (uint16_t)byte_counters[1][i];

    counters[0][i] = c0;
    counters[1][i] = c1;
  }

  uint16_t *counters0 = counters[0];
  {
    uint32_t byte0 = (uint32_t)v[N-1];
    tmp_indices[counters0[0]] = byte0<<24 | uint32_t(N-1);
    counters0[0]--;
  }

  for (int i = (int)N-1; i >= 1; i--) {
    uint32_t byte0 = (uint32_t)v[i-1];
    uint32_t byte1 = (uint32_t)v[i];
    tmp_indices[counters0[v[i]]] = byte0<<24 | byte1<<16 | (uint32_t)(i-1);
    counters0[v[i]]--;
  }

  uint16_t *counters1 = counters[1];
  for (int i = (int)N-1; i >= 0; i--) {
    uint32_t data = tmp_indices[i];
    uint16_t tmp = counters1[data>>24];
    counters1[data>>24]--;
    indices[tmp] = data;
  }

  for (int i = 1; i < (int)N; i++) {
    if (indices[i-1]&0xffff0000 == indices[i]&0xffff0000) {
      fix(v, indices, i-1);
    }
  }

  for (int i = 0; i < N; i++) {
    output[i] = (uint16_t)indices[i];
  }
}

void text_32_0alloc(byte *text, int32_t *sa, int tLen, int sLen) {
  if ((int)(int32_t)tLen != tLen || tLen != sLen) {
    std::cout << "suffixarray: misuse of text_16" << std::endl;
    return;
  }
  for (int i = 0; i < sLen; i++) {
    sa[i] = 0;
  }
  int32_t memory[2*256];
  sais_8_32(text, 256, sa, memory, tLen, sLen, 2*256);
}
