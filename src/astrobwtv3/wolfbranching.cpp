#include "astrobwtv3.h"
#include <inttypes.h>
#include <stdio.h>

// The base for the following code was contributed by @Wolf9466 on Discord

// Last instruction is a special case, and duplicated.
alignas(32) uint32_t CodeLUT[256] =
{
	0x090F020A, 0x060B0500, 0x09080609, 0x0A0D030B, 0x04070A01, 0x09030607, 0x060D0401, 0x000A0904,
	0x040F0F06, 0x030E070C, 0x04020D02, 0x0B0F050A, 0x0C020C04, 0x0B03070F, 0x07060206, 0x0C060501,
	0x0E020B04, 0x03020F04, 0x0E0D0B0F, 0x010F0600, 0x0503080C, 0x0B030005, 0x0608020B, 0x0D0B0905,
	0x00070E0F, 0x090D0A01, 0x02090008, 0x0F050E0F, 0x0600000F, 0x02030700, 0x050E0F06, 0x040C0602,
	0x0C080D0C, 0x0A0E0802, 0x01060601, 0x00040B03, 0x090B0C0B, 0x0A070702, 0x070D090A, 0x0C030705,
	0x0A030903, 0x0F010D0E, 0x0B0D0C0A, 0x05000501, 0x09090D0A, 0x0F0F0509, 0x09000F0E, 0x0F050F06,
	0x0A04040F, 0x0900080E, 0x080D000B, 0x030E0E0F, 0x0A070409, 0x00090E0E, 0x08030404, 0x080E0E0B,
	0x0C02040B, 0x0A0F0D08, 0x080C0500, 0x0B020A04, 0x0304020D, 0x0F060D0F, 0x05040C00, 0x0F090100,
	0x03080E02, 0x0F0D0C02, 0x0C080E0B, 0x0B090C0F, 0x05040E03, 0x00020807, 0x0302070E, 0x0F040206,
	0x08090306, 0x09080F01, 0x020D0805, 0x0209050E, 0x0A0C0F07, 0x0D000609, 0x0A080201, 0x0E0C0002,
	0x0A060005, 0x0E060A09, 0x03040407, 0x06080D08, 0x010B0600, 0x07030A06, 0x0E0A0E04, 0x000D0E00,
	0x0C0B0204, 0x0002040C, 0x080F0B07, 0x09050E08, 0x09040905, 0x0C020500, 0x0B0A0506, 0x0B040F0F,
	0x0C0C090B, 0x0B060907, 0x0E06070E, 0x0E010807, 0x0A060809, 0x07090704, 0x0D01000D, 0x0B08030A,
	0x08090F00, 0x060D0A0C, 0x080E0B02, 0x070C0F0B, 0x0304050C, 0x020A030C, 0x000C0C07, 0x02080207,
	0x0D040F01, 0x0F0B0904, 0x0B080A04, 0x0A0F050D, 0x05030906, 0x060D0605, 0x0700060F, 0x080C0403,
	0x0C020308, 0x07000902, 0x0E0A0F0C, 0x05040D0D, 0x0C0C0304, 0x080C0007, 0x0D0B0F08, 0x06020503,
	0x0A0C0C0F, 0x04090907, 0x070A0B0E, 0x010B0902, 0x05080F0C, 0x030F0C06, 0x040E0B05, 0x070C0008,
	0x0701030F, 0x0F07080A, 0x03030001, 0x0F0D0C0D, 0x0B0C030F, 0x0B010900, 0x050F080C, 0x050D0706,
	0x0A06040A, 0x080E0C0E, 0x05060509, 0x04060E02, 0x050F0601, 0x03080100, 0x06060605, 0x00060206,
	0x0704060C, 0x0B0D0404, 0x0F040309, 0x01030903, 0x07070D0B, 0x07060A0B, 0x090D000B, 0x01030A03,
	0x07080B0D, 0x03030F0A, 0x02080C01, 0x06010E0B, 0x02090104, 0x0E030600, 0x0D000C04, 0x04040207,
	0x0A050A0B, 0x0B060E05, 0x01080102, 0x0D010908, 0x0E01060B, 0x04060200, 0x040A0909, 0x0D01020F,
	0x0302030F, 0x090C0C05, 0x0500040B, 0x0C000708, 0x070E0301, 0x04060C0F, 0x030B0F0E, 0x00010102,
	0x06020F03, 0x040E0F07, 0x0C0E0107, 0x0304000D, 0x0E090E0E, 0x0F0E0301, 0x0F07050C, 0x000D0A07,
	0x00060002, 0x05060A0B, 0x050A0605, 0x090C030E, 0x0D08060B, 0x0E0A0202, 0x0707080B, 0x04000203,
	0x07090808, 0x0D0C0E04, 0x03040A0F, 0x03050B0A, 0x0F0C0A03, 0x090E0600, 0x0E080809, 0x0F0D0909,
	0x0000070D, 0x0F080901, 0x0C0A0F04, 0x0E00010A, 0x0A0C0303, 0x00060D01, 0x03010704, 0x03050602,
	0x0A040105, 0x0F000B0E, 0x08040201, 0x0E0D0508, 0x0B060806, 0x0F030408, 0x07060302, 0x0D030A01,
	0x0C0B0D06, 0x0407080D, 0x08010203, 0x04060105, 0x00070009, 0x0D0A0C09, 0x02050A0A, 0x0D070308,
	0x02020E0F, 0x0B090D09, 0x05020703, 0x0C020D04, 0x03000501, 0x0F060C0D, 0x00000D01, 0x0F0B0205,
	0x04000506, 0x0E09030B, 0x00000103, 0x0F0C090B, 0x040C080F, 0x010F0C07, 0x000B0700, 0x0F0C0F04,
	0x0401090F, 0x080E0E0A, 0x050A090E, 0x0009080C, 0x080E0C06, 0x0D0C030D, 0x090D0C0D, 0x090D0C0D,
};


void wolfBranch_avx2(__m256i &in, uint8_t pos2val, uint32_t opcode, workerData &worker)
{
  for (int i = 3; i >= 0; --i)
  {
    uint8_t insn = (opcode >> (i << 3)) & 0xFF;
    switch (insn)
    {
    case 0:
      in = _mm256_add_epi8(in, in);
      break;
    case 1:
      in = _mm256_sub_epi8(in, _mm256_xor_si256(in, _mm256_set1_epi8(97)));
      break;
    case 2:
      in = _mm256_mul_epi8(in, in);
      break;
    case 3:
      in = _mm256_xor_si256(in, _mm256_set1_epi8(pos2val));
      break;
    case 4:
      in = _mm256_xor_si256(in, _mm256_set1_epi64x(-1LL));
      break;
    case 5:
      in = _mm256_and_si256(in, _mm256_set1_epi8(pos2val));
      break;
    case 6:
      in = _mm256_sllv_epi8(in,_mm256_and_si256(in,vec_3));
      break;
    case 7:
      in = _mm256_srlv_epi8(in,_mm256_and_si256(in,vec_3));
      break;
    case 8:
      in = _mm256_reverse_epi8(in);
      break;
    case 9:
      in = _mm256_xor_si256(in, popcnt256_epi8(in));
      break;
    case 10:
      in = _mm256_rolv_epi8(in, in);
      break;
    case 11:
      in = _mm256_rol_epi8(in, 1);
      break;
    case 12:
      in = _mm256_xor_si256(in, _mm256_rol_epi8(in, 2));
      break;
    case 13:
      in = _mm256_rol_epi8(in, 3);
      break;
    case 14:
      in = _mm256_xor_si256(in, _mm256_rol_epi8(in, 4));
      break;
    case 15:
      in = _mm256_rol_epi8(in, 5);
      break;
    }      
  }
}

uint8_t wolfBranch(uint8_t val, uint8_t pos2val, uint32_t opcode)
{
  for (int i = 3; i >= 0; --i)
  {
    uint8_t insn = (opcode >> (i << 3)) & 0xFF;
    switch (insn)
    {
    case 0:
      val += val;
      break;
    case 1:
      val -= (val ^ 97);
      break;
    case 2:
      val *= val;
      break;
    case 3:
      val ^= pos2val;
      break;
    case 4:
      val = ~val;
      break;
    case 5:
      val &= pos2val;
      break;
    case 6:
      val <<= (val & 3);
      break;
    case 7:
      val >>= (val & 3);
      break;
    case 8:
      val = (((val & 0xAA) >> 1) | ((val & 0x55) << 1));
      val = (((val & 0xCC) >> 2) | ((val & 0x33) << 2));
      val = (((val & 0xF0) >> 4) | ((val & 0x0F) << 4));
      break;
    case 9:
      val ^= (uint8_t)__builtin_popcount(val);
      break;
    case 10:
      val = rl8(val, val);
      break;
    case 11:
      val = rl8(val, 1);
      break;
    case 12:
      val ^= rl8(val, 2);
      break;
    case 13:
      val = rl8(val, 3);
      break;
    case 14:
      val ^= rl8(val, 4);
      break;
    case 15:
      val = rl8(val, 5);
      break;
    }
  }

  return (val);
}

// __attribute__((target("avx2")))
void wolfPermute(uint8_t *in, uint8_t *out, uint16_t op, uint8_t pos1, uint8_t pos2, workerData &worker)
{
  // printf("AVX2 WOLF\n");
	uint32_t Opcode = CodeLUT[op];

  __m256i data = _mm256_loadu_si256((__m256i*)&in[pos1]);
  __m256i old = data;

  wolfBranch_avx2(data, in[pos2], Opcode, worker);
  data = _mm256_blendv_epi8(old, data, genMask(pos2 - pos1));

  _mm256_storeu_si256((__m256i*)&out[pos1], data);
}

// __attribute__((target("default")))
// void wolfPermute(uint8_t *in, uint8_t *out, uint16_t op, uint8_t pos1, uint8_t pos2)
// {
// 	uint32_t Opcode = CodeLUT[op];

// 	for(int i = pos1; i < pos2; ++i)
// 	{
// 		out[i] = wolfBranch(in[i], in[pos2], Opcode);
// 	}		
// }