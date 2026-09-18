#pragma once

// String-only templates: these expand inside ONE volatile assembly block.
// Splitting the stages into separate asm statements would expose live physical
// registers to the compiler and change the qualified allocation/schedule.
//
// D, B and A are inclusive physical VGPR ranges (for example 0:7).
// B precedes A deliberately: this is the gfx1100 SourceSwap=1 ownership.
#define IRIS_NATIVE128_WMMA(D, B, A) \
    "v_wmma_i32_16x16x16_iu8 v[" #D "], v[" #B "], v[" #A "], v[" #D \
    "] neg_lo:[1,1,0] clamp\n\t"

// One K=16 slice updates the sixteen accumulator fragments, in fixed order.
// Each group of four shares B and walks the four A fragments.
#define IRIS_NATIVE128_WMMA_GRID(B0, B1, B2, B3, A0, A1, A2, A3) \
    IRIS_NATIVE128_WMMA(0:7, B0, A0) \
    IRIS_NATIVE128_WMMA(8:15, B0, A1) \
    IRIS_NATIVE128_WMMA(16:23, B0, A2) \
    IRIS_NATIVE128_WMMA(24:31, B0, A3) \
    \
    IRIS_NATIVE128_WMMA(32:39, B1, A0) \
    IRIS_NATIVE128_WMMA(40:47, B1, A1) \
    IRIS_NATIVE128_WMMA(48:55, B1, A2) \
    IRIS_NATIVE128_WMMA(56:63, B1, A3) \
    \
    IRIS_NATIVE128_WMMA(64:71, B2, A0) \
    IRIS_NATIVE128_WMMA(72:79, B2, A1) \
    IRIS_NATIVE128_WMMA(80:87, B2, A2) \
    IRIS_NATIVE128_WMMA(88:95, B2, A3) \
    \
    IRIS_NATIVE128_WMMA(96:103, B3, A0) \
    IRIS_NATIVE128_WMMA(104:111, B3, A1) \
    IRIS_NATIVE128_WMMA(112:119, B3, A2) \
    IRIS_NATIVE128_WMMA(120:127, B3, A3)


// LDS A bytes arrive as two interleaved halfword streams. Pack them only
// after the caller's wait; the wait placement is part of the schedule.
#define IRIS_NATIVE128_READ_A_PAIR(D, P, O) \
    "ds_load_u8 v" #D ", v218 offset:(" #O ")\n\t" \
    "ds_load_u8 v" #P ", v218 offset:(" #O "+128)\n\t" \
    "ds_load_u8_d16_hi v" #D ", v218 offset:(" #O "+256)\n\t" \
    "ds_load_u8_d16_hi v" #P ", v218 offset:(" #O "+384)\n\t"

#define IRIS_NATIVE128_PACK_A(D, P) \
    "v_lshl_or_b32 v" #D ", v" #P ", 8, v" #D "\n\t"

#define IRIS_NATIVE128_FETCH_NEXT \
    "buffer_load_b128 v[202:205], v197, %[ra], %[oa] offen\n\t" \
    "buffer_load_b128 v[206:209], v198, %[ra], %[oa] offen\n\t" \
    "buffer_load_b128 v[210:213], v199, %[rb], %[ob] offen\n\t" \
    "buffer_load_b128 v[214:217], v220, %[rb], %[ob] offen\n\t"

// Preserve the progressive VMEM waits: each completed vector can enter LDS
// while later vectors are still arriving. BANK is 0 or 16384 bytes.
#define IRIS_NATIVE128_STAGE_INPUTS(BANK) \
    "s_waitcnt vmcnt(3)\n\t" \
    "ds_store_b128 v195, v[202:205] offset:(" #BANK ")\n\t" \
    "s_waitcnt vmcnt(2)\n\t" \
    "ds_store_b128 v195, v[206:209] offset:(" #BANK "+512)\n\t" \
    "s_waitcnt vmcnt(1)\n\t" \
    "ds_store_b128 v196, v[210:213] offset:(" #BANK ")\n\t" \
    "s_waitcnt vmcnt(0)\n\t" \
    "ds_store_b128 v196, v[214:217] offset:(" #BANK "+576)\n\t"

#define IRIS_NATIVE128_READ_B_LOW(BANK) \
    "ds_load_b128 v[163:166], v219 offset:(" #BANK ")\n\t" \
    "ds_load_b128 v[167:170], v219 offset:(" #BANK "+32)\n\t" \
    "ds_load_b128 v[171:174], v219 offset:(" #BANK "+64)\n\t" \
    "ds_load_b128 v[175:178], v219 offset:(" #BANK "+96)\n\t"

#define IRIS_NATIVE128_READ_B_HIGH(BANK) \
    "ds_load_b128 v[179:182], v219 offset:(" #BANK "+16)\n\t" \
    "ds_load_b128 v[183:186], v219 offset:(" #BANK "+48)\n\t" \
    "ds_load_b128 v[187:190], v219 offset:(" #BANK "+80)\n\t" \
    "ds_load_b128 v[191:194], v219 offset:(" #BANK "+112)\n\t"

// First K=16 slice: four groups of four A fragments.
#define IRIS_NATIVE128_READ_A_LOW(BANK) \
    IRIS_NATIVE128_READ_A_PAIR(130, 221, BANK+0) \
    IRIS_NATIVE128_READ_A_PAIR(131, 222, BANK+512) \
    IRIS_NATIVE128_READ_A_PAIR(132, 223, BANK+1024) \
    IRIS_NATIVE128_READ_A_PAIR(133, 224, BANK+1536) \
    IRIS_NATIVE128_READ_A_PAIR(134, 225, BANK+32) \
    IRIS_NATIVE128_READ_A_PAIR(135, 226, BANK+544) \
    IRIS_NATIVE128_READ_A_PAIR(136, 227, BANK+1056) \
    IRIS_NATIVE128_READ_A_PAIR(137, 228, BANK+1568) \
    IRIS_NATIVE128_READ_A_PAIR(138, 229, BANK+64) \
    IRIS_NATIVE128_READ_A_PAIR(139, 230, BANK+576) \
    IRIS_NATIVE128_READ_A_PAIR(140, 231, BANK+1088) \
    IRIS_NATIVE128_READ_A_PAIR(141, 232, BANK+1600) \
    IRIS_NATIVE128_READ_A_PAIR(142, 233, BANK+96) \
    IRIS_NATIVE128_READ_A_PAIR(143, 234, BANK+608) \
    IRIS_NATIVE128_READ_A_PAIR(144, 235, BANK+1120) \
    IRIS_NATIVE128_READ_A_PAIR(145, 236, BANK+1632)

// Second K=16 slice before the interleaved next-tile global loads.
#define IRIS_NATIVE128_READ_A_HIGH_HEAD(BANK) \
    IRIS_NATIVE128_READ_A_PAIR(146, 237, BANK+2048) \
    IRIS_NATIVE128_READ_A_PAIR(147, 238, BANK+2560) \
    IRIS_NATIVE128_READ_A_PAIR(148, 239, BANK+3072) \
    IRIS_NATIVE128_READ_A_PAIR(149, 240, BANK+3584) \
    IRIS_NATIVE128_READ_A_PAIR(150, 241, BANK+2080) \
    IRIS_NATIVE128_READ_A_PAIR(151, 242, BANK+2592) \
    IRIS_NATIVE128_READ_A_PAIR(152, 243, BANK+3104) \
    IRIS_NATIVE128_READ_A_PAIR(153, 244, BANK+3616)

// Resume the second slice after its split ninth pair.
#define IRIS_NATIVE128_READ_A_HIGH_TAIL(BANK) \
    IRIS_NATIVE128_READ_A_PAIR(155, 246, BANK+2624) \
    IRIS_NATIVE128_READ_A_PAIR(156, 247, BANK+3136) \
    IRIS_NATIVE128_READ_A_PAIR(157, 248, BANK+3648) \
    IRIS_NATIVE128_READ_A_PAIR(158, 249, BANK+2144) \
    IRIS_NATIVE128_READ_A_PAIR(159, 250, BANK+2656) \
    IRIS_NATIVE128_READ_A_PAIR(160, 251, BANK+3168) \
    IRIS_NATIVE128_READ_A_PAIR(161, 252, BANK+3680)

// Drain path: no next-tile fetch to interleave.
#define IRIS_NATIVE128_READ_A_HIGH(BANK) \
    IRIS_NATIVE128_READ_A_PAIR(146, 237, BANK+2048) \
    IRIS_NATIVE128_READ_A_PAIR(147, 238, BANK+2560) \
    IRIS_NATIVE128_READ_A_PAIR(148, 239, BANK+3072) \
    IRIS_NATIVE128_READ_A_PAIR(149, 240, BANK+3584) \
    IRIS_NATIVE128_READ_A_PAIR(150, 241, BANK+2080) \
    IRIS_NATIVE128_READ_A_PAIR(151, 242, BANK+2592) \
    IRIS_NATIVE128_READ_A_PAIR(152, 243, BANK+3104) \
    IRIS_NATIVE128_READ_A_PAIR(153, 244, BANK+3616) \
    IRIS_NATIVE128_READ_A_PAIR(154, 245, BANK+2112) \
    IRIS_NATIVE128_READ_A_PAIR(155, 246, BANK+2624) \
    IRIS_NATIVE128_READ_A_PAIR(156, 247, BANK+3136) \
    IRIS_NATIVE128_READ_A_PAIR(157, 248, BANK+3648) \
    IRIS_NATIVE128_READ_A_PAIR(158, 249, BANK+2144) \
    IRIS_NATIVE128_READ_A_PAIR(159, 250, BANK+2656) \
    IRIS_NATIVE128_READ_A_PAIR(160, 251, BANK+3168) \
    IRIS_NATIVE128_READ_A_PAIR(161, 252, BANK+3680)

// Combine the two byte streams into signed-int8 WMMA inputs.
#define IRIS_NATIVE128_PACK_A_LOW() \
    IRIS_NATIVE128_PACK_A(130, 221) \
    IRIS_NATIVE128_PACK_A(131, 222) \
    IRIS_NATIVE128_PACK_A(132, 223) \
    IRIS_NATIVE128_PACK_A(133, 224) \
    IRIS_NATIVE128_PACK_A(134, 225) \
    IRIS_NATIVE128_PACK_A(135, 226) \
    IRIS_NATIVE128_PACK_A(136, 227) \
    IRIS_NATIVE128_PACK_A(137, 228) \
    IRIS_NATIVE128_PACK_A(138, 229) \
    IRIS_NATIVE128_PACK_A(139, 230) \
    IRIS_NATIVE128_PACK_A(140, 231) \
    IRIS_NATIVE128_PACK_A(141, 232) \
    IRIS_NATIVE128_PACK_A(142, 233) \
    IRIS_NATIVE128_PACK_A(143, 234) \
    IRIS_NATIVE128_PACK_A(144, 235) \
    IRIS_NATIVE128_PACK_A(145, 236)

// Combine the two byte streams into signed-int8 WMMA inputs.
#define IRIS_NATIVE128_PACK_A_HIGH() \
    IRIS_NATIVE128_PACK_A(146, 237) \
    IRIS_NATIVE128_PACK_A(147, 238) \
    IRIS_NATIVE128_PACK_A(148, 239) \
    IRIS_NATIVE128_PACK_A(149, 240) \
    IRIS_NATIVE128_PACK_A(150, 241) \
    IRIS_NATIVE128_PACK_A(151, 242) \
    IRIS_NATIVE128_PACK_A(152, 243) \
    IRIS_NATIVE128_PACK_A(153, 244) \
    IRIS_NATIVE128_PACK_A(154, 245) \
    IRIS_NATIVE128_PACK_A(155, 246) \
    IRIS_NATIVE128_PACK_A(156, 247) \
    IRIS_NATIVE128_PACK_A(157, 248) \
    IRIS_NATIVE128_PACK_A(158, 249) \
    IRIS_NATIVE128_PACK_A(159, 250) \
    IRIS_NATIVE128_PACK_A(160, 251) \
    IRIS_NATIVE128_PACK_A(161, 252)
