#include "asm_templates.hpp"

// Keep waits, LDS handoffs, and register constraints in this single asm block.
// The templates only remove repetition; they do not reschedule instructions.
template <class Loads> __device__ __forceinline__ void slot_raw(
    I8 (&acc)[16], const Loads& loads, unsigned tiles, unsigned stride,
    unsigned read_a, unsigned read_b, unsigned store_a, unsigned store_b) {
    unsigned offset_a, offset_b, remaining;

    asm volatile(
        // Prime bank 0 before entering the ping/pong loop.
        "s_mov_b32 %[oa], 0\n\t"
        "s_mov_b32 %[ob], 0\n\t"
        "s_mov_b32 %[left], %[tiles]\n\t"
        IRIS_NATIVE128_FETCH_NEXT
        IRIS_NATIVE128_STAGE_INPUTS(0)

        "s_waitcnt lgkmcnt(0)\n\t"
        "s_barrier\n\t"
        IRIS_NATIVE128_READ_A_LOW(0)
        IRIS_NATIVE128_READ_B_LOW(0)

        // Pipeline boundary: branch labels remain local to this asm instance.
        "cross_k_loop_%=:\n\t"
        "s_cmp_eq_u32 %[left], 1\n\t"
        "s_cbranch_scc1 cross_k_drain0_%=\n\t"
        IRIS_NATIVE128_READ_A_HIGH_HEAD(0)
        "ds_load_u8 v154, v218 offset:2112\n\t"
        "ds_load_u8 v245, v218 offset:2240\n\t"
        "s_add_u32 %[oa], %[oa], %[stride]\n\t"
        "s_add_u32 %[ob], %[ob], 32\n\t"
        IRIS_NATIVE128_FETCH_NEXT
        "ds_load_u8_d16_hi v154, v218 offset:2368\n\t"
        "ds_load_u8_d16_hi v245, v218 offset:2496\n\t"
        IRIS_NATIVE128_READ_A_HIGH_TAIL(0)
        IRIS_NATIVE128_READ_B_HIGH(0)
        IRIS_NATIVE128_STAGE_INPUTS(16384)

        "s_waitcnt lgkmcnt(15)\n\t"
        IRIS_NATIVE128_PACK_A_LOW()
        "s_nop 1\n\t"

        // Consume this K=16 slice without changing the physical fragment order.
        IRIS_NATIVE128_WMMA_GRID(163:166, 167:170, 171:174, 175:178,
                                 130:133, 134:137, 138:141, 142:145)

        "s_waitcnt lgkmcnt(0)\n\t"
        "s_barrier\n\t"
        IRIS_NATIVE128_READ_A_LOW(16384)
        IRIS_NATIVE128_READ_B_LOW(16384)
        "s_waitcnt lgkmcnt(15)\n\t"
        IRIS_NATIVE128_PACK_A_HIGH()
        "s_nop 1\n\t"

        // Consume this K=16 slice without changing the physical fragment order.
        IRIS_NATIVE128_WMMA_GRID(179:182, 183:186, 187:190, 191:194,
                                 146:149, 150:153, 154:157, 158:161)

        "s_sub_u32 %[left], %[left], 1\n\t"
        "s_cmp_eq_u32 %[left], 1\n\t"
        "s_cbranch_scc1 cross_k_drain1_%=\n\t"
        IRIS_NATIVE128_READ_A_HIGH_HEAD(16384)
        "ds_load_u8 v154, v218 offset:18496\n\t"
        "ds_load_u8 v245, v218 offset:18624\n\t"
        "s_add_u32 %[oa], %[oa], %[stride]\n\t"
        "s_add_u32 %[ob], %[ob], 32\n\t"
        IRIS_NATIVE128_FETCH_NEXT
        "ds_load_u8_d16_hi v154, v218 offset:18752\n\t"
        "ds_load_u8_d16_hi v245, v218 offset:18880\n\t"
        IRIS_NATIVE128_READ_A_HIGH_TAIL(16384)
        IRIS_NATIVE128_READ_B_HIGH(16384)
        IRIS_NATIVE128_STAGE_INPUTS(0)
        "s_waitcnt lgkmcnt(15)\n\t"
        IRIS_NATIVE128_PACK_A_LOW()
        "s_nop 1\n\t"

        // Consume this K=16 slice without changing the physical fragment order.
        IRIS_NATIVE128_WMMA_GRID(163:166, 167:170, 171:174, 175:178,
                                 130:133, 134:137, 138:141, 142:145)

        "s_waitcnt lgkmcnt(0)\n\t"
        "s_barrier\n\t"
        IRIS_NATIVE128_READ_A_LOW(0)
        IRIS_NATIVE128_READ_B_LOW(0)
        "s_waitcnt lgkmcnt(15)\n\t"
        IRIS_NATIVE128_PACK_A_HIGH()
        "s_nop 1\n\t"

        // Consume this K=16 slice without changing the physical fragment order.
        IRIS_NATIVE128_WMMA_GRID(179:182, 183:186, 187:190, 191:194,
                                 146:149, 150:153, 154:157, 158:161)

        "s_sub_u32 %[left], %[left], 1\n\t"
        "s_branch cross_k_loop_%=\n\t"

        // Pipeline boundary: branch labels remain local to this asm instance.
        "cross_k_drain1_%=:\n\t"
        "v_xor_b32 v218, 0x4000, v218\n\t"
        "v_xor_b32 v219, 0x4000, v219\n\t"

        // Pipeline boundary: branch labels remain local to this asm instance.
        "cross_k_drain0_%=:\n\t"
        IRIS_NATIVE128_READ_A_HIGH(0)
        IRIS_NATIVE128_READ_B_HIGH(0)
        "s_waitcnt lgkmcnt(15)\n\t"
        IRIS_NATIVE128_PACK_A_LOW()
        "s_nop 1\n\t"

        // Consume this K=16 slice without changing the physical fragment order.
        IRIS_NATIVE128_WMMA_GRID(163:166, 167:170, 171:174, 175:178,
                                 130:133, 134:137, 138:141, 142:145)

        "s_waitcnt lgkmcnt(0)\n\t"
        IRIS_NATIVE128_PACK_A_HIGH()
        "s_nop 1\n\t"

        // Consume this K=16 slice without changing the physical fragment order.
        IRIS_NATIVE128_WMMA_GRID(179:182, 183:186, 187:190, 191:194,
                                 146:149, 150:153, 154:157, 158:161)

        : [acc0] "+&{v[0:7]}"(acc[0]),
          [acc1] "+&{v[8:15]}"(acc[1]),
          [acc2] "+&{v[16:23]}"(acc[2]),
          [acc3] "+&{v[24:31]}"(acc[3]),
          [acc4] "+&{v[32:39]}"(acc[4]),
          [acc5] "+&{v[40:47]}"(acc[5]),
          [acc6] "+&{v[48:55]}"(acc[6]),
          [acc7] "+&{v[56:63]}"(acc[7]),
          [acc8] "+&{v[64:71]}"(acc[8]),
          [acc9] "+&{v[72:79]}"(acc[9]),
          [acc10] "+&{v[80:87]}"(acc[10]),
          [acc11] "+&{v[88:95]}"(acc[11]),
          [acc12] "+&{v[96:103]}"(acc[12]),
          [acc13] "+&{v[104:111]}"(acc[13]),
          [acc14] "+&{v[112:119]}"(acc[14]),
          [acc15] "+&{v[120:127]}"(acc[15]),
          [read_a] "+&{v218}"(read_a),
          [read_b] "+&{v219}"(read_b),
          [store_a] "+&{v195}"(store_a),
          [store_b] "+&{v196}"(store_b),
          [oa] "=&s"(offset_a),
          [ob] "=&s"(offset_b),
          [left] "=&s"(remaining)
        : "{v197}"(loads.offset_a[0]),
          "{v198}"(loads.offset_a[1]),
          "{v199}"(loads.offset_b[0]),
          "{v220}"(loads.offset_b[1]),
          [ra] "s"(loads.resource_a),
          [rb] "s"(loads.resource_b),
          [tiles] "s"(tiles),
          [stride] "s"(stride)
        : "v130", "v131", "v132", "v133", "v134", "v135", "v136", "v137",
          "v138", "v139", "v140", "v141", "v142", "v143", "v144", "v145",
          "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228",
          "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236",
          "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170",
          "v171", "v172", "v173", "v174", "v175", "v176", "v177", "v178",
          "v146", "v147", "v148", "v149", "v150", "v151", "v152", "v153",
          "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161",
          "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244",
          "v245", "v246", "v247", "v248", "v249", "v250", "v251", "v252",
          "v179", "v180", "v181", "v182", "v183", "v184", "v185", "v186",
          "v187", "v188", "v189", "v190", "v191", "v192", "v193", "v194",
          "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209",
          "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217",
          "scc", "memory");
}
