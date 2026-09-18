#include "asm_templates.hpp"

// Keep waits, LDS handoffs, and register constraints in this single asm block.
// The templates only remove repetition; they do not reschedule instructions.
template <class Loads> __device__ __forceinline__ void slot_partial(
    I8 (&acc)[16], const Loads& loads, unsigned tiles, unsigned stride,
    unsigned read_a, unsigned read_b, unsigned store_a, unsigned store_b, unsigned base, unsigned wave) {
    unsigned offset_a, offset_b, remaining, scratch, word, temp;

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

        "s_sub_u32 %[scratch], %[tiles], %[left]\n\t"
        "s_add_u32 %[scratch], %[scratch], 1\n\t"
        "s_and_b32 %[word], %[scratch], 3\n\t"
        "s_cmp_eq_u32 %[word], 0\n\t"
        "s_cbranch_scc0 native_skip0_%=\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "v_xor3_b32 v146, v0, v4, v8\n\t"
        "v_xor3_b32 v147, v1, v5, v9\n\t"
        "v_xor3_b32 v148, v2, v6, v10\n\t"
        "v_xor3_b32 v149, v3, v7, v11\n\t"
        "v_xor3_b32 v146, v146, v12, v16\n\t"
        "v_xor3_b32 v147, v147, v13, v17\n\t"
        "v_xor3_b32 v148, v148, v14, v18\n\t"
        "v_xor3_b32 v149, v149, v15, v19\n\t"
        "v_xor3_b32 v146, v146, v20, v24\n\t"
        "v_xor3_b32 v147, v147, v21, v25\n\t"
        "v_xor3_b32 v148, v148, v22, v26\n\t"
        "v_xor3_b32 v149, v149, v23, v27\n\t"
        "v_xor3_b32 v146, v146, v28, v32\n\t"
        "v_xor3_b32 v147, v147, v29, v33\n\t"
        "v_xor3_b32 v148, v148, v30, v34\n\t"
        "v_xor3_b32 v149, v149, v31, v35\n\t"
        "v_xor3_b32 v146, v146, v36, v40\n\t"
        "v_xor3_b32 v147, v147, v37, v41\n\t"
        "v_xor3_b32 v148, v148, v38, v42\n\t"
        "v_xor3_b32 v149, v149, v39, v43\n\t"
        "v_xor3_b32 v146, v146, v44, v48\n\t"
        "v_xor3_b32 v147, v147, v45, v49\n\t"
        "v_xor3_b32 v148, v148, v46, v50\n\t"
        "v_xor3_b32 v149, v149, v47, v51\n\t"
        "v_xor3_b32 v146, v146, v52, v56\n\t"
        "v_xor3_b32 v147, v147, v53, v57\n\t"
        "v_xor3_b32 v148, v148, v54, v58\n\t"
        "v_xor3_b32 v149, v149, v55, v59\n\t"
        "v_xor3_b32 v146, v146, v60, v64\n\t"
        "v_xor3_b32 v147, v147, v61, v65\n\t"
        "v_xor3_b32 v148, v148, v62, v66\n\t"
        "v_xor3_b32 v149, v149, v63, v67\n\t"
        "v_xor3_b32 v146, v146, v68, v72\n\t"
        "v_xor3_b32 v147, v147, v69, v73\n\t"
        "v_xor3_b32 v148, v148, v70, v74\n\t"
        "v_xor3_b32 v149, v149, v71, v75\n\t"
        "v_xor3_b32 v146, v146, v76, v80\n\t"
        "v_xor3_b32 v147, v147, v77, v81\n\t"
        "v_xor3_b32 v148, v148, v78, v82\n\t"
        "v_xor3_b32 v149, v149, v79, v83\n\t"
        "v_xor3_b32 v146, v146, v84, v88\n\t"
        "v_xor3_b32 v147, v147, v85, v89\n\t"
        "v_xor3_b32 v148, v148, v86, v90\n\t"
        "v_xor3_b32 v149, v149, v87, v91\n\t"
        "v_xor3_b32 v146, v146, v92, v96\n\t"
        "v_xor3_b32 v147, v147, v93, v97\n\t"
        "v_xor3_b32 v148, v148, v94, v98\n\t"
        "v_xor3_b32 v149, v149, v95, v99\n\t"
        "v_xor3_b32 v146, v146, v100, v104\n\t"
        "v_xor3_b32 v147, v147, v101, v105\n\t"
        "v_xor3_b32 v148, v148, v102, v106\n\t"
        "v_xor3_b32 v149, v149, v103, v107\n\t"
        "v_xor3_b32 v146, v146, v108, v112\n\t"
        "v_xor3_b32 v147, v147, v109, v113\n\t"
        "v_xor3_b32 v148, v148, v110, v114\n\t"
        "v_xor3_b32 v149, v149, v111, v115\n\t"
        "v_xor3_b32 v146, v146, v116, v120\n\t"
        "v_xor3_b32 v147, v147, v117, v121\n\t"
        "v_xor3_b32 v148, v148, v118, v122\n\t"
        "v_xor3_b32 v149, v149, v119, v123\n\t"
        "v_xor_b32 v146, v146, v124\n\t"
        "v_xor_b32 v147, v147, v125\n\t"
        "v_xor_b32 v148, v148, v126\n\t"
        "v_xor_b32 v149, v149, v127\n\t"
        "v_xor3_b32 v146, v146, v147, v148\n\t"
        "v_xor_b32 v146, v146, v149\n\t"
        "s_lshr_b32 %[scratch], %[scratch], 2\n\t"
        "s_sub_u32 %[word], %[scratch], 1\n\t"
        "s_and_b32 %[word], %[word], 15\n\t"
        "s_cmp_eq_u32 %[word], 15\n\t"
        "s_cselect_b32 %[temp], 8704, 0\n\t"
        "s_lshl_b32 %[word], %[word], 9\n\t"
        "s_add_u32 %[word], %[word], 8704\n\t"
        "s_add_u32 %[word], %[word], %[temp]\n\t"
        "s_add_u32 %[word], %[word], %[base]\n\t"
        "s_lshl_b32 %[temp], %[wave], 5\n\t"
        "v_mbcnt_lo_u32_b32 v150, -1, 0\n\t"
        "v_add_u32 v150, %[temp], v150\n\t"
        "v_lshlrev_b32 v150, 2, v150\n\t"
        "v_add_u32 v150, %[word], v150\n\t"
        "s_cmp_le_u32 %[scratch], 16\n\t"
        "s_cbranch_scc1 native_store0_%=\n\t"
        "ds_load_b32 v151, v150\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "v_alignbit_b32 v151, v151, v151, 19\n\t"
        "v_xor_b32 v146, v146, v151\n\t"
        "native_store0_%=:\n\t"
        "ds_store_b32 v150, v146\n\t"
        "native_skip0_%=:\n\t"
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

        "s_sub_u32 %[scratch], %[tiles], %[left]\n\t"
        "s_add_u32 %[scratch], %[scratch], 1\n\t"
        "s_and_b32 %[word], %[scratch], 3\n\t"
        "s_cmp_eq_u32 %[word], 0\n\t"
        "s_cbranch_scc0 native_skip1_%=\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "v_xor3_b32 v146, v0, v4, v8\n\t"
        "v_xor3_b32 v147, v1, v5, v9\n\t"
        "v_xor3_b32 v148, v2, v6, v10\n\t"
        "v_xor3_b32 v149, v3, v7, v11\n\t"
        "v_xor3_b32 v146, v146, v12, v16\n\t"
        "v_xor3_b32 v147, v147, v13, v17\n\t"
        "v_xor3_b32 v148, v148, v14, v18\n\t"
        "v_xor3_b32 v149, v149, v15, v19\n\t"
        "v_xor3_b32 v146, v146, v20, v24\n\t"
        "v_xor3_b32 v147, v147, v21, v25\n\t"
        "v_xor3_b32 v148, v148, v22, v26\n\t"
        "v_xor3_b32 v149, v149, v23, v27\n\t"
        "v_xor3_b32 v146, v146, v28, v32\n\t"
        "v_xor3_b32 v147, v147, v29, v33\n\t"
        "v_xor3_b32 v148, v148, v30, v34\n\t"
        "v_xor3_b32 v149, v149, v31, v35\n\t"
        "v_xor3_b32 v146, v146, v36, v40\n\t"
        "v_xor3_b32 v147, v147, v37, v41\n\t"
        "v_xor3_b32 v148, v148, v38, v42\n\t"
        "v_xor3_b32 v149, v149, v39, v43\n\t"
        "v_xor3_b32 v146, v146, v44, v48\n\t"
        "v_xor3_b32 v147, v147, v45, v49\n\t"
        "v_xor3_b32 v148, v148, v46, v50\n\t"
        "v_xor3_b32 v149, v149, v47, v51\n\t"
        "v_xor3_b32 v146, v146, v52, v56\n\t"
        "v_xor3_b32 v147, v147, v53, v57\n\t"
        "v_xor3_b32 v148, v148, v54, v58\n\t"
        "v_xor3_b32 v149, v149, v55, v59\n\t"
        "v_xor3_b32 v146, v146, v60, v64\n\t"
        "v_xor3_b32 v147, v147, v61, v65\n\t"
        "v_xor3_b32 v148, v148, v62, v66\n\t"
        "v_xor3_b32 v149, v149, v63, v67\n\t"
        "v_xor3_b32 v146, v146, v68, v72\n\t"
        "v_xor3_b32 v147, v147, v69, v73\n\t"
        "v_xor3_b32 v148, v148, v70, v74\n\t"
        "v_xor3_b32 v149, v149, v71, v75\n\t"
        "v_xor3_b32 v146, v146, v76, v80\n\t"
        "v_xor3_b32 v147, v147, v77, v81\n\t"
        "v_xor3_b32 v148, v148, v78, v82\n\t"
        "v_xor3_b32 v149, v149, v79, v83\n\t"
        "v_xor3_b32 v146, v146, v84, v88\n\t"
        "v_xor3_b32 v147, v147, v85, v89\n\t"
        "v_xor3_b32 v148, v148, v86, v90\n\t"
        "v_xor3_b32 v149, v149, v87, v91\n\t"
        "v_xor3_b32 v146, v146, v92, v96\n\t"
        "v_xor3_b32 v147, v147, v93, v97\n\t"
        "v_xor3_b32 v148, v148, v94, v98\n\t"
        "v_xor3_b32 v149, v149, v95, v99\n\t"
        "v_xor3_b32 v146, v146, v100, v104\n\t"
        "v_xor3_b32 v147, v147, v101, v105\n\t"
        "v_xor3_b32 v148, v148, v102, v106\n\t"
        "v_xor3_b32 v149, v149, v103, v107\n\t"
        "v_xor3_b32 v146, v146, v108, v112\n\t"
        "v_xor3_b32 v147, v147, v109, v113\n\t"
        "v_xor3_b32 v148, v148, v110, v114\n\t"
        "v_xor3_b32 v149, v149, v111, v115\n\t"
        "v_xor3_b32 v146, v146, v116, v120\n\t"
        "v_xor3_b32 v147, v147, v117, v121\n\t"
        "v_xor3_b32 v148, v148, v118, v122\n\t"
        "v_xor3_b32 v149, v149, v119, v123\n\t"
        "v_xor_b32 v146, v146, v124\n\t"
        "v_xor_b32 v147, v147, v125\n\t"
        "v_xor_b32 v148, v148, v126\n\t"
        "v_xor_b32 v149, v149, v127\n\t"
        "v_xor3_b32 v146, v146, v147, v148\n\t"
        "v_xor_b32 v146, v146, v149\n\t"
        "s_lshr_b32 %[scratch], %[scratch], 2\n\t"
        "s_sub_u32 %[word], %[scratch], 1\n\t"
        "s_and_b32 %[word], %[word], 15\n\t"
        "s_cmp_eq_u32 %[word], 15\n\t"
        "s_cselect_b32 %[temp], 8704, 0\n\t"
        "s_lshl_b32 %[word], %[word], 9\n\t"
        "s_add_u32 %[word], %[word], 8704\n\t"
        "s_add_u32 %[word], %[word], %[temp]\n\t"
        "s_add_u32 %[word], %[word], %[base]\n\t"
        "s_lshl_b32 %[temp], %[wave], 5\n\t"
        "v_mbcnt_lo_u32_b32 v150, -1, 0\n\t"
        "v_add_u32 v150, %[temp], v150\n\t"
        "v_lshlrev_b32 v150, 2, v150\n\t"
        "v_add_u32 v150, %[word], v150\n\t"
        "s_cmp_le_u32 %[scratch], 16\n\t"
        "s_cbranch_scc1 native_store1_%=\n\t"
        "ds_load_b32 v151, v150\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "v_alignbit_b32 v151, v151, v151, 19\n\t"
        "v_xor_b32 v146, v146, v151\n\t"
        "native_store1_%=:\n\t"
        "ds_store_b32 v150, v146\n\t"
        "native_skip1_%=:\n\t"
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

        "s_sub_u32 %[scratch], %[tiles], %[left]\n\t"
        "s_add_u32 %[scratch], %[scratch], 1\n\t"
        "s_and_b32 %[word], %[scratch], 3\n\t"
        "s_cmp_eq_u32 %[word], 0\n\t"
        "s_cbranch_scc0 native_skip2_%=\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "v_xor3_b32 v146, v0, v4, v8\n\t"
        "v_xor3_b32 v147, v1, v5, v9\n\t"
        "v_xor3_b32 v148, v2, v6, v10\n\t"
        "v_xor3_b32 v149, v3, v7, v11\n\t"
        "v_xor3_b32 v146, v146, v12, v16\n\t"
        "v_xor3_b32 v147, v147, v13, v17\n\t"
        "v_xor3_b32 v148, v148, v14, v18\n\t"
        "v_xor3_b32 v149, v149, v15, v19\n\t"
        "v_xor3_b32 v146, v146, v20, v24\n\t"
        "v_xor3_b32 v147, v147, v21, v25\n\t"
        "v_xor3_b32 v148, v148, v22, v26\n\t"
        "v_xor3_b32 v149, v149, v23, v27\n\t"
        "v_xor3_b32 v146, v146, v28, v32\n\t"
        "v_xor3_b32 v147, v147, v29, v33\n\t"
        "v_xor3_b32 v148, v148, v30, v34\n\t"
        "v_xor3_b32 v149, v149, v31, v35\n\t"
        "v_xor3_b32 v146, v146, v36, v40\n\t"
        "v_xor3_b32 v147, v147, v37, v41\n\t"
        "v_xor3_b32 v148, v148, v38, v42\n\t"
        "v_xor3_b32 v149, v149, v39, v43\n\t"
        "v_xor3_b32 v146, v146, v44, v48\n\t"
        "v_xor3_b32 v147, v147, v45, v49\n\t"
        "v_xor3_b32 v148, v148, v46, v50\n\t"
        "v_xor3_b32 v149, v149, v47, v51\n\t"
        "v_xor3_b32 v146, v146, v52, v56\n\t"
        "v_xor3_b32 v147, v147, v53, v57\n\t"
        "v_xor3_b32 v148, v148, v54, v58\n\t"
        "v_xor3_b32 v149, v149, v55, v59\n\t"
        "v_xor3_b32 v146, v146, v60, v64\n\t"
        "v_xor3_b32 v147, v147, v61, v65\n\t"
        "v_xor3_b32 v148, v148, v62, v66\n\t"
        "v_xor3_b32 v149, v149, v63, v67\n\t"
        "v_xor3_b32 v146, v146, v68, v72\n\t"
        "v_xor3_b32 v147, v147, v69, v73\n\t"
        "v_xor3_b32 v148, v148, v70, v74\n\t"
        "v_xor3_b32 v149, v149, v71, v75\n\t"
        "v_xor3_b32 v146, v146, v76, v80\n\t"
        "v_xor3_b32 v147, v147, v77, v81\n\t"
        "v_xor3_b32 v148, v148, v78, v82\n\t"
        "v_xor3_b32 v149, v149, v79, v83\n\t"
        "v_xor3_b32 v146, v146, v84, v88\n\t"
        "v_xor3_b32 v147, v147, v85, v89\n\t"
        "v_xor3_b32 v148, v148, v86, v90\n\t"
        "v_xor3_b32 v149, v149, v87, v91\n\t"
        "v_xor3_b32 v146, v146, v92, v96\n\t"
        "v_xor3_b32 v147, v147, v93, v97\n\t"
        "v_xor3_b32 v148, v148, v94, v98\n\t"
        "v_xor3_b32 v149, v149, v95, v99\n\t"
        "v_xor3_b32 v146, v146, v100, v104\n\t"
        "v_xor3_b32 v147, v147, v101, v105\n\t"
        "v_xor3_b32 v148, v148, v102, v106\n\t"
        "v_xor3_b32 v149, v149, v103, v107\n\t"
        "v_xor3_b32 v146, v146, v108, v112\n\t"
        "v_xor3_b32 v147, v147, v109, v113\n\t"
        "v_xor3_b32 v148, v148, v110, v114\n\t"
        "v_xor3_b32 v149, v149, v111, v115\n\t"
        "v_xor3_b32 v146, v146, v116, v120\n\t"
        "v_xor3_b32 v147, v147, v117, v121\n\t"
        "v_xor3_b32 v148, v148, v118, v122\n\t"
        "v_xor3_b32 v149, v149, v119, v123\n\t"
        "v_xor_b32 v146, v146, v124\n\t"
        "v_xor_b32 v147, v147, v125\n\t"
        "v_xor_b32 v148, v148, v126\n\t"
        "v_xor_b32 v149, v149, v127\n\t"
        "v_xor3_b32 v146, v146, v147, v148\n\t"
        "v_xor_b32 v146, v146, v149\n\t"
        "s_lshr_b32 %[scratch], %[scratch], 2\n\t"
        "s_sub_u32 %[word], %[scratch], 1\n\t"
        "s_and_b32 %[word], %[word], 15\n\t"
        "s_cmp_eq_u32 %[word], 15\n\t"
        "s_cselect_b32 %[temp], 8704, 0\n\t"
        "s_lshl_b32 %[word], %[word], 9\n\t"
        "s_add_u32 %[word], %[word], 8704\n\t"
        "s_add_u32 %[word], %[word], %[temp]\n\t"
        "s_add_u32 %[word], %[word], %[base]\n\t"
        "s_lshl_b32 %[temp], %[wave], 5\n\t"
        "v_mbcnt_lo_u32_b32 v150, -1, 0\n\t"
        "v_add_u32 v150, %[temp], v150\n\t"
        "v_lshlrev_b32 v150, 2, v150\n\t"
        "v_add_u32 v150, %[word], v150\n\t"
        "s_cmp_le_u32 %[scratch], 16\n\t"
        "s_cbranch_scc1 native_store2_%=\n\t"
        "ds_load_b32 v151, v150\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "v_alignbit_b32 v151, v151, v151, 19\n\t"
        "v_xor_b32 v146, v146, v151\n\t"
        "native_store2_%=:\n\t"
        "ds_store_b32 v150, v146\n\t"
        "native_skip2_%=:\n\t"
        : [scratch] "=&s"(scratch), [word] "=&s"(word), [temp] "=&s"(temp),
          [acc0] "+&{v[0:7]}"(acc[0]),
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
          [stride] "s"(stride), [base] "s"(base), [wave] "s"(wave)
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
          "vcc", "scc", "memory");
}
