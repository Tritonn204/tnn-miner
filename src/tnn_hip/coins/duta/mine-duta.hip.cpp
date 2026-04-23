#ifdef TNN_DUTAHASH

#include <coins/miners.hpp>
#include <net/net.hpp>
#include "tnn-hugepages.hpp"
#include <stratum/stratum.h>
#include <base64.hpp>

#include "../../common/gpu_compat.hpp"
#include "../../common/gpu_miner.hpp"
#include "../../common/hip_algo_registry.hpp"
#include "../../common/gpu_submit_queue.hpp"
#include <job_safe.hpp>
#include "../../common/tnn_log.hpp"
#include <algo_definitions.h>

#include <crypto/xelis-hash/xelis-hash.hpp>
#include <thread>

struct TestVectorV4 {
    const char* name;
    uint64_t height;
    uint64_t nonce;
    int mem_mb;
    uint8_t anchor[32];
    uint8_t header[80];
    uint8_t digest[32];
};

static void fill_seq_header(uint8_t header[80]) {
    for (int i = 0; i < 80; i++) header[i] = (uint8_t)i;
}
static void fill_seed_header(uint8_t header[80], uint8_t seed) {
    for (int i = 0; i < 80; i++) header[i] = (uint8_t)(seed + (uint8_t)(i * 13));
}
static void hex_to_bytes(const char* s, uint8_t* out, int n) {
    for (int i = 0; i < n; i++) {
        unsigned v = 0;
        sscanf(s + i * 2, "%02x", &v);
        out[i] = (uint8_t)v;
    }
}

static void make_vectors_v4(TestVectorV4* v) {
    memset(v, 0, sizeof(TestVectorV4) * 6);

    v[0].name = "ref_seq_h12345_n42_anchor11_ds1m";
    v[0].height = 12345; v[0].nonce = 42; v[0].mem_mb = 1;
    memset(v[0].anchor, 0x11, 32); fill_seq_header(v[0].header);
    hex_to_bytes("758770774ab2bbb9a3ac22fe0c7b93eca3cfcc554d4557a24bd49bb10ce90ae2", v[0].digest, 32);

    v[1].name = "seq_h0_n0_anchoraa_ds1m";
    v[1].height = 0; v[1].nonce = 0; v[1].mem_mb = 1;
    memset(v[1].anchor, 0xaa, 32); fill_seq_header(v[1].header);
    hex_to_bytes("e189530d1f1d2c1400cf80fba2ae6bffb048c8f5bd3f3b81b012b3a12a7da04f", v[1].digest, 32);

    v[2].name = "seq_h2048_n1_anchor22_ds1m";
    v[2].height = 2048; v[2].nonce = 1; v[2].mem_mb = 1;
    memset(v[2].anchor, 0x22, 32); fill_seq_header(v[2].header);
    hex_to_bytes("0c602e53818f44dca6f1bc96c36d200280c17e680b97a2fd51a57d0acf8e6e4f", v[2].digest, 32);

    v[3].name = "seq_h4097_n999_anchor55_ds1m";
    v[3].height = 4097; v[3].nonce = 999; v[3].mem_mb = 1;
    memset(v[3].anchor, 0x55, 32); fill_seq_header(v[3].header);
    hex_to_bytes("8468e88ee1280cedad626bd9bdac95ea9cbe050837ffa704d56da13914d3c41d", v[3].digest, 32);

    v[4].name = "seed7_h12345_n42_anchor11_ds1m";
    v[4].height = 12345; v[4].nonce = 42; v[4].mem_mb = 1;
    memset(v[4].anchor, 0x11, 32); fill_seed_header(v[4].header, 7);
    hex_to_bytes("10bcf0c86387926d8b6ee3dde281d52e92319fe70bc4bbcde3fbe3eeef28df8e", v[4].digest, 32);

    v[5].name = "seedc3_h99999_n123456789_anchor77_ds1m";
    v[5].height = 99999; v[5].nonce = 123456789ULL; v[5].mem_mb = 1;
    memset(v[5].anchor, 0x77, 32); fill_seed_header(v[5].header, 0xc3);
    hex_to_bytes("7accba3ed81aba92aa691902e66166817462afdff4e3cb4d5707e1883a7c2605", v[5].digest, 32);
}

// v4 is current as of 04/22/2026. Prepared for versionsed support after v4
static int run_vector_checks(void) {
    TestVectorV4 vecs[6];
    make_vectors_v4(vecs);
    int pass_count = 0;

    printf("=== correctness preflight ===\n");
    for (int i = 0; i < 6; i++) {
        size_t dataset_bytes = (size_t)vecs[i].mem_mb * 1024ull * 1024ull;
        uint32_t nblocks = (uint32_t)(dataset_bytes / 64u);
        uint8_t* h_dataset = (uint8_t*)malloc(dataset_bytes);
        if (!h_dataset) {
            printf("vector[%d] %s : ERROR host alloc failed\n", i, vecs[i].name);
            continue;
        }
        build_dataset_v4_host(vecs[i].height / EPOCH_LEN, vecs[i].anchor, (size_t)vecs[i].mem_mb, h_dataset);

        uint8_t *d_dataset = nullptr, *d_header = nullptr, *d_anchor = nullptr;
        uint64_t *d_scratch = nullptr, *d_digest = nullptr;
        uint64_t h_digest_u64[4] = {0};

        HIP_CHECK(oroMalloc(&d_dataset, dataset_bytes));
        HIP_CHECK(oroMalloc(&d_header, 80));
        HIP_CHECK(oroMalloc(&d_anchor, 32));
        HIP_CHECK(oroMalloc(&d_scratch, SCRATCHPAD_BYTES));
        HIP_CHECK(oroMalloc(&d_digest, 4 * sizeof(uint64_t)));

        HIP_CHECK(oroMemcpy(d_dataset, h_dataset, dataset_bytes, oroMemcpyHostToDevice));
        HIP_CHECK(oroMemcpy(d_header, vecs[i].header, 80, oroMemcpyHostToDevice));
        HIP_CHECK(oroMemcpy(d_anchor, vecs[i].anchor, 32, oroMemcpyHostToDevice));

        (void)oroLaunchKernelGGL(init_scratch_kernel_v4, dim3(1), dim3(1), 0, 0,
                           d_header, d_anchor, vecs[i].height, vecs[i].nonce, d_scratch);
        HIP_CHECK(oroGetLastError());
        HIP_CHECK(oroDeviceSynchronize());

        (void)oroLaunchKernelGGL(digest_kernel_v4, dim3(1), dim3(1), 0, 0,
                           d_dataset, nblocks, d_header, d_anchor,
                           vecs[i].height, vecs[i].nonce, d_scratch, d_digest);
        HIP_CHECK(oroGetLastError());
        HIP_CHECK(oroDeviceSynchronize());
        HIP_CHECK(oroMemcpy(h_digest_u64, d_digest, 4 * sizeof(uint64_t), oroMemcpyDeviceToHost));

        uint8_t got[32];
        for (int w = 0; w < 4; w++) store_u64_le(got + w * 8, h_digest_u64[w]);
        int ok = (memcmp(got, vecs[i].digest, 32) == 0);
        pass_count += ok;

        printf("[%s] %s\n", ok ? "PASS" : "FAIL", vecs[i].name);
        if (!ok) {
            printf("  expected=");
            for (int k = 0; k < 32; k++) printf("%02x", vecs[i].digest[k]);
            printf("\n  got     =");
            for (int k = 0; k < 32; k++) printf("%02x", got[k]);
            printf("\n");
        }

        (void)oroFree(d_dataset); oroFree(d_header); oroFree(d_anchor); oroFree(d_scratch); oroFree(d_digest);
        free(h_dataset);
    }
    printf("vector_summary=%d/6 passed\n\n", pass_count);
    return pass_count;
}

#endif