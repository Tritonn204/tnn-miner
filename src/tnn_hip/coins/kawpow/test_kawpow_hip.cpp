#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>

#include <ethash/progpow.hpp>
#include "test_kawpow_hip.h"

// ============================================================
// KawPow (ProgPoW 0.9.4 / Ravencoin) test vectors
// Source: https://github.com/AustraliaCash/cpp-kawpow
//         test/unittests/progpow_test_vectors.hpp
// ============================================================

struct kawpow_test_vector
{
    int block_number;
    const char* header_hash_hex;
    const char* nonce_hex;
    const char* mix_hash_hex;
    const char* final_hash_hex;
};

static const kawpow_test_vector test_vectors[] = {
    {0,
     "0000000000000000000000000000000000000000000000000000000000000000",
     "0000000000000000",
     "6e97b47b134fda0c7888802988e1a373affeb28bcd813b6e9a0fc669c935d03a",
     "e601a7257a70dc48fccc97a7330d704d776047623b92883d77111fb36870f3d1"},
    {49,
     "63155f732f2bf556967f906155b510c917e48e99685ead76ea83f4eca03ab12b",
     "0000000007073c07",
     "d36f7e815ee09e74eceb9c96993a3d681edf2bf0921fc7bb710364042db99777",
     "e7ced124598fd2500a55ad9f9f48e3569327fe50493c77a4ac9799b96efb9463"},
    {50,
     "9e7248f20914913a73d80a70174c331b1d34f260535ac3631d770e656b5dd922",
     "00000000076e482e",
     "d6dc634ae837e2785b347648ea515e25e5d8821ae0b95e1c2a9c2d497e0dcfbd",
     "ab0ad7ef8d8ee317dd12d10310aceed7321d34fb263791c2de5776a6658d177e"},
    {99,
     "de37e1824c86d35d154cf65a88de6d9286aec4f7f10c3fc9f0fa1bcc2687188d",
     "000000003917afab",
     "fa706860e5e0e830d5d1d7157e5bea7f5f8a350c7c8612ac1d1fcf2974d64244",
     "aa85340690f2e907054324a5021937910e15edfd1ef1577231843e7d32ec3a61"},
    {29950,
     "ac7b55e801511b77e11d52e9599206101550144525b5679f2dab19386f23dcce",
     "005d409dbc23a62a",
     "5359807b77a74878269c3a3044df8618a576ce8dc52e1c48d927d4a60e7c6b79",
     "022019e5408683f7f8326b4e46b42864a3a069f17b6151e434fcaedecaadd918"},
    {29999,
     "e43d7e0bdc8a4a3f6e291a5ed790b9fa1a0948a2b9e33c844888690847de19f5",
     "005db5fa4c2a3d03",
     "d15de3f9bfedd9b6d0f498273eb3b437115bdc8326c96c6457ac06deb5c9f389",
     "4e93630b81198752f876b24380999189b7b9366c08222ac05e4237b87114f305"},
    {30000,
     "d34519f72c97cae8892c277776259db3320820cb5279a299d0ef1e155e5c6454",
     "005db8607994ff30",
     "de0348b69bf91dfe2c3d3dba6f0132e9048a5284e57b8d9d20adc5f3dc0d3236",
     "c7953d848cda6e304f77b4c6d735645c8e8508a5e74c9e9814ef37b19087cd6c"},
    {30049,
     "8b6ce5da0b06d18db7bd8492d9e5717f8b53e7e098d9fef7886d58a6e913ef64",
     "005e2e215a8ca2e7",
     "975c6a9decc89cba7ace69338d4de8510d9619aef42b1d35d0bef7e0ce0614a9",
     "c262d8055e288d04b951a844bfca8ba529f5b4d652b408e3942727d7dd90957a"},
    {30050,
     "c2c46173481b9ced61123d2e293b42ede5a1b323210eb2a684df0874ffe09047",
     "005e30899481055e",
     "362f2fabdb9699d3634b6499703f939f378ee4eac803396c2b0ed0fe1d154972",
     "4cd7e6e79e0b63d42b2b06716a919ccc7834077ec727a9ea94edcdaff2fefab8"},
    {30099,
     "ea42197eb2ba79c63cb5e655b8b1f612c5f08aae1a49ff236795a3516d87bc71",
     "005ea6aef136f88b",
     "b1196457261bd05ccb387a8ff3fd02687bf496bd7943d89419465289669e27aa",
     "39d1ebfa783b61a6fa8e9747d0f9f134efae5cfba284a2c80e8deabae6b98676"},
    {59950,
     "49e15ba4bf501ce8fe8876101c808e24c69a859be15de554bf85dbc095491bd6",
     "02ebe0503bd7b1da",
     "df3dbb1669fd35dbb0ae96bbea2d498f0c6992cbddd092aeace42dd933505f95",
     "b8984cf4021c4433f753654848d721f33a0792b4417241f0cf7c7c2db011a54a"},
    {59999,
     "f5c50ba5c0d6210ddb16250ec3efda178de857b2b1703d8d5403bd0f848e19cf",
     "02edb6275bd221e3",
     "5017df70e97ca35638cf439cdbe54f30383d335e18eb4a74d6e166736f1038fa",
     "4cf1fa62f25b577ac822a6a28d55f8b7e3ae7fe983abd868ae00927e68c41016"},
    {170915,
     "5b3e8dfa1aafd3924a51f33e2d672d8dae32fa528d8b1d378d6e4db0ec5d665d",
     "0000000044975727",
     "efb29147484c434f1cc59629da90fd0343e3b047407ecd36e9ad973bd51bbac5",
     "e7e6bb3b2f9acd3864bc86f72f87237eaf475633ef650c726ac80eb0adf116b6"},
};

static const int NUM_TEST_VECTORS = sizeof(test_vectors) / sizeof(test_vectors[0]);

// ============================================================
// Hex utilities
// ============================================================

static ethash::hash256 hex_to_hash256(const char* hex)
{
    ethash::hash256 hash = {};
    for (size_t i = 0; i < 64 && hex[i] && hex[i + 1]; i += 2)
    {
        auto nibble = [](char c) -> int {
            return c <= '9' ? (c - '0') : (c - 'a' + 10);
        };
        hash.bytes[i / 2] = (uint8_t)((nibble(hex[i]) << 4) | nibble(hex[i + 1]));
    }
    return hash;
}

static uint64_t hex_to_u64(const char* hex)
{
    uint64_t v = 0;
    for (int i = 0; hex[i]; i++)
    {
        v <<= 4;
        v |= (hex[i] <= '9') ? (hex[i] - '0') : (hex[i] - 'a' + 10);
    }
    return v;
}

static std::string hash256_to_hex(const ethash::hash256& h)
{
    static const char hex_chars[] = "0123456789abcdef";
    std::string s;
    s.reserve(64);
    for (int i = 0; i < 32; i++)
    {
        s.push_back(hex_chars[h.bytes[i] >> 4]);
        s.push_back(hex_chars[h.bytes[i] & 0xf]);
    }
    return s;
}

// ============================================================
// Test runner
// ============================================================

static int test_kawpow_cpu()
{
    printf("\n");
    printf("========================================\n");
    printf("[hip-test-kawpow] Phase 1: CPU reference validation\n");
    printf("========================================\n\n");
    fflush(stdout);

    int pass = 0, fail = 0;

    // Group test vectors by epoch to reuse epoch contexts
    int prev_epoch = -1;
    ethash::epoch_context_ptr ctx{nullptr, ethash_destroy_epoch_context};

    for (int t = 0; t < NUM_TEST_VECTORS; t++)
    {
        const auto& tv = test_vectors[t];
        int epoch = ethash::get_epoch_number(tv.block_number);

        printf("[CPU %2d] block=%d epoch=%d nonce=%s\n", t + 1, tv.block_number, epoch, tv.nonce_hex);
        fflush(stdout);

        // Create epoch context if epoch changed
        if (epoch != prev_epoch)
        {
            printf("         Creating epoch context %d...\n", epoch);
            fflush(stdout);
            ctx = ethash::create_epoch_context(epoch);
            if (!ctx)
            {
                printf("         \033[31mFAILED: could not create epoch context\033[0m\n");
                fflush(stdout);
                fail++;
                continue;
            }
            prev_epoch = epoch;
        }

        ethash::hash256 header = hex_to_hash256(tv.header_hash_hex);
        uint64_t nonce = hex_to_u64(tv.nonce_hex);

        // Compute hash using CPU reference (light/verification mode)
        ethash::result r = progpow::hash(*ctx, tv.block_number, header, nonce);

        std::string got_mix = hash256_to_hex(r.mix_hash);
        std::string got_final = hash256_to_hex(r.final_hash);

        bool mix_ok = (got_mix == tv.mix_hash_hex);
        bool final_ok = (got_final == tv.final_hash_hex);

        if (mix_ok && final_ok)
        {
            printf("         \033[32mPASS\033[0m mix=%s\n", got_mix.c_str());
            pass++;
        }
        else
        {
            printf("         \033[31mFAIL\033[0m\n");
            if (!mix_ok)
            {
                printf("           mix expected: %s\n", tv.mix_hash_hex);
                printf("           mix got:      %s\n", got_mix.c_str());
            }
            if (!final_ok)
            {
                printf("           final expected: %s\n", tv.final_hash_hex);
                printf("           final got:      %s\n", got_final.c_str());
            }
            fail++;
        }
        fflush(stdout);
    }

    printf("\n========================================\n");
    printf("[CPU] %d/%d passed", pass, NUM_TEST_VECTORS);
    if (fail > 0)
        printf(", \033[31m%d FAILED\033[0m", fail);
    else
        printf(", \033[32mall passed\033[0m");
    printf("\n========================================\n\n");
    fflush(stdout);

    return (fail == 0) ? 0 : 1;
}

int test_kawpow_hip()
{
    // Phase 1: CPU reference validation against known test vectors
    int rc = test_kawpow_cpu();
    if (rc != 0)
        return rc;

    // Phase 2: GPU kernel validation against CPU reference
    return kawpow_gpu_test();
}
