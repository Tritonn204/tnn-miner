// Intentionally included inside the mainloop scope: an inline function changed ISA.
// Requires c, transcript, kk and the Pearl policy type. No independent include guard.

uint32_t parts[Pearl::chains] = {};
#pragma unroll
for (unsigned chain = 0; chain < Pearl::chains; ++chain) {
#pragma unroll
    for (unsigned z = 0; z < 128 / Pearl::chains; ++z) {
        constexpr unsigned span = 128 / Pearl::chains;
        unsigned i = chain * span + z;
        parts[chain] ^= uint32_t(c[i / 8][i % 8]);
    }
    if constexpr (Pearl::chains > 1)
        asm volatile("" : "+v"(parts[chain]));
}
uint32_t reduction = 0;
#pragma unroll
for (unsigned i = 0; i < Pearl::chains; ++i)
    reduction ^= parts[i];
unsigned word = ((kk + 32) / Pearl::rank - 1) % 16;
if constexpr (Pearl::update == 1) {
    switch (word) {
    case 0:
        transcript[0] = (transcript[0] << 13 | transcript[0] >> 19) ^ reduction;
        break;
    case 1:
        transcript[1] = (transcript[1] << 13 | transcript[1] >> 19) ^ reduction;
        break;
    case 2:
        transcript[2] = (transcript[2] << 13 | transcript[2] >> 19) ^ reduction;
        break;
    case 3:
        transcript[3] = (transcript[3] << 13 | transcript[3] >> 19) ^ reduction;
        break;
    case 4:
        transcript[4] = (transcript[4] << 13 | transcript[4] >> 19) ^ reduction;
        break;
    case 5:
        transcript[5] = (transcript[5] << 13 | transcript[5] >> 19) ^ reduction;
        break;
    case 6:
        transcript[6] = (transcript[6] << 13 | transcript[6] >> 19) ^ reduction;
        break;
    case 7:
        transcript[7] = (transcript[7] << 13 | transcript[7] >> 19) ^ reduction;
        break;
    case 8:
        transcript[8] = (transcript[8] << 13 | transcript[8] >> 19) ^ reduction;
        break;
    case 9:
        transcript[9] = (transcript[9] << 13 | transcript[9] >> 19) ^ reduction;
        break;
    case 10:
        transcript[10] = (transcript[10] << 13 | transcript[10] >> 19) ^ reduction;
        break;
    case 11:
        transcript[11] = (transcript[11] << 13 | transcript[11] >> 19) ^ reduction;
        break;
    case 12:
        transcript[12] = (transcript[12] << 13 | transcript[12] >> 19) ^ reduction;
        break;
    case 13:
        transcript[13] = (transcript[13] << 13 | transcript[13] >> 19) ^ reduction;
        break;
    case 14:
        transcript[14] = (transcript[14] << 13 | transcript[14] >> 19) ^ reduction;
        break;
    case 15:
        transcript[15] = (transcript[15] << 13 | transcript[15] >> 19) ^ reduction;
        break;
    }
} else {
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) {
        if constexpr (Pearl::update == 2) {
            uint32_t old = transcript[i], next = (old << 13 | old >> 19) ^ reduction;
            asm volatile("v_cmp_eq_u32_e64 vcc_lo, %2, %3\n\tv_cndmask_b32_e32 %0, %0, %1, vcc_lo"
                         : "+v"(old)
                         : "v"(next), "s"(word), "n"(i)
                         : "vcc");
            transcript[i] = old;
        } else if (word == i)
            transcript[i] = (transcript[i] << 13 | transcript[i] >> 19) ^ reduction;
    }
}
