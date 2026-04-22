#include "kawpow_algo.h"

const kawpow_coin_padding_t* currentKawpowPadding = &KAWPOW_PADDING_RVN;
const kawpow_coin_padding_t* devKawpowPadding = &KAWPOW_PADDING_RVN;

bool initKawpowPaddingForCoin(int coinId, const kawpow_coin_padding_t** padding)
{
    switch (coinId) {
    case COIN_KAWPOW:
    case COIN_RVN:
    case COIN_QUAI:
        // standard KawPow padding ("rAVENCOINKAWPOW")
        *padding = &KAWPOW_PADDING_RVN;
        return true;
    default:
        return false;
    }
}
