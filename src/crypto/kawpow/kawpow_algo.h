#pragma once
#include <ethash/kawpow_coins.h>
#include <algo_definitions.h>

// Global padding pointers — set during coin initialization, consumed by GPU kernel RTC
extern const kawpow_coin_padding_t* currentKawpowPadding;
extern const kawpow_coin_padding_t* devKawpowPadding;

// Initialize padding for a given coin. Returns true if recognized.
bool initKawpowPaddingForCoin(int coinId, const kawpow_coin_padding_t** padding);
