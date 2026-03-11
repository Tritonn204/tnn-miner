// yespower_algo.cpp
#include "yespower_algo.h"
#include "tnn-common.hpp"

// Global parameters
yespower_params_t currentYespowerParams = {
    YESPOWER_1_0,
    2048,
    32,
    (const uint8_t*)"Let the quest begin",
    19,
};

yespower_params_t devYespowerParams = {
    YESPOWER_1_0,
    2048,
    32,
    (const uint8_t*)"Let the quest begin",
    19,
};

void initADVCParams(yespower_params_t* params) {
    params->version = YESPOWER_1_0;
    params->N = 2048;
    params->r = 32;
    params->pers = (const uint8_t*)"Let the quest begin";
    params->perslen = 19;
}

void setManualYespowerParams(yespower_params_t* params, uint32_t N, uint32_t R, const char* pers) {
    params->version = YESPOWER_1_0;
    params->N = N;
    params->r = R;
    params->pers = (const uint8_t*)pers;
    params->perslen = pers ? strlen(pers) : 0;
}

void setManualYespowerParamsV(yespower_params_t* params, yespower_version_t ver, uint32_t N, uint32_t R, const char* pers) {
    params->version = ver;
    params->N = N;
    params->r = R;
    params->pers = (const uint8_t*)pers;
    params->perslen = pers ? strlen(pers) : 0;
}

bool initYespowerParamsForCoin(int coinId, yespower_params_t* params) {
    switch (coinId) {
    case COIN_ADVC:
        initADVCParams(params);
        return true;
    case COIN_TIDE:
        // YespowerTIDE: v1.0, N=2048, r=8, no pers
        setManualYespowerParamsV(params, YESPOWER_1_0, 2048, 8, NULL);
        return true;
    case COIN_YPR16:
        // YespowerR16: v1.0, N=4096, r=16, no pers
        setManualYespowerParamsV(params, YESPOWER_1_0, 4096, 16, NULL);
        return true;
    case COIN_YCR16:
        // YescryptR16: v0.5, N=4096, r=16, pers="Client Key"
        setManualYespowerParamsV(params, YESPOWER_0_5, 4096, 16, "Client Key");
        return true;
    case COIN_YCR8:
        // YescryptR8: v0.5, N=2048, r=8, pers="Client Key"
        setManualYespowerParamsV(params, YESPOWER_0_5, 2048, 8, "Client Key");
        return true;
    case COIN_MGPC:
        // YespowerMGPC: v1.0, N=2048, r=32, pers="Magpies are birds of the Corvidae family."
        setManualYespowerParamsV(params, YESPOWER_1_0, 2048, 32, "Magpies are birds of the Corvidae family.");
        return true;
    case COIN_URX:
        // YespowerURX: v1.0, N=2048, r=32, pers="UraniumX"
        setManualYespowerParamsV(params, YESPOWER_1_0, 2048, 32, "UraniumX");
        return true;
    case COIN_LTNCG:
        // YespowerLTNCG: v1.0, N=2048, r=32, pers="LTNCGYES"
        setManualYespowerParamsV(params, YESPOWER_1_0, 2048, 32, "LTNCGYES");
        return true;
    case COIN_YSC:
        // Yescrypt (plain): v0.5, N=2048, r=8, no pers
        setManualYespowerParamsV(params, YESPOWER_0_5, 2048, 8, NULL);
        return true;
    case COIN_YCR32:
        // YescryptR32: v0.5, N=4096, r=32, pers="WaviBanana"
        setManualYespowerParamsV(params, YESPOWER_0_5, 4096, 32, "WaviBanana");
        return true;
    // EQPAY disabled — hashes 181 bytes instead of 80, needs special header handling
    // case COIN_EQPAY:
    //     setManualYespowerParamsV(params, YESPOWER_1_0, 2048, 32,
    //         "The gods had gone away, and the ritual of the religion continued senselessly, uselessly.");
    //     return true;
    default:
        return false;
    }
}
