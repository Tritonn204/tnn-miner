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