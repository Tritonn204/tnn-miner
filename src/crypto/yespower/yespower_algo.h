// yespower_algo.h
#pragma once

#include "yespower.h"
#include "algo_definitions.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Global parameter storage
extern yespower_params_t currentYespowerParams;
extern yespower_params_t devYespowerParams;

// Function declarations
void initADVCParams(yespower_params_t* params);
void setManualYespowerParams(yespower_params_t* params, uint32_t N, uint32_t R, const char* pers);

#ifdef __cplusplus
}
#endif