#pragma once

#include <astrobwtv3/astrobwtv3.h>
#include <astrobwtv3/astrotest.hpp>
// #include <astrobwtv3/lookupcompute.h>

#include <xelis-hash/xelis-hash.hpp>
#include <spectrex/spectrex.h>
#include <astrix-hash/astrix-hash.h>
#include <nxl-hash/nxl-hash.h>
#include <hoohash/hoohash.h>
#include <wala-hash/wala-hash.h>
#include <shai/shai-hive.h>
#include <yespower/yespower.h>

#include <randomx/randomx.h>
#include <randomx/tests/randomx_test.h>

#define DEFINE_UNSUPPORTED_MSG(name, algo) \
    const char* unsupported_##name = "This Binary was compiled without " algo " support... \n" \
    "Please source a TNN Miner binary with " algo " support";

// Use it for each algo
DEFINE_UNSUPPORTED_MSG(astro, "AstroBWTv3")
DEFINE_UNSUPPORTED_MSG(xelishash, "XelisHash")
DEFINE_UNSUPPORTED_MSG(randomx, "RandomX")
DEFINE_UNSUPPORTED_MSG(astrix, "AstrixHash")
DEFINE_UNSUPPORTED_MSG(nexellia, "Nexell-AI")
DEFINE_UNSUPPORTED_MSG(hoohash, "Hoohash")
DEFINE_UNSUPPORTED_MSG(waglayla, "Waglayla")
DEFINE_UNSUPPORTED_MSG(shai, "Shai")
DEFINE_UNSUPPORTED_MSG(yespower, "YesPower")
DEFINE_UNSUPPORTED_MSG(rinhash, "RinHash")