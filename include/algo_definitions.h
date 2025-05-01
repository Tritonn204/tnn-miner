#pragma once

// Each new blockchain has 10 reserved net protocol slots
// AstroBWTv3
#define PROTO_DERO_SOLO 0

// XelisHash v1/v2
#define PROTO_XELIS_SOLO 10
#define PROTO_XELIS_XATUM 11
#define PROTO_XELIS_STRATUM 12

// SpectreX
#define PROTO_SPECTRE_SOLO 20
#define PROTO_SPECTRE_STRATUM 21

// RandomX Family
#define PROTO_RX0_SOLO 30
#define PROTO_RX0_STRATUM 31

// VerusHash
#define PROTO_VERUS_SOLO 50
#define PROTO_VERUS_STRATUM 51

// KAS family
#define PROTO_KAS_SOLO 60
#define PROTO_KAS_STRATUM 61

#define COIN_UNKNOWN -1
#define COIN_DERO 0
#define COIN_XELIS 1
#define COIN_SPECTRE 2
#define COIN_RX0 3
#define COIN_XMR 4
#define COIN_SAL 5
#define COIN_ZEPH 6
#define COIN_VERUS 7
#define COIN_AIX 8
#define COIN_NXL 9
#define COIN_HTN 10
#define COIN_WALA 11
#define COIN_SHAI 12
#define COIN_COUNT 13

// Corresponding to the ALGO_POW[] array in miners.hpp
// Also used in coins[COIN_COUNT] from tnn-common.hpp
#define ALGO_UNSUPPORTED 0
#define ALGO_ASTROBWTV3 10
#define ALGO_XELISV2 20
#define ALGO_SPECTRE_X 30
#define ALGO_RX0 40
#define ALGO_VERUS 50
#define ALGO_ASTRIX_HASH 60
#define ALGO_NXL_HASH 70
#define ALGO_HOOHASH 80
#define ALGO_WALA_HASH 90
#define ALGO_SHAI_HIVE 100
