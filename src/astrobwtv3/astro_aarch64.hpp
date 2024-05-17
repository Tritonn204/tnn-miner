#include <arm_neon.h>
#include <bitset>

#include <fnv1a.h>
#include <xxhash64.h>
#include <highwayhash/sip_hash.h>
#include "astrobwtv3.h"

#include "lookup.h"

void branchComputeCPU_aarch64(workerData &worker, bool isTest);