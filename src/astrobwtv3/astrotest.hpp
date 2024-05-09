#ifndef astrotest_h
#define astrotest_h

#include <bit>
#include <bitset>
#include <filesystem>
#include <fstream>

#include "astrobwtv3.h"
#include "lookup.h"
#include "lookupcompute.h"
#include "num.h"

#include "hex.h"

#include <xxhash64.h>

#if defined(__aarch64__)
  #include "astro_aarch64.hpp"
#endif

struct OpTestResult {
  unsigned char input[256];
  unsigned char result[256];
  std::chrono::nanoseconds duration_ns;
};

struct PowTest{
	std::string out;
	std::string in;
	bool expectFail = false;
};

inline PowTest random_pow_tests[] = {
	{"54e2324ddacc3f0383501a9e5760f85d63e9bc6705e9124ca7aef89016ab81ea", "a"},
	{"faeaff767be60134f0bcc5661b5f25413791b4df8ad22ff6732024d35ec4e7d0", "ab"},
	{"715c3d8c61a967b7664b1413f8af5a2a9ba0005922cb0ba4fac8a2d502b92cd6", "abc"},
	{"74cc16efc1aac4768eb8124e23865da4c51ae134e29fa4773d80099c8bd39ab8", "abcd"},
	{"d080d0484272d4498bba33530c809a02a4785368560c5c3eac17b5dacd357c4b", "abcde"},
	{"813e89e0484cbd3fbb3ee059083af53ed761b770d9c245be142c676f669e4607", "abcdef"}, 
	{"3972fe8fe2c9480e9d4eff383b160e2f05cc855dc47604af37bc61fdf20f21ee", "abcdefg"},
	{"f96191b7e39568301449d75d42d05090e41e3f79a462819473a62b1fcc2d0997", "abcdefgh"},
	{"8c76af6a57dfed744d5b7467fa822d9eb8536a851884aa7d8e3657028d511322", "abcdefghi"},
	{"f838568c38f83034b2ff679d5abf65245bd2be1b27c197ab5fbac285061cf0a7", "abcdefghij"},
	{"ff9f23980870b4dd3521fcf6807b85d8bf70c5fbbd9736c87c23fac0114e2b8b", "4145bd000025fbf83b29cddc000000009b6d4f3ecaaaea9e99ff5630b7c9d01d000000000e326f0593a9000000339a10", true}
};

template <std::size_t N>
inline void generateRandomBytes(std::uint8_t (&iv_buff)[N])
{
  auto const hes = std::random_device{}();

  using random_bytes_engine = std::independent_bits_engine<std::default_random_engine,
                                                           CHAR_BIT, unsigned short>;

  random_bytes_engine rbe;
  rbe.seed(hes);

  std::generate(std::begin(iv_buff), std::end(iv_buff), std::ref(rbe));
}

void runDivsufsortBenchmark();

int DeroTesting(int testOp, int testLen, bool useLookup);
int runDeroOpTests(int testOp, int dataLen=15);

int TestAstroBWTv3(bool useLookup);
int TestAstroBWTv3repeattest(bool useLookup);

void optest_ref(int op, workerData &worker, byte testData[32], OpTestResult &testRes, bool print=true);
void optest_branchcpu(int op, workerData &worker, byte testData[32], OpTestResult &testRes, bool print=true);

void optest_lookup(int op, workerData &worker, byte testData[32], OpTestResult &testRes, bool print=true);

void optest_avx2(int op, workerData &worker, byte testData[32], OpTestResult &testRes, bool print=true);

#if defined(__aarch64__)
void optest_aarch64(int op, workerData &worker, byte testData[32], OpTestResult &testRes, bool print=true);
#endif

#endif
