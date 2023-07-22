#ifndef POWTEST
#define POWTEST

#include <string>

using byte = unsigned char;

void init_basic();

struct PowTest{
	std::string out;
	std::string in;
};

template<class T, size_t N, class V>
std::array<T, N> to_array(const V& v)
{
    assert(v.size() == N);
    std::array<T, N> d;
    std::copy(v.begin(), v.end(), d.data());
    return d;
}

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
  {"00017c48ba235f82c2fa2b06993e9393fe9f629b4630f98c731abd5e4f08bcb9", "41a1dd0000229d3a47100c65000000005b4fd5bd674799179c3819a83d70a84600000000a1cc177b2f9fee0000000d01"}
};

void TestAstroBWTv3();
void TestAstroBWTv3repeattest();

#endif