#include "sha_detect.h"

#include <cstdio>

int main()
{
    // The former Windows constructor exited before this marker whenever
    // stdout was redirected. This executable has no GPU dependencies.
    std::puts("SHA_STARTUP_MAIN");

    const int supported = has_sha_ni_support();
    if ((supported != 0 && supported != 1) ||
        has_sha_ni_support_cached() != supported ||
        has_sha_ni_support_cached() != supported)
    {
        return 1;
    }

    std::printf("SHA_STARTUP_PASS supported=%d\n", supported);
    return 0;
}
