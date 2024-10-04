#include "hipkill.h"
#include <hip/hip_runtime.h>

void hipDeviceReset_wrapper() {
  hipDeviceReset();
}