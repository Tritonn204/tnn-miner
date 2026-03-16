#include "hipkill.h"
#include <tnn_hip/common/gpu_compat.hpp>

void hipDeviceReset_wrapper() {
  oroDeviceReset();
}