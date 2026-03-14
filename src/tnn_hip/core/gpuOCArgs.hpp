#pragma once
#include "gpuOC.hip.h"
#include <cstdint>
#include <vector>
#include <string>
#include <array>

namespace hip_oc_args
{

  enum class Knob : uint8_t
  {
    CoreOff,
    CoreClock,
    MemOff,
    PowerLimit
  };
  struct KnobDef
  {
    const char *name;
    Knob knob;
  };

  inline constexpr std::array<KnobDef, 4> kKnobs = {{
      {"gpu-coff", Knob::CoreOff},
      {"gpu-cclock", Knob::CoreClock},
      {"gpu-moff", Knob::MemOff},
      {"gpu-plimit", Knob::PowerLimit},
  }};

  struct Slot
  {
    bool set;
    int val;
  };

  struct RawArg
  {
    Knob knob;
    int devIdx; // -1 = list/broadcast
    std::vector<Slot> values;
    std::string token; // "--gpu-coff0" for diagnostics
  };

  struct ExtractResult
  {
    std::vector<RawArg> tuneArgs;
    std::vector<std::string> remaining; // feed to po::command_line_parser
    std::vector<std::string> errors;    // only truly unrecoverable ones
  };

  ExtractResult extract(int argc, char **argv);

  std::vector<GpuTuneParams> expand(const std::vector<RawArg> &args,
                                    int deviceCount,
                                    std::vector<bool> *touched = nullptr);

} // namespace hip_oc_args