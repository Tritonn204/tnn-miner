#include "gpuOCArgs.hpp"
#include "tnn_log.hpp"
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <algorithm>

namespace hip_oc_args
{

  // ──────────────────────────────────────────────────────────────────
  //  List parsing
  //
  //  Forgiving: strips all whitespace, blanks (,,) are skip-this-device,
  //  any token that's not an integer is also treated as a skip.  Never
  //  fails — always produces *something* so the rest of the list applies.
  //
  //    "300, 250 , 200"  → [300, 250, 200]
  //    "300,,200"        → [300, skip, 200]
  //    "300,x,200"       → [300, skip, 200]   + warn
  //    "300,"            → [300, skip]
  //    ",300"            → [skip, 300]
  //    ""                → []                 (caller reports "missing value")
  // ──────────────────────────────────────────────────────────────────

  static std::vector<Slot> parseIntList(const std::string &raw,
                                        const std::string &optName) // for warn msgs
  {
    // strip ALL whitespace so strtol can't silently eat some of it
    std::string s;
    s.reserve(raw.size());
    for (char c : raw)
      if (!std::isspace((unsigned char)c))
        s += c;

    std::vector<Slot> out;
    if (s.empty())
      return out;

    size_t pos = 0;
    int slot = 0;
    while (pos <= s.size())
    {
      size_t comma = s.find(',', pos);
      size_t end = (comma == std::string::npos) ? s.size() : comma;
      std::string tok = s.substr(pos, end - pos);

      if (tok.empty())
      {
        out.push_back({false, 0}); // intentional skip
      }
      else
      {
        char *e = nullptr;
        long v = std::strtol(tok.c_str(), &e, 10);
        if (*e == '\0')
        {
          out.push_back({true, (int)v});
        }
        else
        {
          TNN_LOG_ERROR("%s: slot %d value \"%s\" unparseable, skipping that GPU\n",
                        optName.c_str(), slot, tok.c_str());
          out.push_back({false, 0});
        }
      }

      ++slot;
      if (comma == std::string::npos)
        break;
      pos = comma + 1;
    }
    return out;
  }

  // ──────────────────────────────────────────────────────────────────
  //  Option-name matching
  //
  //    gpu-coff       → devIdx = -1
  //    gpu-coff3      → devIdx = 3
  //    gpu-coff(3)    → devIdx = 3
  //    gpu-coff-3     → devIdx = 3
  //    gpu-coff_3     → devIdx = 3
  //    gpu-coffee     → no match
  // ──────────────────────────────────────────────────────────────────
  static bool matchKnob(const std::string &tok, const char *base, int &devIdx)
  {
    size_t blen = std::strlen(base);
    if (tok.size() < blen || tok.compare(0, blen, base) != 0)
      return false;

    const char *s = tok.c_str() + blen;
    devIdx = -1;
    if (*s == '\0')
      return true;

    char open = 0;
    if (*s == '(' || *s == '-' || *s == '_')
      open = *s++;
    if (!std::isdigit((unsigned char)*s))
      return false;

    char *e = nullptr;
    long v = std::strtol(s, &e, 10);
    if (open == '(')
    {
      if (*e != ')')
        return false;
      ++e;
    }
    if (*e != '\0')
      return false;

    devIdx = (int)v;
    return true;
  }

  // Is the next argv token a value (maybe negative) or another flag?
  //   "300" yes   "-50" yes   "-t" no   "--foo" no
  static bool looksLikeValue(const char *s)
  {
    if (!s || !*s)
      return false;
    if (s[0] != '-')
      return true;
    if (s[1] == '-')
      return false;
    return std::isdigit((unsigned char)s[1]);
  }

  // ──────────────────────────────────────────────────────────────────
  //  Phase 1: extract()
  // ──────────────────────────────────────────────────────────────────
  ExtractResult extract(int argc, char **argv)
  {
    ExtractResult res;
    res.remaining.reserve(argc > 0 ? argc - 1 : 0);

    for (int i = 1; i < argc; ++i)
    {
      std::string a = argv[i];

      if (a.size() < 3 || a[0] != '-' || a[1] != '-')
      {
        res.remaining.push_back(std::move(a));
        continue;
      }

      // "--gpu-coff0=-50" → key="gpu-coff0" inlineVal="-50"
      std::string key, inlineVal;
      size_t eq = a.find('=', 2);
      if (eq == std::string::npos)
        key = a.substr(2);
      else
      {
        key = a.substr(2, eq - 2);
        inlineVal = a.substr(eq + 1);
      }

      bool consumed = false;
      for (const auto &def : kKnobs)
      {
        int devIdx;
        if (!matchKnob(key, def.name, devIdx))
          continue;

        std::string valStr;
        if (!inlineVal.empty())
          valStr = inlineVal;
        else if (i + 1 < argc && looksLikeValue(argv[i + 1]))
          valStr = argv[++i];
        else
        {
          res.errors.push_back(a + ": missing value");
          consumed = true;
          break;
        }

        auto slots = parseIntList(valStr, a);
        if (slots.empty())
        {
          res.errors.push_back(a + ": missing value");
        }
        else if (devIdx >= 0 && slots.size() > 1)
        {
          // --gpu-coff0 300,200  → keep first, drop the rest, warn
          TNN_LOG_ERROR("%s: list given with explicit device index, using first value only\n",
                       a.c_str());
          slots.resize(1);
          res.tuneArgs.push_back({def.knob, devIdx, std::move(slots), a});
        }
        else
        {
          res.tuneArgs.push_back({def.knob, devIdx, std::move(slots), a});
        }
        consumed = true;
        break;
      }

      if (!consumed)
        res.remaining.push_back(std::move(a));
    }
    return res;
  }

  // ──────────────────────────────────────────────────────────────────
  //  Phase 3: expand()
  //
  //  Hardcoded policy: repeat-last.
  //    --gpu-plimit 250             4 GPUs → 250 250 250 250
  //    --gpu-plimit 250,300         4 GPUs → 250 300 300 300
  //    --gpu-plimit 250,,300        4 GPUs → 250 --- 300 300
  //    --gpu-plimit 250,,           4 GPUs → 250 --- --- ---   (tail blanks don't pad)
  //    --gpu-plimit 1,2,3,4,5       4 GPUs → 1 2 3 4           (debug log surplus)
  //    --gpu-plimit 250 --gpu-plimit0 300 → dev0=300, rest=250
  // ──────────────────────────────────────────────────────────────────
  static void setKnob(GpuTuneParams &p, Knob k, int v)
  {
    switch (k)
    {
    case Knob::CoreOff:
      p.coreClockOffsetMHz = v;
      break;
    case Knob::CoreClock:
      p.coreClockMHz = v;
      break;
    case Knob::MemOff:
      p.memClockOffsetMHz = v;
      break;
    case Knob::PowerLimit:
      p.powerLimitW = v;
      break;
    }
  }

  std::vector<GpuTuneParams> expand(const std::vector<RawArg> &args,
                                    int deviceCount,
                                    std::vector<bool> *touched)
  {
    std::vector<GpuTuneParams> out((size_t)std::max(deviceCount, 0));
    if (touched)
      touched->assign(out.size(), false);
    if (deviceCount <= 0)
      return out;

    for (const auto &ra : args)
    {
      // ── explicit device index ────────────────────────────────
      if (ra.devIdx >= 0)
      {
        if (ra.devIdx >= deviceCount)
        {
          TNN_LOG_ERROR("%s: device %d out of range (%d GPU%s)\n",
                        ra.token.c_str(), ra.devIdx,
                        deviceCount, deviceCount == 1 ? "" : "s");
          continue;
        }
        if (ra.values[0].set)
        {
          setKnob(out[ra.devIdx], ra.knob, ra.values[0].val);
          if (touched)
            (*touched)[ra.devIdx] = true;
        }
        continue;
      }

      // ── list / scalar ────────────────────────────────────────
      const int nvals = (int)ra.values.size();

      // padValue = last *set* slot; if the list ends in blanks
      // (trailing comma), there's no pad — user opted out of the tail.
      int padVal = 0;
      bool havePad = false;
      for (int i = nvals - 1; i >= 0; --i)
        if (ra.values[i].set)
        {
          padVal = ra.values[i].val;
          havePad = true;
          break;
        }

      for (int d = 0; d < deviceCount; ++d)
      {
        if (d < nvals)
        {
          if (ra.values[d].set)
          {
            setKnob(out[d], ra.knob, ra.values[d].val);
            if (touched)
              (*touched)[d] = true;
          }
          // blank slot → explicitly leave this device alone
        }
        else if (havePad)
        {
          setKnob(out[d], ra.knob, padVal);
          if (touched)
            (*touched)[d] = true;
        }
      }

      if (nvals > deviceCount)
        TNN_LOG_DEBUG("%s: %d values, %d GPUs — last %d ignored\n",
                      ra.token.c_str(), nvals, deviceCount, nvals - deviceCount);
    }

    return out;
  }

} // namespace hip_oc_args