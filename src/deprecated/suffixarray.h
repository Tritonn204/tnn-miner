#ifndef suffixarray
#define suffixarray

#include <iostream>
#include <sstream>

#include <stdint.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include <regex>

#include "sa_fast.h"
#include "binary.h"

inline const int realMaxData32 = 2 ^ (31) - 1;
inline int maxData32 = realMaxData32;

using byte = unsigned char;

inline const int bufSize = 16 << 10; // reasonable for BenchmarkSaveRestore

inline const char *toChar(char *bytes)
{
  return reinterpret_cast<char *>(bytes);
}

inline const unsigned char *toByte(char *bytes)
{
  return reinterpret_cast<unsigned char *>(bytes);
}

struct ints
{
  std::vector<int32_t> int32;
  std::vector<int64_t> int64;

  int len() const;
  int64_t get(int i) const;
  void set(int i, int64_t v);
  ints slice(int i, int j) const;
};

struct Index
{
  std::vector<byte> data;
  ints sa;

  void Read(std::istream &r);
  void Write(std::ostream &w);
  std::vector<byte> Bytes() const;
  std::vector<byte> at(int i) const;
  ints lookupAll(const std::vector<byte> &s) const;
  std::vector<int> Lookup(const std::vector<byte> &s, int n) const;
  std::vector<std::vector<int>> FindAllIndex(std::string pattern, std::regex &r, int n) const;
  int Len() const;
  bool Less(int i, int j) const;
  void Swap(int i, int j);
};

Index *New(byte *data, int dataLen);

void writeInt(std::ostream &w, std::stringstream buf, int x);
std::pair<int64_t, int> readInt(std::istream &r, byte *buf);

std::pair<int, int> writeSlice(std::ostream &w, byte *buf, const ints &data, int dataLen);

class TooBigException : public std::exception
{
public:
  const char *what() const noexcept override
  {
    return "suffixarray: data too large";
  }
};

std::pair<int, int> readSlice(std::istream &r, byte *buf, ints &data);

inline std::vector<byte> to_vector(std::string const &str)
{
  // don't forget the trailing 0...
  return std::vector<byte>(str.data(), str.data() + str.length() + 1);
}

inline int ints::len() const
{
  return int32.size() + int64.size();
}
inline int64_t ints::get(int i) const
{
  if (!int32.empty())
    return (int64_t)(int32[i]);
  return int64[i];
}
inline void ints::set(int i, int64_t v)
{
  if (!int32.empty())
  {
    int32[i] = (int32_t)v;
  }
  else
  {
    int64[i] = v;
  }
}
inline ints ints::slice(int i, int j) const
{
  if (!int32.empty())
  {
    return ints{std::vector<int32_t>(int32.begin() + i, int32.begin() + j), std::vector<int64_t>()};
  } else if (int64.empty()) {
    return ints{std::vector<int32_t>(), std::vector<int64_t>(int64.begin() + i, int64.begin() + j)};
  }
  return ints{std::vector<int32_t>(),std::vector<int64_t>()};
}

inline void Index::Read(std::istream &r)
{
  byte buf[bufSize];

  int64_t n64;
  r.read(reinterpret_cast<char *>(buf), MaxVarintLen64);
  n64 = readInt(r, buf).first;

  if (n64 < 0)
  {
    throw TooBigException();
  }
  int n = static_cast<int>(n64);

  if (2 * n < data.capacity() || data.capacity() < n || (sa.int32.size() != 0 && n > maxData32) || (sa.int64.size() != 0 && n <= maxData32))
  {
    data = std::vector<byte>(n);
    sa.int32.clear();
    sa.int64.clear();
    if (n <= maxData32)
    {
      sa.int32 = std::vector<int32_t>(n);
    }
    else
    {
      sa.int64 = std::vector<int64_t>(n);
    }
  }
  else
  {
    data.resize(n);
    sa = sa.slice(0, n);
  }

  r.read(reinterpret_cast<char *>(data.data()), n);

  ints saCopy = sa;
  while (saCopy.len() > 0)
  {
    std::pair<int, int> result = readSlice(r, buf, saCopy);
    int numElements = result.first;
    int err = result.second;
    if (err != 0)
    {
      throw std::runtime_error("Error occurred");
    }
    saCopy = saCopy.slice(numElements, saCopy.len());
  }
}
inline void Index::Write(std::ostream &w)
{
  byte b[bufSize];
  std::string b2(b, b + sizeof(b));

  writeInt(w, std::stringstream(b2), static_cast<int64_t>(data.size()));
  w.write(reinterpret_cast<char *>(data.data()), data.size());

  ints saCopy = sa;
  while (saCopy.len() > 0)
  {
    std::pair<int, int> result = writeSlice(w, b, saCopy, saCopy.len());
    int numElements = result.first;
    int err = result.second;
    if (err != 0)
    {
      throw std::runtime_error("Error occurred");
    }
    saCopy = saCopy.slice(numElements, saCopy.len());
  }
}
inline std::vector<byte> Index::Bytes() const
{
  return data;
}
inline std::vector<byte> Index::at(int i) const
{
  return std::vector<byte>(data.begin() + (sa.get(i)), data.end());
}

inline ints Index::lookupAll(const std::vector<byte> &s) const
{
  byte v32[sizeof(sa.int32)];
  byte v64[sizeof(sa.int64)];

  byte *v;
  int size;

  if (sa.int32.empty())
  {
    memcpy(v32, &sa.int32, sizeof(sa.int32));
    size = sizeof(sa.int32);
    v = v32;
  }
  else
  {
    memcpy(v64, &sa.int64, sizeof(sa.int64));
    size = sizeof(sa.int64);
    v = v64;
  }
  // find matching suffix index range [i:j]
  // find the first index where s would be the prefix
  int i = std::lower_bound(v, v + size, s.data(), [&](const byte &a, const byte *b) -> bool
                           { return std::lexicographical_compare(&a, &a + size, b, b + s.size()); }) -
          v;
  // starting at i, find the first index at which s is not a prefix
  int j = i + std::lower_bound(v + i, v + size, s.data(), [&](const byte &a, const byte *b) -> bool
                               { return !std::equal(&a, &a + size, b, b + s.size()); }) -
          v;
  return sa.slice(i, j);
}

inline std::vector<int> Index::Lookup(const std::vector<byte> &s, int n) const
{
  std::vector<int> result;

  if (!s.empty() && n != 0)
  {
    ints matches = lookupAll(s);
    int count = matches.len();

    if (n < 0 || count < n)
    {
      n = count;
    }

    // 0 <= n <= count
    if (n > 0)
    {
      result.resize(n);

      if (!matches.int32.empty())
      {
        for (int i = 0; i < n; i++)
        {
          result[i] = static_cast<int>(matches.int32[i]);
        }
      }
      else
      {
        for (int i = 0; i < n; i++)
        {
          result[i] = static_cast<int>(matches.int64[i]);
        }
      }
    }
  }

  return result;
}

inline std::vector<std::vector<int>> Index::FindAllIndex(std::string pattern, std::regex &r, int n) const
{
  std::string prefix;
  bool complete = false;

  if (pattern.empty())
  {
    return {};
  }

  // Check if there is a non-empty literal prefix
  std::smatch match;
  std::regex_search(pattern, match, std::regex("^\\[\\^]*\\^?([^.*+?{()|\\[\\]\\\\]+)"));
  if (match.empty())
  {
    prefix = "";
  }
  else
  {
    prefix = match[1].str();
    complete = (pattern == prefix);
  }

  std::vector<byte> lit(prefix.begin(), prefix.end());

  std::vector<std::vector<int>> result;

  // Worst-case scenario: no literal prefix
  if (prefix == "")
  {
    std::string dataStr(data.begin(), data.end());
    std::smatch match;
    std::string::const_iterator searchStart(dataStr.cbegin());
    while (std::regex_search(searchStart, dataStr.cend(), match, r))
    {
      std::vector<int> indices;
      indices.push_back(match.position());
      indices.push_back(match.position() + match.length());
      result.push_back(indices);
      searchStart = match.suffix().first;
      if (result.size() >= n)
        break;
    }
    return result;
  }

  // If regexp is a literal, just use Lookup and convert the result into match pairs
  if (complete)
  {
    for (int n1 = n;; n1 += 2 * (n - result.size()))
    {
      std::vector<int> indices = Lookup(lit, n1);
      if (indices.empty())
      {
        return result;
      }
      std::sort(indices.begin(), indices.end());
      std::vector<int> pairs(2 * indices.size());
      int count = 0;
      int prev = 0;
      for (int i : indices)
      {
        if (count == n)
        {
          break;
        }
        // Ignore indices leading to overlapping matches
        if (prev <= i)
        {
          pairs[2 * count] = i;
          pairs[2 * count + 1] = i + lit.size();
          result.push_back({pairs[2 * count], pairs[2 * count + 1]});
          count++;
          prev = i + lit.size();
        }
      }
      result.resize(count);
      if (result.size() >= n || indices.size() != n1)
      {
        // Found all matches or there's no chance to find more
        // (n and n1 can be negative)
        break;
      }
    }
    if (result.empty())
    {
      result.clear();
    }
    return result;
  }

  // Regexp has a non-empty literal prefix
  r = std::regex("^" + pattern);

  for (int n1 = n;; n1 += 2 * (n - result.size()))
  {
    std::vector<int> indices = Lookup(lit, n1);
    if (indices.empty())
    {
      return result;
    }
    std::sort(indices.begin(), indices.end());
    int pairs[2*indices.size()];
    result.clear();
    int prev = 0;
    for (int i : indices)
    {
      if (result.size() == n)
      {
        break;
      }
      if (prev <= i)
      {
        std::vector<int> indices;
        indices.push_back(i);
        indices.push_back(i+lit.size());
        result.push_back(indices);
        prev = i + lit.size();
      }
    }
    if (result.size() >= n || indices.size() != n1)
    {
      // Found all matches or there's no chance to find more
      // (n and n1 can be negative)
      break;
    }
  }
  if (result.empty())
  {
    result.clear();
  }
  return result;
}

inline int Index::Len() const { return sa.len(); }
inline bool Index::Less(int i, int j) const
{
  return std::memcmp(at(i).data(), at(j).data(), std::min(at(i).size(), at(j).size())) < 0;
}

inline void Index::Swap(int i, int j)
{
  if (!sa.int32.empty())
  {
    std::swap(sa.int32[i], sa.int32[j]);
  }
  else
  {
    std::swap(sa.int64[i], sa.int64[j]);
  }
}

#endif