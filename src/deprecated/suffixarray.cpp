#include <iostream>
#include <sstream>

#include <binary.h>

#include "suffixarray.h"
#include "sais2.h"

using byte = unsigned char;

Index *New(byte *data, int dataLen) {
  Index* ix = new Index();
  ix->data = std::vector<byte>(data,data+dataLen);
  ix->sa.int32 = std::vector<int32_t>();
  ix->sa.int64 = std::vector<int64_t>();

  if (dataLen <= maxData32) {
      ix->sa.int32.resize(dataLen);
      text_32(data, ix->sa.int32.data(), dataLen, ix->sa.int32.size());
  } else {
      ix->sa.int64.resize(dataLen);
      text_64(data, ix->sa.int64.data(), dataLen, ix->sa.int64.size());
  }

  return ix;
}

void writeInt(std::ostream& w, std::stringstream buf, int x) {
    buf << x;
    std::string str = buf.str();
    w.write(str.c_str(), MaxVarintLen64);
}

std::pair<int64_t, int> readInt(std::istream& r, byte* buf) {
    r.read(reinterpret_cast<char *>(buf), MaxVarintLen64);
    int err = r.fail() ? -1 : 0;
    int64_t x = 0;
    if (err == 0) {
        std::stringstream stream(std::string(reinterpret_cast<char *>(buf), r.gcount()));
        stream >> x;
    }
    return std::make_pair(x, err);
}

Index* New(const std::vector<byte>& data) {
    Index* ix = new Index();
    ix->data = data;
    if (data.size() <= maxData32) {
        ix->sa.int32.resize(data.size());
        byte d[data.size()];
        memcpy(d, data.data(), data.size());
        text_32(d, ix->sa.int32.data(), data.size(), data.size());
    } else {
        ix->sa.int64.resize(data.size());
        byte d[data.size()];
        byte c[data.size()];
        memcpy(d, data.data(), data.size());
        text_64(d, ix->sa.int64.data(), data.size(), data.size());
        
    }
    return ix;
}

std::pair<int, int> writeSlice(std::ostream& w, byte* buf, const ints& data, int dataLen) {
    int p = MaxVarintLen64;
    int n = 0;
    int m = data.len();

    while (n < m && p + MaxVarintLen64 <= MaxVarintLen64) {
        uint64_t value = static_cast<uint64_t>(data.get(n));
        p += PutUvarint(reinterpret_cast<byte*>(buf + p), value);
        n++;
    }

    PutVarint(reinterpret_cast<uint8_t*>(buf), static_cast<int64_t>(p));

    w.write(reinterpret_cast<const char*>(buf), p);
    int err = w.fail() ? -1 : 0;

    return std::make_pair(n, err);
}

std::pair<int, int> readSlice(std::istream& r, byte* buf, ints& data) {
  // Read buffer size
  int64_t size64;
  r.read(reinterpret_cast<char*>(buf), MaxVarintLen64);
  size64 = readInt(r, buf).first;

  if ((int)size64 != size64 || size64 < 0) {
    // We never write chunks this big anyway.
    throw TooBigException();
  }
  int size = static_cast<int>(size64);

  // Read buffer without the size
  r.read(reinterpret_cast<char*>(buf) + MaxVarintLen64, size);

  // Decode as many elements as present in buf
  int p = MaxVarintLen64;
  int n = 0;
  while (p < size) {
    int64_t x = readInt(r, buf + p).first;
    int w = PutUvarint(buf + p, static_cast<uint64_t>(x));
    data.set(n, x);
    p += w;
    n++;
  }

  return {n, 0};
}