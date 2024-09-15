__device__ static size_t
feed_text(uint64_t *stateBuffer, uint8_t *buf, const uint8_t *text, size_t bytes, size_t bits, const char *suffix, size_t off, size_t byterate, size_t *bitoffp)
{
  size_t n, bitoff;

  if (off)
  {
    n = bytes < byterate - off ? bytes : byterate - off;
    memcpy(&buf[off], text, n);
    off += n;
    if (off == byterate)
    {
      hipkeccak_zerocopy_update(stateBuffer, buf, byterate);
      off = 0;
    }
    text = &text[n];
    bytes -= n;
  }
  if (bytes)
  {
    n = bytes;
    n -= bytes %= byterate;
    hipkeccak_zerocopy_update(stateBuffer, text, n);
    text = &text[n];
  }
  memcpy(&buf[off], text, bytes + !!bits);
  off += bytes;
  bitoff = bits;
  if (!bitoff)
    buf[off] = 0;
  for (; *suffix; suffix++)
  {
    if (*suffix == '1')
      buf[off] |= static_cast<uint8_t>(1 << bitoff); // Explicit cast to uint8_t
    if (++bitoff == 8)
    {
      if (++off == byterate)
      {
        hipkeccak_zerocopy_update(stateBuffer, buf, byterate);
        off = 0;
      }
      bitoff = 0;
      buf[off] = 0;
    }
  }

  *bitoffp = bitoff;
  return off;
}

__device__ static size_t
encode_left(uint64_t *stateBuffer, uint8_t *buf, size_t byterate, size_t value, size_t off)
{
  size_t x, n, j, i = off;

  for (x = value, n = 0; x; x >>= 8)
    n += 1;
  if (!n)
    n = 1;
  buf[i++] = static_cast<uint8_t>(n); // Explicit cast to uint8_t in C++
  if (i == byterate)
  {
    hipkeccak_zerocopy_update(stateBuffer, buf, byterate);
    i = 0;
  }

  for (j = 0; j < n;)
  {
    buf[i++] = static_cast<uint8_t>(value >> ((n - ++j) << 3)); // Explicit cast to uint8_t
    if (i == byterate)
    {
      hipkeccak_zerocopy_update(stateBuffer, buf, byterate);
      i = 0;
    }
  }

  return i;
}

__device__ static void hipkeccak_cshake_initialise(uint64_t *stateBuffer, // Remove restrict
                                                   uint8_t *memBuffer, uint8_t *s_text, size_t s_len)
{
  size_t off = 0, bitoff = 0;
  size_t byterate = static_cast<size_t>(1088) >> 3;

  // Function calls with adjusted parameters
  off = encode_left(stateBuffer, memBuffer, byterate, byterate, off); // Explicit cast for state->M
  off = encode_left(stateBuffer, memBuffer, byterate, 0, off);

  off = encode_left(stateBuffer, memBuffer, byterate, (15 << 3), off);
  off = feed_text(stateBuffer, memBuffer, s_text, s_len, 0, "", off, byterate, &bitoff);

  memset(&memBuffer[off], 0, 137 - off);
  hipkeccak_zerocopy_update(stateBuffer, memBuffer, 137);
}