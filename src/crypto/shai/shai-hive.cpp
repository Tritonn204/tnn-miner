#include "shai-hive.h"

#include <numeric>
#include <string.h>
#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <openssl/sha.h>
#include <hex.h>

namespace ShaiHive
{
  int timeLimit = 500;

  // Helper function to convert hex string to integer
  template <typename T>
  T hexToType(const std::string &hexString)
  {
    static_assert(std::is_integral<T>::value, "Integral type required.");
    T number;
    std::stringstream ss;
    ss << std::hex << hexString;
    ss >> number;
    return number;
  }

  // Extract seed from hash
  uint64_t extractSeedFromHash(const std::string &hash)
  {
    return hexToType<uint64_t>(hash.substr(48, 64)); // Using first 16 characters (64 bits)
  }

  // Get grid size (V2 implementation)
  uint16_t getGridSize(const std::string &hash)
  {
    const int min_grid_size = 2000;
    const int max_grid_size = GRAPH_SIZE; // Using the GRAPH_SIZE constant
    std::string grid_size_segment = hash.substr(0, 8);
    unsigned long long grid_size = hexToType<unsigned long long>(grid_size_segment);
    auto grid_size_final = min_grid_size + (grid_size % (max_grid_size - min_grid_size));

    if (grid_size_final > max_grid_size)
    {
      grid_size_final = max_grid_size;
    }
    return static_cast<uint16_t>(grid_size_final);
  }

  uint16_t getGridSize(const uint8_t *hash) // or byte* depending on your type
  {
    const int min_grid_size = 2000;
    const int max_grid_size = GRAPH_SIZE;

    // Construct uint64_t from first 4 bytes
    unsigned long long grid_size =
        (static_cast<unsigned long long>(hash[0]) << 24) |
        (static_cast<unsigned long long>(hash[1]) << 16) |
        (static_cast<unsigned long long>(hash[2]) << 8) |
        (static_cast<unsigned long long>(hash[3]));

    auto grid_size_final = min_grid_size + (grid_size % (max_grid_size - min_grid_size));

    if (grid_size_final > max_grid_size)
    {
      grid_size_final = max_grid_size;
    }
    return static_cast<uint16_t>(grid_size_final);
  }

#if defined(__x86_64__)
  __attribute__((target("default")))
  std::vector<std::vector<bool>>
  generateGraph(const std::string &hash, size_t gridSize)
  {
    std::vector<std::vector<bool>> graph(gridSize, std::vector<bool>(gridSize, false));

    size_t numEdges = (gridSize * (gridSize - 1)) / 2;
    size_t bitsNeeded = numEdges; // One bit per edge

    // Extract seed from hash
    uint64_t seed = extractSeedFromHash(hash);

    // Initialize PRNG with seed
    std::mt19937_64 prng(seed);

    // Generate bits
    int bigSize = (bitsNeeded + 31) / 32;
    uint32_t bitStream[bigSize];

    for (size_t i = 0; i < bitsNeeded; i += 32)
    {
      bitStream[(i / 32)] = prng();
    }

#define GET_BIT(x, y) \
  (((x[(y) / 32]) >> (31 - ((y) % 32))) & 1)

    // Fill the adjacency matrix
    size_t bitIndex = 0;
    for (size_t i = 0; i < gridSize; ++i)
    {
      for (size_t j = i + 1; j < gridSize; ++j)
      {
        graph[i][j] = graph[j][i] = GET_BIT(bitStream, bitIndex);
        bitIndex++;
      }
    }

#undef GET_BIT

    return graph;
  }

  __attribute__((target("sse2")))
  std::vector<std::vector<bool>>
  generateGraph(const std::string &hash, size_t gridSize)
  {
    std::vector<std::vector<bool>> graph(gridSize, std::vector<bool>(gridSize, false));

    size_t numEdges = (gridSize * (gridSize - 1)) / 2;
    size_t bitsNeeded = numEdges; // One bit per edge

    // Extract seed from hash
    uint64_t seed = extractSeedFromHash(hash);

    // Initialize PRNG with seed
    std::mt19937_64 prng(seed);

    // Generate bits using SSE
    size_t numUInt32s = (bitsNeeded + 31) / 32;
    alignas(16) uint32_t bitStream[numUInt32s];

    for (size_t i = 0; i < numUInt32s; i += 4)
    {
      // Generate 4 uint32_t values at a time using SSE2
      __m128i randomBits = _mm_set_epi32(prng(), prng(), prng(), prng());
      randomBits = _mm_shuffle_epi32(randomBits, _MM_SHUFFLE(0, 1, 2, 3));

      _mm_storeu_si128(reinterpret_cast<__m128i *>(&bitStream[i]), randomBits);
    }

#define GET_BIT(x, y) \
  (((x[(y) / 32]) >> (31 - ((y) % 32))) & 1)

    // Fill the adjacency matrix using SSE for faster access
    size_t bitIndex = 0;
    for (size_t i = 0; i < gridSize; ++i)
    {
      for (size_t j = i + 1; j < gridSize; ++j)
      {
        // if (bitIndex < 2048) printf("%d", GET_BIT(bitStream, bitIndex));
        graph[i][j] = graph[j][i] = GET_BIT(bitStream, bitIndex);
        bitIndex++;
      }
    }
    // printf("\n");

#undef GET_BIT

    return graph;
  }

  // AVX2-optimized version of generateGraph
  __attribute__((target("avx2")))
  std::vector<std::vector<bool>>
  generateGraph(const std::string &hash, size_t gridSize)
  {
    std::vector<std::vector<bool>> graph(gridSize, std::vector<bool>(gridSize, false));

    size_t numEdges = (gridSize * (gridSize - 1)) / 2;
    size_t bitsNeeded = numEdges;

    // Extract seed from hash
    uint64_t seed = extractSeedFromHash(hash);

    // Initialize PRNG with seed
    std::mt19937_64 prng(seed);

    // Generate bits in 256-bit chunks using AVX2
    alignas(32) uint32_t bitStream[(bitsNeeded + 31) / 32];

    for (size_t i = 0; i < (bitsNeeded + 255) / 256; ++i)
    {
      // Generate eight 32-bit integers (256 bits total) and store them in an AVX2 register
      __m256i randomBits = _mm256_set_epi32(
          prng(), prng(), prng(), prng(),
          prng(), prng(), prng(), prng());
      randomBits = _mm256_permutevar8x32_epi32(randomBits,
                                               _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7));
      // Store the 256-bit chunk into bitStream
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(&bitStream[i * 8]), randomBits);
    }

// Macro to access specific bits in bitStream
#define GET_BIT(x, y) \
  (((x[(y) / 32]) >> (31 - ((y) % 32))) & 1)

    // Fill the adjacency matrix
    size_t bitIndex = 0;
    for (size_t i = 0; i < gridSize; ++i)
    {
      for (size_t j = i + 1; j < gridSize; ++j)
      {
        // if (bitIndex < 2048) printf("%d", GET_BIT(bitStream, bitIndex));
        graph[i][j] = graph[j][i] = GET_BIT(bitStream, bitIndex);
        bitIndex++;
      }
    }
    // printf("\n");

#undef GET_BIT

    return graph;
  }

  // AVX-512-optimized version of generateGraph
  __attribute__((target("avx512f")))
  std::vector<std::vector<bool>>
  generateGraph(const std::string &hash, size_t gridSize)
  {
    std::vector<std::vector<bool>> graph(gridSize, std::vector<bool>(gridSize, false));

    size_t numEdges = (gridSize * (gridSize - 1)) / 2;
    size_t bitsNeeded = numEdges;

    // Extract seed from hash
    uint64_t seed = extractSeedFromHash(hash);

    // Initialize PRNG with seed
    std::mt19937_64 prng(seed);

    // Generate bits in 512-bit chunks using AVX-512
    alignas(64) uint32_t bitStream[(bitsNeeded + 31) / 32];

    for (size_t i = 0; i < (bitsNeeded + 511) / 512; ++i)
    {
      // Generate sixteen 32-bit integers (512 bits total) and store them in an AVX-512 register
      __m512i randomBits = _mm512_set_epi32(
          (uint32_t)prng(), (uint32_t)prng(), (uint32_t)prng(), (uint32_t)prng(),
          (uint32_t)prng(), (uint32_t)prng(), (uint32_t)prng(), (uint32_t)prng(),
          (uint32_t)prng(), (uint32_t)prng(), (uint32_t)prng(), (uint32_t)prng(),
          (uint32_t)prng(), (uint32_t)prng(), (uint32_t)prng(), (uint32_t)prng());
      randomBits = _mm512_permutexvar_epi32(
          _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
          randomBits);

      // Store the 512-bit chunk into bitStream
      _mm512_storeu_si512(reinterpret_cast<__m512i *>(&bitStream[i * 16]), randomBits);
    }

// Macro to access specific bits in bitStream
#define GET_BIT(x, y) \
  (((x[(y) / 32]) >> (31 - ((y) % 32))) & 1)

    // Fill the adjacency matrix
    size_t bitIndex = 0;
    for (size_t i = 0; i < gridSize; ++i)
    {
      for (size_t j = i + 1; j < gridSize; ++j)
      {
        graph[i][j] = graph[j][i] = GET_BIT(bitStream, bitIndex);
        bitIndex++;
      }
    }

#undef GET_BIT

    return graph;
  }

  __attribute__((target("default")))
  void
  generateGraph_packed(const std::string &hash, size_t gridSize, HamiltonGraph::ShaiGraph &G)
  {
    G._size = gridSize;

    size_t numEdges = (gridSize * (gridSize - 1)) / 2;
    size_t bitsNeeded = numEdges; // One bit per edge

    // Extract seed from hash
    uint64_t seed = extractSeedFromHash(hash);

    // Initialize PRNG with seed
    std::mt19937_64 prng(seed);

    // Generate bits
    int bigSize = (bitsNeeded + 31) / 32;
    uint32_t bitStream[bigSize];

    for (size_t i = 0; i < bitsNeeded; i += 32)
    {
      G.d[(i / 32)] = prng();
    }
  }

  __attribute__((target("sse2")))
  void
  generateGraph_packed(const std::string &hash, size_t gridSize, HamiltonGraph::ShaiGraph &G)
  {
    G._size = gridSize;

    size_t numEdges = (gridSize * (gridSize - 1)) / 2;
    size_t bitsNeeded = numEdges; // One bit per edge

    // Extract seed from hash
    uint64_t seed = extractSeedFromHash(hash);

    // Initialize PRNG with seed
    std::mt19937_64 prng(seed);

    // Generate bits using SSE
    size_t numUInt32s = (bitsNeeded + 31) / 32;
    for (size_t i = 0; i < numUInt32s; i += 4)
    {
      // Generate 4 uint32_t values at a time using SSE2
      __m128i randomBits = _mm_set_epi32(prng(), prng(), prng(), prng());
      randomBits = _mm_shuffle_epi32(randomBits, _MM_SHUFFLE(0, 1, 2, 3));

      _mm_storeu_si128(reinterpret_cast<__m128i *>(&G.d[i]), randomBits);
    }
  }

  // AVX2-optimized version of generateGraph
  __attribute__((target("avx2")))
  void
  generateGraph_packed(const std::string &hash, size_t gridSize, HamiltonGraph::ShaiGraph &G)
  {
    G._size = gridSize;

    size_t numEdges = (gridSize * (gridSize - 1)) / 2;
    size_t bitsNeeded = numEdges;

    // Extract seed from hash
    uint64_t seed = extractSeedFromHash(hash);

    // Initialize PRNG with seed
    std::mt19937_64 prng(seed);

    // Generate bits in 256-bit chunks using AVX2
    for (size_t i = 0; i < (bitsNeeded + 255) / 256; ++i)
    {
      // Generate eight 32-bit integers (256 bits total) and store them in an AVX2 register
      __m256i randomBits = _mm256_set_epi32(
          prng(), prng(), prng(), prng(),
          prng(), prng(), prng(), prng());
      randomBits = _mm256_permutevar8x32_epi32(randomBits,
                                               _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7));
      // Store the 256-bit chunk into bitStream
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(&G.d[i * 8]), randomBits);
    }
  }

  // AVX-512-optimized version of generateGraph
  __attribute__((target("avx512f")))
  void
  generateGraph_packed(const std::string &hash, size_t gridSize, HamiltonGraph::ShaiGraph &G)
  {
    G._size = gridSize;

    size_t numEdges = (gridSize * (gridSize - 1)) / 2;
    size_t bitsNeeded = numEdges;

    // Extract seed from hash
    uint64_t seed = extractSeedFromHash(hash);

    // Initialize PRNG with seed
    std::mt19937_64 prng(seed);

    // Generate bits in 512-bit chunks using AVX-512
    for (size_t i = 0; i < (bitsNeeded + 511) / 512; ++i)
    {
      // Generate sixteen 32-bit integers (512 bits total) and store them in an AVX-512 register
      __m512i randomBits = _mm512_set_epi32(
          (uint32_t)prng(), (uint32_t)prng(), (uint32_t)prng(), (uint32_t)prng(),
          (uint32_t)prng(), (uint32_t)prng(), (uint32_t)prng(), (uint32_t)prng(),
          (uint32_t)prng(), (uint32_t)prng(), (uint32_t)prng(), (uint32_t)prng(),
          (uint32_t)prng(), (uint32_t)prng(), (uint32_t)prng(), (uint32_t)prng());
      randomBits = _mm512_permutexvar_epi32(
          _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
          randomBits);

      // Store the 512-bit chunk into bitStream
      _mm512_storeu_si512(reinterpret_cast<__m512i *>(&G.d[i * 16]), randomBits);
    }
  }

  __attribute__((target("default")))
  void freevIota (uint16_t *freev) {
    std::iota(freev, &freev[GRAPH_SIZE], 1);
  }

  __attribute__((target("avx2")))
  void freevIota (uint16_t *freev) {
    int i;

    __m256i sum = _mm256_set_epi16(
      16,15,14,13,12,11,10,9,8,
      7,6,5,4,3,2,1
    );

    __m256i inc = _mm256_set1_epi16(16);

    #pragma unroll
    for (i = 0; i + 15 < GRAPH_SIZE; i += 16) {
      _mm256_store_si256(reinterpret_cast<__m256i *>(&freev[i]), sum);
      sum = _mm256_add_epi16(sum,inc);
      // _mm256_storeu_si256(reinterpret_cast<__m256i *>(&freev[i]), _mm256_set_epi16(
      //   i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8,
      //   i+9, i+10, i+11, i+12, i+13, i+14, i+15, i+16
      // ));
    }
    // if (i < GRAPH_SIZE)
    for (; i < GRAPH_SIZE; i++) {
      freev[i] = i+1;
    }
  }

  __attribute__((target("default")))
  static void resetPath (uint16_t *path) {
    for (int i = 0; i < GRAPH_SIZE; i++) {
      path[i] = -1;
    }
  }

  __attribute__((target("avx2")))
   static void resetPath (uint16_t *path) {
    int i;

    #pragma unroll
    for (i = 0; i + 15 < GRAPH_SIZE; i += 16) {
      _mm256_store_si256(reinterpret_cast<__m256i *>(&path[i]), _mm256_set1_epi16(-1));
    }
    // if (i < GRAPH_SIZE)
    for (; i < GRAPH_SIZE; i++) {
      path[i] = -1;
    }
  }

#elif defined(__aarch64__)
  std::vector<std::vector<bool>> generateGraph(const std::string &hash, size_t gridSize)
  {
    std::vector<std::vector<bool>> graph(gridSize, std::vector<bool>(gridSize, false));

    size_t numEdges = (gridSize * (gridSize - 1)) / 2;
    size_t bitsNeeded = numEdges; // One bit per edge

    // Extract seed from hash
    uint64_t seed = extractSeedFromHash(hash);

    // Initialize PRNG with seed
    std::mt19937_64 prng(seed);

    // Generate bits using NEON
    size_t numUInt32s = (bitsNeeded + 31) / 32;
    alignas(16) uint32_t bitStream[numUInt32s];

    for (size_t i = 0; i < numUInt32s; i += 4)
    {
      // Generate 4 uint32_t values at a time
      uint32_t p1 = prng();
      uint32_t p2 = prng();
      uint32_t p3 = prng();
      uint32_t p4 = prng();

      uint32x4_t randomBits = vdupq_n_u32(0); // Initialize the vector
      randomBits = vsetq_lane_u32(p1, randomBits, 0);
      randomBits = vsetq_lane_u32(p2, randomBits, 1);
      randomBits = vsetq_lane_u32(p3, randomBits, 2);
      randomBits = vsetq_lane_u32(p4, randomBits, 3);

      // Store the generated randomBits into bitStream
      vst1q_u32(&bitStream[i], randomBits);
    }

#define GET_BIT(x, y) \
  (((x[(y) / 32]) >> (31 - ((y) % 32))) & 1)

    // Fill the adjacency matrix using bitStream directly
    size_t bitIndex = 0;
    for (size_t i = 0; i < gridSize; ++i)
    {
      for (size_t j = i + 1; j < gridSize; ++j)
      {
        graph[i][j] = graph[j][i] = GET_BIT(bitStream, bitIndex);
        bitIndex++;
      }
    }

#undef GET_BIT

    return graph;
  }

  void freevIota (uint16_t *freev) {
    std::iota(freev, &freev[GRAPH_SIZE], 1);
  }

  void generateGraph_packed(const std::string &hash, size_t gridSize, HamiltonGraph::ShaiGraph &G) {
    G._size = gridSize;

    size_t numEdges = (gridSize * (gridSize - 1)) / 2;
    size_t bitsNeeded = numEdges; // One bit per edge

    // Extract seed from hash
    uint64_t seed = extractSeedFromHash(hash);

    // Initialize PRNG with seed
    std::mt19937_64 prng(seed);

    // Generate bits
    int bigSize = (bitsNeeded + 31) / 32;
    uint32_t bitStream[bigSize];

    for (size_t i = 0; i < bitsNeeded; i += 32)
    {
      G.d[(i / 32)] = prng();
    }
  }
#endif

  static void benchmarkLoop()
  {
    const int NUM_HASHES = 10000;
    ShaiCtx ctx;
    uint16_t *nonce = reinterpret_cast<uint16_t *>(ctx.data); // Treat first 2 bytes as nonce

    // Start timing
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < NUM_HASHES; ++i)
    {
      if (hash(ctx, ctx.data))
      {
        // You might want to handle found solutions separately
      }
      nonce[0]++; // Increment nonce
    }

    // End timing
    auto end = std::chrono::steady_clock::now();

    // Calculate duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double seconds = duration.count() / 1e6;

    // Calculate and print hash rate
    double hashrate_khs = NUM_HASHES / seconds / 1000.0;

    std::cout << std::fixed << std::setprecision(2)
              << "Hashed " << NUM_HASHES << " in " << seconds << " seconds. "
              << "Speed: " << hashrate_khs << " KH/s" << std::endl;
  }

  // Benchmark function
  template <typename F>
  static double benchmark(F func)
  {
    auto start = std::chrono::steady_clock::now();
    func();
    auto end = std::chrono::steady_clock::now();
    // std::cout << "took " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;
    return std::chrono::duration<double, std::milli>(end - start).count();
  }

  static inline void SHA256(SHA256_CTX &sha256, const uint8_t *input, uint8_t *digest, unsigned long inputSize)
  {
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, input, inputSize);
    SHA256_Final(digest, &sha256);
  }

  void tuneTimeLimit()
  {
    printf("pre tune\n");
    std::vector<int> times;
    alignas(64) HamiltonGraph::ShaiGraph gREF;
    alignas(64) uint16_t freev[GRAPH_SIZE];
    uint16_t path[GRAPH_SIZE];
    for (int i = 0; i < 100; i++)
    {
      uint8_t HASH_bytes[32];
      for (int j = 0; j < 32; j++)
      {
        HASH_bytes[j] = rand() % 256;
      }
      std::string HASH = hexStr(HASH_bytes, 32);
      int nchecks = 0;
      times.push_back(benchmark([&]()
                                { 
        size_t gridSize = getGridSize(HASH);

        generateGraph_packed(HASH, gridSize, gREF);

        
        std::fill_n(path, gridSize, -1);
        path[0] = 0;

        freevIota(freev);
        HamiltonGraph::solveGraph_opt(gREF, path, 1, freev, nchecks); }));
    }

    std::sort(times.begin(), times.end());
    int median = times[times.size() / 2];

    timeLimit = (int)(median * 1.5) + 1;
    // setcolor(BRIGHT_YELLOW);
    printf("Tuned Hamilton Graph VDF time limit to %dms\n", timeLimit);
    fflush(stdout);
    // setcolor(BRIGHT_WHITE);
  }

  int test()
  {
    std::string in = "000000203bb93d254cc2e50bd5f7905625984a687b4a314ca805165c9588d74305000000b6238a92a621aec22dbe7449f37bf603aeaef3efcaabc4608c2ca1452c2a353cf6302867577b061d";
    uint8_t DATA[80 + GRAPH_SIZE * 2]{0};
    uint8_t SHA[32];

    hexstrToBytes(in, DATA);
    memset(DATA + 80, 0xFF, GRAPH_SIZE * 2);

    SHA256_CTX ctx_sha;
    SHA256(ctx_sha, DATA, SHA, 80 + GRAPH_SIZE * 2);

    std::reverse(SHA, SHA + 32);
    std::string HASH = hexStr(SHA, 32);
    printf("initial SHA: %s\n", HASH.c_str());
    if(0 == strcmp("947c0b0fce52de20d6e1eaec85791ecb16ca9eeb8d07690e9537ef16cd410e6e", HASH.c_str())) {
      printf("Initial Hash is correct\n");
    } else {
      printf("Initial hash was incorrect.\n");
      return 1;
    }

    size_t gridSize = getGridSize(HASH);
    printf("gridSize = %ld\n", gridSize);
    alignas(64) HamiltonGraph::ShaiGraph graph;
    uint16_t path[GRAPH_SIZE];
    generateGraph_packed(HASH, gridSize, graph);
    std::fill_n(path, gridSize, -1);
    path[0] = 0;

    alignas(64) uint16_t freev[GRAPH_SIZE];
    freevIota(freev);

    int nchecks = 0;

    bool found = HamiltonGraph::solveGraph(graph, path, 1, freev, 500);
    if (found)
    {
      if (gridSize < GRAPH_SIZE)
      {
        std::fill(&path[gridSize], &path[GRAPH_SIZE], 0xFFFF);
      }

      // for (int i = 0; i < path.size(); i++) {
      //   printf("%d, ", path[i]);
      // }
      // printf("\n\n");

      memset(DATA + 80 + 4000, 0xFF, 16);
      memcpy(DATA + 80, path, GRAPH_SIZE * sizeof(uint16_t));
      SHA256(ctx_sha, DATA, SHA, 80 + GRAPH_SIZE * 2);
      std::reverse(SHA, SHA + 32);
      HASH = hexStr(SHA, 32);
      printf("final SHA: %s\n", HASH.c_str());
      if(0 == strcmp("439097b932de6864e0dca529b74dd2085f193f9e9c04164b90b4f8bbf21f8f9e", HASH.c_str())) {
        printf("final Hash is correct\n");
      } else {
        printf(" expected: 439097b932de6864e0dca529b74dd2085f193f9e9c04164b90b4f8bbf21f8f9e\n");
        return 2;
      }
    }

    benchmarkLoop();
    return 0;
  }

  bool hash(ShaiCtx &ctx, uint8_t *data)
  {
    memcpy(ctx.data, data, 80);
    memset(ctx.data + 80, 0xFF, GRAPH_SIZE * 2);

    SHA256_CTX ctx_sha;
    SHA256(ctx_sha, ctx.data, ctx.sha, 80 + GRAPH_SIZE * 2);

    std::reverse(ctx.sha, ctx.sha + 32);
    std::string HASH = hexStr(ctx.sha, 32);
    size_t gridSize = getGridSize(HASH);

    // ctx.graph = generateGraph(HASH, gridSize);
    generateGraph_packed(HASH, gridSize, ctx.packedGraph);
    // ctx.path.resize(gridSize);
    // ctx.path = std::vector<uint16_t>(gridSize, -1);
    std::fill_n(ctx.path, GRAPH_SIZE, -1);
    ctx.path[0] = 0;

    freevIota(ctx.freev);

    int nchecks = 0;

    bool found = HamiltonGraph::solveGraph_opt(ctx.packedGraph, ctx.path, 1, ctx.freev, nchecks);
    if (found)
    {
      // if (gridSize < GRAPH_SIZE)
      // {
      //   std::fill(&ctx.path[gridSize], &ctx.path[GRAPH_SIZE], 0xFFFF);
      // }
      SHA256_CTX ctx_sha_two;
      memcpy(ctx.data + 80, ctx.path, GRAPH_SIZE * sizeof(uint16_t));
      SHA256(ctx_sha_two, ctx.data, ctx.sha, 80 + GRAPH_SIZE * 2);
      std::reverse(ctx.sha, ctx.sha + 32);

      return true;
    }
    return false;
  }
}