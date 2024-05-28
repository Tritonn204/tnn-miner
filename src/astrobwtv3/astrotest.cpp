#include <astrotest.hpp>
#include <fstream>

#if defined(__AVX2__)
void testPopcnt256_epi8() {
    __m256i data = _mm256_setzero_si256();
    __m256i expected = _mm256_setzero_si256();

    // Test all possible byte values
    for (uint32_t i = 0; i < 256; ++i) {
        // Set the byte value in the data vector
        data = _mm256_set1_epi8(static_cast<int8_t>(i));

        // Calculate the expected population count for the byte value
        uint8_t popcnt = __builtin_popcount(i);
        expected = _mm256_set1_epi8(static_cast<int8_t>(popcnt));

        // Perform the population count using the function
        __m256i result = popcnt256_epi8(data);

        // Compare the result with the expected value
        if (!_mm256_testc_si256(result, expected)) {
            std::cout << "Test failed for byte value: " << i << std::endl;
            return;
        }
    }

    std::cout << "testPopcnt256_epi8 tests passed" << std::endl;
}
#endif

void mapZeroes() {
  std::vector<std::tuple<int,int,int>> opMap; 
  std::vector<std::tuple<int,int,int>> noChange; 

  for (int o = 0; o < 256; o++) {
    for (int i = 0; i < 256; i++) {
      for (int j = 0; j < 256; j++) {
        byte r = i;
        branchResult(r, o, j);
        if (std::find(branchedOps_global.begin(), branchedOps_global.end(), i) == branchedOps_global.end()) {
          if (r == 0) opMap.push_back(std::tuple<int,int,int>(o,i,j));
          if (r == i) noChange.push_back(std::tuple<int,int,int>(o,i,j));
          break;
        }
      }
    }
  }

  std::ofstream outputFile("unchangedOps.txt");
  if (outputFile.is_open()) {
    for (const auto& [op, input, salt] : noChange) {
      outputFile << "op: " << op << ", input: " << input << ", salt: " << salt << std::endl;
    }
  }
  outputFile.close();
  std::cout << "Output written to unchangedOps.txt" << std::endl;

  outputFile.open("clippedOps.txt");
  if (outputFile.is_open()) {
    for (const auto& [op, input, salt] : opMap) {
      outputFile << "op: " << op << ", input: " << input << ", salt: " << salt << std::endl;
    }
  }
  outputFile.close();
  std::cout << "Output written to clippedOps.txt" << std::endl;
}

int DeroTesting(int testOp, int testLen, bool useLookup) {
  int failedTests = 0;
  Num diffTest("1234567890123456789", 10);

  if (testLen >= 0) {
    failedTests += runDeroOpTests(testOp, testLen);
  } else {
    failedTests += runDeroOpTests(testOp);
  }

  failedTests += TestAstroBWTv3(useLookup);
  // TestAstroBWTv3_cuda();

  return failedTests;
}

const uint64_t NUM_ITERATIONS = 1000;

#pragma clang optimize off
void benchmarkSIMDMath() {
    alignas(32) uint8_t prev_chunk[32] = {0};
    alignas(32) uint8_t chunk[32] = {0};
    int pos1 = 0;
    int pos2 = 16;

    prev_chunk[pos2] = 29;

    // Fill the first 15 bytes of prev_chunk with 0x00
    for (int i = 0; i < pos2; ++i) {
        prev_chunk[i] = 0;
    }

    // Fill the remaining bytes of prev_chunk with random values
    srand(0);
    for (int i = pos2 + 1; i < 32; ++i) {
        prev_chunk[i] = static_cast<uint8_t>(rand());
    }

    uint64_t accumulator = 0;
    auto start = std::chrono::steady_clock::now();

    for (uint64_t i = 0; i < NUM_ITERATIONS; ++i) {
        __m256i data = _mm256_loadu_si256((__m256i*)&prev_chunk[pos1]);
        __m256i old = data;

        data = _mm256_and_si256(data, _mm256_set1_epi8(prev_chunk[pos2]));
        data = _mm256_xor_si256(data, _mm256_set1_epi8(prev_chunk[pos2]));
        data = _mm256_xor_si256(data, popcnt256_epi8(data));
        data = _mm256_sllv_epi8(data, _mm256_and_si256(data, _mm256_set1_epi8(3)));

        data = _mm256_blendv_epi8(old, data, genMask(pos2 - pos1));
        _mm256_storeu_si256((__m256i*)&chunk[pos1], data);

        accumulator += chunk[pos1];
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    std::cout << "SIMD Math Operations: " << duration.count() << " nanoseconds" << std::endl;
    std::cout << "Accumulator: " << accumulator << std::endl;
}

void benchmarkLoadCompare() {
    alignas(32) uint8_t prev_chunk[32] = {0};
    alignas(32) uint8_t chunk[32] = {0};
    int pos1 = 0;
    int pos2 = 16;

    workerData *controlWorker = (workerData*)malloc_huge_pages(sizeof(workerData));
    initWorker(*controlWorker);
    lookupGen(*controlWorker, nullptr, nullptr);

    prev_chunk[pos2] = 29;

    // Fill the first 15 bytes of prev_chunk with 0x00
    for (int i = 0; i < pos2; ++i) {
        prev_chunk[i] = 0;
    }

    // Fill the remaining bytes of prev_chunk with random values
    srand(0);
    for (int i = pos2 + 1; i < 32; ++i) {
        prev_chunk[i] = static_cast<uint8_t>(rand());
    }

    uint64_t accumulator = 0;
    auto start = std::chrono::steady_clock::now();

    for (uint64_t i = 0; i < NUM_ITERATIONS; ++i) {

        __m256i data = _mm256_loadu_si256((__m256i*)&prev_chunk[pos1]);
        // if ((result & mask) == mask) {
          byte newVal = controlWorker->lookup3D[controlWorker->branched_idx[116]*256*256 + prev_chunk[pos2]*256 + prev_chunk[pos1]];

          __m256i newVec = _mm256_set1_epi8(newVal);
          data = _mm256_blendv_epi8(data, newVec, genMask(pos2-pos1));
          _mm256_storeu_si256((__m256i*)&chunk[pos1], data);
          accumulator += chunk[pos1];
        // } 
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    std::cout << "Load, Compare, Broadcast, Store: " << duration.count() << " nanoseconds" << std::endl;
    std::cout << "Accumulator: " << accumulator << std::endl;
}
#pragma clang optimize on

int runDeroOpTests(int testOp, int dataLen) {

  benchmarkSIMDMath();
  benchmarkLoadCompare();

  bool useLookup = false;
  int numOpsFailed = 0;
  #if defined(__AVX2__)
  testPopcnt256_epi8();
  #endif

  workerData *controlWorker = (workerData*)malloc_huge_pages(sizeof(workerData));
  initWorker(*controlWorker);
  lookupGen(*controlWorker, nullptr, nullptr);

  workerData *testWorker = (workerData*)malloc_huge_pages(sizeof(workerData));
  initWorker(*testWorker);
  lookupGen(*testWorker, nullptr, nullptr);

  workerData *z_testWorker = (workerData*)malloc_huge_pages(sizeof(workerData));
  initWorker(*z_testWorker);
  lookupGen(*z_testWorker, nullptr, nullptr);

  controlWorker->pos1 = 0; controlWorker->pos2 = dataLen+1;
  testWorker->pos1 = 0; testWorker->pos2 = dataLen+1;
  z_testWorker->pos1 = 0; z_testWorker->pos2 = dataLen+1;

  z_testWorker->isSame = true;

  byte test[32];
  //byte test2[32];
  std::srand(time(NULL));
  generateInitVector<32>(test);
  memset(test, 0, dataLen);
  
  printf("Initial Input\n");
  for (int i = 0; i < 32; i++) {
    printf("%02x", test[i]);
  }
  printf("\n");

  std::string resultText = std::string("Lookup");
  void (*testFunc)(int op, workerData &worker, byte testData[32], OpTestResult &testRes, bool print);
  // the ampersand is actually optional
  testFunc = &optest_branchcpu;
  if(useLookup) {
    testFunc = &optest_lookup;
  } else {
    #if defined(__AVX2__)
    resultText = "AVX2";
    testFunc = &optest_avx2;
    #elif defined(__x86_64__)
    resultText = "Branch";
    testFunc = &optest_branchcpu;
    #endif
    #if defined(__aarch64__)
    resultText = "AA64";
    testFunc = &optest_aarch64;
    #endif
  }

  int startOp = 0;
  int maxOp = 255;
  if(testOp >= 0) {
    startOp = testOp;
    maxOp = testOp;
  }
                                            //
  printf("%-7s:   Branch vs %-7s ns         - Valid| Branch vs %-7s ns        - Valid\n", resultText.c_str(), resultText.c_str(), "AVX2_zOp");
  for(int op = startOp; op <= maxOp; op++) {
    // WARMUP, don't print times
    OpTestResult *controlResult = new OpTestResult;
    OpTestResult *testResult = new OpTestResult;
    OpTestResult *z_testResult = new OpTestResult;
    // WARMUP, don't print times

    controlWorker->pos1 = 0; controlWorker->pos2 = dataLen+1;
    //memset(&controlWorker->step_3, 0, 256);
    //memcpy(&controlWorker->step_3, test, dataLen);
    optest_branchcpu(0, *controlWorker, test, *controlResult, false);

    controlWorker->pos1 = 0; controlWorker->pos2 = dataLen+1;    
    //memset(&controlWorker->step_3, 0, 256);
    //memcpy(&controlWorker->step_3, test, dataLen);
    optest_branchcpu(op, *controlWorker, test, *controlResult, false);
    //printf("  Op: %3d - %6ld ns\n", op, controlResult->duration_ns);

    z_testWorker->pos1 = 0; z_testWorker->pos2 = dataLen+1;
    testFunc(op, *z_testWorker, test, *z_testResult, false);

    z_testWorker->pos1 = 0; z_testWorker->pos2 = dataLen+1;
    testFunc(op, *z_testWorker, test, *z_testResult, false);

    testWorker->pos1 = 0; testWorker->pos2 = dataLen+1;
    testFunc(op, *testWorker, test, *testResult, false);

    testWorker->pos1 = 0; testWorker->pos2 = dataLen+1;
    testFunc(op, *testWorker, test, *testResult, false);

    auto control_dur = controlResult->duration_ns.count();
    auto test_dur = testResult->duration_ns.count();
    auto z_test_dur = z_testResult->duration_ns.count();

    auto percent_speedup = double(double(control_dur-test_dur)/double(test_dur))*100;
    auto z_percent_speedup = double(double(control_dur-z_test_dur)/double(z_test_dur))*100;
    bool valid = 0 == memcmp(controlResult->result, testResult->result, dataLen);
    bool z_valid = 0 == memcmp(controlResult->result, z_testResult->result, dataLen);
    char isOpt = ' ';
    if(testWorker->opt[op]) {
      isOpt = '*';
    }
    printf("%cOp: %3d - %7ld / %7ld = %6.2f %% - %s | %7ld / %7ld = %6.2f %% - %s\n", isOpt, op, controlResult->duration_ns.count(), testResult->duration_ns.count(), 
      percent_speedup, valid ? "true" : "false", controlResult->duration_ns.count(), z_testResult->duration_ns.count(), z_percent_speedup, z_valid ? "true" : "false");
    if(!valid) {
      numOpsFailed++;
      printf("Vanilla: ");
      for (int i = 0; i < dataLen+1; i++) {
        printf("%02x", controlResult->result[i]);
      }
      printf("\n");
      printf("%7s: ", resultText.c_str());
      for (int i = 0; i < dataLen+1; i++) {
        printf("%02x", testResult->result[i]);
      }
      printf("\n");
    }

    if(!z_valid) {
      numOpsFailed++;
      printf("Vanilla: ");
      for (int i = 0; i < dataLen+1; i++) {
        printf("%02x", controlResult->result[i]);
      }
      printf("\nZ optimized: ");
      printf("%7s: ", resultText.c_str());
      for (int i = 0; i < dataLen+1; i++) {
        printf("%02x", z_testResult->result[i]);
      }
      printf("\n");
    }
  }
  return numOpsFailed;
}

int rakeDeroOpTests(int testOp, int dataLen) {
  bool useLookup = false;
  int numOpsFailed = 0;
  #if defined(__AVX2__)
  testPopcnt256_epi8();
  #endif

  workerData *controlWorker = (workerData*)malloc_huge_pages(sizeof(workerData));
  initWorker(*controlWorker);
  lookupGen(*controlWorker, nullptr, nullptr);

  workerData *testWorker = (workerData*)malloc_huge_pages(sizeof(workerData));
  initWorker(*testWorker);
  lookupGen(*testWorker, nullptr, nullptr);

  controlWorker->pos1 = 0; controlWorker->pos2 = dataLen+1;
  testWorker->pos1 = 0; testWorker->pos2 = dataLen+1;

  byte test[32];

  for (int o = 0; o < 256; o++) {
    //byte test2[32];
    std::srand(time(NULL));
    generateInitVector<32>(test);
    memset(test, o, dataLen);

    printf("Initial Input\n");
    for (int i = 0; i < 32; i++) {
      printf("%02x", test[i]);
    }
    printf("\n");

    std::string resultText = std::string("Lookup");
    void (*testFunc)(int op, workerData &worker, byte testData[32], OpTestResult &testRes, bool print);
    // the ampersand is actually optional
    testFunc = &optest_branchcpu;
    if(useLookup) {
      testFunc = &optest_lookup;
    } else {
      #if defined(__AVX2__)
      resultText = "AVX2";
      testFunc = &optest_avx2;
      #elif defined(__x86_64__)
      resultText = "Branch";
      testFunc = &optest_branchcpu;
      #endif
      #if defined(__aarch64__)
      resultText = "AA64";
      testFunc = &optest_aarch64;
      #endif
    }

    int startOp = 0;
    int maxOp = 255;
    if(testOp >= 0) {
      startOp = testOp;
      maxOp = testOp;
    }

    // printf("%-7s:   Branch vs %-7s ns         - Valid\n", resultText.c_str(), resultText.c_str());
    for(int op = startOp; op <= maxOp; op++) {
      // WARMUP, don't print times
      OpTestResult *controlResult = new OpTestResult;
      OpTestResult *testResult = new OpTestResult;
      // WARMUP, don't print times

      controlWorker->pos1 = 0; controlWorker->pos2 = dataLen;
      //memset(&controlWorker->step_3, 0, 256);
      //memcpy(&controlWorker->step_3, test, dataLen);
      optest_branchcpu(0, *controlWorker, test, *controlResult, false);

      controlWorker->pos1 = 0; controlWorker->pos2 = dataLen;    
      //memset(&controlWorker->step_3, 0, 256);
      //memcpy(&controlWorker->step_3, test, dataLen);
      optest_branchcpu(op, *controlWorker, test, *controlResult, false);
      //printf("  Op: %3d - %6ld ns\n", op, controlResult->duration_ns);

      testWorker->pos1 = 0; testWorker->pos2 = dataLen;
      testFunc(op, *testWorker, test, *testResult, false);

      auto control_dur = controlResult->duration_ns.count();
      auto test_dur = testResult->duration_ns.count();

      auto percent_speedup = double(double(control_dur-test_dur)/double(test_dur))*100;
      bool valid = 0 == memcmp(controlResult->result, testResult->result, dataLen);
      char isOpt = ' ';
      if(testWorker->opt[op]) {
        isOpt = '*';
      }
      // printf("%cOp: %3d - %7ld / %7ld = %6.2f %% - %s\n", isOpt, op, controlResult->duration_ns.count(), testResult->duration_ns.count(), percent_speedup, valid ? "true" : "false");
      if(!valid) {
        printf("input: %d, op: %d\n", o, op);
        numOpsFailed++;
        printf("Vanilla: ");
        for (int i = 0; i < dataLen+1; i++) {
          printf("%02x", controlResult->result[i]);
        }
        printf("\n");
        printf("%7s: ", resultText.c_str());
        for (int i = 0; i < dataLen+1; i++) {
          printf("%02x", testResult->result[i]);
        }
        printf("\n");
      }
    }
  }

  return numOpsFailed;
}

int TestAstroBWTv3(bool useLookup=false)
{
  int failedTests = 0;
  std::srand(1);
  int n = -1;
  workerData *worker = (workerData *)malloc_huge_pages(sizeof(workerData));
  initWorker(*worker);
  lookupGen(*worker, nullptr, nullptr);
  //workerData *worker2 = (workerData *)malloc_huge_pages(sizeof(workerData));
  //initWorker(*worker2);
  //lookupGen(*worker2, nullptr, nullptr);

  int i = 0;
  for (PowTest t : random_pow_tests)
  {
    // if (i > 0)
    //   break;
    byte *buf = new byte[t.in.size()];
    memcpy(buf, t.in.c_str(), t.in.size());
    byte res[32];
    //byte res2[32];
    //AstroBWTv3(buf, (int)t.in.size(), res2, *worker2, true);
    // printf("vanilla result: %s\n", hexStr(res, 32).c_str());
    AstroBWTv3(buf, (int)t.in.size(), res, *worker, useLookup);
    // printf("lookup result: %s\n", hexStr(res, 32).c_str());
    std::string actualRes = hexStr(res, 32);
    if (actualRes.c_str() != t.out)
    {
      if(t.expectFail) {
        printf("SUCCESS! Pow function failed as expected! pow(%s) expected !%s received  %s\n", t.in.c_str(), t.out.c_str(), actualRes.c_str());
      } else {
        printf("FAIL. pow(%s) expected %s received %s\n", t.in.c_str(), t.out.c_str(), actualRes.c_str());
        failedTests++;
      }

      // Section below is for debugging modifications to the branched compute operation

      // debugOpOrder = true;
      // worker = (workerData *)malloc_huge_pages(sizeof(workerData));
      // initWorker(*worker);
      // lookupGen(*worker, lookup2D, lookup3D);
      // AstroBWTv3(buf, (int)t.in.size(), res2, *worker, false);
      // worker2 = (workerData *)malloc_huge_pages(sizeof(workerData));
      // initWorker(*worker2);
      // lookupGen(*worker2, lookup2D, lookup3D);
      // AstroBWTv3(buf, (int)t.in.size(), res, *worker2, true);
      // debugOpOrder = false;
    }
    else
    {
      printf("SUCCESS! pow(%s) expected %s received %s\n", t.in.c_str(), t.out.c_str(), actualRes.c_str());
    }

    // for (const auto& [a, b, c] : worker->repeats) {
    //   printf("op: %d, input: %d, dataLen: %d\n", a, b, c);
    // }

    // printf("repeated branch ops: %d, total ops: %d\n", worker->repeats.size(), worker->tries);

    delete[] buf;
    i++;
  }

  byte *data = new byte[48];
  byte *data2 = new byte[48];

  std::string c("7199110000261dfb0b02712100000000c09a113bf2050b1e55c79d15116bd94e00000000a9041027027fa800000314bb");
  std::string c2("7199110000261dfb0b02712100000000c09a113bf2050b1e55c79d15116bd94e00000000a9041027027fa800002388bb");
  hexstrToBytes(c, data);
  hexstrToBytes(c2, data2);

  printf("Str: %s\nHex: %s\n", c.c_str(), hexStr(data, 48).c_str());
  printf("Str: %s\nHex: %s\n", c2.c_str(), hexStr(data2, 48).c_str());

  failedTests += TestAstroBWTv3repeattest(true);
  failedTests += TestAstroBWTv3repeattest(false);

  // for (int i = 0; i < 1024; i++)
  // {
  //   std::generate(buf.begin(), buf.end(), [&dist, &gen]()
  //                 { return dist(gen); });
  //   std::memcpy(random_data, buf.data(), buf.size());

  //   // std::cout << hexStr(data, 48) << std::endl;
  //   // std::cout << hexStr(random_data, 48) << std::endl;

  //   if (i % 2 == 0)
  //   {
  //     byte res[32];
  //     AstroBWTv3(data, 48, res, *worker, false);

  //     // hexStr(res, 64);
  //     std::string s = hexStr(res, 32);f
  //     if (s != "c392762a462fd991ace791bfe858c338c10c23c555796b50f665b636cb8c8440")
  //     {
  //       printf("%d test failed hash %s\n", i, s.c_str());
  //     }
  //   }
  //   else
  //   {
  //     byte res[32];
  //     AstroBWTv3(buf.data(), 48, res, *worker, false);
  //   }
  // }
  // std::cout << "Repeated test over" << std::endl;
  // libcubwt_free_device_storage(storage);
  // cudaFree(storage);
  return failedTests;
}

int TestAstroBWTv3repeattest(bool useLookup)
{
  int rc = 0;
  workerData *worker = (workerData *)malloc_huge_pages(sizeof(workerData));
  initWorker(*worker);
  lookupGen(*worker, nullptr, nullptr);

  byte *data = new byte[48];
  byte random_buffer[48];

  std::string c("419ebb000000001bbdc9bf2200000000635d6e4e24829b4249fe0e67878ad4350000000043f53e5436cf610000086b00");
  hexstrToBytes(c, data);

  for (int i = 0; i < 1024; i++)
  {
    generateRandomBytes<48>(random_buffer);

    // std::cout << hexStr(data, 48) << std::endl;
    // std::cout << hexStr(random_buffer, 48) << std::endl;

    if (i % 2 == 0)
    {
      byte res[32];
      AstroBWTv3(data, 48, res, *worker, false);

      // hexStr(res, 64);
      std::string s = hexStr(res, 32);
      if (s != "c392762a462fd991ace791bfe858c338c10c23c555796b50f665b636cb8c8440")
      {
        printf("%d test failed hash %s\n", i, s.c_str());
        rc += 1;
      }
    }
    else
    {
      byte res[32];
      AstroBWTv3(random_buffer, 48, res, *worker, false);
    }
    if(rc >= 5) {
      return rc;
    }
  }
  std::cout << "Repeated test over" << std::endl;
  return rc;
}


void optest_ref(int op, workerData &worker, byte testData[32], OpTestResult &testRes, bool print) {
  // Set us up the bomb
  memset(worker.step_3, 0, 256);
  memcpy(worker.step_3, testData, 32);
  if (print) {
    printf("Scalar:\n");
    //printf("---------------\n");
    printf("SC Input %3d  : ", op);
    for (int i = worker.pos1; i < worker.pos1 + 32; i++) {
      printf("%02X ", worker.step_3[i]);
    }
    printf("\n");
  }

  auto start = std::chrono::steady_clock::now();
  for(int n = 0; n < 256; n++){
        switch (op)
    {
    case 0:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random

        // INSERT_RANDOM_CODE_END
        worker.t1 = worker.step_3[worker.pos1];
        worker.t2 = worker.step_3[worker.pos2];
        worker.step_3[worker.pos1] = reverse8(worker.t2);
        worker.step_3[worker.pos2] = reverse8(worker.t1);
      }
      break;
    case 1:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] += worker.step_3[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 2:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 3:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 4:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 5:
    {
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {

        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right

        // INSERT_RANDOM_CODE_END
      }
    }
    break;
    case 6:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -

        // INSERT_RANDOM_CODE_END
      }
      break;
    case 7:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                             // +
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 8:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
        worker.step_3[i] = std::rotl(worker.step_3[i], 10); // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 5);// rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 9:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 10:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
        worker.step_3[i] *= worker.step_3[i];              // *
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 11:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 6); // rotate  bits by 1
        // worker.step_3[i] = std::rotl(worker.step_3[i], 5);            // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 12:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] *= worker.step_3[i];               // *
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 13:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 14:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 15:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 16:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] *= worker.step_3[i];               // *
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 17:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 18:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 9);  // rotate  bits by 3
        // worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
        // worker.step_3[i] = std::rotl(worker.step_3[i], 5);         // rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 19:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] += worker.step_3[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 20:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 21:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] += worker.step_3[i];                             // +
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 22:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 23:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 4); // rotate  bits by 3
        // worker.step_3[i] = std::rotl(worker.step_3[i], 1);                           // rotate  bits by 1
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 24:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 25:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 26:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                 // *
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 27:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 28:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 29:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] += worker.step_3[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 30:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 31:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] *= worker.step_3[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 32:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 33:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
        worker.step_3[i] *= worker.step_3[i];                             // *
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 34:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 35:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];              // +
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
        worker.step_3[i] = std::rotl(worker.step_3[i], 1); // rotate  bits by 1
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 36:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);   // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 37:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] *= worker.step_3[i];                             // *
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 38:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 39:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 40:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 41:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
        worker.step_3[i] -= (worker.step_3[i] ^ 97);        // XOR and -
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 42:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 4); // rotate  bits by 1
        // worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 43:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] += worker.step_3[i];                             // +
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 44:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 45:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 10); // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 5);                       // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 46:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 47:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 48:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        // worker.step_3[i] = ~worker.step_3[i];                    // binary NOT operator
        // worker.step_3[i] = ~worker.step_3[i];                    // binary NOT operator
        worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 49:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 50:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);     // reverse bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
        worker.step_3[i] += worker.step_3[i];              // +
        worker.step_3[i] = std::rotl(worker.step_3[i], 1); // rotate  bits by 1
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 51:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];     // XOR
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 52:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 53:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 54:

#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);  // reverse bits
        worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
        // worker.step_3[i] = ~worker.step_3[i];    // binary NOT operator
        // worker.step_3[i] = ~worker.step_3[i];    // binary NOT operator
        // INSERT_RANDOM_CODE_END
      }

      break;
    case 55:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 56:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] *= worker.step_3[i];               // *
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 57:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = std::rotl(worker.step_3[i], 8);                // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = reverse8(worker.step_3[i]); // reverse bits
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 58:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] += worker.step_3[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 59:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 60:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 61:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = std::rotl(worker.step_3[i], 8);             // rotate  bits by 3
        // worker.step_3[i] = std::rotl(worker.step_3[i], 5);// rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 62:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] += worker.step_3[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 63:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] += worker.step_3[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 64:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];     // XOR
        worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] *= worker.step_3[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 65:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 8); // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] *= worker.step_3[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 66:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 67:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);   // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 68:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 69:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 70:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 71:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 72:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 73:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 74:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 75:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 76:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 77:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 78:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 79:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] += worker.step_3[i];               // +
        worker.step_3[i] *= worker.step_3[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 80:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] += worker.step_3[i];                             // +
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 81:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 82:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
        // worker.step_3[i] = ~worker.step_3[i];        // binary NOT operator
        // worker.step_3[i] = ~worker.step_3[i];        // binary NOT operator
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 83:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 84:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] += worker.step_3[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 85:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 86:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 87:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];               // +
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] += worker.step_3[i];               // +
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 88:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
        worker.step_3[i] *= worker.step_3[i];               // *
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 89:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];               // +
        worker.step_3[i] *= worker.step_3[i];               // *
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 90:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);     // reverse bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 6); // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 91:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 92:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 93:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] += worker.step_3[i];                             // +
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 94:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 95:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
        worker.step_3[i] = std::rotl(worker.step_3[i], 10); // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 96:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);   // rotate  bits by 2
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);   // rotate  bits by 2
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 97:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 98:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 99:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 100:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 101:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 102:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
        worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
        worker.step_3[i] += worker.step_3[i];              // +
        worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 103:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 104:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
        worker.step_3[i] += worker.step_3[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 105:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 106:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
        worker.step_3[i] *= worker.step_3[i];               // *
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 107:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], 6);             // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 108:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 109:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 110:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 111:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 112:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
        worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
        worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 113:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 6); // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 1);                           // rotate  bits by 1
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = ~worker.step_3[i];                 // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 114:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 115:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 116:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 117:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 118:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 119:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
        worker.step_3[i] ^= worker.step_3[worker.pos2];     // XOR
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 120:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] *= worker.step_3[i];               // *
        worker.step_3[i] ^= worker.step_3[worker.pos2];     // XOR
        worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 121:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] *= worker.step_3[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 122:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 123:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] = std::rotl(worker.step_3[i], 6);                // rotate  bits by 3
        // worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 124:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] ^= worker.step_3[worker.pos2];     // XOR
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 125:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 126:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 9); // rotate  bits by 3
        // worker.step_3[i] = std::rotl(worker.step_3[i], 1); // rotate  bits by 1
        // worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
        worker.step_3[i] = reverse8(worker.step_3[i]); // reverse bits
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 127:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 128:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 129:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 130:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 131:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] *= worker.step_3[i];                 // *
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 132:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 133:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 134:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 135:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 136:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 137:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 138:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
        worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
        worker.step_3[i] += worker.step_3[i];           // +
        worker.step_3[i] -= (worker.step_3[i] ^ 97);    // XOR and -
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 139:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 8); // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 140:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] ^= worker.step_3[worker.pos2];     // XOR
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 141:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] += worker.step_3[i];                 // +
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 142:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 143:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 144:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 145:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 146:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 147:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] *= worker.step_3[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 148:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 149:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
        worker.step_3[i] = reverse8(worker.step_3[i]);  // reverse bits
        worker.step_3[i] -= (worker.step_3[i] ^ 97);    // XOR and -
        worker.step_3[i] += worker.step_3[i];           // +
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 150:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 151:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 152:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 153:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 4); // rotate  bits by 1
        // worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
        // worker.step_3[i] = ~worker.step_3[i];     // binary NOT operator
        // worker.step_3[i] = ~worker.step_3[i];     // binary NOT operator
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 154:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
        worker.step_3[i] = ~worker.step_3[i];                 // binary NOT operator
        worker.step_3[i] ^= worker.step_3[worker.pos2];       // XOR
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 155:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] ^= worker.step_3[worker.pos2];       // XOR
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];       // XOR
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 156:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = std::rotl(worker.step_3[i], 4);             // rotate  bits by 3
        // worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 157:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 158:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);    // rotate  bits by 3
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 159:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 160:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 4);             // rotate  bits by 1
        // worker.step_3[i] = std::rotl(worker.step_3[i], 3);    // rotate  bits by 3
        // INSERT_RANDOM_CODE_END
      }
      break;
    case 161:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 162:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];               // *
        worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] -= (worker.step_3[i] ^ 97);        // XOR and -
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 163:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 164:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                 // *
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] = ~worker.step_3[i];                 // binary NOT operator
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 165:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] += worker.step_3[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 166:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] += worker.step_3[i];               // +
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 167:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        // worker.step_3[i] = ~worker.step_3[i];        // binary NOT operator
        // worker.step_3[i] = ~worker.step_3[i];        // binary NOT operator
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 168:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 169:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 170:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);   // XOR and -
        worker.step_3[i] = reverse8(worker.step_3[i]); // reverse bits
        worker.step_3[i] -= (worker.step_3[i] ^ 97);   // XOR and -
        worker.step_3[i] *= worker.step_3[i];          // *
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 171:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);    // rotate  bits by 3
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 172:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 173:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] += worker.step_3[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 174:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 175:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
        worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 176:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] *= worker.step_3[i];              // *
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 177:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 178:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] += worker.step_3[i];                             // +
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 179:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 180:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 181:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 182:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] = std::rotl(worker.step_3[i], 6); // rotate  bits by 1
        // worker.step_3[i] = std::rotl(worker.step_3[i], 5);         // rotate  bits by 5
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 183:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];        // +
        worker.step_3[i] -= (worker.step_3[i] ^ 97); // XOR and -
        worker.step_3[i] -= (worker.step_3[i] ^ 97); // XOR and -
        worker.step_3[i] *= worker.step_3[i];        // *
                                                     // INSERT_RANDOM_CODE_END
      }
      break;
    case 184:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] *= worker.step_3[i];                          // *
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 185:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 186:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 187:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
        worker.step_3[i] += worker.step_3[i];              // +
        worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 188:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 189:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] ^= worker.step_3[worker.pos2];     // XOR
        worker.step_3[i] -= (worker.step_3[i] ^ 97);        // XOR and -
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 190:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 191:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                             // +
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 192:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] *= worker.step_3[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 193:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 194:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 195:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);   // rotate  bits by 2
        worker.step_3[i] ^= worker.step_3[worker.pos2];       // XOR
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 196:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 197:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] *= worker.step_3[i];                             // *
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 198:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 199:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];           // binary NOT operator
        worker.step_3[i] += worker.step_3[i];           // +
        worker.step_3[i] *= worker.step_3[i];           // *
        worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 200:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 201:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 202:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 203:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);                // rotate  bits by 1
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 204:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 205:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] += worker.step_3[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 206:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 207:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 8); // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 3);                           // rotate  bits by 3
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 208:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 209:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 210:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);                // rotate  bits by 5
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 211:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] += worker.step_3[i];                             // +
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 212:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 213:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 214:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 215:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] *= worker.step_3[i];                             // *
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 216:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 217:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
        worker.step_3[i] += worker.step_3[i];               // +
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);  // rotate  bits by 1
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 218:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]); // reverse bits
        worker.step_3[i] = ~worker.step_3[i];          // binary NOT operator
        worker.step_3[i] *= worker.step_3[i];          // *
        worker.step_3[i] -= (worker.step_3[i] ^ 97);   // XOR and -
                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 219:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 220:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 221:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5); // rotate  bits by 5
        worker.step_3[i] ^= worker.step_3[worker.pos2];    // XOR
        worker.step_3[i] = ~worker.step_3[i];              // binary NOT operator
        worker.step_3[i] = reverse8(worker.step_3[i]);     // reverse bits
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 222:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] *= worker.step_3[i];                          // *
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 223:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 224:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], 4);  // rotate  bits by 1
        // worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       //
      }
      break;
    case 225:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                          // binary NOT operator
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 226:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);  // reverse bits
        worker.step_3[i] -= (worker.step_3[i] ^ 97);    // XOR and -
        worker.step_3[i] *= worker.step_3[i];           // *
        worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 227:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 228:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];          // ones count bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 229:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                // rotate  bits by 3
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);               // rotate  bits by 2
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 230:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 231:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] ^= worker.step_3[worker.pos2];                // XOR
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 232:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] *= worker.step_3[i];               // *
        worker.step_3[i] *= worker.step_3[i];               // *
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4); // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 233:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);    // rotate  bits by 3
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 234:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3);    // shift right
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 235:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] *= worker.step_3[i];               // *
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 236:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= worker.step_3[worker.pos2];                   // XOR
        worker.step_3[i] += worker.step_3[i];                             // +
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                      // XOR and -
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 237:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);             // rotate  bits by 3
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 238:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];              // +
        worker.step_3[i] += worker.step_3[i];              // +
        worker.step_3[i] = std::rotl(worker.step_3[i], 3); // rotate  bits by 3
        worker.step_3[i] -= (worker.step_3[i] ^ 97);       // XOR and -
                                                           // INSERT_RANDOM_CODE_END
      }
      break;
    case 239:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 6); // rotate  bits by 5
        // worker.step_3[i] = std::rotl(worker.step_3[i], 1); // rotate  bits by 1
        worker.step_3[i] *= worker.step_3[i];                             // *
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 240:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                             // binary NOT operator
        worker.step_3[i] += worker.step_3[i];                             // +
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3);    // shift left
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 241:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);   // rotate  bits by 4
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] ^= worker.step_3[worker.pos2];       // XOR
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 242:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];           // +
        worker.step_3[i] += worker.step_3[i];           // +
        worker.step_3[i] -= (worker.step_3[i] ^ 97);    // XOR and -
        worker.step_3[i] ^= worker.step_3[worker.pos2]; // XOR
                                                        // INSERT_RANDOM_CODE_END
      }
      break;
    case 243:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);   // rotate  bits by 2
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);    // rotate  bits by 1
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 244:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = reverse8(worker.step_3[i]);      // reverse bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 245:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] -= (worker.step_3[i] ^ 97);                   // XOR and -
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);             // rotate  bits by 5
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 246:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                          // +
        worker.step_3[i] = std::rotl(worker.step_3[i], 1);             // rotate  bits by 1
        worker.step_3[i] = worker.step_3[i] >> (worker.step_3[i] & 3); // shift right
        worker.step_3[i] += worker.step_3[i];                          // +
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 247:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);  // rotate  bits by 5
        worker.step_3[i] = ~worker.step_3[i];               // binary NOT operator
                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    case 248:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = ~worker.step_3[i];                 // binary NOT operator
        worker.step_3[i] -= (worker.step_3[i] ^ 97);          // XOR and -
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 5);    // rotate  bits by 5
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 249:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);                    // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 250:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = worker.step_3[i] & worker.step_3[worker.pos2]; // AND
        worker.step_3[i] = std::rotl(worker.step_3[i], worker.step_3[i]); // rotate  bits by random
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]];             // ones count bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);               // rotate  bits by 4
                                                                          // INSERT_RANDOM_CODE_END
      }
      break;
    case 251:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] += worker.step_3[i];                 // +
        worker.step_3[i] ^= (byte)bitTable[worker.step_3[i]]; // ones count bits
        worker.step_3[i] = reverse8(worker.step_3[i]);        // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);   // rotate  bits by 2
                                                              // INSERT_RANDOM_CODE_END
      }
      break;
    case 252:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = reverse8(worker.step_3[i]);                 // reverse bits
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 4);            // rotate  bits by 4
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);            // rotate  bits by 2
        worker.step_3[i] = worker.step_3[i] << (worker.step_3[i] & 3); // shift left
                                                                       // INSERT_RANDOM_CODE_END
      }
      break;
    case 253:
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2); // rotate  bits by 2
        worker.step_3[i] ^= worker.step_3[worker.pos2];     // XOR
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);  // rotate  bits by 3
        // INSERT_RANDOM_CODE_END

        worker.prev_lhash = worker.lhash + worker.prev_lhash;
        worker.lhash = XXHash64::hash(worker.step_3, worker.pos2,0);
      }
      break;
    case 254:
    case 255:
      RC4_set_key(&worker.key, 256,  worker.step_3);
// worker.step_3 = highwayhash.Sum(worker.step_3[:], worker.step_3[:])
#pragma GCC unroll 32
      for (int i = worker.pos1; i < worker.pos2; i++)
      {
        // INSERT_RANDOM_CODE_START
        worker.step_3[i] ^= static_cast<uint8_t>(std::bitset<8>(worker.step_3[i]).count()); // ones count bits
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                                  // rotate  bits by 3
        worker.step_3[i] ^= std::rotl(worker.step_3[i], 2);                                 // rotate  bits by 2
        worker.step_3[i] = std::rotl(worker.step_3[i], 3);                                  // rotate  bits by 3
                                                                                            // INSERT_RANDOM_CODE_END
      }
      break;
    default:
      break;
    }
}
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
  testRes.duration_ns = time;
  memcpy(testRes.result, worker.step_3, 256);
  if (print) {
    printf("SC result     : %03d -> %03d", worker.pos1, worker.pos2);
    for (int i = worker.pos1; i < worker.pos1 + 32; i++) {
      printf("%02x ", worker.step_3[i]);
    }
    printf("\n took %ld ns\n---------------\n", time.count());
  }
}


void optest_branchcpu(int op, workerData &worker, byte testData[32], OpTestResult &testRes, bool print) {
  // Set us up the bomb
  memset(worker.step_3, 0, 256);
  memcpy(worker.step_3, testData, 32);

  // Because branchComputeCPU references .chunk (which is a pointer)
  worker.chunk = &worker.step_3[0];
  worker.prev_chunk = worker.chunk;
  if (print){
    printf("Lookup\n");
    printf("LT Input %3d  : ", op);
    for (int i = worker.pos1; i < worker.pos1 + 32; i++) {
      printf("%02X ", worker.chunk[i]);
    }
    printf("\n");
  }

  auto start = std::chrono::steady_clock::now();
  for(int x = 0; x < 256; x++) {
    worker.op = op;
    //worker.pos1 = 0; worker.pos2 = 32;
    worker.chunk = worker.step_3;
    worker.prev_chunk = worker.chunk;
    branchComputeCPU(worker, true);
  }

  auto test_end = std::chrono::steady_clock::now();
  auto test_time = std::chrono::duration_cast<std::chrono::nanoseconds>(test_end-start);
  testRes.duration_ns = test_time;
  memcpy(testRes.result, worker.chunk, 256);
  //memcpy(testRes.result, worker.salsaInput, 256);
  if (print){
    printf("LT result     : ");
    for (int i = worker.pos1; i < worker.pos1 + 32; i++) {
      printf("%02x ", worker.chunk[i]);
    }
    printf("\n took %ld ns\n---------------\n", test_time.count());
  }
  return; 
}

void optest_lookup(int op, workerData &worker, byte testData[32], OpTestResult &testRes, bool print) {
  // Set us up the bomb
  memset(worker.step_3, 0, 256);
  memcpy(worker.step_3, testData, 32);

  // Because lookupCompute references .chunk (which is a pointer)
  worker.chunk = &worker.step_3[0];
  worker.prev_chunk = worker.chunk;
  if (print){
    printf("Lookup\n");
    printf("LT Input %3d  : ", op);
    for (int i = worker.pos1; i < worker.pos1 + 32; i++) {
      printf("%02X ", worker.chunk[i]);
    }
    printf("\n");
  }

  auto start = std::chrono::steady_clock::now();
  for(int x = 0; x < 256; x++) {
    worker.op = op;
    //worker.pos1 = 0; worker.pos2 = 32;
    worker.chunk = worker.step_3;
    worker.prev_chunk = worker.chunk;
    lookupCompute(worker, true);
  }

  auto test_end = std::chrono::steady_clock::now();
  auto test_time = std::chrono::duration_cast<std::chrono::nanoseconds>(test_end-start);
  testRes.duration_ns = test_time;
  memcpy(testRes.result, worker.chunk, 256);
  //memcpy(testRes.result, worker.salsaInput, 256);
  if (print){
    printf("LT result     : ");
    for (int i = worker.pos1; i < worker.pos1 + 32; i++) {
      printf("%02x ", worker.chunk[i]);
    }
    printf("\n took %ld ns\n---------------\n", test_time.count());
  }
  return; 
}

#if defined(__AVX2__)
void optest_avx2(int op, workerData &worker, byte testData[32], OpTestResult &testRes, bool print) {
  // Set us up the bomb
  memset(worker.step_3, 0, 256);
  memcpy(worker.step_3, testData, 32);

  // Because branchComputeCPU_avx2 references .chunk (which is a pointer)
  worker.chunk = &worker.step_3[0];
  worker.prev_chunk = worker.chunk;
  if (print){
    printf("SIMD\n");
    printf("SIMD Input %3d: ", op);
    for (int i = worker.pos1; i < worker.pos1 + 32; i++) {
      printf("%02X ", worker.chunk[i]);
    }
    printf("\n");
  }

  auto start = std::chrono::steady_clock::now();
  for(int x = 0; x < 256; x++) {
    worker.op = op;
    //worker.pos1 = 0; worker.pos2 = 32;
    worker.chunk = worker.step_3;
    worker.prev_chunk = worker.chunk;
    branchComputeCPU_avx2_zOptimized(worker, true);
  }

  auto test_end = std::chrono::steady_clock::now();
  auto test_time = std::chrono::duration_cast<std::chrono::nanoseconds>(test_end-start);
  testRes.duration_ns = test_time;
  memcpy(testRes.result, worker.chunk, 256);
  //memcpy(testRes.result, worker.salsaInput, 256);
  if (print){
    printf("SIMD result   : ");
    for (int i = worker.pos1; i < worker.pos1 + 32; i++) {
      printf("%02x ", worker.chunk[i]);
    }
    printf("\n took %ld ns\n---------------\n", test_time.count());
  }
  return; 
}
#endif

#if defined(__aarch64__)
void optest_aarch64(int op, workerData &worker, byte testData[32], OpTestResult &testRes, bool print) {
  // Set us up the bomb
  memset(worker.step_3, 0, 256);
  memcpy(worker.step_3, testData, 32);

  // Because branchComputeCPU_avx2 references .chunk (which is a pointer)
  worker.chunk = &worker.step_3[0];
  worker.prev_chunk = worker.chunk;
  if (print){
    printf("AA64\n");
    printf("AA64 Input %3d: ", op);
    for (int i = worker.pos1; i < worker.pos1 + 32; i++) {
      printf("%02X ", worker.chunk[i]);
    }
    printf("\n");
  }

  auto start = std::chrono::steady_clock::now();
  for(int x = 0; x < 256; x++) {
    worker.op = op;
    //worker.pos1 = 0; worker.pos2 = 32;
    worker.chunk = worker.step_3;
    worker.prev_chunk = worker.chunk;
    branchComputeCPU_aarch64(worker, true);
  }

  auto test_end = std::chrono::steady_clock::now();
  auto test_time = std::chrono::duration_cast<std::chrono::nanoseconds>(test_end-start);
  testRes.duration_ns = test_time;
  memcpy(testRes.result, worker.chunk, 256);
  //memcpy(testRes.result, worker.salsaInput, 256);
  if (print){
    printf("AA64 result   : %03d -> %03d", worker.pos1, worker.pos2);
    for (int i = worker.pos1; i < worker.pos1 + 32; i++) {
      printf("%02x ", worker.chunk[i]);
    }
    printf("\n took %ld ns\n---------------\n", test_time.count());
  }
  return; 
}
#endif
