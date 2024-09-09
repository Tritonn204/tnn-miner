#include "miners.hpp"

#include <net/rx0/rx0_jobCache.hpp>

#include <randomx/randomx.h>
#include <randomx/dataset.hpp>
#include <randomx/common.hpp>
#include <randomx/jit_compiler.hpp>

#include <stratum/stratum.h>
#include <thread>

randomx_dataset* rxDataset;
randomx_cache* rxCache;

randomx_dataset* rxDataset_dev;
randomx_cache* rxCache_dev;

randomx_flags rxFlags;

bool rx_hugePages;

std::string randomx_cacheKey = "0000000000000000000000000000000000000000000000000000000000000000";
std::string randomx_cacheKey_dev = "0000000000000000000000000000000000000000000000000000000000000000";

std::string randomx_login;
std::string randomx_login_dev;

void randomx_init_intern(int threadCount) {
  if (nullptr == randomx::selectArgonImpl(rxFlags)) {
    throw std::runtime_error("Unsupported Argon2 implementation");
  }
  if ((rxFlags & RANDOMX_FLAG_JIT) && !RANDOMX_HAVE_COMPILER) {
    throw std::runtime_error("JIT compilation is not supported on this platform. Try without --jit");
  }
  if (!(rxFlags & RANDOMX_FLAG_JIT) && RANDOMX_HAVE_COMPILER) {
    std::cout << "WARNING: You are using the interpreter mode. Use --jit for optimal performance." << std::endl;
  }

  rxCache = randomx_alloc_cache(rxFlags);
  rxCache_dev = randomx_alloc_cache(rxFlags);

  rxDataset = randomx_alloc_dataset(rxFlags);
  rxDataset_dev = randomx_alloc_dataset(rxFlags);

  if (rxCache == nullptr || rxCache_dev == nullptr) {
    throw std::runtime_error("RandomX Cache Alloc Failed");
  }
}

void randomx_update_data(randomx_cache* rc, randomx_dataset* rd, void *seed, size_t seedSize, int threadCount) {
  randomx_init_cache(rc, seed, seedSize);
  
  uint32_t datasetItemCount = randomx_dataset_item_count();
  std::vector<std::thread> threads;

  if (threadCount > 1) {
    auto perThread = datasetItemCount / threadCount;
    auto remainder = datasetItemCount % threadCount;
    uint32_t startItem = 0;

    for (int i = 0; i < threadCount; ++i)
    {
      auto count = perThread + (i == threadCount - 1 ? remainder : 0);
      threads.push_back(std::thread(&randomx_init_dataset, rd, rc, startItem, count));
      startItem += count;
    }
    for (unsigned i = 0; i < threads.size(); ++i) {
      threads[i].join();
    }
  } else {
    randomx_init_dataset(rd, rc, 0, datasetItemCount);
  }
  threads.clear();
}

void randomx_set_flags(bool autoFlags) {
  if (autoFlags) {
    rxFlags = randomx_get_flags();
  }

  rxFlags |= RANDOMX_FLAG_FULL_MEM;
  if (rx_hugePages) rxFlags |= RANDOMX_FLAG_LARGE_PAGES; // TODO: Make this a toggle from CLI
  fflush(stdout);

  setcolor(BRIGHT_YELLOW);
  if (rxFlags & RANDOMX_FLAG_ARGON2_AVX2) {
    std::cout << " Argon2 implementation: AVX2" << std::endl;
  }
  else if (rxFlags & RANDOMX_FLAG_ARGON2_SSSE3) {
    std::cout << " Argon2 implementation: SSSE3" << std::endl;
  }
  else {
    std::cout << " Argon2 implementation: reference" << std::endl;
  }

  if (rxFlags & RANDOMX_FLAG_FULL_MEM) {
    std::cout << " full memory mode (2080 MiB)" << std::endl;
  }
  else {
    std::cout << " light memory mode (256 MiB)" << std::endl;
  }

  if (rxFlags & RANDOMX_FLAG_JIT) {
    std::cout << " JIT compiled mode ";
    if (rxFlags & RANDOMX_FLAG_SECURE) {
      std::cout << "(secure)";
    }
    std::cout << std::endl;
  }
  else {
    std::cout << " interpreted mode" << std::endl;
  }

  if (rxFlags & RANDOMX_FLAG_HARD_AES) {
    std::cout << " hardware AES mode" << std::endl;
  }
  else {
    std::cout << " software AES mode" << std::endl;
  }

  if (rxFlags & RANDOMX_FLAG_LARGE_PAGES) {
    std::cout << " large pages mode" << std::endl;
  }
  else {
    std::cout << " small pages mode" << std::endl;
  }
  printf("\n");
  fflush(stdout);
  setcolor(BRIGHT_WHITE);
}

void rxRPCTest() {
  const char* expected = "2b21c0534efc99db149d3fdb6e2f2938958f85229157f7da1586da641c041004";
  const char* seedHash = "78a2c61c1cac2577a9a65287a27e933ab56929c1bf428a6e1768636bdfdc9989";

  const char* blob = "0103bdefedb6065feba13815bf0a1ce2de984dcc449feaac5dc953cbc66de0fab9c082d4321c4"
  "300000000000000000000000000000000000000000000000000000000c8e633f791e0084871c36dd329edf7ab5e42eb6"
  "cef27d9136b1732b9dc01e6a501";

  const char* daemonBlob = "0103bdefedb6065feba13815bf0a1ce2de984dcc449feaac5dc953cbc66de0fab9c082d"
  "4321c430608b900000000000000000000000000000000000000000000000000c8e633f791e0084871c36dd329edf7ab5"
  "e42eb6cef27d9136b1732b9dc01e6a501";
  
  const char* fullTemplate = "0103bdefedb6065feba13815bf0a1ce2de984dcc449feaac5dc953cbc66de0fab9c08"
  "2d4321c4300000000000000000000000000000000000000000000000000000000024601ff0a02908188929bd90302f56"
  "a855c3ddbbb45aa2de64ff2e61f3569f398daa904b541049e9c82095966a8045a455048a19baf9bfaf31802f5fd6ee52"
  "d8f869455ca565e0c34755f9abc2b2ad6e9168689e92808526bee5e045a45504867800101f14963fa44a05664d1b904a"
  "87fd9e8e984a74378cb40ba552a11f7fba2cfc633023c000000000000000000000000000000000000000000000000000"
  "000000000000000000000000000000000000000000000000000000000000000000000012c7be86ab07488ba43e8e03d8"
  "5a67625cfbf98c8544de4c877241b7aaafc7fe30000000000";

  const char *nonceHex = "0608b900";

  int th = std::thread::hardware_concurrency();

  byte seedBuffer[32];
  hexstrToBytes(seedHash, seedBuffer);

  printf("%s | seedHash\n%s | seedBuffer\n", seedHash, hexStr(seedBuffer, 32).c_str());

  randomx_set_flags(true);
  randomx_init_intern(th);
  
  randomx_vm *vm = randomx_create_vm(rxFlags, rxCache, rxDataset);
  randomx_update_data(rxCache, rxDataset, seedBuffer, 32, th);
  randomx_vm_set_cache(vm, rxCache);

  // hashing

  byte work[RANDOMX_TEMPLATE_SIZE];
  byte powHash[32];

  hexstrToBytes(blob, work);
  hexstrToBytes(nonceHex, &work[39]);

  printf("%s | work\n%s: | daemonWork\n", 
    hexStr(work, std::string(blob).size()/2).c_str(), daemonBlob
  );

  randomx_calculate_hash(vm, work, std::string(blob).size()/2, powHash);
  randomx_calculate_hash(vm, work, std::string(blob).size()/2, powHash);

  printf("%s | powHash\n%s | expected\n", hexStr(powHash, 32).c_str(), expected);
}

void mineRx0(int tid){
  // const char* minerSig = "tnn-miner";

  // byte random_buf[12 + strlen(minerSig)];
  // std::random_device rd;
  // std::mt19937 gen(rd());
  // std::uniform_int_distribution<uint8_t> dist(0, 255);
  // std::array<uint8_t, 12> buf;
  // std::generate(buf.begin(), buf.end(), [&dist, &gen]()
  //               { return dist(gen); });
  // std::memcpy(random_buf, buf.data(), buf.size());

  // memcpy(random_buf + 12, minerSig, strlen(minerSig));

  boost::this_thread::sleep_for(boost::chrono::milliseconds(125));

  bool updateCache = false;

  int64_t localJobCounter;
  byte powHash[32];
  byte devWork[RANDOMX_TEMPLATE_SIZE];
  byte work[RANDOMX_TEMPLATE_SIZE];

  randomx_vm *vm = randomx_create_vm(rxFlags, rxCache, rxDataset);
  randomx_vm *vmDev = randomx_create_vm(rxFlags, rxCache, rxDataset);

  if (vm == nullptr) {
    if ((rxFlags & RANDOMX_FLAG_HARD_AES)) {
      throw std::runtime_error("Cannot create VM with the selected options. Try using --softAes");
    }
    if (rxFlags & RANDOMX_FLAG_LARGE_PAGES) {
      throw std::runtime_error("Cannot create VM with the selected options. Try without --largePages");
    }
    throw std::runtime_error("Cannot create VM");
  }

  if (vmDev == nullptr) {
    if ((rxFlags & RANDOMX_FLAG_HARD_AES)) {
      throw std::runtime_error("DEV: Cannot create VM with the selected options. Try using --softAes");
    }
    if (rxFlags & RANDOMX_FLAG_LARGE_PAGES) {
      throw std::runtime_error("DEV: Cannot create VM with the selected options. Try without --largePages");
    }
    throw std::runtime_error("DEV: Cannot create VM");
  }

waitForJob:

  while (!isConnected)
  {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }


  while (true)
  {
    while(!randomx_ready && !randomx_ready_dev) {
      boost::this_thread::yield();
    }
    try
    {
      boost::json::value myJob;
      boost::json::value myJobDev;
      {
        std::scoped_lock<boost::mutex> lockGuard(mutex);
        myJob = job;
        myJobDev = devJob;
        localJobCounter = jobCounter;
      }

      randomx_vm_set_cache(vm, rxCache);
      byte *b2 = new byte[RANDOMX_TEMPLATE_SIZE];
      memset(b2, 0, RANDOMX_TEMPLATE_SIZE);
      hexstrToBytes(std::string(myJob.at("blob").as_string()), b2);
      memcpy(work, b2, std::string(myJob.at("blob").as_string().c_str()).size() / 2);
      delete[] b2;

      if (devConnected)
      {
        randomx_vm_set_cache(vmDev, rxCache_dev);
        byte *b2d = new byte[RANDOMX_TEMPLATE_SIZE];
        memset(b2d, 0, RANDOMX_TEMPLATE_SIZE);
        hexstrToBytes(std::string(myJobDev.at("blob").as_string()), b2d);
        memcpy(devWork, b2d, std::string(myJobDev.at("blob").as_string().c_str()).size() / 2);
        delete[] b2d;
      }

      double which;
      bool devMine = false;
      bool submit = false;
      Num cmpDiff;
      // DIFF = 5000;

      std::string hex;

      uint32_t userNonce = 0;
      uint32_t devNonce = 0;

      int userLen = std::string(myJob.at("blob").as_string().c_str()).size() / 2;
      int devLen = std::string(myJobDev.at("blob").as_string().c_str()).size() / 2;

      Num cmpTarget = rx0_calcTarget(devMine ? myJobDev : myJob);

      while (localJobCounter == jobCounter)
      {
        which = (double)(rand() % 10000);
        devMine = (devConnected && which < devFee * 100.0 && randomx_ready_dev) || (randomx_ready_dev && !randomx_ready);

        uint32_t &nonce = devMine ? devNonce : userNonce;

        // printf("Difficulty: %" PRIx64 "\n", DIFF);

        nonce ++;

        byte *WORK = devMine ? &devWork[0] : &work[0];

        uint32_t N = nonce << 11 | tid;
        memcpy(&WORK[39], &N, 4);
        // if (littleEndian())
        // {
        //   std::swap(WORK[39], WORK[42]);
        //   std::swap(WORK[40], WORK[41]);

        // }

        randomx_calculate_hash(devMine ? vmDev : vm, WORK, devMine ? devLen : userLen, powHash);
        counter.fetch_add(1);
        submit = devMine ? !submittingDev : !submitting;

        // std::vector<char> tmp;
        // cmpTarget.print(tmp, 16);

        std::reverse(powHash, powHash + 32);
        if (Num(hexStr(powHash, 32).c_str(), 16) < cmpTarget)
        {
          std::reverse(powHash, powHash + 32);
          // std::cout << hexStr(powHash, 32).c_str() << "\n" << &tmp[0] << "\n" << std::endl << std::flush;
          // std::cout << hexStr(powHash, 32).c_str() << " | hash\n" << std::flush;
          // std::cout << hexStr(WORK, devMine ? devLen : userLen).c_str() << " | blob\n" << std::flush;
          if (!submit) {
            for(;;) {
              submit = (devMine && devConnected) ? !submittingDev : !submitting;
              if (submit || localJobCounter != jobCounter)
                break;
              boost::this_thread::yield();
            }
          }
          if (localJobCounter != jobCounter)
                break;
          // printf("work: %s, hash: %s\n", hexStr(&WORK[0], MINIBLOCK_SIZE).c_str(), hexStr(powHash, 32).c_str());
          // boost::lock_guard<boost::mutex> lock(mutex);
          if (devMine)
          {
            submittingDev = true;
            setcolor(CYAN);
            std::cout << "\n(DEV) Thread " << tid << " found a dev share\n" << std::flush;
            setcolor(BRIGHT_WHITE);

            N = __builtin_bswap32(N);
            devShare = {
              {"method", rx0Stratum::submit.method.c_str()},
              {"id", rx0Stratum::submit.id},
              {"params", {
                {"id", randomx_login_dev.c_str()},
                {"job_id", myJobDev.at("job_id").as_string().c_str()},
                {"nonce", uint32ToHex(N).c_str()},
                {"result", hexStr(powHash, 32).c_str()}
              }}
            };
            data_ready = true;
          }
          else
          {
            submitting = true;
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
            setcolor(BRIGHT_WHITE);
                      
            switch(protocol) {
              case RX0_SOLO:
              {
                int fbSize = myJob.at("template").as_string().size() / 2;
                byte *fullBlob = new byte[fbSize];
                hexstrToBytes(std::string(myJob.at("template").as_string()), fullBlob);
                memcpy(&fullBlob[39], &N, 4);

                share = {
                  {"jsonrpc", "2.0"},
                  {"method", "submit_block"},
                  {"id", 7}, // hardcoded for now
                  {"params", {hexStr(fullBlob, fbSize).c_str()/*,hexStr(WORK, userLen).c_str()*/}}
                };
                delete[] fullBlob;
                break;
              }
              case RX0_STRATUM:
              {
                N = __builtin_bswap32(N);
                share = {
                  {"method", rx0Stratum::submit.method.c_str()},
                  {"id", rx0Stratum::submit.id},
                  {"params", {
                    {"id", randomx_login.c_str()},
                    {"job_id", myJob.at("job_id").as_string().c_str()},
                    {"nonce", uint32ToHex(N).c_str()},
                    {"result", hexStr(powHash, 32).c_str()}
                  }}
                };
                break;                
              }
            }
            data_ready = true;
          }
          cv.notify_all();
        }

        if (!isConnected)
          break;
      }
      if (!isConnected)
        break;
    }
    catch (std::exception& e)
    {
      setcolor(RED);
      std::cerr << "Error in POW Function" << std::endl;
      std::cerr << e.what() << std::endl << std::flush;
      setcolor(BRIGHT_WHITE);

      localJobCounter = -1;
    }
    if (!isConnected)
      break;
  }
  goto waitForJob;
}