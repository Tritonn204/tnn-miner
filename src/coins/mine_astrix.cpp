#include "miners.hpp"
#include <astrix-hash/astrix-hash.h>
#include <hugepages.h>
#include <stratum/stratum.h>

void mineAstrix(int tid)
{
  for(;;) {
    boost::this_thread::yield();
  }
//   int64_t localJobCounter;
//   int64_t localOurHeight = 0;
//   int64_t localDevHeight = 0;

//   uint64_t i = 0;
//   uint64_t i_dev = 0;

//   byte powHash[32];
//   byte work[AstrixHash::INPUT_SIZE] = {0};
//   byte devWork[AstrixHash::INPUT_SIZE] = {0};

//   AstrixHash::worker *worker = (AstrixHash::worker *)malloc_huge_pages(sizeof(AstrixHash::worker));
//   AstrixHash::worker *devWorker = (AstrixHash::worker *)malloc_huge_pages(sizeof(AstrixHash::worker));

// waitForJob:

//   while (!isConnected)
//   {
//     boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
//   }

//   while (true)
//   {
//     try
//     {
//       bool assigned = false;
//       boost::json::value myJob;
//       boost::json::value myJobDev;
//       {
//         std::scoped_lock<boost::mutex> lockGuard(mutex);
//         myJob = job;
//         myJobDev = devJob;
//         localJobCounter = jobCounter;
//       }
      
//       if (!myJob.at("template").is_string()) {
//         continue;
//       }
//       if (ourHeight == 0 && devHeight == 0)
//         continue;

//       if (ourHeight == 0 || localOurHeight != ourHeight)
//       {
//         byte *b2 = new byte[AstrixHash::INPUT_SIZE];
//         switch (protocol)
//         {
//         case SPECTRE_SOLO:
//           hexstrToBytes(std::string(myJob.at("template").as_string()), b2);
//           break;
//         case SPECTRE_STRATUM:
//           hexstrToBytes(std::string(myJob.at("template").as_string()), b2);
//           break;
//         }
//         memcpy(work, b2, AstrixHash::INPUT_SIZE);
//         AstrixHash::newMatrix(work, worker->matBuffer, *worker);
//         // AstrixHash::genPrePowHash(b2, *worker);/
//         // AstrixHash::newMatrix(b2, worker->mat);
//         delete[] b2;
//         localOurHeight = ourHeight;
//         i = 0;
//       }

//       if (devConnected && myJobDev.at("template").is_string())
//       {
//         if (devHeight == 0 || localDevHeight != devHeight)
//         {
//           byte *b2d = new byte[AstrixHash::INPUT_SIZE];
//           switch (protocol)
//           {
//           case SPECTRE_SOLO:
//             hexstrToBytes(std::string(myJobDev.at("template").as_string()), b2d);
//             break;
//           case SPECTRE_STRATUM:
//             hexstrToBytes(std::string(myJobDev.at("template").as_string()), b2d);
//             break;
//           }
//           memcpy(devWork, b2d, AstrixHash::INPUT_SIZE);
//           AstrixHash::newMatrix(devWork, devWorker->matBuffer, *devWorker);
//           // AstrixHash::genPrePowHash(b2d, *devWorker);
//           // AstrixHash::newMatrix(b2d, devWorker->mat);
//           delete[] b2d;
//           localDevHeight = devHeight;
//           i_dev = 0;
//         }
//       }

//       bool devMine = false;
//       double which;
//       bool submit = false;
//       double DIFF = 1;
//       Num cmpDiff;

//       // printf("end of job application\n");
//       while (localJobCounter == jobCounter)
//       {
//         which = (double)(rand() % 10000);
//         devMine = (devConnected && devHeight > 0 && which < devFee * 100.0);
//         DIFF = devMine ? doubleDiffDev : doubleDiff;
//         if (DIFF == 0)
//           continue;

//         // cmpDiff = ConvertDifficultyToBig(DIFF, SPECTRE_X);
//         cmpDiff = AstrixHash::diffToTarget(DIFF);

//         uint64_t *nonce = devMine ? &i_dev : &i;
//         (*nonce)++;

//         // printf("nonce = %llu\n", *nonce);

//         byte *WORK = (devMine && devConnected) ? &devWork[0] : &work[0];
//         byte *nonceBytes = &WORK[72];
//         uint64_t n;
        
//         int enLen = 0;
        
//         boost::json::value &J = devMine ? myJobDev : myJob;
//         if (!J.as_object().if_contains("extraNonce") || J.at("extraNonce").as_string().size() == 0)
//           n = ((tid - 1) % (256 * 256)) | ((rand() % 256) << 16) | ((*nonce) << 24);
//         else {
//           uint64_t eN = std::stoull(std::string(J.at("extraNonce").as_string().c_str()), NULL, 16);
//           enLen = std::string(J.at("extraNonce").as_string()).size()/2;
//           int offset = (64 - enLen*8);
//           n = ((tid - 1) % (256 * 256)) | (((*nonce) << 16) & ((1ULL << offset)-1)) | (eN << offset);
//         }
//         memcpy(nonceBytes, (byte *)&n, 8);

//         // printf("after nonce: %s\n", hexStr(WORK, AstrixHash::INPUT_SIZE).c_str());

//         if (localJobCounter != jobCounter) {
//           // printf("thread %d updating job before hash\n", tid);
//           break;
//         }

//         AstrixHash::worker &usedWorker = devMine ? *devWorker : *worker;
//         AstrixHash::hash(usedWorker, WORK, AstrixHash::INPUT_SIZE, powHash);

//         // if (littleEndian())
//         // {
//         //   std::reverse(powHash, powHash + 32);
//         // }

//         counter.fetch_add(1);
//         submit = (devMine && devConnected) ? !submittingDev : !submitting;

//         if (localJobCounter != jobCounter || localOurHeight != ourHeight) {
//           // printf("thread %d updating job after hash\n", tid);
//           break;
//         }


//         if (Num(hexStr(powHash, 32).c_str(), 16) <= cmpDiff)
//         {
//           // printf("thread %d entered submission process\n", tid);
//           if (!submit) {
//             for(;;) {
//               submit = (devMine && devConnected) ? !submittingDev : !submitting;
//               if (submit || localJobCounter != jobCounter || localOurHeight != ourHeight)
//                 break;
//               boost::this_thread::yield();
//             }
//           }
//           if (localJobCounter != jobCounter) {
//             // printf("thread %d updating job after check\n", tid);
//             break;
//           }
//           // if (littleEndian())
//           // {
//           //   std::reverse(powHash, powHash + 32);
//           // }
//         //   std::string b64 = base64::to_base64(std::string((char *)&WORK[0], XELIS_TEMPLATE_SIZE));
//           // boost::lock_guard<boost::mutex> lock(mutex);
//           if (devMine)
//           {
//             submittingDev = true;
//             // std::scoped_lock<boost::mutex> lockGuard(devMutex);
//             // if (localJobCounter != jobCounter || localDevHeight != devHeight)
//             // {
//             //   break;
//             // }
//             setcolor(CYAN);
//             std::cout << "\n(DEV) Thread " << tid << " found a dev share\n" << std::flush;
//             setcolor(BRIGHT_WHITE);
//             switch (protocol)
//             {
//             case SPECTRE_SOLO:
//               devShare = {{"block_template", hexStr(&WORK[0], AstrixHash::INPUT_SIZE).c_str()}};
//               break;
//             case SPECTRE_STRATUM:
//               std::vector<char> nonceStr;
//               // Num(std::to_string((n << enLen*8) >> enLen*8).c_str(),10).print(nonceStr, 16);
//               Num(std::to_string(n).c_str(),10).print(nonceStr, 16);
//               devShare = {{{"id", SpectreStratum::submitID},
//                         {"method", SpectreStratum::submit.method.c_str()},
//                         {"params", {devWorkerName,                                   // WORKER
//                                     myJobDev.at("jobId").as_string().c_str(), // JOB ID
//                                     std::string(nonceStr.data()).c_str()}}}};

//               break;
//             }
//             data_ready = true;
//           }
//           else
//           {
//             submitting = true;
//             // std::scoped_lock<boost::mutex> lockGuard(userMutex);
//             // if (localJobCounter != jobCounter || localOurHeight != ourHeight)
//             // {
//             //   break;
//             // }
//             setcolor(BRIGHT_YELLOW);
//             std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
//             setcolor(BRIGHT_WHITE);
//             switch (protocol)
//             {
//             case SPECTRE_SOLO:
//               share = {{"block_template", hexStr(&WORK[0], AstrixHash::INPUT_SIZE).c_str()}};
//               break;
//             case SPECTRE_STRATUM:
//               std::vector<char> nonceStr;
//               // Num(std::to_string((n << enLen*8) >> enLen*8).c_str(),10).print(nonceStr, 16);
//               Num(std::to_string(n).c_str(),10).print(nonceStr, 16);
//               share = {{{"id", SpectreStratum::submitID},
//                         {"method", SpectreStratum::submit.method.c_str()},
//                         {"params", {workerName,                                   // WORKER
//                                     myJob.at("jobId").as_string().c_str(), // JOB ID
//                                     std::string(nonceStr.data()).c_str()}}}};

//               // std::cout << "blob: " << hexStr(&WORK[0], AstrixHash::INPUT_SIZE).c_str() << std::endl;
//               // std::cout << "nonce: " << nonceStr.data() << std::endl;
//               // std::cout << "extraNonce: " << hexStr(&WORK[AstrixHash::INPUT_SIZE - 48], enLen).c_str() << std::endl;
//               // std::cout << "hash: " << hexStr(&powHash[0], 32) << std::endl;
//               // std::vector<char> diffHex;
//               // cmpDiff.print(diffHex, 16);
//               // std::cout << "difficulty (LE): " << std::string(diffHex.data()).c_str() << std::endl;
//               // std::cout << "powValue: " << Num(hexStr(powHash, 32).c_str(), 16) << std::endl;
//               // std::cout << "target (decimal): " << cmpDiff << std::endl;

//               // printf("blob: %s\n", foundBlob.c_str());
//               // printf("hash (BE): %s\n", hexStr(&powHash[0], 32).c_str());
//               // printf("nonce (Full bytes for injection): %s\n", hexStr((byte *)&n, 8).c_str());

//               break;
//             }
//             data_ready = true;
//           }
//           // printf("thread %d finished submission process\n", tid);
//           cv.notify_all();
//         }

//         if (!isConnected) {
//           data_ready = true;
//           cv.notify_all();
//           break;
//         }
//       }
//       if (!isConnected) {
//         data_ready = true;
//         cv.notify_all();
//         break;
//       }
//     }
//     catch (std::exception& e)
//     {
//       setcolor(RED);
//       std::cerr << "Error in POW Function" << std::endl;
//       std::cerr << e.what() << std::endl;
//       setcolor(BRIGHT_WHITE);

//       localJobCounter = -1;
//       localOurHeight = -1;
//       localDevHeight = -1;
//     }
//     if (!isConnected) {
//       data_ready = true;
//       cv.notify_all();
//       break;
//     }
//   }
//   goto waitForJob;
}