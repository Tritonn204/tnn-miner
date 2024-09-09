#include "net.hpp"
#include "hex.h"

// #include <grpcpp/grpcpp.h>
// #include <net/proto/astrix/messages.grpc.pb.h>   // This will be the generated file for your gRPC service
// #include <net/proto/astrix/p2p.pb.h>   // Generated from protobuf (e.g., messages.proto)
// #include <net/proto/astrix/rpc.pb.h>   // Generated from protobuf (e.g., messages.proto)

// #include <num.h>

// #include "rx0_jobCache.hpp"
// #include <astrix-hash/astrix-hash.h>

void astrix_session(
    std::string host,
    std::string const &port,
    std::string const &wallet,
    bool isDev)
{
  // httplib::Client daemon(host, stoul(port));

  // std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(host + ":" + port, grpc::InsecureChannelCredentials());
  // auto client = Astrixd::NewStub(channel);

  // // job thread here
  // uint64_t chainHeight = 0;

  // GetBlockTemplateRequestMessage gbtReq;
  // gbtReq.set_pay_address(wallet.c_str());
  // gbtReq.set_extra_data("version");

  // bool submitThread = false;
  // bool abort = false;


  // auto astrix_getTemplate = [&]() -> int {
  //   GetBlockTemplateResponse response;
  //   grpc::ClientContext context;
  //   grpc::Status status = client->GetBlockTemplate(&context, gbtReq, &response);

  //   if (response.has_block())
  //   {
  //     printf("received: %s\n", response.block().DebugString().c_str());
      // if ((isDev ? devJob : job).as_object()["template"].is_null() ||
      //   std::string(newJob.at("blocktemplate_blob").as_string().c_str()).compare(
      //   (isDev ? devJob : job).at("template").as_string().c_str()) != 0)
      // {
      //   chainHeight = newJob.at("height").to_number<uint64_t>();
      //   boost::json::value &J = isDev ? devJob : job;

      //   Num newTarget = maxTarget / Num(newJob.at("difficulty").to_number<uint64_t>());
      //   std::vector<char> tmp;
      //   newTarget.print(tmp, 16);

      //   // std::cout << "new target: " << &tmp[0] << std::endl
      //   //           << std::flush;

      //   std::string tString = (const char *)tmp.data();
      //   if (!isDev) difficulty = newJob.at("difficulty").to_number<uint64_t>();

      //   J = {
      //       {"blob", newJob.at("blockhashing_blob").as_string().c_str()},
      //       {"template", newJob.at("blocktemplate_blob").as_string().c_str()},
      //       {"target", tString.c_str()},
      //       {"seed_hash", newJob.at("seed_hash").as_string().c_str()}
      //   };

      //   // std::cout << "Received template: " << response << std::endl;
      // }

      // // std::cout << "difficulty: " << newJob.at("difficulty").to_number<uint64_t>() << std::endl;

      // bool *C = isDev ? &devConnected : &isConnected;
      // if (!*C)
      // {
      //   if (!isDev)
      //   {
      //     difficulty = newJob.at("difficulty").to_number<uint64_t>();
      //     setcolor(BRIGHT_YELLOW);
      //     printf("Mining at: %s to wallet %s\n", host.c_str(), wallet.c_str());
      //     fflush(stdout);
      //     setcolor(CYAN);
      //     printf("Dev fee: %.2f%% of your total hashrate\n", devFee);
  
      //     fflush(stdout);
      //     setcolor(BRIGHT_WHITE);
      //   }
      //   else
      //   {
      //     setcolor(CYAN);
      //     printf("Connected to dev node: %s\n", host.c_str());
      //     fflush(stdout);
      //     setcolor(BRIGHT_WHITE);
      //   }
      // }

      // updateVM(newJob, isDev);

      // *C = true;
  //     return 0;
  //   }
  //   else
  //   {
  //     fail("getBlockTemplate", (res ? std::to_string(res->status).c_str() : "No response"));
  //     return 1;
  //   }
  // };

  // boost::thread([&](){
  //   submitThread = true;
  //   while(!abort) {
  //     boost::unique_lock<boost::mutex> lock(mutex);
  //     bool *B = isDev ? &submittingDev : &submitting;
  //     cv.wait(lock, [&]{ return (data_ready && (*B)) || abort; });
  //     if (abort) break;
  //     try {
  //       boost::json::object *S = &share;
  //       if (isDev)
  //         S = &devShare;

  //       std::string msg = boost::json::serialize((*S)) + "\n";
  //       // std::cout << "sending in: " << msg << std::endl;
  //       auto res = daemon.Post("/json_rpc", msg, jsonType);
  //       if (res && res->status == 200)
  //       {
  //         boost::json::object result = boost::json::parse(res->body).as_object();
  //         if (!result["error"].is_null()) {
  //           setcolor(isDev ? CYAN : RED);
  //           printf("%s\n", result["error"].as_object()["message"].as_string().c_str());
  //           fflush(stdout);
  //           setcolor(BRIGHT_WHITE);

  //           rejected++;
  //         } else {
  //           // std::cout << boost::json::serialize(result) << std::endl << std::flush;
  //           setcolor(isDev ? CYAN : BRIGHT_YELLOW);
  //           printf("\n");
  //           if (isDev) printf("DEV | ");
  //           printf("Block accepted!\n");
  //           fflush(stdout);
  //           setcolor(BRIGHT_WHITE);
  //           accepted++;

  //           boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
  //           rx0_getTemplate();
  //         }
  //       } else {
  //         fail("submit_block", (res ? std::to_string(res->status).c_str() : "No response"));
  //       }
  //       (*B) = false;
  //       data_ready = false;
  //     } catch (const std::exception &e) {
  //       setcolor(RED);
  //       printf("\nSubmit thread error: %s\n", e.what());
  //       fflush(stdout);
  //       setcolor(BRIGHT_WHITE);
  //       break;
  //     }
  //     //boost::this_thread::sleep_for(boost::chrono::milliseconds(200));
  //     boost::this_thread::yield();
  //   }
  //   submitThread = false;
  // });


  // for (;;)
  // {
  //   bool *C = isDev ? &devConnected : &isConnected;
  //   bool *B = isDev ? &submittingDev : &submitting;
  //   try
  //   {
  //     if (astrix_getTemplate()) {
  //       setForDisconnected(C, B, &abort, &data_ready, &cv);

  //       for (;;)
  //       {
  //         if (!submitThread)
  //           break;
  //         boost::this_thread::yield();
  //       }       
  //       return;
  //     }
  //     boost::this_thread::sleep_for(boost::chrono::seconds(5));
  //   }
  //   catch (const std::exception &e)
  //   {
  //     bool *C = isDev ? &devConnected : &isConnected;
  //     printf("exception\n");
  //     fflush(stdout);
  //     setForDisconnected(C, B, &abort, &data_ready, &cv);

  //     for (;;)
  //     {
  //       if (!submitThread)
  //         break;
  //       boost::this_thread::yield();
  //     }
  //     setcolor(RED);
  //     std::cerr << e.what() << std::endl;
  //     fflush(stdout);
  //     setcolor(BRIGHT_WHITE);
  //     return;
  //   }
  //   boost::this_thread::yield();
  // }
}