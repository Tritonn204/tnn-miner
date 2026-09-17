// Links the actual TNN Pearl session object, but no GPU runtime or miner entry.
#include <net/sessions.hpp>
#include <tnn_hip/common/gpu_submit_queue.hpp>
#include <atomic>
#include <cassert>
#include <iostream>
#include <tnn_hip/coins/pearl/pearl_mining.hpp>
#include <tnn_hip/coins/pearl/pearl_logging.hpp>

bool ABORT_MINER = false;
bool isConnected = false, devConnected = false;
bool submitting = false, submittingDev = false, data_ready = false;
int accepted = 0, rejected = 0, jobCounter = 0;
int64_t ourHeight = 0, devHeight = 0;
double doubleDiff = 0, doubleDiffDev = 0;
std::atomic<int> deviceAccepted[33]{}, deviceRejected[33]{};
std::mutex mutex, wsMutex;
std::condition_variable cv;
std::string stratumPassword = "x";
boost::json::value job, devJob;
boost::json::object share, devShare;

int main(int argc, char** argv) {
    assert(argc == 2);
    if (std::string(argv[1]) == "--logging") {
        using namespace tnn::pearl;
        std::printf("STATUS >> ");
        log_share(false, 3, true, "");
        log_share(false, 3, false, "bad target");
        log_share(true, 3, true, "");
        log_share(true, 3, false, "stale");
        tnn_set_log_level(TnnLogLevel::Off);
        std::printf("QUIET");
        log_share(false, 3, true, "");
        std::printf("END");
        log_stratum_error("test error");
        return 0;
    }
    tnn::pearl::configure_mining();
    net::io_context context;
    ssl::context tls(ssl::context::tlsv12_client);
    GPUSubmitQueue::instance().start(&share, &devShare, &submitting, &submittingDev,
        &data_ready, &cv, [](int64_t, bool) { return true; }, &mutex);
    net::spawn(context, [&](net::yield_context yield) {
        // The first peer closes after authorization. A normal session must
        // return without aborting the application, allowing its owner to retry.
        tnn::pearl::pearl_stratum_session("127.0.0.1", argv[1], "transport-wallet", "worker",
                                          context, tls, yield, false, false);
        assert(!ABORT_MINER);
        tnn::pearl::pearl_stratum_session("127.0.0.1", argv[1], "transport-wallet", "worker",
                                          context, tls, yield, false, false);
    }, net::detached);
    net::steady_timer poll(context);
    unsigned sent = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    std::function<void()> tick;
    tick = [&] {
        if (ABORT_MINER) return;
        {
            std::scoped_lock lock(mutex);
            if (accepted == 1) ABORT_MINER = true;
            if (isConnected && (sent == 0 || (sent == 1 && rejected == 1))) {
                assert(doubleDiff > 0 && doubleDiffDev == 0);
                auto binding = job.as_object();
                boost::json::object packet{{"_pearl_job", binding}, {"_pearl_device", 3},
                    {"params", boost::json::object{{"job_id", binding.at("job_id")}, {"plain_proof", "AAAA"}}}};
                GPUSubmitQueue::instance().push({std::move(packet), false, 1});
                ++sent;
            }
        }
        if (std::chrono::steady_clock::now() > deadline) ABORT_MINER = true;
        poll.expires_after(std::chrono::milliseconds(10));
        poll.async_wait([&](boost::system::error_code error) { if (!error) tick(); });
    };
    tick();
    context.run();
    GPUSubmitQueue::instance().stop();
    assert(sent == 2 && accepted == 1 && rejected == 1 && deviceAccepted[3] == 1);
    std::cout << "PEARL_TRANSPORT_PASS actual_session=true proof_validation=false gpu_launches=0\n";
}
