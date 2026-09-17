#include "pearl_mining.hpp"
#include "pearl_share_audit.hpp"
#include <tnn_hip/common/gpu_miner.hpp>
#include <stratum/pearl-stratum.hpp>
#include <net/net.hpp>

#include <chrono>
#include <map>
#include <optional>
#include <deque>

void minePearl_hip(int tid) {
    (void)tid;
    using namespace tnn::pearl;
    using Clock = stratum::Clock;
    Clock::time_point accounting_begin{}, accounting_end{}, last_report{};
    uint64_t completed_macs = 0, rolling_macs = 0;
    std::deque<std::pair<Clock::time_point, uint64_t>> rolling;
    if (!mining_enabled ||
        miningProfile.protocol != PROTO_PEARL_STRATUM) {
        TNN_LOG_ERROR("\n[PEARL] Pearl requires an enabled mining mode and Pearl Stratum\n");
        ABORT_MINER = true;
        return;
    }

    std::unique_ptr<GPUMiner> miner;
    struct Channel {
        std::optional<stratum::Jobs> jobs;
        uint64_t generation = 0;
        std::map<int64_t, stratum::Job> bindings;
    };
    std::array<Channel, 2> channels;
    std::mutex bindings_mutex;
    try {
        int count = 0;
        if (oroGetDeviceCount(&count) != oroSuccess) throw std::runtime_error("Cannot enumerate devices");
        int selected = -1;
        for (int device = 0; device < count; ++device) {
            if (!shouldUseDevice(device)) continue;
            if (selected != -1) throw std::runtime_error("Pearl requires exactly one selected GPU");
            selected = device;
        }
        if (selected == -1) throw std::runtime_error("Pearl requires a selected GPU");
        miner = std::make_unique<GPUMiner>("pearl", selected);
        if (!miner->initialize()) throw std::runtime_error("Pearl GPU initialization failed");
        miner->set_dev_fee(devFee);
        GPUSubmitQueue::instance().start(&share, &devShare, &submitting, &submittingDev,
            &data_ready, &cv, [&](int64_t token, bool dev) {
                {
                    std::scoped_lock lock(mutex);
                    if (!(dev ? devConnected : isConnected)) return false;
                }
                std::scoped_lock lock(bindings_mutex);
                const auto& channel = channels[dev];
                const auto found = channel.bindings.find(token);
                return channel.jobs && found != channel.bindings.end() &&
                       channel.jobs->eligible(found->second, Clock::now());
            }, &mutex, [](const GPUSubmitEntry&, SubmitDisposition outcome) {
                if (outcome == SubmitDisposition::Stale) ++share_audit.stale;
                else if (outcome == SubmitDisposition::Shutdown) ++share_audit.cancelled;
                else if (outcome == SubmitDisposition::Timeout) ++share_audit.failed;
            }, true);
        pearl_start_proofs();
        miner->set_completed_observer([&](uint64_t work, double iteration_seconds) {
            const auto now = Clock::now();
            if (accounting_begin == Clock::time_point{}) {
                accounting_begin = now - std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(iteration_seconds));
                last_report = now;
            }
            accounting_end = now;
            completed_macs += work;
            rolling_macs += work;
            rolling.emplace_back(now, work);
            while (!rolling.empty() && rolling.front().first <= now - std::chrono::seconds(120)) {
                rolling_macs -= rolling.front().second;
                rolling.pop_front();
            }
            if (now - last_report >= std::chrono::seconds(5)) {
                const double elapsed = std::chrono::duration<double>(now - accounting_begin).count();
                if (elapsed >= 120)
                    TNN_LOG_DEBUG("[PEARL-RATE] cumulative_TMACs=%.3f rolling120_TMACs=%.3f completed_MACs=%llu\n",
                        completed_macs / (elapsed * 1e12), rolling_macs / (120e12),
                        static_cast<unsigned long long>(completed_macs));
                else
                    TNN_LOG_DEBUG("[PEARL-RATE] cumulative_TMACs=%.3f rolling120=pending completed_MACs=%llu\n",
                        completed_macs / (elapsed * 1e12), static_cast<unsigned long long>(completed_macs));
                last_report = now;
            }
        });
        miner->start({}, pearl_build_batch);
        int64_t token = 0;
        while (!ABORT_MINER && !worker_failed && miner->is_running()) {
            std::array<boost::json::value, 2> jobs;
            std::array<bool, 2> connected;
            {
                std::scoped_lock lock(mutex);
                jobs = {job, devJob};
                connected = {isConnected, devConnected};
            }
            for (unsigned dev = 0; dev < 2; ++dev) {
                if (!connected[dev] || !jobs[dev].is_object() ||
                    !jobs[dev].as_object().contains("connection_generation")) continue;
                auto fields = jobs[dev].as_object();
                uint64_t generation = fields.at("connection_generation").to_number<uint64_t>();
                auto parsed = stratum::parse_job({{"method", "mining.notify"}, {"params", fields}}, generation);
                std::scoped_lock lock(bindings_mutex);
                auto& channel = channels[dev];
                if (channel.generation != generation) {
                    channel.jobs.emplace(generation);
                    channel.generation = generation;
                    channel.bindings.clear();
                }
                if (!channel.jobs->update(parsed, Clock::now())) continue;
                for (auto it = channel.bindings.begin(); it != channel.bindings.end();) {
                    if (!channel.jobs->eligible(it->second, Clock::now())) it = channel.bindings.erase(it);
                    else ++it;
                }
                if (token == INT64_MAX) throw std::runtime_error("Pearl job token exhausted");
                ++token;
                channel.bindings.emplace(token, parsed);
                JobSnapshot snapshot(parsed.header.data(), parsed.header.size(), token, 1,
                                     ALGO_PEARL_POUW, parsed.id, dev != 0);
                snapshot.raw_target = parsed.target_le;
                snapshot.connection_generation = generation;
                snapshot.block_height = parsed.height;
                snapshot.pearl_cert_version = parsed.cert_version;
                miner->publish_job(std::move(snapshot));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    } catch (const std::exception& error) {
        fflush(stdout);
        TNN_LOG_ERROR("\n[PEARL-ERROR] %s; stopping\n", error.what());
    }
    if (miner) miner->stop();
    if (completed_macs) {
        const double elapsed = std::chrono::duration<double>(accounting_end - accounting_begin).count();
        TNN_LOG_DEBUG("[PEARL-WORKER-SUMMARY] completed_MACs=%llu active_s=%.6f active_TMACs=%.3f\n",
            static_cast<unsigned long long>(completed_macs), elapsed, completed_macs / (elapsed * 1e12));
    }
    pearl_stop_proofs();
    GPUSubmitQueue::instance().stop();
    TNN_LOG_DEBUG("[PEARL-SHARES-SNAPSHOT] found=%llu proofs=%llu sent=%llu accepted=%llu rejected=%llu stale=%llu cancelled=%llu ambiguous=%llu failed=%llu\n",
        (unsigned long long)share_audit.discovered.load(), (unsigned long long)share_audit.proofs_built.load(),
        (unsigned long long)share_audit.submitted.load(), (unsigned long long)share_audit.accepted.load(),
        (unsigned long long)share_audit.rejected.load(), (unsigned long long)share_audit.stale.load(),
        (unsigned long long)share_audit.cancelled.load(), (unsigned long long)share_audit.ambiguous.load(),
        (unsigned long long)share_audit.failed.load());
    ABORT_MINER = true;
}
