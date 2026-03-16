#include "reporter.hpp"
#include <numeric>
#include <iostream>

const std::string units[] = {" ", " K", " M", " G", " T", " P"};
#ifdef TNN_RANDOMX
extern std::atomic<bool> datasetInitInProgress;
#endif

#ifdef TNN_HIP
extern std::atomic<bool> g_mining_started;
#include <tnn_hip/core/devInfo.hip.h>
#include <tnn_hip/common/gpu_device_filter.hpp>
#endif

extern bool beQuiet;

int update_handler(const boost::system::error_code& error)
{
    CHECK_CLOSE_RET(0);
    if (error == boost::asio::error::operation_aborted) {
        return 1;
    }

    using clock = std::chrono::steady_clock;
    static clock::time_point next_tick = clock::now() + std::chrono::seconds(1);
    next_tick += std::chrono::seconds(1);
    update_timer.expires_at(next_tick);
    update_timer.async_wait(update_handler);

    if (!isConnected) return 1;

#ifdef TNN_RANDOMX
    if (datasetInitInProgress.load()) return 1;
#endif

    static bool first_tick = true;
    reportCounter++;

    auto now = std::chrono::steady_clock::now();
    auto daysUp    = std::chrono::duration_cast<std::chrono::hours>(now - g_start_time).count() / 24;
    auto hoursUp   = std::chrono::duration_cast<std::chrono::hours>(now - g_start_time).count() % 24;
    auto minutesUp = std::chrono::duration_cast<std::chrono::minutes>(now - g_start_time).count() % 60;
    auto secondsUp = std::chrono::duration_cast<std::chrono::seconds>(now - g_start_time).count() % 60;

    // =========================================================================
    // Accumulate GPU stats (always, even before mining_started)
    // =========================================================================
    bool any_gpu_hashing = false;
    std::vector<double> gpu_hashrates;
    std::vector<int> gpu_unit_indices;

    #ifdef TNN_HIP
    if (gpuMine) {
        gpu_hashrates.resize(HIP_deviceCount);
        gpu_unit_indices.resize(HIP_deviceCount);

        bool prev_beQuiet = beQuiet;
        beQuiet = true;

        double elapsed_gpu = first_tick ? std::chrono::duration<double>(now - g_start_time).count() : 0.0;

        for (int i = 0; i < HIP_deviceCount; i++) {
            if (!shouldUseDevice(i)) continue;
            uint64_t currentHashesG = HIP_counters[i].load();
            HIP_counters[i].store(0);

            if (currentHashesG > 0) {
                any_gpu_hashing = true;
                beQuiet = prev_beQuiet;
            }

            // Normalize first-tick GPU sample the same way as CPU
            if (first_tick && elapsed_gpu > 1.0)
                currentHashesG = (uint64_t)(currentHashesG / elapsed_gpu);

            double ratioG = 1.0;
            if (HIP_rates1min[i].size() <= 60) {
                HIP_rates1min[i].push_back((int64_t)(currentHashesG * ratioG));
            } else {
                HIP_rates1min[i].erase(HIP_rates1min[i].begin());
                HIP_rates1min[i].push_back((int64_t)(currentHashesG * ratioG));
            }

            double hashrateG = (double)std::accumulate(HIP_rates1min[i].begin(), HIP_rates1min[i].end(), 0LL) /
                               (double)HIP_rates1min[i].size();

            int unitIdxG = 0;
            while (hashrateG >= 1000 && unitIdxG < 5) {
                unitIdxG++;
                hashrateG /= 1000.0;
            }

            gpu_hashrates[i] = hashrateG;
            gpu_unit_indices[i] = unitIdxG;
        }
    }
    #endif

    // =========================================================================
    // Accumulate CPU stats (always, even before mining_started)
    // =========================================================================
    uint64_t currentHashes = counter.load();
    counter.store(0);

    if (currentHashes > 0) {
        any_gpu_hashing = true;  // Reuse flag for CPU too
    }

    // On the first tick the sample covers an unknown duration since workers
    // started, not exactly 1 second.  Normalize it to a per-second rate so it
    // doesn't inflate the rolling average.
    if (first_tick) {
        double elapsed = std::chrono::duration<double>(now - g_start_time).count();
        if (elapsed > 1.0)
            currentHashes = (uint64_t)(currentHashes / elapsed);
    }

    double ratio = 1.0;
    size_t rate_window = gpuMine ? 60 : 30;
    if (rate30sec.size() <= rate_window) {
        rate30sec.push_back((int64_t)(currentHashes * ratio));
    } else {
        rate30sec.erase(rate30sec.begin());
        rate30sec.push_back((int64_t)(currentHashes * ratio));
    }

    double hashrate =
        (double)std::accumulate(rate30sec.begin(), rate30sec.end(), 0LL) /
        (double)rate30sec.size();

    int unitIdx = 0;
    while (hashrate >= 1000 && unitIdx < 5) {
        unitIdx++;
        hashrate /= 1000.0;
    }
    latest_hashrate = hashrate;
    first_tick = false;

    // =========================================================================
    // Check if mining has started (first non-zero hashrate)
    // =========================================================================

    #ifdef TNN_HIP
    if (!g_mining_started.load()) {
        if (any_gpu_hashing || hashrate > 0.001) {
            g_mining_started.store(true);
            
            // Print a separator to visually mark end of tuning output
            setcolor(BRIGHT_YELLOW);
            printf("\n");
            printf("============================================================\n");
            printf("[MINER] Mining started, hashrate reporting enabled\n");
            printf("============================================================\n");
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
        } else {
            // Mining hasn't started yet - don't print status, just accumulate stats
            return 0;
        }
    }
    #endif

    // =========================================================================
    // Print status (only after mining has started)
    // =========================================================================
    if (reportCounter >= reportInterval) {
        
        // GPU hashrates
#ifdef TNN_HIP
        if (gpuMine) {
            setcolor(BRIGHT_YELLOW);

            for (int i = 0; i < HIP_deviceCount; i++) {
                if (!shouldUseDevice(i)) continue;
                int a = deviceAccepted[i].load(std::memory_order_relaxed);
                int r = deviceRejected[i].load(std::memory_order_relaxed);

                if (g_powerMonAvail) {
                    double watts = getDevicePowerWatts(i);
                    if (watts > 0.0) {
                        // Reconstruct raw H/s for efficiency calc
                        double raw_hs = gpu_hashrates[i];
                        for (int u = 0; u < gpu_unit_indices[i]; u++) raw_hs *= 1000.0;
                        double eff = raw_hs / watts;
                        int effUnit = 0;
                        while (eff >= 1000 && effUnit < 5) { effUnit++; eff /= 1000.0; }

                        printf("\n[ GPU #%d | PCIe ID: %s | %s | %lf%sH/s | %.1fW | %.2f%sH/W | A:%d R:%d ]",
                            i, HIP_pcieID[i].c_str(), HIP_names[i].c_str(),
                            gpu_hashrates[i], units[gpu_unit_indices[i]].c_str(),
                            watts, eff, units[effUnit].c_str(),
                            a, r);
                        continue;
                    }
                }

                printf("\n[ GPU #%d | PCIe ID: %s | %s | %lf%sH/s | A:%d R:%d ]",
                    i, HIP_pcieID[i].c_str(), HIP_names[i].c_str(),
                    gpu_hashrates[i], units[gpu_unit_indices[i]].c_str(),
                    a, r);
            }

            fflush(stdout);
            setcolor(BRIGHT_WHITE);
        }
#endif

        // CPU hashrate line (in hybrid mode when both CPU and GPU are active)
#ifdef TNN_HIP
        if (cpuMine && gpuMine && hashrate > 0.001) {
            int cpuA = deviceAccepted[DEVICE_SHARE_CPU].load(std::memory_order_relaxed);
            int cpuR = deviceRejected[DEVICE_SHARE_CPU].load(std::memory_order_relaxed);
            setcolor(BRIGHT_CYAN);
            printf("\n[ CPU | %lf%sH/s | A:%d R:%d ]",
                hashrate, units[unitIdx].c_str(), cpuA, cpuR);
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
        }
#endif

        // Compute aggregate hashrate for status line (CPU + GPU)
        double agg_hashrate = hashrate;
        int agg_unitIdx = unitIdx;
#ifdef TNN_HIP
        if (gpuMine && cpuMine) {
            // Reconstruct raw H/s for CPU
            double agg_raw = hashrate;
            for (int u = 0; u < unitIdx; u++) agg_raw *= 1000.0;
            // Add all GPU raw H/s
            for (int i = 0; i < HIP_deviceCount; i++) {
                if (!shouldUseDevice(i)) continue;
                double gpu_raw = gpu_hashrates[i];
                for (int u = 0; u < gpu_unit_indices[i]; u++) gpu_raw *= 1000.0;
                agg_raw += gpu_raw;
            }
            agg_hashrate = agg_raw;
            agg_unitIdx = 0;
            while (agg_hashrate >= 1000 && agg_unitIdx < 5) {
                agg_unitIdx++;
                agg_hashrate /= 1000.0;
            }
        } else if (gpuMine) {
            // GPU-only: aggregate from GPU rates
            double agg_raw = 0.0;
            for (int i = 0; i < HIP_deviceCount; i++) {
                if (!shouldUseDevice(i)) continue;
                double gpu_raw = gpu_hashrates[i];
                for (int u = 0; u < gpu_unit_indices[i]; u++) gpu_raw *= 1000.0;
                agg_raw += gpu_raw;
            }
            agg_hashrate = agg_raw;
            agg_unitIdx = 0;
            while (agg_hashrate >= 1000 && agg_unitIdx < 5) {
                agg_unitIdx++;
                agg_hashrate /= 1000.0;
            }
        }
#endif

        // Overall status line
        setcolor(BRIGHT_WHITE);
        if (!gpuMine) std::cout << "\r";
        else std::cout << "\n";

        std::cout << std::setw(2) << std::setfill('0') << consoleLine << versionString << " " << std::flush;
        setcolor(CYAN);
        std::cout << std::setw(2) << std::setprecision(3)
                  << "HASHRATE " << agg_hashrate << units[agg_unitIdx] << "H/s" << " | " << std::flush;

        std::string uptime =
            std::to_string(daysUp) + "d-" +
            std::to_string(hoursUp) + "h-" +
            std::to_string(minutesUp) + "m-" +
            std::to_string(secondsUp) + "s >> ";

        double dPrint;
        switch(miningProfile.coin.miningAlgo) {
            case ALGO_ASTROBWTV3:
            case ALGO_XELISV2:
            case ALGO_XELISV3:
            case ALGO_RX0:
            case ALGO_VERUS:
            case ALGO_SHAI_HIVE:
                dPrint = difficulty;
                break;
            case ALGO_SPECTRE_X:
            case ALGO_ASTRIX_HASH:
            case ALGO_NXL_HASH:
            case ALGO_HOOHASH:
            case ALGO_WALA_HASH:
            case ALGO_YESPOWER:
            case ALGO_RINHASH:
                dPrint = doubleDiff;
                break;
            default:
                dPrint = doubleDiff;
                break;
        }

        std::cout << std::setw(2) << "ACCEPTED " << accepted
                  << std::setw(2) << " | REJECTED " << rejected
                  << std::setw(2) << " | DIFFICULTY "
                  << std::setw(6) << std::setfill(' ') << dPrint
                  << std::setw(2) << " | UPTIME " << uptime
                  << std::flush;

        setcolor(BRIGHT_WHITE);
        fflush(stdout);

        reportCounter = 0;
    }

    return 0;
}