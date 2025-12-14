#include "reporter.hpp"
#include <numeric>
#include <iostream>

const std::string units[] = {" ", " K", " M", " G", " T", " P"};
#ifdef TNN_RANDOMX
extern std::atomic<bool> datasetInitInProgress;
#endif

#ifdef TNN_HIP
extern std::atomic<bool> g_mining_started;
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

        for (int i = 0; i < HIP_deviceCount; i++) {
            uint64_t currentHashesG = HIP_counters[i].load();
            HIP_counters[i].store(0);

            if (currentHashesG > 0) {
                any_gpu_hashing = true;
                beQuiet = prev_beQuiet;
            }

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

    double ratio = 1.0;
    if (rate30sec.size() <= 30) {
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

    // =========================================================================
    // Check if mining has started (first non-zero hashrate)
    // =========================================================================

    #ifdef TNN_HIP
    if (!g_mining_started.load()) {
        if (any_gpu_hashing || hashrate > 0.001) {
            g_mining_started.store(true);
            
            // Print a separator to visually mark end of tuning output
            printf("\n");
            printf("============================================================\n");
            printf("[MINER] Mining started, hashrate reporting enabled\n");
            printf("============================================================\n");
            fflush(stdout);
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
        if (gpuMine) {
            setcolor(BRIGHT_YELLOW);

            for (int i = 0; i < HIP_deviceCount; i++) {
                printf("\n[ GPU #%d | PCIe ID: %s | %s | %lf%sH/s ]",
                    i,
                    HIP_pcieID[i].c_str(),
                    HIP_names[i].c_str(),
                    gpu_hashrates[i],
                    units[gpu_unit_indices[i]].c_str());
            }

            fflush(stdout);
            setcolor(BRIGHT_WHITE);
        }

        // Overall status line
        setcolor(BRIGHT_WHITE);
        if (!gpuMine) std::cout << "\r";
        else std::cout << "\n";

        std::cout << std::setw(2) << std::setfill('0') << consoleLine << versionString << " " << std::flush;
        setcolor(CYAN);
        std::cout << std::setw(2) << std::setprecision(3)
                  << "HASHRATE " << hashrate << units[unitIdx] << "H/s" << " | " << std::flush;

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