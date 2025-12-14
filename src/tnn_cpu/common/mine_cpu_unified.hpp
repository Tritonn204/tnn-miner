#pragma once
#include "cpu_miner.hpp"
#include "../../net/net.hpp"
#include <hex.h>
#include <base64.hpp>
#include <terminal.hpp>
#include <stratum/stratum.h>
#include <random>

// Unified CPU mining function that works with any registered algorithm
// This replaces the individual mine* functions with a unified implementation
inline void mineCPU_unified(int tid, const std::string& algo_name) {
    int64_t localJobCounter = -1;
    int64_t localOurHeight = 0;
    int64_t localDevHeight = 0;
    int64_t localDifficulty = 0;
    int64_t localDifficultyDev = 0;

    std::unique_ptr<CPUMiner> miner;
    std::unique_ptr<CPUMiner> dev_miner;

    try {
        // Create miners for both user and dev mining
        miner = std::make_unique<CPUMiner>(algo_name, tid);
        dev_miner = std::make_unique<CPUMiner>(algo_name, tid);

        if (!miner->initialize() || !dev_miner->initialize()) {
            setcolor(RED);
            std::cerr << "Failed to initialize CPU miner for thread " << tid << std::endl;
            setcolor(BRIGHT_WHITE);
            return;
        }
    } catch (const std::exception& e) {
        setcolor(RED);
        std::cerr << "Error creating CPU miner: " << e.what() << std::endl;
        setcolor(BRIGHT_WHITE);
        return;
    }

    auto& config = miner->get_config();

    // Random number generator for dev mining probability
    thread_local std::random_device rd;
    thread_local std::mt19937 rng(rd());
    thread_local std::uniform_real_distribution<double> dist(0, 10000);

    thread_local uint64_t localCount = 0;

    // Buffers for mining
    uint8_t hash_output[32];
    uint64_t found_nonce;
    std::vector<uint8_t> work_output(config.template_size);

waitForJob:
    while (!isConnected) {
        CHECK_CLOSE;
        boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    }

    while (!ABORT_MINER) {
        try {
            boost::json::value myJob;
            boost::json::value myJobDev;
            {
                std::scoped_lock<boost::mutex> lockGuard(mutex);
                myJob = job;
                myJobDev = devJob;
                localJobCounter = jobCounter;
            }

            // Determine job field name based on protocol
            const char* jobFieldName = "template";  // Default for KAS-family
            if (miningProfile.protocol >= PROTO_XELIS_SOLO && miningProfile.protocol <= PROTO_XELIS_STRATUM) {
                jobFieldName = "miner_work";  // Xelis uses different field
            }

            // Validate job - safely check if field exists and is valid
            if (!myJob.is_object()) {
                continue;
            }
            auto* jobField = myJob.as_object().if_contains(jobFieldName);
            if (!jobField || !jobField->is_string()) {
                continue;
            }
            if (ourHeight == 0 && devHeight == 0) {
                continue;
            }
            // Wait for difficulty to be set before mining
            if (ourHeight > 0 && difficulty == 0) {
                continue;
            }
            if (devHeight > 0 && difficultyDev == 0) {
                continue;
            }

            // Update user work if height changed
            if (ourHeight > 0 && localOurHeight != ourHeight) {
                std::vector<uint8_t> work_data(config.template_size);

                // Protocol-specific work parsing
                switch (miningProfile.protocol) {
                case PROTO_XELIS_SOLO:
                case PROTO_XELIS_STRATUM:
                    hexstrToBytes(std::string(myJob.at(jobFieldName).as_string()), work_data.data());
                    break;
                case PROTO_XELIS_XATUM:
                {
                    std::string b64 = base64::from_base64(std::string(myJob.at(jobFieldName).as_string().c_str()));
                    std::memcpy(work_data.data(), b64.data(), b64.size());
                    break;
                }
                case PROTO_KAS_SOLO:
                case PROTO_KAS_STRATUM:
                    hexstrToBytes(std::string(myJob.at(jobFieldName).as_string()), work_data.data());
                    break;
                default:
                    // Generic hex parsing
                    hexstrToBytes(std::string(myJob.at(jobFieldName).as_string()), work_data.data());
                    break;
                }

                miner->set_work(work_data.data(), config.template_size);
                miner->set_difficulty(difficulty);
                localOurHeight = ourHeight;
                localDifficulty = difficulty;
            }

            // Update difficulty if it changed (without job change)
            if (ourHeight > 0 && localDifficulty != difficulty) {
                miner->set_difficulty(difficulty);
                localDifficulty = difficulty;
            }

            // Update dev work if height changed
            const char* devJobFieldName = "template";  // Default for KAS-family
            if (devMiningProfile.protocol >= PROTO_XELIS_SOLO && devMiningProfile.protocol <= PROTO_XELIS_STRATUM) {
                devJobFieldName = "miner_work";
            }

            // Safely check dev job validity
            bool devJobValid = false;
            if (devConnected && myJobDev.is_object()) {
                auto* devJobField = myJobDev.as_object().if_contains(devJobFieldName);
                devJobValid = (devJobField && devJobField->is_string());
            }

            if (devJobValid) {
                if (devHeight > 0 && localDevHeight != devHeight) {
                    std::vector<uint8_t> dev_work_data(config.template_size);

                    switch (devMiningProfile.protocol) {
                    case PROTO_XELIS_SOLO:
                    case PROTO_XELIS_STRATUM:
                        hexstrToBytes(std::string(myJobDev.at(devJobFieldName).as_string()), dev_work_data.data());
                        break;
                    case PROTO_XELIS_XATUM:
                    {
                        std::string b64 = base64::from_base64(std::string(myJobDev.at(devJobFieldName).as_string().c_str()));
                        std::memcpy(dev_work_data.data(), b64.data(), b64.size());
                        break;
                    }
                    case PROTO_KAS_SOLO:
                    case PROTO_KAS_STRATUM:
                        hexstrToBytes(std::string(myJobDev.at(devJobFieldName).as_string()), dev_work_data.data());
                        break;
                    default:
                        hexstrToBytes(std::string(myJobDev.at(devJobFieldName).as_string()), dev_work_data.data());
                        break;
                    }

                    dev_miner->set_work(dev_work_data.data(), config.template_size);
                    dev_miner->set_difficulty(difficultyDev);
                    localDevHeight = devHeight;
                    localDifficultyDev = difficultyDev;
                }

                // Update dev difficulty if it changed (without job change)
                if (devHeight > 0 && localDifficultyDev != difficultyDev) {
                    dev_miner->set_difficulty(difficultyDev);
                    localDifficultyDev = difficultyDev;
                }
            }

            // Inner mining loop
            while (localJobCounter == jobCounter) {
                CHECK_CLOSE;

                // Decide whether to do dev mining this iteration
                double which = dist(rng);
                bool devMine = (devConnected && devHeight > 0 && which < devFee * 100.0);

                CPUMiner* active_miner = devMine ? dev_miner.get() : miner.get();

                // Mine one hash
                bool found = active_miner->mine_one(hash_output, &found_nonce, work_output.data());

                // Batch update counter every 512 hashes
                if (++localCount >= 512) {
                    counter.fetch_add(localCount);
                    localCount = 0;
                }

                // Check if job changed
                if (localJobCounter != jobCounter || localOurHeight != ourHeight) {
                    if (localCount) {
                        counter.fetch_add(localCount);
                        localCount = 0;
                    }
                    break;
                }

                // Handle solution if found
                if (found) {
                    bool& submittingFlag = devMine ? submittingDev : submitting;
                    int64_t& localHeightCheck = devMine ? localDevHeight : localOurHeight;
                    int64_t& heightCheck = devMine ? devHeight : ourHeight;

                    // Wait for previous submission to complete
                    bool submit = !submittingFlag;
                    if (!submit) {
                        for(;;) {
                            submit = !submittingFlag;
                            if (submit || localJobCounter != jobCounter || localHeightCheck != heightCheck)
                                break;
                            boost::this_thread::yield();
                        }
                    }

                    // Verify still valid
                    if (localJobCounter != jobCounter || localHeightCheck != heightCheck) {
                        if (localCount) {
                            counter.fetch_add(localCount);
                            localCount = 0;
                        }
                        break;
                    }

                    submittingFlag = true;

                    // Print message
                    if (devMine) {
                        setcolor(CYAN);
                        std::cout << "\n(DEV) Thread " << tid << " found a dev share\n" << std::flush;
                        setcolor(BRIGHT_WHITE);
                    } else {
                        setcolor(BRIGHT_YELLOW);
                        std::cout << "\nThread " << tid << " found a nonce!\n" << std::flush;
                        setcolor(BRIGHT_WHITE);
                    }

                    // Build share based on protocol
                    auto& shareTarget = devMine ? devShare : share;
                    auto& profile = devMine ? devMiningProfile : miningProfile;
                    auto& jobData = devMine ? myJobDev : myJob;
                    const std::string& workerNameStr = devMine ? devWorkerName : workerName;

                    switch (profile.protocol) {
                    // Xelis protocols
                    case PROTO_XELIS_SOLO:
                        shareTarget = {{"block_template", hexStr(work_output.data(), config.template_size).c_str()}};
                        break;
                    case PROTO_XELIS_XATUM:
                    {
                        std::string b64 = base64::to_base64(std::string((char*)work_output.data(), config.template_size));
                        shareTarget = {
                            {"data", b64.c_str()},
                            {"hash", hexStr(hash_output, 32).c_str()}
                        };
                        break;
                    }
                    case PROTO_XELIS_STRATUM:
                    {
                        auto* jobId = jobData.as_object().if_contains("jobId");
                        if (jobId && jobId->is_string()) {
                            shareTarget = {{
                                {"id", XelisStratum::submitID},
                                {"method", XelisStratum::submit.method.c_str()},
                                {"params", {
                                    workerNameStr,
                                    jobId->as_string().c_str(),
                                    hexStr((byte*)&found_nonce, 8).c_str()
                                }}
                            }};
                        } else {
                            submittingFlag = false;
                            continue;
                        }
                        break;
                    }

                    // KAS-family protocols
                    case PROTO_KAS_SOLO:
                        shareTarget = {{"block_template", hexStr(work_output.data(), config.template_size).c_str()}};
                        break;
                    case PROTO_KAS_STRATUM:
                    {
                        auto* jobId = jobData.as_object().if_contains("jobId");
                        if (jobId && jobId->is_string()) {
                            std::vector<char> nonceStr;
                            Num(std::to_string(found_nonce).c_str(), 10).print(nonceStr, 16);
                            shareTarget = {{
                                {"id", KasStratum::submitID},
                                {"method", KasStratum::submit.method.c_str()},
                                {"params", {
                                    workerNameStr,
                                    jobId->as_string().c_str(),
                                    std::string(nonceStr.data()).c_str()
                                }}
                            }};
                        } else {
                            submittingFlag = false;
                            continue;
                        }
                        break;
                    }

                    default:
                        // Generic share format
                        shareTarget = {{"block_template", hexStr(work_output.data(), config.template_size).c_str()}};
                        break;
                    }

                    data_ready = true;
                    cv.notify_all();
                }

                if (!isConnected) {
                    if (localCount) {
                        counter.fetch_add(localCount);
                        localCount = 0;
                    }
                    break;
                }
            }

            if (localCount) {
                counter.fetch_add(localCount);
                localCount = 0;
            }

            if (!isConnected) {
                break;
            }
        }
        catch (std::exception& e) {
            setcolor(RED);
            std::cerr << "Error in POW Function: " << e.what() << std::endl;
            setcolor(BRIGHT_WHITE);

            if (localCount) {
                counter.fetch_add(localCount);
                localCount = 0;
            }

            localJobCounter = -1;
            localOurHeight = -1;
            localDevHeight = -1;
            localDifficulty = 0;
            localDifficultyDev = 0;
        }

        if (!isConnected) {
            break;
        }
    }

    goto waitForJob;
}
