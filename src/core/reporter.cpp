#include "reporter.hpp"
#include <numeric>
#include <iostream>

const std::string units[] = {" ", " K", " M", " G", " T", " P"}; // Note the space

int update_handler(const boost::system::error_code& error)
{
  CHECK_CLOSE_RET(0);
  if (error == boost::asio::error::operation_aborted) {
    return 1;
  }

  // Set an expiry time relative to now.
  update_timer.expires_after(std::chrono::seconds(1));

  // Start an asynchronous wait.
  update_timer.async_wait(update_handler);

  if (!isConnected) {
    return 1;
  }

  reportCounter++;

  auto now = std::chrono::steady_clock::now();
  //auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_start_time).count();

  auto daysUp = std::chrono::duration_cast<std::chrono::hours>(now - g_start_time).count() / 24;
  auto hoursUp = std::chrono::duration_cast<std::chrono::hours>(now - g_start_time).count() % 24;
  auto minutesUp = std::chrono::duration_cast<std::chrono::minutes>(now - g_start_time).count() % 60;
  auto secondsUp = std::chrono::duration_cast<std::chrono::seconds>(now - g_start_time).count() % 60;

  if (gpuMine) {
    setcolor(BRIGHT_YELLOW);

    // if (reportCounter >= reportInterval) printf("\n");
    for (int i = 0; i < HIP_deviceCount; i++) {
      uint64_t currentHashesG = HIP_counters[i].load();
      HIP_counters[i].store(0);

      double ratioG = 1.0 * 1;
      if (HIP_rates30sec[i].size() <= (30 / 1))
      {
        HIP_rates30sec[i].push_back((int64_t)(currentHashesG * ratioG));
      }
      else
      {
        HIP_rates30sec[i].erase(HIP_rates30sec[i].begin());
        HIP_rates30sec[i].push_back((int64_t)(currentHashesG * ratioG));
      }

      double hashrateG = 1.0 * (double)std::accumulate(HIP_rates30sec[i].begin(), HIP_rates30sec[i].end(), 0LL) / (HIP_rates30sec[i].size() * 1);

      int unitIdxG = 0;

      for (;;) {
        if (hashrateG < 1000) break;
        unitIdxG++;
        hashrateG /= 1000.0;
      }

      if (reportCounter >= reportInterval)
        printf("\n[ GPU #%d | PCIe ID: %s | %s | %lf%sH/s ]", 
          i, 
          HIP_pcieID[i].c_str(), 
          HIP_names[i].c_str(), 
          hashrateG, 
          units[unitIdxG].c_str()
        );
    }
    fflush(stdout);
    setcolor(BRIGHT_WHITE);
  }

  uint64_t currentHashes = counter.load();
  counter.store(0);

  double ratio = 1.0 * 1;
  if (rate30sec.size() <= (30 / 1))
  {
    rate30sec.push_back((int64_t)(currentHashes * ratio));
  }
  else
  {
    rate30sec.erase(rate30sec.begin());
    rate30sec.push_back((int64_t)(currentHashes * ratio));
  }

  double hashrate = 1.0 * (double)std::accumulate(rate30sec.begin(), rate30sec.end(), 0LL) / (rate30sec.size() * 1);
  // hashrate = (hashrate * 1.0) / (double)1;

  int unitIdx = 0;

  for (;;) {
    if (hashrate < 1000) break;
    unitIdx++;
    hashrate /= 1000.0;
  }

  if (reportCounter >= reportInterval) {
    setcolor(BRIGHT_WHITE);
    if (!gpuMine) std::cout << "\r";
    else std::cout << "\n";
    std::cout << std::setw(2) << std::setfill('0') << consoleLine << versionString << " " << std::flush;
    setcolor(CYAN);
    std::cout << std::setw(2) << std::setprecision(3) << "HASHRATE " << hashrate << units[unitIdx] << "H/s" << " | " << std::flush;

    std::string uptime = std::to_string(daysUp) + "d-" +
                  std::to_string(hoursUp) + "h-" +
                  std::to_string(minutesUp) + "m-" +
                  std::to_string(secondsUp) + "s >> ";

    double dPrint;

    switch(miningAlgo) {
      case DERO_HASH:
      case XELIS_HASH:
        dPrint = difficulty;
        break;
      case SPECTRE_X:
        dPrint = doubleDiff;
        break;
      case RX0:
      case VERUSHASH:
        dPrint = difficulty;
        break;
      case ASTRIX_HASH:
      case NXL_HASH:
      case HOOHASH:
      case WALA_HASH:
        dPrint = doubleDiff;
        break;
    }

    std::cout << std::setw(2) << "ACCEPTED " << accepted << std::setw(2) << " | REJECTED " << rejected
              << std::setw(2) << " | DIFFICULTY " << std::setw(6) << std::setfill(' ') << dPrint << std::setw(2) << " | UPTIME " << uptime << std::flush;
    setcolor(BRIGHT_WHITE); 
    fflush(stdout);

    reportCounter = 0;
  }

  return 0;
}