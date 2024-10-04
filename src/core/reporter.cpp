#include "reporter.hpp"
#include <numeric>
#include <iostream>

int update_handler(const boost::system::error_code& error)
{
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
  std::string units[] = {" ", " K", " M", " G", " T", " P"}; // Note the space

  for (;;) {
    if (hashrate < 1000) break;
    unitIdx++;
    hashrate /= 1000.0;
  }

  if (reportCounter >= reportInterval) {
    setcolor(BRIGHT_WHITE);
    std::cout << "\r" << std::setw(2) << std::setfill('0') << consoleLine << versionString << " " << std::flush;
    setcolor(CYAN);
    std::cout << std::setw(2) << std::setprecision(3) << "HASHRATE " << hashrate << units[unitIdx] << "H/s" << " | " << std::flush;

    std::string uptime = std::to_string(daysUp) + "d-" +
                  std::to_string(hoursUp) + "h-" +
                  std::to_string(minutesUp) + "m-" +
                  std::to_string(secondsUp) + "s >> ";

    double dPrint;

    switch(miningAlgo) {
      case DERO_HASH:
        dPrint = difficulty;
        break;
      case XELIS_HASH:
        dPrint = difficulty;
        break;
      case SPECTRE_X:
        dPrint = doubleDiff;
        break;
      case RX0:
        dPrint = difficulty;
        break;
      case VERUSHASH:
        dPrint = difficulty;
        break;
      case ASTRIX_HASH:
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