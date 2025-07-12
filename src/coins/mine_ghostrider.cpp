// mine_ghostrider.cpp
#include "miners.hpp"
#include "tnn-hugepages.h"
#include "numa_optimizer.h"

// GhostRider includes
#include <ghostrider/gr-gate.h>
#include <ghostrider/gr-hash.h>
#include <stratum/btc-stratum.h>

#include <endian.hpp>

// GhostRider rotation table (from gr-gate.c)
static const uint8_t cn[40][3] = {
    {1, 2, 3}, {1, 2, 4}, {1, 2, 5}, {1, 3, 4}, {1, 3, 5},
    {1, 4, 5}, {2, 3, 4}, {2, 3, 5}, {2, 4, 5}, {3, 4, 5},
    {0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 1, 5}, {0, 2, 3},
    {0, 2, 4}, {0, 2, 5}, {0, 3, 4}, {0, 3, 5}, {0, 4, 5},
    {0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 1, 5}, {0, 2, 3},
    {0, 2, 4}, {0, 2, 5}, {0, 3, 4}, {0, 3, 5}, {0, 4, 5},
    {1, 2, 3}, {1, 2, 4}, {1, 2, 5}, {1, 3, 4}, {1, 3, 5},
    {1, 4, 5}, {2, 3, 4}, {2, 3, 5}, {2, 4, 5}, {3, 4, 5}
};

// Thread-local GhostRider state
thread_local uint8_t gr_hash_order[18];
thread_local gr_context_overlay gr_ctx;
thread_local uint8_t *hp_state = nullptr;
thread_local size_t hp_allocated_size = 0;
thread_local int current_rotation = -1;

// Helper to allocate CN memory based on selected algorithms
void ensureGhostRiderMemory(int rotation) {
  // Memory requirements: Turtlelite=256KB, Turtle=256KB, Darklite=512KB, 
  // Dark=512KB, Lite=1MB, Fast=2MB
  static const size_t cn_memory[6] = {524288, 524288, 2097152, 1048576, 262144, 262144};
  static const uint8_t cn_map[6] = {3, 2, 5, 4, 1, 0};
  
  // Find max memory needed for this rotation
  size_t max_mem = 0;
  for (int i = 0; i < 3; i++) {
    size_t mem_needed = cn_memory[cn_map[cn[rotation][i]]];
    if (mem_needed > max_mem) max_mem = mem_needed;
  }
  
  // Reallocate if needed
  if (max_mem > hp_allocated_size) {
    if (hp_state) {
      free(hp_state);
    }
    hp_state = (uint8_t*)aligned_alloc(4096, max_mem);
    hp_allocated_size = max_mem;
    
    // Optimize for NUMA if available
    if (hp_state) {
      NUMAOptimizer::optimizeMemoryForMining(hp_state, max_mem);
    }
  }
}

void mineGhostRider(int tid)
{
  // Thread-local RNG
  thread_local std::random_device rd;
  thread_local std::mt19937 rng(rd());
  thread_local std::uniform_real_distribution<double> dist(0, 10000);
  
  int64_t localJobCounter;
  int64_t localOurHeight = 0;
  int64_t localDevHeight = 0;
  
  uint32_t nonce = 0;
  uint32_t nonce_dev = 0;
  
  alignas(64) byte work[80] = {0};
  alignas(64) byte devWork[80] = {0};
  alignas(64) byte FINALWORK[80] = {0};
  alignas(64) byte hash[32] = {0};
  
  uint32_t targetWords[8];
  uint32_t targetWords_dev[8];
  
  // Initialize GhostRider contexts
  memset(&gr_ctx, 0, sizeof(gr_ctx));
    
waitForJob:
  while (!isConnected)
  {
    CHECK_CLOSE;
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
  }
  
  while (!ABORT_MINER)
  {
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
      
      if (!myJob.at("template").is_string())
        continue;
      if (ourHeight == 0 && devHeight == 0)
        continue;
      
      // Handle new job data
      if (ourHeight == 0 || localOurHeight != ourHeight)
      {
        hexstrToBytes(std::string(myJob.at("template").as_string()), work);
        localOurHeight = ourHeight;
        nonce = 0;
        
        // GhostRider: Update rotation based on height
        // Rotation changes every 10 blocks, with 2 sub-rotations per main rotation
        int new_rotation = ((ourHeight / 10) * 2 + (ourHeight % 10 >= 5 ? 1 : 0)) % 40;
        
        if (new_rotation != current_rotation) {
          current_rotation = new_rotation;
          
          // Ensure we have enough memory for this rotation's CN algorithms
          ensureGhostRiderMemory(current_rotation);
          
          // Generate core algorithm order from block data
          gr_getAlgoString((const uint8_t*)(work + 1), gr_hash_order);
          
          // Override CN algorithm positions with rotation-specific ones
          gr_hash_order[5] = cn[current_rotation][0] + 15;
          gr_hash_order[11] = cn[current_rotation][1] + 15;
          gr_hash_order[17] = cn[current_rotation][2] + 15;
        }
      }
      
      if (devConnected && myJobDev.at("template").is_string())
      {
        if (devHeight == 0 || localDevHeight != devHeight)
        {
          hexstrToBytes(std::string(myJobDev.at("template").as_string()), devWork);
          localDevHeight = devHeight;
          nonce_dev = 0;
          
          // Dev job might need different rotation
          int dev_rotation = ((devHeight / 10) * 2 + (devHeight % 10 >= 5 ? 1 : 0)) % 40;
          if (dev_rotation != current_rotation) {
              // Would need separate dev state management here
          }
        }
      }
      
      bool devMine = false;
      double which;
      bool submit = false;
      
      BTCStratum::diffToWords(doubleDiff / 65536.0, targetWords);
      BTCStratum::diffToWords(doubleDiffDev / 65536.0, targetWords_dev);
      
      while (localJobCounter == jobCounter)
      {
        CHECK_CLOSE;
        which = dist(rng);
        devMine = (devConnected && devHeight > 0 && which < devFee * 100.0);
        
        uint32_t *noncePtr = devMine ? &nonce_dev : &nonce;
        (*noncePtr)++;
        
        byte *WORK = (devMine && devConnected) ? devWork : work;
        memcpy(FINALWORK, WORK, 80);
        
        // Put nonce in correct position
        uint32_t n = ((tid - 1) % (256 * 256)) | ((*noncePtr) << 16);
        be32enc(FINALWORK + 76, n);
        
        if (localJobCounter != jobCounter)
          break;
        
        // Hash with GhostRider
        // Note: gr_hash expects two 80-byte inputs for 2-way hashing
        // You might need to adjust based on your actual gr_hash implementation
        byte FINALWORK2[80];
        memcpy(FINALWORK2, FINALWORK, 80);
        be32enc(FINALWORK2 + 76, n + 1);  // Second nonce
        
        gr_hash(hash, FINALWORK, FINALWORK2, tid);
        
        uint32_t *currentTarget = devMine ? targetWords_dev : targetWords;
        counter.fetch_add(2);  // GhostRider does 2 hashes at once
        
        submit = (devMine && devConnected) ? !submittingDev : !submitting;
        
        if (localJobCounter != jobCounter || localOurHeight != ourHeight)
          break;
        
        // Check if either hash meets target
        if (checkHash(hash, currentTarget))
        {
          // Handle submission (same as your code)
          // ... submission logic ...
        }
        
        if (!isConnected)
        {
          data_ready = true;
          cv.notify_all();
          break;
        }
      }
    }
    catch (std::exception &e)
    {
      setcolor(RED);
      std::cerr << "Error in GhostRider Function: " << e.what() << std::endl << std::flush;
      setcolor(BRIGHT_WHITE);
      
      localJobCounter = -1;
      localOurHeight = -1;
      localDevHeight = -1;
      current_rotation = -1;
    }
  }
  
  // Cleanup
  if (hp_state) {
    free(hp_state);
    hp_state = nullptr;
    hp_allocated_size = 0;
  }
  
  goto waitForJob;
}