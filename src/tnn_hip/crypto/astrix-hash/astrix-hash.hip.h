#pragma once

#include <inttypes.h>

namespace Astrix_HIP {
  constexpr int INPUT_SIZE = 80;
  constexpr int MAX_NONCES = 640;
  constexpr int THREAD_DIM = 1024;
  /**
   * Update the Astrix matrix data on the currently active GPU
   * @param  in  A host pointer to the start of the new prePowHash
   */
  void newMatrix(uint8_t *in, bool isDev = false);

  /**
   * Call the astrixHash_hip kernel with launch parameters provided
   * 
   * @param  blocks  The amount of blocks to launch the POW kernel with
   * @param  final_nonces  A device pointer to a buffer for storing all found nonces for this POW round
   * @param  nonce_count  A device pointer to a modifiable tally of found nonces for this POW round
   * @param  kIndex  The current run count/index for the current job on the active GPU
   */
  void astrixHash_wrapper(
    int blocks,
    const uint64_t nonce_mask, 
    const uint64_t nonce_fixed, 
    uint64_t *final_nonces, 
    int *nonce_count, 
    int kIndex, 
    size_t batchSize, 
    uint8_t device = 0,
    bool isDev = false
  );
}