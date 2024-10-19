#pragma once

#include <inttypes.h>

namespace Wala_HIP {
  constexpr int INPUT_SIZE = 80;
  constexpr int MAX_NONCES = 640;
  constexpr int SHAKE_THREAD_DIM = 320;
  constexpr int THREAD_DIM = 1024;
  constexpr int HASH_HEADER_SIZE = 72;

  /**
   * Update the Wala matrix data on the currently active GPU
   * @param  in  A host pointer to the start of the new prePowHash
   */
  void newMatrix(uint8_t *in, bool isDev);

  void getHashBlocksPerSM(int *numBlocksPerSm, bool isDev = false);

  void absorbPow(uint8_t *work, bool isDev);

  void nonceCounter(int *d_nonce_count, int *h_nonce_count, uint64_t *d_final_nonces, uint64_t *h_nonce_buffer);

  template<bool isDev>
  void copyWork(uint8_t *work);

  void copyDiff(uint8_t *diff);
  /**
   * Call the walaHash_hip kernel with launch parameters provided
   * 
   * @param  blocks  The amount of blocks to launch the POW kernel with
   * @param  final_nonces  A device pointer to a buffer for storing all found nonces for this POW round
   * @param  nonce_count  A device pointer to a modifiable tally of found nonces for this POW round
   * @param  dataBuf  A device pointer to an array of 32-byte hash buffers
   * @param  kIndex  The current run count/index for the current job on the active GPU
   */
  void walaHash_wrapper(
    int blocks,
    const uint64_t nonce_mask, 
    const uint64_t nonce_fixed, 
    uint64_t *final_nonces,
    int *nonce_count, 
    uint8_t *dataBuf,
    int kIndex, 
    size_t batchSize, 
    uint8_t device,
    bool isDev
  );
}