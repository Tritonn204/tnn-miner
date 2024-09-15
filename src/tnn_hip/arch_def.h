#pragma once

/*
below will be a macro to define all architecture target spec symbols in one go
kernel prefix -> batch size -> threads per block (tuned for shared memory use)

will generate sym_BATCH_SIZE, sym_BLOCKS, sym_THREADS for use in compile-time kernel dimensioning
*/
#define archDim(sym, batchSize, threads) \
  constexpr int sym##_THREADS = threads; \
  constexpr int sym##_BLOCKS = ((batchSize+threads-1)/threads); \ 
  constexpr int sym##_BATCH_SIZE = batchSize;