#pragma once

// below will be a macro to define all architecture target spec symbols in one go
#define archDim(sym, batchSize, threads) \
  constexpr int sym##_THREADS = threads; \
  constexpr int sym##_BLOCKS = ((batchSize+threads-1)/threads);