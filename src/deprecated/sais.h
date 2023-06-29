#ifndef SAIS
#define SAIS

#include <stdint.h>
#include <algorithm>

//begin
template <class T>
inline void trim256(T *arr) {
  // T resize[256];
  // std::copy(arr, arr+256, resize);
  // arr = resize;
}

template <class T>
inline void trimToSize(T *arr, int size) {
  // T resize[size];
  // std::copy(arr, arr+size, resize);
  // arr = resize;
}

/*func*/void text_32(unsigned char *text, int32_t *sa, int tLen, int sLen);

/*func*/void sais_8_32(unsigned char *text, int textMax, int32_t *sa, int32_t *tmp, int textLen, int saLen, int tmpLen);

/*func*/int32_t *freq_8_32(unsigned char *text, int32_t *freq, int32_t *bucket, int tLen);

/*func*/void bucketMin_8_32(unsigned char *text, int32_t *freq, int32_t *bucket, int tLen);

/*func*/void bucketMax_8_32(unsigned char *text, int32_t *freq, int32_t *bucket, int tLen);

/*func*/int placeLMS_8_32(unsigned char *text, int32_t *sa, int32_t *freq, int32_t *bucket, int tLen);

/*func*/void induceSubL_8_32(unsigned char *text, int32_t *sa, int32_t *freq, int32_t *bucket, int tLen, int saLen);

/*func*/void induceSubS_8_32(unsigned char *text, int32_t *sa, int32_t *freq, int32_t *bucket, int tLen, int saLen);

/*func*/void length_8_32(unsigned char *text, int32_t *sa, int numLMS, int tLen);

/*func*/int assignID_8_32(unsigned char *text, int32_t *sa, int textLen, int numLMS, int saLen);

/*func*/void map_32(int32_t *sa, int numLMS, int saLen);

/*func*/void recurse_32(int32_t *sa, int32_t *oldTmp, int numLMS, int maxID, int saLen, int tmpLen);

/*func*/void unmap_8_32(unsigned char *text, int32_t *sa, int numLMS, int tLen, int saLen);

/*func*/void expand_8_32(unsigned char *text, int32_t *freq, int32_t *bucket_8_64, int32_t *sa, int numLMS, int tLen, int saLen);

/*func*/void induceL_8_32(unsigned char *text, int32_t *sa, int32_t *freq, int32_t *bucket, int tLen, int saLen);

/*func*/void induceS_8_32(unsigned char *text, int32_t *sa, int32_t *freq, int32_t *bucket, int tLen, int saLen);

#endif