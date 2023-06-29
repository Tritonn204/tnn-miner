#ifndef ENDIAN
#define ENDIAN

inline bool littleEndian()
{
  int n = 1;
  // little endian if true
  if(*(char *)&n == 1) {
    return true;
  } 
  return false;
}

#endif