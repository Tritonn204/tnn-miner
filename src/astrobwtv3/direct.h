#ifndef _ARCHON_DIRECT_H_
#define _ARCHON_DIRECT_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <memory.h>
//special case for BSD
#if HAVE_MALLOC_H
#include <malloc.h>
#endif
#include <stdlib.h>

#define NDEBUG // defined in Makefile
#define DEAD	10

#define NBIT2 8
#define HI2(num) ((int)(num) << NBIT2)
#define NSYM2 HI2(1)

typedef unsigned char byte;
typedef unsigned short sword;
typedef unsigned long dword;

struct akData {
	int use_backward, use_forward;
	int jbk_let,jbk_rit,jbk_dif;
	int jfw_let,jfw_rit,jfw_dif;

  int rx[NSYM2], *r2, *ra;

  byte *gin;					// input byte array
  int *p;				// suffix array itself
  int n, base, sorted; // numbers

	dword *bar;
	int** jak;
	byte* dep;
};

// Byte swap
static __inline
sword
byteswap_16p(const sword *n) {
#if defined(__i386__)  && defined(__GNUC__)
  sword r = *n;
  __asm__("xchgb %b0, %h0" : "+q" (r));
  return r;
#elif defined(__ppc__) && defined(__GNUC__)
  sword r;
  __asm__("lhbrx %0, 0, %1" : "=r" (r) : "r"  (n), "m" (*n));
  return r;
#else
  return (((*n) << 8) & 0xff00) | (((*n) >> 8) & 0xff);
#endif
}
static __inline
dword
byteswap_32p(const dword *n) {
#if defined(__i386__) && defined(__GNUC__)
  dword r = *n;
  __asm__("bswap %0" : "+r" (r));
  return r;
#elif defined(__ppc__) && defined(__GNUC__)
  dword r;
  __asm__("lwbrx %0, 0, %1" : "=r" (r) : "r"  (n), "m" (*n));
  return r;
#else
  return (((*n) & 0xff) << 24) | (((*n) & 0xff00) << 8) |
         (((*n) >> 8) & 0xff00) | (((*n) >> 24) & 0xff);
#endif
}

#if defined(__BIG_ENDIAN__)
# define CPU2LE16p(_a) byteswap_16p((sword *)(_a))
# define CPU2LE32p(_a) byteswap_32p((dword *)(_a))
# warning Big Endian
//#elif defined(__LITTLE_ENDIAN__)
#else /* little endian or unknown endian... */
# define CPU2LE16p(_a) (*((sword *)(_a)))
# define CPU2LE32p(_a) (*((dword *)(_a)))
# warning Little Endian
#endif


int ankinit(struct akData*);
void ankexit();
void ankprint(void(*)(char*,int,int,int,int));

int compare(int,int,int*,struct akData*);
void ray(int*,unsigned int,int,struct akData*);
int sufcheckArchon(int*,int,char,struct akData*);
void getbounds(int,int**,int**,struct akData*);

// caution: REVER uses 'x' & 'z' values
#define REVER(px,pz,tm)	{ x=px,z=pz;		\
	while(x+1 < z) tm=*x,*x++=*--z,*z=tm;	\
}
#define GETMEM(num,type,mem)			\
	(mem += (num)*sizeof(type),		\
	(type*)malloc((num)*sizeof(type)))
#define FREE(ptr) if(ptr) free(ptr)

#ifdef __cplusplus
}
#endif

#endif /* _ARCHON_DIRECT_H_ */
