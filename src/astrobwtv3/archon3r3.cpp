/*
*	Archon 3-rel-3	(C) kvark, 2006
*		Archon project
*	Burrows-Wheeler Transformation algoritm
*
*		Anchors + DeepRay + Lazy + Isab
*		Direct sorting (a<b>=c)
*		Units: dword, no bswap
*/

//#define SUFCHECK
#ifndef NDEBUG
  #define NDEBUG
#endif
#define VERBLEV	0

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <memory.h>
#include <limits.h>
#include <assert.h>

#include "archon3r3.h"

#define INSERT	10
#define MAXST	64
#define DEEP	150
#define DEELCP	10000
#define OVER	1000
#define TERM	((1+OVER)*sizeof(ulong))
#define LCPART	8
#define ABIT	7

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef int trax2[0x10001];

int BS,n,*p;
int baza,ch;
uchar *s_bin,sym;
uchar *sfin, *bin;
int rb[0x100];
trax2 r,r2a;

int call_indiana, call_fw_anch, call_bk_anch;
int call_fw_buck, call_split, call_pseudo;
int call_digger, call_lazy_ray;
int call_deep_ray, call_smart_ins;

int ndis,nlcp,numst;
uchar *offs;
int **anch;
short *lcp;

clock_t t0;
FILE *fi,*fo;
int lucky;

#define p2b(pt) (*(ushort *)(pt))
#define p4b(pt) (*(ulong *)(pt))
#define BADSUF(pid) (pid[0]>pid[1] && pid[0]>=pid[-1])

void ray(workerData &worker, int *, int *, uchar *);

struct TLonStr	{
	uchar *sb;
	int len,off;
	char up;
}lst[MAXST+1],
*milen,*mac,*lc;

void init_hardcore(workerData &worker)	{ 
  worker.lucky++;
	memset(offs, 0, ndis*sizeof(uchar));
	memset(anch, 0, ndis*sizeof(int*));
}
void exit_hardcore(workerData &worker)	{
	worker.lucky=0;
}

#ifdef SUFCHECK
int sufcheck(int A[], int B[])	{ int *x;
	for(x=A; x<B-1; x++)	{
		uchar *a = bin-3+x[0];
		uchar *b = bin-3+x[1];
		while(p4b(a)==p4b(b)) a-=4,b-=4;
		if(p4b(a) > p4b(b)) return x-A;
	}return -1;
}
#endif

void encode(workerData &worker) {
    uchar cl, *fly;
    int i, pos, lim;
    worker.baza = -1; // Radix sort
    memset(r, 0, sizeof(trax2));
    worker.sfin = worker.bin + worker.data_len; //scan
    for (fly = worker.bin; fly < worker.sfin - 1; fly++)
        r[p2b(fly)]++;
    r[worker.ch = 0x10000] = pos = worker.data_len;
    i = 256; do {
        i--;
        cl = 0; do {
            pos -= r[--worker.ch];
            r2a[worker.ch] = r[worker.ch] = pos;
        } while (--cl);
        rb[i] = pos;
        if ((uchar)i == *worker.bin) {
            worker.sa[--pos] = 0; r[worker.ch]--;
        }//for start
    } while (i);
    worker.sfin[0] = 0xFF; fly = worker.bin; //border
    do if (BADSUF(fly))
        worker.sa[r2a[p2b(fly)]++] = fly + 1 - worker.bin, fly++;
    while (++fly < worker.sfin);
    // Direct sort
    lst[0].len = worker.data_len; numst = 0;
    lst[0].sb = worker.bin - 5;
    for (worker.lucky = 0, worker.ch = 0; worker.ch < 0x10000; worker.ch++) {
        ray(worker, worker.sa + r[worker.ch], worker.sa + r2a[worker.ch], worker.bin - 5);
    }//Right2Left wave
    if (worker.lucky) exit_hardcore(worker);
    memcpy(r2a, r + 1, sizeof(trax2) - sizeof(int));
    *worker.sfin = 0xFF;
    cl = 0; do {
        cl--;
        lim = r2a[(cl << 8) + cl];
        for (i = r[(uint)(cl + 1) << 8] - 1; i >= lim; i--) {
            unsigned char cc = worker.bin[pos = worker.sa[i] + 1];
            if (cc <= cl) worker.sa[--r2a[(cc << 8) + cl]] = pos;
        }
        for (lim = r2a[(cl << 8) + cl]; i >= lim; i--)
            if (worker.bin[pos = worker.sa[i] + 1] == cl)
                worker.sa[--lim] = pos;
    } while (cl);
    //Left2Right wave
    *worker.sfin = 0x00; putc(*worker.bin, fo);
    cl = 0; i = 0; do {
        worker.ch = r[(uint)(cl + 1) << 8] - r[cl << 8];
        while (worker.ch--) {
            if ((pos = 1 + worker.sa[i++]) == worker.data_len) {
                worker.baza = i; putc(*worker.bin, fo);
                continue;
            }//finish
            uchar sym = worker.bin[pos]; putc(sym, fo);
            if (sym >= cl) worker.sa[rb[sym]++] = pos;
        }
    } while (++cl);
    fwrite(&worker.baza, 4, 1, fo);
}


void deep_ray(workerData &worker, int*,int*,uchar*);
#define LENMAX	32000
/*
*	lcp_qsort - ternary qsort with lcp keys
*/
void lcp_qsort(workerData &worker, int P[], int a, int b, uchar *bof)	{
	do	{ int x=a,y=a,z=b;
		short wk = lcp[(a+b)>>1];
		do	{ short qk = lcp[y];
			int s = P[y];
			if(qk > wk)	{ z--;
				P[y] = P[z]; P[z] = s;
				lcp[y] = lcp[z]; lcp[z] = qk;
			}else
			if(qk < wk)	{
				P[y] = P[x]; P[x] = s;
				lcp[y] = lcp[x]; lcp[x] = qk;
				y++; x++;
			}else y++;
		}while(y<z);
		if(a+1 < x) lcp_qsort(worker, P,a,x,bof);
		if(x+1 < y)	{ int cur = lcp[x];
			if(cur <= 0) cur += LENMAX;
			else cur = LENMAX - cur;
			deep_ray(worker, P+x,P+y,bof-(cur<<2));
		}a=z;
	}while(b-a > 1);
}

/*
*	lazy_ray - my invented lcp multi sort
*/
void lazy_ray(workerData &worker, int A[], const int num, uchar bof[])	{
	int i,s;
	call_lazy_ray++; assert(lucky);
	lcp[0] = 0; s = A[0];
	for(i=1; i<num; i++)	{ short last = 0;
		uchar *a = A[i]+bof, *b = s+bof;;
		while(p4b(a)==p4b(b) && ++last<LENMAX) a-=4,b-=4;
		lcp[i] = (p4b(a) > p4b(b) ? LENMAX-last : last-LENMAX);
	}//qsort
	lcp_qsort(worker, A,0,num,bof);
}
#undef LENMAX

/*
*	isab - smart string compare routine
*/
int isab(uchar *a,uchar *b)	{
	int cof,i; uchar *bx;
	cof = (a>b ? (bx=a)-b : (bx=b)-a);
	//if(!cof) return -1;
	mac = milen = lst; //choose period
	for(lc=lst+1; lc <= lst+numst; lc++)	{
		if(lc->len < milen->len) milen=lc;
		if(lc->off == cof  &&  lc->sb - lc->len < bx)
			if(lc->sb > mac->sb) mac=lc;
	}//continue until border
	for(i = bx-mac->sb; i>=0; i-=4)	{
		if(p4b(a) != p4b(b)) break;
		a-=4; b-=4;
	}//replace old bound
	bx += DEEP;
	if(i>0 || mac==lst)	{
		int rez = p4b(a)>p4b(b) || b<=lst[0].sb;
		int clen = bx-(a>b?a:b)-4;
		if(numst < MAXST)	{
			milen = lst+(++numst);
			milen->len = 0;
		}//replace-add
		if(clen > milen->len)	{
			milen->sb = bx;
			milen->len = clen;
			milen->off = cof;
			milen->up = (rez == (a>b));
		}return rez;
	}//update bound
	if(bx > mac->sb)	{
		mac->len += bx-mac->sb;
		mac->sb = bx;
	}return (mac->up == (a>b));
}

int median(int a,int b,int c,uchar bof[])	{
	uint qa = p4b(a+bof), qb = p4b(b+bof), qc = p4b(c+bof);
	if(qa > qb)	return (qa<qc ? a : (qb>qc?b:c));
	else		return (qb<qc ? b : (qa>qc?a:c));
}
/*
*	deep_ray - the deep BeSe implementation
*	key is uint64 instead of uint32
*/
void deep_ray(workerData &worker, int *A, int *B, uchar *boff)	{
	int *x,*y,*z; ulong w,w2;
	call_deep_ray++; assert(lucky);
	while(B-A > INSERT)	{
		int s = median(A[0],A[(B-A)>>1],B[-1],boff);
		x=A; y=A; z=B; w = p4b(s+boff);
		w2 = p4b(s-4+boff);
		while(y<z)	{
			uint q;
			s = *y; q = p4b(s+boff);
			if(q == w)	{
				q = p4b(s-4+boff);
				if(q == w2) y++;
				else if(q > w2)	{
					*y = *--z; *z = s; 
				}else	{ //q < w2
					*y++ = *x; *x++ = s;
				}
			}else if(q > w)	{
				*y = *--z; *z = s; 
			}else	{ // q < w
				*y++ = *x; *x++ = s;
			}
		}//recurse
		if(A+1 < x) ray(worker, A,x,boff);
		if(z+1 < B) ray(worker, z,B,boff);
		A = x; B = z; boff -= 8;
		if(bin-boff>DEELCP && B-A<nlcp)	{
			lazy_ray(worker, A,B-A,boff); return;
		}
	}//insertion
	for(x=A+1; x<B; x++)	{
		int s = (z=x)[0];
		while(--z>=A && isab(boff+z[0],boff+s))
			z[1] = z[0];
		z[1] = s; //in place
	}
}

int icmp(const void *v0, const void *v1)	{
	return *(int*)v0 - *(int*)v1;
}
/*
*	indiana - general anchor sort
*	uses bit array 'mak' for speed
*/
void indiana(int A[], int B[], const int dif, int an[])	{
	int *x,*z,*hi,*lo; int pre,num=B-A;
	call_indiana++; //assert(lucky);
	qsort(A,num,sizeof(int),icmp);
	pre = p2b(bin+A[0]-dif-1);
	hi = p+r2a[pre]-1; lo = p+r[pre];
	an[0]=~an[0]; //now pre = suffixes left
	for(x=z=an, pre=num-1;;)	{ int id;
		while(x > lo && (id=dif+*--x,bsearch(&id,A,num,sizeof(int),icmp)))
			if(x[0]=~x[0],!--pre) goto allok;
		while(z < hi && (id=dif+*++z,bsearch(&id,A,num,sizeof(int),icmp)))
			if(z[0]=~z[0],!--pre) goto allok;
		assert(x>lo || z<hi);
	}allok:
	//if(z-x > (B-A<<2)) fprintf(fd,"\n%d/%d %d",B-A,z-x,dif);
	for(z=A,x--; z<B; z++)	{
		do x++; while(x[0]>=0);
		*z = (x[0]=~x[0]) + dif;
	}
}

/*
*	split - splits suffixes into 3 groups according
*	to the first 'ofs' characters from 'bof'
*/
int * split(int *A, int *B,uchar *bof,int ofs,int piv,int **end)	{
	call_split++;
	piv+=ofs; do	{ int *x,*y,*z;
		ulong w = p4b(bof + piv);
		x=y=A,z=B; do	{
			int s = *y;
			ulong q = p4b(bof + s);
			if(q == w) y++;
			else if(q<w)	{
				*y++ = *x; *x++ = s;
			}else	{ // q>w
				*y = *--z; *z = s;
			}
		}while(y<z);
		A = x; B = y; bof -= 4;
	}while(A<B && (ofs-=4) > 0);
	end[0] = B; return A;
}

#define MAXDIFF		32
#define MAXLINDIFF	0
/*
*	digger - deep routines guider
*	chooses anchor or calls deep_ray
*/
void digger(workerData &worker, int A[], int B[], uchar bof[])	{
	int min_fw,min_bk,min_cr,diff;
	int *x,*afw=0,*abk=0,*acr=0;
	if(B-A <= 1) return;
	if(!lucky) init_hardcore(worker);
	call_digger++;
	min_fw = min_cr = INT_MAX; min_bk = INT_MIN;
	for(x=A; x<B; x++)	{
		uchar *bp; int *an;
		int tj = x[0]>>ABIT;
		if(!(an = anch[tj])) continue;
		bp = bin+(tj<<ABIT)+offs[tj]-1;
		if(bp[-1] > bp[0]) continue;
		diff = bin+x[0]-bp-1;
		if(p2b(bp) == ch)	{
			if(diff>0 && diff<min_cr)
				min_cr=diff,acr=an;
		}else
		if(diff > 0)	{
			if(diff < min_fw) min_fw=diff,afw=an;
		}else	if(diff > min_bk) min_bk=diff,abk=an;
	}diff = 0;
	//if forward anchor sort
	if(afw && min_fw < MAXDIFF)	{
		call_fw_anch++;
		indiana(A,B,diff=min_fw,afw);
	}else //if backward sort
	if(abk && min_bk > -MAXDIFF)	{ int i=0;
		for(x=A+1; x<B && !i; x++)
			for(i=-min_bk; i>0 && bin[A[0]+i] == bin[x[0]+i]; i--);
			if(!i)	{ call_bk_anch++;
				indiana(A,B,diff=min_bk,abk);
			}
	}else //same bucket
	if(acr && min_cr < MAXDIFF)	{ int *z;
		x = split(A,B,bof,min_cr,*acr,&z);
		if(x+1 < z)	{ call_fw_buck++;
			indiana(x,z,diff=min_cr,acr);
		}else diff = -1;
		if(A+1 < x) deep_ray(worker, A,x,bof);
		if(z+1 < B) deep_ray(worker, z,B,bof);
	}//pseudo or deep
	if(!diff)	{ int pre,s,tj;
		uchar *spy = bin+A[0]-1;
		for(diff=0; diff<MAXLINDIFF; diff++,spy--)
			if(BADSUF(spy) && (pre=p2b(spy))<ch)
				if(r[pre+1]-r[pre] > ((B-A)<<4)) break;
		if(diff < MAXLINDIFF)	{ call_pseudo++;
			s = A[0]-diff; tj = s>>ABIT;
			assert(s>=0 && s<n);
			for(acr=p+r[pre]; acr[0]!=s; acr++);
			assert(acr < p+r2a[pre]);
			anch[tj] = acr; offs[tj] = s&AMASK;
			indiana(A,B,diff,acr);
		}else deep_ray(worker, A,B,bof);
	}//update anchors
	for(x=A; x<B; x++)	{
		int tj = x[0] >> ABIT;
		diff = x[0] & AMASK;
		if(!anch[tj] || offs[tj]>diff)	{
			anch[tj] = x; offs[tj] = diff;
		}
	}
}
#undef MAXDIFF
#undef MAXLINDIFF

/*
*	smart_ins - insertion sort limited with DEEP
*	calls 'digger' to sort bad substrings
*/
void smart_ins(workerData &worker, int A[], int B[], uchar *bof)	{
	int *x=A+1,*z;
	int badcase = 0;
	do	{ int s = *x; z=x-1;
		do	{
			int limit = (DEEP+3)>>2;
			uchar *a,*b;
			a = bof + z[0]; b = bof+s;
			while(p4b(a)==p4b(b) && --limit) a-=4,b-=4;
			if(p4b(a) < p4b(b)) break;
			if(!limit)	{
				z[0]=~z[0];
				badcase++; break;
			}//shift
			do z[1] = z[0];
			while(--z>=A && z[0]<0);
		}while(z>=A);	
		z[1] = s; //put in place
	}while(++x<B);
	if(badcase)	{ bof -= DEEP;
		call_smart_ins++;
		x=A; do	{ //skip bad
			for(z=x; z[0]<0; z++) z[0]=~z[0];
			if(x+1 < ++z) digger(worker, x,z,bof);
		}while((x=z)<B);
	}
}

/* 
*	ray - the modified mkqsort
*	deep = bin-boff, dword comparsions
*/
void ray(workerData &worker, int *A, int *B, uchar *boff)	{
	int *x,*y,*z;
	while(B-A > INSERT)	{
		int s = median(A[0],A[(B-A)>>1],B[-1],boff);
		ulong w = p4b(s+boff);
		x=y=A; z=B; do	{
			ulong q = p4b((s=*y)+boff);
			if(q < w)	{
				*y++ = *x; *x++ = s;
			}else
			if(q > w)	{
				*y = *--z; *z = s; 
			}else y++;
		}while(y<z);
		if(A+1 < x) ray(worker, A,x,boff);
		if(z+1 < B) ray(worker, z,B,boff);
		A = x; B = z; boff-=4;
		if(bin-boff > DEEP)	{
			digger(worker,A,B,boff); return;
		}
	}//insertion
	if(A+1 < B) smart_ins(worker,A,B,boff);
}
#undef p2b
#undef p4b
#undef BADSUF

#undef INSERT
#undef MAXST
#undef DEEP
#undef DEELCP
#undef OVER
#undef TERM
#undef LCPART
#undef ABIT