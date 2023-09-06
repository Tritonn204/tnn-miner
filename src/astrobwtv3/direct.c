/*
 *	direct.c	(C) kvark, 2007
 *		Archon project
 *	Direct string sorting routine.
 *	Includes protection technique.
 *	Method:	Powered Anchors
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <assert.h>
#include "direct.h"

#define USE_FWAN // defined in Makefile
#define USE_BKAN // defined in Makefile

// static int use_backward,use_forward;
// static int jbk_let,jbk_rit,jbk_dif;
// static int jfw_let,jfw_rit,jfw_dif;

typedef dword value; // compare unit type
// static dword *bar;	//bit array (in dwords)
// static int **jak;	//anchors array
// static byte *dep;	//new: depth array

#define JALOG (8)
#define DLOG (7)
#define DMASK ((1 << DLOG) - 1)
#define LOPER (7)

static __inline int *divide(int *, int *, byte *, value);
static __inline int *split(int *, int, int, int, struct akData *AK);

#ifdef USE_BKAN
static __inline void jupdate(int *A, int num, byte bdep, struct akData *AK)
{
	assert(num > 1);
	do
	{
		int id = (DEAD + A[0]) >> JALOG;
		if (AK->dep[id] < bdep)
			continue;
		AK->jak[id] = A;
		AK->dep[id] = bdep;
	} while (++A, --num);
}
#endif // USE_BKAN
#ifdef USE_FWAN
static __inline void shell(int *A, int num)
{
	static int h[20];
	int x, i;
	for (i = 0, x = 1; (h[i] = x) < num; x = 3 * x + 1, ++i)
		;
	for (x = i; x;)
	{
		int j, k, sh = h[--x];
		for (j = 0; j != sh; ++j)
		{
			for (k = j + sh; k < num; k += sh)
			{
				int val = A[i = k];
				while ((i -= sh) >= j && A[i] > val)
					A[i + sh] = A[i];
				A[i + sh] = val;
			}
		}
	}
}
#endif // USE_FWAN

int ankinit(struct akData *AK)
{
	int memory;
	int jn, bsiz = 1 + (AK->n >> 5);
	// common init
	AK->bar = GETMEM(bsiz, dword, memory);
	if (!AK->bar)
		return 0;
	memset(AK->bar, 0, bsiz * sizeof(dword));
	// backward init
	jn = 1 + ((DEAD + AK->n) >> JALOG);
	AK->jak = GETMEM(jn, int *, memory);
	AK->dep = GETMEM(jn, byte, memory);
	memset(AK->jak, 0, jn * sizeof(int *));
	memset(AK->dep, -1, jn * sizeof(byte));
	// exit4
	if (!AK->jak || !AK->dep)
		return -1;
	return memory;
}
void ankexit()
{
	// FREE(jak); FREE(dep); FREE(bar);
}
void ankprint(void (*prin)(char *, int, int, int, int))
{
	// prin("Back\t", use_backward, jbk_rit, jbk_let, jbk_dif);
	// prin("Front\t", use_forward, jfw_rit, jfw_let, jfw_dif);
}

/*	anchors: start		*/

#define BAR(op, bar) (bar[(id) >> 5] op(1 << ((id)&31)))
__inline static void mark(int id, dword *bar)
{
	assert(id >= 0);
	BAR(^=, bar);
}
__inline static dword marked(int id, dword *bar)
{
	assert(id >= 0);
	return BAR(&, bar);
}
#undef BAR

#ifdef USE_BKAN
static int *towuse(int *A, int num, int *depth, struct akData *AK)
{
	int per = *depth - 5, k = 0, *ank = NULL;
	do
	{ // sf := current string end
		int sf = DEAD + A[k] - *depth;
		int **lim = AK->jak + (sf >> JALOG);
		int dif = (DEAD + A[k]) >> JALOG; // or (sf+DMASK)
		int **t = AK->jak + dif;
		byte *d = AK->dep + dif;
		assert(sf >= 0);
		do
		{
			if (!*t || (dif = A[k] - **t) <= 0)
				continue;
			if (dif >= *depth - 2 - (*d << DLOG))
				continue;
			if (dif < per)
				per = dif, ank = *t;
			break;
		} while (--d, t-- != lim);
	} while (++k < num);
	if (ank)
		*depth = per;
	return ank;
}
static void towback(int *A, int *B, int per, int *ank, struct akData *AK)
{
	int *x, *z, *lo = A, *hi = B;
	int *bot, *top, sf; // step-1: mark source
	assert(A + 1 < B);
	x = lo;
	do
		mark(x[0] - per, AK->bar);
	while (++x != hi);
	getbounds(*ank, &bot, &top, AK);
	mark(*(x = z = ank, AK->bar), AK->bar);
	*lo++ = per + *x;
	assert(bot <= ank && ank <= top);
	do
	{ // step-2: fill-up destiny
		printf("while loop iteration\n");
		while (x > bot && marked(*--x, AK->bar))
		{
			// printf("*lo: %d\n", *lo);
			*lo++ = per + *x;
			// printf("*lo++: %d\n", *lo);
			mark(*x, AK->bar);
		}
		while (z < top && marked(*++z, AK->bar))
		{
			*--hi = per + *z;
			mark(*z, AK->bar);
		}
		assert(lo == hi || x > bot || z < top);
	} while (lo < hi);
	AK->jbk_dif += per; // step-3: reverse parts
	AK->jbk_let += 1 + z - x;
	AK->jbk_rit += B - A;
	REVER(A, lo, sf);
	REVER(hi, B, sf);
}
#endif // USE_BKAN

#ifdef USE_FWAN
static void towfront(int *A, int *B, int depth, int per, struct akData *AK)
{
	int nof, *x, *z;
	assert(A + 1 < B);
	// step-1: find root nodes
	x = A;
	do
		mark(x[0], AK->bar);
	while (++x != B);
	x = A;
	z = B;
	do
	{
		nof = x[0] - per;
		if (nof < 0 || !marked(nof, AK->bar))
		{
			int id = *x; // get a root node
			*x = *--z;
			*z = id;
		}
		else
			x++;
	} while (x != z);
	// step-2: sort them & prepare
	z = A;
	do
		mark(z[0], AK->bar);
	while (++z != B);
	assert(x != B);
	if (x == A)
	{
		ray(x, B - x, depth, AK);
		return;
	}
	else
		z = split(x, B - x, depth, *A, AK);
	// done roots
	AK->jfw_let += B - A;
	AK->jfw_rit += B - x;
	AK->jfw_dif += per;
	nof = 0;
	do
		mark(A[nof] - per, AK->bar);
	while (++nof != x - A);
	for (nof = 0; x + nof != z; ++nof)
		A[nof] = x[nof];
	x = A + nof;
	// step-3: roll-up front anchors
	while (A < x)
	{
		nof = *A++;
		if (!marked(nof, AK->bar))
			continue;
		*x++ = per + nof;
		mark(nof, AK->bar);
	}
	while (B > z)
	{
		nof = *--B;
		if (!marked(nof, AK->bar))
			continue;
		*--z = per + nof;
		mark(nof, AK->bar);
	} // magic :)
	assert(x == z);
}
#endif // USE_FWAN

/*	anchors: end		*/

/*	BeSe: start		*/

int compare(int x, int y, int *pr, struct akData *AK)
{
	byte *gin = AK->gin;
	int com = 0;
	value *li = (value *)(gin + DEAD) - 1;
	value *pa = (value *)(gin + DEAD + x);
	value *pb = (value *)(gin + DEAD + y);
	while (*--pa == *--pb && pa > li && pb > li)
		com++;
	if (pr)
	{
		if (pa <= li || pb <= li)
			pr[0] = pb > pa;
		else
			pr[0] = CPU2LE32p(pb) > CPU2LE32p(pa);
		//		else pr[0] = pb[0]>pa[0];
	}
	return com;
}

#define STEP (int)sizeof(value)
#define GETVAL(bin, id) CPU2LE32p(bin + (id))

static __inline int *divide(int *A, int *num, byte *bin, value w)
{
	int *x = A, *y = A, *z = A + *num;
	for (;;)
	{
		int s;
		value q = GETVAL(bin, s = *y);
		if (q <= w)
		{
			if (q != w)
				*y = *x, *x++ = s;
			if (++y == z)
				break;
		}
		else
		{
			if (--z == y)
				break;
			*y = *z, *z = s;
		}
	}
	*num = z - x;
	return x;
}

static int jtandem(int *A, int *B, int *x, int num, int depth, struct akData *AK)
{
#ifdef USE_FWAN
	int per = B - A, mid = per - num; // excluded
	/*DO: charging */ {
		static int charge = 0, *emid = NULL;
		if (emid < A || emid >= B)
		{
			charge = 0;
			emid = A + (per >> 1);
		}
		if ((charge += num, !mid) || (mid -= charge >> LOPER,
									  charge = 0, mid >= 0))
			return 0;
	}
	shell(x, num);
	per = depth;
	while (--num)
	{
		int dif = x[1] - x[0];
		++x;
		assert(dif > 0);
		if (dif < per)
			per = dif;
	}
	if (per + STEP > depth)
		return 0;
	towfront(A, B, depth, per, AK);
	AK->use_forward++;
	return -1;
#else  // USE_FWAN
	return 0;
#endif // USE_FWAN
}

static int borders(int *A, int *B, int depth)
{
	static int *emin = NULL;
	if (emin < A || emin >= B)
	{
		emin = --B;
		while (B != A)
			if (*--B < *emin)
				emin = B;
	}
	if (STEP + *emin <= depth)
	{
		depth = *emin;
		*emin = *A;
		*A = depth;
		emin = A;
		return -1;
	}
	else
		return 0;
}

static __inline int *split(int *A, int num, int depth, int kor, struct akData *AK)
{
	byte *gin = AK->gin;
	byte *bin = gin + DEAD - depth;
	while (num)
	{
		value w = GETVAL(bin, kor);
		int *x = A, *z = A + num;
		if (!w && borders(A, z, depth))
			x++;
		x = divide(x, &num, bin, w);
		if (A + 1 < x)
			ray(A, x - A, depth, AK);
		A = x + num;
		if (A + 1 < z)
			ray(A, z - A, depth, AK);
		A -= num;
		bin -= STEP;
		depth += STEP;
	}
	return A;
}

void ray(int *A, unsigned int num, int depth, struct akData *AK)
{
	assert(num > 1);
	for (;;)
	{
		byte *gin = AK->gin;
		int *x, *z;
		/*DO: ternary qsort */ {
			byte *bin = gin + DEAD - depth;
			value w = GETVAL(bin, A[num >> 1]);
			z = (x = A) + num;
			if (!w && borders(A, z, depth))
			{
				int s = A[num - 1];
				w = GETVAL(bin, s);
				assert(s + STEP > depth);
				--num;
				x++;
			}
			x = divide(x, &num, bin, w);
		}
		if (num > 1 && jtandem(A, z, x, num, depth, AK))
			return;
		if (A + 1 < x)
			ray(A, x - A, depth, AK);
		A = x + num;
		if (A + 1 < z)
			ray(A, z - A, depth, AK);
		A = x;
		if (num <= 1)
			return;


#ifdef USE_BKAN
		if ((depth & DMASK) < STEP)
		{
			int bdep, *ank = towuse(A, num, &depth, AK);
			if (ank)
			{
				towback(A, A + num, depth, ank, AK);
				AK->use_backward++;
				return;
			}
			bdep = (depth - sizeof(sword) + DMASK) >> DLOG;
			if (bdep <= 0xFF)
			{
				ray(A, num, depth + STEP, AK);
				jupdate(A, num, (byte)bdep, AK);
				assert(bdep);
				return;
			}
		}
#endif // USE_BKAN
		depth += STEP;
	}
}
#undef GETVAL
#undef STEP

#ifdef __cplusplus
}
#endif

/*	BeSe: end		*/
