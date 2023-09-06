/*
 *	archon.c	(C) kvark, 2007
 *		Archon project
 *	Suffix Array sorting algorithm.
 *	Method:	Two-stage (IT-2)
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <stdio.h>
#include "archon.h"

#define REVERSED  // defined in Makefile
#define USE_IT2   // defined in Makefile

static void timetool(int num)
{
#ifdef VERBOSE
	static const int dots = 10;
	static int total = 0, next = 0;
	for (; num < 0 || num > next; next += (total + dots) / (dots + 1))
	{
		if (num > next)
		{
			putchar('+');
			continue;
		} // now num<0
		total = -num;
		next = 0;
		putchar('[');
		for (num = 1; num <= dots; num++)
			putchar('-');
		putchar(']');
		for (num = 0; num <= dots; num++)
			putchar('\b');
	}
#endif // VERBOSE
}

#define NBIT 8
#define HI(num) ((int)(num) << NBIT)
#define NSYM HI(1)
#define LO(num) (num & (NSYM - 1))
#define N2SYM HI(NSYM)

// byte *gin;					// input byte array
// static int *p;				// suffix array itself
// static int n, base, sorted; // numbers

void getbounds(int id, int **top, int **bot, struct akData* AK)
{
	// works even with USE_IT2 disabled
	id = CPU2LE16p(AK->gin + DEAD + id - 2);
	*top = AK->p + AK->r2[id];
	*bot = AK->p + AK->ra[id] - 1;
}

int sufcheckArchon(int *A, int num, char verb, struct akData *AK)
{
#ifdef VERBOSE
	float sum = 0.0f;
#else  // VERBOSE
	int sum = 0;
#endif // VERBOSE
	int i, k = 0;
	if (num <= 1)
		return 0;
	timetool(-num);
	for (i = 0; i < num - 1; i++)
	{
		int com, rez;
		timetool(i);
		com = compare(A[i], A[i + 1], &rez, AK);
		sum += com;
		if (!rez)
			k++;
		if (k && verb == '-')
			break;
	}
	if (verb != '+')
		return k;
#ifdef VERBOSE
	printf("\nresult: %d faults\n", k);
	printf("Neighbor LCP: %.2f\n", sum / (num - 1));
#endif // VERBOSE
	return 0;
}

// int geninit(FILE *ff)
// {
// 	extern int memory;
// 	n = ftell(ff);
// 	memory = sizeof(rx);
// 	gin = GETMEM(DEAD + n + 1, byte);
// 	p = GETMEM(n + 1, int);
// 	if (!p || !gin)
// 		return 0;
// 	fseek(ff, 0, SEEK_SET);
// 	fread(gin + DEAD, 1, n, ff);
// 	return memory;
// }

int gencode(struct akData *AK)
{
	int memory;
	memset(AK->gin, 0, DEAD);
	AK->r2 = GETMEM(N2SYM + 1, int, memory);
	AK->ra = GETMEM(N2SYM + 1, int, memory);
	if (!AK->r2 || !AK->ra)
		return -1;
	if (!ankinit(AK))
		return -2;
	AK->sorted = 0;
	return memory;
}
// #ifdef VERBOSE
// static void printeff(char *str, int use, int rit, int let, int dif)
// {
// 	int log;
// 	if (!*sorted || !let || !use)
// 		return;
// 	for (log = 0; use >> log; log++)
// 		;
// 	printf("%s count:2^%02d\t usage:%.1f%c\t eff:%.1f%c\t dif:%d\n",
// 		   str, log, 100.f * rit / sorted, '%', 100.f * rit / let, '%', dif / use);
// }
// void genprint()
// {
// 	printf("base: %d\t sorted: %.1f%c\n", base, 100.f * sorted / n, '%');
// 	ankprint(printeff);
// }
// #else  // VERBOSE
// void genprint()
// {
// }
// #endif // VERBOSE

void genexit()
{
	ankexit();
	// FREE(r2);
	// FREE(ra);
	// FREE(gin);
	// FREE(p);
}

static void sortbuckets(struct akData *AK)
{
	enum
	{
		POST = sizeof(dword)
	};
	int px[] = {1, 0};
	byte cl = 0;
	ray(px, 2, POST, AK);
	do
	{
		int ch, high = HI(cl + 1);
		if (AK->rx[cl] + 1 >= AK->rx[cl + 1])
			continue;
		ch = HI(cl);
		do
		{
			int num = AK->ra[ch] - AK->r2[ch];
			if (num < 2)
				continue;
			ray(AK->p + AK->r2[ch], num, sizeof(sword) + POST, AK);
			AK->sorted += num;
		} while (++ch != high);
		timetool(AK->rx[cl]);
	} while (++cl);
	timetool(AK->n);
}

int compute(struct akData *AK)
{
	// returns: base>=0 on success
	byte *in = AK->gin + DEAD;
	if (!AK->n)
		return -1;
	AK->base = -1;
#ifndef REVERSED
	/*DO: reverse */ {
		byte *x, *z, tm;
		REVER(in, in + n, tm);
	}
#endif //! REVERSED
	memset(AK->rx, 0, NSYM * sizeof(int));
	memset(AK->r2, 0, N2SYM * sizeof(int));
	/*DO: count frequences */ {
		byte *fly, *fin = in + AK->n - 1;
		for (fly = in; fly < fin; fly++)
			AK->r2[CPU2LE16p(fly)]++;
	}
	/*DO: shift frequences */ {
		int i, ch = N2SYM;
		AK->r2[ch] = i = AK->n;
		do
		{
			int cl = NSYM;
			do
			{
				i -= AK->r2[--ch];
				AK->ra[ch] = AK->r2[ch] = i;
			} while (--cl);
			AK->rx[cl = ch >> NBIT] = i;
			if (cl == *in)
				AK->p[--i] = 1, --AK->r2[ch];
		} while (ch);
	}
#ifdef USE_IT2
	/*DO: get direct sort values */ {
		byte *fly = in, *fin = in + AK->n;
		*fin = NSYM - 1;
		do
			if (fly[0] > fly[1] && fly[0] >= fly[-1])
				AK->p[AK->ra[CPU2LE16p(fly)]++] = fly + 2 - in, ++fly;
		while (++fly < fin);
	}
	sortbuckets(AK); // direct sort

	/*DO: two wave sort */ {
		byte cl;
		int i;
		memcpy(AK->ra, AK->r2 + 1, N2SYM * sizeof(int));
		in[AK->n] = NSYM - 1; // Right2Left wave
		cl = 0;
		do
		{
			int lim = (--cl, AK->ra[HI(cl) + cl]);
			for (i = AK->r2[HI(cl + 1)]; i-- > lim;)
			{
				int cc = in[AK->p[i]];
				if (cc <= cl)
					AK->p[--AK->ra[HI(cc) + cl]] = AK->p[i] + 1;
			}
			for (lim = AK->ra[HI(cl) + cl]; i >= lim; --i)
				if (in[AK->p[i]] == cl)
					AK->p[--lim] = AK->p[i] + 1;
		} while (cl);
		in[AK->n] = in[0]; // Left2Right wave
		i = 0;
		cl = 0;
		do
		{
			int ch = AK->r2[HI(cl + 1)] - AK->r2[HI(cl)];
			while (ch--)
			{
				int pos;
				byte sym = in[pos = AK->p[i++]];
				if (pos == AK->n)
					AK->base = i - 1;
				else if (sym >= cl)
					AK->p[AK->rx[sym]++] = pos + 1;
			}
		} while (++cl);
	}
#else  // USE_IT2
	/*DO: total direct sort */ {
		/*DO: count stats */ {
			byte *fly, *fin = in + n - 1;
			for (fly = in; fly < fin; fly++)
				p[ra[CPU2LE16p(fly)]++] = fly + 2 - in;
		}
		sortbuckets(AK);
		base = r2[CPU2LE16p(in + n - 2)];
		while (p[*base] != n)
			base++;
	}
#endif // USE_IT2
#ifdef VERBOSE
	printf("\n");
#endif // VERBOSE
	return AK->base;
}

int verify(int *rx, byte *gin, int *p, int n)
{
	// returns: 0 on success
	int i = n;
	byte sym;
	byte *in = gin + DEAD;
	// i = sufcheck(p,n,'+');
	memset(rx, 0, NSYM * sizeof(int));
	while (i--)
		rx[in[p[i] - 1]] = i;
	timetool(-n);
	rx[in[0]]++;
	for (i = 0; i < n; i++)
	{
		if (p[i] == n)
			continue;
		sym = in[p[i]];
		if (p[rx[sym]] != 1 + p[i])
			break;
		rx[sym]++;
		timetool(i);
	}
	timetool(n);
	return n - i;
}

// int encode(FILE *ff)
// {
// 	byte *in = gin + DEAD;
// 	int i;
// 	in[n] = in[0];
// 	for (i = 0; i != n; ++i)
// 		putc(in[p[i]], ff);
// 	fwrite(&base, sizeof(int), 1, ff);
// 	return 0;
// }

// int decode(FILE *ff)
// {
// 	int i, k;
// 	byte *in = gin + DEAD;
// 	n -= sizeof(int);
// 	if (n < 0)
// 		return -1;
// 	base = *(int *)(in + n);

// 	memset(rx, 0, NSYM * sizeof(int));
// 	for (i = 0; i < n; i++)
// 		rx[in[i]]++;
// 	k = NSYM;
// 	do
// 	{
// 		--k;
// 		i -= rx[k];
// 		rx[k] = i;
// 	} while (k);
// #define INEX rx[in[i]]++
// #ifdef REVERSED
// #define IOPER p[i] = INEX
// 	k = base;
// #else // REVERSED
// #define IOPER p[INEX] = i
// 	k = p[base];
// #endif // REVERSED
// 	i = base;
// 	IOPER;
// 	for (i = 0; i < base; i++)
// 		IOPER;
// 	for (i = base + 1; i < n; i++)
// 		IOPER;
// 	for (i = n; i--; k = p[k])
// 		putc(in[k], ff);
// #undef IOPER
// #undef INEX
// 	return n;
// }

extern int archonsort(unsigned char *T, int *SA, int size)
{

	//   gin = T;
	//   p = SA;
	//   n = size;

	struct akData AK;

	AK.gin = T;
	AK.p = SA;
	AK.n = size;

	gencode(&AK);

	compute(&AK);
	/*
	  if(verify() != 0) {
		fprintf(stderr, "error\n");
		exit(1);
	  }
	*/
	// ankexit();

	memcpy(SA, AK.p, size);

	// free(AK.gin);
	// free(AK.p);
	// FREE(r2);
	// FREE(ra);
	// genexit();
	return 0;
}

#ifdef __cplusplus
}
#endif