// This is the sample code for the SA-IS algorithm presented in our article:
// G. Nong, S. Zhang and W. H. Chan, Two Efficient Algorithms for Linear Time Suffix Array Construction, 
// IEEE Transactions on Computers, Vol. 60, No. 10, Oct. 2011. 
// which draft can be retrieved at: http://code.google.com/p/ge-nong/

#include <iostream>
#include <stdlib.h>
using namespace std;
#include <memory.h>
#include <time.h>

const int EMPTY=0xffffffff;
unsigned char mask[]={0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
#define tget(i) ( (t[(i)/8]&mask[(i)%8]) ? 1 : 0 )
#define tset(i, b) t[(i)/8]=(b) ? (mask[(i)%8]|t[(i)/8]) : ((~mask[(i)%8])&t[(i)/8])

#define chr(i) (cs==sizeof(int)?((int*)s)[i]:((unsigned char *)s)[i])
#define isLMS(i) (i>0 && tget(i) && !tget(i-1))

clock_t start, finish;
double  duration;

void timer_start() {
	start = clock();
}

void timer_finish(const char *str) {
	finish = clock(); 
	duration = (double)(finish - start) / CLOCKS_PER_SEC; 
	fprintf( stderr, "\n%s: %.3lf seconds", str, duration);
}

// compute the head or end of each bucket
void getBuckets(unsigned char *s, int *bkt, int n, int K, int cs, bool end) { 
  int i, sum=0;
  for(i=0; i<=K; i++) bkt[i]=0; // clear all buckets
  for(i=0; i<n; i++) bkt[chr(i)]++; // compute the size of each bucket
  for(i=0; i<=K; i++) { sum+=bkt[i]; bkt[i]=end ? sum-1 : sum-bkt[i]; }
}

// compute SAl
void induceSAl(unsigned char *t, int *SA, unsigned char *s, int *bkt, 
               int n, int K, int cs, int level) { 
  int i, j;
  getBuckets(s, bkt, n, K, cs, false); // find heads of buckets
  if(level==0) bkt[0]++; 
  for(i=0; i<n; i++)
    if(SA[i]!=EMPTY) {
	  j=SA[i]-1; 
	  if(j>=0 && !tget(j)) SA[bkt[chr(j)]++]=j;
    }
}

// compute SAs
void induceSAs(unsigned char *t, int *SA, unsigned char *s, int *bkt, 
               int n, int K, int cs) { 
  int i, j;
  getBuckets(s, bkt, n, K, cs, true); // find ends of buckets
  for(i=n-1; i>=0; i--)
    if(SA[i]!=EMPTY) {
	  j=SA[i]-1; 
	  if(j>=0 && tget(j)) SA[bkt[chr(j)]--]=j;
    }
}

// find the suffix array SA of s[0..n-1] in {0..K}^n;
// require s[n-1]=0 (the virtual sentinel!), n>=2;
// use a space of at most 6.25n+(1) for a constant alphabet;
// level starts from 0.
void SA_IS(unsigned char *s, int *SA, int n, int K, int cs, int level) {
  static double redu_ratio=0;
  static long sum_n=0, sum_n1=0;
  fprintf(stderr, "\nLevel: %d", level);

  int i, j;
  unsigned char *t=(unsigned char *)malloc(n/8+1); // LS-type array in bits

  // stage 1: reduce the problem by at least 1/2

  timer_start();
  // Classify the type of each character
  tset(n-2, 0); tset(n-1, 1); // the sentinel must be in s1, important!!!
  for(i=n-3; i>=0; i--) tset(i, (chr(i)<chr(i+1) || (chr(i)==chr(i+1) && tget(i+1)==1))?1:0);

  int *bkt = (int *)malloc(sizeof(int)*(K+1)); // bucket counters

  // sort all the S-substrings
  getBuckets(s, bkt, n, K, cs, true); // find ends of buckets
  for(i=0; i<n; i++) SA[i]=EMPTY;
  for(i=n-2; i>=0; i--)
    if(isLMS(i)) SA[bkt[chr(i)]--]=i;
  SA[0]=n-1; // set the single sentinel LMS-substring

  induceSAl(t, SA, s, bkt, n, K, cs, level); 
  induceSAs(t, SA, s, bkt, n, K, cs); 

  timer_finish("Time for sorting all the LMS-substrings");

  free(bkt);

  timer_start();

  // compact all the sorted substrings into the first n1 items of s
  // 2*n1 must be not larger than n (proveable)
  int n1=0;
  for(i=0; i<n; i++)
    if(isLMS(SA[i]))
      SA[n1++]=SA[i];

  // Init the name array buffer
  for(i=n1; i<n; i++) SA[i]=EMPTY;
  // find the lexicographic names of all substrings
  int name=0, prev=-1;
  for(i=0; i<n1; i++) {
	  int pos=SA[i]; bool diff=false;
    for(int d=0; d<n; d++)
      if(prev==-1 || pos+d==n-1 || prev+d==n-1 ||
         chr(pos+d)!=chr(prev+d) ||
         tget(pos+d)!=tget(prev+d))
      { diff=true; break; }
      else
        if(d>0 && (isLMS(pos+d) || isLMS(prev+d)))
          break;

    if(diff) 
      { name++; prev=pos; }
	  pos=pos/2; //(pos%2==0)?pos/2:(pos-1)/2;
    SA[n1+pos]=name-1; 
  }
  for(i=n-1, j=n-1; i>=n1; i--)
	  if(SA[i]!=EMPTY) SA[j--]=SA[i];

   // s1 is done now
  int *SA1=SA, *s1=SA+n-n1;

  timer_finish("Time for naming");

  // stage 2: solve the reduced problem

  fprintf(stderr, "\nReduction ratio: %.2lf", (double)n1/n);
  redu_ratio += (double)n1/n;
  sum_n1+=n1; sum_n+=n;
  // recurse if names are not yet unique
  if(name<n1) {
    SA_IS((unsigned char*)s1, SA1, n1, name-1, sizeof(int), level+1);
  } else { // generate the suffix array of s1 directly
    for(i=0; i<n1; i++) SA1[s1[i]] = i;
	cerr << endl << "Recusion ends";
	fprintf(stderr, "\nMean reduction ratio over iterations: %.2lf", redu_ratio/(level+1));
	fprintf(stderr, "\nMean reduction ratio over characters: %.2lf", (double)sum_n1/sum_n);
  }

  fprintf(stderr, "\nLevel: %d", level);

  // stage 3: induce the result for the original problem

  timer_start();

  bkt = (int *)malloc(sizeof(int)*(K+1)); // bucket counters

  // put all left-most S characters into their buckets
  getBuckets(s, bkt, n, K, cs, true); // find ends of buckets
  j=0;
  for(i=1; i<n; i++)
    if(isLMS(i)) s1[j++]=i; // get p1
  for(i=0; i<n1; i++) SA1[i]=s1[SA1[i]]; // get index in s1
  for(i=n1; i<n; i++) SA[i]=EMPTY; // init SA[n1..n-1]
  for(i=n1-1; i>=0; i--) {
	  j=SA[i]; SA[i]=EMPTY;
	  if(level==0 && i==0)
          SA[0]=n-1;
      else
          SA[bkt[chr(j)]--]=j;
  }

  induceSAl(t, SA, s, bkt, n, K, cs, level); 
  induceSAs(t, SA, s, bkt, n, K, cs); 
  timer_finish("Time for sorting all the suffixes");

  free(bkt); 
  free(t);
}