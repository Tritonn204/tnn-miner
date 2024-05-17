#include <assert.h>
#include <string.h>
#include <stdio.h>

#include "archon4r0.h"

//--------------------------------------------------------------------------//
// This SAC implementation is based on SAIS algorithm explained here:
// http://www.cs.sysu.edu.cn/nong/t_index.files/Two%20Efficient%20Algorithms%20for%20Linear%20Suffix%20Array%20Construction.pdf
// It also uses some nice optimizations found in Yuta Mori version:
// https://sites.google.com/site/yuta256/sais
//--------------------------------------------------------------------------//

// Constants and macros
const bool sInduction = true;
const bool sTracking = true;

#define p2b(pt) (*(ushort *)(pt))
#define p4b(pt) (*(ulong *)(pt))
#define BADSUF(pid) (pid[0] > pid[1] && pid[0] >= pid[-1])

#define BIT_LMS 31
#define FLAG_LMS 1<<BIT_LMS
#define BIT_JUMP 30
#define FLAG_JUMP 1<<BIT_JUMP

void directSort(byte *data, suffix *P, t_index N)	{
  for(t_index i=0; i!=N; ++i)
    P[i] = i+1;
  ray(data, P, N, 1);
}

bool sufCompare(byte *data, suffix a, suffix b)	{
  assert(a!=b);
  t_index d=0; do ++d;
  while(a>=d && b>=d && data[a-d]==data[b-d]);
  return a>=d && (b<d || data[a-d]<data[b-d]);
}

bool bruteCheck(byte *data, suffix *A, suffix *B)	{
  for(suffix *x=A; ++x<B; )
    assert(sufCompare(data, x[-1], x[0]));
  return true;
}

void ray(byte *data, suffix *A, t_index num, unsigned depth)	{
  while(num>1)	{
    suffix *x=A, *z=A+num, s=A[num>>1];
    if(s<depth)	{
      assert(s+1==depth);
      if(num==2)
        return;
      suffix t = A[num>>1] = A[0];
      *--z=s; s=t; --num;
    }
    assert(s>=depth);
    byte w = data[s-depth];
    suffix *y = x;
    for(;;)	{
      s = *y;
      if(s>=depth)	{
        byte q = data[s-depth];
        if(q <= w)	{
          if(q != w)
            *y=*x,*x++=s;
          if(++y == z)
            break;
          continue;
        }
      }
      if(--z == y)
        break;
      *y=*z,*z=s;
    }
    y=z; z=A+num;
    num = y-x;
    ray(data, A, x-A, depth); A = x+num;
    ray(data, A, z-A, depth); A = x;
    ++depth;
  }
}

void checkData(byte *data, t_index N, t_index K)	{
  if(sizeof(byte) == sizeof(t_index))
    return;
  byte shift = sizeof(byte)<<3;
  assert(!( (K-1)>>shift ));
  if(K>>shift)
    return;
  for(t_index i=0; i<N; ++i)
    assert(data[i]>=0 && data[i]<K);
}

bool checkUnique(suffix *P, t_index num)	{
  for(t_index i=0; i!=num; ++i)
    for(t_index j=i+1; j!=num; ++j)
      if(P[i]==P[j])
        return false;
  return true;
}

void makeBuckets(byte *data, t_index *R, t_index N, t_index K)	{
  memset(R, 0, K*sizeof(t_index));
  t_index i,sum;
  for(i=0; i<N; ++i)
    ++R[data[i]];
  for(R[i=K]=sum=N; i--;)
    sum = R[i] = sum-R[i];	// cf
  assert(!sum);
}

void buckets(byte *data, t_index *R, t_index *RE, t_index *R2, t_index K, t_index N)	{
  if(R2)	{
    memcpy(RE, R2, (K-1U)*sizeof(t_index));
    R[0] = 0; R[K] = N;
  } else
    makeBuckets(data, R, N, K);
}

void fillEmpty(suffix *P, t_index off, t_index num, t_index N)	{
  //memset( P+off, 0, num*sizeof(suffix) );
  t_index i = off; off+=num;
  while(i!=off)
    P[i++] = N|FLAG_LMS;
}

template<class X>
void parseLMS(byte *data, X &x, t_index &n1, t_index N)	{
  if(!n1)
    return;
  t_index i=0,k=0;
  for(;;)	{
    while(++i, assert(i<N), data[i-1] >= data[i]);
    x.parse(k,i);
    if(++k == n1)
      break;
    while(++i, assert(i<N), data[i-1] <= data[i]);
  }
}

void findLMS(byte *data, suffix *P, t_index *R, t_index *RE, t_index *R2, t_index N, t_index K, t_index &n1)	{
  assert(!n1 && N);
  fillEmpty(P,0,N,N);
  buckets(data, R, RE, R2, K, N);
  for(t_index i=0; ; )	{
    do if(++i>=N)
      return;
    while(data[i-1] >= data[i]);
    ++n1;	//found LMS!
    P[--RE[data[i-1]]] = i;
    while(++i<N && data[i-1] <= data[i]);
  }
}

void packTargetIndices(suffix *P, t_index &n1, t_index N)	{
  // pack LMS into the first n1 suffixes
  t_index i=-1,j=0;
  while(j!=n1)	{
    suffix s;
    while(++i, assert(i<N), (s=P[i]^FLAG_LMS) & FLAG_LMS);
    P[j++] = s;
  }
}

template<typename Q>
void packTargetValues(Q *const input, suffix *P, t_index &n1, t_index &d1, t_index N)	{
  // number of words occupied by new data
  d1 = 1 + (sizeof(Q)*n1-1) / sizeof(suffix);
  assert(d1<=n1);
  // pack values into [0,d1] and
  // move suffixes into [d1,d1+n1]
  t_index j, i=n1-1;
  for(j=0; j!=n1; ++j)	{
    suffix s;
    while( !(s=P[++i]) );
    assert( i<N );
    P[d1+j] = P[j];
    input[j] = s-1U;
  }
}

void computeTargetValues(suffix *P, byte *data, t_index &n1, t_index &name)	{
  // compare LMS using known lengths
  suffix *const s1 = P+n1;
  t_index i, prev_len = 0;
  suffix prev = 0;
  for(name=0,i=0; i!=n1; ++i)	{
    const suffix cur = P[i];
    suffix &cur_len = s1[cur>>1];
    assert(cur_len);
    if(cur_len == prev_len)	{
      suffix j = 1;
      do if(data[cur-j] != data[prev-j])
        goto greater;
      while(++j<=cur_len);
    }else	{
      greater:	//warning!
      ++name; prev = cur;
      prev_len = cur_len;
    }
    cur_len = name;
  }
}

void inducePre(byte *data, suffix *P, t_index *R, t_index *RE, t_index *R2, t_index K, t_index N)	{
  // we are not interested in s>=N-1 here so we skip it
  t_index i;
  byte prev; suffix *pr=NULL;
  assert(N);
  //left2right
  buckets(data, R, RE, R2, K, N);
  pr = P + R[prev=0];
  for(i=0; i!=N; ++i)	{
    const suffix s = P[i];
    // empty space is supposed to be flagged
    if(s >= N-1)	{
      P[i] = s & ~FLAG_LMS;
      continue;
    }
    assert(s>0 && s<N-1);
    P[i] = N;	//skipped value
    const byte cur = data[s];
    assert( data[s-1] <= cur );
    if(cur != prev)	{
      R[prev] = pr-P;
      pr = P + R[prev=cur];
    }
    assert( pr>P+i && pr<P+RE[cur] );
    const suffix q = s+1;
    *pr++ = q + (cur>data[q] ? FLAG_LMS:0);
  }
  //right2left
  buckets(data, R, RE, R2, K, N);
  pr = P + RE[prev=data[0]];
  *--pr = 1 + (prev<data[1] ? FLAG_LMS:0);
  i=N; do	{
    const suffix s = P[--i];
    if(s >= N-1)
      continue;
    assert(s>0 && s<N-1);
    //P[i] = N;
    const byte cur = data[s];
    assert( data[s-1] >= cur );
    if(cur != prev)	{
      RE[prev] = pr-P;
      pr = P + RE[prev=cur];
    }
    assert( pr>P+R[cur] && pr<=P+i );
    const suffix q = s+1;
    *--pr = q + (cur<data[q] ? FLAG_LMS:0);
  }while(i);
}

// the pre-pass to sort LMS
// using additional 2K space

void inducePreFast(byte *data, suffix *P, t_index *R, t_index *RE, t_index *R2, t_index *const D, t_index N, t_index K)	{
  assert( BIT_LMS+1 == (sizeof(suffix)<<3) );
  t_index i;
  byte prev; suffix *pr=NULL;
  assert(N);
  memset( D, 0, 2*K*sizeof(t_index) );
  //left2right
  unsigned d=0;
  buckets(data, R, RE, R2, K, N);
  pr = P + R[prev=0];
  for(i=0; i!=N; ++i)	{
    suffix s = P[i];
    // empty space is supposed to be flagged
    if((s&FLAG_LMS) || (s&~FLAG_JUMP)==N-1)	{
      P[i] = s & ~FLAG_LMS;
      continue;
    }
    assert(s>0 && (s&~FLAG_JUMP)<N-1);
    P[i] = N;	//skipped value
    d += s>>BIT_JUMP;
    s &= ~FLAG_JUMP;
    const byte cur = data[s];
    assert( data[s-1] <= cur );
    if(cur != prev)	{
      R[prev] = pr-P;
      pr = P + R[prev=cur];
    }
    assert( pr>P+i && pr<P+RE[cur] );
    suffix q = s+1;
    unsigned t = (cur<<1) + (data[q]<cur);
    if(D[t] != d)	{
      q |= FLAG_JUMP;
      D[t] = d;
    }
    *pr++ = q | (t<<BIT_LMS);
  }
  //reverse flags order
  i=N; do	{
    const suffix s = P[--i];
    // exclude N+
    if(s>=N)
      continue;
    assert(s>0 && s<N);
    P[i] ^= FLAG_JUMP;
    while( assert(i>0), !(P[--i]&FLAG_JUMP) );
    P[i] ^= FLAG_JUMP;
  }while(i);
  //right2left
  buckets(data, R, RE, R2, K, N);
  pr = P + RE[prev=data[0]];
  *--pr = 1 | FLAG_JUMP | (prev<data[1] ? FLAG_LMS:0);
  i=N; ++d; do	{
    suffix s = P[--i];
    // exclude N-1 and LMS flags
    if((s&~FLAG_JUMP) >= N-1)
      continue;
    assert(s>0 && (s&~FLAG_JUMP)<N-1);
    //P[i] = N;
    d += s>>BIT_JUMP;
    s &= ~FLAG_JUMP;
    const byte cur = data[s];
    assert( data[s-1] >= cur );
    if(cur != prev)	{
      RE[prev] = pr-P;
      pr = P + RE[prev=cur];
    }
    assert( pr>P+R[cur] && pr<=P+i );
    suffix q = s+1;
    unsigned t = (cur<<1) + (data[q]>cur);
    if(D[t] != d)	{
      q |= FLAG_JUMP;
      D[t] = d;
    }
    *--pr = q | (t<<BIT_LMS);
  }while(i);
}

// the post-pass to figure out all non-LMS suffixes

void inducePost(byte *data, suffix *P, t_index *R, t_index *RE, t_index *R2, t_index K, t_index N)	{
  t_index i;
  byte prev; suffix *pr=NULL;
  assert(N);
  //left2right
  buckets(data, P, RE, R2, K, N);
  pr = P + R[prev=0];
  for(i=0; i!=N; ++i)	{
    const suffix s = P[i];
    P[i] = s ^ FLAG_LMS;
    if(s & FLAG_LMS)
      continue;
    assert(s && s<N);
    const byte cur = data[s];
    assert( data[s-1] <= cur );
    if(cur != prev)	{
      R[prev] = pr-P;
      pr = P + R[prev=cur];
    }
    assert( pr>P+i && pr<P+RE[cur] );
    const suffix q = s+1;
    *pr++ = q | (q==N || cur>data[q] ? FLAG_LMS:0);
  }
  //right2left
  buckets(data, P, RE, R2, K, N);
  pr = P + RE[prev=data[0]];
  *--pr = 1 | (prev<data[1] ? FLAG_LMS:0);
  i=N; do	{
    const suffix s = P[--i];
    if(s >= N)	{
      P[i] = s & ~FLAG_LMS;
      continue;
    }
    assert(s);
    const byte cur = data[s];
    assert( data[s-1] >= cur );
    if(cur != prev)	{
      RE[prev] = pr-P;
      pr = P + RE[prev=cur];
    }
    assert( pr>P+R[cur] && pr<=P+i );
    const suffix q = s+1;
    *--pr = q | (q!=N && cur<data[q] ? FLAG_LMS:0);
  }while(i);
}

// find the length of each LMS substring
// and write it into P[n1+(x>>1)]
// no collisions guaranteed because LMS distance>=2

struct XTargetLength	{
  suffix *const target;
  mutable t_index last;
  
  XTargetLength(suffix *const s1)
  : target(s1), last(0)	{}

  __inline void parse(t_index k, t_index i) const	{
    target[i>>1] = i-last;	//length
    last = i;
  }
};

void reduce(byte *data, suffix *P, t_index *R, t_index *RE, t_index *R2, t_index N, t_index K, t_index &n1, t_index &name) {
    // scatter LMS into bucket positions
    findLMS(data, P, R, RE, R2, N, K, n1);
    // sort by induction (evil technology!)
#   ifdef USE_TEMPLATE
    induce(data, P, R, RE, IPre(N), N, n1);
#   else
    inducePre(data, P, R, RE, R2, K, N);
#   endif
    // pack LMS indices
    packTargetIndices(P, n1, N);
    // fill in the lengths
    memset(P + n1, 0, (N - n1) * sizeof(suffix));
    XTargetLength TL = XTargetLength(P + n1);
    parseLMS(data, TL, n1, N);
    // compute values
    computeTargetValues(P, data, n1, name);
}

bool reduceFast(byte *data, suffix *P, t_index *R, t_index *RE, t_index *R2, t_index *D, t_index N, t_index K, t_index &n1, t_index &name) {
    findLMS(data, P, R, RE, R2, N, K, n1);
    // mark next-char borders
    assert(R2);
    t_index i = K - 1, top = N;
    for (;;) {
        if (RE[i] != top) {
            assert(RE[i] < top && P[RE[i]]);
            P[RE[i]] |= FLAG_JUMP;
        }
        if (!i)
            break;
        top = R2[--i];
    }
#   ifdef USE_TEMPLATE
    induce(data, P, R, RE, ITrack(N, D, K), N, n1);
#   else
    inducePreFast(data, P, R, RE, R2, D, N, K);
#   endif
    name = 0;
    // pack target indices
    for (top = i = 0;; ++i) {
        assert(i < N);
        suffix s = P[i];
        if (!(s & FLAG_LMS))
            continue;
        P[top] = (s ^= FLAG_LMS);
        name += s >> BIT_JUMP;
        assert(s && (s & ~FLAG_JUMP) < N);
        if (++top == n1)
            break;
    }
    // store names or unset flags
    if (name < n1) {
        suffix *const s1 = P + n1;
        memset(s1, 0, (N - n1) * sizeof(suffix));
        top = name + 1;
        i = n1;
        do {
            suffix suf = P[--i];
            top -= suf >> BIT_JUMP;
            suf &= ~FLAG_JUMP;
            s1[suf >> 1] = top;
        } while (i);
        return true;
    } else {
        for (i = 0; i != n1; ++i)
            P[i] &= ~FLAG_JUMP;
        return false;
    }
}

template<typename Q>
void solve(byte *data, suffix *P, t_index *R, t_index *RE, t_index *R2, t_index N, t_index K, t_index Nreserve, t_index &n1, t_index &d1, t_index &name) {
    Q *const input = reinterpret_cast<Q *>(P);
    packTargetValues(input, P, n1, d1, N);
    if (name < n1) {
        // count the new memory reserve
        t_index left = Nreserve;
        assert(n1 + d1 <= N);
        left += N - n1 - d1; // take into account new data and suffix storage
        if (R2) {
            assert(left >= K - 1);
            left -= K - 1;
        }
        // finally, solve the sub-problem
        constructSuffixArray(input, P + d1, n1, name, left);
    } else {
        // permute back from values into indices
        assert(name == n1);
        for (t_index i = n1; i--;)
            P[d1 + input[i]] = i + 1;
    }
}

struct XListBad	{
  suffix *const target;

  XListBad(suffix *const s1)
  : target(s1)	{}

  __inline void parse(t_index k, t_index i) const	{
    target[k] = i;
  }
};


struct XListGood : public XListBad	{
  t_index *const freq;
  const byte *const input;

  XListGood(suffix *const s1, t_index *const R, t_index K, const byte *const data)
  : XListBad(s1), freq(R), input(data-1)	{
    memset( freq, 0, K*sizeof(t_index) );
  }

  __inline void parse(t_index k, t_index i) const	{
    XListBad::parse(k,i);
    ++freq[input[i]];
  }
};

void derive(byte *data, suffix *P, t_index *R, t_index *RE, t_index *R2, t_index N, t_index K, t_index &n1, t_index &d1, bool need) {
    if (need) {
        memmove(P, P + d1, n1 * sizeof(suffix));
        // get the list of LMS strings into [n1,2*n1]
        // LMS number -> actual string number
        if (R2) {
          XListGood L = XListGood(P + n1, R, K, data);
          parseLMS(data, L, n1, N);
        } else {
          XListBad L = XListBad(P + n1);
          parseLMS(data, L, n1, N);
        }
        // update the indices in the sorted array
        // LMS t_index -> string t_index
        // todo: try to combine with the next pass
        suffix *const s2 = P + n1 - 1;
        for (t_index i = 0; i != n1; ++i) {
            assert(P[i] > 0 && P[i] <= (suffix)n1);
            P[i] = s2[P[i]];
        }
    }
    // scatter LMS back into proper positions
    if (R2 && need) {
        R2[-1] = 0; // either unoccupied or R[K], which we don't use
        t_index top = N, i = K;
        suffix *x = P + n1;
        while (i--) {
            t_index num = R[i];
            t_index bot = R2[(int)i - 1]; // arg -1 is OK here
            t_index space = top - bot - num;
            x -= num;
            memmove(P + top - num, x, num * sizeof(suffix));
            fillEmpty(P, bot, space, N);
            top = bot;
        }
        assert(x == P);
    } else {
        buckets(data, R, RE, R2, K, N);
        fillEmpty(P, n1, N - n1, N);
        byte prev_sym = K - 1;
        suffix *pr = P + RE[prev_sym];
        for (t_index i = n1; i--;) {
            suffix suf = P[i];
            P[i] = N | FLAG_LMS;
            assert(suf > 0 && suf <= (suffix)N && "Invalid suffix!");
            byte cur = data[suf - 1];
            if (cur != prev_sym) {
                assert(cur < prev_sym && "Not sorted!");
                pr = P + RE[prev_sym = cur];
            }
            *--pr = suf;
            assert(RE[cur] >= R[cur] && "Stepped twice on the same suffix!");
            assert(RE[cur] >= i && "Not sorted properly!");
        }
    }
    // induce the rest of suffixes
#   ifdef USE_TEMPLATE
    induce(data, P, R, RE, IPost(N), N);
#   else
    inducePost(data, P, R, RE, R2, K, N);
#   endif
}

void constructSuffixArray(byte *data, suffix *P, t_index N, t_index K, t_index reserved) {
    checkData(data, N, K);
    
    t_index *R = reinterpret_cast<t_index*>(P + N);
    t_index *RE = R + 1;
    t_index *R2 = reserved >= K * 2 ? R + reserved - K + 1 : NULL;
    t_index n1 = 0, d1 = 0, name = 0;
    
    if (sInduction) {
        t_index *D = R + K + 1;
        bool need = true;
        if (sTracking && R2 && D + K + K <= R2 && !(N >> 30))
            need = reduceFast(data, P, R, RE, R2, D, N, K, n1, name);
        else
            reduce(data, P, R, RE, R2, N, K, n1, name);
        
        if (need) {
            solve<byte>(data, P, R, RE, R2, N, K, reserved, n1, d1, name);
        }
        
        derive(data, P, R, RE, R2, N, K, n1, d1, need);
    } else {
        directSort(data, P, N);
    }
}