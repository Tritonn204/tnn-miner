#include <assert.h>
#include <string.h>
#include <stdio.h>

#include "archon.h"

//--------------------------------------------------------------------------//
// This SAC implementation is based on SAIS algorithm explained here:
// http://www.cs.sysu.edu.cn/nong/t_index.files/Two%20Efficient%20Algorithms%20for%20Linear%20Suffix%20Array%20Construction.pdf
// It also uses some nice optimizations found in Yuta Mori version:
// https://sites.google.com/site/yuta256/sais
//--------------------------------------------------------------------------//
static const bool	sInduction	= true;
static const bool	sTracking	= true;


template<typename T>
class	Constructor	{
	const T	*const data;	// input string
	suffix	*const P;		// suffix array
	t_index	*const R,		// radix start
			*const RE,		// radix end
			*const R2;		// radix backup
	const	t_index N;		// input length
	const	t_index K;		// number of unique values
	t_index	n1;				// number of LMS suffixes
	t_index	d1;				// memory units occupied by the new data
	t_index	name;			// new number of unique values

	enum	{
		BIT_LMS		= 31,
		FLAG_LMS	= 1<<BIT_LMS,
		BIT_JUMP	= 30,
		FLAG_JUMP	= 1<<BIT_JUMP,
	};

	//---------------------------------
	//	Strategy-0 implementation	//

	void directSort()	{
		for(t_index i=0; i!=N; ++i)
			P[i] = i+1;
		ray(P,N,1);
	}

	void ray(suffix *A, t_index num, unsigned depth)	{
		while(num>1)	{
			suffix *x=A, *z=A+num, s=A[num>>1];
			if(s<depth)	{
				assert(s+1==depth);
				if(num==2)
					return;
				const suffix t = A[num>>1] = A[0];
				*--z=s; s=t; --num;
			}
			assert(s>=depth);
			const T w = data[s-depth];
			suffix *y = x;
			for(;;)	{
				s = *y;
				if(s>=depth)	{
					const T q = data[s-depth];
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
			ray(A,x-A,depth); A = x+num;
			ray(A,z-A,depth); A = x;
			++depth;
		}
	}

	bool sufCompare(const suffix a, const suffix b)	{
		assert(a!=b);
		t_index d=0; do ++d;
		while(a>=d && b>=d && data[a-d]==data[b-d]);
		return a>=d && (b<d || data[a-d]<data[b-d]);
	}

	bool bruteCheck(suffix *const A, suffix *const B)	{
		for(suffix *x=A; ++x<B; )
			assert(sufCompare( x[-1], x[0] ));
		return true;
	}

	//---------------------------------
	//	Commonly used routines	//

	void checkData()	{
		if(sizeof(T) == sizeof(t_index))
			return;
		byte shift = sizeof(T)<<3;
		assert(!( (K-1)>>shift ));
		if(K>>shift)
			return;
		for(t_index i=0; i<N; ++i)
			assert(data[i]>=0 && data[i]<K);
	}

	bool checkUnique(const t_index num)	{
		for(t_index i=0; i!=num; ++i)
			for(t_index j=i+1; j!=num; ++j)
				if(P[i]==P[j])
					return false;
		return true;
	}

	void makeBuckets()	{
		memset( R, 0, K*sizeof(t_index) );
		t_index i,sum;
		for(i=0; i<N; ++i)
			++R[data[i]];
		for(R[i=K]=sum=N; i--;)
			sum = R[i] = sum-R[i];	// cf
		assert(!sum);
	}

	void buckets()	{
		if(R2)	{
			memcpy( RE, R2, (K-1U)*sizeof(t_index) );
			R[0] = 0; R[K] = N;
		}else
			makeBuckets();
	}

	void fillEmpty(t_index off, t_index num)	{
		//memset( P+off, 0, num*sizeof(suffix) );
		t_index i = off; off+=num;
		while(i!=off)
			P[i++] = N|FLAG_LMS;
	}

	template<class X>
	void parseLMS(const X &x)	{
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

	//---------------------------------
	//	Intermediate routines	//

	void findLMS()	{
		assert(!n1 && N);
		fillEmpty(0,N);
		buckets();
		for(t_index i=0; ; )	{
			do if(++i>=N)
				return;
			while(data[i-1] >= data[i]);
			++n1;	//found LMS!
			P[--RE[data[i-1]]] = i;
			while(++i<N && data[i-1] <= data[i]);
		}
	}

	void packTargetIndices()	{
		// pack LMS into the first n1 suffixes
		t_index i=-1,j=0;
		while(j!=n1)	{
			suffix s;
			while(++i, assert(i<N), (s=P[i]^FLAG_LMS) & FLAG_LMS);
			P[j++] = s;
		}
	}

	void computeTargetValues()	{
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

	//---------------------------------
	//	Induction implementation	//

	// This section contains a try to share the code by using templated induce() method
	// Unfortunately, it works 4% slower by unknown reasons so far
	// Can be enabled with USE_TEMPLATE definition

	struct IPre	{
		const t_index n1;
		IPre(const suffix n)
		: n1(n-1)	{}
		// elem check
		inline suffix getUp(suffix &orig) const	{
			const suffix s = orig;
			if(s >= n1)
				return 0;
			orig = n1;
			assert(s>0 && s<n1);
			return s;
		}
		inline suffix getDown(const suffix orig) const	{
			if(orig >= n1)
				return 0;
			assert(orig>0 && orig<n1);
			return orig;
		}
		// elem mod
		inline suffix flagUp	(const suffix q, const T cur, const T prev) const	{
			return (cur>prev ? q|FLAG_LMS : q);
		}
		inline suffix flagDown	(const suffix q, const T cur, const T prev) const	{
			return (cur<prev ? q|FLAG_LMS : q);
		}
		void middle(suffix *const P) const	{}
	};


	struct ITrack	{
		const t_index n1;
		t_index *const mask;
		mutable t_index d;
		// construct
		ITrack(const t_index N, t_index *const D, const suffix K)
		: n1(N-1U), mask(D), d(0)	{
			assert( BIT_LMS+1 == (sizeof(suffix)<<3) );
			memset( D, 0, 2*K*sizeof(t_index) );
		}
		// elem check
		inline suffix get(const suffix x) const	{
			d += x>>BIT_JUMP;
			return x & ~FLAG_JUMP;
		}
		inline suffix getUp(suffix &orig) const	{
			const suffix s = orig;
			if((s&FLAG_LMS) || (s&~FLAG_JUMP)==n1)	{
				orig = s & ~FLAG_LMS;
				return 0;
			}
			orig = n1+1;
			return get(s);
		}
		inline suffix getDown(const suffix orig) const	{
			if((orig&~FLAG_JUMP) >= n1)
				return 0;
			return get(orig);
		}
		// elem mod
		inline suffix flag(const unsigned t) const	{
			suffix rez = t<<BIT_LMS;
			if(mask[t] != d)	{
				rez |= FLAG_JUMP;
				mask[t] = d;
			}
			return rez;
		}
		inline suffix flagUp	(const suffix q, const T cur, const T& prev) const	{
			return q | flag( (cur<<1) + (prev<cur) );
		}
		inline suffix flagDown	(const suffix q, const T cur, const T& prev) const	{
			return q | flag( (cur<<1) + (prev>cur) );
		}
		// middle
		void middle(suffix *const P) const	{
			++d; //reverse flags order
			t_index i=n1+1;
			do	{
				const suffix s = P[--i];
				if(s > n1)
					continue;
				assert(s>0 && s<=n1);
				P[i] |= FLAG_JUMP;
				while( assert(i>0), !(P[--i]&FLAG_JUMP) );
				P[i] ^= FLAG_JUMP;
			}while(i);
		}
	};


	struct IPost	{
		const t_index N;
		IPost(const suffix n)
		: N(n)	{}
		// elem check
		inline suffix getUp(suffix &orig) const	{
			const suffix s = orig;
			orig = s ^ FLAG_LMS;
			if(s & FLAG_LMS)
				return 0;
			assert(s && s<N);
			return s;
		}
		inline suffix getDown(suffix &orig) const	{
			const suffix s = orig;
			if(s >= N)	{
				orig = s & ~FLAG_LMS;
				return 0;
			}
			return s;
		}
		// elem mod
		inline suffix flagUp	(const suffix q, const T cur, const T prev) const	{
			return (q==N || cur>prev ? q|FLAG_LMS : q);
		}
		inline suffix flagDown	(const suffix q, const T cur, const T prev) const	{
			return (q!=N && cur<prev ? q|FLAG_LMS : q);
		}
		void middle(suffix *const P) const	{}
	};


	template<class I>
	void induce(const I &in)	{
		t_index i; T prev;
		suffix *pr = NULL;
		assert(N);
		//go up
		buckets();
		pr = P + R[prev=0];
		for(i=0; i!=N; ++i)	{
			const suffix s = in.getUp(P[i]);
			if(!s)
				continue;
			const T cur = data[s];
			assert( data[s-1] <= cur );
			if(cur != prev)	{
				R[prev] = pr-P;
				pr = P + R[prev=cur];
			}
			assert( pr>P+i && pr<P+RE[cur] );
			const suffix q = s+1;
			*pr++ = in.flagUp(q,cur,data[q]);
		}
		//middle
		in.middle(P);
		//go down
		buckets();
		pr = P + RE[prev=data[0]];
		*--pr = in.flagDown(1,prev,data[1]);
		i=N+1; do	{--i;
			const suffix s = in.getDown(P[i]);
			if(!s)
				continue;
			const T cur = data[s];
			assert( data[s-1] >= cur );
			if(cur != prev)	{
				RE[prev] = pr-P;
				pr = P + RE[prev=cur];
			}
			assert( pr>P+R[cur] && pr<=P+i );
			const suffix q = s+1;
			*--pr = in.flagDown(q,cur,data[q]);
		}while(i);
	};

	
	// This section contains a real induction code in 3 different variants
	// the pre-pass to sort LMS
	//todo: use buckets to traverse the SA efficiently
	// if R2 is available

	void inducePre()	{
		// we are not interested in s>=N-1 here so we skip it
		t_index i;
		T prev; suffix *pr=NULL;
		assert(N);
		//left2right
		buckets();
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
			const T cur = data[s];
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
		buckets();
		pr = P + RE[prev=data[0]];
		*--pr = 1 + (prev<data[1] ? FLAG_LMS:0);
		i=N; do	{
			const suffix s = P[--i];
			if(s >= N-1)
				continue;
			assert(s>0 && s<N-1);
			//P[i] = N;
			const T cur = data[s];
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

	void inducePreFast(t_index *const D)	{
		assert( BIT_LMS+1 == (sizeof(suffix)<<3) );
		t_index i;
		T prev; suffix *pr=NULL;
		assert(N);
		memset( D, 0, 2*K*sizeof(t_index) );
		//left2right
		unsigned d=0;
		buckets();
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
			const T cur = data[s];
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
		buckets();
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
			const T cur = data[s];
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

	void inducePost()	{
		t_index i;
		T prev; suffix *pr=NULL;
		assert(N);
		//left2right
		buckets();
		pr = P + R[prev=0];
		for(i=0; i!=N; ++i)	{
			const suffix s = P[i];
			P[i] = s ^ FLAG_LMS;
			if(s & FLAG_LMS)
				continue;
			assert(s && s<N);
			const T cur = data[s];
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
		buckets();
		pr = P + RE[prev=data[0]];
		*--pr = 1 | (prev<data[1] ? FLAG_LMS:0);
		i=N; do	{
			const suffix s = P[--i];
			if(s >= N)	{
				P[i] = s & ~FLAG_LMS;
				continue;
			}
			assert(s);
			const T cur = data[s];
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

	void reduce()	{
		// scatter LMS into bucket positions
		findLMS();
		// sort by induction (evil technology!)
#		ifdef USE_TEMPLATE
		induce(IPre(N));
#		else
		inducePre();
#		endif
		// pack LMS indices
		packTargetIndices();
		// fill in the lengths
		memset( P+n1, 0, (N-n1)*sizeof(suffix) );
		parseLMS( XTargetLength(P+n1) );
		// compute values
		computeTargetValues();
	}

	bool reduceFast(t_index *const D)	{
		findLMS();
		// mark next-char borders
		assert(R2);
		t_index i=K-1, top=N;
		for(;;)	{
			if(RE[i]!=top)	{
				assert( RE[i]<top && P[RE[i]] );
				P[RE[i]] |= FLAG_JUMP;
			}
			if(!i)
				break;
			top = R2[--i];
		}
#		ifdef USE_TEMPLATE
		induce( ITrack(N,D,K) );
#		else
		inducePreFast(D);
#		endif
		name = 0;
		// pack target indices
		for(top=i=0; ;++i)	{
			assert(i<N);
			suffix s = P[i];
			if(!(s&FLAG_LMS))
				continue;
			P[top] = (s ^= FLAG_LMS);
			name += s>>BIT_JUMP;
			assert( s && (s&~FLAG_JUMP)<N );
			if(++top==n1)
				break;
		}
		// store names or unset flags
		if(name < n1)	{
			suffix *const s1 = P+n1;
			memset( s1, 0, (N-n1)*sizeof(suffix) );
			top = name+1;
			i=n1; do	{
				suffix suf = P[--i];
				top -= suf>>BIT_JUMP;
				suf &= ~FLAG_JUMP;
				s1[suf>>1] = top;
			}while(i);
			return true;
		}else	{
			d1 = n1;
			for(i=0; i!=n1; ++i)
				P[i] &= ~FLAG_JUMP;
			return false;
		}
	}

	template<typename Q>
	void packTargetValues(Q *const input)	{
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

	template<typename Q>
	void solve(const t_index reserve)	{
		Q *const input = reinterpret_cast<Q*>(P);
		packTargetValues(input);
		if(name<n1)	{
			// count the new memory reserve
			t_index left = reserve;
			assert( n1+d1 <= N );
			left += N-n1-d1; // take in account new data and suffix storage
			if(R2)	{
				assert( left >= K-1 );
				left -= K-1;
			}
			// finally, solve the sub-problem
			Constructor<Q>( input, P+d1, n1, name, left );
		}else	{
			// permute back from values into indices
			assert(name == n1);
			for(t_index i=n1; i--; )
				P[d1+input[i]] = i+1;
		}
	}

	//	Derivation helper classes

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
		const T *const input;

		XListGood(suffix *const s1, t_index *const R, t_index K, const T *const data)
		: XListBad(s1), freq(R), input(data-1)	{
			memset( freq, 0, K*sizeof(t_index) );
		}

		__inline void parse(t_index k, t_index i) const	{
			XListBad::parse(k,i);
			++freq[input[i]];
		}
	};


	void derive(bool sorted)	{
		if(sorted)	{
			memmove( P, P+d1, n1*sizeof(suffix) );
			// get the list of LMS strings into [n1,2*n1]
			// LMS number -> actual string number
			if(R2)
				parseLMS( XListGood(P+n1,R,K,data) );
			else
				parseLMS( XListBad(P+n1) );
			// update the indices in the sorted array
			// LMS t_index -> string t_index
			//todo: try to combine with the next pass
			suffix *const s2 = P+n1-1;
			for(t_index i=0; i!=n1; ++i)	{
				assert( P[i]>0 && P[i]<=(suffix)n1 );
				P[i] = s2[P[i]];
			}
		}
		// scatter LMS back into proper positions
		if(R2 && sorted)	{
			R2[-1] = 0;	// either unoccupied or R[K], which we don't use
			t_index top = N, i=K;
			suffix *x = P+n1;
			while(i--)	{
				const t_index num = R[i];
				const t_index bot = R2[(int)i-1];		// arg -1 is OK here
				const t_index space = top-bot-num;
				x -= num;
				memmove( P+top-num, x, num	*sizeof(suffix) );
				fillEmpty(bot,space);
				top = bot;
			}
			assert( x==P );
		}else	{
			buckets();
			fillEmpty(n1,N-n1);
			T prev_sym = K-1;
			suffix *pr = P + RE[prev_sym];
			for(t_index i=n1; i--; )	{
				const suffix suf = P[i];
				P[i] = N | FLAG_LMS;
				assert(suf>0 && suf<=(suffix)N	&& "Invalid suffix!");
				const T cur = data[suf-1];
				if(cur != prev_sym)	{
					assert(cur<prev_sym			&& "Not sorted!");
					pr = P + RE[prev_sym=cur];
				}
				*--pr = suf;
				assert(RE[cur] >= R[cur] 	&& "Stepped twice on the same suffix!");
				assert(RE[cur] >= i			&& "Not sorted properly!");
			}
		}
		// induce the rest of suffixes
#		ifdef USE_TEMPLATE
		induce( IPost(N) );
#		else
		inducePost();
#		endif
	}

	//---------------------------------
	//	Main entry point	//

public:
	Constructor(const T *const _data, suffix *const _P, const t_index _N, const t_index _K, const t_index reserved)
	: data(_data), P(_P), R(reinterpret_cast<t_index*>(_P+_N)), RE(R+1)
	, R2(reserved>=_K*2 ? R+reserved-_K+1 : NULL)
	, N(_N), K(_K), n1(0), d1(0), name(0)	{
		assert( N>0 && K>0 && K<reserved );
		checkData();
		if(R2)	{
			assert(R2 >= R+K+1);
			makeBuckets();
			memcpy( R2, RE, (K-1)*sizeof(t_index) );
		}
		if(!sInduction)	{
			directSort();
			return;
		}
		// reduce the problem to LMS sorting
		t_index *const D = R+K+1;
		bool need = true;
		if(sTracking && R2 && D+K+K<=R2 && !(N>>30))
			need = reduceFast(D);
		else
			reduce();
		// solve the reduced problem
		if(need)	{
#		ifndef NO_SQUEEZE
			if(name<=0x100)
				solve<byte>(reserved);
			else if(name<=0x10000)
				solve<dbyte>(reserved);
			else
#		endif
				solve<unsigned>(reserved);
		}
		// derive all other suffixes
		derive(need);
	}
};


//--------------------------------------------------------//

//	INITIALIZATION	//

t_index Archon::estimateReserve(const t_index n)	{
	t_index total = 0x10000;
#	ifndef NO_SQUEEZE
	// R(0x101) + R2(0xFF) + D(0x200)
	const int add = 0x400;
	if(!(n>>17))
		total = n/4+1;
	if(total<add)
		total = add;
#	endif
	return total;
}

Archon::Archon(const t_index Nx)
: Nmax(Nx)
, Nreserve(estimateReserve(Nx))
, P(new suffix[Nmax+Nreserve])
, str(new byte[Nmax+1])
, N(0), baseId(0) {
	assert(P && str);
	assert(sizeof(t_index) == sizeof(suffix));
}

Archon::~Archon()	{
	delete[] P;
	delete[] str;
}

unsigned Archon::countMemory() const	{
	return Nmax + (Nmax+Nreserve)*sizeof(suffix);
}


//	ENCODING	//

bool Archon::validate()	{
	t_index *const R = reinterpret_cast<t_index*>(P+N);
	memset( R,0, 0x100*sizeof(t_index) );
	t_index i = N;
	while(i--)
		R[str[P[i]-1]]=i;
	for(i=0; i!=N; ++i)	{
		const suffix s = P[i];
		if(s!=N && P[R[str[s]]++] != 1+s)
			return false;
	}
	return true;
}

int Archon::enRead(FILE *const fx, t_index ns)	{
	assert(ns>=0 && ns<=Nmax);
	N = fread( str, 1, ns, fx );
	return N;
}

int Archon::enCompute()	{
	Constructor<byte>( str, P, N, 0x100, Nmax-N+Nreserve );
	return 0;
}

int Archon::enWrite(FILE *const fx)	{
	baseId = N;
	for(t_index i=0; i!=N; ++i)	{
		suffix pos = P[i];
		if(pos == N)	{
			baseId = i;
			pos = 0;
		}
		fputc( str[pos], fx);
	}
	assert(baseId != N);
	fwrite( &baseId, sizeof(t_index), 1, fx);
	return 0;
}


//	DECODING	//

void Archon::roll(const t_index i)	{
	P[i] = P[N+1+str[i]]++;
	assert(N==1 || i != P[i]);
}

int Archon::deRead(FILE *const fx, t_index ns)	{
	enRead(fx,ns);
	int ok = fread( &baseId, sizeof(t_index), 1, fx );
	assert(ok && baseId<N);
	return N;
}

int Archon::deCompute()	{
	t_index i,k;
	// compute buchet heads
	t_index *const R = reinterpret_cast<t_index*>(P+N+1);
	assert( N+0x101 <= Nmax+Nreserve );
	memset( R, 0, 0x100*sizeof(t_index) );
	for(i=0; i!=N; ++i)
		++R[str[i]];
	for(k=N,i=0x100; i--;)
		R[i] = (k-=R[i]);
	// fill the jump table
#	ifndef	NDEBUG
	memset( P, 0, (N+1)*sizeof(t_index) );
#	endif
	for(i=0; i<baseId; i++)		roll(i);
	for(i=baseId+1U; i<N; i++)	roll(i);
	roll(baseId);
	return 0;
}

int Archon::deWrite(FILE *const fx)	{
	t_index i, k=baseId;
	for(i=0; i!=N; ++i,k=P[k])
		putc(str[k],fx);
	assert( k==baseId );
	return 0;
}