#pragma once
#include <vector>
#include <string>
#include <array>
#include <limits>
#include <numeric>
#include <iomanip>
#include <future>
#include <ranges>
#include <execution>
#include <omp.h>

#include <boost/core/noinit_adaptor.hpp>

#include "psais/utility/parallel.hpp"
#include "psais/utility/thread_pool.hpp"

// #pSAIS::detail
namespace psais::detail {

#define L_TYPE 0
#define S_TYPE 1
#define NUM_THREADS 32u
#define INDUCE_NUM_THREADS 16u

constexpr auto BLOCK_SIZE = 1u << 20;

constexpr inline auto mask = std::array<uint8_t, 8>{0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};

template <typename T>
using NoInitVector = std::vector<T, boost::noinit_adaptor<std::allocator<T>>>;

template <typename T>
constexpr auto EMPTY = std::numeric_limits<T>::max();

// #TypeVector
struct TypeVector {
	TypeVector(std::unsigned_integral auto size) : T(size / 8 + 1) {}

	bool get(auto idx) const {
		return T[idx >> 3] & mask[idx & 7];
	}

	void set(auto idx, bool val) {
		T[idx >> 3] = val
			? (mask[idx & 7] | T[idx >> 3])
			: ((~mask[idx & 7]) & T[idx >> 3]);
	}

	bool is_LMS(auto i) const {
		return i > 0 and get(i - 1) == L_TYPE and get(i) == S_TYPE;
	}

  private:
	NoInitVector<uint8_t> T;
	static constexpr inline auto mask = std::array<uint8_t, 8>{0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
};

// #name_substr
template<typename IndexType>
auto name_substr(
	const std::ranges::random_access_range auto &S,
	const TypeVector &T,
	const NoInitVector<IndexType> &SA,
	std::ranges::random_access_range auto S1,
	size_t kmer
) {
	auto is_same_substr = [&S, &T] (auto x, auto y, auto k) {
		do {
			k--;
			if (S[x++] != S[y++]) return false;
		} while (!T.is_LMS(x) and !T.is_LMS(y) and k);

		return k and T.is_LMS(x) and T.is_LMS(y) and S[x] == S[y];
	};

	IndexType n = (IndexType)S.size();
	auto SA1 = NoInitVector<IndexType>{};
	psais::utility::parallel_take_if(n, NUM_THREADS, SA1,
		[&](IndexType i) { return T.is_LMS(SA[i]); },
		[&](IndexType i) { return SA[i]; }
	);

	IndexType n1 = (IndexType)SA1.size();

	auto is_same = NoInitVector<IndexType>(n1);
	psais::utility::parallel_init(is_same, 0);

	{
		auto result = std::vector<std::future<void>>{};
		result.reserve(n1 / BLOCK_SIZE + 1);

		auto pool = psais::utility::ThreadPool(NUM_THREADS);
		for (IndexType x = 1; x < n1; x += BLOCK_SIZE) {
			IndexType L = x, R = std::min(n1, L + BLOCK_SIZE);
			result.push_back(
				pool.enqueue(
					[&](IndexType l, IndexType r) {
						for (IndexType i = l; i < r; i++)
							is_same[i] = not is_same_substr(SA1[i - 1], SA1[i], kmer);
					}, L, R
				)
			);
		}

		for (auto &f : result) {
			f.get();
		}
	}

	psais::utility::parallel_prefix_sum(is_same, NUM_THREADS);

	NoInitVector<IndexType> name(n);
	psais::utility::parallel_init(name, EMPTY<IndexType>);

	psais::utility::parallel_do(n1, NUM_THREADS,
		[&](IndexType L, IndexType R, IndexType) {
			for (IndexType i = L; i < R; i++)
				name[SA1[i]] = is_same[i];
		}
	);

	psais::utility::parallel_take_if(n, NUM_THREADS, S1,
		[&](IndexType i) { return name[i] != EMPTY<IndexType>; },
		[&](IndexType i) { return name[i]; }
	);

	return is_same.back();
}

// #induce_sort

// ##put_lms
template<typename IndexType>
auto put_lms(
	const std::ranges::random_access_range auto &S,
	const NoInitVector<IndexType> &LMS,
	const NoInitVector<IndexType> &SA1,
	const NoInitVector<IndexType> &BA,
	NoInitVector<IndexType> &SA
) {
	IndexType n1 = (IndexType)SA1.size();
	IndexType K = (IndexType)BA.size() - 1;

	NoInitVector<IndexType> S1(n1);
	std::transform(std::execution::par_unseq, SA1.begin(), SA1.end(), S1.begin(),
		[&LMS](auto &x) { return LMS[x]; }
	);

	NoInitVector<IndexType> local_BA(1ull * (K + 1) * NUM_THREADS);
	psais::utility::parallel_do(NUM_THREADS, NUM_THREADS,
		[&](IndexType, IndexType, IndexType tid) {
			IndexType *ptr = local_BA.data() + tid * (K + 1);
			for (IndexType i = 0; i < K + 1; i++)
				ptr[i] = 0;
		}
	);

	psais::utility::parallel_do(n1, NUM_THREADS,
		[&](IndexType L, IndexType R, IndexType tid) {
			IndexType *ptr = local_BA.data() + tid * (K + 1);
			for (IndexType i = L; i < R; i++) {
				IndexType idx = S1[i];
				ptr[S[idx]]++;
			}
		}
	);

	psais::utility::parallel_do(K + 1, NUM_THREADS,
		[&](IndexType L, IndexType R, IndexType) {
			for (IndexType i = NUM_THREADS - 2; ~i; i--) {
				auto *w_ptr = local_BA.data() + (i    ) * (K + 1);
				auto *r_ptr = local_BA.data() + (i + 1) * (K + 1);
				for (IndexType j = L; j < R; j++)
					w_ptr[j] += r_ptr[j];
			}
		}
	);

	psais::utility::parallel_do(n1, NUM_THREADS,
		[&](IndexType L, IndexType R, IndexType tid) {
			auto *ptr = local_BA.data() + tid * (K + 1);
			for (IndexType i = L; i < R; i++) {
				IndexType idx = S1[i];
				IndexType offset = ptr[S[idx]]--;
				SA[BA[S[idx]] - offset] = idx;
			}
		}
	);
}

// ##prepare
template<typename IndexType, typename CharType>
void prepare(
	const size_t L,
	const std::ranges::random_access_range auto &S,
	const NoInitVector<IndexType> &SA,
	const TypeVector &T,
	NoInitVector<std::pair<CharType, uint8_t>> &RB
) {
	if (L >= SA.size()) return;
	decltype(L) R = std::min(SA.size(), L + BLOCK_SIZE);

	#pragma omp parallel for num_threads(INDUCE_NUM_THREADS / 2)
	for (auto i = L; i < R; i++) {
		auto induced_idx = SA[i] - 1;

		if (SA[i] == EMPTY<IndexType> or SA[i] == 0) {
			RB[i - L] = {EMPTY<CharType>, 0};
		} else {
			RB[i - L] = {S[induced_idx], T.get(induced_idx)};
		}
	}
}

// ##update
template<typename IndexType>
void update(
	const size_t L,
	const NoInitVector<std::pair<IndexType, IndexType>> &WB,
	NoInitVector<IndexType> &SA
) {
	if (L >= SA.size()) return;
	decltype(L) R = std::min(SA.size(), L + BLOCK_SIZE);

	#pragma omp parallel for num_threads(INDUCE_NUM_THREADS / 2)
	for (auto i = L; i < R; i++) {
		auto& [idx, val] = WB[i - L];
		if (idx != EMPTY<IndexType>) {
			SA[idx] = val;
		}
	}
}

// ##induce_impl
template<auto InduceType, typename IndexType, typename CharType>
void induce_impl (
	const std::ranges::random_access_range auto &S,
	const TypeVector &T,
	const std::ranges::input_range auto &rng,
	const IndexType L,
	NoInitVector<IndexType> &SA,
	NoInitVector<std::pair<CharType, uint8_t>> &RB,
	NoInitVector<std::pair<IndexType, IndexType>> &WB,
	NoInitVector<IndexType> &ptr
) {
	for (IndexType i : rng) {
		auto induced_idx = SA[i] - 1;

		if (SA[i] != EMPTY<IndexType> and SA[i] != 0) {
			auto chr = EMPTY<CharType>;
			if (auto [c, t] = RB[i - L]; c != EMPTY<CharType>) {
				if (t == InduceType) chr = c;
			} else if (T.get(induced_idx) == InduceType) {
				chr = S[induced_idx];
			}

			if (chr == EMPTY<CharType>) continue;

			bool is_adjacent;
			auto pos = ptr[chr];
			if constexpr (InduceType == L_TYPE) {
				ptr[chr] += 1;
				is_adjacent = pos < L + (BLOCK_SIZE << 1);
			} else {
				ptr[chr] -= 1;
				is_adjacent = pos + BLOCK_SIZE >= L;
			}

			// if pos is in adjacent block -> directly write it
			// otherwise, write it to WB
			if (is_adjacent) {
				SA[pos] = induced_idx;
				WB[i - L].first = EMPTY<IndexType>;
			} else {
				WB[i - L] = {pos, induced_idx};
			}
		}
	}
}

// ##induce
template<auto InduceType, typename IndexType, typename CharType>
void induce (
	const std::ranges::random_access_range auto &S,
	const TypeVector &T,
	NoInitVector<IndexType> &SA,
	NoInitVector<std::pair<CharType, uint8_t>> &RBP,
	NoInitVector<std::pair<CharType, uint8_t>> &RBI,
	NoInitVector<std::pair<IndexType, IndexType>> &WBU,
	NoInitVector<std::pair<IndexType, IndexType>> &WBI,
	NoInitVector<IndexType> &ptr
) {
	// views
	constexpr auto iter_view = [] {
		if constexpr (InduceType == L_TYPE) {
			return std::views::all;
		} else {
			return std::views::reverse;
		}
	}();

	IndexType size = SA.size();
	auto blocks = std::views::iota(IndexType(0), size)
		| std::views::filter([](IndexType n) { return n % BLOCK_SIZE == 0; });

	// prepare for first block
	if constexpr (InduceType == L_TYPE) {
		prepare(0, S, SA, T, RBP);
	} else {
		prepare(size / BLOCK_SIZE * BLOCK_SIZE, S, SA, T, RBP);
	}

	auto pool = psais::utility::ThreadPool(2);
	auto stage = std::array<std::future<void>, 2>{};

	// pipeline
	for (IndexType L : blocks | iter_view) {
		for (auto &s : stage) if (s.valid()) s.wait();
		RBI.swap(RBP);
		WBI.swap(WBU);

		// prepare && update
		IndexType P_L = L + BLOCK_SIZE;
		IndexType U_L = L - BLOCK_SIZE;
		if constexpr (InduceType == S_TYPE) {
			std::swap(P_L, U_L);
		}

		stage[0] = pool.enqueue(prepare<IndexType, CharType, decltype(S)>, P_L,
				std::ref(S), std::ref(SA), std::ref(T), std::ref(RBP));
		stage[1] = pool.enqueue(update<IndexType>, U_L,
				std::ref(WBU), std::ref(SA));

		// induce
		auto rng = std::views::iota(L, std::min(L + BLOCK_SIZE, size)) | iter_view;
		induce_impl<InduceType>(S, T, rng, L, SA, RBI, WBI, ptr);
	}
}

// ##induce_sort
template<typename IndexType>
void induce_sort(
	const std::ranges::random_access_range auto &S,
	const TypeVector &T,
	const NoInitVector<IndexType> &SA1,
	const NoInitVector<IndexType> &LMS,
	NoInitVector<IndexType> &BA,
	NoInitVector<IndexType> &SA
) {
	using CharType = decltype(S.begin())::value_type;

	// induce LMS
	put_lms(S, LMS, SA1, BA, SA);

	// declare ptr, RBP, RBI, WBI, WBU
	NoInitVector<IndexType> ptr(BA.size());
	NoInitVector<std::pair<CharType, uint8_t>> RBP(BLOCK_SIZE), RBI(BLOCK_SIZE);
	NoInitVector<std::pair<IndexType, IndexType>> WBU(BLOCK_SIZE), WBI(BLOCK_SIZE);

	// init buffer
	psais::utility::parallel_init(RBP, std::pair{EMPTY<CharType>, uint8_t(0)});
	psais::utility::parallel_init(RBI, std::pair{EMPTY<CharType>, uint8_t(0)});
	psais::utility::parallel_init(WBU, std::pair{EMPTY<IndexType>, EMPTY<IndexType>});
	psais::utility::parallel_init(WBI, std::pair{EMPTY<IndexType>, EMPTY<IndexType>});

	// induce L
	ptr[0] = 0;
	std::transform(std::execution::par_unseq, BA.begin(), BA.end() - 1, ptr.begin() + 1,
		[](IndexType &b) { return b; }
	);
	induce<L_TYPE>(S, T, SA, RBP, RBI, WBU, WBI, ptr);

	// init buffer
	psais::utility::parallel_init(RBP, std::pair{EMPTY<CharType>, uint8_t(0)});
	psais::utility::parallel_init(RBI, std::pair{EMPTY<CharType>, uint8_t(0)});
	psais::utility::parallel_init(WBU, std::pair{EMPTY<IndexType>, EMPTY<IndexType>});
	psais::utility::parallel_init(WBI, std::pair{EMPTY<IndexType>, EMPTY<IndexType>});

	// clean S_TYPE
	std::for_each(std::execution::par_unseq, SA.begin() + 1, SA.end(),
		[&T](IndexType &idx) {
			if (idx != EMPTY<IndexType> and T.get(idx) == S_TYPE) {
				idx = EMPTY<IndexType>;
			}
		}
	);

	// induce S
	std::transform(std::execution::par_unseq, BA.begin(), BA.end(), ptr.begin(),
		[](auto &bucket) { return bucket - 1; }
	);
	induce<S_TYPE>(S, T, SA, RBP, RBI, WBU, WBI, ptr);
}

// #preprocess

// ##get_type
template<typename IndexType>
auto get_type(const std::ranges::random_access_range auto &S) {
	auto T = TypeVector(S.size());
	std::vector<IndexType> same_char_suffix_len(NUM_THREADS, 0);
	std::vector<IndexType> block_size(NUM_THREADS, 0);
	std::vector<IndexType> block_left(NUM_THREADS, 0);

	T.set(S.size() - 1, S_TYPE);
	IndexType rest = S.size() % 8;
	IndexType n = S.size() / 8 * 8;

	auto cal_type = [&](auto x) -> bool {
		auto x1 = S[x], x2 = S[x + 1];
		if (x1 < x2)
			return S_TYPE;
		else if (x1 > x2)
			return L_TYPE;
		else
			return T.get(x + 1);
	};

	if (rest != 0) {
		for (IndexType i = rest - 2; ~i; i--) {
			T.set(n + i, cal_type(n + i));
		}

		if (n != 0)
			T.set(n - 1, cal_type(n - 1));
	}

	psais::utility::parallel_do(n, NUM_THREADS,
		[&](IndexType L, IndexType R, IndexType tid) {
			if (L == R)
				return ;

			if (R != n)
				T.set(R - 1, cal_type(R - 1));

			same_char_suffix_len[tid] = 1;
			bool same = true;
			for (IndexType i = R - L - 2; ~i; i--) {
				IndexType x = L + i;
				T.set(x, cal_type(x));

				if (S[x] != S[x + 1])
					same = false;

				if (same)
					same_char_suffix_len[tid]++;
			}

			block_size[tid] = R - L;
			block_left[tid] = L;
		}
	);

	std::vector<uint8_t> flip(NUM_THREADS, false);
	for (IndexType i = NUM_THREADS - 2; ~i; i--) {
		if (block_size[i + 1] == 0)
			continue;

		IndexType x1 = block_left[i + 1] - 1;
		IndexType x2 = block_left[i + 1];
		// ...-|----|----|-...
		//        x1 x2

		if (S[x1] != S[x2])
			continue;

		uint8_t prev_left_type = T.get(x2);
		if (same_char_suffix_len[i + 1] == block_size[i + 1] and flip[i + 1])
			prev_left_type ^= 1;

		if (T.get(x1) != prev_left_type)
			flip[i] = true;
	}

	psais::utility::parallel_do(n, NUM_THREADS,
		[&](IndexType L, IndexType R, IndexType tid) {
			if (not flip[tid])
				return ;

			T.set(R - 1, !T.get(R - 1));
			for (IndexType i = R - L - 2; ~i; i--) {
				IndexType x = L + i;
				if (S[x] != S[x + 1])
					return ;
				T.set(x, !T.get(x));
			}
		}
	);

	return T;
}

// ##get_bucket
template<typename IndexType>
auto get_bucket(const std::ranges::random_access_range auto &S, IndexType K) {
	NoInitVector<IndexType> local_BA(1ull * (K + 1) * NUM_THREADS);
	psais::utility::parallel_init(local_BA, 0);

	IndexType n = S.size();
	psais::utility::parallel_do(n, NUM_THREADS,
		[&](IndexType L, IndexType R, IndexType tid) {
			IndexType *ptr = local_BA.data() + tid * (K + 1);
			for (IndexType i = L; i < R; i++)
				ptr[S[i]]++;
		}
	);

	auto BA = NoInitVector<IndexType>(K + 1);
	psais::utility::parallel_init(BA, 0);

	psais::utility::parallel_do(K + 1, NUM_THREADS,
		[&](IndexType L, IndexType R, IndexType) {
			for (IndexType i = 0; i < NUM_THREADS; i++) {
				IndexType *ptr = local_BA.data() + i * (K + 1);
				for (IndexType j = L; j < R; j++)
					BA[j] += ptr[j];
			}
		}
	);

	psais::utility::parallel_prefix_sum(BA, NUM_THREADS);

	return BA;
}

// ##get_lms
template<typename IndexType>
auto get_lms(const TypeVector &T, const auto size) {
	auto LMS = NoInitVector<IndexType>{};
	psais::utility::parallel_take_if(size, NUM_THREADS, LMS,
		[&](IndexType i) { return T.is_LMS(i); },
		[ ](IndexType i) { return i; }
	);
	return LMS;
}

// #suffix_array
template<typename IndexType>
NoInitVector<IndexType> suffix_array(
	const std::ranges::random_access_range auto &S,
	IndexType K,
	size_t kmer
) {
	// 1. get type && bucket array
	auto T = get_type<IndexType>(S);
	auto BA = get_bucket(S, K);

	// 2. induce LMS-substring
	auto LMS = get_lms<IndexType>(T, S.size());

	auto SA = NoInitVector<IndexType>(S.size());
	psais::utility::parallel_init(SA, EMPTY<IndexType>);

	auto SA1 = NoInitVector<IndexType>(LMS.size());

	// iota SA1
	auto iota = std::views::iota(IndexType(0), static_cast<IndexType>(SA1.size()));
	std::transform(std::execution::par_unseq, iota.begin(), iota.end(), SA1.begin(),
		[](auto &idx) { return idx; }
	);

	induce_sort(S, T, SA1, LMS, BA, SA);

	auto S1 = std::ranges::subrange(SA.begin() + SA.size() - SA1.size(), SA.end());
	auto K1 = name_substr(S, T, SA, S1, kmer);

	// 3. recursively solve LMS-suffix
	if (K1 + 1 == LMS.size()) {
		for (size_t i = 0; i < LMS.size(); i++) {
			SA1[S1[i]] = i;
		}
	} else {
		SA1 = suffix_array(S1, K1, kmer >> 1);
	}

	// 4. induce orig SA
	psais::utility::parallel_init(SA, EMPTY<IndexType>);
	induce_sort(S, T, SA1, LMS, BA, SA);

	return SA;
}

#undef L_TYPE
#undef S_TYPE
#undef NUM_THREADS
#undef INDUCE_NUM_THREADS

} // namespace psais::detail

// #pSAIS
namespace psais {

template <typename IndexType>
auto suffix_array(std::string_view s, size_t kmer = std::string::npos) {
	IndexType K = 0;
	auto idx = std::array<IndexType, 128>{};
	for (auto c : s) idx[c] = 1;
	for (auto &x : idx) if(x) x = ++K;

	auto res = psais::detail::NoInitVector<uint8_t>(s.size() + 1);
	std::transform(std::execution::par_unseq, s.begin(), s.end(), res.begin(),
		[&idx](auto &c) { return idx[c]; }
	);
	res[s.size()] = 0;

	return psais::detail::suffix_array(res, K, kmer);
}

} //namespace psais
