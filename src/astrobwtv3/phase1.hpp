#pragma once

#include "fgsaca_common.hpp"
#include "fgsaca_util.hpp"

namespace fgsaca_internal {

enum alg_t {
	SACA_t = 0,
	BBWT_t = 1,
	EBWT_t = 2
};

constexpr size_t phase_1_prefetch_distance = 8;
	
template <typename Ix, typename mem_ix_t>
FGSACA_INLINE Ix
phase_1_prefetch(
	const mem_ix_t* FGSACA_RESTRICT ISA,
	const mem_ix_t* FGSACA_RESTRICT pss,
	const Ix* FGSACA_RESTRICT A,
	const size_t gstart,
	size_t dat)
{
	while (fgsaca_likely(dat > 0) && dat + phase_1_prefetch_distance >= gstart && is_marked<Ix, Ix>(A[dat])) {
		fgsaca_prefetch(&pss[unmark(A[dat])]);
		fgsaca_prefetch(&ISA[unmark(A[dat])]);
		dat--;
	}
	return dat;
}

template <bool use_prefetching, typename Ix>
FGSACA_INLINE void
reduce_old_group_size(
	Ix* FGSACA_RESTRICT SA,
	Ix* FGSACA_RESTRICT isas,
	const size_t cnt)
{
	if constexpr (use_prefetching) {
		constexpr ptrdiff_t prefetch_distance = 16;
		ptrdiff_t i, j;
		for (i = 0; i < prefetch_distance; i++) {
			fgsaca_prefetchw(&SA[isas[i]]);
		}
		for (i = 0, j = cnt - prefetch_distance - 3; i < j; i += 4) {
			fgsaca_prefetch(&isas[i + 2 * prefetch_distance]);

			fgsaca_prefetchw(&SA[isas[i + 0 + prefetch_distance]]);
			fgsaca_prefetchw(&SA[isas[i + 1 + prefetch_distance]]);
			fgsaca_prefetchw(&SA[isas[i + 2 + prefetch_distance]]);
			fgsaca_prefetchw(&SA[isas[i + 3 + prefetch_distance]]);

			SA[isas[i + 0]]--;
			SA[isas[i + 1]]--;
			SA[isas[i + 2]]--;
			SA[isas[i + 3]]--;
		}
		for (j += prefetch_distance + 3; i < j; i += 1) {
			SA[isas[i + 0]]--;
		}
	} else {
		for (size_t i = 0; i < cnt; i++) {
			SA[isas[i]]--;
		}
	}
}

template <bool use_prefetching, typename Ix>
FGSACA_INLINE void
set_new_group_start(
	Ix* FGSACA_RESTRICT SA,
	Ix* FGSACA_RESTRICT isas,
	const size_t cnt)
{
	if constexpr (use_prefetching) {
		constexpr ptrdiff_t prefetch_distance = 16;
		ptrdiff_t i, j;
		for (i = 0; i < prefetch_distance; i++) {
			fgsaca_prefetch(&SA[isas[i]]);
		}
		for (i = 0, j = cnt - prefetch_distance - 3; i < j; i += 4) {
			fgsaca_prefetchw(&isas[i + 2 * prefetch_distance]);

			fgsaca_prefetch(&SA[isas[i + 0 + prefetch_distance]]);
			fgsaca_prefetch(&SA[isas[i + 1 + prefetch_distance]]);
			fgsaca_prefetch(&SA[isas[i + 2 + prefetch_distance]]);
			fgsaca_prefetch(&SA[isas[i + 3 + prefetch_distance]]);

			isas[i + 0] += SA[isas[i + 0]];
			isas[i + 1] += SA[isas[i + 1]];
			isas[i + 2] += SA[isas[i + 2]];
			isas[i + 3] += SA[isas[i + 3]];
		}
		for (j += prefetch_distance + 3; i < j; i += 1) {
			isas[i + 0] += SA[isas[i + 0]];
		}
	} else {
		for (size_t i = 0; i < cnt; i++) {
			isas[i] += SA[isas[i]];
		}
	}
}

template <bool use_prefetching, typename Ix, typename mem_ix_t>
FGSACA_INLINE void
increment_new_group_start(
	Ix* FGSACA_RESTRICT SA,
	Ix* FGSACA_RESTRICT isas,
	Ix* FGSACA_RESTRICT sorted,
	mem_ix_t* FGSACA_RESTRICT ISA,
	const size_t cnt)
{
	if constexpr (use_prefetching) {
		constexpr ptrdiff_t prefetch_distance = 16;
		ptrdiff_t i, j;
		for (i = 0; i < prefetch_distance; i++) {
			fgsaca_prefetchw(&ISA[sorted[i]]);
			fgsaca_prefetchw(&SA[isas[i]]);
		}
		for (i = 0, j = cnt - prefetch_distance - 3; i < j; i += 4) {
			fgsaca_prefetch(&isas[i + 2 * prefetch_distance]);
			fgsaca_prefetch(&sorted[i + 2 * prefetch_distance]);

			fgsaca_prefetchw(&ISA[sorted[i + 0 + prefetch_distance]]);
			fgsaca_prefetchw(&ISA[sorted[i + 1 + prefetch_distance]]);
			fgsaca_prefetchw(&ISA[sorted[i + 2 + prefetch_distance]]);
			fgsaca_prefetchw(&ISA[sorted[i + 3 + prefetch_distance]]);

			fgsaca_prefetchw(&SA[isas[i + 0 + prefetch_distance]]);
			fgsaca_prefetchw(&SA[isas[i + 1 + prefetch_distance]]);
			fgsaca_prefetchw(&SA[isas[i + 2 + prefetch_distance]]);
			fgsaca_prefetchw(&SA[isas[i + 3 + prefetch_distance]]);

			SA[ISA[sorted[i + 0]] = isas[i + 0]]++;
			SA[ISA[sorted[i + 1]] = isas[i + 1]]++;
			SA[ISA[sorted[i + 2]] = isas[i + 2]]++;
			SA[ISA[sorted[i + 3]] = isas[i + 3]]++;
		}
		for (j += prefetch_distance + 3; i < j; i += 1) {
			SA[ISA[sorted[i + 0]] = isas[i + 0]]++;
		}
	} else {
		for (size_t i = 0; i < cnt; i++) {
			SA[ISA[sorted[i]] = isas[i]]++;
		}
	}
}

template <typename Ix, typename mem_ix_t>
FGSACA_INLINE void
fetch_isas(
	Ix* FGSACA_RESTRICT isas,
	const mem_ix_t* FGSACA_RESTRICT ISA,
	const Ix* FGSACA_RESTRICT sorted,
	const ptrdiff_t cnt)
{
	constexpr ptrdiff_t prefetch_distance = 32;
	ptrdiff_t i, j;
	for (i = 0, j = std::min(cnt, prefetch_distance); i < j; i++)
		fgsaca_prefetch(&ISA[sorted[i]]);
	for (i = 0, j = cnt - prefetch_distance - 3; i < j; i += 4) {
		fgsaca_prefetch(&ISA[sorted[prefetch_distance + 0 + i]]);
		fgsaca_prefetch(&ISA[sorted[prefetch_distance + 1 + i]]);
		fgsaca_prefetch(&ISA[sorted[prefetch_distance + 2 + i]]);
		fgsaca_prefetch(&ISA[sorted[prefetch_distance + 3 + i]]);

		isas[i + 0] = ISA[sorted[i + 0]];
		isas[i + 1] = ISA[sorted[i + 1]];
		isas[i + 2] = ISA[sorted[i + 2]];
		isas[i + 3] = ISA[sorted[i + 3]];
	}
	for (; i < cnt; i++) {
		isas[i] = ISA[sorted[i]];
	}
}

template <bool use_prefetching, typename Ix>
FGSACA_INLINE void
insert_last_children(
	const Ix* FGSACA_RESTRICT isas,
	Ix* FGSACA_RESTRICT SA,
	const Ix* FGSACA_RESTRICT sorted,
	const ptrdiff_t cnt)
{
	if constexpr (use_prefetching) {
		constexpr ptrdiff_t prefetch_distance = 16;

		for (ptrdiff_t i = cnt - 1, j = std::max(ptrdiff_t {}, cnt - 2 * prefetch_distance); i >= j; i--)
			fgsaca_prefetchw(&SA[isas[i]]);
		for (ptrdiff_t i = cnt - 1, j = std::max(ptrdiff_t {}, cnt - prefetch_distance); i >= j; i--)
			if (SA[isas[i]] >= 8)
				fgsaca_prefetchw(&SA[isas[i] + SA[isas[i]] - 1]);

		ptrdiff_t i;
		for (i = cnt - 1; i >= 2 * prefetch_distance + 3; i -= 4) {
			fgsaca_prefetch(&isas[i - 3 * prefetch_distance]);

			fgsaca_prefetchw(&SA[isas[i - 2 * prefetch_distance - 0]]);
			fgsaca_prefetchw(&SA[isas[i - 2 * prefetch_distance - 1]]);
			fgsaca_prefetchw(&SA[isas[i - 2 * prefetch_distance - 2]]);
			fgsaca_prefetchw(&SA[isas[i - 2 * prefetch_distance - 3]]);

			fgsaca_prefetchw(&SA[isas[i - prefetch_distance - 0] + SA[isas[i - prefetch_distance - 0]] - 1]);
			fgsaca_prefetchw(&SA[isas[i - prefetch_distance - 1] + SA[isas[i - prefetch_distance - 1]] - 1]);
			fgsaca_prefetchw(&SA[isas[i - prefetch_distance - 2] + SA[isas[i - prefetch_distance - 2]] - 1]);
			fgsaca_prefetchw(&SA[isas[i - prefetch_distance - 3] + SA[isas[i - prefetch_distance - 3]] - 1]);

			SA[isas[i - 0] + --SA[isas[i - 0]]] = mark(sorted[i - 0]);
			SA[isas[i - 1] + --SA[isas[i - 1]]] = mark(sorted[i - 1]);
			SA[isas[i - 2] + --SA[isas[i - 2]]] = mark(sorted[i - 2]);
			SA[isas[i - 3] + --SA[isas[i - 3]]] = mark(sorted[i - 3]);
		}
		for (; i >= 0; i -= 1) {
			SA[isas[i - 0] + --SA[isas[i - 0]]] = mark(sorted[i - 0]);
		}
	} else {
		constexpr ptrdiff_t prefetch_distance = 32;
		const ptrdiff_t j = std::max(ptrdiff_t {}, cnt - prefetch_distance);
		for (ptrdiff_t i = cnt - 1; i >= j; i--)
			fgsaca_prefetchw(&SA[isas[i]]);
		ptrdiff_t i;
		for (i = cnt - 1; i >= prefetch_distance; i--) {
			fgsaca_prefetchw(&SA[isas[i - prefetch_distance]]);
			const auto pos = isas[i] + --SA[isas[i]];
			SA[pos] = mark(sorted[i]);
		}
		for (; i >= 0; i--) {
			const auto pos = isas[i] + --SA[isas[i]];
			SA[pos] = mark(sorted[i]);
		}
	}
}

template <bool use_prefetching, typename Ix, typename mem_ix_t>
FGSACA_INLINE void
update_gstart_last_children(
	const Ix* FGSACA_RESTRICT isas,
	const Ix* FGSACA_RESTRICT SA,
	const Ix* FGSACA_RESTRICT sorted,
	mem_ix_t* FGSACA_RESTRICT ISA,
	const ptrdiff_t cnt)
{
	if constexpr (use_prefetching) {
		const ptrdiff_t prefetch_distance = 32;
		ptrdiff_t i, j;
		for (i = 0; i < prefetch_distance; i++) {
			fgsaca_prefetch(&SA[isas[i]]);
			fgsaca_prefetchw(&ISA[sorted[i]]);
		}
		for (i = 0, j = cnt - prefetch_distance - 3; i < j; i += 4) {
			fgsaca_prefetch(&isas[i + 2 * prefetch_distance]);
			fgsaca_prefetch(&sorted[i + 2 * prefetch_distance]);

			fgsaca_prefetch(&SA[isas[i + 0 + prefetch_distance]]);
			fgsaca_prefetch(&SA[isas[i + 1 + prefetch_distance]]);
			fgsaca_prefetch(&SA[isas[i + 2 + prefetch_distance]]);
			fgsaca_prefetch(&SA[isas[i + 3 + prefetch_distance]]);

			fgsaca_prefetchw(&ISA[sorted[i + 0 + prefetch_distance]]);
			fgsaca_prefetchw(&ISA[sorted[i + 1 + prefetch_distance]]);
			fgsaca_prefetchw(&ISA[sorted[i + 2 + prefetch_distance]]);
			fgsaca_prefetchw(&ISA[sorted[i + 3 + prefetch_distance]]);

			const auto p0 = isas[i + 0];
			if (const Ix s = SA[p0]; not is_marked(s))
				ISA[sorted[i + 0]] = p0 + s;
			const auto p1 = isas[i + 1];
			if (const Ix s = SA[p1]; not is_marked(s))
				ISA[sorted[i + 1]] = p1 + s;
			const auto p2 = isas[i + 2];
			if (const Ix s = SA[p2]; not is_marked(s))
				ISA[sorted[i + 2]] = p2 + s;
			const auto p3 = isas[i + 3];
			if (const Ix s = SA[p3]; not is_marked(s))
				ISA[sorted[i + 3]] = p3 + s;
		}
		for (j += prefetch_distance + 3; i < j; i += 1) {
			const auto p0 = isas[i + 0];
			if (const Ix s = SA[p0]; not is_marked(s))
				ISA[sorted[i + 0]] = p0 + s;
		}
	} else {
		for (ptrdiff_t i = 0; i < cnt; i++) {
			const auto pos = isas[i];
			if (const Ix s = SA[pos]; not is_marked(s))
				ISA[sorted[i]] = pos + s;
		}
	}
}

template <typename sa_ix_t>
void sort_by_multiplicity(
	sa_ix_t* FGSACA_RESTRICT B,
	sa_ix_t* FGSACA_RESTRICT tmp,
	sa_ix_t* FGSACA_RESTRICT buckets,
	const size_t single_cnt,
	const size_t no_single_cnt,
	const size_t sigma)
{
	// elements and counts are in tmp,
	// sorted into B
	// rearrange non-singles
	memset(buckets, 0, sizeof(buckets[0]) * sigma);
	if (sigma > 256) [[unlikely]] {
		size_t i;
		constexpr size_t prefetch_distance = 32;
		for (i = 0; i + 3 + prefetch_distance < no_single_cnt; i += 4) {
			fgsaca_prefetchw(&buckets[tmp[2 * i + 1 + prefetch_distance]]);
			fgsaca_prefetchw(&buckets[tmp[2 * i + 3 + prefetch_distance]]);
			fgsaca_prefetchw(&buckets[tmp[2 * i + 5 + prefetch_distance]]);
			fgsaca_prefetchw(&buckets[tmp[2 * i + 7 + prefetch_distance]]);

			buckets[tmp[2 * i + 1]]++;
			buckets[tmp[2 * i + 3]]++;
			buckets[tmp[2 * i + 5]]++;
			buckets[tmp[2 * i + 7]]++;
		}
		for (; i < no_single_cnt; i += 1) {
			buckets[tmp[2 * i + 1]]++;
		}

		size_t s;
		for (i = 2, s = single_cnt; i < sigma; i++) {
			s = buckets[i] += s;
		}

		for (i = no_single_cnt; i > 3 + prefetch_distance; i -= 4) {
			fgsaca_prefetchw(&buckets[tmp[2 * i - 1 - prefetch_distance]]);
			fgsaca_prefetchw(&buckets[tmp[2 * i - 3 - prefetch_distance]]);
			fgsaca_prefetchw(&buckets[tmp[2 * i - 5 - prefetch_distance]]);
			fgsaca_prefetchw(&buckets[tmp[2 * i - 7 - prefetch_distance]]);

			B[--buckets[tmp[2 * i - 1]]] = tmp[2 * i - 2];
			B[--buckets[tmp[2 * i - 3]]] = tmp[2 * i - 4];
			B[--buckets[tmp[2 * i - 5]]] = tmp[2 * i - 6];
			B[--buckets[tmp[2 * i - 7]]] = tmp[2 * i - 8];
		}
		for (; i-- > 0;)
			B[--buckets[tmp[2 * i + 1]]] = tmp[2 * i];
	} else {
		for (size_t i = 0; i < no_single_cnt; i++) {
			buckets[tmp[1 + 2 * i]]++;
		}
		for (size_t i = 2, s = single_cnt; i < sigma; i++) {
			s = buckets[i] += s;
		}
		for (size_t i = no_single_cnt; i-- > 0;)
			B[--buckets[tmp[1 + 2 * i]]] = tmp[2 * i];
	}
}

// single_cnt: no. of elements which are the only children of their parent
// single_lc_cnt: no. of singles that are last children
// no_single_cnt: no. of unique parents - single_cnt
// sigma: max key + 1
template <typename sa_ix_t, typename mem_ix_t, alg_t alg>
FGSACA_INLINE std::tuple<size_t, size_t, size_t, size_t, size_t> // sigma, single_cnt, single_lc_cnt, no_single_cnt, num_fac
fgsaca_phase_1_find_prevs(
	const mem_ix_t* FGSACA_RESTRICT pss,
	sa_ix_t* FGSACA_RESTRICT tmp,
	const size_t n,
	sa_ix_t* FGSACA_RESTRICT G,
	std::vector<std::pair<sa_ix_t, sa_ix_t>>& FGSACA_RESTRICT factor_groups,
	size_t num, const size_t mem_end, const size_t gstart, const size_t gend)
{
	size_t single_cnt = 0,
		single_lc_cnt = 0, // finalists
		no_single_cnt = 0;
	size_t sigma = 2;

	// for SA and BBWT we can guarantee that Lyndons factors
	// in a group always have smaller indices than the non-Lyndon factors
	// in that group. (Follows by the fact that for two Lyndon factors i,j with
	// i < j we have L_i >= L_j, and that for each non-Lyndon factor k there is l < k
	// s.t. L_l < l, e.g. l = pss[k]).
	// Since we have pss[j] = n for each Lyndon factor, we can easily find all
	// the Lyndon factors in this group (they are at the beginning).
	//
	// This is of course not true for the EBWT
	size_t num_factors = 0;
	if constexpr (alg == EBWT_t) {
		for (size_t i = 0; i < num; i++) {
			const sa_ix_t p = transfer_mark<sa_ix_t>(pss[unmark(G[i])]);
			if (p == n) [[unlikely]] {
				num_factors = 1;
				for (i++; i < num; i++) {
					const sa_ix_t p = transfer_mark<sa_ix_t>(pss[unmark(G[i])]);
					if (p == n) {
						num_factors++;
					} else {
						G[i - num_factors] = p;
					}
				}
				factor_groups.emplace_back(gstart, gend);
				num -= num_factors;
				if (num == 0)
					return std::make_tuple(0, 0, 0, 0, num_factors);
				break;
			}
			G[i] = p;
		}
	} else if (transfer_mark<sa_ix_t>(pss[unmark(G[0])]) == n) [[unlikely]] {
		num_factors = 1;
		while (num_factors < num and transfer_mark<sa_ix_t>(pss[unmark(G[num_factors])]) == n)
			num_factors++;
		for (size_t i = num_factors; i < num; i++) {
			G[i - num_factors] = transfer_mark<sa_ix_t>(pss[unmark(G[i])]);
			assert(unmark(G[i - num_factors]) < n);
		}
		if constexpr (alg != SACA_t) {
			factor_groups.emplace_back(gstart, gend);
		}
		num -= num_factors;
		if (num == 0)
			return std::make_tuple(0, 0, 0, 0, num_factors);
	} else {
		for (size_t i = 0; i < num; i++) {
			G[i] = transfer_mark<sa_ix_t>(pss[unmark(G[i])]);
			assert(unmark(G[i]) < n);
		}
	}

	size_t i = 0;

	sa_ix_t s = G[i];
	while (i < num) {
		sa_ix_t ss = s;
		size_t k = i + 1;

		while (k < num && unmark(s = G[k]) == ss) {
			k++;
			ss = s;
		}
		ASSUME(k == num || unmark(ss) < unmark(s));

		if (k == i + 1) {
			single_cnt++;
			if (is_marked(ss)) {
				G[single_lc_cnt++] = unmark(ss);
			} else {
				assert(mem_end - (single_cnt - single_lc_cnt) >= 2 * no_single_cnt);
				tmp[mem_end - (single_cnt - single_lc_cnt)] = ss;
			}
		} else {
			const size_t key = (k - i - 1u) * 2u + (1u ^ (ss >> MSB_i<sa_ix_t>()));
			sigma = std::max(sigma, key);

			assert(num - (single_cnt - single_lc_cnt) >= 2 * no_single_cnt + 2);
			tmp[2 * no_single_cnt + 0] = unmark(ss);
			tmp[2 * no_single_cnt + 1] = key;

			no_single_cnt++;
		}

		i = k;
	}
	return std::make_tuple(sigma + 1, single_cnt, single_lc_cnt, no_single_cnt, num_factors);
}

// returns the last lyndon factor and its position in SA
template <typename sa_ix_t, typename mem_ix_t, alg_t alg>
when_t<alg != SACA_t, size_t> fgsaca_phase1(
	sa_ix_t* FGSACA_RESTRICT SA,
	const mem_ix_t* FGSACA_RESTRICT pss,
	CVec<mem_ix_t>& FGSACA_RESTRICT gstarts,
	mem_ix_t* FGSACA_RESTRICT ISA,
	size_t n)
{
	static_assert(std::numeric_limits<size_t>::max() >= std::numeric_limits<sa_ix_t>::max());
	static_assert(std::numeric_limits<size_t>::max() >= std::numeric_limits<mem_ix_t>::max());
	size_t prefetch_ptr = n - 1;

	std::vector<std::pair<sa_ix_t, sa_ix_t>> factor_groups;

	size_t gend;
	for (gend = n; gend-- > 0; ) {
		ASSUME(gend < n);
		assert(unmark(SA[gend]) < n);
		const size_t gstart = ISA[unmark(SA[gend])];

		ASSUME(gstart <= gend);

		auto* const FGSACA_RESTRICT G = SA + gstart;
		size_t gsize = gend + 1 - gstart;

		if (prefetch_ptr >= gstart) {
			prefetch_ptr = gstart - 1u;
		}
		if (fgsaca_likely(gstart > 0) and gsize < 64) {
			prefetch_ptr = phase_1_prefetch<sa_ix_t>(ISA, pss, SA, gstart, prefetch_ptr);
		}

		if (gsize == 1) {
			const size_t s = unmark(G[0]);
			ISA[s] = mark<mem_ix_t>(gstart);

			const mem_ix_t p_ = pss[s];
			const size_t p = unmark(p_);
			if (p < n) [[likely]] {
				const auto gs = ISA[p];
				if (SA[gs] == 1 and is_marked(p_)) {
					SA[gs] = transfer_mark<sa_ix_t>(p_);
				} else if (SA[gs] != 1) {
					const auto pos = gs + --SA[gs];
					if (is_marked(p_)) {
						SA[pos] = transfer_mark<sa_ix_t>(p_);
					} else {
						SA[pos] = 1;
					}
					ISA[p] = pos;
				}
			} else assert(p == n);
			continue;
		}

		gstarts.emplace_back(gstart);

		size_t fs = gend + 1; // first unused index


		sa_ix_t* FGSACA_RESTRICT tmp;
		const size_t tmp_sz = fgsaca_unlikely(2*gsize >= n) ? n/2 : gsize;
		const bool dealloc_tmp = [&] {
			if (n - fs < tmp_sz) 
			{
				tmp = new sa_ix_t[tmp_sz];
				return true;
			}
			else
			{
				tmp = SA + fs;
				fs += tmp_sz;
				return false;
			}
			// TODO: re-enable

 			// && (gstarts_offset + gsize > gstarts.capacity());
			// } else {
			// tmp = gstarts.data() + gstarts_offset;
			// gstarts_offset += gsize;
		}();

		const auto [sigma, single_cnt, single_lc_cnt, no_single_cnt, num_factors]
			= fgsaca_phase_1_find_prevs<sa_ix_t, mem_ix_t, alg>(pss, tmp, n, G, factor_groups, gsize, tmp_sz, gstart, gend);
		gsize -= num_factors, gend -= num_factors;

		if (gsize == 0) [[unlikely]] {
			if (dealloc_tmp)
				delete[] tmp;
			gend = gstart;
			continue;
		}

		const size_t single_nlc_cnt = single_cnt - single_lc_cnt;
		memcpy(&G[single_lc_cnt], &tmp[tmp_sz - single_nlc_cnt], single_nlc_cnt * sizeof(G[0]));

		const size_t gs = single_cnt + no_single_cnt;

		sa_ix_t* FGSACA_RESTRICT buckets;
		const bool dealloc_buckets = [&] {
			if (false and 2 * no_single_cnt + sigma <= tmp_sz) {
				buckets = tmp + 2 * no_single_cnt;
				return false;
			}
			else if (gs + sigma <= gsize){
				buckets = G + gs;
				return false;
			} else if (fs + sigma < n) {
				buckets = SA + fs;
				fs += sigma;
				return false;
				// TODO: re-enable
				// } else if (sigma + gstarts_offset <= gstarts.capacity()) {
				// buckets = gstarts.data() + gstarts_offset;
				// gstarts_offset += sigma;
				// return false;
			}
			// TODO
			buckets = new sa_ix_t[sigma];
			return true;
		}();

		sort_by_multiplicity<sa_ix_t>(G, tmp, buckets, single_cnt, no_single_cnt, sigma);
		buckets[0] = 0;
		buckets[1] = single_lc_cnt;

		ASSUME(buckets[2] == single_cnt);

		if (gs < 32) {
			for (size_t i = sigma, k = gs; i-- > 0;) {
				const auto start = buckets[i];
				if (k == start)
					continue;

				if (i % 2 == 0) { // final
					for (size_t i = k; i-- > start;) {
						const auto s = G[i];
						const auto p = ISA[s];
						assert(s < n);
						SA[p + --SA[p]] = mark<sa_ix_t>(s);
					}
					for (size_t i = start; i < k; i++) {
						const auto s = G[i];
						const auto p = ISA[s];
						if (const auto o = SA[p]; not is_marked(o))
							ISA[s] = p + o;
					}

				} else { // strictly preliminary
					for (size_t i = start; i < k; i++) {
						SA[ISA[G[i]]]--;
					}
					for (size_t i = start; i < k; i++) {
						auto& p = ISA[G[i]];
						p += SA[p];
					}
					for (size_t i = start; i < k; i++) {
						SA[ISA[G[i]]]++;
					}
				}

				k = start;
			}
		} else {
			sa_ix_t* const FGSACA_RESTRICT isas = tmp;

			// fetch isas
			fetch_isas(isas, ISA, G, gs);

			// reordering
			for (size_t i = sigma, k = gs; i-- > 0;) {
				const auto start = buckets[i];
				if (k == start)
					continue;

				if (start < 64 && k >= 64) {
					prefetch_ptr = phase_1_prefetch<sa_ix_t>(ISA, pss, SA, gstart, prefetch_ptr);
				}

				if (i % 2 == 0) { // final
					if (k - start >= 512) {
						insert_last_children<true, sa_ix_t>(isas + start, SA, G + start, k - start);
						update_gstart_last_children<true, sa_ix_t>(isas + start, SA, G + start, ISA, k - start);
					} else {
						insert_last_children<false, sa_ix_t>(isas + start, SA, G + start, k - start);
						update_gstart_last_children<false, sa_ix_t>(isas + start, SA, G + start, ISA, k - start);
					}
				} else { // strictly preliminary
					if (k - start >= 512) {
						reduce_old_group_size<true, sa_ix_t>(SA, isas + start, k - start);
						set_new_group_start<true, sa_ix_t>(SA, isas + start, k - start);
						increment_new_group_start<true, sa_ix_t>(SA, isas + start, G + start, ISA, k - start);
					} else {
						reduce_old_group_size<false, sa_ix_t>(SA, isas + start, k - start);
						set_new_group_start<false, sa_ix_t>(SA, isas + start, k - start);
						increment_new_group_start<false, sa_ix_t>(SA, isas + start, G + start, ISA, k - start);
					}
				}

				k = start;
			}
		}

		if (dealloc_buckets)
			delete[] buckets;

		if (dealloc_tmp)
			delete[] tmp;

		gend = gstart;
	}

	[[maybe_unused]] mem_ix_t pos;
	if constexpr (alg != SACA_t)
		{ // insert Lyndon facators
			// for each Lyndon factor i, insert nss[i] at the position of i (using ISA)
			for (const auto& [gstart, gend] : factor_groups)
				SA[gstart] = gend;


			assert(pss[0] == n);
			size_t last = 0;
			pos = 0;
			for (size_t i = 0; i < n; i++)
				if (pss[i] == n) [[unlikely]] { // Lyndon factor
					const auto isa = ISA[i];
					ISA[i] = mark(pos);
					const_cast<mem_ix_t*>(pss)[i] = last; // TODO: this is ugly

					last = i;
					pos = is_marked(isa)
						? unmark(isa) // already in singleton group
						: (mem_ix_t) SA[isa]--;
				}
		}
	for (size_t i = 0; i < gstarts.size(); i++) {
		ASSUME(gstarts[i] < n);
		SA[gstarts[i]] = i;
	}

	for (size_t i = 0; i < n; i++) {
		if (const auto isa = ISA[i]; not is_marked(isa)) {
			ISA[i] = SA[isa];
			assert(gstarts[ISA[i]] == isa);
		}
	}
	if constexpr (alg != SACA_t) {
		return (size_t) pos;
	} else {
		return nothing_t {};
	}
}
	
} // namespace fgsaca_internal
