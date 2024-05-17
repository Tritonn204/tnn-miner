/*
 * fgsaca.hpp for FGSACA
 * Copyright (c) 2021-2023 Jannik Olbrich All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// This implementation is inspired by
// https://github.com/jonas-ellert/nearest-smaller-suffixes

#pragma once

#include "40bit.hpp"
#include "fgsaca_common.hpp"

namespace pss {

using namespace fgsaca_internal;

template <typename index_type, index_type mark_v>
FGSACA_INLINE constexpr index_type mark(
	const index_type v)
{
	if constexpr (mark_v == 0u) {
		return v;
	} else if constexpr (std::is_same_v<index_type, uint40_t>) {
		if constexpr (mark_v.m_high == 0) {
			index_type res = v;
			res.m_low |= mark_v.m_low;
			return res;
		} else if constexpr (mark_v.m_low == 0) {
			index_type res = v;
			res.m_high |= mark_v.m_high;
			return res;
		} else {
			return v | mark_v;
		}
	} else {
		return v | mark_v;
	}
}
template <typename index_type, index_type mark_v>
FGSACA_INLINE constexpr index_type unmark(
	const index_type v)
{
	if constexpr (mark_v == 0u) {
		return v;
	} else if constexpr (std::is_same_v<index_type, uint40_t>) {
		if constexpr (mark_v.m_high == 0) {
			index_type res = v;
			res.m_low &= ~mark_v.m_low;
			return res;
		} else if constexpr (mark_v.m_low == 0) {
			index_type res = v;
			res.m_high &= ~mark_v.m_high;
			return res;
		} else {
			return v & ~mark_v;
		}
	} else {
		return v & ~mark_v;
	}
}

template <size_t max_lce, typename value_type>
FGSACA_INLINE size_t lcp_const_bounded(
	const value_type* FGSACA_RESTRICT text,
	const size_t i,
	const size_t j)
{
	ASSUME(i > j);
	for (size_t lce = 0; fgsaca_likely(lce < max_lce); lce++)
		if (text[i + lce] != text[j + lce])
			return lce;
	return max_lce;
}
template <typename value_type>
FGSACA_INLINE size_t lcp_unbounded(
	const value_type* FGSACA_RESTRICT text,
	const size_t n,
	const size_t i,
	const size_t j,
	size_t lce = 0)
{
	ASSUME(i > j);
	while (fgsaca_likely(i + lce < n) and text[i + lce] == text[j + lce])
		lce++;
	return lce;
}
template <typename value_type>
FGSACA_INLINE size_t lcp_bounded(
	const value_type* FGSACA_RESTRICT text,
	size_t n,
	const size_t i,
	const size_t j,
	size_t lce,
	const size_t lce_hi)
{
	ASSUME(i > j);
	if (i + lce_hi < n) [[likely]]
		n = i + lce_hi;
	while (i + lce < n and text[i + lce] == text[j + lce])
		lce++;
	return lce;
}
template <typename value_type>
FGSACA_INLINE bool cmp_pos(
	const value_type* FGSACA_RESTRICT text,
	const size_t n,
	const size_t i,
	const size_t j,
	const size_t lce)
{
	return fgsaca_unlikely(not(i + lce < n)) or text[i + lce] < text[j + lce];
}

template <
	typename index_type,
	typename value_type>
FGSACA_INLINE index_type run_extension(
	index_type* FGSACA_RESTRICT pss,
	const value_type* FGSACA_RESTRICT text,
	const size_t max_lce,
	const size_t l,
	size_t r,
	const size_t period)
{
	const bool l_smaller = text[l + max_lce] < text[r + max_lce];
	const size_t num_repetitions = max_lce / period - 1;
	const size_t rb = r + num_repetitions * period;

	for (size_t k = r + 1; k < rb; k++) {
		pss[k] = pss[k - period] + period;
	}

	if (l_smaller) { // increasing run
		assert(pss[r] >= l);
		for (size_t rep = 0; rep < num_repetitions; rep++) {
			r += period;
			pss[r] = r - period;
		}
	} else { // decreasing run
		const index_type parent = pss[r];
		for (size_t rep = 0; rep < num_repetitions; rep++) {
			r += period;
			pss[r] = parent;
		}
	}
	return r;
}
template <typename value_type>
FGSACA_INLINE std::pair<size_t, size_t> is_extended_lyndon_run(
	const value_type* text,
	const size_t n)
{
	size_t factor_length = 0, suf_length = 0;
	for (size_t i = 0; i < n;) {
		size_t j = i + 1, k = i;
		while (j < n and text[k] <= text[j]) {
			if (text[k] < text[j]) {
				k = i;
			} else {
				k++;
			}
			j++;
		}
		if (j - k > factor_length) [[unlikely]]
			factor_length = j - k, suf_length = i;

		do {
			i += j - k;
		} while (i <= k);
		ASSUME(i > k and i <= k + (j - k));
	}
	if (2 * factor_length > n)
		return std::make_pair(0, 0);
	for (size_t i = factor_length; i < n; i++)
		if (text[i - factor_length] != text[i]) [[unlikely]]
			return std::make_pair(0, 0);
	return std::make_pair(factor_length, suf_length);
}
template <
	typename value_type>
FGSACA_INLINE size_t get_anchor(
	const value_type* FGSACA_RESTRICT text,
	const size_t lce)
{
	ASSUME(lce >= 8);
	const size_t l = lce / 4;
	const auto [factor_length, suf_length] = is_extended_lyndon_run(text + l, lce - l);

	if (factor_length == 0)
		return l;
	const auto rep_eq = [&](const size_t l, const size_t r) {
		for (size_t k = 0; k < factor_length; k++)
			if (text[l + k] != text[r + k])
				return false;
		return true;
	};
	ptrdiff_t lhs = static_cast<ptrdiff_t>(l + suf_length) - factor_length;
	while (lhs >= 0 and rep_eq(lhs, lhs + factor_length))
		lhs -= factor_length;
	ASSUME(lhs + factor_length * 2 >= 0);
	return std::min<size_t>(l, lhs + factor_length * 2);
}
template <
	typename index_type,
	typename value_type,
	index_type last_child_mark>
FGSACA_INLINE size_t amortized_lookahead(
	index_type* FGSACA_RESTRICT pss,
	const value_type* FGSACA_RESTRICT text,
	const size_t max_lce,
	const size_t l,
	const size_t r,
	const size_t distance)
{
	constexpr bool mark_last_children = last_child_mark != 0u;
	ASSUME(l < r);
	const size_t anchor = get_anchor(text + r, max_lce);

	for (size_t k = 1; k < anchor; k++) {
		pss[r + k] = pss[l + k] + distance;
	}
	if constexpr (mark_last_children) {
		size_t k = r + anchor - 1;
		while (k > r) {
			pss[k] = unmark<index_type, last_child_mark>(pss[k]);
			if (pss[k] >= k) [[unlikely]]
				break;
			k = pss[k];
		}
	}
	return r + anchor - 1;
}

template <
	typename index_type,
	typename value_type,
	index_type last_child_mark>
FGSACA_INLINE size_t find_pss_large_lcp(
	const value_type* FGSACA_RESTRICT text,
	index_type* FGSACA_RESTRICT pss,
	index_type* FGSACA_RESTRICT arr,
	const size_t n,
	const size_t i,
	size_t& FGSACA_RESTRICT max_lce,
	size_t& FGSACA_RESTRICT max_lce_idx)
{
	constexpr bool mark_last_children = last_child_mark != 0u;

	size_t hi = max_lce_idx, lce_hi = max_lce, lo = pss[hi];
	ASSUME(lo < hi);

	if (lo == n) [[unlikely]] {
		return n;
	}
	size_t lce_lo = 0;
	size_t num = 0; // number of pss-steps between pss[hi] and lo, i.e. lo = pss^(num+1)[hi]

	// we want: S_lo < S_i and S_hi > S_i
	// we have: S_hi > S_i
	// last child of hi is marked, hi is unmarked
	assert(( unmark<index_type, last_child_mark>(pss[hi]) == pss[hi] ));

	ASSUME(i + lce_hi == n or text[i + lce_hi] < text[hi + lce_hi]);
	lce_lo = lcp_bounded(text, n, i, lo, size_t { 0 }, lce_hi);
	while (lce_lo == lce_hi and cmp_pos(text, n, i, lo, lce_lo = lcp_unbounded(text, n, i, lo, lce_lo))) {
		// lo and arr[n-num..n) are not solution
		if constexpr (mark_last_children) {
			assert(( unmark<index_type, last_child_mark>(pss[hi]) == pss[hi] ));
			pss[hi] = mark<index_type, last_child_mark>(pss[hi]);
			// arr[n-num..n) are last children
			for (size_t k = num; k >= 1; k--) {
				const size_t c = arr[n - k];
				pss[c] = mark<index_type, last_child_mark>(pss[c]);
			}
		}
		ASSUME(lce_hi <= lce_lo);
		hi = lo, lce_hi = lce_lo;
		lo = pss[hi];
		if (lo == n) [[unlikely]] {
			max_lce = lce_hi, max_lce_idx = hi;
			return n;
		}
		for (num = 0; num < lce_lo; num++)
			if (const auto p = pss[lo]; p != n) [[likely]] {
				arr[n - num - 1] = lo;
				lo = p;
			} else {
				break;
			}
		lce_lo = lcp_bounded(text, n, i, lo, size_t { 0 }, lce_hi);
	}
	ASSUME(lce_hi >= max_lce);
	max_lce = lce_hi, max_lce_idx = hi;
	ASSUME(hi < n and lo < hi);
	if (num == 0) {
		return lo;
	}

	if (lce_lo > max_lce) {
		max_lce = lce_lo;
		max_lce_idx = lo;
	}

	assert(pss[hi] == arr[n - 1]);

	size_t lo_idx = n - num - 1;
	size_t hi_idx = n;
	//  solution is in arr (or lo), arr is sorted in increasing order
	while (lo_idx + 1 < hi_idx) {
		if (lce_lo <= lce_hi) {
			lce_lo = lcp_unbounded(text, n, i, arr[lo_idx + 1], lce_lo);
			if (max_lce < lce_lo) {
				max_lce = lce_lo;
				max_lce_idx = arr[lo_idx + 1];
			}
			if (const size_t n_lo = arr[lo_idx + 1]; cmp_pos(text, n, i, n_lo, lce_lo)) {
				// arr[lo_idx+1] is not a valid solution => arr[lo_idx] = lo is the solution
				break;
			} else {
				// arr[lo_idx+1] is a valid solution => arr[lo_idx] = lo is not needed
				lo = n_lo;
				lo_idx++;
			}
		} else {
			const size_t n_hi = arr[hi_idx - 1];
			lce_hi = lcp_unbounded(text, n, i, n_hi, lce_hi);
			if (max_lce < lce_hi) {
				max_lce = lce_hi;
				max_lce_idx = n_hi;
			}
			if (cmp_pos(text, n, i, n_hi, lce_hi)) {
				// arr[hi_idx - 1] = n_hi is not a valid solution
				hi_idx--;
			} else {
				// arr[hi_idx - 1] = n_hi is a valid solution
				lo = n_hi;
				lo_idx = hi_idx - 1;
				break;
			}
		}
	}
	if constexpr (mark_last_children) {
		// either lo_idx = n - num - 1 or lo == arr[lo_idx]
		assert(lo_idx == n - num - 1 or lo == arr[lo_idx]);
		// pss[arr[lo_idx + 1]] = pss[i] = lo
		// => arr[lo_idx + 1] is not last child
		for (size_t k = lo_idx + 2; k < n; k++) {
			const size_t c = arr[k];
			pss[c] = mark<index_type, last_child_mark>(pss[c]);
		}
		if (lo_idx + 2 <= n)
			pss[hi] = mark<index_type, last_child_mark>(arr[n - 1]);
	}
	// arr[lo_idx]
	return lo;
}
template <
	typename index_type,
	typename value_type,
	size_t threshold,
	index_type last_child_mark>
FGSACA_INLINE size_t find_pss(
	const value_type* FGSACA_RESTRICT text,
	index_type* FGSACA_RESTRICT pss,
	index_type* FGSACA_RESTRICT mem,
	const size_t n,
	const size_t i)
{
	constexpr bool mark_last_children = last_child_mark != 0u;

	ASSUME(i + threshold < n);

	[[maybe_unused]] size_t last = n;
	size_t j = i - 1, lce_hi = lcp_unbounded(text, n, i, j);
	if (lce_hi <= threshold) [[likely]] {
		while (ASSUME(lce_hi <= threshold), text[j + lce_hi] > text[i + lce_hi]) {
			// j is not a smaller suffix
			if constexpr (mark_last_children) {
				if (last < n) [[likely]] {
					pss[last] = mark<index_type, last_child_mark>(j);
				}
				last = j;
			}

			j = pss[j];
			if (j == n) [[unlikely]] {
				goto pss_immediate_accept;
			}
			lce_hi = lcp_const_bounded<threshold + 1>(text, i, j);
			if (lce_hi > threshold) [[unlikely]]
				break;
		}
		if (lce_hi <= threshold) [[likely]] {
		pss_immediate_accept:
			assert(j == n or (j < i and text[j + lce_hi] < text[i + lce_hi]));
			pss[i] = j;
			return i;
		}
	}
	ASSUME(j < i);
	lce_hi = lcp_unbounded(text, n, i, j, lce_hi);
	size_t max_lce = lce_hi, max_lce_idx = j;
	if (not cmp_pos(text, n, i, j, lce_hi)) {
		pss[i] = j;
	} else {
		// pss[last] = j > pss[i]
		// => last is last child of j
		if constexpr (mark_last_children) {
			if (last != n) [[likely]] {
				pss[last] = mark<index_type, last_child_mark>(j);
			}
		}
		if (pss[j] == n) [[unlikely]] {
			pss[i] = n;
		} else {
			pss[i] = find_pss_large_lcp<index_type, value_type, last_child_mark>(text, pss, mem, n, i, max_lce, max_lce_idx);
		}
	}
	const size_t distance = i - max_lce_idx;
	if (max_lce >= 2 * distance) [[unlikely]] {
		return run_extension(pss, text, max_lce, max_lce_idx, i, distance);
	} else {
		return amortized_lookahead<index_type, value_type, last_child_mark>(
			pss, text, max_lce, max_lce_idx, i, distance);
	}
}

template <
	typename index_type,
	typename value_type,
	index_type last_child_mark = 0>
FGSACA_INLINE void __attribute__((optimize("no-unroll-loops"))) compute_pss(
	const value_type* FGSACA_RESTRICT text,
	const size_t n,
	index_type* FGSACA_RESTRICT pss,
	index_type* FGSACA_RESTRICT mem = nullptr)
{
	const bool alloc_stack_mem = (mem == nullptr);
	if (alloc_stack_mem)
		mem = new index_type[n];

	constexpr bool mark_last_children = last_child_mark != 0u;

	pss[0] = n;
	constexpr size_t threshold = 127;

	size_t i = 1;
	if (n >= threshold) [[likely]]
		for (; i < n - threshold; i++)
			i = find_pss<index_type, value_type, threshold, last_child_mark>(
				text, pss, mem, n, i);
	for (; i < n; i++) {
		if (not cmp_pos(text, n, i, i - 1, lcp_unbounded(text, n, i, i - 1))) {
			pss[i] = i - 1;
			continue;
		}
		[[maybe_unused]] size_t lst = i - 1;
		size_t j = pss[i - 1];
		while (fgsaca_likely(j != n) and cmp_pos(text, n, i, j, lcp_unbounded(text, n, i, j))) {
			if constexpr (mark_last_children) {
				pss[lst] = mark<index_type, last_child_mark>(j);
				lst = j;
			}
			j = pss[j];
		}
		pss[i] = j;
	}

	if constexpr (mark_last_children) {
		// mark P_n as last children
		size_t i = n - 1;
		while (true) {
			const index_type j_ = pss[i];
			const size_t j = j_;
			if (j == n)
				break;
			pss[i] = mark<index_type, last_child_mark>(j_);
			i = j;
		}
	}

	if (alloc_stack_mem)
		delete[] mem;
}

template <
	typename index_type,
	typename value_type,
	index_type last_child_mark>
inline void compute_pss_naive(
	const value_type* FGSACA_RESTRICT text,
	const index_type n,
	index_type* FGSACA_RESTRICT pss)
{
	constexpr bool mark_last_children = last_child_mark > 0;
	pss[0] = n;
	for (size_t i = 1; i < n; i++) {
		if (not cmp_pos(text, n, i, i - 1, lcp_unbounded(text, n, i, i - 1))) {
			pss[i] = i - 1;
			continue;
		}
		pss[i] = pss[i - 1];
		[[maybe_unused]] size_t lst = i - 1;
		while (pss[i] != n and cmp_pos(text, n, i, pss[i], lcp_unbounded(text, n, i, pss[i]))) {
			if constexpr (mark_last_children) {
				pss[lst] = mark<index_type, last_child_mark>(pss[i]);
				lst = pss[i];
			}
			pss[i] = pss[pss[i]];
		}
	}
	if constexpr (mark_last_children) {
		size_t i = n - 1;
		while (true) {
			const index_type j_ = pss[i];
			const size_t j = j_;
			if (j == n)
				break;
			pss[i] = mark<index_type, last_child_mark>(j_);
			i = j;
		}
	}
}

} // namespace pss