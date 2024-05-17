/*
 * fgsaca_internal.hpp for FGSACA
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
#pragma once

#include "pss.hpp"
#include "canonicalisation.hpp"
#include "phase1.hpp"
#include "phase2.hpp"

#include <cassert>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <functional>
#include <utility>

namespace fgsaca_internal {

constexpr bool use_40_bit_ints = true;

template <typename T, typename F>
FGSACA_INLINE void
with_ls_types(
	T const* FGSACA_RESTRICT S,
	const size_t n,
	const F& f)
{
	size_t type = 0;
	size_t c0, c1;

	c1 = S[n - 1];
	// NOTE: S[n-1] will be judged as L-type, since false == (S[n-1] < S[n-1] + 0)
	{
		constexpr ptrdiff_t prefetch_distance = 64;
		ptrdiff_t i, j;
		for (i = n - 1, j = prefetch_distance + 3; i >= j; i -= 4) {
			fgsaca_prefetch(&S[i - prefetch_distance]);

			c0 = S[i - 0], type = c0 < c1 + type, f(i - 0, c0, type);
			c1 = S[i - 1], type = c1 < c0 + type, f(i - 1, c1, type);
			c0 = S[i - 2], type = c0 < c1 + type, f(i - 2, c0, type);
			c1 = S[i - 3], type = c1 < c0 + type, f(i - 3, c1, type);
		}
		for (j -= prefetch_distance + 3; i >= j; i -= 1) {
			c0 = S[i - 0], type = c0 < c1 + type, f(i - 0, c0, type);
			c1 = c0;
		}
	}
}
template <typename T, typename F>
FGSACA_INLINE void
with_ls_types_shifted(
	T const* FGSACA_RESTRICT S,
	const size_t shift,
	const size_t n,
	const F& f)
{
	ASSUME(shift < n);
	if (shift == 0) {
		with_ls_types<T, F>(S, n, f);
		return;
	}
	size_t type = 0;
	size_t c0, c1;

	c1 = S[shift - 1];
	// NOTE: S[n-1] will be judged as L-type, since false == (S[n-1] < S[n-1] + 0)
	constexpr ptrdiff_t prefetch_distance = 64;
	{
		ptrdiff_t i, j;
		for (i = shift - 1, j = prefetch_distance + 3; i >= j; i -= 4) {
			fgsaca_prefetch(&S[i - prefetch_distance]);

			c0 = S[i - 0], type = c0 < c1 + type, f(n - shift + i - 0, c0, type);
			c1 = S[i - 1], type = c1 < c0 + type, f(n - shift + i - 1, c1, type);
			c0 = S[i - 2], type = c0 < c1 + type, f(n - shift + i - 2, c0, type);
			c1 = S[i - 3], type = c1 < c0 + type, f(n - shift + i - 3, c1, type);
		}
		for (j -= prefetch_distance + 3; i >= j; i -= 1) {
			c0 = S[i - 0], type = c0 < c1 + type, f(n - shift + i - 0, c0, type);
			c1 = c0;
		}
	}
	{
		ptrdiff_t i, j;
		for (i = n - 1, j = shift + prefetch_distance + 3; i >= j; i -= 4) {
			fgsaca_prefetch(&S[i - prefetch_distance]);

			c0 = S[i - 0], type = c0 < c1 + type, f(2 * shift + i - 0 - n, c0, type);
			c1 = S[i - 1], type = c1 < c0 + type, f(2 * shift + i - 1 - n, c1, type);
			c0 = S[i - 2], type = c0 < c1 + type, f(2 * shift + i - 2 - n, c0, type);
			c1 = S[i - 3], type = c1 < c0 + type, f(2 * shift + i - 3 - n, c1, type);
		}
		for (j -= prefetch_distance + 3; i >= j; i -= 1) {
			c0 = S[i - 0], type = c0 < c1 + type, f(2 * shift + i - 0 - n, c0, type);
			c1 = c0;
		}
	}
}


template <typename Ix>
FGSACA_INLINE void
fgsaca_write_group_starts(
	Ix* FGSACA_RESTRICT SA,
	const Ix* FGSACA_RESTRICT C,
	const size_t sigma)
{ // write sizes of groups to starts in SA
	constexpr size_t prefetch_distance = 32;
	size_t i = 0, j;
	if (2 * sigma > prefetch_distance)
		for (j = 2 * sigma - prefetch_distance; i < j; i += 1) {
			if (C[i + 1 + prefetch_distance] > C[i + prefetch_distance])
				fgsaca_prefetchw(&SA[C[i + prefetch_distance]]);
			if (C[i + 1] > C[i]) {
				SA[C[i]] = C[i + 1] - C[i];
			}
		}
	for (; i < 2 * sigma; i++) {
		if (C[i + 1] > C[i]) {
			SA[C[i]] = C[i + 1] - C[i];
		}
	}
}


template <typename Ix, typename mem_ix_t, typename T>
FGSACA_INLINE void
fgsaca_insert_leaves_set_ISA(
	T const* FGSACA_RESTRICT S,
	const size_t n,
	Ix* FGSACA_RESTRICT SA,
	mem_ix_t* FGSACA_RESTRICT ISA,
	const Ix* FGSACA_RESTRICT C,
	const size_t offset = 0)
{
	size_t c1 = S[n - 1], c0, type = 0;
	for (size_t i = n; i-- > 0;) {
		c0 = S[i], type = c0 < c1 + type;
		c1 = c0;
		const size_t cc = c0 * 2 + type;
		const Ix gstart = C[cc];
		ISA[i + offset] = gstart;
		if (type == 0) {
			const auto pos = gstart + --SA[gstart];
			SA[pos] = mark(i + offset); // inserted in increasing order
			assert(unmark<Ix>(SA[pos]) == i + offset);
		}
	}
}
template <typename Ix, typename mem_ix_t, typename T>
FGSACA_INLINE void
fgsaca_insert_leaves_set_ISA_shifted(
	T const* FGSACA_RESTRICT S,
	const size_t n,
	const size_t shift,
	Ix* FGSACA_RESTRICT SA,
	mem_ix_t* FGSACA_RESTRICT ISA,
	const Ix* FGSACA_RESTRICT C,
	const size_t offset = 0)
{
	ASSUME(shift < n);
	if (shift == 0) {
		fgsaca_insert_leaves_set_ISA<Ix, mem_ix_t, T>(S, n, SA, ISA, C, offset);
		return;
	}

	size_t c1 = S[shift - 1], c0, type = 0;
	size_t p = offset + n - 1;
	for (size_t i = shift; i-- > 0; p--) {
		c0 = S[i], type = c0 < c1 + type;
		c1 = c0;
		const size_t cc = c0 * 2 + type;
		const size_t gstart = C[cc];
		ISA[p] = gstart;
		if (type == 0) {
			const auto pos = gstart + --SA[gstart];
			SA[pos] = mark(p); // inserted in increasing order
		}
	}
	for (size_t i = n; i-- > shift; p--) {
		c0 = S[i], type = c0 < c1 + type;
		c1 = c0;
		const size_t cc = c0 * 2 + type;
		const Ix gstart = C[cc];
		ISA[p] = gstart;
		if (type == 0) {
			const auto pos = gstart + --SA[gstart];
			SA[pos] = mark(p); // inserted in increasing order
		}
	}
}



template <typename Ix, typename mem_ix_t, bool bbwt, typename T>
void fgsaca_main(
	T const* FGSACA_RESTRICT S,
	Ix* FGSACA_RESTRICT SA,
	size_t n,
	size_t sigma,
	mem_ix_t* FGSACA_RESTRICT mem,
	size_t s_mem)
{
	static_assert(not std::is_same_v<Ix, uint40_t>);
	
	static_assert(std::numeric_limits<Ix>::max() >= std::numeric_limits<mem_ix_t>::max());
	assert(n < std::numeric_limits<Ix>::max() / 2);

	const bool isa_allocated = s_mem < 2 * n;
	mem_ix_t* FGSACA_RESTRICT const ISA = [&] {
		if (isa_allocated) {
			return new mem_ix_t[2 * n];
		} else {
			mem += 2 * n;
			s_mem -= 2 * n;
			return mem - 2 * n;
		}
	}();
	mem_ix_t* FGSACA_RESTRICT const pss = ISA + n;

	CVec<mem_ix_t> gstarts(mem, s_mem);

	pss::compute_pss<mem_ix_t, T, MSB<mem_ix_t>()>(S, n, pss, ISA);

	// build initial group structure
	{
		auto C = std::make_unique<Ix[]>(2 * sigma + 1);

		memset(C.get(), 0, sizeof(C[0]) * (2 * sigma + 1));

		with_ls_types<T>(S, n, [&C](const auto /* i */, const T c, const size_t type) {
			C[2 * c + type]++;
		});

		memset(SA, 0, sizeof(SA[0]) * n);

		// group starts
		for (size_t i = 0, j = 0; i < 2 * sigma + 1; ++i) {
			j += C[i];
			C[i] = j - C[i];
		}
		fgsaca_write_group_starts<Ix>(SA, C.get(), sigma);

		// insert leaves into groups, set ISA to start of group
		fgsaca_insert_leaves_set_ISA<Ix, mem_ix_t, T>(S, n, SA, ISA, C.get());
	}

	// process groups from highest to lowest
	const auto last_pos = fgsaca_phase1<Ix, mem_ix_t, bbwt ? BBWT_t : SACA_t> (SA, pss, gstarts, ISA, n);

	pss[0] = uint32_t { 0 };

	{ // phase II
		// interleave pss & ISA
		memcpy(SA, ISA, sizeof(ISA[0]) * n);
		for (size_t i = 0; i < n; i++) {
			ISA[2 * i + 0] = reinterpret_cast<const mem_ix_t*>(SA)[i];
			ISA[2 * i + 1] = pss[i];
		}

		fgsaca_phase2<Ix, mem_ix_t, bbwt>(SA, gstarts.data(), ISA, n, gstarts.size(), last_pos);
	}

	if (isa_allocated)
		delete[] ISA;
	else
		s_mem += 2 * n, mem -= 2 * n;
}
template <
	typename Ix,
	typename T>
void fbbwt(
	T const* FGSACA_RESTRICT S,
	T* FGSACA_RESTRICT res,
	size_t n,
	size_t sigma)
{
	const auto A = std::make_unique<Ix[]>(n); // SA_o

	using mem_ix_t = std::conditional_t
		< use_40_bit_ints and std::is_same_v<Ix, uint64_t>
		, uint40_t
		, Ix>;

	// align. this may not be portable, TODO: use std::align
	auto tmp_mem_ = reinterpret_cast<std::uintptr_t>(res);
	size_t tmp_sz = n * sizeof(T);
	while (tmp_sz > 0 && tmp_mem_ % sizeof(mem_ix_t) != 0)
		tmp_mem_++, tmp_sz--;
	tmp_sz /= sizeof(mem_ix_t);
	mem_ix_t* FGSACA_RESTRICT tmp_mem = tmp_sz > 0 ? reinterpret_cast<mem_ix_t*>(tmp_mem_) : nullptr;

	fgsaca_main<Ix, mem_ix_t, true, T>(S, A.get(), n, sigma, tmp_mem, tmp_sz);

	{ // BBWT from SA_o
		for (size_t i = 0; i < n; i++) {
			ASSUME(A[i] != 0 and A[i] <= n);
			res[i] = S[A[i] - 1];
		}
	}
}


template <
	typename Ix,
	typename IsaIx,
	typename get_N_t>
void febwt_find_idx(
	const size_t k,
	const size_t n,
	Ix* FGSACA_RESTRICT res_i,
	get_N_t get_N,
	const Ix* FGSACA_RESTRICT SA,
	IsaIx* FGSACA_RESTRICT ISA)
{
	memset(ISA, 0, sizeof(ISA[0]) * n);
	for (size_t i = 0, s = 0; i < k; i++) {
		const size_t n_i = get_N(i);
		const size_t rot = res_i[i];
		const size_t orig = (n_i * 2 - 1 - rot) % n_i; // index of successor of original rotation in (rotated - s)
		const size_t pos = orig + s;
		assert(pos < n);
		ISA[pos] = (IsaIx) (i + 1);
		s += n_i;
	}
	for (size_t i = 0; i < n; i++)
		if (const IsaIx seq_i = ISA[SA[i] - 1]; seq_i != 0) [[unlikely]] {
			res_i[seq_i - 1] = (Ix) i;
		}
}

template <
	typename Ix,
	typename get_S_t,
	typename get_N_t,
	typename T = decltype(std::declval<get_S_t>()((size_t) 0, (size_t) 0)[0]) >
void febwt(
	get_S_t get_S,
	get_N_t get_N,
	T* FGSACA_RESTRICT res,
	Ix* FGSACA_RESTRICT res_i,
	const size_t k,
	const size_t sigma)
{

	static_assert(sizeof(Ix) >= sizeof(T));
	const size_t n = [&] {
			size_t r = 0;
			for (size_t i = 0; i < k; i++)
				r += get_N(i);
			return r;
		}();

	using mem_ix_t = std::conditional_t
		< use_40_bit_ints and std::is_same_v<uint64_t, Ix>
		, uint40_t
		, Ix>;

	const auto ISA_ = std::make_unique<mem_ix_t[]>(2 * n);
	const auto SA_ = std::make_unique<Ix[]>(n);
	mem_ix_t* FGSACA_RESTRICT const ISA = ISA_.get();
	mem_ix_t* FGSACA_RESTRICT const pss = ISA_.get() + n;
	Ix* FGSACA_RESTRICT const SA = SA_.get();

	{ // find the minimum rotations and compute pss
		static_assert(sizeof(T) <= sizeof(mem_ix_t));
		T* FGSACA_RESTRICT const rotated = reinterpret_cast<T*>(ISA);
		{ // canonicalisation
			for (size_t i = 0, s = 0; i < k; i++) {
				const size_t n_i = get_N(i);
				const auto S = get_S(i, s);
				res_i[i] = canonicalise<T>(S, n_i);
				const auto mid = S + res_i[i];
				std::copy(mid, S + n_i, &rotated[s]);
				std::copy(S, mid, &rotated[s + n_i - res_i[i]]);
				s += n_i;
			}
		}

		{ // pss
			for (size_t i = 0, s = 0; i < k; i++) {
				const size_t n_i = get_N(i);
				pss::compute_pss<mem_ix_t, T, MSB<mem_ix_t>()>(rotated + s, n_i, pss + s, reinterpret_cast<mem_ix_t*>(SA));
				const mem_ix_t m_n = n;
				const mem_ix_t m_s = s;

				for (size_t j = 0; j < n_i; j++) // TODO: adapt compute_marked_pss s.t. -1 can be changed
					if (pss[j + s] == n_i) {
						pss[j + s] = m_n;
					} else {
						assert(unmark(pss[j + s]) < n_i);
						pss[j + s] += m_s;
					}

				s += n_i;
			}
		}
	}

	{ // build initial group structure
		auto C = std::make_unique<Ix[]>(2 * sigma + 1);

		memset(C.get(), 0, sizeof(C[0]) * (2 * sigma + 1));

		// Compute types for each sequence separately. This avoids
		// wrong types at the boundaries
		for (size_t i = k, s = n; i-- > 0;) {
			const size_t n_i = get_N(i);
			s -= n_i;
			const auto S = get_S(i, s);
			with_ls_types_shifted<T>(S, res_i[i], n_i, [&C](const size_t /* i */, const T c, const size_t type) {
				C[2 * c + type]++;
			});
		}

		memset(SA, 0, sizeof(SA[0]) * n);

		// group starts
		for (size_t i = 0, j = 0; i < 2 * sigma + 1; ++i) {
			j += C[i];
			C[i] = j - C[i];
		}
		fgsaca_write_group_starts<Ix>(SA, C.get(), sigma);

		// insert leaves into groups, set ISA to start of group
		// for each sequence separately
		for (size_t i = k, s = n; i-- > 0;) {
			const size_t n_i = get_N(i);
			s -= n_i;
			const auto S = get_S(i, s);
			fgsaca_insert_leaves_set_ISA_shifted<Ix, mem_ix_t, T>(S, n_i, res_i[i], SA, ISA, C.get(), s);
		}
	}

	CVec<mem_ix_t> gstarts(nullptr, 0); // TODO: use memory from res

	const auto last_pos = fgsaca_phase1<Ix, mem_ix_t, EBWT_t>(SA, pss, gstarts, ISA, n);

	pss[0] = 0;

	{ // interleave pss & ISA
		memcpy(SA, ISA, sizeof(ISA[0]) * n);
		for (size_t i = 0; i < n; i++) {
			ISA[2 * i + 0] = reinterpret_cast<const mem_ix_t*>(SA)[i];
			ISA[2 * i + 1] = pss[i];
		}
	}

	fgsaca_phase2<Ix, mem_ix_t, true>(SA, gstarts.data(), ISA, n, gstarts.size(), last_pos);

	{ // ebwt from SA_o
		T* FGSACA_RESTRICT const rotated = reinterpret_cast<T*>(ISA);
		for (size_t i = 0, s = 0; i < k; i++) {
			const size_t n_i = get_N(i);
			const auto S = get_S(i, s);
			const auto mid = S + res_i[i];
			std::copy(mid, S + n_i, &rotated[s]);
			std::copy(S, mid, &rotated[s + n_i - res_i[i]]);
			s += n_i;
		}
		for (size_t i = 0; i < n; i++) {
			assert(SA[i] > 0 and SA[i] <= n);
			res[i] = rotated[SA[i] - 1];
		}
	}
	{ // idx
		if (k < std::numeric_limits<uint8_t>::max()) {
			assert(std::numeric_limits<Ix>::max() >= std::numeric_limits<uint8_t>::max());
			febwt_find_idx<Ix, uint8_t>(k, n, res_i, std::move(get_N), SA, reinterpret_cast<uint8_t*>(ISA));
		} else if (k < std::numeric_limits<uint16_t>::max()) {
			assert(std::numeric_limits<Ix>::max() >= std::numeric_limits<uint16_t>::max());
			febwt_find_idx<Ix, uint16_t>(k, n, res_i, std::move(get_N), SA, reinterpret_cast<uint16_t*>(ISA));
		} else if (k < std::numeric_limits<uint32_t>::max()) {
			assert(std::numeric_limits<Ix>::max() >= std::numeric_limits<uint32_t>::max());
			febwt_find_idx<Ix, uint32_t>(k, n, res_i, std::move(get_N), SA, reinterpret_cast<uint32_t*>(ISA));
		} else {
			febwt_find_idx<Ix, Ix>(k, n, res_i, std::move(get_N), SA, reinterpret_cast<Ix*>(ISA));
		}
	}
}

} // namespace fgsaca_internal
