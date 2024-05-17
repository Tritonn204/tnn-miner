/*
 * fgsaca_util.hpp for FGSACA
 * Copyright (c) 2022 Jannik Olbrich All Rights Reserved.
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

#include "fgsaca_common.hpp"
#include "canonicalisation.hpp"
#include "40bit.hpp"

#include <algorithm>
#include <cstdint>
#include <cassert>
#include <numeric>
#include <set>
#include <memory>
#include <optional>
#include <vector>
#include <cstring>
#include <string>

namespace fgsaca_internal {

template<typename T>
consteval T bit(size_t i) noexcept {
	assert(i < sizeof(T)*8);
	return ((T) 1u) << i;
}
template<>
consteval uint40_t bit<uint40_t>(size_t i) noexcept {
	assert(i < sizeof(uint40_t)*8);
	if (i < sizeof(uint40_t::low_t) * 8)
		return uint40_t{ bit<uint40_t::low_t>(i), 0u };
	else
		return uint40_t{ 0u, bit<uint40_t::high_t>(i - sizeof(uint40_t::low_t) * 8u) };
}

// use consteval functions here because of a bug in GCC13
template <typename Ix>
consteval size_t MSB_i() { return 8u * sizeof(Ix) - 1; }
template<typename Ix, typename Ix2 = Ix>
consteval Ix MSB() { return bit<Ix>(MSB_i<Ix2>()); }

template <typename T, typename Alloc = std::allocator<T>>
struct CVec {
	T* m_data = nullptr;
	size_t m_size = 0, m_capacity = 0;
	bool m_mine = false;

	T* data() const { return m_data; }

	size_t size() const { return m_size; }

	size_t capacity() const { return m_capacity; }

	template <typename... Args>
	void
	emplace_back(
		Args&&... v)
	{
		if (m_size == m_capacity) [[unlikely]] {
			realloc();
		}
		ASSUME(m_size < m_capacity);
		std::construct_at(m_data + m_size, std::forward<Args>(v)...);
		m_size++;
	}
	CVec(
		T* mem,
		size_t cap)
		: m_data(mem)
		, m_size(0)
		, m_capacity(cap)
		, m_mine(false)
	{
	}
	CVec()
		: CVec(nullptr, 0)
	{
	}
	~CVec()
	{
		Alloc alloc;
		for (size_t i = 0; i < m_size; i++)
			std::destroy_at(m_data + i);
		if (m_mine) {
			alloc.deallocate(m_data, m_capacity);
		}
	}

	FGSACA_INLINE T&
	operator[](
		const size_t i)
	{
		ASSUME(i < m_size);
		return m_data[i];
	}
	FGSACA_INLINE const T&
	operator[](
		const size_t i) const
	{
		ASSUME(i < m_size);
		return m_data[i];
	}

	void
	realloc()
	{
		Alloc alloc;
		size_t new_cap = static_cast<size_t>(m_capacity * 1.61);
		{
			constexpr size_t step = (4096 + sizeof(T) - 1) / sizeof(T);
			new_cap += step;
			new_cap = (new_cap + step - new_cap % step);
			ASSUME(new_cap % step == 0);
		}
		T* FGSACA_RESTRICT const tmp = alloc.allocate(new_cap);

		for (size_t i = 0; i < m_size; i++)
			std::construct_at(tmp + i, std::move(m_data[i]));
		if (m_mine)
			alloc.deallocate(m_data, m_capacity);

		m_mine = true, m_capacity = new_cap, m_data = tmp;
	}
};

template <size_t offset, size_t stride, size_t cons = 1>
struct idx_comp {
	static_assert(cons > 0);
	static_assert(stride > 0);
	static FGSACA_INLINE constexpr size_t
	idx(
		size_t i)
	{
		return offset + (i / cons) * stride + (i % cons);
	}
	FGSACA_INLINE constexpr size_t
	operator[](
		size_t i) const
	{
		return idx(i);
	}
};


template <typename Ix, typename Ix2 = Ix>
FGSACA_INLINE bool
is_marked(
	const Ix v)
{
	return (v & MSB<Ix, Ix2>()) != 0;
}
template <>
FGSACA_INLINE bool
is_marked<uint40_t>(
	const uint40_t v)
{
	return is_marked<uint40_t::high_t>(v.m_high);
}
template <typename Ix, typename Ix2 = Ix>
FGSACA_INLINE constexpr Ix
mark(
	const Ix v)
{
	ASSUME(v < (MSB<Ix, Ix2>()));
	return v | MSB<Ix, Ix2>();
}
template <>
FGSACA_INLINE constexpr uint40_t
mark(
	uint40_t v)
{
	v.m_high = mark<uint40_t::high_t>(v.m_high);
	return v;
}
template <typename Ix, typename Ix2 = Ix>
FGSACA_INLINE constexpr Ix
unmark(
	const Ix v)
{
	return v & ~MSB<Ix, Ix2>();
}

template <typename Ix, typename Ix2>
FGSACA_INLINE std::enable_if_t<std::is_same_v<Ix, Ix2>, Ix>
transfer_mark(
	const Ix2 v)
{
	return v;
}

template <typename Ix, typename Ix2>
FGSACA_INLINE std::enable_if_t<not std::is_same_v<Ix, Ix2>, Ix>
transfer_mark(
	const Ix2 v)
{
	static_assert(std::is_unsigned_v<Ix>);
	static_assert(sizeof(Ix) >= sizeof(Ix2));
	using low_t = typename Ix2::low_t;
	using high_t = typename Ix2::high_t;

	static_assert(std::is_same_v<Ix2, uint40_t>);
	Ix res = ((Ix) v.m_low) // the lower 32 bits can be taken as are
		| (((Ix) unmark<high_t>(v.m_high)) << (8 * sizeof(low_t))) //  the upper 8 bits must be unmarked and then shifted
		| (((Ix) (v.m_high & MSB<high_t>())) << (MSB_i<Ix>() - MSB_i<high_t>())); // extract and shift the mark-bit

	assert(is_marked<Ix>(res) == is_marked<Ix2>(v));
	assert(unmark<Ix>(res) == unmark<Ix2>(v));

	return res;
}

struct nothing_t { };
template <bool use, typename T>
using when_t = std::conditional_t<use, T, nothing_t>;

// returns std::nullopt if SA is correct and otherwise the first error found
template<typename Ix, typename T>
std::optional<std::string> validateSA(const T* FGSACA_RESTRICT text, const Ix n, const Ix sigma, const Ix* FGSACA_RESTRICT SA) {
	static_assert(std::is_unsigned_v<Ix>);

	using namespace std::literals;

	// verify range
	for (Ix i = 0; i < n; i++)
		if (SA[i] >= n)
			return "\'"s + std::to_string(SA[i]) + "\' is out of range (n=" + std::to_string(n) + ", i=" + std::to_string(i) + ")";

	// verify uniqueness
	{
		std::vector<bool> vis(n,false);
		for (Ix i = 0; i < n; i++) {
			if (vis[SA[i]])
				return "SA is not a permutation: "s + std::to_string(SA[i]) + " occured (at least) twice (e.g. at i=" + std::to_string(i) + ")";
			vis[SA[i]] = true;
		}
	}

	// check correct bucket
	for (Ix i = 1; i < n; i++)
		if (text[SA[i - 1]] > text[SA[i]])
			return "buckets are not correct: \'"s + std::to_string(text[SA[i - 1]]) + "\' is before \'" + std::to_string(text[SA[i]]) + "\'";

	Ix C[sigma];
	memset(C, 0, sizeof(C[0]) * sigma);
	for (Ix i = 0; i < n; i++)
		C[text[i]]++;
	for (Ix i = 0, s = 0; i < sigma; i++) {
		s += C[i];
		C[i] = s - C[i];
	}

	constexpr ptrdiff_t prefetch_distance = 32 * 4 / sizeof(Ix);
	ptrdiff_t i, j;
	if (n-1 != SA[C[text[n-1]]++]) return "order within buckets is wrong"s;

	for (i = 0, j = n - 2*prefetch_distance - 4; i < j; i += 4) {
		fgsaca_prefetch(&SA[i + 3 * prefetch_distance]);

		fgsaca_prefetch(&text[SA[i + 2 * prefetch_distance + 0] - 1]);
		fgsaca_prefetch(&text[SA[i + 2 * prefetch_distance + 1] - 1]);
		fgsaca_prefetch(&text[SA[i + 2 * prefetch_distance + 2] - 1]);
		fgsaca_prefetch(&text[SA[i + 2 * prefetch_distance + 3] - 1]);

		Ix v0 = SA[i + prefetch_distance + 0] - 1;
		Ix v1 = SA[i + prefetch_distance + 1] - 1;
		Ix v2 = SA[i + prefetch_distance + 2] - 1;
		Ix v3 = SA[i + prefetch_distance + 3] - 1;
		if (v0 < n) fgsaca_prefetchw(&C[text[v0]]);
		if (v1 < n) fgsaca_prefetchw(&C[text[v1]]);
		if (v2 < n) fgsaca_prefetchw(&C[text[v2]]);
		if (v3 < n) fgsaca_prefetchw(&C[text[v3]]);

		v0 = SA[i + 0] - 1;
		v1 = SA[i + 1] - 1;
		v2 = SA[i + 2] - 1;
		v3 = SA[i + 3] - 1;
		if (v0 < n && v0 != SA[C[text[v0]]++]) { return "order within buckets is wrong: i="s + std::to_string(i + 0); }
		if (v1 < n && v1 != SA[C[text[v1]]++]) { return "order within buckets is wrong: i="s + std::to_string(i + 1); }
		if (v2 < n && v2 != SA[C[text[v2]]++]) { return "order within buckets is wrong: i="s + std::to_string(i + 2); }
		if (v3 < n && v3 != SA[C[text[v3]]++]) { return "order within buckets is wrong: i="s + std::to_string(i + 3); }
	}
	for (j += 2 * prefetch_distance + 4; i < j; i += 1) {
		// the suffix S_{SA[i]-1} must be the (current) first in its bucket (the c1-bucket)
		Ix v0 = SA[i + 0] - 1;
		if (v0 < n && v0 != SA[C[text[v0]]++]) { return "order within buckets is wrong: i="s + std::to_string(i + 0); }
	}

	return std::nullopt;
}


template<typename Ix, typename T>
std::vector<Ix>
compute_lf(
	const T* FGSACA_RESTRICT S,
	const size_t n,
	const size_t sigma
)  {
	std::vector<Ix> C(sigma);
	for (Ix i = 0; i < n; i++)
		C[S[i]]++;
	for (Ix i = 0, s = 0; i < sigma; i++) {
		s += C[i];
		C[i] = s - C[i];
	}
	std::vector<Ix> lf(n);
	for (size_t i = 0; i < n; i++)
		lf[i] = C[S[i]]++;
	return lf;
}

	
template <typename T>
void invert_bbwt(const T* FGSACA_RESTRICT bbwt, const size_t n, T* FGSACA_RESTRICT out)
{
	const size_t sigma = size_t { 1 } + *std::max_element(bbwt, bbwt + n);
	auto lf = compute_lf<size_t>(bbwt, n, sigma);

	constexpr size_t VIS_MARKER = ~(std::numeric_limits<size_t>::max() >> 1);

	std::vector<std::vector<T>> factors;
	for (size_t i = 0; i < n; i++) {
		if (lf[i] & VIS_MARKER) [[likely]]
			continue;
		factors.emplace_back();
		auto& fac = factors.back();
		for (size_t p = lf[i]; ; ) {
			fac.emplace_back(bbwt[p]);

			const auto np = lf[p];
			assert((np & VIS_MARKER) == 0);
			lf[p] = np | VIS_MARKER;
			if (p == i) break;
			p = np;
		}
		fac.shrink_to_fit();
		std::reverse(fac.begin(), fac.end());
		const auto r = canonicalise(fac.data(), fac.size());
		std::rotate(fac.begin(), fac.begin() + r, fac.end());
		assert(canonicalise(fac.data(), fac.size()) == 0);
	}
	{
		auto tmp = std::move(lf);
	} // delete lf

	std::sort(factors.rbegin(), factors.rend());

	for (size_t i = 0, s = 0; i < factors.size(); i++) {
		assert(s + factors[i].size() <= n);
		std::copy(factors[i].begin(), factors[i].end(), out + s);
		s += factors[i].size();
	}
}

template<typename Ix, typename T>
void invert_ebwt(
	const T* FGSACA_RESTRICT ebwt,
	const Ix* FGSACA_RESTRICT start,
	const size_t k,
	const size_t n,
	T* FGSACA_RESTRICT out,
	Ix* FGSACA_RESTRICT idx)
{
	const Ix sigma = Ix { 1 } + *std::max_element(ebwt, ebwt + n);

	auto lf = compute_lf<Ix>(ebwt, n, sigma);
	constexpr Ix VIS_MARKER = ~(std::numeric_limits<Ix>::max() >> 1);

	std::vector<Ix> start_sorted(start, start + k);
	std::sort(start_sorted.begin(), start_sorted.end());

	const auto walk_lf = [&] (const Ix i, Ix& s) {
		assert((lf[i] & VIS_MARKER) == 0);
		for (Ix p = i; ; ) {
			assert(s < n);
			out[s++] = ebwt[p];
			const auto np = lf[p];
			assert((np & VIS_MARKER) == 0);
			lf[p] = np | VIS_MARKER;
			p = np;
			if (p == i)
				break;
		}
	};
	const auto check_lf = [&] (const Ix i, const Ix ref_last, const Ix len) {
		assert((lf[i] & VIS_MARKER) == 0);
		Ix l = 0;
		for (Ix p = i; ; ) {
			if (out[ref_last - l++] != ebwt[p])
				return false;
			const auto np = lf[p];
			assert((np & VIS_MARKER) == 0);
			p = np;
			if (p == i)
				return l == len;
			if (l >= len)
				return false;
		}
	};

	Ix s = 0;
	for (Ix j = 0; j < k; j++) {
		const Ix i = start_sorted[j];
		idx[j] = s;
		walk_lf(i, s);
		std::reverse(out + idx[j], out + s);
	}
	std::copy_backward(out, out + s, out + n);
	for (Ix j = 0; j < k; j++)
		idx[j] = n + idx[j] - s;

	s = 0;
	for (Ix j = 0; j < k; j++) {
		const Ix len = (j + 1 < k ? idx[j + 1] : n) - idx[j];

		std::copy(out + idx[j], out + idx[j] + len, out + s);
		idx[j] = s;
		s += len;
		assert(s <= n);

		Ix i = start_sorted[j] + 1;
		while (i < n and (lf[i] & VIS_MARKER) == 0 and check_lf(i, s - 1, len)) {
			walk_lf(i, s);
			std::reverse(out + s - len, out + s);
			i++;
		}
	}
}
template <typename T>
bool check_bbwt(const T* FGSACA_RESTRICT text, const size_t n, const T* FGSACA_RESTRICT bbwt)
{
	const auto reconstructed = std::make_unique<T[]>(n);
	invert_bbwt<T>(bbwt, n, reconstructed.get());
	return std::equal(text, text + n, reconstructed.get());
}
template <typename Ix, typename T>
bool
check_ebwt(
	T const* FGSACA_RESTRICT S,
	Ix const* FGSACA_RESTRICT ns,
	const size_t k,
	const uint8_t* FGSACA_RESTRICT ebwt,
	const Ix* FGSACA_RESTRICT idx
) {
	const size_t n = std::accumulate(ns, ns + k, 0u);
	const auto reconstructed = std::make_unique<uint8_t[]>(n);
	const auto reconstructed_idx = std::make_unique<Ix[]>(k);

	invert_ebwt(ebwt, idx, k, n, reconstructed.get(), reconstructed_idx.get());

	using seq_t = std::pair<Ix, const T*>;
	const auto cmp = [] (const seq_t& lhs, const seq_t& rhs) -> bool {
			const Ix len = std::min(lhs.first, rhs.first);
			for (size_t i = 0; i < len; i++)
				if (lhs.second[i] != rhs.second[i])
					return lhs.second[i] < rhs.second[i];
			return lhs.first < rhs.first;
		};
	std::multiset<seq_t, decltype(cmp)> seen(cmp);

	for (size_t i = 0, s = 0; i < k; i++)
	{
		seen.emplace(ns[i], S + s);
		s += ns[i];
	}

	for (size_t i = 0; i < k; i++)
	{
		const size_t len = (i + 1 < k ? reconstructed_idx[i + 1] : n) - reconstructed_idx[i];
		const auto it = seen.find(std::make_pair(len, reconstructed.get() + reconstructed_idx[i]));
		if (it == seen.end()) {
			return false;
		} else {
			seen.erase(it);
		}
	}

	return seen.empty();
}

} // namespace fgsaca_internal
