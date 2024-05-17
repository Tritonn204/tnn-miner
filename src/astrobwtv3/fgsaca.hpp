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

#pragma once

#include "fgsaca_internal.hpp"

template<typename Ix, typename T>
void fgsaca(T const* S, Ix* SA, size_t n, size_t sigma, size_t fs = 0) {
	using mem_ix_t = std::conditional_t
		< fgsaca_internal::use_40_bit_ints and std::is_same_v<Ix, uint64_t>
		, uint40_t
		, Ix>;
	fgsaca_internal::fgsaca_main<Ix, mem_ix_t, false, T>(
		S, SA, n, sigma, nullptr, fs);
}

template<typename T>
void fbbwt(T const* S, T* res, size_t n, size_t sigma) {
	if (n < fgsaca_internal::MSB<uint32_t>())
		fgsaca_internal::fbbwt<uint32_t, T>(S, res, n, sigma);
	else
		fgsaca_internal::fbbwt<uint64_t, T>(S, res, n, sigma);
}

template<
	typename Ix,
	typename T>
void febwt(T const* S, Ix const* ns, T* res, Ix* res_i, const size_t k, const size_t sigma) {
	fgsaca_internal::febwt<Ix>(
		[&] (const size_t, const size_t s) { return S + s; },
		[&] (const size_t i) { return ns[i]; },
		res,
		res_i,
		k,
		sigma);
}
template<
	typename Ix,
	typename T>
void febwt(T const* const* S, Ix const* ns, T* res, Ix* res_i, const size_t k, const size_t sigma) {
	fgsaca_internal::febwt<Ix>(
		[&] (const size_t i, const size_t) { return S[i]; },
		[&] (const size_t i) { return ns[i]; },
		res,
		res_i,
		k,
		sigma);
}
template<
	typename Ix,
	typename T>
void febwt(std::vector<T> const* S, T* res, Ix* res_i, const size_t k, const size_t sigma) {
	fgsaca_internal::febwt<Ix>(
		[&] (const size_t i, const size_t) { return S[i].data(); },
		[&] (const size_t i) { return S[i].size(); },
		res,
		res_i,
		k,
		sigma);
}
template<typename Ix>
void febwt(std::string const* S, char* res, Ix* res_i, const size_t k, const size_t sigma) {
	fgsaca_internal::febwt<Ix>(
		[&] (const size_t i, const size_t) { return (const uint8_t*) S[i].data(); },
		[&] (const size_t i) { return S[i].size(); },
		(uint8_t*) res,
		res_i,
		k,
		sigma);
}
template<typename Ix>
void febwt(std::string_view const* S, char* res, Ix* res_i, const size_t k, const size_t sigma) {
	fgsaca_internal::febwt<Ix>(
		[&] (const size_t i, const size_t) { return (const uint8_t*) S[i].data(); },
		[&] (const size_t i) { return S[i].size(); },
		(uint8_t*) res,
		res_i,
		k,
		sigma);
}
