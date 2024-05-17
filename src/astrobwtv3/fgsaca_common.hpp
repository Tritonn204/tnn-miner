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

#include <type_traits>
#include <cassert>

#define FGSACA_RESTRICT	          __restrict__
#define FGSACA_INLINE             __attribute__((always_inline)) inline
#define fgsaca_prefetch(address)  __builtin_prefetch((const void *)(address), 0, 0)
#define fgsaca_prefetchw(address) __builtin_prefetch((const void *)(address), 1, 0)
#define fgsaca_likely(x)          __builtin_expect((x),1)
#define fgsaca_unlikely(x)        __builtin_expect((x),0)
#define FGSACA_UNREACHABLE()     __builtin_unreachable()

namespace fgsaca_internal {

FGSACA_INLINE constexpr void IMPOSSIBLE(bool x) noexcept
{
	if (std::is_constant_evaluated()) {
		assert(x);
	} else {
		if (x)
			FGSACA_UNREACHABLE();
	}
}
FGSACA_INLINE constexpr void ASSUME(bool x) noexcept
{
	IMPOSSIBLE(not x);
}

} // fgsaca_internal
