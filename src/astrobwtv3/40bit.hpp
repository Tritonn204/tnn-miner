/*******************************************************************************
 * thrill/common/uint_types.hpp
 *
 * Class representing a 40-bit or 48-bit unsigned integer encoded in five or
 * six bytes.
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2013 Timo Bingmann <tb@panthema.net>
 *
 * All rights reserved. Published under the BSD-2 license printed below
 * 
 * 
 * MODIFIED FOR gsaca-double-sort
 * Copyright (C) 2020 Jonas Ellert
 *
 * FURTHER MODIFIED FOR fgsaca
 * Copyright (C) 2023 Jannik Olbrich
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#pragma once

#include <cassert>
#include <limits>
#include <memory>
#include <type_traits>

#include "fgsaca_common.hpp"

template<typename _high_t = uint8_t>
class two_word_int {
public:
	// lower part type, always 32-bit
	using low_t = uint32_t;
	// higher part type, currently either 8-bit or 16-bit
	using high_t = _high_t;

	// member containing lower significant integer value
	low_t m_low;
	// member containing higher significant integer value
	high_t m_high;

private:
	//! return highest value storable in lower part, also used as a mask.
	static constexpr unsigned low_max()
	{
		return std::numeric_limits<low_t>::max();
	}

	//! number of bits in the lower integer part, used a bit shift value.
	static constexpr size_t low_bits = 8 * sizeof(low_t);

	//! return highest value storable in higher part, also used as a mask.
	static constexpr unsigned high_max()
	{
		return std::numeric_limits<high_t>::max();
	}

	//! number of bits in the higher integer part, used a bit shift value.
	static constexpr size_t high_bits = 8 * sizeof(high_t);

public:
	//! number of binary digits (bits) in two_word_int
	static constexpr size_t digits = low_bits + high_bits;

	//! number of bytes in two_word_int
	static constexpr size_t bytes = sizeof(low_t) + sizeof(high_t);

	// compile-time assertions about size of Low
	static_assert(8 * sizeof(low_t) == 32, "sizeof Low is 32-bit");
	static_assert(digits / 8 == bytes, "digit and bytes ratio is wrong");

	//! empty constructor, does not even initialize to zero!
	constexpr two_word_int() = default;

	//! construct unit pair from lower and higher parts.
	constexpr two_word_int(const low_t& l, const high_t& h)
		: m_low(l)
		, m_high(h)
	{
	}
	constexpr two_word_int(const two_word_int&) = default;
	constexpr two_word_int(two_word_int&&) = default;
	constexpr explicit two_word_int(const uint32_t& a)
		: m_low(a)
		, m_high(0)
	{
	}

	constexpr two_word_int(const uint64_t& a)
		: m_low((low_t)(a & low_max()))
		, m_high((high_t)((a >> low_bits) & high_max()))
	{
		assert((a >> (low_bits + high_bits)) == 0);
	}
	constexpr two_word_int& operator=(const two_word_int&) = default;
	constexpr two_word_int& operator=(two_word_int&&) = default;

	constexpr operator uint64_t() const
	{
		return uint64_t { m_high } << low_bits | uint64_t { m_low };
	}

	constexpr two_word_int& operator++()
	{
		if (fgsaca_unlikely(m_low == low_max()))
			++m_high, m_low = 0;
		else
			++m_low;
		return *this;
	}

	constexpr two_word_int& operator--()
	{
		if (fgsaca_unlikely(m_low == 0))
			--m_high, m_low = (low_t)low_max();
		else
			--m_low;
		return *this;
	}

	constexpr two_word_int operator++(int)
	{
		two_word_int result = *this;
		if (fgsaca_unlikely(m_low == low_max()))
			++m_high, m_low = 0;
		else
			++m_low;
		return result;
	}

	constexpr two_word_int operator--(int)
	{
		two_word_int result = *this;
		if (fgsaca_unlikely(m_low == 0))
			--m_high, m_low = (low_t)low_max();
		else
			--m_low;
		return result;
	}

	constexpr two_word_int& operator+=(const two_word_int& b)
	{
		uint64_t add = uint64_t { m_low } + b.m_low;
		m_low = (low_t)(add & low_max());
		m_high = (high_t)(m_high + b.m_high + ((add >> low_bits) & high_max())); // TODO: remove if overflow cannot happen
		return *this;
	}

	constexpr two_word_int operator+(const two_word_int& b) const
	{
		uint64_t add = uint64_t { m_low } + b.m_low;
		return two_word_int(
			(low_t)(add & low_max()),
			(high_t)(m_high + b.m_high + ((add >> low_bits) & high_max())));
	}

	constexpr two_word_int& operator-=(const two_word_int& b)
	{
		uint64_t sub = uint64_t { m_low } - b.m_low;
		m_low = (low_t)(sub & low_max());
		m_high = (high_t)(m_high - b.m_high + ((sub >> low_bits) & high_max()));
		return *this;
	}

	constexpr two_word_int operator-(const two_word_int& b) const
	{
		uint64_t sub = m_low - uint64_t(b.m_low);
		return two_word_int(
			(low_t)(sub & low_max()),
			(high_t)(m_high - b.m_high + ((sub >> low_bits) & high_max())));
	}

	constexpr bool operator==(const two_word_int& b) const
	{
		return (m_low == b.m_low) && (m_high == b.m_high);
	}

	constexpr bool operator!=(const two_word_int& b) const
	{
		return (m_low != b.m_low) || (m_high != b.m_high);
	}

	constexpr bool operator<(const two_word_int& b) const
	{
		return (m_high < b.m_high) || (m_high == b.m_high && m_low < b.m_low);
	}

	constexpr bool operator<=(const two_word_int& b) const
	{
		return (m_high < b.m_high) || (m_high == b.m_high && m_low <= b.m_low);
	}

	constexpr bool operator>(const two_word_int& b) const
	{
		return (m_high > b.m_high) || (m_high == b.m_high && m_low > b.m_low);
	}

	constexpr bool operator>=(const two_word_int& b) const
	{
		return (m_high > b.m_high) || (m_high == b.m_high && m_low >= b.m_low);
	}

	constexpr two_word_int& operator>>=(uint8_t const& shift)
	{
		*this = (two_word_int) ((uint64_t) *this >> shift);
		return *this;
	}

#define uint64_binary_operator(x, cnst)                    \
	template <typename T>                                  \
	constexpr auto operator x(const T& b) cnst             \
	{                                                      \
		static_assert(std::is_convertible_v<T, two_word_int>); \
		static_assert(!std::is_same_v<T, two_word_int>);       \
		return *this x(two_word_int) b;                        \
	}

	uint64_binary_operator(==, const);

	uint64_binary_operator(!=, const);

	uint64_binary_operator(<, const);

	uint64_binary_operator(<=, const);

	uint64_binary_operator(>, const);

	uint64_binary_operator(>=, const);

	uint64_binary_operator(-, const);

	uint64_binary_operator(+, const);

	uint64_binary_operator(-=, );
	uint64_binary_operator(+=, );
#undef uint64_binary_operator

	//! return an two_word_int instance containing the smallest value possible
	static constexpr two_word_int min()
	{
		return two_word_int { std::numeric_limits<low_t>::min(),
			std::numeric_limits<high_t>::min() };
	}

	//! return an two_word_int instance containing the largest value possible
	static constexpr two_word_int max()
	{
		return two_word_int { std::numeric_limits<low_t>::max(),
			std::numeric_limits<high_t>::max() };
	}
} __attribute((packed));

using uint40_t = two_word_int<uint8_t>;

template<typename _high_t>
std::ostream& operator<<(std::ostream& out, const two_word_int<_high_t>& a)
{
	return out << (uint64_t)a;
}
