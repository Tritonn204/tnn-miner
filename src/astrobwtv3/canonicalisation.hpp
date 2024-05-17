#pragma once

#include "fgsaca_common.hpp"

template <typename T>
size_t
canonicalise(
	const T* FGSACA_RESTRICT s,
	const size_t n)
{
	size_t cur_min = 0;

	// invariant: [0..i) \ cur_min is not the minimum rotation
	for (size_t i = 1; i < n;) {
		auto c1 = s[i], c2 = s[cur_min];
		if (c1 < c2) [[unlikely]] {
			cur_min = i++;
			continue;
		}
		if (c1 > c2) {
			i++;
			continue;
		}
		size_t l = 0;
		while (i + l < n and (c1 = s[i + l]) == (c2 = s[cur_min + l]))
			l++;
		if (i + l == n) [[unlikely]] {
			size_t ai = 0, ac = cur_min + l;
			while (fgsaca_likely(ac < n) and (c1 = s[ai]) == (c2 = s[ac]))
				ai++, ac++, l++;
			if (ac == n) [[unlikely]] {
				ac = 0;
				while (ac != cur_min and (c1 = s[ai]) == (c2 = s[ac]))
					ai++, ac++, l++;
			}
		}
		if (c1 < c2) {
			// i is the new optimum
			// [cur_min..cur_min + l] can't be the optimum
			const size_t new_i = std::max(i + 1, cur_min + l + 1);
			cur_min = i;
			i = new_i;
		} else {
			// [i..i+l] can't be the optimum
			i += l + 1;
		}
	}

	return cur_min;
}