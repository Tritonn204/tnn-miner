#pragma once

#include "fgsaca_common.hpp"
#include "fgsaca_util.hpp"

#include <functional>

namespace fgsaca_internal {

template <typename Ix, typename IIx>
Ix process_phase2(
	Ix* FGSACA_RESTRICT SA,
	IIx* FGSACA_RESTRICT gstarts,
	Ix& FGSACA_RESTRICT s,
	const IIx isa,
	const IIx p // p = PREV[s]
)
{
	if (is_marked(isa))
		return false;

	const Ix sr = is_marked(isa) ? unmark(isa) : gstarts[isa]++;

	const bool msb = (unmark(p) + 1u < s or fgsaca_unlikely(unmark(p) > s));
	SA[sr] = msb ? mark(s) : s;

	s = unmark(p);
	return s != p;
}

template <typename Ix, typename IIx, bool bbwt>
void phase_2_induce(
	Ix* FGSACA_RESTRICT SA,
	IIx* FGSACA_RESTRICT gstarts,
	const IIx* FGSACA_RESTRICT mem,
	const idx_comp<0, 2> ISA,
	const idx_comp<1, 2> PREV,
	const Ix n,
	const Ix ngroups)
{
	const auto process = std::bind_front(process_phase2<Ix, IIx>, SA, gstarts);
	constexpr Ix undefined = std::numeric_limits<Ix>::max();

	Ix i = 0;
	if constexpr (false) { // trivial Phase 2
		for (; i < n; i++) {
			auto s = SA[i];

			if (not is_marked(s)) {
				continue;
			}
			SA[i] = unmark(s);
			ASSUME(unmark(s) > 0);
			ASSUME(unmark(s) < (n + bbwt));
			s = unmark(s) - 1;
			while (process(s, mem[ISA[s]], mem[PREV[s]])) { }
		}
	} else if (false) { // alternative prefetching for BFS
		constexpr Ix max_size = 1 << 8;
		std::tuple<Ix, IIx, Ix> todo[max_size + 1]; // (s, isa, prev)
		Ix cur_size = 0, fetched = 0;
		do {
			while (fetched < cur_size) {
				const Ix s = std::get<0>(todo[fetched]);
				todo[fetched] = std::make_tuple(s, mem[ISA[s]], mem[PREV[s]]);
				fetched++;
			}
			fetched = 0;

			Ix new_size = 0;
			for (Ix i = 0; i < cur_size; i++) {
				if (i + 16 < cur_size) {
					const auto isa = std::get<1>(todo[i + 16]);
					if (!is_marked(isa))
						fgsaca_prefetchw(&gstarts[isa]);
				}
				auto [s, isa, p] = todo[i];

				if (process(s, isa, p)) {
					fgsaca_prefetch(&mem[ISA[s]]);
					std::get<0>(todo[new_size++]) = s;
					if (new_size >= 16) {
						const auto s = std::get<0>(todo[fetched]);
						todo[fetched] = std::make_tuple(s, mem[ISA[s]], mem[PREV[s]]);
						fetched++;
					}
				}
			}
			for (; i < n && new_size < max_size; i++) {
				auto s = SA[i];
				if (s == undefined)
					break;
				if (!is_marked(s))
					continue;
				SA[i] = unmark(s);

				std::get<0>(todo[new_size++]) = unmark(s) - 1;
				fgsaca_prefetch(&mem[ISA[unmark(s) - 1]]);
				if (new_size >= 16) {
					const auto s = std::get<0>(todo[fetched]);
					todo[fetched] = std::make_tuple(s, mem[ISA[s]], mem[PREV[s]]);
					fetched++;
				}
			}

			cur_size = new_size;
		} while (cur_size > 0);
	} else if (ngroups >= 128) { // use bfs with prefetching
		constexpr Ix max_size = (1 << 10u) * 4 / sizeof(Ix);
		std::tuple<Ix, IIx, Ix> todo[max_size + 1]; // (s, isa, prev)
		Ix cur_size = 0;
		constexpr Ix prefetch_distance = 32;
		do {
			{ // fetch first <= prefetch_distance ISAs
				const Ix sz = std::min(cur_size, prefetch_distance);
				if (cur_size >= prefetch_distance / 4)
					for (Ix i = 0; i < sz; i++)
						fgsaca_prefetch(&mem[ISA[get<0>(todo[i])]]);

				for (Ix i = 0; i < sz; i++) {
					const Ix s = get<0>(todo[i]);
					const IIx isa = mem[ISA[s]];
					const Ix p = mem[PREV[s]];
					get<1>(todo[i]) = isa;
					get<2>(todo[i]) = p;
					if (!is_marked(isa))
						fgsaca_prefetchw(&gstarts[isa]);
				}
			}
			Ix new_size = 0;

			ptrdiff_t k, j;
			for (k = 0, j = cur_size - (ptrdiff_t)2 * prefetch_distance; k < j; k++) {
				fgsaca_prefetch(&mem[ISA[get<0>(todo[k + 2 * prefetch_distance])]]);

				{
					const Ix s = get<0>(todo[k + prefetch_distance]);
					const IIx isa = mem[ISA[s]];
					const Ix p = mem[PREV[s]];
					get<1>(todo[k + prefetch_distance]) = isa;
					get<2>(todo[k + prefetch_distance]) = p;
					if (!is_marked(isa))
						fgsaca_prefetchw(&gstarts[isa]);
				}
				auto [s, isa, p] = todo[k];
				if (process(s, isa, p)) {
					get<0>(todo[new_size++]) = s;
				}
			}
			for (j += prefetch_distance; k < j; k++) {
				{
					const Ix s = get<0>(todo[k + prefetch_distance]);
					const IIx isa = mem[ISA[s]];
					const Ix p = mem[PREV[s]];
					get<1>(todo[k + prefetch_distance]) = isa;
					get<2>(todo[k + prefetch_distance]) = p;
					if (!is_marked(isa))
						fgsaca_prefetchw(&gstarts[isa]);
				}
				auto [s, isa, p] = todo[k];
				if (process(s, isa, p)) {
					get<0>(todo[new_size++]) = s;
				}
			}
			for (j += prefetch_distance; k < j; k++) {
				auto [s, isa, p] = todo[k];
				if (process(s, isa, p)) {
					get<0>(todo[new_size++]) = s;
				}
			}

			// add new elements to back of queue
			for (; i < n && new_size < max_size; i++) {
				auto s = SA[i];
				if (s == undefined)
					break;
				if (!is_marked(s))
					continue;
				SA[i] = unmark(s);

				get<0>(todo[new_size++]) = unmark(s) - 1;
			}

			cur_size = new_size;
		} while (cur_size > 0);
	} else { // dfs with prefetching
		constexpr Ix prefetch_distance = 8;

		uint64_t mask = 0;
		if (n > 3 * prefetch_distance)
			for (Ix j = n - 3 * prefetch_distance; i < j; i++) {
				if (const Ix s = SA[i + 3 * prefetch_distance]; s != undefined and is_marked(s)) {
					const Ix k = unmark(s) - 1;
					fgsaca_prefetch(&mem[ISA[k]]);
					mask |= bit<uint64_t>(2 * prefetch_distance);
				}
				if ((mask & bit<uint64_t>(prefetch_distance)) != 0) {
					const Ix k = unmark(SA[i + 2 * prefetch_distance]) - 1;
					const auto isa = mem[ISA[k]];
					if (is_marked(isa)) {
						SA[i + 2 * prefetch_distance] = unmark(SA[i + 2 * prefetch_distance]);
					} else {
						fgsaca_prefetchw(&gstarts[isa]);
					}

					const auto p = unmark(mem[PREV[k]]);
					if (p < k)
						fgsaca_prefetch(&mem[ISA[p]]);
				}

				mask >>= 1;
				auto s = SA[i];
				ASSUME(s != undefined);
				if (!is_marked(s)) {
					continue;
				}
				s = unmark(s) - 1;
				IIx isa = mem[ISA[s]];
				SA[i] = s + 1;
				while (process(s, isa, mem[PREV[s]]))
					isa = mem[ISA[s]];
			}
		for (; i < n; i++) {
			auto s = SA[i];
			ASSUME(s != undefined);
			if (!is_marked(s)) {
				continue;
			}
			s = unmark(s) - 1;
			IIx isa = mem[ISA[s]];
			SA[i] = s + 1;
			while (process(s, isa, mem[PREV[s]]))
				isa = mem[ISA[s]];
		}
	}
}

template <typename Ix, typename mem_ix_t, bool bbwt>
void fgsaca_insert_singletons_phase2(
	Ix* FGSACA_RESTRICT SA,
	mem_ix_t* FGSACA_RESTRICT ISA_PREV,
	const Ix n,
	const when_t<bbwt, Ix> last_pos)
{
	memset(SA, 0xFF, sizeof(SA[0]) * n); // set to 0xff (for prefetching)
	for (Ix i = (bbwt ? 1 : 0); i < n; i++) {
		if (const auto isa = ISA_PREV[2 * i]; is_marked(isa)) { // singleton
			const auto p = ISA_PREV[2 * i + 1];
			ISA_PREV[2 * i + 1] = unmark(p); // set as "not-last-child" since parent must also be singleton

			const bool msb = unmark(p) + 1u < i or fgsaca_unlikely(unmark(p) == n);
			SA[unmark(isa)] = i | (msb ? mark<Ix>(0) : 0u);
		}
	}
	if constexpr (bbwt) {
		// insert n where last_pos should have been
		SA[last_pos] = mark(n);
	}
}

template <typename Ix, typename mem_ix_t, bool bbwt>
void fgsaca_phase2(
	Ix* FGSACA_RESTRICT SA,
	mem_ix_t* FGSACA_RESTRICT gstarts,
	mem_ix_t* FGSACA_RESTRICT mem, // mem[2*i] = ISA[i], mem[2*i+1] = PREV[i]
	const size_t n,
	const size_t ngroups,
	const when_t<bbwt, size_t> last_pos)
{
	const idx_comp<0, 2> ISA {};
	const idx_comp<1, 2> PREV {};

	fgsaca_insert_singletons_phase2<Ix, mem_ix_t, bbwt>(SA, mem, n, last_pos);

	if constexpr (not bbwt) {
		const auto process = std::bind_front(process_phase2<Ix, mem_ix_t>, SA, gstarts);
		{ // handle P_n explicitly
			Ix s = n - 1;
			while (process(s, mem[ISA[s]], mem[PREV[s]])) { }
		}
	}

	phase_2_induce<Ix, mem_ix_t, bbwt>(
		SA,
		gstarts,
		mem,
		ISA,
		PREV,
		n,
		ngroups);
}

} // fgsaca_internal
