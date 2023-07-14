#pragma once
#include <vector>
#include <thread>
#include <boost/core/noinit_adaptor.hpp>

// #psais::detail
namespace psais::detail {

template<typename T>
std::reference_wrapper<std::remove_reference_t<T>> wrapper(T& t) { return std::ref(t); }

template<typename T>
T&& wrapper(std::remove_reference_t<T>&& t) { return std::move(t); }

} //namespace psais::detail

// #parallel_do
namespace psais::utility {

template <typename Func, typename ... Args>
void parallel_do (
	std::integral auto n_jobs,
	std::integral auto n_threads,
	Func&& func,
	Args&& ...args
) {
	std::vector<std::jthread> threads;
	threads.reserve(n_jobs);
	auto counts = n_jobs / n_threads;
	auto remain = n_jobs % n_threads;

	for (decltype(n_jobs) tid = 0, L = 0; tid < n_threads; tid++) {
		auto block_size = counts + (tid < remain);
		if (block_size == 0) break;

		auto R = std::min(n_jobs, L + block_size);
		threads.emplace_back(std::forward<Func>(func), L, R, tid, psais::detail::wrapper<Args>(args)...);
		L = R;
	}
}

// #parallel_init
template <std::ranges::input_range R>
void parallel_init(R&& r, const auto& value) {
	std::fill(std::execution::par_unseq, r.begin(), r.end(), value);
}

// #parallel_prefix_sum
template <typename Vec>
void parallel_prefix_sum(
	Vec& vec,
	std::integral auto n_threads
) {
	auto n_jobs = vec.size();

	using T = Vec::value_type;

	std::vector<T, boost::noinit_adaptor<std::allocator<T>>> block_prefix_sum(n_jobs);
	std::vector<T> block_last_pos(n_threads, 0);
	psais::utility::parallel_do(vec.size(), (size_t) n_threads,
		[&](auto L, auto R, auto tid) {
			if (L == R)
				return ;

			block_prefix_sum[L] = vec[L];
			for (auto i = L + 1; i < R; i++)
				block_prefix_sum[i] = vec[i] + block_prefix_sum[i - 1];
			block_last_pos[tid] = R - 1;
		}
	);

	for (decltype(n_threads) i = 1; i < n_threads; i++) {
		if (block_last_pos[i] == 0)
			break;
		block_prefix_sum[block_last_pos[i]] += block_prefix_sum[block_last_pos[i - 1]];
	}

	psais::utility::parallel_do(n_jobs, (size_t) n_threads,
		[&](auto L, auto R, auto tid) {
			if (L == R)
				return ;

			auto offset = (tid == 0 ? 0 : block_prefix_sum[L - 1]);
			for (auto i = L; i < R - 1; i++)
				vec[i] = offset + block_prefix_sum[i];
			vec[R - 1] = block_prefix_sum[R - 1];
		}
	);
}

// #parallel_take_if
template <typename Compare, typename Project>
auto parallel_take_if(
	std::integral auto n_jobs,
	std::integral auto n_threads,
	std::ranges::random_access_range auto& ret,
	Compare &&compare,
	Project &&project
) {
	std::vector<size_t> buf(n_threads, 0);

	psais::utility::parallel_do(n_jobs, n_threads,
		[&](auto L, auto R, auto tid) {
			for (auto i = L; i < R; i++)
				buf[tid] += !!compare(i);
		}
	);

	std::vector<size_t> offset(n_threads + 1, 0);
	for (decltype(n_threads) i = 0; i < n_threads; i++)
		offset[i + 1] = offset[i] + buf[i];

	if constexpr (requires { ret.resize(0); }) {
		if (ret.size() < offset.back()) {
			ret.resize(offset.back());
		}
	}

	psais::utility::parallel_do(n_jobs, n_threads,
		[&](auto L, auto R, auto tid) {
			for (size_t i = L; i < R; i++)
				if (compare(i))
					ret[offset[tid]++] = project(i);
		}
	);
}

} //namespace psais::utility
