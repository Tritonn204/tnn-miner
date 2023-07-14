#pragma once
#include <thread>
#include <queue>
#include <functional>

namespace psais::utility {

class ThreadPool {
  public:
	ThreadPool(int cnt) : thread_cnt(cnt) {
		auto init_thread = [this](std::stop_token tok) {
			auto job = std::packaged_task<void()>{};

			while (!tok.stop_requested()) {
				if (auto lock = std::scoped_lock(q_m); jobs.empty()) {
					continue;
				} else {
					job = std::move(jobs.front());
					jobs.pop();
				}
				job();
			}
		};

		for (int i = 0; i < thread_cnt; i++) {
			threads.emplace_back(init_thread);
		}
	}

	template <typename F, typename ... Args>
	requires std::invocable<F, Args...>
	auto enqueue(F&& f, Args&& ... args) {
		using res_type = typename std::invoke_result_t<F, Args...>;

		auto task = std::packaged_task<res_type()>(
			std::bind(std::forward<F>(f), std::forward<Args>(args)...)
		);
		auto res = task.get_future();

		auto lock = std::unique_lock(q_m);
		jobs.emplace(std::move(task));
		return res;
	}

  private:
	std::queue<std::packaged_task<void()>> jobs;
	std::vector<std::jthread> threads;
	std::mutex q_m;

	int thread_cnt;
};

} //namespace psais::utility
