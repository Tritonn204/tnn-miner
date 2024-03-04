#pragma once
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>


const int ITEMS_PER_THREAD = 20;
const int BLOCK_SIZE = 256;

#define safe_cuda(ans) { throw_on_cuda_error((ans), __FILE__, __LINE__); }

void throw_on_cuda_error(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess)
	{
		std::cout << file;
		std::cout << line;
		std::stringstream ss;
		ss << file << "(" << line << ")";
		std::string file_and_line;
		ss >> file_and_line;
		throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
	}
}


__device__ int tid()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

//Utility function: rounds up integer division.
//Can overflow on large numbers and does not work with negatives
__host__ __device__ int div_round_up(int a, int b){
	return (a + b - 1) / b;
}

//Calculate number of cuda blocks we need
int get_num_blocks(int n){

	return div_round_up(n, BLOCK_SIZE*ITEMS_PER_THREAD);
}

template <typename T>
void print(const thrust::device_vector<T>& v)
{
	for (auto elem : v)
		std::cout << " " << (int)elem;
	std::cout << "\n";
}

template <typename T>
void print(char *label, const thrust::device_vector<T>& v, int max = 10)
{
	int i = 0;
	std::cout << label << ":\n";
	for (auto elem : v){
		std::cout << " " << (int)elem;
		i++;
		if (i >= max) break;
	}
	std::cout << "\n";
}

class range {
public:
	class iterator {
		friend class range;
	public:
		__host__ __device__
		long int operator *() const { return i_; }
		__host__ __device__
		const iterator &operator ++() { i_ += step_; return *this; }
		__host__ __device__
		iterator operator ++(int) { iterator copy(*this); i_ += step_; return copy; }

		__host__ __device__
		bool operator ==(const iterator &other) const { return i_ >= other.i_; }
		__host__ __device__
		bool operator !=(const iterator &other) const { return i_ < other.i_; }

		__host__ __device__
		void step(int s){ step_ = s; }
	protected:
		__host__ __device__
		iterator(long int start) : i_(start) { }

	//private:
	public:
		unsigned long i_;
		int step_ = 1;
	};

	__host__ __device__
	iterator begin() const { return begin_; }
	__host__ __device__
	iterator end() const { return end_; }
	__host__ __device__
	range(long int  begin, long int end) : begin_(begin), end_(end) {}
	__host__ __device__
	void step(int s) { begin_.step(s); }
private:
	iterator begin_;
	iterator end_;
};

template <typename T>
__device__ range grid_stride_range(T begin, T end){
	begin += blockDim.x * blockIdx.x + threadIdx.x;
	range r(begin, end);
	r.step(gridDim.x * blockDim.x);
	return r;
}

//Converts device_vector to raw pointer
template <typename T>
T * raw(thrust::device_vector<T>& v){
	return raw_pointer_cast(v.data());
}