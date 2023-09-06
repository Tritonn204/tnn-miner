#include "cuda_runtime.h"
#include <stdint.h>
#include <iostream>

//Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

#include "cuda_helpers.cuh"

typedef uint8_t U8;
typedef uint32_t U32;

//Pack 4 U8's into an int
__device__ void pack(const U8 * data, int * keys, int n){
	auto r = thrust::counting_iterator<int>(0);

	thrust::for_each(thrust::device, r, r + n, [=] __device__(int i) {
		
		int packed = data[i];
		
		packed <<= 8;
		if (i + 1 < n)
			packed |= data[i+1];
		packed <<= 8;
		if (i + 2 < n)
			packed |= data[i + 2];
		packed <<= 8;
		if (i + 3 < n)
			packed |= data[i + 3];
		
		keys[i] = packed;

	});
}

__device__ void mark_head(int * keys, U8 * buckets, int n){

	auto r = thrust::counting_iterator<int>(0);

	thrust::for_each(thrust::device, r, r + n, [=] __device__(int i) {
		//Already marked - don't need to do anything
		if (buckets[i] == 1){
			return;
		}
		//First item is always head
		else if (i == 0){
			buckets[i] = 1;
		}
		//Is different than previous item - must be a bucket head
		else if (keys[i] != keys[i - 1]){
			buckets[i] = 1;
		}
	});
}

__device__ void get_rank(U8 * buckets, int * b_scan, int * rank, int * sa, int n){

	//Scan bucket heads
	//Copy buckets into rank before we scan it into b_scan - scanning 8 bit types creates problems
	thrust::copy(thrust::device, buckets, buckets + n, rank);
	thrust::inclusive_scan(thrust::device, rank, rank + n, b_scan);

	//Calculate rank - stores rank inverse to the suffix array
	// e.g. rank[3] stores the bucket position of sa[?] = 3

	auto r = thrust::counting_iterator<int>(0);

	thrust::for_each(thrust::device, r, r + n, [=] __device__(int i) {
		int suffix = sa[i];
		rank[suffix] = b_scan[i];
	});

}

__device__ void get_sort_keys(int * keys, int * rank, int * sa, U8 * buckets, int step, int n){

	auto r = thrust::counting_iterator<int>(0);

	thrust::for_each(thrust::device, r, r + n, [=] __device__(int i) {
		//Check if already sorted
		//If is last item - just need to check its flag
		if (buckets[i] == 1 && i == n - 1)
			return;
		//Otherwise, if the current item and its next item are flagged, current item must be already sorted
		else if (buckets[i] == 1 && buckets[i + 1])
			return;

		//Set sort keys
		int next_suffix = sa[i] + step;
		//Went off end of string - must be lexicographically less than rest of bucket
		if (next_suffix >= n)
			keys[i] = -next_suffix;

		//Else set sort key to rank of next suffix
		else
			keys[i] = rank[next_suffix];
	});
}


//We have to do a 2 pass sort here to get a "segmented sort"
__device__ void sort_sa(int * keys, int * b_scan, int * sa, int n){
	stable_sort_by_key(thrust::device, keys, keys + n, thrust::make_zip_iterator(thrust::make_tuple(sa, b_scan)));
	stable_sort_by_key(thrust::device, b_scan, b_scan + n, thrust::make_zip_iterator(thrust::make_tuple(sa, keys)));
}

__device__ int device_sa(const unsigned char *data_in, int *sa_in, int n){
	
	// try{
		int* sa = new int[n];

		//Init suffix array
		thrust::sequence(thrust::device, sa, sa + n);

		int* keys = new int[n];
		U8* buckets = new U8[n];
		int* b_scan = new int[n];
		int* rank = new int[n];

		//Pack 4 bytes into keys so we can radix sort to H order 4 before prefix doubling
		pack(data_in, keys, n);
		
		//Radix sort as unsigned 
		//We have to cast keys to a raw pointer then to a thrust::device_ptr to convince thrust its unsigned
		stable_sort_by_key(thrust::device, keys, keys + n, sa);

		int step = 4;
		//Begin prefix doubling loop - runs at most log(n) times
		while (true){

			//Mark bucket heads
			mark_head(keys, buckets, n);

			//Check if we are done, i.e. every item is a bucket head
			int result = reduce(thrust::device, buckets, buckets + n, INT_MAX, thrust::minimum<int>());
			if (result == 1) break;

			//Get rank of suffixes
			get_rank(buckets, b_scan, rank, sa, n);

			//Use rank as new sort keys
			get_sort_keys(keys, rank, sa, buckets, step, n);

			//Sort
			sort_sa(keys, b_scan, sa, n);
			/*
			std::cout << "-----\n";
			print("SA", sa);
			print("Keys", keys);
			print("Buckets", buckets);
			print("rank", rank);
			std::cout << "-----\n";
			*/
			step *= 2;

			//Just in case, check for infinite loop
			if (step < 0){
				printf("Error: Prefix doubling infinite loop.\n");
				return 1;
			}
		}

		//Copy SA back to host
		memcpy(sa_in, sa, sizeof(int)*n);
		delete[] sa;
		delete[] keys;
		delete[] buckets;
		delete[] b_scan;
		delete[] rank;
	// }
	// catch (thrust::system_error &e)
	// {
	// 	printf("CUDA error: %s\n", e.what());
	// }

	return 0;
}