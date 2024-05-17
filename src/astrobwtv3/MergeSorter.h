/*
 * MergeSorter.h
 *
 *  Created on: Jul 19, 2012
 *      Author: marius
 */

#ifndef MERGESORTER_H_
#define MERGESORTER_H_
#include "radix_utils.h"
#include <vector>

template<class Word>
class MergeSorter: public Sorter<Word> {
private:
	static const uint insertSortThreshold = 32;
	static const uint tileSize = 0x4000;
	std::vector<Word> destK;
	std::vector<uint> destS;

public:
	MergeSorter() {

	}

	~MergeSorter() {

	}

	void sort(uint length, uint *sa, Word *key) {
		if (length <= insertSortThreshold) {
			insertSort(sa, length, key);
			return;
		}

		mergeSort(length, sa, key);
		return;

		/*
		 uint s = 0;
		 for (; s + tileSize < length; s += tileSize)
		 mergeSort(tileSize, sa + s, key + s);
		 if (length - s > 1)
		 mergeSort(length - s, sa + s, key + s);

		 if (length > tileSize)
		 iterativeMergeSort(length, sa, key, tileSize);
		 */
	}

	void mergeSort(uint length, uint *sa, Word*key) {
		//	uint tiles = (length / insertSortThreshold) + ((length
		//			% insertSortThreshold) ? 1 : 0);

		//	uint insSortThr = (logNextPowOfTwo(tiles) & 1) // odd
		//	? (insertSortThreshold >> 1)
		//			: insertSortThreshold;
		uint insSortThr = insertSortThreshold;

		uint s = 0;
		for (; s + insSortThr < length; s += insSortThr)
			insertSort(sa + s, insSortThr, key + s);
		if (length - s > 1)
			insertSort(sa + s, length - s, key + s);

		iterativeMergeSort(length, sa, key, insSortThr);
	}

	void iterativeMergeSort(uint length, uint *sa, Word*key,
			int initialTileSize) {
		// iterative merge sort
		destS.reserve(length);
		destK.reserve(length);
		uint *dS = &destS[0];
		Word *dK = &destK[0];
		for (uint bSize = initialTileSize; bSize < length; bSize <<= 1) {
			int n = 0;
			for (uint s = 0; s < length; s += (bSize << 1)) {
				uint i = s;
				uint m = s + bSize;
				if (m > length)
					m = length;
				uint j = m;
				uint e = m + bSize;
				if (e > length)
					e = length;
				for (; i < m && j < e; ++n) {
					if (key[i] <= key[j]) {
						dK[n] = key[i];
						dS[n] = sa[i];
						++i;
					} else {
						dK[n] = key[j];
						dS[n] = sa[j];
						++j;
					}
				}
				for (; i < m; ++n, ++i) {
					dK[n] = key[i];
					dS[n] = sa[i];
				}
				for (; j < e; ++n, ++j) {
					dK[n] = key[j];
					dS[n] = sa[j];
				}
			}
			uint *tmp = sa;
			sa = dS;
			dS = tmp;
			Word *tmpk = key;
			key = dK;
			dK = tmpk;
		}
		if (sa == &destS[0]) {
			//		cout << "Avoidable " << length << endl;
			memcpy(dK, key, length * sizeof(Word));
			memcpy(dS, sa, length * sizeof(uint));
		}
	}

};

#endif /* MERGESORTER_H_ */