/*
 * SOrter.h
 *
 *  Created on: Jul 21, 2012
 *      Author: marius
 */

#ifndef SORTER_H_
#define SORTER_H_

template<class Word>
class Sorter {
public:
	virtual ~Sorter() {
	}
	virtual void sort(uint length, uint *data, Word *key) {
	}
};

#endif /* SORTER_H_ */