/*********************************************************************************
** MIT License
**
** Copyright (c) 2021 VIKAS AWADHIYA
**
** Permission is hereby granted, free of charge, to any person obtaining a copy
** of this software and associated documentation files (the "Software"), to deal
** in the Software without restriction, including without limitation the rights
** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
** copies of the Software, and to permit persons to whom the Software is
** furnished to do so, subject to the following conditions:

** The above copyright notice and this permission notice shall be included in all
** copies or substantial portions of the Software.

** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
** OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
** SOFTWARE.
*********************************************************************************/

#include "r0.hpp"
#include "r12.hpp"

#include <numeric>
#include <algorithm>

namespace dc3 {

namespace r0 {

namespace  {

std::string::size_type highestR0BucketIndex(
        std::vector<std::pair<std::string::size_type, std::string::size_type>>::iterator firstIt,
        std::vector<std::pair<std::string::size_type, std::string::size_type>>::iterator lastIt){

    std::string::size_type index = firstIt->first;
    ++firstIt;

    for(; lastIt != firstIt; ++firstIt){

        if(index < firstIt->first){

            index = firstIt->first;
        }
    }

    return index;
}

std::vector<std::string::size_type> r0CountingSort(
        std::vector<std::pair<std::string::size_type, std::string::size_type>>::iterator firstIt,
        std::vector<std::pair<std::string::size_type, std::string::size_type>>::iterator lastIt,
        bool orderOffsetRequires){

    std::vector<std::string::size_type> countingBuckets(highestR0BucketIndex(firstIt, lastIt) + 1, 0);

    for(std::vector<std::pair<std::string::size_type, std::string::size_type>>::iterator it = firstIt; lastIt != it;
        ++it){

        ++countingBuckets[it->first];
    }

    std::vector<std::string::size_type> r0OrderOffset{};

    if(orderOffsetRequires){
        r0OrderOffset = std::vector<std::string::size_type>(std::count_if(countingBuckets.cbegin(), countingBuckets.cend(),
                                                                        [](std::string::size_type val){return val > 0;}));

        std::copy_if(countingBuckets.cbegin(), countingBuckets.cend(), r0OrderOffset.begin(),
                     [](std::string::size_type val){return val > 0;});
    }

    std::partial_sum(countingBuckets.cbegin(), countingBuckets.cend(), countingBuckets.begin());

    std::vector<std::string::size_type> tempR0Order(lastIt - firstIt);

    for(std::reverse_iterator<std::vector<std::pair<std::string::size_type, std::string::size_type>>::iterator> rIt(lastIt),
        rEndIt(firstIt); rEndIt != rIt; ++rIt){

        tempR0Order[countingBuckets[rIt->first] - 1] = rIt->second;
        --countingBuckets[rIt->first];
    }

    for(std::vector<std::string::size_type>::const_iterator it = tempR0Order.cbegin(), endIt = tempR0Order.cend();
        endIt != it; ++it, ++firstIt){

        firstIt->second = *it;
    }

    return r0OrderOffset;
}

void updateR0InexesForSecondRoundOfSort(
        std::vector<std::pair<std::string::size_type, std::string::size_type>>::iterator firstIt,
        std::vector<std::pair<std::string::size_type, std::string::size_type>>::iterator lastIt,
        const std::vector<std::string::size_type>& r12Order,
        std::string::size_type numStrSize){

    std::vector<std::string::size_type> mappingArr = r12::indexToR12OrderMapping(r12Order.cbegin(), r12Order.cend(),
                                                                                numStrSize);
    for(; lastIt != firstIt; ++firstIt){

        firstIt->first = mappingArr[firstIt->second + 1];
    }
}



}

std::vector<std::string::size_type> r0Order(const std::vector<std::string::size_type>& r12Order,
                                            const std::vector<std::string::size_type>& numStr){

    std::vector<std::pair<std::string::size_type, std::string::size_type>> r0Index((numStr.size() - r12Order.size()) - 2);

    std::string::size_type i = 0;

    for(std::vector<std::pair<std::string::size_type, std::string::size_type>>::iterator it = r0Index.begin(),
        endIt = r0Index.end(); endIt != it; ++it, i += 3){

        it->first = numStr[i];
        it->second = i;
    }

    std::vector<std::string::size_type> r0OrderOffset = r0CountingSort(r0Index.begin(), r0Index.end(), true);

    if(r0OrderOffset.size() != r0Index.size()){

        updateR0InexesForSecondRoundOfSort(r0Index.begin(), r0Index.end(), r12Order, numStr.size());

        std::vector<std::pair<std::string::size_type, std::string::size_type>>::iterator it = r0Index.begin();

        for(std::string::size_type offset : r0OrderOffset){

            if(offset > 1){
                r0CountingSort(it, it + offset, false);
            }

            it += offset;
        }
    }

    std::vector<std::string::size_type> orderOfR0;
    orderOfR0.reserve(r0Index.size());

    for(std::vector<std::pair<std::string::size_type, std::string::size_type>>::const_iterator it = r0Index.cbegin(),
        endIt = r0Index.cend(); endIt != it; ++it){

        orderOfR0.push_back(it->second);
    }

    return orderOfR0;
}

}

}