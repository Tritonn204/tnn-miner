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

#include "r12.hpp"

#include <numeric>
#include <algorithm>

namespace dc3 {

namespace r12 {

namespace  {

std::string::size_type highestR12BucketIndex(std::vector<std::string::size_type>::iterator firstIt,
                                           std::vector<std::string::size_type>::iterator lastIt,
                                           std::vector<std::string::size_type>::const_iterator strFirstIt,
                                           unsigned digitPlace){

    std::string::size_type index = *(strFirstIt + (*firstIt + digitPlace));

    for(std::vector<std::string::size_type>::iterator it = firstIt + 1; lastIt != it; ++it){

        if(index < *(strFirstIt + (*it + digitPlace))){

            index = *(strFirstIt + (*it + digitPlace));
        }
    }

    return index;
}

std::vector<std::string::size_type> r12CountingSort(std::vector<std::string::size_type>::iterator firstIt,
                                                    std::vector<std::string::size_type>::iterator lastIt,
                                                    std::vector<std::string::size_type>::const_iterator strFirstIt,
                                                    unsigned digitPlace){

    std::vector<std::string::size_type> countingBuckets(highestR12BucketIndex(firstIt, lastIt, strFirstIt, digitPlace)
                                                        + 1, 0);

    for(std::vector<std::string::size_type>::iterator it = firstIt; lastIt != it; ++it){
        ++countingBuckets[*(strFirstIt + (*it + digitPlace))];
    }

    std::vector<std::string::size_type> r12OrderOffset(std::count_if(countingBuckets.cbegin(), countingBuckets.cend(),
                                                                     [](std::string::size_type val){return val > 0;}), 0);

    std::copy_if(countingBuckets.cbegin(), countingBuckets.cend(), r12OrderOffset.begin(),
                 [](std::string::size_type val){return val > 0;});

    std::partial_sum(countingBuckets.cbegin(), countingBuckets.cend(), countingBuckets.begin());

    std::vector<std::string::size_type> tempR12Order(lastIt - firstIt);

    for(std::reverse_iterator<std::vector<std::string::size_type>::iterator> rIt(lastIt), rEndIt(firstIt);
        rEndIt != rIt; ++rIt){

        std::string::size_type digit = *(strFirstIt + (*rIt + digitPlace));

        tempR12Order[countingBuckets[digit] - 1] = *rIt;
        --countingBuckets[digit];
    }

    for(std::vector<std::string::size_type>::const_iterator it = tempR12Order.cbegin(), endIt = tempR12Order.cend();
        endIt != it; ++it, ++firstIt){

        *firstIt = *it;
    }

    return r12OrderOffset;
}

}

std::vector<std::string::size_type> r12Indexes(std::string::size_type numStrSize){

    std::string::size_type strSize = numStrSize - 2;
    std::vector<std::string::size_type> indexes(((strSize / 3) * 2) + (((strSize % 3) > 1) ? 1 : 0));

    std::string::size_type rI = 0;

    for(std::string::size_type i1 = 1; i1 < strSize; i1 +=3, ++rI){

        indexes[rI] = i1;
    }

    for(std::string::size_type i2 = 2; i2 < strSize; i2 +=3, ++rI){

        indexes[rI] = i2;
    }

    return indexes;
}

std::vector<std::string::size_type> indexToR12OrderMapping(std::vector<std::string::size_type>::const_iterator r12FirstIt,
                                                           std::vector<std::string::size_type>::const_iterator r12LastIt,
                                                           std::string::size_type numStrSize){

    std::vector<std::string::size_type> mappingArr(numStrSize, 0);

    std::string::size_type order = 1;

    for(; r12LastIt != r12FirstIt; ++r12FirstIt, ++order){

        mappingArr[*r12FirstIt] = order;
    }

    return mappingArr;
}

std::vector<std::string::size_type> r12NumString(const std::vector<std::string::size_type>& orderOfR12,
                                                 const std::vector<std::string::size_type>& offsetOfR12,
                                                 std::string::size_type numStrSize){

    std::vector<std::string::size_type> mappingArr(numStrSize, 0);

    std::vector<std::string::size_type>::const_iterator it = orderOfR12.cbegin();
    std::string::size_type order = 1;

    for(std::string::size_type offset : offsetOfR12){

        for(std::vector<std::string::size_type>::const_iterator lastIt = it + offset; lastIt != it; ++it){
            mappingArr[*it] = order;
        }

        ++order;
    }

    std::vector<std::string::size_type> r12NumStr = r12Indexes(numStrSize);

    for(std::vector<std::string::size_type>::iterator it = r12NumStr.begin(), endIt = r12NumStr.end(); endIt != it; ++it){

        std::string::size_type temp = mappingArr[*it];
        *it = temp;
    }

    return  r12NumStr;
}

std::pair<std::vector<std::string::size_type>, std::vector<std::string::size_type>> r12Order(
        const std::vector<std::string::size_type> &numStr){

    std::vector<std::string::size_type> r12Index = r12Indexes(numStr.size());

    std::vector<std::string::size_type> r12OrderOffset = r12CountingSort(r12Index.begin(), r12Index.end(),
                                                                         numStr.cbegin(), 0);

    if(r12OrderOffset.size() == r12Index.size()){
        return {std::move(r12OrderOffset), std::move(r12Index)};
    }

    for(unsigned digit = 1; digit < 3; ++digit){

        std::vector<std::string::size_type> digitWiseR12OrderOffset;
        digitWiseR12OrderOffset.reserve(r12OrderOffset.size());

        std::vector<std::string::size_type>::iterator it = r12Index.begin();

        for(std::string::size_type offset : r12OrderOffset){

            if(1 == offset){
                digitWiseR12OrderOffset.push_back(offset);
            }
            else{
                std::vector<std::string::size_type> tempR12OrderOffset = r12CountingSort(it, it + offset,
                                                                                         numStr.cbegin(), digit);

                digitWiseR12OrderOffset.insert(digitWiseR12OrderOffset.cend(), tempR12OrderOffset.cbegin(),
                                               tempR12OrderOffset.cend());
            }

            it += offset;
        }

        r12OrderOffset = std::move(digitWiseR12OrderOffset);

        if(r12OrderOffset.size() == r12Index.size()){
            break;
        }
    }

    return {std::move(r12OrderOffset), std::move(r12Index)};
}

}

}
