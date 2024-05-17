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

#include "dc3.hpp"
#include "r0.hpp"
#include "r12.hpp"

#include <iterator>

#include <numeric>
#include <algorithm>

namespace dc3 {


std::vector<std::string::size_type> mergeR0AndR12Orders(const std::vector<std::string::size_type>& r0Order,
                                                        const std::vector<std::string::size_type>& r12Order,
                                                        const std::vector<std::string::size_type>& numStr){

    std::vector<std::string::size_type> mappingArr = r12::indexToR12OrderMapping(r12Order.cbegin(), r12Order.cend(),
                                                                            numStr.size());

    std::vector<std::string::size_type> rOrder;
    rOrder.reserve(numStr.size() - 2);

    std::vector<std::string::size_type>::const_iterator r0It = r0Order.cbegin();
    std::vector<std::string::size_type>::const_iterator r12It = r12Order.cbegin();

    if(numStr.size() - 3 == *r0It){
        ++r0It;
    }
    else{
        ++r12It;
    }

    for(std::vector<std::string::size_type>::const_iterator r0EndIt = r0Order.cend(), r12EndIt = r12Order.cend();
        (r0EndIt != r0It && r12EndIt != r12It);){

        unsigned rNum = 0;

        if(numStr[*r0It] == numStr[*r12It]){

            if(2 == *r12It % 3){

                if(numStr[*r0It + 1] == numStr[*r12It + 1]){
                    rNum = mappingArr[*r0It + 2] < mappingArr[*r12It + 2] ? 0 : 1;
                }
                else{
                    rNum = numStr[*r0It + 1] < numStr[*r12It + 1] ? 0 : 1;
                }
            }
            else{
                rNum = (mappingArr[*r0It + 1] < mappingArr[*r12It + 1]) ? 0 : 1;
            }
        }
        else{
            rNum = numStr[*r0It] < numStr[*r12It] ? 0 : 1;
        }

        if(0 == rNum){
            rOrder.push_back(*r0It);
            ++r0It;
        }
        else{
            rOrder.push_back(*r12It);
            ++r12It;
        }
    }

    if(r0Order.cend() != r0It){
        rOrder.insert(rOrder.cend(), r0It, r0Order.cend());
    }
    else{
        rOrder.insert(rOrder.cend(), r12It, r12Order.cend());
    }

    return rOrder;
}

std::vector<std::string::size_type> uniqueR12Order(const std::vector<std::string::size_type>& numStr){

    std::pair<std::vector<std::string::size_type>, std::vector<std::string::size_type>> orderR12Info = r12::r12Order(
                numStr);

    if(orderR12Info.first.size() == orderR12Info.second.size()){
        return orderR12Info.second;
    }

    std::vector<std::string::size_type> r12NumStr = r12::r12NumString(orderR12Info.second, orderR12Info.first,
                                                                      numStr.size());

    r12NumStr.insert(r12NumStr.cend(), {0, 0, 0});

    std::vector<std::string::size_type> secondLevelR12Order = uniqueR12Order(r12NumStr);

    std::vector<std::string::size_type> secondLevelROrder = mergeR0AndR12Orders(
                r0::r0Order(secondLevelR12Order, r12NumStr), secondLevelR12Order, r12NumStr);

    std::vector<std::string::size_type> orderMappingArr = r12::r12Indexes(numStr.size());

    for(std::vector<std::string::size_type>::iterator it = secondLevelROrder.begin(), endIt = secondLevelROrder.end();
        endIt != it; ++it){

        std::string::size_type temp = orderMappingArr[*it];
        *it = temp;
    }

    return secondLevelROrder;
}


std::vector<std::string::size_type> suffixArray(std::string::const_iterator firstIt,
                                                std::string::const_iterator lastIt){

    std::vector<std::string::size_type> numStr((lastIt - firstIt) + 3);
    std::vector<std::string::size_type>::iterator it = numStr.begin();

    for(; lastIt != firstIt; ++firstIt, ++it){

        *it = *firstIt;
    }

    numStr.back() = 0;
    numStr[numStr.size() - 2] = 0;
    numStr[numStr.size() - 3] = 0;

    std::vector<std::string::size_type> orderR12 = uniqueR12Order(numStr);

    return mergeR0AndR12Orders(r0::r0Order(orderR12, numStr), orderR12, numStr);
}


}