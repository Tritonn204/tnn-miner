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

#ifndef R12_HPP
#define R12_HPP

#include <vector>
#include <string>

namespace dc3 {

namespace r12 {

std::vector<std::string::size_type> r12Indexes(std::string::size_type numStrSize);

std::vector<std::string::size_type> indexToR12OrderMapping(
        std::vector<std::string::size_type>::const_iterator r12FirstIt,
        std::vector<std::string::size_type>::const_iterator r12LastIt,
        std::string::size_type numStrSize);

std::vector<std::string::size_type> r12NumString(const std::vector<std::string::size_type>& orderOfR12,
                                                 const std::vector<std::string::size_type>& offsetOfR12,
                                                 std::string::size_type numStrSize);

std::pair<std::vector<std::string::size_type>, std::vector<std::string::size_type>> r12Order(
        const std::vector<std::string::size_type> &numStr);
}

}


#endif // R12_HPP