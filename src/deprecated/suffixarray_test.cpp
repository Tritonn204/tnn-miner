#include <suffixarray.h>
#include "suffixarray_test.h"
#include <sais2.h>

// int main() {
//   // TestSuffixArray();
//   int a = -1;
//   for (testCase T : testCases) {
//     // a++;
//     // if (a == 0) {
//     //   continue;
//     // }
//     byte *buffer = reinterpret_cast<unsigned char*>(const_cast<char*>(T.source.c_str()));
//     Index *x = New(buffer, T.source.size());

//     testConstruction(&T, x);
// 		testSaveRestore(T, *x);
// 		testLookups(std::cout, T, x, 0);
// 		testLookups(std::cout, T, x, 1);
// 		testLookups(std::cout, T, x, 10);
// 		testLookups(std::cout, T, x, 2e9);
// 		testLookups(std::cout, T, x, -1);
//   }
// }

std::vector<int> find(std::string src, std::string s, int n)
{
  std::vector<int> res;
  if (s != "" && n != 0)
  {
    int i = -1;
    while (n < 0 || res.size() < n)
    {
      int j = src.find(s.substr(i + 1, src.size() - (i + 1)));
      if (j < 0)
        break;
      i += j + 1;
      res.push_back(i);
    }
  }
  return res;
}

void TestSuffixArray() {
	std::string s("abcabxabcd");
	int32_t result32[] = {0, 6, 3, 1, 7, 4, 2, 8, 9, 5};

	int32_t sa32[10];
	int64_t sa16[10];
	text_32(reinterpret_cast<unsigned char*>(const_cast<char*>(s.c_str())), sa32, s.length(), 10);
	text_64(reinterpret_cast<unsigned char*>(const_cast<char*>(s.c_str())), sa16, s.length(), 10);

	for (int i = 0; i < 10; i++){
		if (result32[i] != sa32[i] || result32[i] != int32_t(sa16[i])) {
			std::cerr << "suffix array failed" << std::endl;
		}
	}
}

void testLookup(const testCase &tc, const Index &x, const std::string &s, int n)
{
  std::vector<int> res = x.Lookup(to_vector(s), n);
  std::vector<int> exp = find(tc.source, s, n);

  // Check that the lengths match
  if (res.size() != exp.size())
  {
    std::cout << "test " << tc.name << ", lookup " << s << " (n = " << n
              << "): expected " << exp.size() << " results; got " << res.size() << std::endl;
  }

  // If n >= 0 the number of results is limited --- unless n >= all results,
  // we may obtain different positions from the Index and from find (because
  // Index may not find the results in the same order as find) => in general
  // we cannot simply check that the res and exp lists are equal

  // Check that each result is in fact a correct match and there are no duplicates
  std::sort(res.begin(), res.end());
  for (size_t i = 0; i < res.size(); i++)
  {
    int r = res[i];
    if (r < 0 || r >= tc.source.size())
    {
      std::cout << "test " << tc.name << ", lookup " << s << ", result " << i << " (n = " << n
                << "): index " << r << " out of range [0, " << tc.source.size() << "["
                << std::endl;
    }
    else if (!std::equal(s.begin(), s.end(), tc.source.begin() + r))
    {
      std::cout << "test " << tc.name << ", lookup " << s << ", result " << i << " (n = " << n
                << "): index " << r << " not a match" << std::endl;
    }
    if (i > 0 && res[i - 1] == r)
    {
      std::cout << "test " << tc.name << ", lookup " << s << ", result " << i << " (n = " << n
                << "): found duplicate index " << r << std::endl;
    }
  }

  if (n < 0)
  {
    // All results computed - sorted res and exp must be equal
    for (size_t i = 0; i < res.size(); i++)
    {
      int r = res[i];
      int e = exp[i];
      if (r != e)
      {
        std::cout << "test " << tc.name << ", lookup " << s << ", result " << i
                  << ": expected index " << e << "; got " << r << std::endl;
      }
    }
  }
}

void testFindAllIndex(const testCase &tc, const Index &x, std::regex &rx, const std::string &pattern, int n)
{
  std::vector<std::vector<int>> res = x.FindAllIndex(pattern, rx, n);
  std::vector<std::vector<int>> exp = FindAllStringIndex(tc.source, rx, n);

  // Check that the lengths match
  if (res.size() != exp.size())
  {
    std::cout << "test " << tc.name << ", FindAllIndex " << tc.source << " (n = " << n
              << "): expected " << exp.size() << " results; got " << res.size() << std::endl;
  }

  // If n >= 0 the number of results is limited --- unless n >= all results,
  // we may obtain different positions from the Index and from regexp (because
  // Index may not find the results in the same order as regexp) => in general
  // we cannot simply check that the res and exp lists are equal

  // Check that each result is in fact a correct match and the result is sorted
  for (size_t i = 0; i < res.size(); i++)
  {
    std::vector<int> r = res[i];
    if (r[0] < 0 || r[0] > r[1] || r[1] > tc.source.size())
    {
      std::cout << "test " << tc.name << ", FindAllIndex " << pattern << ", result " << i << " (n == " << n
                << "): illegal match [" << r[0] << ", " << r[1] << "]" << std::endl;
    }
    else if (!std::regex_match(tc.source.substr(r[0], r[1] - r[1]), rx))
    {
      std::cout << "test " << tc.name << ", FindAllIndex " << pattern << ", result " << i << " (n = " << n
                << "): [" << r[0] << ", " << r[1] << "] not a match" << std::endl;
    }
  }

  if (n < 0)
  {
    // All results computed - sorted res and exp must be equal
    for (size_t i = 0; i < res.size(); i++)
    {
      std::vector<int> r = res[i];
      std::vector<int> e = exp[i];
      if (r[0] != e[0] || r[1] != e[1])
      {
        std::cout << "test " << tc.name << ", FindAllIndex " << pattern << ", result " << i
                  << ": expected match [" << e[0] << ", " << e[1]
                  << "]; got [" << r[0] << ", " << r[1] << "]" << std::endl;
      }
    }
  }
}

void testLookups(std::ostream &os, const testCase &tc, const Index *x, int n)
{
  for (const std::string &pat : tc.patterns)
  {
    testLookup(tc, *x, pat, n);
    std::regex rx;
    try
    {
      rx = std::regex(pat);
      testFindAllIndex(tc, *x, rx, pat, n);
    }
    catch (const std::regex_error &e)
    {
      // Handle regex compilation error
      std::cerr << "Regex compilation error: " << e.what() << std::endl;
    }
  }
}

void testConstruction(testCase *tc, Index *x)
{
  if (x->sa.int32.empty())
  {
    auto begin = x->sa.int64.begin();
    auto end = x->sa.int64.end();
    if (!std::is_sorted(begin, end))
    {
      printf("failed testConstruction %s\n", tc->name.c_str());
    }
  } else {
    auto begin = x->sa.int32.begin();
    auto end = x->sa.int32.end();
    if (!std::is_sorted(begin, end))
    {
      printf("failed testConstruction %s\n", tc->name.c_str());
    }
  }
}

bool equal(const Index* x, const Index* y) {
    if (x->data != y->data) {
        return false;
    }
    if (x->sa.len() != y->sa.len() || x->sa.len() != y->sa.len()) {
        return false;
    }
    int n = x->sa.len();
    for (int i = 0; i < n; i++) {
        if (x->sa.int32[i] != y->sa.int32[i] || x->sa.int64[i] != y->sa.int64[i]) {
            return false;
        }
    }
    return true;
}

int testSaveRestore(testCase &tc, Index& x) {
    std::stringstream buf;
    for (const auto& byte : x.data) {
        buf << byte;
    }

    size_t size = buf.str().size();

    Index y;
    std::string dataStr = buf.str();
    y.Read(buf);

    if (!equal(&x,&y)) {
        printf("Restored index doesn't match saved index %s\n", tc.name.c_str());
    }

    // Restoring index using forced 32
    int old = maxData32;
    maxData32 = realMaxData32;
    y.Read(buf);

    if (!equal(&x,&y)) {
        printf("Restored index doesn't match saved index %s\n", tc.name.c_str());
    }

    // Restoring index using forced 64
    maxData32 = -1;
    y.Read(buf);

    if (!equal(&x,&y)) {
        printf("Restored index doesn't match saved index %s\n", tc.name.c_str());
    }

    maxData32 = old;

    return size;
}