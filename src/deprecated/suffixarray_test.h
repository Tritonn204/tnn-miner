#ifndef SATEST
#define SATEST

#include <string>
#include <vector>

#include "suffixarray.h"

struct testCase {
  std::string name;
  std::string source;
  std::vector<std::string> patterns;
};

testCase testCases [] = {
  {
		"empty string",
		"",
		{
			"",
			"foo",
			"(foo)",
			".*",
			"a*",
		},
	},

	{
		"all a's",
		"aaaaaaaaaa", // 10 a's
		{
			"",
			"a",
			"aa",
			"aaa",
			"aaaa",
			"aaaaa",
			"aaaaaa",
			"aaaaaaa",
			"aaaaaaaa",
			"aaaaaaaaa",
			"aaaaaaaaaa",
			"aaaaaaaaaaa", // 11 a's
			".",
			".*",
			"a+",
			"aa+",
			"aaaa[b]?",
			"aaa*",
		},
	},

	{
		"abc",
		"abc",
		{
			"a",
			"b",
			"c",
			"ab",
			"bc",
			"abc",
			"a.c",
			"a(b|c)",
			"abc?",
		},
	},

	{
		"barbara*3",
		"barbarabarbarabarbara",
		{
			"a",
			"bar",
			"rab",
			"arab",
			"barbar",
			"bara?bar",
		},
	},

	{
		"typing drill",
		"Now is the time for all good men to come to the aid of their country.",
		{
			"Now",
			"the time",
			"to come the aid",
			"is the time for all good men to come to the aid of their",
			"to (come|the)?",
		},
	},

	{
		"godoc simulation",
		"package main\n\nimport(\n    \"rand\"\n    ",
		{},
	}
};

void TestSuffixArray();

std::vector<std::vector<int>> FindAllStringIndex(const std::string& s, const std::regex& re, int n) {
    if (n < 0) {
        n = s.length() + 1;
    }
    
    std::vector<std::vector<int>> result;
    std::smatch match;
    auto callback = [&](const std::string& str) {
        std::vector<int> indices;
        indices.push_back(match.position());
        indices.push_back(match.position() + match.length());
        result.push_back(indices);
    };
    
    std::sregex_iterator begin(s.begin(), s.end(), re);
    std::sregex_iterator end;
    int count = 0;
    
    while (begin != end && count < n) {
        match = *begin;
        callback(match.str());
        ++begin;
        ++count;
    }
    
    return result;
}

void testLookups(std::ostream& os, const testCase& tc, const Index* x, int n);
std::vector<int> find(std::string src, std::string s, int n);
void testFindAllIndex(const testCase &tc, const Index &x, std::regex &rx, const std::string &pattern, int n);
void testLookup(const testCase& tc, const Index& x, const std::string& s, int n);

void testConstruction(testCase* tc, Index* x);
bool equal(const Index* x, const Index* y);
int testSaveRestore(testCase &tc, Index& x);


#endif