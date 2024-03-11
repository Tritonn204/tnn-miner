#include <inttypes.h>
#include <thrust/sort.h>

const uint32_t MAXX2 = (256 * 384) - 1; // this is the maximum

bool cmp(int x, int y, int *rank, int S, int &gap)
{
  // if (rank[x] != rank[y])
  //   return rank[x] < rank[y];
  // x += gap, y += gap;
  // return (x < S && y < S) ? rank[x] < rank[y] : x > y;
}
void Sa_build(unsigned char *in, int* rank, int *sa, int S)
{
  // int tmp[MAXX2] = {0};
  // int gap;
  // for (int i = 0; i < S; ++i)
  //   rank[i] = in[i],
  //   sa[i] = i;
  // for (gap = 1;; gap <<= 1)
  // {
  //   thrust::sort(thrust::device, sa, sa + S, cmp);
  //   for (int i = 1; i < S; ++i)
  //     tmp[i] = tmp[i - 1] + cmp(sa[i - 1], sa[i], rank, S, &gap);
  //   for (int i = 0; i < S; ++i)
  //     rank[sa[i]] = tmp[i];
  //   if (tmp[S - 1] == S - 1)
  //     break;
  // }
}