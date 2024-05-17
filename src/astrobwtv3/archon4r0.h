#pragma once

typedef unsigned int suffix;
typedef unsigned int t_index;
typedef unsigned char byte;

void constructSuffixArray(byte *data, suffix *P, t_index N, t_index K, t_index reserved);
void ray(byte *data, suffix *A, t_index num, unsigned depth);