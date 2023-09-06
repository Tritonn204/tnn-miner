#ifndef _ARCHON_H_
#define _ARCHON_H_

#include <stdio.h>
#include "direct.h"

#ifdef __cplusplus
extern "C" {
#endif
//#define VERBOSE

int geninit(FILE*);
int gencode(struct akData*);
void genprint();
void genexit();

int compute(struct akData*);
int verify();
int encode(FILE*);
int decode(FILE*);

int archonsort(unsigned char *T, int *SA, int n);

#ifdef __cplusplus
}
#endif

#endif /* _ARCHON_H_ */
