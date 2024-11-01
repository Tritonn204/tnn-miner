/**
* Equihash solver interface for ccminer (compatible with linux and windows)
* Solver taken from nheqminer, by djeZo (and NiceHash)
* tpruvot - 2017 (GPL v3)
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#define VERUS_KEY_SIZE 8832
#define VERUS_KEY_SIZE128 552
#include <stdexcept>
#include <vector>
#include "verus_clhash.h"
#include "uint256.h"
//#include "hash.h"
#include <miner.h>
//#include "primitives/block.h"
//extern "C"
//{
//#include "haraka.h"

//}
enum
{
	// primary actions
	SER_NETWORK = (1 << 0),
	SER_DISK = (1 << 1),
	SER_GETHASH = (1 << 2),
};
// input here is 140 for the header and 1344 for the solution (equi.cpp)
static const int PROTOCOL_VERSION = 170002;

//#include <cuda_helper.h>

#define EQNONCE_OFFSET 30 /* 27:34 */
#define NONCE_OFT EQNONCE_OFFSET

static bool init[MAX_GPUS] = { 0 };

static __thread uint32_t throughput = 0;



#ifndef htobe32
#define htobe32(x) swab32(x)
#endif

extern "C" inline void GenNewCLKey(unsigned char *seedBytes32, u128 *keyback)
{
	// generate a new key by chain hashing with Haraka256 from the last curbuf
	int n256blks = VERUS_KEY_SIZE >> 5;  //8832 >> 5
	int nbytesExtra = VERUS_KEY_SIZE & 0x1f;  //8832 & 0x1f
	unsigned char *pkey = (unsigned char*)keyback;
	unsigned char *psrc = seedBytes32;
	for (int i = 0; i < n256blks; i++)
	{
		haraka256(pkey, psrc);

		psrc = pkey;
		pkey += 32;
	}
	if (nbytesExtra)
	{
		unsigned char buf[32];
		haraka256(buf, psrc);
		memcpy(pkey, buf, nbytesExtra);
	}
}

extern "C" inline void FixKey(uint32_t *fixrand, uint32_t *fixrandex, u128 *keyback,
	u128 * g_prand, u128 *g_prandex)
{

	for (int i = 31; i > -1; i--)
	{
		keyback[fixrandex[i]] = g_prandex[i];
		keyback[fixrand[i]] = g_prand[i];
	}

}


 extern "C" inline void VerusHashHalf(void *result2, unsigned char *data, int len)
{
	alignas(32) unsigned char buf1[64] = { 0 }, buf2[64];
	unsigned char *curBuf = buf1, *result = buf2;
	int curPos = 0;
	//unsigned char result[64];
	curBuf = buf1;
	result = buf2;
	curPos = 0;
	std::fill(buf1, buf1 + sizeof(buf1), 0);

	unsigned char *tmp;

	load_constants();

	// digest up to 32 bytes at a time
	for (int pos = 0; pos < len; )
	{
		int room = 32 - curPos;

		if (len - pos >= room)
		{
			memcpy(curBuf + 32 + curPos, data + pos, room);
			haraka512(result, curBuf);
			tmp = curBuf;
			curBuf = result;
			result = tmp;
			pos += room;
			curPos = 0;
		}
		else
		{
			memcpy(curBuf + 32 + curPos, data + pos, len - pos);
			curPos += len - pos;
			pos = len;
		}
	}

	memcpy(curBuf + 47, curBuf, 16);
	memcpy(curBuf + 63, curBuf, 1);
	//	FillExtra((u128 *)curBuf);
	memcpy(result2, curBuf, 64);
};




extern "C" void inline Verus2hash(unsigned char *hash, unsigned char *curBuf, unsigned char *nonce,
	u128  * __restrict data_key, uint8_t *gpu_init, uint32_t *fixrand, uint32_t *fixrandex, u128 *g_prand,
	u128 *g_prandex, int version)
{
	//uint64_t mask = VERUS_KEY_SIZE128; //552
	static const __m128i shuf1 = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0);
	const __m128i fill1 = _mm_shuffle_epi8(_mm_load_si128((u128 *)curBuf), shuf1);
	static const __m128i shuf2 = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0);
	unsigned char ch = curBuf[0];
	_mm_store_si128((u128 *)(&curBuf[32 + 16]), fill1);
	curBuf[32 + 15] = ch;
	//	FillExtra((u128 *)curBuf);
	uint64_t intermediate;
	memcpy(curBuf + 32, nonce, 15);  //copy the 15bytes nonce

	intermediate = verusclhashv2_2(data_key, curBuf, 511, fixrand, fixrandex, g_prand, g_prandex);
		//FillExtra
	__m128i fill2 = _mm_shuffle_epi8(_mm_loadl_epi64((u128 *)&intermediate), shuf2);
	_mm_store_si128((u128 *)(&curBuf[32 + 16]), fill2);
	curBuf[32 + 15] = *((unsigned char *)&intermediate);
	intermediate &= 511;
	haraka512_keyed(hash, curBuf, data_key + intermediate);
	FixKey(fixrand, fixrandex, data_key, g_prand, g_prandex);
}


extern "C" int scanhash_verus(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint8_t blockhash_half[64] = { 0 };
	uint8_t gpuinit = 0;
	struct timeval tv_start, tv_end;
	u128 *data_key =  (u128*)malloc(VERUS_KEY_SIZE + 1024);
	u128 *data_key_prand = data_key + VERUS_KEY_SIZE128 ;
	u128 *data_key_prandex = data_key + VERUS_KEY_SIZE128 + 32;

	uint32_t nonce_buf = 0;
	uint32_t fixrand[32];
	uint32_t fixrandex[32];

	unsigned char block_41970[3] = { 0xfd, 0x40, 0x05};
	uint8_t  full_data[140 + 3 + 1344] = { 0 };
	uint8_t* sol_data = &full_data[140];

	memcpy(full_data, pdata, 140);
	memcpy(sol_data, block_41970, 3);
	memcpy(sol_data + 3, work->solution, 1344);
	uint8_t version = work->solution[0];
	uint8_t nonceSpace[15] = {0};  //pool nonce (32bit) + round(32bit) + thrd id (byte) + padding(2bytes) + counting nonce(32bit)
	
    if (version >= 7 && work->solution[5] > 0) {

        // clear non-canonical data from header/solution before hashing; required for merged mining 
		memset(full_data + 4, 0, 96);                        // hashPrevBlock, hashMerkleRoot, hashFinalSaplingRoot
        memset(full_data + 4 + 32 + 32 + 32 + 4, 0, 4);      // nBits
        memset(full_data + 4 + 32 + 32 + 32 + 4 + 4, 0, 32); // nNonce
        memset(sol_data + 3 + 8, 0, 64);                     // hashPrevMMRRoot, hashBlockMMRRoot
		memcpy(nonceSpace, &pdata[EQNONCE_OFFSET - 3], 7 );			// transfer the nonce values that would be in the header to
//		memcpy(nonceSpace + 4, &pdata[EQNONCE_OFFSET + 1], 3 );		// the 15 bytes available
		memcpy(nonceSpace + 7, &pdata[EQNONCE_OFFSET + 2], 4 );	
	}

	uint32_t  vhash[8] = { 0 };

	VerusHashHalf(blockhash_half, (unsigned char*)full_data, 1487);

	GenNewCLKey((unsigned char*)blockhash_half, data_key);  //data_key a global static 2D array data_key[16][8832];


	gettimeofday(&tv_start, NULL);

	throughput = 1;
	const uint32_t Htarg = ptarget[7];
	do {

		*hashes_done = nonce_buf + throughput;

		((uint32_t *)(&nonceSpace[11]))[0] = nonce_buf;

		Verus2hash((unsigned char *)vhash, (unsigned char *)blockhash_half, nonceSpace, data_key, 
				&gpuinit, fixrand, fixrandex , data_key_prand, data_key_prandex, version);


		if (vhash[7] <= Htarg )
		{
			work->valid_nonces++;
			memcpy(work->data, full_data, 140);
			int nonce = work->valid_nonces - 1;
			memcpy(work->extra, sol_data, 1347);
			memcpy(work->extra + 1332, nonceSpace, 15);  //copy in the valid nonce 15 bytes to the solution part
			bn_store_hash_target_ratio(vhash, work->target, work, nonce);

			work->nonces[work->valid_nonces - 1] = ((uint32_t*)full_data)[NONCE_OFT];
			//pdata[NONCE_OFT] = endiandata[NONCE_OFT] + 1;
			goto out;
		}

		if ((uint64_t)throughput + (uint64_t)nonce_buf >= (uint64_t)max_nonce) {

			break;
		}
		nonce_buf += throughput;

	} while (!work_restart[thr_id].restart);


out:
	gettimeofday(&tv_end, NULL);


	pdata[NONCE_OFT] = ((uint32_t*)full_data)[NONCE_OFT] + 1;
	free(data_key);

	return work->valid_nonces;
}

// cleanup
void free_verushash(int thr_id)
{
	if (!init[thr_id])
		return;



	init[thr_id] = false;
}