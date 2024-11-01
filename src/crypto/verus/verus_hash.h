// (C) 2018 Michael Toutonghi
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

/*
This provides the PoW hash function for Verus, enabling CPU mining.
*/
#ifndef VERUS_HASH_H_
#define VERUS_HASH_H_

// verbose output when defined
//#define VERUSHASHDEBUG 1

#include <cstring>
#include <vector>

#include "uint256.h"
#include "verus_clhash.h"

extern "C" 
{
#include "haraka.h"
#include "haraka_portable.h"

}

class CVerusHash
{
    public:
        static void Hash(void *result, const void *data, size_t len);
        static void (*haraka512Function)(unsigned char *out, const unsigned char *in);

        static void init();

        CVerusHash() { }

        CVerusHash &Write(const unsigned char *data, size_t len);

        CVerusHash &Reset()
        {
            curBuf = buf1;
            result = buf2;
            curPos = 0;
            std::fill(buf1, buf1 + sizeof(buf1), 0);
            return *this;
        }

        int64_t *ExtraI64Ptr() { return (int64_t *)(curBuf + 32); }
        void ClearExtra()
        {
            if (curPos)
            {
                std::fill(curBuf + 32 + curPos, curBuf + 64, 0);
            }
        }
        void ExtraHash(unsigned char hash[32]) { (*haraka512Function)(hash, curBuf); }

        void Finalize(unsigned char hash[32])
        {
            if (curPos)
            {
                std::fill(curBuf + 32 + curPos, curBuf + 64, 0);
                (*haraka512Function)(hash, curBuf);
            }
            else
                std::memcpy(hash, curBuf, 32);
        }

    private:
        // only buf1, the first source, needs to be zero initialized
        unsigned char buf1[64] = {0}, buf2[64];
        unsigned char *curBuf = buf1, *result = buf2;
        size_t curPos = 0;
};

class CVerusHashV2
{
    public:
        static void Hash(void *result, const void *data, size_t len);
        static void (*haraka512Function)(unsigned char *out, const unsigned char *in);
        static void (*haraka512KeyedFunction)(unsigned char *out, const unsigned char *in, const u128 *rc);
        static void (*haraka256Function)(unsigned char *out, const unsigned char *in);

        static void init();

        verusclhasher vclh;

        CVerusHashV2() : vclh() {
            // we must have allocated key space, or can't run
            if (!verusclhasher_key.get())
            {
                printf("ERROR: failed to allocate hash buffer - terminating\n");
                assert(false);
            }
        }

        CVerusHashV2 &Write(const unsigned char *data, size_t len);

        inline CVerusHashV2 &Reset()
        {
            curBuf = buf1;
            result = buf2;
            curPos = 0;
            std::fill(buf1, buf1 + sizeof(buf1), 0);

			return *this;

            return *this;

        }

        inline int64_t *ExtraI64Ptr() { return (int64_t *)(curBuf + 32); }
        inline void ClearExtra()
        {
            if (curPos)
            {
                std::fill(curBuf + 32 + curPos, curBuf + 64, 0);
            }
        }

        template <typename T>
        inline void FillExtra(const T *_data)
        {
            unsigned char *data = (unsigned char *)_data;
            int pos = curPos;
            int left = 32 - pos;
            do
            {
                int len = left > sizeof(T) ? sizeof(T) : left;
                std::memcpy(curBuf + 32 + pos, data, len);
                pos += len;
                left -= len;
            } while (left > 0);
        }
        inline void ExtraHash(unsigned char hash[32]) { (*haraka512Function)(hash, curBuf); }
        inline void ExtraHashKeyed(unsigned char hash[32], u128 *key) { (*haraka512KeyedFunction)(hash, curBuf, key); }

        void Finalize(unsigned char hash[32])
        {
            if (curPos)
            {
                std::fill(curBuf + 32 + curPos, curBuf + 64, 0);
                (*haraka512Function)(hash, curBuf);
            }
            else
                std::memcpy(hash, curBuf, 32);
        }

        // chains Haraka256 from 32 bytes to fill the key
        static u128 *GenNewCLKey(unsigned char *seedBytes32)
        {
	
			unsigned char *key = (unsigned char *)verusclhasher_key.get();
            verusclhash_descr *pdesc = (verusclhash_descr *)verusclhasher_descr.get();
            // skip keygen if it is the current key
            if (pdesc->seed != *((uint256 *)seedBytes32))
            {
                // generate a new key by chain hashing with Haraka256 from the last curbuf
                int n256blks = pdesc->keySizeInBytes >> 5;
                int nbytesExtra = pdesc->keySizeInBytes & 0x1f;
                unsigned char *pkey = key + pdesc->keySizeInBytes;
                unsigned char *psrc = seedBytes32;
                for (int i = 0; i < n256blks; i++)
                {
                    (*haraka256Function)(pkey, psrc);

                    psrc = pkey;
                    pkey += 32;
                }
                if (nbytesExtra)
                {
                    unsigned char buf[32];
                    (*haraka256Function)(buf, psrc);
                    memcpy(pkey, buf, nbytesExtra);
                }
                pdesc->seed = *((uint256 *)seedBytes32);
            }
            memcpy(key, key + pdesc->keySizeInBytes, pdesc->keySizeInBytes);
            return (u128 *)key;
        }

        inline uint64_t IntermediateTo128Offset(uint64_t intermediate)
        {
            // the mask is where we wrap
            uint64_t mask = vclh.keyMask >> 4;
            return intermediate & mask;
        }

        void Finalize2b(unsigned char hash[32])
        {
            // fill buffer to the end with the beginning of it to prevent any foreknowledge of
            // bits that may contain zero
			//uint8_t temp[64] = { 0x0c, 0x4b, 0x23, 0x67, 0x8e, 0x9d, 0xc3, 0x5e, 0xaa, 0xed, 0x49, 0x3e, 0x32, 0x27, 0x3b, 0x24, 0x3b, 0xae, 0xc9, 0x7b, 0x9a, 0xcc, 0x02, 0x72, 0x38, 0x61, 0xb0, 0xc6, 0x58, 0x30, 0x23, 0x8e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x4b, 0x23, 0x67, 0x8e, 0x9d, 0xc3, 0x5e, 0xaa, 0xed, 0x49, 0x3e, 0x32, 0x27, 0x3b, 0x24, 0x0c };

		//	memcpy(curBuf, temp, 64);
            FillExtra((u128 *)curBuf);

            u128 *key = GenNewCLKey(curBuf);

            uint64_t intermediate = vclh(curBuf, key);

            FillExtra(&intermediate);


            // get the final hash with a mutated dynamic key for each hash result
            (*haraka512KeyedFunction)(hash, curBuf, key + IntermediateTo128Offset(intermediate));
#ifdef VERUSHASHDEBUG
			printf("[cpu]Final hash    : ");
			for (int i = 0; i < 32; i++)
				printf("%02x", ((uint8_t*)&hash[0])[i]);
			printf("\n");
#endif
            /*
            // TEST BEGIN
            // test against the portable version
            uint256 testHash1 = *(uint256 *)hash, testHash2;
            FillExtra((u128 *)curBuf);
            u128 *hashKey = ((u128 *)vclh.gethashkey());
            uint64_t temp = verusclhash_port(key, curBuf, vclh.keyMask);
            FillExtra(&temp);
            haraka512_keyed((unsigned char *)&testHash2, curBuf, hashKey + IntermediateTo128Offset(intermediate));
            if (testHash1 != testHash2)
            {
                printf("Portable version failed! intermediate1: %lx, intermediate2: %lx\n", intermediate, temp);
            }
            // END TEST
            */
        }

        inline unsigned char *CurBuffer()
        {
            return curBuf;
        }

    private:
        // only buf1, the first source, needs to be zero initialized
        alignas(32) unsigned char buf1[64] = {0}, buf2[64];
        unsigned char *curBuf = buf1, *result = buf2;
        size_t curPos = 0;
};

extern void verus_hash(void *result, const void *data, size_t len);
extern void verus_hash_v2(void *result, const void *data, size_t len);

#endif