from kaspy.kaspa_clients import RPCClient

import copy
import numpy as np
import math
import hashlib
import struct
import ctypes

# from cSHAKE import *
from Crypto.Hash import cSHAKE256


# so = ctypes.cdll.LoadLibrary('./_generate_matrix.so')

# generate_matrix = so.generate_matrix
# generate_matrix.argtypes = [ctypes.c_char_p]
# generate_matrix.restype = ctypes.c_void_p
# free = so.free
# free.argtypes = [ctypes.c_void_p]


def to_little(val):
	little_hex = bytearray.fromhex(val)
	little_hex.reverse()

	str_little = ''.join(format(x, '02x') for x in little_hex)

	return str_little


def gen_pre_pow_hash(serialized_header, blake2b_key=b'BlockHash'):

	pre_pow_hash_g = hashlib.blake2b(digest_size=32, key=blake2b_key)
	pre_pow_hash_g.update(serialized_header)
	pre_pow_hash_g = pre_pow_hash_g.hexdigest()

	return pre_pow_hash_g


def transform_pre_pow_hash(pre_pow_hash_g):

	pre_pow_hash_np = np.array([0, 0, 0, 0], dtype=np.uint64)

	for i in range(0, 4):
		# Разбиение результата blake2b на 8 байтовые отрезки для представления в uint64
		pre_pow_hash_np[i] = np.uint64(
		    int(to_little(pre_pow_hash_g[i * 16: i * 16 + 16]), 16))

	return pre_pow_hash_np


def to_le_bytes(keccak_hash):

	size = int(math.ceil(math.log(np.iinfo(keccak_hash.dtype).max, 2) / 4))

	keccak_bytes = ""

	for i in range(0, len(keccak_hash)):
		part = hex(keccak_hash[i])[2:].rjust(size, '0')

		reversed_by_bytes = ''

		for i in range(len(part), 0, -2):
			reversed_by_bytes += part[i - 2:i]

		keccak_bytes += reversed_by_bytes

	return np.array([int(keccak_bytes[i:i+2], 16) for i in range(0, len(keccak_bytes), 2)], dtype=np.uint8)


def decode_to_slice(slice):

	return np.array([int(slice[i:i+2], 16) for i in range(0, len(slice), 2)], dtype=np.uint8)


def header_serialization(header, for_pre_pow=True):

	bytes_header = np.empty([], dtype=np.uint8)

	if for_pre_pow:
		nonce = 0
		timestamp = 0
	else:
		nonce = header['nonce']
		timestamp = header['timestamp']

	version = header['version']

	# adding version and number of parents
	version = to_le_bytes(np.array([version], dtype=np.uint16))
	num_parents = to_le_bytes(np.array([len(header['parents'])], dtype=np.uint64))

	bytes_header = np.concatenate((version, num_parents))

	# adding each parent hashes and their lenght
	for parent in header['parents']:
		bytes_header = np.concatenate((bytes_header, to_le_bytes(
		    np.array([len(parent['parentHashes'])], dtype=np.uint64))))

		for hash_string in parent['parentHashes']:
			bytes_header = np.concatenate((bytes_header, decode_to_slice(hash_string)))

	# adding hash_merkle_root
	bytes_header = np.concatenate(
	    (bytes_header, decode_to_slice(header['hashMerkleRoot'])))

	# adding accepted_id_merkle_root
	bytes_header = np.concatenate(
	    (bytes_header, decode_to_slice(header['acceptedIdMerkleRoot'])))

	# adding uxto_commitment
	bytes_header = np.concatenate(
	    (bytes_header, decode_to_slice(header['utxoCommitment'])))

	# adding timestamp
	bytes_header = np.concatenate(
	    (bytes_header, to_le_bytes(np.array([timestamp], dtype=np.uint64))))

	# adding bits
	bytes_header = np.concatenate(
	    (bytes_header, to_le_bytes(np.array([header['bits']], dtype=np.uint32))))

	# adding nonce
	bytes_header = np.concatenate(
	    (bytes_header, to_le_bytes(np.array([nonce], dtype=np.uint64))))

	# adding daa_score
	bytes_header = np.concatenate((bytes_header, to_le_bytes(
	    np.array([header['daaScore']], dtype=np.uint64))))

	# adding blue score
	bytes_header = np.concatenate((bytes_header, to_le_bytes(
	    np.array([header['blueScore']], dtype=np.uint64))))

	# adding blue work
	blue_work_len = (len(header['blueWork']) + 1) // 2

	if len(header['blueWork']) % 2 == 0.0:
		blue_work = decode_to_slice(header['blueWork']);
	else:
		blue_work = decode_to_slice("0" + header['blueWork']);

	bytes_header = np.concatenate(
	    (bytes_header, to_le_bytes(np.array([blue_work_len], dtype=np.uint64))))
	bytes_header = np.concatenate((bytes_header, blue_work))

	# adding pruning_point
	bytes_header = np.concatenate(
	    (bytes_header, decode_to_slice(header['pruningPoint'])))

	return bytes_header


def main():

	proofOfWorkDomain = "ProofOfWorkHash"

	client = RPCClient()
	client.connect(host='127.0.0.1', port='18110')

	command = 'getBlockRequest'
	payload = {
	    "hash": "17b59f5c71174ea5b6a796dce5ad64ad053769e895b53b8a3f31dabf2a7465e6"}

	resp = client.request(command=command, payload=payload)
	resp = copy.copy(resp)

	block = resp['getBlockResponse']['block']
	header = block['header']
	header['version'] = 1

	serialized_header = header_serialization(header)

	pre_pow_hash = gen_pre_pow_hash(serialized_header)

	nonce_hash = hex(int(header['nonce']))[2:].rjust(16, '0')
	nonce_hash = struct.pack('<Q', int(header['nonce'])).hex()
	timestamp = struct.pack('<Q', int(header['timestamp'])).hex()

	data = pre_pow_hash + timestamp + "0000000000000000000000000000000000000000000000000000000000000000" + nonce_hash
	print(data)

	# so = ctypes.cdll.LoadLibrary('./spectre_lib/_spectre_lib.so')

	# spectre_lib = so.spectre_lib
	# spectre_lib.argtypes = [ctypes.c_char_p]
	# spectre_lib.restype = ctypes.c_void_p
	# free = so.free
	# free.argtypes = [ctypes.c_void_p]

	hex_string = bytes(data.encode())
	c_hex_string2 = ctypes.create_string_buffer(hex_string, len(hex_string))
	print("c_hex_string2: ", c_hex_string2)

	# ptr = spectre_lib(c_hex_string2)

	# out = ctypes.string_at(ptr)
	# hex_string = out.decode('utf-8')
	# free(ptr)


	# AstroBWTv3Hash = so.AstroBWTv3Hash
	# AstroBWTv3Hash.argtypes = [ctypes.c_char_p]
	# AstroBWTv3Hash.restype = ctypes.c_void_p
	# free = so.free
	# free.argtypes = [ctypes.c_void_p]

	# hex_string = bytes(hex_string.encode())
	c_hex_string2 = ctypes.create_string_buffer(hex_string, len(hex_string))
	print("input for astrobwtv3: ", hex_string)

	# ptr = AstroBWTv3Hash(c_hex_string2)

	# out = ctypes.string_at(ptr)
	# hex_string = out.decode('utf-8')

	# AstroBWT_hash = ctypes.create_string_buffer(bytes(hex_string.encode()), len(hex_string))

	# free(ptr)


	# keccak = np.copy(transform_pre_pow_hash(pre_pow_hash))

	# bArr = []

	# for i in range(len(keccak)):
	# 	bArr = bArr + list(map(int, struct.pack('<Q', keccak[i])))

	# hex_string = bytes(bytes(bArr).hex().encode())
	# c_hex_string = ctypes.create_string_buffer(hex_string, len(hex_string))

	# ptr = generate_matrix(c_hex_string)

	# out = ctypes.string_at(ptr)
	# matrix = out.decode('utf-8')
	# free(ptr)


	# matrix_prepared = bytes(matrix.encode())
	# c_matrix_prepared = ctypes.create_string_buffer(matrix_prepared, len(matrix_prepared))

	# verify_pow = so.verify_pow
	# verify_pow.argtypes = [ctypes.c_char_p]
	# verify_pow.restype = ctypes.c_void_p
	# free = so.free
	# free.argtypes = [ctypes.c_void_p]

	# matrix_prepared = bytes(matrix.encode())
	# c_matrix_prepared = ctypes.create_string_buffer(matrix_prepared, len(matrix_prepared))

	# ptr = verify_pow(AstroBWT_hash, c_matrix_prepared)

	# out = ctypes.string_at(ptr)
	# hex_string = out.decode('utf-8')
	# free(ptr)

	# print(hex_string)



	# client = RPCClient()
	# client.connect(host='127.0.0.1', port='18110')

	# command = 'getBlockTemplateRequest'
	# payload = {"payAddress": "spectre:qrcdstvtcjy49a7u4v39jdke7337tgumyk5jd5kxyp45ykddge9nyr75djwy5", "extraData": "test"}

	# resp  = client.request(command=command, payload=payload)
	# resp = copy.copy(resp)



if __name__ == '__main__':	
	main()