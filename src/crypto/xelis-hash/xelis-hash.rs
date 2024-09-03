#![cfg_attr(feature = "nightly", feature(portable_simd))]

#[cfg(feature = "nightly")]
use std::{ptr::read_unaligned, simd::{cmp::SimdPartialEq, num::SimdUint, u32x16, u32x8}};

use aes::cipher::generic_array::GenericArray;
use thiserror::Error as ThisError;
use tiny_keccak::keccakp;

// These are tweakable parameters
pub const MEMORY_SIZE: usize = 32768;
pub const SCRATCHPAD_ITERS: usize = 5000;
pub const ITERS: usize = 1;
pub const BUFFER_SIZE: usize = 42;
pub const SLOT_LENGTH: usize = 256;

// Untweakable parameters
pub const KECCAK_WORDS: usize = 25;
pub const BYTES_ARRAY_INPUT: usize = KECCAK_WORDS * 8;
pub const HASH_SIZE: usize = 32;
pub const STAGE_1_MAX: usize = MEMORY_SIZE / KECCAK_WORDS;

pub struct ScratchPad([u64; MEMORY_SIZE]);

impl ScratchPad {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn as_mut_slice(&mut self) -> &mut [u64; MEMORY_SIZE] {
        &mut self.0
    }
}

impl Default for ScratchPad {
    fn default() -> Self {
        Self([0; MEMORY_SIZE])
    }
}

#[derive(Debug, bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C, align(8))]
pub struct Bytes8Alignment([u8; 8]);

#[derive(Debug, Clone)]
pub struct Input {
    data: Vec<Bytes8Alignment>,
}

impl Default for Input {
    fn default() -> Self {
        let mut n = BYTES_ARRAY_INPUT / 8;
        if BYTES_ARRAY_INPUT % 8 != 0 {
            n += 1;
        }
    
        Self {
            data: vec![Bytes8Alignment([0; 8]); n]
        }
    }
} 

impl Input {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr() as *mut u8
    }

    pub fn as_mut_slice(&mut self) -> Result<&mut [u8; BYTES_ARRAY_INPUT], Error> {
        bytemuck::cast_slice_mut(&mut self.data).try_into().map_err(|_| Error)
    }

    pub fn as_slice(&self) -> Result<&[u8; BYTES_ARRAY_INPUT], Error> {
        bytemuck::cast_slice(&self.data).try_into().map_err(|_| Error)
    }
}

#[derive(Debug, ThisError)]
#[error("Error while hashing")]
pub struct Error;

pub type Hash = [u8; HASH_SIZE];

// This will auto allocate the scratchpad
pub fn xelis_hash_no_scratch_pad(input: &mut [u8]) -> Result<Hash, Error> {
    let mut scratchpad = ScratchPad::default();
    xelis_hash_scratch_pad(input, &mut scratchpad)
}

pub fn xelis_hash_scratch_pad(input: &mut [u8], scratch_pad: &mut ScratchPad) -> Result<Hash, Error> {
    xelis_hash(input, scratch_pad.as_mut_slice())
}

#[inline(always)]
fn stage_1(int_input: &mut [u64; KECCAK_WORDS], scratch_pad: &mut [u64; MEMORY_SIZE], a: (usize, usize), b: (usize, usize)) {
  // println!("int_input: {}", int_input.iter().map(|&x| format!("{:016x} ", x)).collect::<String>());
  for i in a.0..=a.1 {
      keccakp(int_input);
      // println!("after keccak: {}", int_input.iter().map(|&x| format!("{:016x} ", x)).collect::<String>());

      let mut rand_int: u64 = 0;
      for j in b.0..=b.1 {
          let pair_idx = (j + 1) % KECCAK_WORDS;
          let pair_idx2 = (j + 2) % KECCAK_WORDS;

          let target_idx = i * KECCAK_WORDS + j;
          let a = int_input[j] ^ rand_int;
          // Branching
          let left = int_input[pair_idx];
          let right = int_input[pair_idx2];
          let xor = left ^ right;
          // println!("target_idx: {}", target_idx);
          // println!("left: {}, right: {}", left, right);
          // println!("xor_result: {}", xor & 0x3);
          let v = match xor & 0x3 {
              0 => left & right,
              1 => !(left & right),
              2 => !xor,
              3 => xor,
              _ => unreachable!(),
          };
          let b = a ^ v;
          rand_int = b;
          scratch_pad[target_idx] = b;
      }
  }
}

pub fn xelis_hash(input: &mut [u8], scratch_pad: &mut [u64; MEMORY_SIZE]) -> Result<Hash, Error> {
    if input.len() < BYTES_ARRAY_INPUT {
        return Err(Error);
    }
    // println!("initial input: {}", input.iter().map(|&x| format!("{:02x}", x)).collect::<String>());  

    if scratch_pad.len() < MEMORY_SIZE {
        return Err(Error);
    }

    let int_input: &mut [u64; KECCAK_WORDS] = bytemuck::try_from_bytes_mut(&mut input[0..BYTES_ARRAY_INPUT])
    .map_err(|_| Error)?;

    // stage 1
    stage_1(int_input, scratch_pad, (0, STAGE_1_MAX - 1), (0, KECCAK_WORDS - 1));
    stage_1(int_input, scratch_pad, (STAGE_1_MAX, STAGE_1_MAX), (0, 17));

    // stage 2
    let mut slots: [u32; SLOT_LENGTH] = [0; SLOT_LENGTH];
    // this is equal to MEMORY_SIZE, just in u32 format
    let small_pad: &mut [u32; MEMORY_SIZE * 2] = bytemuck::try_cast_slice_mut(scratch_pad)
        .map_err(|_| Error)?
        .try_into()
        .map_err(|_| Error)?;

    // println!("first small_pad: {}", small_pad[0]);
    
    slots.copy_from_slice(&small_pad[small_pad.len() - SLOT_LENGTH..]);

    let mut indices: [u16; SLOT_LENGTH] = [0; SLOT_LENGTH];
    for _ in 0..ITERS {
        for j in 0..small_pad.len() / SLOT_LENGTH {
            // Initialize indices
            for k in 0..SLOT_LENGTH {
                indices[k] = k as u16;
            }

            for slot_idx in (0..SLOT_LENGTH).rev() {
                let index_in_indices = (small_pad[j * SLOT_LENGTH + slot_idx] % (slot_idx as u32 + 1)) as usize;
                let index = indices[index_in_indices] as usize;
                indices[index_in_indices] = indices[slot_idx];

                #[cfg(feature = "nightly")]
                {
                    let mut sum_buffer = u32x16::splat(0);
                
                    // println!("Initial sum_buffer: {:?}", sum_buffer);
                
                    for k in (0..SLOT_LENGTH).step_by(16) {
                        let slot_vector = u32x16::from_array(unsafe { read_unaligned(&slots[k] as *const u32 as *const [u32; 16]) });
                        let values = u32x16::from_array(unsafe { read_unaligned(&small_pad[j * SLOT_LENGTH + k] as *const u32 as *const [u32; 16]) });
                
                        // println!("Iteration {} - slot_vector: {:?}", k / 16, slot_vector);
                        // println!("Loop {} | Iteration {} - values: {:?}", slot_idx, k / 8, values);
                
                        let sign_mask = (slot_vector >> 31).simd_eq(u32x16::splat(0));
                        sum_buffer = sign_mask.select(sum_buffer + values, sum_buffer - values);
                        // println!("Iteration {} - sign_mask: {:08x?}", k / 16, sign_mask.to_array());
                
                        // println!("Iteration {} - sum_buffer: {:?}", k / 8, sum_buffer);
                        // println!("");
                    }

                    if j == 0 {print!("Complete Sum: {}, ", sum_buffer.reduce_sum());}
                
                    if slots[index] >> 31 == 0 {
                        sum_buffer[index % 8] -= small_pad[j * SLOT_LENGTH + index];
                    } else {
                        sum_buffer[index % 8] += small_pad[j * SLOT_LENGTH + index];
                    }

                    if j == 0 {println!("Adjusted Sum: {}", sum_buffer.reduce_sum());}
                
                    // println!("Final sum_buffer: {:?}", sum_buffer);
                    // println!("Reduced Sum: {}", sum_buffer.reduce_sum());
                    // if slot_idx == 0 {println!("slot: {}, index: {}", j, index);}
                
                    slots[index] += sum_buffer.reduce_sum();
                }
                
                #[cfg(not(feature = "nightly"))]
                {
                    let mut sum = slots[index];
                    let offset = j * SLOT_LENGTH;
                    for k in 0..index {
                        let pad = small_pad[offset + k];
                        sum = if slots[k] >> 31 == 0 {
                            sum.wrapping_add(pad)
                        } else {
                            sum.wrapping_sub(pad)
                        };
                    }
                    for k in (index + 1)..SLOT_LENGTH {
                        let pad = small_pad[offset + k];
                        sum = if slots[k] >> 31 == 0 {
                            sum.wrapping_add(pad)
                        } else {
                            sum.wrapping_sub(pad)
                        };
                    }

                    slots[index] = sum;
                    // println!("slot: {}, index: {}", j, index);
                }
            }
        }
    }

    small_pad[(MEMORY_SIZE * 2) - SLOT_LENGTH..].copy_from_slice(&slots);

    // stage 3
    let key = GenericArray::from([0u8; 16]);
    let mut block = GenericArray::from([0u8; 16]);

    let mut addr_a = (scratch_pad[MEMORY_SIZE - 1] >> 15) & 0x7FFF;
    let mut addr_b = scratch_pad[MEMORY_SIZE - 1] & 0x7FFF;

    // println!("addr_a: {}", addr_a);
    // println!("addr_b: {}", addr_b);

    let mut mem_buffer_a: [u64; BUFFER_SIZE] = [0; BUFFER_SIZE];
    let mut mem_buffer_b: [u64; BUFFER_SIZE] = [0; BUFFER_SIZE];

    for i in 0..BUFFER_SIZE as u64 {
        mem_buffer_a[i as usize] = scratch_pad[((addr_a + i) % MEMORY_SIZE as u64) as usize];
        mem_buffer_b[i as usize] = scratch_pad[((addr_b + i) % MEMORY_SIZE as u64) as usize];
    }

    // println!("{}", mem_buffer_a.iter().map(|&x| format!("{:016x} ", x)).collect::<String>());
    // println!("{}", mem_buffer_b.iter().map(|&x| format!("{:016x} ", x)).collect::<String>());

    let mut final_result = [0; HASH_SIZE];

    for i in 0..SCRATCHPAD_ITERS {
        let mem_a = mem_buffer_a[i % BUFFER_SIZE];
        let mem_b = mem_buffer_b[i % BUFFER_SIZE];

        block[..8].copy_from_slice(&mem_b.to_le_bytes());
        block[8..].copy_from_slice(&mem_a.to_le_bytes());

        // println!("pre block: {}", block.iter().map(|&x| format!("{:02x?}", x)).collect::<String>());

        aes::hazmat::cipher_round(&mut block, &key);

        // println!("block: {}", block.iter().map(|&x| format!("{:02x?}", x)).collect::<String>());

        let hash1 = u64::from_le_bytes(block[0..8].try_into().map_err(|_| Error)?);
        let hash2 = mem_a ^ mem_b;

        let mut result = !(hash1 ^ hash2);

        // println!("pre result: {}", result);

        for j in 0..HASH_SIZE {
            let a = mem_buffer_a[(j + i) % BUFFER_SIZE];
            let b = mem_buffer_b[(j + i) % BUFFER_SIZE];

            // more branching
            let v = match (result >> (j * 2)) & 0xf {
                0 => result.rotate_left(j as u32) ^ b,
                1 => !(result.rotate_left(j as u32) ^ a),
                2 => !(result ^ a),
                3 => result ^ b,
                4 => result ^ (a.wrapping_add(b)),
                5 => result ^ (a.wrapping_sub(b)),
                6 => result ^ (b.wrapping_sub(a)),
                7 => result ^ (a.wrapping_mul(b)),
                8 => result ^ (a & b),
                9 => result ^ (a | b),
                10 => result ^ (a ^ b),
                11 => result ^ (a.wrapping_sub(result)),
                12 => result ^ (b.wrapping_sub(result)),
                13 => result ^ (a.wrapping_add(result)),
                14 => result ^ (result.wrapping_sub(a)),
                15 => result ^ (result.wrapping_sub(b)),
                _ => unreachable!(),
            };

            result = v;
        }

        addr_b = result & 0x7FFF;
        mem_buffer_a[i % BUFFER_SIZE] = result;
        mem_buffer_b[i % BUFFER_SIZE] = scratch_pad[addr_b as usize];

        addr_a = (result >> 15) & 0x7FFF;
        scratch_pad[addr_a as usize] = result;

        // println!("post result: {}", result);

        let index = SCRATCHPAD_ITERS - i - 1;
        if index < 4 {
            final_result[index * 8..(SCRATCHPAD_ITERS - i) * 8].copy_from_slice(&result.to_be_bytes());
        }
    }

    Ok(final_result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{time::Instant, hint};

    fn test_input(input: &mut [u8], expected_hash: Hash) {
        let mut scratch_pad = [0u64; MEMORY_SIZE];
        let hash = xelis_hash(input, &mut scratch_pad).unwrap();
        assert_eq!(hash, expected_hash);
    }

    #[test]
    fn benchmark_cpu_hash() {
        const ITERATIONS: u32 = 1000;
        let mut input = [0u8; 200];
        let mut scratch_pad = [0u64; 32768];

        let start = Instant::now();
        for i in 0..ITERATIONS {
            input[0] = i as u8;
            input[1] = (i >> 8) as u8;
            let _ = hint::black_box(xelis_hash(&mut input, &mut scratch_pad)).unwrap();
        }

        let elapsed = start.elapsed();
        println!("Time took: {:?}", elapsed);
        println!("H/s: {:.2}", (ITERATIONS as f64 * 1000.) / (elapsed.as_millis() as f64));
        println!("ms per hash: {:.3}", (elapsed.as_millis() as f64) / ITERATIONS as f64);
    }

    #[test]
    fn test_zero_input() {
        let mut input = [0u8; 200];
        let expected_hash = [
            0x0e, 0xbb, 0xbd, 0x8a, 0x31, 0xed, 0xad, 0xfe, 0x09, 0x8f, 0x2d, 0x77, 0x0d, 0x84,
            0xb7, 0x19, 0x58, 0x86, 0x75, 0xab, 0x88, 0xa0, 0xa1, 0x70, 0x67, 0xd0, 0x0a, 0x8f,
            0x36, 0x18, 0x22, 0x65,
        ];

        test_input(&mut input, expected_hash);
    }

    #[test]
    fn test_xelis_input() {
        let mut input = [0u8; BYTES_ARRAY_INPUT];

        let custom = b"xelis-hashing-algorithm";
        input[0..custom.len()].copy_from_slice(custom);

        let expected_hash = [
            106, 106, 173, 8, 207, 59, 118, 108, 176, 196, 9, 124, 250, 195, 3,
            61, 30, 146, 238, 182, 88, 83, 115, 81, 139, 56, 3, 28, 176, 86, 68, 21
        ];
        test_input(&mut input, expected_hash);
    }

    #[test]
    fn test_scratch_pad() {
        let mut scratch_pad = ScratchPad::default();
        let mut input = Input::default();

        let hash = xelis_hash_scratch_pad(input.as_mut_slice().unwrap(), &mut scratch_pad).unwrap();
        let expected_hash = [
            0x0e, 0xbb, 0xbd, 0x8a, 0x31, 0xed, 0xad, 0xfe, 0x09, 0x8f, 0x2d, 0x77, 0x0d, 0x84,
            0xb7, 0x19, 0x58, 0x86, 0x75, 0xab, 0x88, 0xa0, 0xa1, 0x70, 0x67, 0xd0, 0x0a, 0x8f,
            0x36, 0x18, 0x22, 0x65,
        ];
        assert_eq!(hash, expected_hash);
    }
}

fn main() {
  let args: Vec<String> = std::env::args().collect();
  if args.len() != 2 {
      println!("Usage: {} <input>", args[0]);
      std::process::exit(1);
  }

  let input = args[1].as_bytes();
  let mut padded_input = [0u8; BYTES_ARRAY_INPUT];
  padded_input[..input.len()].copy_from_slice(input);

  let mut scratch_pad = [0u64; MEMORY_SIZE];
}