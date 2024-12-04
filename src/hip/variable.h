#include "hip/hip_runtime.h"
#ifndef ZFP_HIP_VARIABLE_H
#define ZFP_HIP_VARIABLE_H

// implementation of variable-rate compression based on compacting in-place
// a stream of variable-length blocks initially stored in fixed-length slots

#include "shared.h"

namespace zfp {
namespace hip {
namespace internal {

namespace cg = cooperative_groups;

// kernel for initializing prefix sum over zfp block lengths
__global__
void
copy_length_kernel(
  unsigned long long* d_offset, // block offsets; first is base of prefix sum
  const ushort* d_length,       // block lengths in bits
  uint blocks_per_chunk         // number of blocks in chunk to process
)
{
  uint block = threadIdx.x + blockIdx.x * blockDim.x;
  if (block < blocks_per_chunk)
    d_offset[block + 1] = d_length[block];
}

// initialize prefix sum by copying a chunk of 16-bit lengths to 64-bit offsets
void
copy_length_launch(
  unsigned long long* d_offset, // block offsets; first is base of prefix sum
  const ushort* d_length,       // block lengths in bits
  uint blocks_per_chunk         // number of blocks in chunk to process
)
{
  dim3 blocks((int)count_up(blocks_per_chunk, 1024), 1, 1);
  copy_length_kernel<<<blocks, 1024>>>(d_offset, d_length, blocks_per_chunk);
}

// load a single unaligned block to a 32-bit aligned slot in shared memory
template <uint threads>
__device__
inline void
load_block(
  uint32* sm_stream,         // shared-memory buffer of 32-bit aligned slots
  uint words_per_slot,       // slot size in number of 32-bit words
  const uint32* d_stream,    // beginning of uncompacted input stream
  unsigned long long offset, // block offset in bits
  uint length                // block length in bits
)
{
  // block start within first 32-bit word
  const uint shift = (uint)offset & 31u;

  // advance stream to beginning 32-bit word of block
  d_stream += offset / 32;

  // advance shared-memory pointer to where block is to be stored
  sm_stream += threadIdx.y * words_per_slot;

  // copy compressed data for one block one 32-bit word at a time
  for (uint i = threadIdx.x; i * 32 < length; i += threads) {
    // fetch two consecutive words and funnel shift them to one output word
    uint32 lo = d_stream[i];
    uint32 hi = 0;
    if ((i + 1) * 32 < shift + length)
      hi = d_stream[i + 1];
    sm_stream[i] = __funnelshift_r(lo, hi, shift);
  }
}

// copy a single block from its 32-bit aligned slot to its compacted location
template <uint threads>
__device__
inline void
copy_block(
  uint32* sm_out,                 // shared-memory pointer to compacted chunk
  unsigned long long base_offset, // global offset to first block in subchunk
  unsigned long long offset,      // global offset to this block in bits
  uint length,                    // block length in bits
  const uint32* sm_in,            // shared-memory pointer to uncompacted data
  uint words_per_slot             // slot size in number of 32-bit words
)
{
  // in-word offset to first block in chunk
  const uint base_shift = (uint)base_offset & 31u;

  // block start within first 32-bit word
  const uint shift = (uint)offset & 31u;

  // advance shared-memory pointer to block source data
  sm_in += threadIdx.y * words_per_slot;

  // advance pointer to block destination data
  sm_out += (offset - base_offset + base_shift) / 32;

  for (uint i = threadIdx.x; i * 32 < shift + length; i += threads) {
    // fetch two consecutive words and funnel shift them to one output word
    uint32 lo = i > 0 ? sm_in[i - 1] : 0;
    uint32 hi = sm_in[i];
    uint32 word = __funnelshift_l(lo, hi, shift);

    // mask out bits from next block
    if ((i + 1) * 32 > shift + length)
      word &= ~(0xffffffffu << ((shift + length) & 31u));

    // store (partial) word in a thread-safe manner
    atomicAdd(sm_out + i, word);
  }
}

// atomically write masked bits of word to *ptr
template <typename T>
__device__
inline void
atomic_deposit(T* ptr, T word, T mask)
{
  if (~mask) {
    // deposit partial word using atomicCAS
    T expected;
    T old = *ptr;
    do {
      expected = old;
      T value = (word & mask) + (expected & ~mask);
      old = atomicCAS(ptr, expected, value);
    } while (old != expected);
  }
  else
    *ptr = word;
}

// copy one subchunk of num_tiles blocks from shared to global memory
template <int tile_size, int num_tiles>
__device__
inline void
store_subchunk(
  uint32* d_stream,          // output pointer
  unsigned long long offset, // bit offset to first block in subchunk
  uint length,               // subchunk length in bits
  uint32* sm_src,            // compacted compressed data in shared memory
  uint tid                   // global thread index inside the thread block
)
{
  // Copy compacted subchunk from shared memory to its final location in global
  // memory using coalesced writes.  Use atomic only for the first and last
  // word of the subchunk.

  const uint begin = offset & 31u;
  const uint end = begin + length;

  // advance output pointer to first word of subchunk
  d_stream += offset / 32;

  // use all threads to copy compacted subchunk to global memory
  for (uint i = tid; i * 32 < end; i += num_tiles * tile_size) {
    // fetch word and zero out for next subchunk
    uint32 word = sm_src[i];
    sm_src[i] = 0;
    // mask out the beginning and end of subchunk if unaligned
    uint32 mask = 0xffffffffu;
    if (i == 0)
      mask &= 0xffffffffu << begin;
    if ((i + 1) * 32 > end)
      mask &= ~(0xffffffffu << (end & 31u));
    // write (possibly partial) word to global memory
    atomic_deposit(d_stream + i, word, mask);
  }
}

// cooperative kernel for stream compaction of one chunk of blocks
template <int tile_size, int num_tiles>
__launch_bounds__(tile_size * num_tiles)
__global__
void
compact_stream_kernel(
  uint32* __restrict__ d_stream,             // compressed bit stream
  unsigned long long* __restrict__ d_offset, // destination bit offsets
  size_t first_block,    // global index of first block in chunk
  uint blocks_per_chunk, // number of blocks per chunk
  uint bits_per_slot,    // number of bits per fixed-size slot holding a block
  uint words_per_slot    // number of 32-bit words per slot
)
{
  // In-place stream compaction of variable-length blocks initially stored in
  // d_stream as fixed-length slots of size bits_per_slot.  Compaction is done
  // in parallel in chunks of blocks_per_chunk blocks.  Each chunk is broken
  // into subchunks of num_tiles (one of {8, 32, 128, 512}) blocks each.
  // Compaction first loads a subchunk of blocks to 32-bit aligned slots in
  // shared memory, sm_in, then compacts that subchunk by concatenating the
  // blocks to sm_out, and then copies the compacted data back to global
  // memory in d_stream.  This repeats for all subchunks of the chunk.  The
  // destination offset of each block in the chunk has already been computed
  // as a prefix sum over block lengths and stored in d_offset, which holds
  // blocks_per_chunk + 1 bit offsets.  At the end of the kernel, d_offset[0]
  // is set to point to the beginning of the next chunk.
  //
  // This parallel compaction is executed by num_tiles * tile_size = 512
  // threads, with one such thread block processing one subchunk at a time.
  // Thread indices are:
  //
  //   threadIdx.x = thread among tile_size threads working on the same block
  //   threadIdx.y = block index within subchunk
  //
  // The caller must launch dim3(tile_size, num_tiles, 1) threads per thread
  // block.  The caller also allocates shared memory for sm_in and sm_out.

  cg::grid_group grid = cg::this_grid();
  // sm_in[num_tiles * words_per_slot]
  extern __shared__ uint32 sm_in[];
  // sm_out[num_tiles * words_per_slot + 2]
  uint32* sm_out = sm_in + num_tiles * words_per_slot;
  // thread within thread block
  const uint tid = threadIdx.x + threadIdx.y * tile_size;
  // number of blocks per group
  const uint blocks_per_group = gridDim.x * num_tiles;
  // first block in this subchunk
  const uint first_subchunk_block = blockIdx.x * num_tiles;

  // zero-initialize compacted buffer (also done in store_subchunk())
  for (uint i = tid; i < num_tiles * words_per_slot + 2; i += num_tiles * tile_size)
    sm_out[i] = 0;

  // compact chunk one group at a time
  for (uint i = 0; i < blocks_per_chunk; i += blocks_per_group) {
    // first block in this subchunk
    const uint base_block = first_subchunk_block + i;
    // block assigned to this thread
    const uint block = base_block + threadIdx.y;
    // is this thread block assigned any compressed blocks?
    const bool active_thread_block = (base_block < blocks_per_chunk);
    // is this thread assigned to valid block?
    const bool valid_block = (block < blocks_per_chunk);
    // destination offset to beginning of subchunk in compacted stream
    const unsigned long long base_offset = active_thread_block ? d_offset[base_block] : 0;

    unsigned long long offset_out = 0;
    uint length = 0;

    if (valid_block) {
      // source offset within uncompacted stream
      unsigned long long offset_in = (first_block + block) * bits_per_slot;
      // destination offset within compacted stream
      offset_out = d_offset[block];
      // bit length of this block
      length = (uint)(d_offset[block + 1] - offset_out);
      // buffer block in fixed-size slot in shared memory
      load_block<tile_size>(sm_in, words_per_slot, d_stream, offset_in, length);
      // compact subchunk by copying block to target location in shared memory
      __syncthreads();
      copy_block<tile_size>(sm_out, base_offset, offset_out, length, sm_in, words_per_slot);
    }

    // synchronize across group if there is overlap between input and output
    const uint last_block = min(i + blocks_per_group, blocks_per_chunk);
    const size_t output_end = d_offset[last_block] / 32;
    const size_t input_begin = (first_block + i) * bits_per_slot / 32;
    if (output_end < input_begin)
      __syncthreads();
    else
      grid.sync();

    // copy compacted subchunk from shared memory to global memory
    if (active_thread_block) {
      unsigned long long last_offset = d_offset[min(base_block + num_tiles, blocks_per_chunk)];
      uint subchunk_length = (uint)(last_offset - base_offset);
      // store compacted subchunk to global memory
      store_subchunk<tile_size, num_tiles>(d_stream, base_offset, subchunk_length, sm_out, tid);
    }
  }

  // update the base of the offset array for the next chunk's prefix sum
  if (blockIdx.x == 0 && tid == 0)
    d_offset[0] = d_offset[blocks_per_chunk];
}

// launch stream compaction kernel for one chunk
template <int tile_size, int num_tiles>
bool
compact_stream_launch(
  uint32* d_stream,             // compressed bit stream
  unsigned long long* d_offset, // global bit offsets to blocks in chunk
  size_t first_block,           // index of first block in chunk
  uint blocks_per_chunk,        // number of blocks per chunk
  uint bits_per_slot,           // fixed-size slot size in bits
  uint processors               // number of device multiprocessors
)
{
  // Assign number of threads ("tile_size") per zfp block in proportion to
  // bits_per_slot.  Compromise between coalescing, keeping threads active,
  // and limiting shared memory usage.  The total dynamic shared memory used
  // equals (2 * num_tiles * words_per_slot + 2) 32-bit words.  The extra
  // two words of shared memory are needed to handle output data that is not
  // aligned on 32-bit words.  The number of zfp blocks per thread block
  // ("num_tiles") is set to ensure that shared memory is at most 48 KB.

  const uint words_per_slot = count_up(bits_per_slot, 32);
  const size_t shmem = (2 * num_tiles * words_per_slot + 2) * sizeof(uint32);

  // compute number of blocks to process concurrently
  int thread_blocks = 0;
  hipOccupancyMaxActiveBlocksPerMultiprocessor(
    &thread_blocks,
    compact_stream_kernel<tile_size, num_tiles>,
    tile_size * num_tiles,
    shmem
  );
  thread_blocks *= processors;
  thread_blocks = min(thread_blocks, (int)count_up(blocks_per_chunk, num_tiles));

  void* kernel_args[] = {
    (void *)&d_stream,
    (void *)&d_offset,
    (void *)&first_block,
    (void *)&blocks_per_chunk,
    (void *)&bits_per_slot,
    (void *)&words_per_slot
  };

  return hipLaunchCooperativeKernel(
    (void *)compact_stream_kernel<tile_size, num_tiles>,
    dim3(thread_blocks, 1, 1),
    dim3(tile_size, num_tiles, 1),
    kernel_args,
    shmem,
    0
  ) == hipSuccess;
}

// compact a single chunk of blocks
bool
compact_stream_chunk(
  uint32* d_stream,             // compressed bit stream
  unsigned long long* d_offset, // global bit offsets to blocks in chunk
  size_t first_block,           // index of first block in chunk
  uint blocks_per_chunk,        // number of blocks per chunk
  uint bits_per_slot,           // fixed-size slot size in bits
  uint processors               // number of device multiprocessors
)
{
  const uint bytes_per_slot = count_up(bits_per_slot, 32) * sizeof(uint32);
  const size_t shared_memory = 48 * 1024 - 2 * sizeof(uint32);

  // choose number of tiles such that shared memory usage is at most 48 KB
  if (512 * 2 * bytes_per_slot <= shared_memory) // bits_per_slot <= 352
    return compact_stream_launch< 1, 512>(d_stream, d_offset, first_block, blocks_per_chunk, bits_per_slot, processors);
  if (128 * 2 * bytes_per_slot <= shared_memory) // bits_per_slot <= 1504
    return compact_stream_launch< 4, 128>(d_stream, d_offset, first_block, blocks_per_chunk, bits_per_slot, processors);
  if ( 32 * 2 * bytes_per_slot <= shared_memory) // bits_per_slot <= 6112
    return compact_stream_launch<16,  32>(d_stream, d_offset, first_block, blocks_per_chunk, bits_per_slot, processors);
  if (  8 * 2 * bytes_per_slot <= shared_memory) // bits_per_slot <= 24544
    return compact_stream_launch<64,   8>(d_stream, d_offset, first_block, blocks_per_chunk, bits_per_slot, processors);

  // zfp blocks are at most ZFP_MAX_BITS = 16658 bits < 2084 bytes;
  // should never arrive here

  return false;
}

// zero-pad stream to align it on a whole word
__global__
void
align_stream_kernel(
  Word* d_stream,              // compacted, compressed stream
  unsigned long long* d_offset // offset to end of stream
)
{
  const size_t alignment = sizeof(Word) * CHAR_BIT;
  const unsigned long long offset = *d_offset;
  const uint shift = (uint)(offset % alignment);

  if (shift) {
    // mask out any nonzero bits at the end and advance offset
    d_stream[offset / alignment] &= ~(~Word(0) << shift);
    *d_offset = round_up(offset, alignment);
  }
}

// compact in place variable-length blocks stored in fixed-length slots
unsigned long long
compact_stream(
  Word* d_stream,         // pointer to compressed blocks
  uint bits_per_slot,     // fixed size of slots holding variable-length blocks
  const ushort* d_length, // lengths of zfp blocks in bits
  size_t blocks,          // number of zfp blocks
  size_t processors       // number of device multiprocessors
)
{
  bool success = true;
  unsigned long long* d_offset;
  size_t chunk_size;
  size_t cubtmp_size;
  void* d_cubtmp;

  if (!setup_device_compact(&chunk_size, &d_offset, &cubtmp_size, &d_cubtmp, processors))
    return 0;

  // perform compaction one chunk of blocks at a time
  for (size_t block = 0; block < blocks && success; block += chunk_size) {
    // determine chunk size
    size_t blocks_per_chunk = min(chunk_size, blocks - block);

    // initialize block offsets to block lengths
    copy_length_launch(d_offset, d_length + block, blocks_per_chunk);

    // compute prefix sum to turn block lengths into offsets
    hipcub::DeviceScan::InclusiveSum(d_cubtmp, cubtmp_size, d_offset, d_offset, blocks_per_chunk + 1);

    // compact the stream in place
    if (!compact_stream_chunk((uint32*)d_stream, d_offset, block, blocks_per_chunk, bits_per_slot, processors))
      success = false;
  }

  // update compressed size and pad to whole words
  unsigned long long bits_written = 0;
  if (success) {
    align_stream_kernel<<<1, 1>>>(d_stream, d_offset);
    hipMemcpy(&bits_written, d_offset, sizeof(bits_written), hipMemcpyDeviceToHost);
  }

  // free temporary buffers
  cleanup_device(d_offset);
  cleanup_device(d_cubtmp);

  return bits_written;
}

} // namespace internal
} // namespace hip
} // namespace zfp

#endif
