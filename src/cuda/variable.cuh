#ifndef ZFP_CUDA_VARIABLE_CUH
#define ZFP_CUDA_VARIABLE_CUH

// implementation of variable-rate compression based on compacting in-place
// a stream of variable-length blocks initially stored in fixed-length slots

#include "shared.cuh"

namespace zfp {
namespace cuda {
namespace internal {

namespace cg = cooperative_groups;

// kernel for initializing prefix sum over zfp block lengths
__global__
void
copy_length_kernel(
  unsigned long long* d_offset, // block offsets; first is base of prefix sum
  const ushort* d_length,       // block lengths in bits
  size_t blocks_per_chunk       // number of blocks in chunk to process
)
{
  size_t block = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
  if (block < blocks_per_chunk)
    d_offset[block + 1] = d_length[block];
}

// initialize prefix sum by copying a chunk of 16-bit lengths to 64-bit offsets
void
copy_length_launch(
  unsigned long long* d_offset, // block offsets; first is base of prefix sum
  const ushort* d_length,       // block lengths in bits
  size_t blocks_per_chunk       // number of blocks in chunk to process
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
  unsigned long long base_offset, // global offset to first block in chunk
  unsigned long long offset,      // global output block offset in bits
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

    // mask out bits from neighboring blocks
/*
    // TODO: LSBs must already be zero as lo == 0
    if (i == 0)
      word &= 0xffffffffu << shift;
*/
    if ((i + 1) * 32 > shift + length)
      word &= ~(0xffffffffu << ((shift + length) & 31u));

    // store (partial) word in a thread-safe manner
    atomicAdd(sm_out + i, word);
  }
}

// Read the input bitstreams from shared memory, align them relative to the
// final output alignment, compact all the aligned bitstreams in sm_out,
// then write all the data (coalesced) to global memory, using atomics only
// for the first and last elements

// compact one subchunk of num_tiles blocks from sm_in to sm_out
template <int tile_size, int num_tiles>
__device__
inline void
process(
  bool valid_block,               // is block valid?
  unsigned long long base_offset, // offset in bits of the first bitstream of the block
  unsigned long long offset,      // block offset in bits
  uint length,                    // block length in bits
  uint padding,                   // padding at the end of the block, in bits
  uint tid,                       // global thread index inside the thread block
  const uint32* sm_in,            // shared memory containing the compressed input data
  uint32* sm_out,                 // shared memory to stage the compacted compressed data
  uint words_per_slot,            // leading dimension of the shared memory (padded maxbits)
  uint32* sm_length,              // shared memory to compute a prefix-sum inside the block
  uint32* d_stream                // output pointer
)
{
  const uint base_shift = base_offset & 31u;

  if (valid_block)
    copy_block<tile_size>(sm_out, base_offset, offset, length, sm_in, words_per_slot);

  // TODO: We can compute total_length from d_offset; no need to form prefix sum here

  // First thread working on each block writes the length in shared memory
  // Add zero-padding bits if needed (last bitstream of last chunk)
  // The extra bits in shared memory are already zeroed.
  if (threadIdx.x == 0)
    sm_length[threadIdx.y] = length + padding;

  // this synchthreads protects sm_out and sm_length
  __syncthreads();

  // compute total length for the threadblock
  uint total_length = 0;
  for (uint i = tid & 31u; i < num_tiles; i += 32)
    total_length += sm_length[i];
  for (uint i = 1; i < 32; i *= 2)
    total_length += SHFL_XOR(total_length, i);

//if (tid == 0)
//printf("base_offset=%llu, totlen=%u len=%u padding=%u\n", base_offset, total_length, length, padding);

  // advance output pointer to first word of chunk
  d_stream += base_offset / 32;

  // Write the shared memory output data to global memory, using all the threads
  for (uint i = tid; i * 32 < base_shift + total_length; i += tile_size * num_tiles) {
    // mask out the beginning and end of chunk if unaligned
    uint32 mask = 0xffffffffu;
    if (i == 0)
      mask &= 0xffffffffu << base_shift;
    if ((i + 1) * 32 > base_shift + total_length)
      mask &= ~(0xffffffffu << ((base_shift + total_length) & 31u));
    // fetch word and zero out for next chunk
    uint32 word = sm_out[i];
    sm_out[i] = 0;
    // Write to global memory. Use atomicCAS for partially masked values
    // Working in-place, the output buffer has not been memset to zero
    if (mask == 0xffffffffu)
      d_stream[i] = word;
    else {
      // deposit partial word (only needed for first and last word in chunk)
      uint32 assumed;
      uint32 old = d_stream[i];
      do {
        assumed = old;
        old = atomicCAS(d_stream + i, assumed, (assumed & ~mask) + (word & mask));
      } while (assumed != old);
    }
  }
}

// cooperative kernel for per-chunk stream compaction
template <int tile_size, int num_tiles>
__launch_bounds__(tile_size * num_tiles)
__global__
void
compact_stream_kernel(
  uint32 * __restrict__ d_stream,             // compressed bit stream
  unsigned long long * __restrict__ d_offset, // destination bit offsets
  size_t first_block,    // global index of first block in chunk
  uint blocks_per_chunk, // number of blocks per chunk
  bool last_chunk,       // is this the last chunk?
  uint bits_per_slot,    // number of bits per fixed-size slot holding a block
  uint words_per_slot    // number of 32-bit words per slot
)
{
  // In-place stream compaction of variable-length blocks initially stored in
  // d_stream as fixed-length slots of size bits_per_slot.  Compaction is done
  // in parallel in chunks of blocks_per_chunk blocks.  Each chunk is broken
  // into subchunks of num_tiles blocks (num_tiles in {8, 32, 128, 512}).
  // Compaction first loads a subchunk of blocks to 32-bit aligned slots in
  // shared memory, sm_in, then compacts that subchunk by concatenating the
  // blocks to sm_out, and then copyies the compacted data back to global
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
  __shared__ uint sm_length[num_tiles];
  extern __shared__ uint32 sm_in[];                           // sm_in[num_tiles * words_per_slot]
  uint32* sm_out = sm_in + num_tiles * words_per_slot;        // sm_out[num_tiles * words_per_slot + 2]
  const uint tid = threadIdx.x + threadIdx.y * tile_size;     // 
  const uint grid_stride = gridDim.x * num_tiles;             //
  const uint first_bitstream_block = blockIdx.x * num_tiles;  // first block in this subchunk
  const uint my_stream = first_bitstream_block + threadIdx.y; // first block within chunk assigned to this thread

  // zero-initialize compacted shared-memory buffer (also done in process())
  for (uint i = tid; i < num_tiles * words_per_slot + 2; i += num_tiles * tile_size)
    sm_out[i] = 0;

  // Loop on all the blocks of the current chunk, using the whole resident grid.
  // All threads must enter this loop, as they have to synchronize inside.
  for (uint i = 0; i < blocks_per_chunk; i += grid_stride) {
    const uint block = my_stream + i;

    bool valid_block = block < blocks_per_chunk;
    bool active_thread_block = first_bitstream_block + i < blocks_per_chunk;
    const unsigned long long base_offset = active_thread_block ? d_offset[first_bitstream_block + i] : 0; // 
    unsigned long long offset_out = 0;
    uint length = 0;
    uint padding = 0;

    if (valid_block) {
      unsigned long long offset_in = (first_block + block) * bits_per_slot;
      offset_out = d_offset[block]; // 
      length = (uint)(d_offset[block + 1] - offset_out);
      load_block<tile_size>(sm_in, words_per_slot, d_stream, offset_in, length);
      // pad last block in stream to align stream on 64-bit boundary
      if (last_chunk && (block == blocks_per_chunk - 1)) {
        uint partial = d_offset[blocks_per_chunk] & 63u;
        padding = (64 - partial) & 63u;
      }
    }

    // Check if there is overlap between input and output at the grid level.
    // Grid sync if needed, otherwise just syncthreads to protect the shared memory.
    // All the threads launched must participate in a grid::sync
    uint last_block = min(i + grid_stride, blocks_per_chunk);

    // TODO: bug here; writing_to is in words, reading from is in bits
    unsigned long long writing_to = (d_offset[last_block] + 31) / 32;
    unsigned long long reading_from = (first_block + i) * bits_per_slot;
    if (writing_to >= reading_from)
      grid.sync();
    else
      __syncthreads();

    // Compact the shared memory data of the whole thread block and write it to global memory
    if (active_thread_block)
{
//if (tid == 0)
//printf("block=%u tile_size=%u num_tiles=%u blocks_per_chunk=%u grid_stride=%u totlen=%llu\n", first_bitstream_block, tile_size, num_tiles, blocks_per_chunk, grid_stride, d_offset[first_bitstream_block + num_tiles] - d_offset[first_bitstream_block]);
      process<tile_size, num_tiles>(valid_block, base_offset, offset_out, length, padding, tid, sm_in, sm_out, words_per_slot, sm_length, d_stream);
}
  }

  // update the base of the offset array for the next chunk's prefix sum
  if (blockIdx.x == 0 && tid == 0)
    d_offset[0] = d_offset[blocks_per_chunk];
}

// launch stream compaction kernel using prescribed 
template <int tile_size, int num_tiles>
bool
compact_stream_launch(
  uint32* d_stream,
  unsigned long long* d_offset,
  size_t first_block,
  uint blocks_per_chunk,
  bool last_chunk,
  uint bits_per_slot,
  uint processors
)
{
  const dim3 threads(tile_size, num_tiles, 1);
  const uint words_per_slot = count_up(bits_per_slot, 32);
  void* kernel_args[] = {
    (void *)&d_stream,
    (void *)&d_offset,
    (void *)&first_block,
    (void *)&blocks_per_chunk,
    (void *)&last_chunk,
    (void *)&bits_per_slot,
    (void *)&words_per_slot
  };

  // Increase the number of threads per zfp block ("tile") as bits_per_slot increases
  // Compromise between coalescing, inactive threads and shared memory size <= 48KB
  // Total shared memory used = (2 * num_tiles * words_per_slot + 2) x 32-bit dynamic shared memory
  // and num_tiles x 32-bit static shared memory.
  // The extra 2 elements of dynamic shared memory are needed to handle unaligned output data
  // and potential zero-padding to the next multiple of 64 bits.
  // Block sizes set so that the shared memory stays < 48KB.

  int max_blocks = 0;
  size_t shmem = (2 * num_tiles * words_per_slot + 2) * sizeof(uint32);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_blocks,
    compact_stream_kernel<tile_size, num_tiles>,
    tile_size * num_tiles,
    shmem
  );
  max_blocks *= processors;
  max_blocks = min(blocks_per_chunk, max_blocks);

  cudaLaunchCooperativeKernel(
    (void *)compact_stream_kernel<tile_size, num_tiles>,
    dim3(max_blocks, 1, 1),
    threads,
    kernel_args,
    shmem,
    0
  );

  return true;
}

// compact a single chunk of blocks
bool
compact_stream_chunk(
  uint32* d_stream,             // compressed bit stream
  unsigned long long* d_offset, // global bit offsets to blocks in chunk
  size_t first_block,           // index of first block in chunk
  uint blocks_per_chunk,        // number of blocks per chunk
  bool last_chunk,              // is this the last chunk?
  uint bits_per_slot,           // fixed-size slot size in bits
  uint processors               // number of device multiprocessors
)
{
  const uint bytes_per_slot = count_up(bits_per_slot, 32) * sizeof(uint32);
  const size_t shared_memory = 48 * 1024 - 2 * sizeof(uint32);

#if 0
  // bits_per_slot <= 32 floor((48KB - 8B) / (2 * num_tiles * 4B))
  if (bits_per_slot <= 352)
    return compact_stream_launch< 1, 512>(d_stream, d_offset, first_block, blocks_per_chunk, last_chunk, bits_per_slot, processors);
  else if (bits_per_slot <= 1504)
    return compact_stream_launch< 4, 128>(d_stream, d_offset, first_block, blocks_per_chunk, last_chunk, bits_per_slot, processors);
  else if (bits_per_slot <= 6112)
    return compact_stream_launch<16,  32>(d_stream, d_offset, first_block, blocks_per_chunk, last_chunk, bits_per_slot, processors);
  else if (bits_per_slot <= 24544)
    return compact_stream_launch<64,   8>(d_stream, d_offset, first_block, blocks_per_chunk, last_chunk, bits_per_slot, processors);
#else
  // choose number of tiles such that shared memory usage is at most 48 KB
  if (512 * 2 * bytes_per_slot <= shared_memory)
    return compact_stream_launch< 1, 512>(d_stream, d_offset, first_block, blocks_per_chunk, last_chunk, bits_per_slot, processors);
  else if (128 * 2 * bytes_per_slot <= shared_memory)
    return compact_stream_launch< 4, 128>(d_stream, d_offset, first_block, blocks_per_chunk, last_chunk, bits_per_slot, processors);
  else if (32 * 2 * bytes_per_slot <= shared_memory)
    return compact_stream_launch<16,  32>(d_stream, d_offset, first_block, blocks_per_chunk, last_chunk, bits_per_slot, processors);
  else if (8 * 2 * bytes_per_slot <= shared_memory)
    return compact_stream_launch<64,   8>(d_stream, d_offset, first_block, blocks_per_chunk, last_chunk, bits_per_slot, processors);
#endif
  else {
    // zfp blocks are at most ZFP_MAX_BITS = 16658 bits < 2084 bytes;
    // should never arrive here
    return false;
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
  size_t lcubtemp;
  void* d_cubtemp;

  if (!setup_device_compact(&chunk_size, &d_offset, &lcubtemp, &d_cubtemp, processors))
    return 0;

//printf("chunk_size=%zu\n", chunk_size);

  // perform compaction one chunk of blocks at a time
  for (size_t block = 0; block < blocks && success; block += chunk_size) {
    // determine chunk size cur_blocks
    size_t blocks_per_chunk = chunk_size;
    bool last_chunk = false;
    if (block + blocks_per_chunk > blocks) {
      blocks_per_chunk = blocks - block;
      last_chunk = true;
    }

    // initialize block offsets to block lengths
    copy_length_launch(d_offset, d_length + block, blocks_per_chunk);

    // compute prefix sum to turn block lengths into offsets
    cub::DeviceScan::InclusiveSum(d_cubtemp, lcubtemp, d_offset, d_offset, blocks_per_chunk + 1);

    // compact the stream in place
    if (!compact_stream_chunk((uint32*)d_stream, d_offset, block, blocks_per_chunk, last_chunk, bits_per_slot, processors))
      success = false;
  }

  // update compressed size and pad to whole words
  unsigned long long bits_written = 0;
  if (success) {
    cudaMemcpy(&bits_written, d_offset, sizeof(bits_written), cudaMemcpyDeviceToHost);
    bits_written = round_up(bits_written, sizeof(Word) * CHAR_BIT);
  }

  // free temporary buffers
  cleanup_device(NULL, d_offset);
  cleanup_device(NULL, d_cubtemp);

  return bits_written;
}

} // namespace internal
} // namespace cuda
} // namespace zfp

#endif
