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
load_block_to_shared_memory(
  uint32* sm,                // shared memory output data
  uint words_per_slot,       // slot size in number of 32-bit words
  const uint32* d_stream,    // beginning of compressed input stream
  unsigned long long offset, // block offset in bits
  uint length                // block length in bits
)
{
  // block start within first 32-bit word
  const uint shift = (uint)offset & 31u;

  // advance stream to beginning 32-bit word of block
  d_stream += offset / 32;

  // advance shared-memory pointer to where block is to be stored
  sm += threadIdx.y * words_per_slot;

  // copy compressed data for one block one 32-bit word at a time
  for (uint i = threadIdx.x; i * 32 < length; i += threads) {
    // fetch two consecutive words and funnel shift them to one output word
    uint32 lo = d_stream[i];
    uint32 hi = 0;
    if ((i + 1) * 32 < shift + length)
      hi = d_stream[i + 1];
    sm[i] = __funnelshift_r(lo, hi, shift);
  }
}

// Read the input bitstreams from shared memory, align them relative to the
// final output alignment, compact all the aligned bitstreams in sm_out,
// then write all the data (coalesced) to global memory, using atomics only
// for the first and last elements
template <int tile_size, int num_tiles>
__device__
inline void
process(
  bool valid_stream,               //
  unsigned long long& offset0,     // offset in bits of the first bitstream of the block
  const unsigned long long offset, // offset in bits for this stream
  const uint& bit_length,          // length of this stream
  const uint& add_padding,         // padding at the end of the block, in bits
  const uint& tid,                 // global thread index inside the thread block
  uint32* sm_in,                   // shared memory containing the compressed input data
  uint32* sm_out,                  // shared memory to stage the compacted compressed data
  uint slot_words,                 // leading dimension of the shared memory (padded maxbits)
  uint32* sm_length,               // shared memory to compute a prefix-sum inside the block
  uint32* output                   // output pointer
)
{
  // all streams in the block will align themselves on the first stream of the block
  uint misaligned0 = offset0 & 31u;
  uint misaligned = offset & 31u;
  uint off_smin = threadIdx.y * slot_words;
  uint off_smout = ((int)(offset - offset0) + misaligned0) / 32;
  offset0 /= 32;

  if (valid_stream) {
    // loop over the whole bitstream (including misalignment), 32 bits per thread
    for (uint i = threadIdx.x; i * 32 < misaligned + bit_length; i += tile_size) {
      // merge two 32-bit words to create an aligned word
      uint32 v0 = i > 0 ? sm_in[off_smin + i - 1] : 0;
      uint32 v1 = sm_in[off_smin + i];
      v1 = __funnelshift_l(v0, v1, misaligned);

      // mask out neighbor bitstreams
      uint mask = 0xffffffffu;
      if (i == 0)
        mask &= 0xffffffffu << misaligned;
      if ((i + 1) * 32 > misaligned + bit_length)
        mask &= ~(0xffffffffu << ((misaligned + bit_length) & 31u));

      atomicAdd(sm_out + off_smout + i, v1 & mask);
    }
  }

  // First thread working on each bitstream writes the length in shared memory
  // Add zero-padding bits if needed (last bitstream of last chunk)
  // The extra bits in shared memory are already zeroed.
  if (threadIdx.x == 0)
    sm_length[threadIdx.y] = bit_length + add_padding;

  // this synchthreads protects sm_out and sm_length
  __syncthreads();

  // compute total length for the threadblock
  uint total_length = 0;
  for (uint i = tid & 31u; i < num_tiles; i += 32)
    total_length += sm_length[i];
  for (uint i = 1; i < 32; i *= 2)
    total_length += SHFL_XOR(total_length, i);

  // Write the shared memory output data to global memory, using all the threads
  for (int i = tid; i * 32 < misaligned0 + total_length; i += tile_size * num_tiles) {
    // Mask out the beginning and end of the block if unaligned
    uint mask = 0xffffffffu;
    if (i == 0)
      mask &= 0xffffffffu << misaligned0;
    if ((i + 1) * 32 > misaligned0 + total_length)
      mask &= ~(0xffffffffu << ((misaligned0 + total_length) & 31u));
    // reset the shared memory to zero for the next iteration
    uint value = sm_out[i];
    sm_out[i] = 0;
    // Write to global memory. Use atomicCAS for partially masked values
    // Working in-place, the output buffer has not been memset to zero
    if (mask == 0xffffffffu)
      output[offset0 + i] = value;
    else {
      uint assumed, old = output[offset0 + i];
      do {
        assumed = old;
        old = atomicCAS(output + offset0 + i, assumed, (assumed & ~mask) + (value & mask));
      } while (assumed != old);
    }
  }
}






    // In-place bitstream concatenation: compacting blocks containing different number
    // of bits, with the input blocks stored in bins of the same size
    // Using a 2D tile of threads,
    // threadIdx.y = Index of the stream
    // threadIdx.x = Threads working on the same stream
    // Must launch dim3(tile_size, num_tiles, 1) threads per block.
    // Offsets has a length of (nstreams_chunk + 1), offsets[0] is the offset in bits
    // where stream 0 starts, it must be memset to zero before launching the very first chunk,
    // and is updated at the end of this kernel.
    template <int tile_size, int num_tiles>
    __launch_bounds__(tile_size * num_tiles)
        __global__ void concat_bitstreams_chunk(uint *__restrict__ streams,
                                                unsigned long long *__restrict__ offsets,
                                                unsigned long long first_stream_chunk,
                                                int nstreams_chunk,
                                                bool last_chunk,
                                                int maxbits,
                                                int maxpad32)
    {
        cg::grid_group grid = cg::this_grid();
        __shared__ uint sm_length[num_tiles];
        extern __shared__ uint sm_in[];              // sm_in[num_tiles * maxpad32]
        uint *sm_out = sm_in + num_tiles * maxpad32; // sm_out[num_tiles * maxpad32 + 2]
        int tid = threadIdx.y * tile_size + threadIdx.x;
        int grid_stride = gridDim.x * num_tiles;
        int first_bitstream_block = blockIdx.x * num_tiles;
        int my_stream = first_bitstream_block + threadIdx.y;

        // Zero the output shared memory. Will be reset again inside process().
        for (int i = tid; i < num_tiles * maxpad32 + 2; i += tile_size * num_tiles)
            sm_out[i] = 0;

        // Loop on all the bitstreams of the current chunk, using the whole resident grid.
        // All threads must enter this loop, as they have to synchronize inside.
        for (int i = 0; i < nstreams_chunk; i += grid_stride)
        {
            bool valid_stream = my_stream + i < nstreams_chunk;
            bool active_thread_block = first_bitstream_block + i < nstreams_chunk;
            unsigned long long offset0 = 0;
            unsigned long long offset = 0;
            uint length_bits = 0;
            uint add_padding = 0;
            if (active_thread_block)
                offset0 = offsets[first_bitstream_block + i];

            if (valid_stream)
            {
                offset = offsets[my_stream + i];
                unsigned long long offset_bits = (first_stream_chunk + my_stream + i) * maxbits;
                unsigned long long next_offset_bits = offsets[my_stream + i + 1];
                length_bits = (uint)(next_offset_bits - offset);
                load_block_to_shared_memory<tile_size>(sm_in, maxpad32, streams, offset_bits, length_bits);
                if (last_chunk && (my_stream + i == nstreams_chunk - 1))
                {
                    uint partial = next_offset_bits & 63;
                    add_padding = (64 - partial) & 63;
                }
            }

            // Check if there is overlap between input and output at the grid level.
            // Grid sync if needed, otherwise just syncthreads to protect the shared memory.
            // All the threads launched must participate in a grid::sync
            int last_stream = min(nstreams_chunk, i + grid_stride);
            unsigned long long writing_to = (offsets[last_stream] + 31) / 32;
            unsigned long long reading_from = (first_stream_chunk + i) * maxbits;
            if (writing_to >= reading_from)
                grid.sync();
            else
                __syncthreads();

            // Compact the shared memory data of the whole thread block and write it to global memory
            if (active_thread_block)
                process<tile_size, num_tiles>(valid_stream, offset0, offset, length_bits, add_padding,
                                            tid, sm_in, sm_out, maxpad32, sm_length, streams);
        }

        // Reset the base of the offsets array, for the next chunk's prefix sum
        if (blockIdx.x == 0 && tid == 0)
            offsets[0] = offsets[nstreams_chunk];
    }


// launch stream compaction kernel using prescribed 
template <int tile_size, int num_tiles>
bool
compact_stream_launch(
  uint* streams,
  unsigned long long* chunk_offsets,
  unsigned long long first,
  int nstream_chunk,
  bool last_chunk,
  uint bits_per_slot,
  int num_sm
)
{
  const dim3 threads(tile_size, num_tiles, 1);
  uint words_per_slot = count_up(bits_per_slot, 32);
  void* kernel_args[] = {
    (void *)&streams,
    (void *)&chunk_offsets,
    (void *)&first,
    (void *)&nstream_chunk,
    (void *)&last_chunk,
    (void *)&bits_per_slot,
    (void *)&words_per_slot
  };

  // Increase the number of threads per ZFP block ("tile") as bits_per_slot increases
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
    concat_bitstreams_chunk<tile_size, num_tiles>,
    tile_size * num_tiles,
    shmem
  );
  max_blocks *= num_sm;
  max_blocks = min(nstream_chunk, max_blocks);

  cudaLaunchCooperativeKernel(
    (void *)concat_bitstreams_chunk<tile_size, num_tiles>,
    dim3(max_blocks, 1, 1),
    threads,
    kernel_args,
    shmem,
    0
  );

  return true;
}

bool
compact_stream_chunk(
  uint* streams,
  unsigned long long* chunk_offsets,
  unsigned long long first,
  int nstream_chunk,
  bool last_chunk,
  uint bits_per_slot,
  int num_sm
)
{
  // choose tile size to ensure the amount of shared memory stays below 48 KB
  // bits_per_slot <= 32 floor((48KB - 8B) / (2 * num_tiles * 4B))
  if (bits_per_slot <= 352)
    return compact_stream_launch< 1, 512>(streams, chunk_offsets, first, nstream_chunk, last_chunk, bits_per_slot, num_sm);
  else if (bits_per_slot <= 1504)
    return compact_stream_launch< 4, 128>(streams, chunk_offsets, first, nstream_chunk, last_chunk, bits_per_slot, num_sm);
  else if (bits_per_slot <= 6112)
    return compact_stream_launch<16,  32>(streams, chunk_offsets, first, nstream_chunk, last_chunk, bits_per_slot, num_sm);
  else if (bits_per_slot <= 24544)
    return compact_stream_launch<64,   8>(streams, chunk_offsets, first, nstream_chunk, last_chunk, bits_per_slot, num_sm);
  else {
    // zfp blocks are at most ZFP_MAX_BITS = 16658, so we should never arrive here
    return false;
  }
}

// the above replaces the following code
#if 0
    if (nbitsmax <= 352) {
      constexpr int tile_size = 1;
      constexpr int num_tiles = 512;
      size_t shmem = (2 * num_tiles * maxpad32 + 2) * sizeof(uint);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks,
        concat_bitstreams_chunk<tile_size, num_tiles>,
        tile_size * num_tiles,
        shmem
      );
      max_blocks *= num_sm;
      max_blocks = min(nstream_chunk, max_blocks);
      dim3 threads(tile_size, num_tiles, 1);
      cudaLaunchCooperativeKernel(
        (void *)concat_bitstreams_chunk<tile_size, num_tiles>,
        dim3(max_blocks, 1, 1),
        threads,
        kernel_args,
        shmem,
        0
      );
    }
    else if (nbitsmax <= 1504) {
      constexpr int tile_size = 4;
      constexpr int num_tiles = 128;
      size_t shmem = (2 * num_tiles * maxpad32 + 2) * sizeof(uint);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
                              concat_bitstreams_chunk<tile_size, num_tiles>,
                              tile_size * num_tiles, shmem);
      max_blocks *= num_sm;
      max_blocks = min(nstream_chunk, max_blocks);
      dim3 threads(tile_size, num_tiles, 1);
      cudaLaunchCooperativeKernel((void *)concat_bitstreams_chunk<tile_size, num_tiles>,
                    dim3(max_blocks, 1, 1), threads, kernel_args, shmem, 0);
    }
    else if (nbitsmax <= 6112) {
      constexpr int tile_size = 16;
      constexpr int num_tiles = 32;
      size_t shmem = (2 * num_tiles * maxpad32 + 2) * sizeof(uint);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
                              concat_bitstreams_chunk<tile_size, num_tiles>,
                              tile_size * num_tiles, shmem);
      max_blocks *= num_sm;
      max_blocks = min(nstream_chunk, max_blocks);
      dim3 threads(tile_size, num_tiles, 1);
      cudaLaunchCooperativeKernel((void *)concat_bitstreams_chunk<tile_size, num_tiles>,
                    dim3(max_blocks, 1, 1), threads, kernel_args, shmem, 0);
    }
    else if (nbitsmax <= 24541) { // Up to 24512 bits, so works even for largest 4D.
      constexpr int tile_size = 64;
      constexpr int num_tiles = 8;
      size_t shmem = (2 * num_tiles * maxpad32 + 2) * sizeof(uint);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
                              concat_bitstreams_chunk<tile_size, num_tiles>,
                              tile_size * num_tiles, shmem);
      max_blocks *= num_sm;
      max_blocks = min(nstream_chunk, max_blocks);
      dim3 threads(tile_size, num_tiles, 1);
      cudaLaunchCooperativeKernel((void *)concat_bitstreams_chunk<tile_size, num_tiles>,
                    dim3(max_blocks, 1, 1), threads, kernel_args, shmem, 0);
    }
  }
#endif

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
