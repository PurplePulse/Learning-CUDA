#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "file_io.h"
#include "search.h"

// 这是第一版 baseline：一个 CUDA 线程计算一对 query/vector 的完整距离。
// 优点是代码直观，缺点是暂时没有在线程间共同处理一个向量。
__device__ __forceinline__ float loadInput(const float* values,
                                           std::uint64_t index) {
  return values[index];
}

__device__ __forceinline__ float loadInput(const __half* values,
                                           std::uint64_t index) {
  return __half2float(values[index]);
}

template <typename InputType>
__global__ void exactDistanceKernel(const InputType* database,
                                    const InputType* queries, float* scores,
                                    std::uint64_t num_vectors,
                                    std::uint32_t dim,
                                    std::uint32_t batch_count, Metric metric) {
  const std::uint64_t vector_id =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint32_t query_in_batch = blockIdx.y;
  if (vector_id >= num_vectors || query_in_batch >= batch_count) {
    return;
  }

  const std::uint64_t vector_offset = vector_id * dim;
  const std::uint64_t query_offset =
      static_cast<std::uint64_t>(query_in_batch) * dim;

  float dot = 0.0F;
  float query_norm = 0.0F;
  float vector_norm = 0.0F;
  float squared_l2 = 0.0F;
  for (std::uint32_t d = 0; d < dim; ++d) {
    const float q = loadInput(queries, query_offset + d);
    const float x = loadInput(database, vector_offset + d);
    if (metric == Metric::kL2) {
      const float difference = q - x;
      squared_l2 = fmaf(difference, difference, squared_l2);
    } else {
      dot = fmaf(q, x, dot);
      if (metric == Metric::kCosine) {
        query_norm = fmaf(q, q, query_norm);
        vector_norm = fmaf(x, x, vector_norm);
      }
    }
  }

  float score = dot;
  if (metric == Metric::kL2) {
    score = squared_l2;
  } else if (metric == Metric::kCosine) {
    score = (query_norm == 0.0F || vector_norm == 0.0F)
                ? 0.0F
                : dot * rsqrtf(query_norm * vector_norm);
  }
  scores[static_cast<std::uint64_t>(query_in_batch) * num_vectors + vector_id] =
      score;
}

// 优化版距离计算：一个 warp（32 个线程）共同处理一个向量。
// lane 0 计算维度 0、32、64...，lane 1 计算 1、33、65...，因此访问是连续的。
template <typename InputType>
__global__ void exactDistanceWarpKernel(const InputType* database,
                                        const InputType* queries, float* scores,
                                        std::uint64_t num_vectors,
                                        std::uint32_t dim,
                                        std::uint32_t batch_count,
                                        Metric metric) {
  constexpr std::uint32_t kWarpSize = 32;
  const std::uint32_t lane = threadIdx.x % kWarpSize;
  const std::uint32_t warp_in_block = threadIdx.x / kWarpSize;
  const std::uint32_t warps_per_block = blockDim.x / kWarpSize;
  const std::uint64_t first_vector_id =
      static_cast<std::uint64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const std::uint32_t query_in_batch = blockIdx.y;
  if (query_in_batch >= batch_count) {
    return;
  }
  const std::uint64_t query_offset =
      static_cast<std::uint64_t>(query_in_batch) * dim;

  // grid-stride 循环让同一个 warp 连续处理多个向量，避免启动过多 block。
  const std::uint64_t vector_stride =
      static_cast<std::uint64_t>(gridDim.x) * warps_per_block;
  for (std::uint64_t vector_id = first_vector_id; vector_id < num_vectors;
       vector_id += vector_stride) {
    const std::uint64_t vector_offset = vector_id * dim;
    float dot = 0.0F;
    float query_norm = 0.0F;
    float vector_norm = 0.0F;
    float squared_l2 = 0.0F;

    for (std::uint32_t d = lane; d < dim; d += kWarpSize) {
      const float q = loadInput(queries, query_offset + d);
      const float x = loadInput(database, vector_offset + d);
      if (metric == Metric::kL2) {
        const float difference = q - x;
        squared_l2 = fmaf(difference, difference, squared_l2);
      } else {
        dot = fmaf(q, x, dot);
        if (metric == Metric::kCosine) {
          query_norm = fmaf(q, q, query_norm);
          vector_norm = fmaf(x, x, vector_norm);
        }
      }
    }

    // 每次让后半部分 lane 的结果加到前半部分，最终总和位于 lane 0。
    for (std::uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      squared_l2 += __shfl_down_sync(0xffffffff, squared_l2, offset);
      dot += __shfl_down_sync(0xffffffff, dot, offset);
      query_norm += __shfl_down_sync(0xffffffff, query_norm, offset);
      vector_norm += __shfl_down_sync(0xffffffff, vector_norm, offset);
    }

    if (lane == 0) {
      float score = dot;
      if (metric == Metric::kL2) {
        score = squared_l2;
      } else if (metric == Metric::kCosine) {
        score = (query_norm == 0.0F || vector_norm == 0.0F)
                    ? 0.0F
                    : dot * rsqrtf(query_norm * vector_norm);
      }
      scores[static_cast<std::uint64_t>(query_in_batch) * num_vectors +
             vector_id] = score;
    }
  }
}

__device__ bool isBetter(float candidate_score, std::uint64_t candidate_id,
                         float current_score, std::uint64_t current_id,
                         Metric metric) {
  if (candidate_score == current_score) {
    return candidate_id < current_id;
  }
  if (metric == Metric::kL2) {
    return candidate_score < current_score;
  }
  return candidate_score > current_score;
}

// 教学版 GPU Top-K：每个 query 使用一个线程，按顺序维护有序的前 K 个结果。
// 它的并行度不高，但逻辑清楚，并且避免把完整距离矩阵复制回 CPU。
// 下一版会用一个 block 共同处理一个 query，作为明确的性能优化步骤。
__global__ void simpleTopKKernel(const float* scores, float* top_scores,
                                 std::uint64_t* top_ids,
                                 std::uint64_t num_vectors, std::uint32_t top_k,
                                 Metric metric) {
  const std::uint32_t query_in_batch = blockIdx.x;
  if (threadIdx.x != 0) {
    return;
  }

  constexpr std::uint32_t kMaximumK = 100;
  float best_scores[kMaximumK];
  std::uint64_t best_ids[kMaximumK];
  // 使用 float 最大值作为哨兵，避免依赖额外的 CUDA 数学常量头文件。
  constexpr float kFloatMaximum = 3.402823466e+38F;
  const float worst_score =
      (metric == Metric::kL2) ? kFloatMaximum : -kFloatMaximum;
  for (std::uint32_t rank = 0; rank < top_k; ++rank) {
    best_scores[rank] = worst_score;
    best_ids[rank] = UINT64_MAX;
  }

  const float* query_scores =
      scores + static_cast<std::uint64_t>(query_in_batch) * num_vectors;
  for (std::uint64_t vector_id = 0; vector_id < num_vectors; ++vector_id) {
    const float score = query_scores[vector_id];
    for (std::uint32_t rank = 0; rank < top_k; ++rank) {
      if (isBetter(score, vector_id, best_scores[rank], best_ids[rank],
                   metric)) {
        for (std::uint32_t shift = top_k - 1; shift > rank; --shift) {
          best_scores[shift] = best_scores[shift - 1];
          best_ids[shift] = best_ids[shift - 1];
        }
        best_scores[rank] = score;
        best_ids[rank] = vector_id;
        break;
      }
    }
  }

  const std::uint64_t output_offset =
      static_cast<std::uint64_t>(query_in_batch) * top_k;
  for (std::uint32_t rank = 0; rank < top_k; ++rank) {
    top_scores[output_offset + rank] = best_scores[rank];
    top_ids[output_offset + rank] = best_ids[rank];
  }
}

// 并行版 GPU Top-K：一个 block 共同处理一个 query。
// 每一轮中，各线程先在自己负责的候选中找最好结果，再用共享内存归约出全局最好结果。
// 找到一个结果后将其标记为“最差”，重复 K 轮。这不是最终最优算法，但比单线程版本
// 更容易理解，也能直接展示线程协作带来的收益。
__global__ void blockTopKKernel(float* scores,
                                const std::uint64_t* candidate_ids,
                                float* top_scores, std::uint64_t* top_ids,
                                std::uint64_t num_vectors, std::uint32_t top_k,
                                Metric metric) {
  constexpr std::uint32_t kThreads = 256;
  __shared__ float shared_scores[kThreads];
  __shared__ std::uint64_t shared_ids[kThreads];
  __shared__ std::uint64_t shared_positions[kThreads];

  const std::uint32_t query_in_batch = blockIdx.x;
  const std::uint32_t thread_id = threadIdx.x;
  float* query_scores =
      scores + static_cast<std::uint64_t>(query_in_batch) * num_vectors;
  constexpr float kFloatMaximum = 3.402823466e+38F;
  const float worst_score =
      (metric == Metric::kL2) ? kFloatMaximum : -kFloatMaximum;

  for (std::uint32_t rank = 0; rank < top_k; ++rank) {
    float local_best_score = worst_score;
    std::uint64_t local_best_id = UINT64_MAX;
    std::uint64_t local_best_position = UINT64_MAX;

    for (std::uint64_t position = thread_id; position < num_vectors;
         position += blockDim.x) {
      const float score = query_scores[position];
      const std::uint64_t vector_id =
          candidate_ids == nullptr
              ? position
              : candidate_ids[static_cast<std::uint64_t>(query_in_batch) *
                                  num_vectors +
                              position];
      if (isBetter(score, vector_id, local_best_score, local_best_id, metric)) {
        local_best_score = score;
        local_best_id = vector_id;
        local_best_position = position;
      }
    }

    shared_scores[thread_id] = local_best_score;
    shared_ids[thread_id] = local_best_id;
    shared_positions[thread_id] = local_best_position;
    __syncthreads();

    for (std::uint32_t stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (thread_id < stride &&
          isBetter(shared_scores[thread_id + stride],
                   shared_ids[thread_id + stride], shared_scores[thread_id],
                   shared_ids[thread_id], metric)) {
        shared_scores[thread_id] = shared_scores[thread_id + stride];
        shared_ids[thread_id] = shared_ids[thread_id + stride];
        shared_positions[thread_id] = shared_positions[thread_id + stride];
      }
      __syncthreads();
    }

    if (thread_id == 0) {
      const std::uint64_t output_offset =
          static_cast<std::uint64_t>(query_in_batch) * top_k + rank;
      top_scores[output_offset] = shared_scores[0];
      top_ids[output_offset] = shared_ids[0];
      query_scores[shared_positions[0]] = worst_score;
    }
    __syncthreads();
  }
}

// 两阶段 Top-K 的第一阶段：每个线程只扫描自己负责的候选，并保留本线程的 Top-K。
// 局部数组不需要有序：绝大多数候选只与当前最差项比较一次；只有候选真正
// 进入 Top-K 时，才扫描 K 个槽位重新找最差项。256 个线程产生 256*K 个候选后，
// 再交给 blockTopKKernel 做最终有序归并。
__global__ void localTopKCandidatesKernel(const float* scores,
                                          float* local_scores,
                                          std::uint64_t* local_ids,
                                          std::uint64_t num_vectors,
                                          std::uint32_t top_k, Metric metric) {
  constexpr std::uint32_t kMaximumLocalK = 10;
  const std::uint32_t query_in_batch = blockIdx.x;
  const std::uint32_t thread_id = threadIdx.x;
  constexpr float kFloatMaximum = 3.402823466e+38F;
  const float worst_score =
      (metric == Metric::kL2) ? kFloatMaximum : -kFloatMaximum;

  float best_scores[kMaximumLocalK];
  std::uint64_t best_ids[kMaximumLocalK];
  for (std::uint32_t rank = 0; rank < top_k; ++rank) {
    best_scores[rank] = worst_score;
    best_ids[rank] = UINT64_MAX;
  }
  std::uint32_t worst_rank = 0;

  const float* query_scores =
      scores + static_cast<std::uint64_t>(query_in_batch) * num_vectors;
  for (std::uint64_t vector_id = thread_id; vector_id < num_vectors;
       vector_id += blockDim.x) {
    const float score = query_scores[vector_id];
    if (isBetter(score, vector_id, best_scores[worst_rank],
                 best_ids[worst_rank], metric)) {
      best_scores[worst_rank] = score;
      best_ids[worst_rank] = vector_id;
      worst_rank = 0;
      for (std::uint32_t rank = 1; rank < top_k; ++rank) {
        if (isBetter(best_scores[worst_rank], best_ids[worst_rank],
                     best_scores[rank], best_ids[rank], metric)) {
          worst_rank = rank;
        }
      }
    }
  }

  const std::uint64_t output_offset =
      (static_cast<std::uint64_t>(query_in_batch) * blockDim.x + thread_id) *
      top_k;
  for (std::uint32_t rank = 0; rank < top_k; ++rank) {
    local_scores[output_offset + rank] = best_scores[rank];
    local_ids[output_offset + rank] = best_ids[rank];
  }
}

// K=1/10 是项目的主要评测档。将 K 变成编译期常量后，编译器可以展开
// 局部槽位扫描，并尽量避免通用 kernel 因动态数组索引产生的 thread stack。
template <std::uint32_t TopK>
__global__ void localTopKCandidatesFixedKernel(const float* scores,
                                               float* local_scores,
                                               std::uint64_t* local_ids,
                                               std::uint64_t num_vectors,
                                               Metric metric) {
  static_assert(TopK >= 1 && TopK <= 10, "fixed Top-K 必须位于 [1, 10]");
  const std::uint32_t query_in_batch = blockIdx.x;
  const std::uint32_t thread_id = threadIdx.x;
  constexpr float kFloatMaximum = 3.402823466e+38F;
  const float worst_score =
      (metric == Metric::kL2) ? kFloatMaximum : -kFloatMaximum;

  float best_scores[TopK];
  std::uint64_t best_ids[TopK];
#pragma unroll
  for (std::uint32_t rank = 0; rank < TopK; ++rank) {
    best_scores[rank] = worst_score;
    best_ids[rank] = UINT64_MAX;
  }
  std::uint32_t worst_rank = 0;

  const float* query_scores =
      scores + static_cast<std::uint64_t>(query_in_batch) * num_vectors;
  for (std::uint64_t vector_id = thread_id; vector_id < num_vectors;
       vector_id += blockDim.x) {
    const float score = query_scores[vector_id];
    if (isBetter(score, vector_id, best_scores[worst_rank],
                 best_ids[worst_rank], metric)) {
      best_scores[worst_rank] = score;
      best_ids[worst_rank] = vector_id;
      worst_rank = 0;
#pragma unroll
      for (std::uint32_t rank = 1; rank < TopK; ++rank) {
        if (isBetter(best_scores[worst_rank], best_ids[worst_rank],
                     best_scores[rank], best_ids[rank], metric)) {
          worst_rank = rank;
        }
      }
    }
  }

  const std::uint64_t output_offset =
      (static_cast<std::uint64_t>(query_in_batch) * blockDim.x + thread_id) *
      TopK;
#pragma unroll
  for (std::uint32_t rank = 0; rank < TopK; ++rank) {
    local_scores[output_offset + rank] = best_scores[rank];
    local_ids[output_offset + rank] = best_ids[rank];
  }
}

// Fused exact scan: each block owns one query/chunk pair and contains eight
// cooperative warps. A warp computes one vector score and maintains a shared
// Top-K heap. The block emits only K candidates, so the full batch*N score
// matrix is never materialized. A small final merge combines all chunks.
template <typename InputType, std::uint32_t MaximumK, Metric MetricValue>
__global__ void exactFusedWarpTopKKernel(
    const InputType* database, const InputType* queries, float* local_scores,
    std::uint64_t* local_ids, std::uint64_t num_vectors, std::uint32_t dim,
    std::uint32_t top_k, bool cache_query) {
  constexpr std::uint32_t kWarpSize = 32;
  constexpr std::uint32_t kWarps = 8;
  constexpr float kFloatMaximum = 3.402823466e+38F;
  __shared__ float warp_scores[kWarps][MaximumK];
  __shared__ std::uint64_t warp_ids[kWarps][MaximumK];
  __shared__ float merge_scores[kWarps];
  __shared__ std::uint64_t merge_ids[kWarps];
  __shared__ std::uint32_t warp_ranks[kWarps];
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto* shared_query = reinterpret_cast<InputType*>(dynamic_shared);

  const std::uint32_t thread_id = threadIdx.x;
  const std::uint32_t lane = thread_id % kWarpSize;
  const std::uint32_t warp = thread_id / kWarpSize;
  const std::uint32_t query_id = blockIdx.y;
  const float worst_score =
      MetricValue == Metric::kL2 ? kFloatMaximum : -kFloatMaximum;
  if (lane == 0) {
    for (std::uint32_t rank = 0; rank < top_k; ++rank) {
      warp_scores[warp][rank] = worst_score;
      warp_ids[warp][rank] = UINT64_MAX;
    }
  }

  const InputType* query_values =
      queries + static_cast<std::uint64_t>(query_id) * dim;
  if (cache_query) {
    for (std::uint32_t d = thread_id; d < dim; d += blockDim.x) {
      shared_query[d] = query_values[d];
    }
  }
  __syncthreads();
  if (cache_query) {
    query_values = shared_query;
  }

  float query_norm = 0.0F;
  if constexpr (MetricValue == Metric::kCosine) {
    for (std::uint32_t d = lane; d < dim; d += kWarpSize) {
      const float query = loadInput(query_values, d);
      query_norm = fmaf(query, query, query_norm);
    }
    for (std::uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      query_norm += __shfl_down_sync(0xffffffffU, query_norm, offset);
    }
  }

  const std::uint64_t first_vector =
      static_cast<std::uint64_t>(blockIdx.x) * kWarps + warp;
  const std::uint64_t vector_stride =
      static_cast<std::uint64_t>(gridDim.x) * kWarps;
  for (std::uint64_t vector_id = first_vector; vector_id < num_vectors;
       vector_id += vector_stride) {
    const std::uint64_t vector_offset = vector_id * dim;
    float dot = 0.0F;
    float vector_norm = 0.0F;
    float squared_l2 = 0.0F;
    for (std::uint32_t d = lane; d < dim; d += kWarpSize) {
      const float query = loadInput(query_values, d);
      const float vector = loadInput(database, vector_offset + d);
      if constexpr (MetricValue == Metric::kL2) {
        const float difference = query - vector;
        squared_l2 = fmaf(difference, difference, squared_l2);
      } else {
        dot = fmaf(query, vector, dot);
        if constexpr (MetricValue == Metric::kCosine) {
          vector_norm = fmaf(vector, vector, vector_norm);
        }
      }
    }
    for (std::uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      if constexpr (MetricValue == Metric::kL2) {
        squared_l2 += __shfl_down_sync(0xffffffffU, squared_l2, offset);
      } else {
        dot += __shfl_down_sync(0xffffffffU, dot, offset);
        if constexpr (MetricValue == Metric::kCosine) {
          vector_norm += __shfl_down_sync(0xffffffffU, vector_norm, offset);
        }
      }
    }
    if (lane == 0) {
      float score = dot;
      if constexpr (MetricValue == Metric::kL2) {
        score = squared_l2;
      } else if constexpr (MetricValue == Metric::kCosine) {
        score = query_norm == 0.0F || vector_norm == 0.0F
                    ? 0.0F
                    : dot * rsqrtf(query_norm * vector_norm);
      }
      if (isBetter(score, vector_id, warp_scores[warp][0], warp_ids[warp][0],
                   MetricValue)) {
        warp_scores[warp][0] = score;
        warp_ids[warp][0] = vector_id;
        std::uint32_t parent = 0;
        while (true) {
          const std::uint32_t left = parent * 2 + 1;
          if (left >= top_k) {
            break;
          }
          const std::uint32_t right = left + 1;
          std::uint32_t worse_child = left;
          if (right < top_k &&
              isBetter(warp_scores[warp][left], warp_ids[warp][left],
                       warp_scores[warp][right], warp_ids[warp][right],
                       MetricValue)) {
            worse_child = right;
          }
          if (!isBetter(warp_scores[warp][parent], warp_ids[warp][parent],
                        warp_scores[warp][worse_child],
                        warp_ids[warp][worse_child], MetricValue)) {
            break;
          }
          const float swap_score = warp_scores[warp][parent];
          const std::uint64_t swap_id = warp_ids[warp][parent];
          warp_scores[warp][parent] = warp_scores[warp][worse_child];
          warp_ids[warp][parent] = warp_ids[warp][worse_child];
          warp_scores[warp][worse_child] = swap_score;
          warp_ids[warp][worse_child] = swap_id;
          parent = worse_child;
        }
      }
    }
  }
  __syncthreads();

  // Heap sort each warp into best-to-worst order.
  if (lane == 0) {
    for (std::uint32_t heap_size = top_k; heap_size > 1; --heap_size) {
      const std::uint32_t last = heap_size - 1;
      const float swap_score = warp_scores[warp][0];
      const std::uint64_t swap_id = warp_ids[warp][0];
      warp_scores[warp][0] = warp_scores[warp][last];
      warp_ids[warp][0] = warp_ids[warp][last];
      warp_scores[warp][last] = swap_score;
      warp_ids[warp][last] = swap_id;
      std::uint32_t parent = 0;
      while (true) {
        const std::uint32_t left = parent * 2 + 1;
        if (left >= last) {
          break;
        }
        const std::uint32_t right = left + 1;
        std::uint32_t worse_child = left;
        if (right < last &&
            isBetter(warp_scores[warp][left], warp_ids[warp][left],
                     warp_scores[warp][right], warp_ids[warp][right],
                     MetricValue)) {
          worse_child = right;
        }
        if (!isBetter(warp_scores[warp][parent], warp_ids[warp][parent],
                      warp_scores[warp][worse_child],
                      warp_ids[warp][worse_child], MetricValue)) {
          break;
        }
        const float parent_score = warp_scores[warp][parent];
        const std::uint64_t parent_id = warp_ids[warp][parent];
        warp_scores[warp][parent] = warp_scores[warp][worse_child];
        warp_ids[warp][parent] = warp_ids[warp][worse_child];
        warp_scores[warp][worse_child] = parent_score;
        warp_ids[warp][worse_child] = parent_id;
        parent = worse_child;
      }
    }
    warp_ranks[warp] = 0;
  }
  __syncthreads();

  for (std::uint32_t output_rank = 0; output_rank < top_k; ++output_rank) {
    if (lane == 0) {
      const std::uint32_t rank = warp_ranks[warp];
      merge_scores[warp] = warp_scores[warp][rank];
      merge_ids[warp] = warp_ids[warp][rank];
    }
    __syncthreads();
    if (thread_id == 0) {
      std::uint32_t best_warp = 0;
      for (std::uint32_t candidate_warp = 1; candidate_warp < kWarps;
           ++candidate_warp) {
        if (isBetter(merge_scores[candidate_warp], merge_ids[candidate_warp],
                     merge_scores[best_warp], merge_ids[best_warp],
                     MetricValue)) {
          best_warp = candidate_warp;
        }
      }
      const std::uint64_t output_offset =
          (static_cast<std::uint64_t>(query_id) * gridDim.x + blockIdx.x) *
              top_k +
          output_rank;
      local_scores[output_offset] = merge_scores[best_warp];
      local_ids[output_offset] = merge_ids[best_warp];
      ++warp_ranks[best_warp];
    }
    __syncthreads();
  }
}

namespace {

void checkCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) {
    checkCuda(cudaMalloc(&data_, bytes), "cudaMalloc");
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* data() { return data_; }

 private:
  void* data_ = nullptr;
};

class CudaEvent {
 public:
  CudaEvent() { checkCuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~CudaEvent() { cudaEventDestroy(event_); }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  void record() { checkCuda(cudaEventRecord(event_), "cudaEventRecord"); }
  void synchronize() {
    checkCuda(cudaEventSynchronize(event_), "cudaEventSynchronize");
  }
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_{};
};

double elapsedMilliseconds(const CudaEvent& start, const CudaEvent& end) {
  float milliseconds = 0.0F;
  checkCuda(cudaEventElapsedTime(&milliseconds, start.get(), end.get()),
            "cudaEventElapsedTime");
  return milliseconds;
}

template <typename InputType, std::uint32_t MaximumK>
void launchExactFusedWarpTopK(dim3 grid, const InputType* database,
                              const InputType* queries, float* local_scores,
                              std::uint64_t* local_ids,
                              std::uint64_t num_vectors, std::uint32_t dim,
                              std::uint32_t top_k, Metric metric) {
  constexpr std::uint32_t kThreads = 256;
  constexpr std::size_t kMaximumCachedQueryBytes = 8192;
  const std::size_t query_bytes =
      static_cast<std::size_t>(dim) * sizeof(InputType);
  const bool cache_query = query_bytes <= kMaximumCachedQueryBytes;
  const std::size_t shared_bytes = cache_query ? query_bytes : 0;
  if (metric == Metric::kL2) {
    exactFusedWarpTopKKernel<InputType, MaximumK, Metric::kL2>
        <<<grid, kThreads, shared_bytes>>>(database, queries, local_scores,
                                           local_ids, num_vectors, dim, top_k,
                                           cache_query);
  } else if (metric == Metric::kInnerProduct) {
    exactFusedWarpTopKKernel<InputType, MaximumK, Metric::kInnerProduct>
        <<<grid, kThreads, shared_bytes>>>(database, queries, local_scores,
                                           local_ids, num_vectors, dim, top_k,
                                           cache_query);
  } else {
    exactFusedWarpTopKKernel<InputType, MaximumK, Metric::kCosine>
        <<<grid, kThreads, shared_bytes>>>(database, queries, local_scores,
                                           local_ids, num_vectors, dim, top_k,
                                           cache_query);
  }
}

}  // namespace

struct GpuExactSearchEngine::Impl {
  explicit Impl(const VectorDatabase& database, const QuerySet& queries,
                const SearchParams& search_params)
      : num_vectors(database.num_vectors),
        dim(database.dim),
        dtype(database.dtype),
        metric(database.metric),
        params(search_params),
        fused_topk(search_params.topk_mode == "fused"),
        use_fp16(database.dtype == DataType::kFloat16),
        input_element_bytes(use_fp16 ? sizeof(std::uint16_t) : sizeof(float)),
        batch_capacity(search_params.batch_size) {
    validateInputs(database, queries, params);
    constexpr std::uint32_t kThreads = 256;
    const std::size_t database_value_count =
        static_cast<std::size_t>(num_vectors) * dim;
    const std::size_t database_bytes =
        database_value_count * input_element_bytes;
    const std::size_t query_bytes =
        static_cast<std::size_t>(batch_capacity) * dim * input_element_bytes;
    const std::size_t score_bytes =
        fused_topk ? 0
                   : static_cast<std::size_t>(batch_capacity) * num_vectors *
                         sizeof(float);
    const std::size_t result_count =
        static_cast<std::size_t>(batch_capacity) * params.top_k;
    const std::uint32_t required_fused_blocks =
        static_cast<std::uint32_t>((num_vectors + 8 - 1) / 8);
    // Keep enough query/chunk blocks in flight to occupy the GPU, without
    // producing 256*K candidates per query for large batches.
    const std::uint32_t occupancy_blocks_per_query = std::max<std::uint32_t>(
        1, (1024U + batch_capacity - 1) / batch_capacity);
    fused_blocks_x =
        std::min(required_fused_blocks, occupancy_blocks_per_query);
    const std::uint32_t local_blocks = fused_topk ? fused_blocks_x : kThreads;
    const std::size_t local_candidate_count =
        uses_local_candidates ? result_count * local_blocks : 0;
    device_bytes =
        database_bytes + query_bytes + score_bytes +
        result_count * (sizeof(float) + sizeof(std::uint64_t)) +
        local_candidate_count * (sizeof(float) + sizeof(std::uint64_t));

    device_database = std::make_unique<DeviceBuffer>(database_bytes);
    device_queries = std::make_unique<DeviceBuffer>(query_bytes);
    if (!fused_topk) {
      device_scores = std::make_unique<DeviceBuffer>(score_bytes);
    }
    device_top_scores =
        std::make_unique<DeviceBuffer>(result_count * sizeof(float));
    device_top_ids =
        std::make_unique<DeviceBuffer>(result_count * sizeof(std::uint64_t));
    if (uses_local_candidates) {
      device_local_scores =
          std::make_unique<DeviceBuffer>(local_candidate_count * sizeof(float));
      device_local_ids = std::make_unique<DeviceBuffer>(local_candidate_count *
                                                        sizeof(std::uint64_t));
    }

    CudaEvent database_copy_start;
    CudaEvent database_copy_end;
    database_copy_start.record();
    const void* database_data =
        use_fp16 ? static_cast<const void*>(database.half_values.data())
                 : static_cast<const void*>(database.values.data());
    checkCuda(cudaMemcpy(device_database->data(), database_data, database_bytes,
                         cudaMemcpyHostToDevice),
              "复制向量库到 GPU");
    database_copy_end.record();
    database_copy_end.synchronize();
    database_h2d_ms =
        elapsedMilliseconds(database_copy_start, database_copy_end);
  }

  std::uint64_t num_vectors;
  std::uint32_t dim;
  DataType dtype;
  Metric metric;
  SearchParams params;
  bool fused_topk;
  bool uses_local_candidates =
      params.topk_mode == "two_stage" || params.topk_mode == "fused";
  bool use_fp16;
  std::size_t input_element_bytes;
  std::uint32_t batch_capacity;
  std::uint32_t fused_blocks_x = 0;
  double database_h2d_ms = 0.0;
  std::size_t device_bytes = 0;
  std::unique_ptr<DeviceBuffer> device_database;
  std::unique_ptr<DeviceBuffer> device_queries;
  std::unique_ptr<DeviceBuffer> device_scores;
  std::unique_ptr<DeviceBuffer> device_top_scores;
  std::unique_ptr<DeviceBuffer> device_top_ids;
  std::unique_ptr<DeviceBuffer> device_local_scores;
  std::unique_ptr<DeviceBuffer> device_local_ids;
};

GpuExactSearchEngine::GpuExactSearchEngine(const VectorDatabase& database,
                                           const QuerySet& initial_queries,
                                           const SearchParams& params)
    : impl_(std::make_unique<Impl>(database, initial_queries, params)) {}

GpuExactSearchEngine::~GpuExactSearchEngine() = default;
GpuExactSearchEngine::GpuExactSearchEngine(GpuExactSearchEngine&&) noexcept =
    default;
GpuExactSearchEngine& GpuExactSearchEngine::operator=(
    GpuExactSearchEngine&&) noexcept = default;

double GpuExactSearchEngine::databaseH2DMilliseconds() const {
  return impl_->database_h2d_ms;
}

std::size_t GpuExactSearchEngine::deviceBytes() const {
  return impl_->device_bytes;
}

SearchResults GpuExactSearchEngine::search(const QuerySet& queries,
                                           SearchStats* stats) {
  if (queries.num_queries == 0 || queries.dim != impl_->dim ||
      queries.dtype != impl_->dtype) {
    throw std::runtime_error("查询集与 GPU 常驻引擎的维度或 dtype 不匹配");
  }
  const std::uint64_t expected_query_values = queries.num_queries * queries.dim;
  if ((impl_->use_fp16 &&
       queries.half_values.size() != expected_query_values) ||
      (!impl_->use_fp16 && queries.values.size() != expected_query_values)) {
    throw std::runtime_error("查询数据长度与元数据不一致");
  }

  SearchStats measured_stats;
  const SearchParams& params = impl_->params;
  const bool use_fp16 = impl_->use_fp16;
  const std::size_t input_element_bytes = impl_->input_element_bytes;
  const std::uint32_t batch_capacity = impl_->batch_capacity;
  const std::uint64_t num_vectors = impl_->num_vectors;
  const std::uint32_t dim = impl_->dim;
  const Metric metric = impl_->metric;
  DeviceBuffer& device_database = *impl_->device_database;
  DeviceBuffer& device_queries = *impl_->device_queries;
  DeviceBuffer* device_scores = impl_->device_scores.get();
  DeviceBuffer& device_top_scores = *impl_->device_top_scores;
  DeviceBuffer& device_top_ids = *impl_->device_top_ids;
  DeviceBuffer* device_local_scores = impl_->device_local_scores.get();
  DeviceBuffer* device_local_ids = impl_->device_local_ids.get();
  constexpr std::uint32_t kThreads = 256;

  SearchResults results(queries.num_queries);
  std::vector<float> host_top_scores(static_cast<std::size_t>(batch_capacity) *
                                     params.top_k);
  std::vector<std::uint64_t> host_top_ids(
      static_cast<std::size_t>(batch_capacity) * params.top_k);
  const std::uint32_t simple_blocks_x =
      static_cast<std::uint32_t>((num_vectors + kThreads - 1) / kThreads);
  constexpr std::uint32_t kWarpsPerBlock = kThreads / 32;
  const std::uint32_t required_warp_blocks = static_cast<std::uint32_t>(
      (num_vectors + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const std::uint32_t warp_blocks_x =
      std::min<std::uint32_t>(required_warp_blocks, 4096);
  const std::uint32_t fused_blocks_x = impl_->fused_blocks_x;

  CudaEvent query_copy_start;
  CudaEvent query_copy_end;
  CudaEvent distance_end;
  CudaEvent topk_end;
  CudaEvent result_copy_end;

  for (std::uint64_t batch_start = 0; batch_start < queries.num_queries;
       batch_start += batch_capacity) {
    const std::uint32_t batch_count =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(
            batch_capacity, queries.num_queries - batch_start));
    const std::size_t current_query_bytes =
        static_cast<std::size_t>(batch_count) * queries.dim *
        input_element_bytes;
    const std::size_t query_value_offset =
        static_cast<std::size_t>(batch_start) * queries.dim;
    const void* query_data =
        use_fp16 ? static_cast<const void*>(queries.half_values.data() +
                                            query_value_offset)
                 : static_cast<const void*>(queries.values.data() +
                                            query_value_offset);
    query_copy_start.record();
    checkCuda(cudaMemcpy(device_queries.data(), query_data, current_query_bytes,
                         cudaMemcpyHostToDevice),
              "复制查询到 GPU");
    query_copy_end.record();

    if (params.topk_mode == "fused") {
      const dim3 grid(fused_blocks_x, batch_count);
      if (use_fp16) {
        const auto* database_values =
            static_cast<const __half*>(device_database.data());
        const auto* query_values =
            static_cast<const __half*>(device_queries.data());
        if (params.top_k <= 10) {
          launchExactFusedWarpTopK<__half, 10>(
              grid, database_values, query_values,
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, dim, params.top_k, metric);
        } else if (params.top_k <= 50) {
          launchExactFusedWarpTopK<__half, 50>(
              grid, database_values, query_values,
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, dim, params.top_k, metric);
        } else {
          launchExactFusedWarpTopK<__half, 100>(
              grid, database_values, query_values,
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, dim, params.top_k, metric);
        }
      } else {
        const auto* database_values =
            static_cast<const float*>(device_database.data());
        const auto* query_values =
            static_cast<const float*>(device_queries.data());
        if (params.top_k <= 10) {
          launchExactFusedWarpTopK<float, 10>(
              grid, database_values, query_values,
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, dim, params.top_k, metric);
        } else if (params.top_k <= 50) {
          launchExactFusedWarpTopK<float, 50>(
              grid, database_values, query_values,
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, dim, params.top_k, metric);
        } else {
          launchExactFusedWarpTopK<float, 100>(
              grid, database_values, query_values,
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, dim, params.top_k, metric);
        }
      }
      checkCuda(cudaGetLastError(), "启动 exactFusedWarpTopKKernel");
      distance_end.record();
      const std::uint64_t candidates_per_query =
          static_cast<std::uint64_t>(fused_blocks_x) * params.top_k;
      blockTopKKernel<<<batch_count, kThreads>>>(
          static_cast<float*>(device_local_scores->data()),
          static_cast<const std::uint64_t*>(device_local_ids->data()),
          static_cast<float*>(device_top_scores.data()),
          static_cast<std::uint64_t*>(device_top_ids.data()),
          candidates_per_query, params.top_k, metric);
      checkCuda(cudaGetLastError(), "启动 fused exact 最终归并");
    } else {
      if (params.distance_mode == "simple") {
        const dim3 grid(simple_blocks_x, batch_count);
        if (use_fp16) {
          exactDistanceKernel<<<grid, kThreads>>>(
              static_cast<const __half*>(device_database.data()),
              static_cast<const __half*>(device_queries.data()),
              static_cast<float*>(device_scores->data()), num_vectors, dim,
              batch_count, metric);
        } else {
          exactDistanceKernel<<<grid, kThreads>>>(
              static_cast<const float*>(device_database.data()),
              static_cast<const float*>(device_queries.data()),
              static_cast<float*>(device_scores->data()), num_vectors, dim,
              batch_count, metric);
        }
        checkCuda(cudaGetLastError(), "启动 exactDistanceKernel");
      } else {
        const dim3 grid(warp_blocks_x, batch_count);
        if (use_fp16) {
          exactDistanceWarpKernel<<<grid, kThreads>>>(
              static_cast<const __half*>(device_database.data()),
              static_cast<const __half*>(device_queries.data()),
              static_cast<float*>(device_scores->data()), num_vectors, dim,
              batch_count, metric);
        } else {
          exactDistanceWarpKernel<<<grid, kThreads>>>(
              static_cast<const float*>(device_database.data()),
              static_cast<const float*>(device_queries.data()),
              static_cast<float*>(device_scores->data()), num_vectors, dim,
              batch_count, metric);
        }
        checkCuda(cudaGetLastError(), "启动 exactDistanceWarpKernel");
      }
      distance_end.record();

      if (params.topk_mode == "simple") {
        simpleTopKKernel<<<batch_count, 1>>>(
            static_cast<const float*>(device_scores->data()),
            static_cast<float*>(device_top_scores.data()),
            static_cast<std::uint64_t*>(device_top_ids.data()), num_vectors,
            params.top_k, metric);
        checkCuda(cudaGetLastError(), "启动 simpleTopKKernel");
      } else if (params.topk_mode == "block") {
        blockTopKKernel<<<batch_count, kThreads>>>(
            static_cast<float*>(device_scores->data()), nullptr,
            static_cast<float*>(device_top_scores.data()),
            static_cast<std::uint64_t*>(device_top_ids.data()), num_vectors,
            params.top_k, metric);
        checkCuda(cudaGetLastError(), "启动 blockTopKKernel");
      } else {
        if (params.top_k == 1) {
          localTopKCandidatesFixedKernel<1><<<batch_count, kThreads>>>(
              static_cast<const float*>(device_scores->data()),
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, metric);
        } else if (params.top_k == 10) {
          localTopKCandidatesFixedKernel<10><<<batch_count, kThreads>>>(
              static_cast<const float*>(device_scores->data()),
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, metric);
        } else {
          localTopKCandidatesKernel<<<batch_count, kThreads>>>(
              static_cast<const float*>(device_scores->data()),
              static_cast<float*>(device_local_scores->data()),
              static_cast<std::uint64_t*>(device_local_ids->data()),
              num_vectors, params.top_k, metric);
        }
        checkCuda(cudaGetLastError(), "启动 localTopKCandidatesKernel");

        const std::uint64_t candidates_per_query =
            static_cast<std::uint64_t>(kThreads) * params.top_k;
        blockTopKKernel<<<batch_count, kThreads>>>(
            static_cast<float*>(device_local_scores->data()),
            static_cast<const std::uint64_t*>(device_local_ids->data()),
            static_cast<float*>(device_top_scores.data()),
            static_cast<std::uint64_t*>(device_top_ids.data()),
            candidates_per_query, params.top_k, metric);
        checkCuda(cudaGetLastError(), "启动两阶段归并 blockTopKKernel");
      }
    }
    topk_end.record();

    const std::size_t result_count =
        static_cast<std::size_t>(batch_count) * params.top_k;
    checkCuda(cudaMemcpy(host_top_scores.data(), device_top_scores.data(),
                         result_count * sizeof(float), cudaMemcpyDeviceToHost),
              "复制 Top-K 分数到 CPU");
    checkCuda(cudaMemcpy(host_top_ids.data(), device_top_ids.data(),
                         result_count * sizeof(std::uint64_t),
                         cudaMemcpyDeviceToHost),
              "复制 Top-K ID 到 CPU");
    result_copy_end.record();
    result_copy_end.synchronize();

    measured_stats.query_h2d_ms +=
        elapsedMilliseconds(query_copy_start, query_copy_end);
    measured_stats.distance_kernel_ms +=
        elapsedMilliseconds(query_copy_end, distance_end);
    measured_stats.topk_kernel_ms +=
        elapsedMilliseconds(distance_end, topk_end);
    measured_stats.result_d2h_ms +=
        elapsedMilliseconds(topk_end, result_copy_end);
    measured_stats.batch_latency_ms.push_back(
        elapsedMilliseconds(query_copy_start, result_copy_end));

    for (std::uint32_t query_in_batch = 0; query_in_batch < batch_count;
         ++query_in_batch) {
      std::vector<Neighbor>& query_results =
          results[batch_start + query_in_batch];
      query_results.resize(params.top_k);
      const std::size_t offset =
          static_cast<std::size_t>(query_in_batch) * params.top_k;
      for (std::uint32_t rank = 0; rank < params.top_k; ++rank) {
        query_results[rank] = {host_top_ids[offset + rank],
                               host_top_scores[offset + rank]};
      }
    }
  }
  if (stats != nullptr) {
    *stats = measured_stats;
  }
  return results;
}

SearchResults gpuExactSearch(const VectorDatabase& database,
                             const QuerySet& queries,
                             const SearchParams& params, SearchStats* stats) {
  GpuExactSearchEngine engine(database, queries, params);
  SearchResults results = engine.search(queries, stats);
  if (stats != nullptr) {
    stats->database_h2d_ms = engine.databaseH2DMilliseconds();
  }
  return results;
}
