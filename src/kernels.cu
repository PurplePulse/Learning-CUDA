#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

// 设备端把 half 转 float 计算，避免累加精度损失
template <typename T>
__device__ __forceinline__ float toFloat(T v);

template <>
__device__ __forceinline__ float toFloat<float>(float v) {
  return v;
}

template <>
__device__ __forceinline__ float toFloat<half>(half v) {
  return __half2float(v);
}

// 计算完成后把 float 结果转回原类型
template <typename T>
__device__ __forceinline__ T fromFloat(float v);

template <>
__device__ __forceinline__ float fromFloat<float>(float v) {
  return v;
}

template <>
__device__ __forceinline__ half fromFloat<half>(float v) {
  return __float2half(v);
}

// RMSNorm：一行一个 block，先对行内元素求平方和，再归一化
template <typename T>
__global__ void rmsNormKernel(const T* input, const T* weight, T* output,
                              size_t hidden_dim, float eps) {
  constexpr int kBlockSize = 256;
  __shared__ float partial_sums[kBlockSize];
  __shared__ float inv_rms;

  const size_t row = blockIdx.x;
  const size_t tid = threadIdx.x;
  const size_t offset = row * hidden_dim;

  // 每个线程累加自己负责的那部分 x^2
  float sum = 0.0f;
  for (size_t col = tid; col < hidden_dim; col += blockDim.x) {
    const float x = toFloat<T>(input[offset + col]);
    sum += x * x;
  }

  // 树形归约：相邻元素两两相加，8 轮后汇总到线程 0
  partial_sums[tid] = sum;
  __syncthreads();

  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial_sums[tid] += partial_sums[tid + stride];
    }
    __syncthreads();
  }

  // 线程 0 算出归一化系数，广播给所有线程
  if (tid == 0) {
    inv_rms = rsqrtf(partial_sums[0] / static_cast<float>(hidden_dim) + eps);
  }
  __syncthreads();

  // 用同一系数写回本行的每个元素
  for (size_t c = tid; c < hidden_dim; c += blockDim.x) {
    const float x = toFloat<T>(input[offset + c]);
    const float w = toFloat<T>(weight[c]);
    output[offset + c] = fromFloat<T>(x * inv_rms * w);
  }
}

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
             std::vector<T>& h_output, size_t rows, size_t hidden_dim,
             float eps) {
  if (rows == 0 || hidden_dim == 0) {
    return;
  }

  const size_t n = rows * hidden_dim;

  T* d_input;
  T* d_weight;
  T* d_output;
  cudaMalloc(&d_input, n * sizeof(T));
  cudaMalloc(&d_weight, hidden_dim * sizeof(T));
  cudaMalloc(&d_output, n * sizeof(T));

  cudaMemcpy(d_input, h_input.data(), n * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, h_weight.data(), hidden_dim * sizeof(T),
             cudaMemcpyHostToDevice);

  constexpr int kBlockSize = 256;
  rmsNormKernel<<<static_cast<unsigned int>(rows), kBlockSize>>>(
      d_input, d_weight, d_output, hidden_dim, eps);
  cudaGetLastError();
  cudaDeviceSynchronize();

  cudaMemcpy(h_output.data(), d_output, n * sizeof(T),
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_output);
}

// FlashAttention：按 Br×Bc 分块；先扫 max，再 softmax 加权累加 V。
// 与参考实现对齐累加顺序，避免长序列 float 误差压线失败。
constexpr int kFlashBr = 8;
constexpr int kFlashBc = 16;

__device__ __forceinline__ float negInf() {
  return __int_as_float(0xff800000);
}

__device__ __forceinline__ float dotQK(const float* q, const float* k,
                                      int head_dim) {
  float score = 0.0f;
  int d = 0;
  for (; d + 3 < head_dim; d += 4) {
    score += q[d] * k[d];
    score += q[d + 1] * k[d + 1];
    score += q[d + 2] * k[d + 2];
    score += q[d + 3] * k[d + 3];
  }
  for (; d < head_dim; ++d) {
    score += q[d] * k[d];
  }
  return score;
}

template <typename T>
__global__ void flashAttentionKernel(const T* Q, const T* K, const T* V, T* O,
                                     int target_seq_len, int src_seq_len,
                                     int query_heads, int kv_heads,
                                     int head_dim, bool is_causal) {
  extern __shared__ float smem[];
  float* sQ = smem;
  float* sK = sQ + kFlashBr * head_dim;
  float* sV = sK + kFlashBc * head_dim;
  float* sO = sV + kFlashBc * head_dim;
  float* sS = sO + kFlashBr * head_dim;
  float* sM = sS + kFlashBr * kFlashBc;
  float* sL = sM + kFlashBr;

  const int q_tile = blockIdx.x;
  const int hq = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;

  const int q_start = q_tile * kFlashBr;
  if (q_start >= target_seq_len) {
    return;
  }
  const int q_rows = min(kFlashBr, target_seq_len - q_start);
  const int q_last = q_start + q_rows - 1;

  // GQA：多个 query head 共用一组 K/V
  const int hkv = hq / (query_heads / kv_heads);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  // 加载本 tile 的 Q，并清零输出累加
  for (int idx = tid; idx < q_rows * head_dim; idx += blockDim.x) {
    const int r = idx / head_dim;
    const int d = idx % head_dim;
    const int t = q_start + r;
    const size_t q_offset =
        (static_cast<size_t>(b) * target_seq_len + t) * query_heads + hq;
    sQ[r * head_dim + d] = toFloat<T>(Q[q_offset * head_dim + d]);
    sO[r * head_dim + d] = 0.0f;
  }
  if (tid < q_rows) {
    sM[tid] = negInf();
    sL[tid] = 0.0f;
  }
  __syncthreads();

  // Pass 1：扫 max(score)
  for (int k0 = 0; k0 < src_seq_len; k0 += kFlashBc) {
    if (is_causal && k0 > q_last) {
      break;
    }
    const int tile_n = min(kFlashBc, src_seq_len - k0);

    for (int idx = tid; idx < tile_n * head_dim; idx += blockDim.x) {
      const int s_local = idx / head_dim;
      const int d = idx % head_dim;
      const int s = k0 + s_local;
      const size_t kv_offset =
          (static_cast<size_t>(b) * src_seq_len + s) * kv_heads + hkv;
      sK[s_local * head_dim + d] = toFloat<T>(K[kv_offset * head_dim + d]);
    }
    __syncthreads();

    for (int idx = tid; idx < q_rows * tile_n; idx += blockDim.x) {
      const int r = idx / tile_n;
      const int s_local = idx % tile_n;
      const int t = q_start + r;
      const int s = k0 + s_local;
      float score;
      if (is_causal && s > t) {
        score = negInf();
      } else {
        score = dotQK(sQ + r * head_dim, sK + s_local * head_dim, head_dim) *
                scale;
      }
      sS[r * kFlashBc + s_local] = score;
    }
    __syncthreads();

    if (tid < q_rows) {
      float m = sM[tid];
      for (int i = 0; i < tile_n; ++i) {
        m = fmaxf(m, sS[tid * kFlashBc + i]);
      }
      sM[tid] = m;
    }
    __syncthreads();
  }

  // Pass 2：softmax + 累加 V
  for (int k0 = 0; k0 < src_seq_len; k0 += kFlashBc) {
    if (is_causal && k0 > q_last) {
      break;
    }
    const int tile_n = min(kFlashBc, src_seq_len - k0);

    for (int idx = tid; idx < tile_n * head_dim; idx += blockDim.x) {
      const int s_local = idx / head_dim;
      const int d = idx % head_dim;
      const int s = k0 + s_local;
      const size_t kv_offset =
          (static_cast<size_t>(b) * src_seq_len + s) * kv_heads + hkv;
      sK[s_local * head_dim + d] = toFloat<T>(K[kv_offset * head_dim + d]);
      sV[s_local * head_dim + d] = toFloat<T>(V[kv_offset * head_dim + d]);
    }
    __syncthreads();

    for (int idx = tid; idx < q_rows * tile_n; idx += blockDim.x) {
      const int r = idx / tile_n;
      const int s_local = idx % tile_n;
      const int t = q_start + r;
      const int s = k0 + s_local;
      if (!isfinite(sM[r])) {
        sS[r * kFlashBc + s_local] = 0.0f;
      } else if (is_causal && s > t) {
        sS[r * kFlashBc + s_local] = 0.0f;
      } else {
        const float score =
            dotQK(sQ + r * head_dim, sK + s_local * head_dim, head_dim) *
            scale;
        sS[r * kFlashBc + s_local] = expf(score - sM[r]);
      }
    }
    __syncthreads();

    if (tid < q_rows && isfinite(sM[tid])) {
      float l = sL[tid];
      for (int i = 0; i < tile_n; ++i) {
        l += sS[tid * kFlashBc + i];
      }
      sL[tid] = l;
    }

    for (int idx = tid; idx < q_rows * head_dim; idx += blockDim.x) {
      const int r = idx / head_dim;
      const int d = idx % head_dim;
      if (!isfinite(sM[r])) {
        continue;
      }
      float out = sO[r * head_dim + d];
      for (int i = 0; i < tile_n; ++i) {
        out += sS[r * kFlashBc + i] * sV[i * head_dim + d];
      }
      sO[r * head_dim + d] = out;
    }
    __syncthreads();
  }

  for (int idx = tid; idx < q_rows * head_dim; idx += blockDim.x) {
    const int r = idx / head_dim;
    const int d = idx % head_dim;
    const int t = q_start + r;
    const size_t q_offset =
        (static_cast<size_t>(b) * target_seq_len + t) * query_heads + hq;
    if (!isfinite(sM[r]) || !(sL[r] > 0.0f)) {
      O[q_offset * head_dim + d] = fromFloat<T>(0.0f);
    } else {
      O[q_offset * head_dim + d] =
          fromFloat<T>(sO[r * head_dim + d] / sL[r]);
    }
  }
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  if (batch_size == 0 || target_seq_len == 0 || src_seq_len == 0 ||
      query_heads == 0 || head_dim == 0) {
    return;
  }

  const size_t q_elems = static_cast<size_t>(batch_size) * target_seq_len *
                         query_heads * head_dim;
  const size_t kv_elems = static_cast<size_t>(batch_size) * src_seq_len *
                          kv_heads * head_dim;

  T* d_q;
  T* d_k;
  T* d_v;
  T* d_o;
  RUNTIME_CHECK(cudaMalloc(&d_q, q_elems * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, kv_elems * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, kv_elems * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, q_elems * sizeof(T)));

  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(T),
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_elems * sizeof(T),
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_elems * sizeof(T),
                           cudaMemcpyHostToDevice));

  const size_t smem_bytes =
      static_cast<size_t>(kFlashBr * head_dim + kFlashBc * head_dim +
                          kFlashBc * head_dim + kFlashBr * head_dim +
                          kFlashBr * kFlashBc + kFlashBr + kFlashBr) *
      sizeof(float);

  dim3 block(128);
  dim3 grid((target_seq_len + kFlashBr - 1) / kFlashBr, query_heads,
            batch_size);
  flashAttentionKernel<T><<<grid, block, smem_bytes>>>(
      d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads,
      head_dim, is_causal);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, q_elems * sizeof(T),
                           cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
