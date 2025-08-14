#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 16
#endif
#ifndef ILP_FACTOR
#define ILP_FACTOR 8
#endif
#ifndef ITERATION
#define ITERATION 10000
#endif
#ifndef STRIDE
#define STRIDE 1
#endif

#if !defined(CALC_BW) && !defined(CALC_LATENCY)
#define CALC_BW
#endif

namespace cg = cooperative_groups;

__device__ inline unsigned int get_smid() {
    unsigned int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1)
void kernel(unsigned long long *out,
            int num_ints,
            int rt_destSM,
            int rt_srcSM,
            int *l2buf) {                    // *** NEW: 全局(L2)缓冲区
  extern __shared__ int sdata[];

  cg::cluster_group cluster = cg::this_cluster();
  unsigned int rank = cluster.block_rank();
  int cluster_id = blockIdx.x / CLUSTER_SIZE;

  // --- 计算本 cluster 在 l2buf 中的切片基址 ---
  int * __restrict__ ws = l2buf + cluster_id * num_ints;   // *** NEW: 用全局内存做“共享区”

  // === WARM & record destSM ===
  if (rank == rt_destSM && threadIdx.x == 0) {
    unsigned int my_smid = get_smid();
  #ifdef CALC_BW
    out[3 * cluster_id + 0] = my_smid;
  #else
    out[cluster_id * (blockDim.x + 2) + 0] = my_smid;
  #endif
    // 将数据写入全局内存（驻留 L2）
    ws[0] = my_smid;
    for (int i = 1; i < num_ints; i++) {
      ws[i] = my_smid + i;
    }
    __threadfence();   // *** NEW: 保证写入对其他 SM 可见（推送到 L2）
  }

  int local_sum = 0;
#ifdef CALC_LATENCY
  unsigned long long lat_acc = 0, lat_cnt = 0;
#endif

  // 等待写端完成
  cluster.sync();

  // === READ & record srcSM + timing ===
  if (rank == rt_srcSM) {
#ifdef CALC_BW
    unsigned long long startCycles = clock64();
#endif

    for (int rep = 0; rep < ITERATION; rep++) {
      int idx = threadIdx.x * STRIDE;

      for (; idx + (ILP_FACTOR - 1) * blockDim.x * STRIDE < num_ints;
            idx += blockDim.x * ILP_FACTOR * STRIDE) {
#ifdef CALC_LATENCY
        unsigned long long t0 = clock();
#endif
#pragma unroll
        for (int j = 0; j < ILP_FACTOR; j++) {
          // *** NEW: 强制从 L2 读取，绕过各自 L1
          local_sum += __ldcg(&ws[idx + j * blockDim.x * STRIDE]);
        }
#ifdef CALC_LATENCY
        unsigned long long t1 = clock();
        lat_acc += (t1 - t0);
        lat_cnt++;
#endif
      }
      // tail
      for (; idx < num_ints; idx += blockDim.x * STRIDE) {
#ifdef CALC_LATENCY
        unsigned long long t0 = clock();
#endif
        local_sum += __ldcg(&ws[idx]);       // *** NEW
#ifdef CALC_LATENCY
        unsigned long long t1 = clock();
        lat_acc += (t1 - t0);
        lat_cnt++;
#endif
      }
    }

    __syncthreads();

#ifdef CALC_BW
    unsigned long long totalCycles = clock64() - startCycles;
    unsigned int my_smid = get_smid();
    out[3 * cluster_id + 1] = my_smid;
    out[3 * cluster_id + 2] = totalCycles;
#endif
#ifdef CALC_LATENCY
    unsigned long long avgLat = lat_cnt ? (lat_acc / lat_cnt) : 0;
    int base = cluster_id * (blockDim.x + 2);
    out[base + 1] = get_smid();
    out[base + 2 + threadIdx.x] = avgLat;
#endif
  }

  // 防优化
  sdata[threadIdx.x] = local_sum;

  cluster.sync();
}

int main(int argc, char **argv) {
  int rt_destSM   = 1;
  int rt_srcSM    = 0;
  int numClusters = 1;
  int blockSize   = 1024;

  if (argc > 1) rt_destSM   = atoi(argv[1]);
  if (argc > 2) rt_srcSM    = atoi(argv[2]);
  if (argc > 3) numClusters = atoi(argv[3]);
  if (argc > 4) blockSize   = atoi(argv[4]);

  int dev = 0;
  cudaDeviceProp prop;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);

  int maxSMemOptin;
  cudaDeviceGetAttribute(&maxSMemOptin,
    cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  size_t shared_bytes = maxSMemOptin;
  int num_ints = (int)(shared_bytes / sizeof(int));  // 仍复用同样的数据规模

  int total_blocks = numClusters * CLUSTER_SIZE;
  size_t out_size;
#ifdef CALC_BW
  out_size = numClusters * 3 * sizeof(unsigned long long);
#else
  out_size = numClusters * (blockSize + 2) * sizeof(unsigned long long);
#endif

  unsigned long long *d_out, *h_out;
  h_out = (unsigned long long*)malloc(out_size);
  cudaMalloc(&d_out, out_size);

  // *** NEW: 为 L2“共享区”分配全局内存（每个 cluster 一段）
  int *d_l2buf = nullptr;
  cudaMalloc(&d_l2buf, (size_t)numClusters * num_ints * sizeof(int));   // *** NEW

  cudaFuncSetAttribute(kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
  cudaFuncSetAttribute(kernel,
    cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

  kernel<<< total_blocks, blockSize, shared_bytes >>>(
    d_out, num_ints, rt_destSM, rt_srcSM, d_l2buf);  // *** NEW: 传入 d_l2buf
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost);

  double clkHz = prop.clockRate * 1e3;

#ifdef CALC_BW
  for (int i = 0; i < numClusters; i++) {
    unsigned long long destSM = h_out[3*i + 0];
    unsigned long long srcSM  = h_out[3*i + 1];
    unsigned long long cycles = h_out[3*i + 2];
    unsigned long long bytes  =
      (unsigned long long)num_ints * sizeof(int) * ITERATION;
    double bpc = (double)bytes / (double)cycles;
    double bw  = bpc * clkHz / 1e9;
    printf("Cluster %d destSM %llu srcSM %llu Bandwidth %.4f GB/s\n",
           i, destSM, srcSM, bw);
  }
#else
  for (int i = 0; i < numClusters; i++) {
    double average = 0;
    int base = i * (blockSize + 2);
    unsigned long long destSM = h_out[base + 0];
    unsigned long long srcSM  = h_out[base + 1];
    for (int j = 0; j < blockSize; j++) {
      unsigned long long avgLat = h_out[base + 2 + j];
      average += avgLat;
    }
    average /= blockSize;
    printf("Cluster %d destSM %llu srcSM %llu Avg Latency %.4f clock cycles\n",
           i, destSM, srcSM, average);
  }
#endif

  cudaFree(d_l2buf);   // *** NEW
  cudaFree(d_out);
  free(h_out);
  return 0;
}
