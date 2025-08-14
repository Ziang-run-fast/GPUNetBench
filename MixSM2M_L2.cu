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

// 选择测带宽或延迟（默认带宽）
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
            int *l2buf)    // L2 路径的全局缓冲（每个 cluster 一段）
{
  extern __shared__ int sdata[];    // DSM 路径：写端的共享内存被映射给读端

  cg::cluster_group cluster = cg::this_cluster();
  const unsigned int rank = cluster.block_rank();
  const int cluster_id = blockIdx.x / CLUSTER_SIZE;

  // 将 rank 划分为两半：前半 DSM，后半 L2
  const int HALF = CLUSTER_SIZE / 2;
  const int dsm_dest = rt_destSM % CLUSTER_SIZE;
  const int dsm_src  = rt_srcSM  % CLUSTER_SIZE;
  const int l2_dest  = (rt_destSM + HALF) % CLUSTER_SIZE;
  const int l2_src   = (rt_srcSM  + HALF) % CLUSTER_SIZE;

  // L2 路径：本 cluster 在 l2buf 中的切片
  int * __restrict__ ws_l2 = l2buf + cluster_id * num_ints;

  // ===== 预热/写端：DSM 写共享内存并记录 destSM =====
  if (rank == dsm_dest && threadIdx.x == 0) {
    const unsigned int my_smid = get_smid();
#ifdef CALC_BW
    // [DSM] per-cluster: [0]=destSM
    out[6 * cluster_id + 0] = my_smid;
#else
    // [DSM] per-cluster 段1: [base+0]=destSM
    const int base_dsm = cluster_id * 2 * (blockDim.x + 2);
    out[base_dsm + 0] = my_smid;
#endif
    sdata[0] = my_smid;
    for (int i = 1; i < num_ints; i++) sdata[i] = my_smid + i;
  }

  // ===== 预热/写端：L2 写全局内存并记录 destSM =====
  if (rank == l2_dest && threadIdx.x == 0) {
    const unsigned int my_smid = get_smid();
#ifdef CALC_BW
    // [L2] per-cluster: [3]=destSM
    out[6 * cluster_id + 3] = my_smid;
#else
    // [L2] per-cluster 段2: [base+0]=destSM
    const int base_l2 = cluster_id * 2 * (blockDim.x + 2) + (blockDim.x + 2);
    out[base_l2 + 0] = my_smid;
#endif
    ws_l2[0] = my_smid;
    for (int i = 1; i < num_ints; i++) ws_l2[i] = my_smid + i;
    __threadfence(); // 确保写入在 L2 可见
  }

  // 写端与读端对齐
  cluster.sync();

  int local_sum = 0;

  // ===================== DSM 读路径 =====================
  if (rank == dsm_src) {
    int * __restrict__ ws_dsm = cluster.map_shared_rank(sdata, dsm_dest);

#ifdef CALC_BW
    unsigned long long startCycles = clock64();
#else
    unsigned long long lat_acc = 0, lat_cnt = 0;
#endif

    for (int rep = 0; rep < ITERATION; rep++) {
      int idx = threadIdx.x * STRIDE;

      // 主体 ILP 组
      for (; idx + (ILP_FACTOR - 1) * blockDim.x * STRIDE < num_ints;
             idx += blockDim.x * ILP_FACTOR * STRIDE) {
#ifdef CALC_LATENCY
        unsigned long long t0 = clock();
#endif
#pragma unroll
        for (int j = 0; j < ILP_FACTOR; j++) {
          local_sum += ws_dsm[idx + j * blockDim.x * STRIDE];
        }
#ifdef CALC_LATENCY
        unsigned long long t1 = clock();
        lat_acc += (t1 - t0);
        lat_cnt++;
#endif
      }
      // 尾部
      for (; idx < num_ints; idx += blockDim.x * STRIDE) {
#ifdef CALC_LATENCY
        unsigned long long t0 = clock();
#endif
        local_sum += ws_dsm[idx];
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
    const unsigned int my_smid = get_smid();
    // [DSM] per-cluster: [1]=srcSM, [2]=cycles
    out[6 * cluster_id + 1] = my_smid;
    out[6 * cluster_id + 2] = totalCycles;
#else
    unsigned long long avgLat = (lat_cnt ? (lat_acc / lat_cnt) : 0);
    const int base_dsm = cluster_id * 2 * (blockDim.x + 2);
    out[base_dsm + 1] = get_smid();                // srcSM
    out[base_dsm + 2 + threadIdx.x] = avgLat;      // 每线程平均（单位：cycle）
#endif
  }

  // ===================== L2 读路径 =====================
  if (rank == l2_src) {
#ifdef CALC_BW
    unsigned long long startCycles = clock64();
#else
    unsigned long long lat_acc = 0, lat_cnt = 0;
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
          local_sum += __ldcg(&ws_l2[idx + j * blockDim.x * STRIDE]); // 强制走 L2
        }
#ifdef CALC_LATENCY
        unsigned long long t1 = clock();
        lat_acc += (t1 - t0);
        lat_cnt++;
#endif
      }
      for (; idx < num_ints; idx += blockDim.x * STRIDE) {
#ifdef CALC_LATENCY
        unsigned long long t0 = clock();
#endif
        local_sum += __ldcg(&ws_l2[idx]);
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
    const unsigned int my_smid = get_smid();
    // [L2] per-cluster: [4]=srcSM, [5]=cycles
    out[6 * cluster_id + 4] = my_smid;
    out[6 * cluster_id + 5] = totalCycles;
#else
    unsigned long long avgLat = (lat_cnt ? (lat_acc / lat_cnt) : 0);
    const int base_l2 = cluster_id * 2 * (blockDim.x + 2) + (blockDim.x + 2);
    out[base_l2 + 1] = get_smid();                 // srcSM
    out[base_l2 + 2 + threadIdx.x] = avgLat;       // 每线程平均（单位：cycle）
#endif
  }

  // 防优化（让编译器保留读写）
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

  int maxSMemOptin = 0;
  cudaDeviceGetAttribute(&maxSMemOptin,
      cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

  // 动态共享内存大小（尽量大，供 DSM 写端使用）
  size_t shared_bytes = (size_t)maxSMemOptin;
  int num_ints = (int)(shared_bytes / sizeof(int));

  // 网格大小：numClusters 个 cluster，每个 cluster 有 CLUSTER_SIZE 个 block
  int total_blocks = numClusters * CLUSTER_SIZE;

  // 输出缓冲大小
  size_t out_size = 0;
#ifdef CALC_BW
  // 每个 cluster 6 个值：DSM [dest,src,cycles] + L2 [dest,src,cycles]
  out_size = (size_t)numClusters * 6 * sizeof(unsigned long long);
#else
  // 每个 cluster 两段：(blockSize + 2) × 2
  out_size = (size_t)numClusters * 2 * (blockSize + 2) * sizeof(unsigned long long);
#endif

  // 分配输出和 L2 缓冲
  unsigned long long *d_out = nullptr;
  unsigned long long *h_out = (unsigned long long*)malloc(out_size);
  cudaMalloc(&d_out, out_size);

  int *d_l2buf = nullptr;
  cudaMalloc(&d_l2buf, (size_t)numClusters * num_ints * sizeof(int));

  // Kernel 属性
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

  // 启动
  kernel<<< total_blocks, blockSize, shared_bytes >>>(
    d_out, num_ints, rt_destSM, rt_srcSM, d_l2buf);
  cudaDeviceSynchronize();

  // 回拷并打印
  cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost);

  const double clkHz = prop.clockRate * 1e3; // kHz->Hz

#ifdef CALC_BW
  for (int i = 0; i < numClusters; i++) {
    unsigned long long dsm_dest = h_out[6*i + 0];
    unsigned long long dsm_src  = h_out[6*i + 1];
    unsigned long long dsm_cyc  = h_out[6*i + 2];
    unsigned long long l2_dest  = h_out[6*i + 3];
    unsigned long long l2_src   = h_out[6*i + 4];
    unsigned long long l2_cyc   = h_out[6*i + 5];

    unsigned long long bytes = (unsigned long long)num_ints * sizeof(int) * ITERATION;
    auto toGBs = [&](unsigned long long cyc){
      double bpc = (double)bytes / (double)cyc;
      return bpc * clkHz / 1e9;
    };

    printf("[Cluster %d] DSM  destSM %llu  srcSM %llu  BW %.4f GB/s\n",
           i, dsm_dest, dsm_src, toGBs(dsm_cyc));
    printf("[Cluster %d] L2   destSM %llu  srcSM %llu  BW %.4f GB/s\n",
           i, l2_dest,  l2_src,  toGBs(l2_cyc));
  }
#else
  for (int i = 0; i < numClusters; i++) {
    const int base_dsm = i * 2 * (blockSize + 2);
    const int base_l2  = base_dsm + (blockSize + 2);

    unsigned long long dsm_dest = h_out[base_dsm + 0];
    unsigned long long dsm_src  = h_out[base_dsm + 1];
    unsigned long long l2_dest  = h_out[base_l2 + 0];
    unsigned long long l2_src   = h_out[base_l2 + 1];

    double avg_dsm = 0.0, avg_l2 = 0.0;
    for (int t = 0; t < blockSize; t++) {
      avg_dsm += (double)h_out[base_dsm + 2 + t];
      avg_l2  += (double)h_out[base_l2  + 2 + t];
    }
    avg_dsm /= blockSize;
    avg_l2  /= blockSize;

    printf("[Cluster %d] DSM  destSM %llu  srcSM %llu  AvgLat %.2f cycles\n",
           i, dsm_dest, dsm_src, avg_dsm);
    printf("[Cluster %d] L2   destSM %llu  srcSM %llu  AvgLat %.2f cycles\n",
           i, l2_dest,  l2_src,  avg_l2);
  }
#endif

  cudaFree(d_l2buf);
  cudaFree(d_out);
  free(h_out);
  return 0;
}
