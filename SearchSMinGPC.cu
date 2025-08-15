
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

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
#ifndef MAX_ATTEMPTS
#define MAX_ATTEMPTS 32   // 多次尝试以提高命中概率
#endif

// 默认为“延迟模式”；如需带宽，可 -DCALC_BW（但探测本身看成功与否即可）
#if !defined(CALC_BW) && !defined(CALC_LATENCY)
#define CALC_LATENCY
#endif

__device__ __forceinline__ unsigned get_smid() {
  unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid)); return smid;
}

// 每个“逻辑 cluster”的控制区
struct Ctrl {
  int writer_rank;   // -1 表示尚未选出
  int reader_rank;   // -1 表示尚未选出
  int success;       // 0/1：是否在该 cluster 成功完成一次 DSM 通信
};

__global__ __cluster_dims__(CLUSTER_SIZE,1,1)
void kernel_probe_dsm(
    double               *avg_lat_out,   // 每 cluster 一个输出：cycles/group
    Ctrl                 *ctrl,          // 每 cluster 控制区
    int                   num_ints,
    int                   dstSM,         // 目标写端 SM
    int                   srcSM)         // 目标读端 SM
{
  extern __shared__ int sdata[]; // 写端 block 的共享内存将被映射给读端
  cg::cluster_group cluster = cg::this_cluster();
  const int   cluster_id = blockIdx.x / CLUSTER_SIZE;
  const int   rank       = (int)cluster.block_rank();
  Ctrl *C = &ctrl[cluster_id];

  // 把 writer/reader 的 rank 初始化为 -1（只做一次即可；并发写无所谓）
  if (threadIdx.x == 0) {
    if (C->writer_rank != -1 && C->reader_rank != -1) {
      // 上一轮遗留；这里不强制清理，因为 host 每次 attempt 都会 memset
    }
  }

  // 依据实际运行的 SM 选择角色候选
  const unsigned mySM = get_smid();
  if (mySM == (unsigned)dstSM) {
    // 只有一个 block 能成为“写端”
    atomicCAS(&C->writer_rank, -1, rank);
  }
  if (mySM == (unsigned)srcSM) {
    // 只有一个 block 能成为“读端”
    atomicCAS(&C->reader_rank, -1, rank);
  }

  // 同一个 cluster 内部对齐
  cluster.sync();

  const int w = C->writer_rank;
  const int r = C->reader_rank;

  // 只有当写端和读端都在同一个 cluster 才能进行 DSMEM 通信
  if (w != -1 && r != -1) {
    const bool isWriter = (rank == w);
    const bool isReader = (rank == r);

    // 写端准备数据
    if (isWriter && threadIdx.x == 0) {
      sdata[0] = 1;
      for (int i=1; i<num_ints; ++i) sdata[i] = sdata[i-1] + 1;
    }

    // 写端准备好后，整个 cluster 同步
    cluster.sync();

#ifdef CALC_BW
    unsigned long long start = 0;
#else
    unsigned long long lat_acc = 0ULL, lat_cnt = 0ULL;
#endif

    if (isReader) {
      // 读端把写端 block 的共享内存映射进来
      int * __restrict__ ws = cluster.map_shared_rank(sdata, w);

#ifdef CALC_BW
      start = clock64();
#else
      int sink = 0;
#endif
      // 读循环
      for (int rep=0; rep<ITERATION; ++rep) {
        int idx = threadIdx.x * STRIDE;

        // 主体段：ILP 组
        for (; idx + (ILP_FACTOR - 1)*blockDim.x*STRIDE < num_ints;
               idx += blockDim.x * ILP_FACTOR * STRIDE)
        {
#ifdef CALC_LATENCY
          unsigned long long t0 = clock();
#endif
#pragma unroll
          for (int j=0; j<ILP_FACTOR; ++j) {
#ifdef CALC_LATENCY
            sink += ws[idx + j*blockDim.x*STRIDE];
#else
            (void)ws[idx + j*blockDim.x*STRIDE];
#endif
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
          sink += ws[idx];
          unsigned long long t1 = clock();
          lat_acc += (t1 - t0);
          lat_cnt++;
#else
          (void)ws[idx];
#endif
        }
      }
#ifdef CALC_LATENCY
      (void)sink;
#endif
    }

    // 再同步一次，确保计时结果就绪
    cluster.sync();

    if (isReader && threadIdx.x == 0) {
#ifdef CALC_BW
      unsigned long long cyc = clock64() - start;
      // 如需带宽可在 host 侧换算；这里我们仍写入“平均每组周期”风格
      avg_lat_out[cluster_id] = (double)cyc; // 带宽模式下留给 host 处理
#else
      const double avg = (lat_cnt ? (double)lat_acc / (double)lat_cnt : 0.0);
      avg_lat_out[cluster_id] = avg;  // cycles per ILP-group
#endif
      C->success = 1; // 标记该 cluster 成功
    }
  }
  // 该 cluster 若两端不齐（w或r为-1），所有 block 都直接返回，本次 attempt 失败
}

int main(int argc, char **argv) {
  int dstSM = 1, srcSM = 0, numClusters = 1, blockSize = 1024;
  if (argc > 1) dstSM      = atoi(argv[1]);
  if (argc > 2) srcSM      = atoi(argv[2]);
  if (argc > 3) numClusters= atoi(argv[3]);
  if (argc > 4) blockSize  = atoi(argv[4]);

  // 设备信息
  int dev = 0; cudaSetDevice(dev);
  cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);
  const double clkHz = prop.clockRate * 1e3;

  // 动态共享内存大小沿用“共享内存上限”作为数据规模
  int maxSMemOptin=0;
  cudaDeviceGetAttribute(&maxSMemOptin,
     cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  const size_t shared_bytes = (size_t)maxSMemOptin;
  const int num_ints = (int)(shared_bytes / sizeof(int));

  // 为 cluster kernel 设置属性
  cudaFuncSetAttribute(kernel_probe_dsm,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(kernel_probe_dsm,
    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
  cudaFuncSetAttribute(kernel_probe_dsm,
    cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

  // 设备端缓冲
  const int total_blocks = numClusters * CLUSTER_SIZE;
  double *d_avg = nullptr;  cudaMalloc(&d_avg, numClusters * sizeof(double));
  Ctrl   *d_ctl = nullptr;  cudaMalloc(&d_ctl, numClusters * sizeof(Ctrl));

  // 主机端缓冲
  double *h_avg = (double*)malloc(numClusters * sizeof(double));
  Ctrl   *h_ctl = (Ctrl*)  malloc(numClusters * sizeof(Ctrl));

  bool ok = false;
  int  ok_cluster = -1;

  // 多次尝试：随机性来自调度，增大 numClusters 和 MAX_ATTEMPTS 可提高命中
  for (int attempt = 1; attempt <= MAX_ATTEMPTS && !ok; ++attempt) {
    // 清空控制区：writer_rank/reader_rank 设为 -1，success=0
    cudaMemset(d_ctl, 0xFF, numClusters * sizeof(Ctrl)); // -1
    // 把 success 字段清零（位于结构体中，重新覆盖）
    // 简单方案：再发一个小核清 success=0；这里用 cudaMemset2D-like 不方便，改为小核
    // 但为简洁，host 再 copy 回来时以 success!=1 判定即可。

    // 启动 kernel
    kernel_probe_dsm<<< total_blocks, blockSize, shared_bytes >>>(
      d_avg, d_ctl, num_ints, dstSM, srcSM);
    cudaDeviceSynchronize();

    // 拿回结果，检查是否有 cluster 成功
    cudaMemcpy(h_ctl, d_ctl, numClusters * sizeof(Ctrl), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_avg, d_avg, numClusters * sizeof(double), cudaMemcpyDeviceToHost);

    for (int c=0; c<numClusters; ++c) {
      if (h_ctl[c].success == 1) { ok = true; ok_cluster = c; break; }
    }
  }

#ifdef CALC_BW
  if (ok) {
    // 带宽口径：我们在 device 端记录的是总 cycles，换算 GB/s
    // bytes = num_ints*sizeof(int) * ITERATION * （近似每线程一次？这里带宽口径仅供参考）
    // 建议在 LATENCY 模式下用于判定；若坚持带宽，可按你的统计口径调整 bytes 定义。
    const unsigned long long cycles = (unsigned long long)h_avg[ok_cluster];
    const unsigned long long bytes  = (unsigned long long)num_ints * sizeof(int) * ITERATION;
    const double bpc = (double)bytes / (double)cycles;
    const double bw  = bpc * clkHz / 1e9;
    printf("Cluster %d destSM%d srcSM%d [DSMEM] Bandwidth %.4f GB/s  -> SAME_GPC\n",
           ok_cluster, dstSM, srcSM, bw);
  } else {
    printf("destSM%d srcSM%d [DSMEM] pairing failed after %d attempts -> LIKELY DIFFERENT_GPC\n",
           dstSM, srcSM, MAX_ATTEMPTS);
  }
#else
  if (ok) {
    // 延迟口径：cycles/group
    const double avg = h_avg[ok_cluster];
    printf("Cluster %d destSM%d srcSM%d [DSMEM] Avg %.2f cycles/group  -> SAME_GPC\n",
           ok_cluster, dstSM, srcSM, avg);
  } else {
    printf("destSM%d srcSM%d [DSMEM] pairing failed after %d attempts -> LIKELY DIFFERENT_GPC\n",
           dstSM, srcSM, MAX_ATTEMPTS);
  }
#endif

  cudaFree(d_avg); cudaFree(d_ctl); free(h_avg); free(h_ctl);
  return 0;
}
