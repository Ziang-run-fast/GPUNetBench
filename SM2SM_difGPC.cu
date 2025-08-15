// SM2SM_interGPC_L2_only.cu
#include <stdio.h>
#include <stdlib.h>
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

// 默认带宽；若要测延迟：-DCALC_LATENCY
#if !defined(CALC_BW) && !defined(CALC_LATENCY)
#define CALC_BW
#endif

__device__ __forceinline__ unsigned get_smid() {
  unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid)); return smid;
}

__device__ __forceinline__ int atomic_read(const int* addr) {
  return atomicAdd((int*)addr, 0);
}

// 每个“逻辑 cluster”有一份：ready 标志、角色锁、输出区
struct ClusterCtrl {
  int writer_claimed;
  int reader_claimed;
  int ready_flag;
};

__global__ void kernel_sm2sm_l2(
    unsigned long long *out,   // (numClusters * 3) 或 (numClusters * (blockSize+2))
    int *gbuf,                 // (numClusters * num_ints)
    int num_ints,
    ClusterCtrl *ctrl,         // (numClusters)
    int rt_destSM, int rt_srcSM)
{
  const int cluster_id = blockIdx.x / CLUSTER_SIZE;
  ClusterCtrl *C = &ctrl[cluster_id];
  int *buf = gbuf + cluster_id * num_ints;

  // 只让每个逻辑 cluster 有“唯一写端/唯一读端”CTA
  const unsigned mySM = get_smid();

  // —— 写端：必须落在 rt_destSM，且抢到 writer_claimed
  bool i_am_writer = (mySM == (unsigned)rt_destSM) &&
                     (atomicCAS(&C->writer_claimed, 0, 1) == 0);

  if (i_am_writer) {
    // 只写一次数据集，避免测量期间的写流量干扰
    if (threadIdx.x == 0) {
      for (int i = 0; i < num_ints; ++i) buf[i] = i + 1;
      __threadfence();               // 推到 L2/对其他 SM 可见
      atomicExch(&C->ready_flag, 1); // 发布“数据就绪”
    }
    return; // 写端退出，避免占用带宽
  }

  // —— 读端：必须落在 rt_srcSM，且抢到 reader_claimed
  bool i_am_reader = (mySM == (unsigned)rt_srcSM) &&
                     (atomicCAS(&C->reader_claimed, 0, 1) == 0);

  if (i_am_reader) {
    // 等待数据就绪
    if (threadIdx.x == 0) {
      while (atomic_read(&C->ready_flag) == 0) { /* spin */ }
      __threadfence(); // 看到 ready 后再读，保序
    }
    __syncthreads();

#ifdef CALC_BW
    unsigned long long start = clock64();
#else
    unsigned long long lat_acc = 0ULL, lat_cnt = 0ULL;
#endif

    for (int rep = 0; rep < ITERATION; ++rep) {
      int idx = threadIdx.x * STRIDE;

      // 主体：ILP_FACTOR 组
      for (; idx + (ILP_FACTOR - 1) * blockDim.x * STRIDE < num_ints;
             idx += blockDim.x * ILP_FACTOR * STRIDE) {
#ifdef CALC_LATENCY
        unsigned long long t0 = clock();
#endif
#pragma unroll
        for (int j = 0; j < ILP_FACTOR; ++j) {
          // 强制从 L2 读取
          (void)__ldcg(&buf[idx + j * blockDim.x * STRIDE]);
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
        (void)__ldcg(&buf[idx]);
#ifdef CALC_LATENCY
        unsigned long long t1 = clock();
        lat_acc += (t1 - t0);
        lat_cnt++;
#endif
      }
    }
    __syncthreads();

#ifdef CALC_BW
    unsigned long long cyc = clock64() - start;
    // out: 每 cluster 3 个值：[destSM, srcSM, cycles]
    out[3 * cluster_id + 0] = rt_destSM;
    out[3 * cluster_id + 1] = rt_srcSM;
    out[3 * cluster_id + 2] = cyc;
#else
    unsigned long long avg = lat_cnt ? (lat_acc / lat_cnt) : 0ULL;
    // out: 每 cluster (blockDim.x + 2)：
    // [0]=destSM, [1]=srcSM, [2..] 每线程平均（当前只写 thread0）
    const size_t base = (size_t)cluster_id * (blockDim.x + 2);
    out[base + 0] = rt_destSM;
    out[base + 1] = rt_srcSM;
    out[base + 2 + 0] = avg; // 只写 thread 0；主机端再做平均或直接读取
#endif
    return;
  }

  // 其他 CTA：不参与，直接返回
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
  cudaSetDevice(dev);

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, dev);
  const double clkHz = prop.clockRate * 1e3;

  // 用共享内存上限作为数据规模（与原逻辑一致的“规模基准”）
  int maxSMemOptin = 0;
  cudaDeviceGetAttribute(&maxSMemOptin,
      cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  const size_t shared_bytes = (size_t)maxSMemOptin;
  const int num_ints = (int)(shared_bytes / sizeof(int));

  // 为每个 cluster 分配一段全局缓冲
  int *d_gbuf = nullptr;
  cudaMalloc(&d_gbuf, (size_t)numClusters * num_ints * sizeof(int));

  // 控制块
  ClusterCtrl *d_ctrl = nullptr;
  cudaMalloc(&d_ctrl, (size_t)numClusters * sizeof(ClusterCtrl));
  cudaMemset(d_ctrl, 0, (size_t)numClusters * sizeof(ClusterCtrl));

  // 输出区
  unsigned long long *d_out = nullptr, *h_out = nullptr;
#ifdef CALC_BW
  cudaMalloc(&d_out, (size_t)numClusters * 3 * sizeof(unsigned long long));
  h_out = (unsigned long long*)malloc((size_t)numClusters * 3 * sizeof(unsigned long long));
#else
  cudaMalloc(&d_out, (size_t)numClusters * (blockSize + 2) * sizeof(unsigned long long));
  h_out = (unsigned long long*)malloc((size_t)numClusters * (blockSize + 2) * sizeof(unsigned long long));
#endif

  // 网格：保持“逻辑 cluster”划分，与原接口一致
  const int total_blocks = numClusters * CLUSTER_SIZE;

  // 启动（建议加 -Xptxas -dlcm=cg 或完全依赖 __ldcg）
  kernel_sm2sm_l2<<<total_blocks, blockSize>>>(
      d_out, d_gbuf, num_ints, d_ctrl, rt_destSM, rt_srcSM);
  cudaDeviceSynchronize();

#ifdef CALC_BW
  cudaMemcpy(h_out, d_out, (size_t)numClusters * 3 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  for (int i = 0; i < numClusters; ++i) {
    const unsigned long long destSM = h_out[3*i + 0];
    const unsigned long long srcSM  = h_out[3*i + 1];
    const unsigned long long cycles = h_out[3*i + 2];
    const unsigned long long bytes  = (unsigned long long)num_ints * sizeof(int) * ITERATION;
    const double bpc = (double)bytes / (double)cycles;
    const double bw  = bpc * clkHz / 1e9;
    printf("Running one cluster...\n"); // 可按需删除
    printf("Cluster %d destSM%llu srcSM%llu Bandwidth %.4f GB/s\n",
           i, destSM, srcSM, bw);
  }
#else
  cudaMemcpy(h_out, d_out, (size_t)numClusters * (blockSize + 2) * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  for (int i = 0; i < numClusters; ++i) {
    const size_t base = (size_t)i * (blockSize + 2);
    const unsigned long long destSM = h_out[base + 0];
    const unsigned long long srcSM  = h_out[base + 1];
    // 这里只写了 thread0 的平均值；为了与原打印口径一致，我们就用它
    const double avg_cycles = (double)h_out[base + 2 + 0];
    printf("Running one cluster...\n"); // 可按需删除
    printf("Cluster %d destSM%llu srcSM%llu Avg Latency %.4f clock cycles\n",
           i, destSM, srcSM, avg_cycles);
  }
#endif

  cudaFree(d_out);
  cudaFree(d_gbuf);
  cudaFree(d_ctrl);
  free(h_out);
  return 0;
}
