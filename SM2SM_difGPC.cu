// sm2sm_intergpc_l2.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef ILP_FACTOR
#define ILP_FACTOR 8
#endif
#ifndef ITERATION
#define ITERATION 10000
#endif
#ifndef STRIDE
#define STRIDE 1
#endif

// 选择测带宽或延迟（默认延迟更符合“RTT”）
#if !defined(CALC_BW) && !defined(CALC_LATENCY)
#define CALC_LATENCY
#endif

// 取 SMID
__device__ __forceinline__ unsigned get_smid() {
  unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid)); return smid;
}

// 简单的设备原子读
__device__ __forceinline__ int atomic_read(const int* addr) {
  return atomicAdd((int*)addr, 0);
}

// 核函数：仅在目标 SM 上执行相应角色；其他 SM 空转退出
__global__ void kernel_intergpc_l2(unsigned long long *out,
                                   int *gbuf, int num_ints,
                                   int *ready_flag,
                                   int destSM, int srcSM)
{
  const unsigned mySM = get_smid();
  int local_sum = 0;

  // 只有一个 warp 必要就足够测量；但保持任意 blockSize 也能跑
  // ——可在主机侧用 blockSize 控制吞吐/并发

  // 1) 写端：落在 destSM 的 CTA 负责把数据写到全局内存，然后发布可见
  if (mySM == (unsigned)destSM) {
    if (threadIdx.x == 0) {
      // 填充一个可复用的数据集（只写一次，避免测量期间的写干扰）
      for (int i = 0; i < num_ints; ++i) gbuf[i] = i + 1;
      __threadfence();                 // 推送到 L2/保证其他 SM 可见
      atomicExch(ready_flag, 1);       // 置位“就绪”
    }
    return; // 写端到此结束，避免给 NoC 增压
  }

  // 2) 读端：落在 srcSM 的 CTA 轮询等待写端完成，然后进行测量
  if (mySM == (unsigned)srcSM) {
    // 等待数据就绪
    if (threadIdx.x == 0) {
      while (atomic_read(ready_flag) == 0) { /* 自旋等待 */ }
      __threadfence(); // 保守起见：保证看到 ready_flag 之后的访问有序
    }
    __syncthreads();

#ifdef CALC_BW
    unsigned long long start = clock64();
#else
    unsigned long long lat_acc = 0, lat_cnt = 0;
#endif

    // 主测量循环：强制从 L2 读取（绕过 L1）
    for (int rep = 0; rep < ITERATION; ++rep) {
      int idx = threadIdx.x * STRIDE;

      // ILP 组
      for (; idx + (ILP_FACTOR - 1) * blockDim.x * STRIDE < num_ints;
             idx += blockDim.x * ILP_FACTOR * STRIDE) {
#ifdef CALC_LATENCY
        unsigned long long t0 = clock();
#endif
#pragma unroll
        for (int j = 0; j < ILP_FACTOR; ++j) {
          local_sum += __ldcg(&gbuf[idx + j * blockDim.x * STRIDE]);
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
        local_sum += __ldcg(&gbuf[idx]);
#ifdef CALC_LATENCY
        unsigned long long t1 = clock();
        lat_acc += (t1 - t0);
        lat_cnt++;
#endif
      }
    }
    __syncthreads();

#ifdef CALC_BW
    unsigned long long cycles = clock64() - start;
    if (threadIdx.x == 0) {
      // out: [0]=destSM, [1]=srcSM, [2]=cycles
      out[0] = destSM; out[1] = srcSM; out[2] = cycles;
    }
#else
    // 线程内平均（单位：cycles/ILP组）
    unsigned long long avg = lat_cnt ? (lat_acc / lat_cnt) : 0ull;
    if (threadIdx.x == 0) {
      // out: [0]=destSM, [1]=srcSM, [2]=avg_cycles_per_group
      out[0] = destSM; out[1] = srcSM; out[2] = avg;
    }
#endif
  }

  // 防优化
  if (threadIdx.x == 0) ((volatile int*)gbuf)[0] = local_sum;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    printf("Usage: %s <destSM> <srcSM> <blockSize> [numIntsKB]\n", argv[0]);
    printf("Example: %s 30 5 256 64\n", argv[0]);
    return 0;
  }
  const int destSM   = atoi(argv[1]);  // 选择位于 GPC-A 的 SM
  const int srcSM    = atoi(argv[2]);  // 选择位于 GPC-B 的 SM（不同 GPC）
  const int blockSize= atoi(argv[3]);  // 线程数（影响并发/合并度）
  const int numIntsKB= (argc > 4 ? atoi(argv[4]) : 64); // 每次数据集大小(KB)

  int dev=0; cudaSetDevice(dev);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  const double clkHz = prop.clockRate * 1e3;

  // 准备缓冲
  const int num_ints = (numIntsKB * 1024) / sizeof(int);
  int *d_gbuf=nullptr, *d_flag=nullptr;
  cudaMalloc(&d_gbuf, num_ints * sizeof(int));
  cudaMalloc(&d_flag, sizeof(int));
  cudaMemset(d_flag, 0, sizeof(int));

  // 输出区：3 个 unsigned long long
  unsigned long long *d_out=nullptr, h_out[3]={0};
  cudaMalloc(&d_out, 3 * sizeof(unsigned long long));

  // 启动方式：发很多 CTA，但只有落在目标 SM 的 CTA 真正干活
  // 为提高“命中”概率，grid 设为 >= SM 数量
  int numSM = prop.multiProcessorCount;
  dim3 grid(max(2*numSM, 64));   // 适当大一点提高分布覆盖
  dim3 block(blockSize);

  // 重要：将 L1 的默认缓存策略改为 .cg，可在编译时加：-Xptxas -dlcm=cg（或全靠 __ldcg）
  kernel_intergpc_l2<<<grid, block>>>(d_out, d_gbuf, num_ints, d_flag, destSM, srcSM);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, 3*sizeof(unsigned long long), cudaMemcpyDeviceToHost);

#ifdef CALC_BW
  const unsigned long long cycles = h_out[2];
  const unsigned long long bytes  = (unsigned long long)num_ints * sizeof(int) * ITERATION;
  const double bpc = (double)bytes / (double)cycles;
  const double GBs = bpc * clkHz / 1e9;
  printf("Inter-GPC L2  destSM %llu  srcSM %llu  Bandwidth %.4f GB/s\n",
         h_out[0], h_out[1], GBs);
#else
  const double avgGrp = (double)h_out[2];           // 每“ILP组”的平均周期
  const double avgPerLoad = avgGrp / (double)ILP_FACTOR; // 近似每次 load 周期（主体占主导）
  printf("Inter-GPC L2  destSM %llu  srcSM %llu  Avg %.2f cycles/group  (≈ %.2f cycles/load)\n",
         h_out[0], h_out[1], avgGrp, avgPerLoad);
#endif

  cudaFree(d_out); cudaFree(d_gbuf); cudaFree(d_flag);
  return 0;
}
