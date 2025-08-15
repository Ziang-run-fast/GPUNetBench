#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>

#ifndef WARMUP_PASSES
#define WARMUP_PASSES 2
#endif

__device__ __forceinline__ unsigned get_smid() {
  unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid)); return smid;
}

// 目标：强制顺序 = t0 -> (L2 load) -> t1
// 用内联 PTX + 数据依赖把 t1 绑定在本次 load 的结果 v 上
__device__ __forceinline__ unsigned timed_ldcg_u32(const unsigned* ptr, unsigned long long& cyc) {
  unsigned v;
  unsigned long long t0, t1;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(ptr));
  // 把 v 作为输入依赖传给第二次读 clock，禁止其越过 load
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) : "r"(v));
  cyc = t1 - t0;
  return v;
}

__global__ void scan_kernel_full(const unsigned* __restrict__ bufU32,
                                 size_t words,          // 缓冲长度（以 4B 计）
                                 int targetSM,          // 目标 SM
                                 unsigned* __restrict__ out_lat_cycles,
                                 size_t step_words,     // 步长（以 4B 为单位）
                                 int* __restrict__ done_flag)
{
  if ((int)get_smid() != targetSM) return;
  // 极少数情况下同一 SM 可能被调度多个 block，留下单一执行者
  if (atomicCAS(done_flag, 0, 1) != 0) return;

  if (threadIdx.x != 0) return;  // 单线程串行，避免 MLP

  // 预热：把测量范围都触达一次（仍走 L2，绕过 L1）
  for (int w = 0; w < WARMUP_PASSES; ++w) {
    for (size_t i = 0; i < words; i += step_words) {
      unsigned long long dummy;
      (void)timed_ldcg_u32(bufU32 + i, dummy);
    }
  }
  __threadfence();

  // 正式测量：每个地址一次 load，记录 cycles
  size_t idx_out = 0;
  unsigned sink = 0;
  for (size_t i = 0; i < words; i += step_words) {
    unsigned long long cyc;
    unsigned v = timed_ldcg_u32(bufU32 + i, cyc);
    sink += v;                                  // 依赖，防优化
    out_lat_cycles[idx_out++] = (unsigned)cyc;  // 记录每次的往返周期
  }
  if (sink == 0xFFFFFFFFu) out_lat_cycles[0] = sink; // 防优化
}

int main(int argc, char** argv) {
  // 用法： ./l2full <targetSM> <byte_stride>
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <targetSM> <byte_stride>\n", argv[0]);
    fprintf(stderr, "  <byte_stride> 必须为 4 的整数倍（按 32-bit 加载）。\n");
    return 1;
  }
  int    targetSM = atoi(argv[1]);
  size_t strideB  = strtoull(argv[2], nullptr, 0);
  if (strideB == 0 || (strideB % 4) != 0) {
    fprintf(stderr, "ERROR: <byte_stride> 必须为 4 的整数倍。\n");
    return 1;
  }

  int dev = 0; cudaSetDevice(dev);
  cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);
  const int numSM = prop.multiProcessorCount;
  if (targetSM < 0 || targetSM >= numSM) {
    fprintf(stderr, "ERROR: targetSM=%d 越界（0..%d）。\n", targetSM, numSM - 1);
    return 1;
  }
  printf("GPU: %s  SMs=%d  clock=%.1f MHz  targetSM=%d\n",
         prop.name, numSM, prop.clockRate/1000.0, targetSM);

  size_t l2_bytes = prop.l2CacheSize;
  if (l2_bytes == 0) {
    fprintf(stderr, "ERROR: 读取到的 L2 容量为 0，设备/驱动可能未暴露该属性。\n");
    return 1;
  }
  printf("L2 Cache Size: %zu bytes (%.2f MB), stride=%zu bytes\n",
         l2_bytes, l2_bytes/(1024.0*1024.0), strideB);

  // 分配并初始化缓冲（作为 L2 后备）
  unsigned *d_buf = nullptr;
  cudaMalloc(&d_buf, l2_bytes);
  std::vector<unsigned> h_init(l2_bytes/4);
  for (size_t i=0;i<h_init.size();++i)
    h_init[i] = (unsigned)i*1664525u + 1013904223u;   // 非零花纹
  cudaMemcpy(d_buf, h_init.data(), l2_bytes, cudaMemcpyHostToDevice);

  const size_t step_words = strideB / 4;
  const size_t words      = l2_bytes / 4;
  const size_t samples    = (words + step_words - 1) / step_words;

  unsigned *d_lat=nullptr; cudaMalloc(&d_lat, samples*sizeof(unsigned));
  cudaMemset(d_lat, 0, samples*sizeof(unsigned));
  int *d_done=nullptr; cudaMalloc(&d_done, sizeof(int));
  cudaMemset(d_done, 0, sizeof(int));

  // 启动 numSM 个 blocks；只有 targetSM 上的那个真测量
  scan_kernel_full<<<numSM, 32>>>(d_buf, words, targetSM, d_lat, step_words, d_done);
  cudaDeviceSynchronize();

  // 回读并全量写 CSV
  std::vector<unsigned> h_lat(samples);
  cudaMemcpy(h_lat.data(), d_lat, samples*sizeof(unsigned), cudaMemcpyDeviceToHost);

  FILE* fp = fopen("l2_latency_full.csv", "w");
  if (!fp) { fprintf(stderr, "ERROR: 无法创建 l2_latency_full.csv\n"); return 1; }
  fprintf(fp, "Offset_bytes,Latency_cycles\n");
  for (size_t i=0;i<samples;++i) {
    const size_t offsetB = i * strideB;
    fprintf(fp, "%zu,%u\n", offsetB, h_lat[i]);
  }
  fclose(fp);
  printf("已写出 %zu 条样本到 l2_latency_full.csv。\n", samples);

  // 简要统计（看下是否仍然“全一样”）
  double avg=0.0; for (auto v: h_lat) avg += v; avg /= (double)samples;
  auto sorted = h_lat; std::sort(sorted.begin(), sorted.end());
  unsigned med = sorted[sorted.size()/2];
  printf("Summary: avg=%.2f cycles  median=%u cycles\n", avg, med);

  cudaFree(d_done);
  cudaFree(d_lat);
  cudaFree(d_buf);
  return 0;
}
