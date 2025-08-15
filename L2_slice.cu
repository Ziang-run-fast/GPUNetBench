// l2_scan_latency_targetSM_L2size.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>

#ifndef LINE_STRIDE_BYTES
#define LINE_STRIDE_BYTES 128   // 建议 64/128/256
#endif
#ifndef WARMUP_PASSES
#define WARMUP_PASSES 2
#endif

// 读当前 SMID
__device__ __forceinline__ unsigned get_smid() {
  unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid)); return smid;
}
// 绕过 L1、直达 L2 的 32-bit 读取
__device__ __forceinline__ unsigned ldg_l2_u32(const unsigned* p) {
#if __CUDA_ARCH__ >= 700
  return __ldcg(p);
#else
  unsigned v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
#endif
}

// 启动 numSM 个 blocks；只有位于 targetSM 的 block 执行测量
__global__ void scan_kernel(const unsigned* __restrict__ bufU32,
                            size_t words,
                            int targetSM,
                            unsigned* __restrict__ out_lat_cycles,
                            size_t step_words,
                            int* __restrict__ done_flag)
{
  // 不是目标 SM 的 block 直接退出
  if ((int)get_smid() != targetSM) return;

  // 极少数情况下同一 SM 可能被调度多个 block，确保只留一个执行者
  if (atomicCAS(done_flag, 0, 1) != 0) return;

  // 单线程串行测量，避免产生 MLP 隐藏单次延迟
  if (threadIdx.x != 0) return;

  // 预热：触发驻留/页表（仍走 L2）
  for (int w = 0; w < WARMUP_PASSES; ++w) {
    for (size_t i = 0; i < words; i += step_words) (void)ldg_l2_u32(bufU32 + i);
  }
  __threadfence();

  // 正式测量：对每个采样点执行一次 load，并记录 clock 差
  size_t idx_out = 0;
  unsigned sink = 0;
  for (size_t i = 0; i < words; i += step_words) {
    unsigned long long t0 = clock();
    unsigned v = ldg_l2_u32(bufU32 + i);
    unsigned long long t1 = clock();
    sink += v;  // 数据依赖，避免指令被消除/乱序
    out_lat_cycles[idx_out++] = (unsigned)(t1 - t0);
  }

  // 防止编译器优化掉 sink
  if (sink == 0xFFFFFFFFu) out_lat_cycles[0] = sink;
}

int main(int argc, char** argv) {
  // 用法： ./l2scan <targetSM> [line_stride_bytes]
  int    targetSM = 0;
  size_t strideB  = LINE_STRIDE_BYTES;
  if (argc > 1) targetSM = atoi(argv[1]);
  if (argc > 2) strideB  = strtoull(argv[2], nullptr, 0);
  if (strideB == 0 || (strideB % 4) != 0) {
    fprintf(stderr, "line_stride_bytes 必须是 4 的整数倍\n"); return 1;
  }

  // 设备信息
  int dev = 0; cudaSetDevice(dev);
  cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);
  const int numSM = prop.multiProcessorCount;
  if (targetSM < 0 || targetSM >= numSM) {
    fprintf(stderr, "targetSM=%d 越界（应在 0..%d）\n", targetSM, numSM - 1); return 1;
  }
  printf("GPU: %s  SMs=%d  clock=%.1f MHz  targetSM=%d\n",
         prop.name, numSM, prop.clockRate / 1000.0, targetSM);

  // 使用“L2 Cache 的大小”作为扫描缓冲的大小
  size_t buf_bytes = prop.l2CacheSize;
  if (buf_bytes == 0) {
    fprintf(stderr, "读取到的 L2 容量为 0，设备/驱动可能不支持该属性。\n");
    return 1;
  }
  printf("L2 Cache Size: %zu bytes (%.2f MB)\n", buf_bytes, buf_bytes / (1024.0 * 1024.0));

  // 分配并初始化全局缓冲（作为 L2 的后备）
  unsigned *d_buf = nullptr;
  cudaMalloc(&d_buf, buf_bytes);
  std::vector<unsigned> h_init(buf_bytes / 4);
  for (size_t i = 0; i < h_init.size(); ++i)
    h_init[i] = (unsigned)i * 1664525u + 1013904223u; // 简单非零花纹，避免压缩/零页快路
  cudaMemcpy(d_buf, h_init.data(), buf_bytes, cudaMemcpyHostToDevice);

  // 采样点数量
  const size_t words      = buf_bytes / 4;
  const size_t step_words = strideB / 4;
  const size_t samples    = (words + step_words - 1) / step_words;

  // 输出数组与“唯一执行者”标志
  unsigned *d_lat = nullptr; cudaMalloc(&d_lat, samples * sizeof(unsigned));
  cudaMemset(d_lat, 0, samples * sizeof(unsigned));
  int *d_done = nullptr; cudaMalloc(&d_done, sizeof(int));
  cudaMemset(d_done, 0, sizeof(int));

  // 启动 numSM 个 blocks；只有 targetSM 上的那个会真正执行测量
  // 用较小线程数即可（单线程测量），这里给 32
  scan_kernel<<<numSM, 32>>>(d_buf, words, targetSM, d_lat, step_words, d_done);
  cudaDeviceSynchronize();

  // 拷回结果并做简单统计
  std::vector<unsigned> h_lat(samples);
  cudaMemcpy(h_lat.data(), d_lat, samples * sizeof(unsigned), cudaMemcpyDeviceToHost);

  double avg = 0.0; for (auto v : h_lat) avg += v; avg /= (double)samples;
  auto sorted = h_lat; std::sort(sorted.begin(), sorted.end());
  unsigned med = sorted[sorted.size() / 2];

  printf("Samples=%zu  stride=%zub  avg=%.2f cycles  median=%u cycles\n",
         samples, strideB, avg, med);

  // 示例输出前 32 个点（需要全量可自行改为写文件）
  size_t to_print = std::min(samples, (size_t)32);
  for (size_t i = 0; i < to_print; ++i) {
    printf("addr+%8zub : %u cycles\n", i * strideB, h_lat[i]);
  }
  // host 端 main 末尾
  // 假设 total_bytes = L2 Cache 大小，strideB = 访问步长
  size_t total_bytes = l2_size_bytes; // 由 cudaDeviceProp.l2CacheSize 获取
  size_t strideB = 4; // 每次访问 4 字节（int）
  
  size_t total_accesses = total_bytes / strideB;
  std::vector<unsigned> h_lat(total_accesses);
  
  // device kernel 中每个线程负责测试一个地址的延迟
  // 然后在 host 端收集 h_lat
  
  // --- 保存到 CSV 文件 ---
  FILE* fp = fopen("l2_latency_full.csv", "w");
  if (!fp) {
      fprintf(stderr, "无法创建 CSV 文件\n");
      return 1;
  }
  fprintf(fp, "Offset_bytes,Latency_cycles\n");
  for (size_t i = 0; i < total_accesses; ++i) {
      fprintf(fp, "%zu,%u\n", i * strideB, h_lat[i]);
  }
  fclose(fp);
  printf("已将延迟数据写入 l2_latency_full.csv\n");
  cudaFree(d_done);
  cudaFree(d_lat);
  cudaFree(d_buf);
  return 0;
}
