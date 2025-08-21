// nvcc -O3 -arch=sm_90 l2_line_min.cu -o l2_line_min
// Ampere 用 -arch=sm_80；其它架构改成对应 sm_XX

#include <cstdio>
#include <cuda_runtime.h>

#ifndef REPEATS
#define REPEATS 256          // 每个 delta 重复次数
#endif
#ifndef FLUSH_BYTES
#define FLUSH_BYTES (128ull<<20) // 128 MB 冲刷区，足以顶掉 H100 的 L2
#endif

// L2-only 读取（绕过 L1）
__device__ __forceinline__ uint32_t ldg_l2_u32(const uint32_t* p){
    return __ldcg(p);
}

__global__ void probe_kernel(
    uint8_t* base,            // 至少 max_delta+4 字节，建议 256B 对齐
    uint8_t* flush_buf,       // FLUSH_BYTES 大小
    const int* deltas, int nd,
    unsigned long long* out_cycles)
{
    if (blockIdx.x || threadIdx.x) return;

    for (int i = 0; i < nd; ++i){
        int delta = deltas[i];
        unsigned long long acc = 0;

        for (int r = 0; r < REPEATS; ++r){
            // 1) 冲刷 L2：顺序读大缓冲（步长 128B）
            for (size_t off = 0; off < FLUSH_BYTES; off += 128){
                (void)ldg_l2_u32(reinterpret_cast<const uint32_t*>(flush_buf + off));
            }

            // 2) 读 A，把 A 所在扇区/行带入 L2
            (void)ldg_l2_u32(reinterpret_cast<const uint32_t*>(base));

            // 防乱序：确保 B 的访问不会跑到前面
            asm volatile("membar.gl;");

            // 3) 只对 B 的读取计时
            unsigned long long t0 = clock64();
            (void)ldg_l2_u32(reinterpret_cast<const uint32_t*>(base + delta));
            unsigned long long t1 = clock64();

            acc += (t1 - t0);
        }
        out_cycles[i] = acc / REPEATS;
    }
}

static void ck(cudaError_t e, const char* m){
    if (e != cudaSuccess){ fprintf(stderr, "%s: %s\n", m, cudaGetErrorString(e)); std::exit(1); }
}

int main(){
    // 准备若干 delta（全部为 8 的倍数，覆盖常见扇区/行尺寸）
    const int deltas_host[] = {16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512};
    const int nd = (int)(sizeof(deltas_host)/sizeof(deltas_host[0]));
    const int max_delta = deltas_host[nd-1];

    // 分配显存：base（对齐到 256B）、flush（128MB）、输出
    uint8_t* d_base_raw = nullptr; ck(cudaMalloc(&d_base_raw, max_delta + 512), "malloc base");
    uintptr_t a = (reinterpret_cast<uintptr_t>(d_base_raw) + 255ull) & ~255ull;
    uint8_t* d_base = reinterpret_cast<uint8_t*>(a);
    ck(cudaMemset(d_base, 0xA5, max_delta + 4), "memset base");

    uint8_t* d_flush = nullptr; ck(cudaMalloc(&d_flush, FLUSH_BYTES), "malloc flush");
    ck(cudaMemset(d_flush, 0, FLUSH_BYTES), "memset flush");

    int* d_deltas = nullptr; ck(cudaMalloc(&d_deltas, nd * sizeof(int)), "malloc deltas");
    ck(cudaMemcpy(d_deltas, deltas_host, nd * sizeof(int), cudaMemcpyHostToDevice), "copy deltas");

    unsigned long long* d_out = nullptr; ck(cudaMalloc(&d_out, nd * sizeof(unsigned long long)), "malloc out");

    // 启动（单线程即可）
    probe_kernel<<<1,1>>>(d_base, d_flush, d_deltas, nd, d_out);
    ck(cudaDeviceSynchronize(), "sync");

    // 回读结果
    unsigned long long* out = new unsigned long long[nd];
    ck(cudaMemcpy(out, d_out, nd * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "copy out");

    printf("# delta_bytes, avg_cycles (B after A via __ldcg, L2-only)\n");
    for (int i = 0; i < nd; ++i){
        printf("%4d, %llu\n", deltas_host[i], out[i]);
    }

    // 释放
    delete[] out;
    cudaFree(d_out);
    cudaFree(d_deltas);
    cudaFree(d_flush);
    cudaFree(d_base_raw);
    return 0;
}
