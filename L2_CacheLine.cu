// nvcc -O3 -arch=sm_80 -DDISABLE_L1 l2_line_probe.cu -o l2_line_probe
// (Hopper/Blackwell 改成对应 -arch，例如 sm_90 / sm_100)

#include <cstdio>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef REPEATS
#define REPEATS 200        // 每个 delta 重复次数
#endif

#ifndef FLUSH_MULT
#define FLUSH_MULT 3       // 冲刷区大小 = FLUSH_MULT * L2_size
#endif

// 选择加载修饰符：默认关闭L1，只有L2（.cg）
#ifdef DISABLE_L1
#define LD_MOD "ld.global.cg.u32"   // L2-only
#else
#define LD_MOD "ld.global.ca.u32"   // L1+L2
#endif

// 只从 L2 加载一个 u32（绕过 L1）
__device__ __forceinline__ uint32_t ldg_l2_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile(LD_MOD " %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

// 冲刷 L2：读取 flush_buf（步长取 128B）保证替换掉之前的缓存行
__device__ __forceinline__ void flush_L2(const uint8_t* flush_buf, size_t flush_bytes) {
    volatile uint32_t sink = 0;
    for (size_t off = 0; off + 128 <= flush_bytes; off += 128) {
        sink += ldg_l2_u32(reinterpret_cast<const uint32_t*>(flush_buf + off));
    }
    if (sink == 0xdeadbeef) asm volatile(""); // 防止被优化
}

// 计时某个 delta 的平均延迟（cycles）
__global__ void measure_kernel(uint8_t* base_aligned,
                               uint8_t* flush_buf,
                               size_t   flush_bytes,
                               const int* deltas,
                               int      ndeltas,
                               unsigned long long* out_cycles) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    for (int i = 0; i < ndeltas; ++i) {
        int delta = deltas[i];
        unsigned long long acc = 0;

        // 每个 delta 重复多次求均值
        for (int r = 0; r < REPEATS; ++r) {
            // 1) 冲刷 L2，清空历史
            flush_L2(flush_buf, flush_bytes);

            // 2) 预取/加载 A，确保 A 所在的 cache line 进入 L2
            (void)ldg_l2_u32(reinterpret_cast<const uint32_t*>(base_aligned));

            // 内存屏障，避免后续访问跑在前面
            asm volatile("membar.gl;");

            // 3) 计时访问 B = A + delta
            unsigned long long t0 = clock64();
            uint32_t v = ldg_l2_u32(reinterpret_cast<const uint32_t*>(base_aligned + delta));
            unsigned long long t1 = clock64();

            acc += (t1 - t0);

            // 避免优化（使用读到的值）
            if ((v & 0xFFFFFFFFu) == 0x12345678u) asm volatile("");
        }
        out_cycles[i] = acc / REPEATS;
    }
}

int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    int l2_bytes = 0;
    cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, dev);
    if (l2_bytes <= 0) {
        // 某些旧驱动可能返回 0，取保守值
        l2_bytes = 64 << 20; // 64MB
    }

    printf("Device: %s\n", prop.name);
    printf("L2 size: %d bytes\n", l2_bytes);
#ifdef DISABLE_L1
    printf("Load mode: ld.global.cg (L2 only, L1 disabled)\n");
#else
    printf("Load mode: ld.global.ca (L1+L2)\n");
#endif
    printf("Repeats per delta: %d\n", REPEATS);

    // 扫描的 delta（字节）
    std::vector<int> deltas = {
        16, 32, 48, 64, 80, 96, 112, 128,
        144, 160, 192, 224, 256, 288, 320, 384, 448, 512
    };
    const int nd = (int)deltas.size();

    // ---- 分配显存 ----
    // 冲刷区：FLUSH_MULT × L2 容量
    size_t flush_bytes = static_cast<size_t>(l2_bytes) * FLUSH_MULT;
    uint8_t* d_flush = nullptr;
    cudaMalloc(&d_flush, flush_bytes);
    cudaMemset(d_flush, 0, flush_bytes);

    // 基地址区域：需要 >= 最大 delta + 256 对齐裕量 + 4 字节
    int max_delta = deltas.back();
    size_t base_bytes = (size_t)max_delta + 512; // 多给一点，便于 256B 对齐
    uint8_t* d_base_raw = nullptr;
    cudaMalloc(&d_base_raw, base_bytes);
    cudaMemset(d_base_raw, 0xA5, base_bytes);

    // 让 base 对齐到 256B（通常 >= cache line）
    uintptr_t addr = reinterpret_cast<uintptr_t>(d_base_raw);
    uintptr_t aligned = (addr + 255u) & ~uintptr_t(255u);
    uint8_t* d_base_aligned = reinterpret_cast<uint8_t*>(aligned);

    // deltas、结果数组
    int* d_deltas = nullptr;
    unsigned long long* d_out = nullptr;
    cudaMalloc(&d_deltas, nd * sizeof(int));
    cudaMalloc(&d_out,    nd * sizeof(unsigned long long));
    cudaMemcpy(d_deltas, deltas.data(), nd * sizeof(int), cudaMemcpyHostToDevice);

    // ---- 启动核函数 ----
    measure_kernel<<<1,1>>>(d_base_aligned, d_flush, flush_bytes, d_deltas, nd, d_out);
    cudaDeviceSynchronize();

    std::vector<unsigned long long> out(nd);
    cudaMemcpy(out.data(), d_out, nd * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // ---- 打印结果 ----
    printf("\n# delta_bytes, avg_cycles (L2-only)\n");
    for (int i = 0; i < nd; ++i) {
        printf("%4d, %llu\n", deltas[i], out[i]);
    }

    // 资源回收
    cudaFree(d_out);
    cudaFree(d_deltas);
    cudaFree(d_base_raw);
    cudaFree(d_flush);
    return 0;
}
