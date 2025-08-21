// nvcc -O3 -arch=sm_90 l2_sector_probe_ldcg.cu -o l2_sector_probe_ldcg
// Ampere 用 -arch=sm_80；Blackwell 用对应 sm_XX

#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef WARM_STEPS
#define WARM_STEPS (1<<18)   // 预热解引用次数（把工作集放进 L2）
#endif
#ifndef MEAS_STEPS
#define MEAS_STEPS (1<<21)   // 计时解引用次数（依赖链，真实延迟）
#endif
#ifndef REPEATS
#define REPEATS 3            // 每个 delta 重复次数，取中位数更稳
#endif

// L2-only 加载
__device__ __forceinline__ unsigned long long ld_l2_u64(const unsigned long long* p){
    return __ldcg(p); // 只经由 L2，绕过 L1
}

// 依赖链：p = *p
__device__ __forceinline__ unsigned long long* chase_once(unsigned long long* p){
    unsigned long long nxt = ld_l2_u64((const unsigned long long*)p);
    return (unsigned long long*)nxt;
}

__global__ void pc_kernel(unsigned long long* start,
                          const unsigned long long* flush, size_t flush_u64s,
                          unsigned long long* out_cycles) {
    if (blockIdx.x || threadIdx.x) return;

    // 轻量冲刷：触碰一片远大于工作集的区域，替换掉旧 L2 内容
    unsigned long long sink = 0;
    for (size_t i = 0; i < flush_u64s; i += 32)
        sink += __ldcg(flush + i);
    if (sink == 0xdeadbeefULL) asm volatile("");

    // 预热：把链表工作集驻留到 L2
    unsigned long long* p = start;
    for (int i = 0; i < WARM_STEPS; ++i) p = chase_once(p);

    // 计时（依赖链确保串行化，无法被乱序/合并隐藏）
    unsigned long long* q = p;
    unsigned long long t0 = clock64();
#pragma unroll 1
    for (int i = 0; i < MEAS_STEPS; ++i) q = chase_once(q);
    unsigned long long t1 = clock64();

    if (((uintptr_t)q & 1ULL) == 0xFFFFFFFFULL) asm volatile(""); // 防优化
    *out_cycles = (t1 - t0) / (unsigned long long)MEAS_STEPS;
}

static void ck(cudaError_t e, const char* msg){
    if (e != cudaSuccess){ fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e)); exit(1); }
}

// 在 d_region 上按 delta 字节一节点构造环形链表；返回起点
unsigned long long* build_ring(void* d_region, size_t region_bytes, int delta_bytes){
    if (delta_bytes < 8) delta_bytes = 8;
    delta_bytes = (delta_bytes + 7) & ~7; // 8B 对齐
    size_t nodes = region_bytes / (size_t)delta_bytes;
    if (nodes < 2) { fprintf(stderr, "region too small for delta=%d\n", delta_bytes); exit(1); }

    // 在 host 侧写入“下一节点”的设备地址
    std::vector<unsigned char> host(region_bytes, 0);
    auto base = (uintptr_t)d_region;
    for (size_t i = 0; i < nodes; ++i){
        uintptr_t curr = base + i * (uintptr_t)delta_bytes;
        uintptr_t next = base + ((i + 1) % nodes) * (uintptr_t)delta_bytes;
        *reinterpret_cast<unsigned long long*>(&host[i * (size_t)delta_bytes]) =
            (unsigned long long)next;
    }
    ck(cudaMemcpy(d_region, host.data(), region_bytes, cudaMemcpyHostToDevice), "memcpy ring");
    return (unsigned long long*)base;
}

int main(){
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp prop{};  cudaGetDeviceProperties(&prop, dev);
    int l2_bytes = 0;       cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, dev);
    if (l2_bytes <= 0) l2_bytes = 64<<20; // 兜底

    printf("Device: %s\nL2 size: %d bytes\n", prop.name, l2_bytes);

    // 工作集：<= L2 的 1/8（最大 8MB），确保可完全命中
    size_t ring_bytes  = std::min((size_t)l2_bytes / 8, (size_t)8<<20);
    // 冲刷水：2×L2（最少 16MB）
    size_t flush_bytes = std::max((size_t)l2_bytes * 2ull, (size_t)16<<20);

    // 256B 对齐
    void* d_region_raw = nullptr; ck(cudaMalloc(&d_region_raw, ring_bytes + 256), "malloc region");
    uintptr_t a = ((uintptr_t)d_region_raw + 255ull) & ~255ull;
    void* d_region = (void*)a;

    unsigned long long* d_flush = nullptr; ck(cudaMalloc((void**)&d_flush, flush_bytes), "malloc flush");
    ck(cudaMemset(d_flush, 0, flush_bytes), "memset flush");

    // 扫描的 delta（可按需扩展）
    std::vector<int> deltas = {8,16,24,32,40,48,56,64,80,96,112,128,160,192,224,256,320,384,448,512};

    unsigned long long* d_out = nullptr; ck(cudaMalloc(&d_out, sizeof(unsigned long long)), "malloc out");

    printf("\n# delta_bytes, avg_cycles (pointer-chasing via __ldcg, L2-only)\n");
    for (int delta : deltas){
        // 每个 delta 重建一次环，确保严格的物理间隔
        unsigned long long* start = build_ring(d_region, ring_bytes, delta);

        // 重复多次，取中位数
        std::vector<unsigned long long> reps;
        for (int r = 0; r < REPEATS; ++r){
            pc_kernel<<<1,1>>>(start, d_flush, flush_bytes/8, d_out);
            cudaDeviceSynchronize();
            unsigned long long cyc;
            ck(cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost), "copy out");
            reps.push_back(cyc);
        }
        std::sort(reps.begin(), reps.end());
        unsigned long long med = reps[REPEATS/2];
        printf("%4d, %llu\n", delta, med);
    }

    cudaFree(d_out);
    cudaFree(d_flush);
    cudaFree(d_region_raw);
    return 0;
}
