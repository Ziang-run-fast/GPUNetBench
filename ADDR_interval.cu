#include <cstdio>
#include <cuda_runtime.h>
#include <stdint.h>

// 测试配置
#define TEST_SIZE (1 << 20)          // 1MB连续地址空间
#define PTR_ARRAY_SIZE (TEST_SIZE / sizeof(uint64_t))  // 指针数组大小
#define STRIDE_BYTES 128             // 步长：128字节
#define CHASE_LENGTH 100000          // 追逐迭代次数
#define WARMUP_CHASE 1000            // 预热追逐次数

// 计算128字节步长对应的指针索引偏移（每个指针8字节）
#define INDEX_STRIDE (STRIDE_BYTES / sizeof(uint64_t))  // 16（128/8）

// 指针追逐测试Kernel
__global__ void hbmPointerChaseKernel(unsigned long long *d_latency, uint64_t *d_ptr_chain) {
    // 仅使用单个线程避免缓存干扰
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long start, end;
        uint64_t *current_ptr;
        volatile uint64_t temp;  // 防止编译器优化

        // 初始化追逐起点
        current_ptr = &d_ptr_chain[0];

        // 预热：先追逐一定次数，确保HBM控制器激活
        for (int i = 0; i < WARMUP_CHASE; ++i) {
            current_ptr = (uint64_t*)(*current_ptr);
        }

        // 开始测量HBM访问延迟
        start = clock64();
        
        // 指针追逐主循环：按128字节步长访问1MB空间
        for (int i = 0; i < CHASE_LENGTH; ++i) {
            current_ptr = (uint64_t*)(*current_ptr);
            temp = (uint64_t)current_ptr;  // 强制使用结果
        }
        
        end = clock64();

        // 计算单次访问的平均周期数
        *d_latency = (end - start) / CHASE_LENGTH;
    }
}

// 创建128字节步长的指针链（限制在1MB空间内）
void create128ByteStrideChain(uint64_t *h_ptr_chain, uint64_t *d_ptr_chain) {
    for (int i = 0; i < PTR_ARRAY_SIZE; ++i) {
        // 计算下一个索引（128字节步长），超过1MB范围则循环
        int next_i = (i + INDEX_STRIDE) % PTR_ARRAY_SIZE;
        
        // 确保指针指向1MB空间内的下一个地址（严格限制在测试范围内）
        h_ptr_chain[i] = (uint64_t)&d_ptr_chain[next_i];
    }
}

int main() {
    int dev = 3;
    cudaSetDevice(dev);

    // 主机和设备内存分配
    uint64_t *h_ptr_chain = (uint64_t*)malloc(PTR_ARRAY_SIZE * sizeof(uint64_t));
    uint64_t *d_ptr_chain;  // 1MB连续空间的指针链
    unsigned long long *d_latency, h_latency;

    cudaMalloc(&d_ptr_chain, TEST_SIZE);  // 精确分配1MB空间
    cudaMalloc(&d_latency, sizeof(unsigned long long));

    // 创建128字节步长的指针链
    create128ByteStrideChain(h_ptr_chain, d_ptr_chain);
    cudaMemcpy(d_ptr_chain, h_ptr_chain, TEST_SIZE, cudaMemcpyHostToDevice);

    // 获取GPU属性
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    float clock_rate = prop.clockRate * 1000.0f;  // 转换为Hz

    // 执行指针追逐测试
    hbmPointerChaseKernel<<<1, 1>>>(d_latency, d_ptr_chain);
    cudaDeviceSynchronize();

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // 读取结果
    cudaMemcpy(&h_latency, d_latency, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // 计算延迟（纳秒）
    float latency_ns = (h_latency * 1e9f) / clock_rate;

    // 输出测试信息
    printf("GPU: %s (HBM总容量: %.2f GB)\n", 
           prop.name, (float)prop.totalGlobalMem / (1024*1024*1024));
    printf("测试配置: 1MB连续地址空间，步长=%d字节\n", STRIDE_BYTES);
    printf("追逐次数: %d\n", CHASE_LENGTH);
    printf("平均访问周期: %llu cycles\n", h_latency);
    printf("平均HBM访问延迟: %.2f ns\n", latency_ns);

    // 释放资源
    free(h_ptr_chain);
    cudaFree(d_ptr_chain);
    cudaFree(d_latency);

    return 0;
}
