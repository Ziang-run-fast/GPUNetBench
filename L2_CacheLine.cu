#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256
#define NUM_BLOCKS 1
#define ITERATIONS 10000

// 使用volatile和__ldg确保绕过L1缓存
__global__ void stride_test(volatile float* __restrict__ data, int stride, int* clocks) {
    int tid = threadIdx.x;
    int idx = tid * stride;
    
    // 同步所有线程
    __syncthreads();
    
    // 开始计时
    unsigned int start_clock = clock();
    
    float sum = 0.0f;
    // 多次访问同一位置
    for (int i = 0; i < ITERATIONS; i++) {
        // 强制通过L2缓存访问
        sum += __ldg((const float*)&data[idx]);
    }
    
    // 结束计时
    unsigned int end_clock = clock();
    
    // 防止编译器优化掉计算
    if (sum > 1e20f) {
        data[idx] = sum;
    }
    
    // 只有线程0记录时间
    if (tid == 0) {
        *clocks = end_clock - start_clock;
    }
}

void run_stride_test() {
    printf("GPU L2 Cache Line Size Test (L1 Disabled)\n");
    printf("==========================================\n\n");
    
    // 分配大量内存确保数据在全局内存中
    size_t total_size = 256 * 1024 * 1024; // 256MB
    size_t num_elements = total_size / sizeof(float);
    
    float* h_data = (float*)malloc(total_size);
    float* d_data;
    int* d_clocks;
    int h_clocks;
    
    // 初始化数据
    for (size_t i = 0; i < num_elements; i++) {
        h_data[i] = (float)(i % 1000) + 1.0f;
    }
    
    cudaMalloc(&d_data, total_size);
    cudaMalloc(&d_clocks, sizeof(int));
    cudaMemcpy(d_data, h_data, total_size, cudaMemcpyHostToDevice);
    
    printf("Testing different strides to find cache line size...\n");
    printf("Each thread accesses the same location %d times\n\n", ITERATIONS);
    
    printf("Stride | Stride | Avg Clocks | Clocks per | Cache Line\n");
    printf("(ints) | (bytes)| per Access | Access     | Boundary?\n");
    printf("-------|--------|------------|------------|-----------\n");
    
    // 测试不同的stride值（以32位整数为单位）
    int strides[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int num_strides = sizeof(strides) / sizeof(strides[0]);
    
    double baseline_clocks = 0;
    
    for (int i = 0; i < num_strides; i++) {
        int stride = strides[i];
        
        // 确保stride不会导致越界访问
        if (stride * BLOCK_SIZE >= num_elements) {
            continue;
        }
        
        double total_clocks = 0;
        int num_runs = 10;
        
        // 多次运行取平均值
        for (int run = 0; run < num_runs; run++) {
            // 预热
            stride_test<<<NUM_BLOCKS, BLOCK_SIZE>>>((volatile float*)d_data, stride, d_clocks);
            cudaDeviceSynchronize();
            
            // 实际测试
            stride_test<<<NUM_BLOCKS, BLOCK_SIZE>>>((volatile float*)d_data, stride, d_clocks);
            cudaDeviceSynchronize();
            
            cudaMemcpy(&h_clocks, d_clocks, sizeof(int), cudaMemcpyDeviceToHost);
            total_clocks += h_clocks;
        }
        
        double avg_clocks = total_clocks / num_runs;
        double clocks_per_access = avg_clocks / ITERATIONS;
        
        if (i == 0) {
            baseline_clocks = clocks_per_access;
        }
        
        // 判断是否跨越了cache line边界
        double ratio = clocks_per_access / baseline_clocks;
        const char* boundary = (ratio > 1.5) ? "   YES" : "    NO";
        
        printf("%6d | %7d | %10.1f | %10.3f | %s\n", 
               stride, stride * 4, avg_clocks, clocks_per_access, boundary);
    }
    
    printf("\nAnalysis:\n");
    printf("---------\n");
    printf("Look for the first significant increase in 'Clocks per Access'.\n");
    printf("This indicates the cache line boundary has been crossed.\n");
    printf("Common GPU cache line sizes:\n");
    printf("- 32 bytes  (8 floats)   - stride 8\n");
    printf("- 64 bytes  (16 floats)  - stride 16  \n");
    printf("- 128 bytes (32 floats)  - stride 32\n");
    
    // 清理
    cudaFree(d_data);
    cudaFree(d_clocks);
    free(h_data);
}

int main() {
    // 检查CUDA支持
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        printf("No CUDA-capable devices found!\n");
        return -1;
    }
    
    // 设置设备并获取属性
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    printf("Clock Rate: %d MHz\n", prop.clockRate / 1000);
    printf("\n");
    
    run_stride_test();
    
    return 0;
}
