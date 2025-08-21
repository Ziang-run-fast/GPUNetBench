#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define WARMUP_ITERATIONS 1000
#define TEST_ITERATIONS 10000
#define MAX_REQUEST_SIZE 256  // 最大请求大小（字节）

// 获取SM ID的设备函数
__device__ __forceinline__ unsigned int get_smid() {
    unsigned int smid;
    asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
}

// 获取高精度时钟
__device__ __forceinline__ uint64_t get_clock64() {
    uint64_t clock_val;
    asm volatile("mov.u64 %0, %clock64;" : "=l"(clock_val));
    return clock_val;
}

// 内存屏障，确保内存访问顺序
__device__ __forceinline__ void memory_fence() {
    __threadfence();
    asm volatile("bar.sync 0;");
}

// 强制L2访问（绕过L1）的内联汇编函数
__device__ __forceinline__ float load_l2_only(const float* addr) {
    float value;
    // 使用.cg（缓存在全局级别）修饰符强制L2访问
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(value) : "l"(addr));
    return value;
}

// 测试不同大小连续内存访问的延迟
__global__ void measure_cacheline_latency(
    const float* __restrict__ data,
    uint64_t* latencies,
    int* request_sizes,
    int num_sizes,
    size_t data_stride
) {
    // 只让SMID=0的第一个warp的第一个线程执行测试
    unsigned int smid = get_smid();
    if (smid != 0 || blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    
    // 基地址，确保对齐
    const float* base_addr = data;
    
    for (int size_idx = 0; size_idx < num_sizes; size_idx++) {
        int request_size_bytes = request_sizes[size_idx];
        int num_floats = request_size_bytes / sizeof(float);
        
        // 预热阶段 - 确保数据不在任何缓存中
        for (int w = 0; w < WARMUP_ITERATIONS; w++) {
            volatile float dummy = 0;
            const float* addr = base_addr + (w * data_stride);
            
            for (int i = 0; i < num_floats; i++) {
                dummy += load_l2_only(addr + i);
            }
            
            // 防止编译器优化
            if (dummy > 1e20f) {
                *((volatile float*)&base_addr[0]) = dummy;
            }
        }
        
        memory_fence();
        
        uint64_t total_cycles = 0;
        
        // 实际测试阶段
        for (int iter = 0; iter < TEST_ITERATIONS; iter++) {
            const float* test_addr = base_addr + (iter * data_stride);
            
            memory_fence();
            uint64_t start_time = get_clock64();
            
            // 连续访问指定大小的内存
            volatile float sum = 0;
            for (int i = 0; i < num_floats; i++) {
                sum += load_l2_only(test_addr + i);
            }
            
            memory_fence();
            uint64_t end_time = get_clock64();
            
            total_cycles += (end_time - start_time);
            
            // 防止编译器优化
            if (sum > 1e20f) {
                *((volatile float*)&base_addr[0]) = sum;
            }
        }
        
        latencies[size_idx] = total_cycles;
    }
}

// 测试随机访问模式以区分cacheline效应
__global__ void measure_random_access_latency(
    const float* __restrict__ data,
    uint64_t* latencies,
    int* offsets,
    int num_offsets,
    size_t total_elements
) {
    unsigned int smid = get_smid();
    if (smid != 0 || blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    
    // 预热
    for (int w = 0; w < WARMUP_ITERATIONS; w++) {
        volatile float dummy = load_l2_only(&data[w % 1024]);
        if (dummy > 1e20f) {
            *((volatile float*)&data[0]) = dummy;
        }
    }
    
    memory_fence();
    
    for (int offset_idx = 0; offset_idx < num_offsets; offset_idx++) {
        int offset = offsets[offset_idx];
        uint64_t total_cycles = 0;
        
        for (int iter = 0; iter < TEST_ITERATIONS; iter++) {
            // 计算随机但固定的地址
            size_t addr_idx = (iter * 1009 + offset) % (total_elements - offset);
            
            memory_fence();
            uint64_t start_time = get_clock64();
            
            volatile float val = load_l2_only(&data[addr_idx]);
            
            memory_fence();
            uint64_t end_time = get_clock64();
            
            total_cycles += (end_time - start_time);
            
            if (val > 1e20f) {
                *((volatile float*)&data[0]) = val;
            }
        }
        
        latencies[offset_idx] = total_cycles;
    }
}

void print_device_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("=== GPU Device Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessor Count: %d\n", prop.multiProcessorCount);
    printf("L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate * 2.0 / 1e6);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("\n");
}

void test_sequential_access() {
    printf("=== Sequential Access Latency Test (SM 0 only) ===\n");
    printf("Testing different request sizes to identify cache line boundaries\n\n");
    
    // 分配大内存确保不会有缓存冲突
    size_t total_elements = 64 * 1024 * 1024;  // 256MB
    size_t bytes = total_elements * sizeof(float);
    size_t data_stride = 1024;  // 每次测试间隔1024个float以避免缓存影响
    
    float* h_data = (float*)malloc(bytes);
    float* d_data;
    uint64_t* d_latencies;
    int* d_request_sizes;
    
    // 初始化数据
    for (size_t i = 0; i < total_elements; i++) {
        h_data[i] = (float)(i & 0xFF) + 1.0f;
    }
    
    // 测试的请求大小（字节）
    int request_sizes[] = {4, 8, 16, 32, 64, 128, 256};
    int num_sizes = sizeof(request_sizes) / sizeof(request_sizes[0]);
    
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_latencies, num_sizes * sizeof(uint64_t));
    cudaMalloc(&d_request_sizes, num_sizes * sizeof(int));
    
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_request_sizes, request_sizes, num_sizes * sizeof(int), cudaMemcpyHostToDevice);
    
    // 执行测试
    measure_cacheline_latency<<<1, 32>>>(d_data, d_latencies, d_request_sizes, num_sizes, data_stride);
    cudaDeviceSynchronize();
    
    // 获取结果
    uint64_t* h_latencies = (uint64_t*)malloc(num_sizes * sizeof(uint64_t));
    cudaMemcpy(h_latencies, d_latencies, num_sizes * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    printf("Request Size | Total Cycles | Avg Cycles | Cycles/Byte | Relative\n");
    printf("   (bytes)   |   (%d iter)  |  per Req   |             | Latency\n", TEST_ITERATIONS);
    printf("-------------|--------------|------------|-------------|----------\n");
    
    uint64_t baseline_cycles = h_latencies[0];
    
    for (int i = 0; i < num_sizes; i++) {
        double avg_cycles = (double)h_latencies[i] / TEST_ITERATIONS;
        double cycles_per_byte = avg_cycles / request_sizes[i];
        double relative_latency = (double)h_latencies[i] / baseline_cycles;
        
        printf("%12d | %12lu | %10.2f | %11.3f | %8.2fx\n",
               request_sizes[i], h_latencies[i], avg_cycles, 
               cycles_per_byte, relative_latency);
    }
    
    printf("\nAnalysis:\n");
    printf("- Look for plateaus in 'Cycles/Byte' - these indicate cache line boundaries\n");
    printf("- The cache line size is where cycles/byte stops decreasing significantly\n");
    printf("- Typical GPU L2 cache line sizes: 32B, 64B, or 128B\n\n");
    
    // 清理
    cudaFree(d_data);
    cudaFree(d_latencies);
    cudaFree(d_request_sizes);
    free(h_data);
    free(h_latencies);
}

void test_offset_access() {
    printf("=== Offset Access Pattern Test ===\n");
    printf("Testing memory access with different byte offsets\n\n");
    
    size_t total_elements = 16 * 1024 * 1024;
    size_t bytes = total_elements * sizeof(float);
    
    float* h_data = (float*)malloc(bytes);
    float* d_data;
    uint64_t* d_latencies;
    int* d_offsets;
    
    for (size_t i = 0; i < total_elements; i++) {
        h_data[i] = (float)(i & 0xFF) + 1.0f;
    }
    
    // 测试不同的字节偏移
    int offsets[] = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64};
    int num_offsets = sizeof(offsets) / sizeof(offsets[0]);
    
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_latencies, num_offsets * sizeof(uint64_t));
    cudaMalloc(&d_offsets, num_offsets * sizeof(int));
    
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets, num_offsets * sizeof(int), cudaMemcpyHostToDevice);
    
    measure_random_access_latency<<<1, 32>>>(d_data, d_latencies, d_offsets, num_offsets, total_elements);
    cudaDeviceSynchronize();
    
    uint64_t* h_latencies = (uint64_t*)malloc(num_offsets * sizeof(uint64_t));
    cudaMemcpy(h_latencies, d_latencies, num_offsets * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    printf("Offset | Total Cycles | Avg Cycles | Relative\n");
    printf("(bytes)|  (%d iter)  |  per Access| Latency\n", TEST_ITERATIONS);
    printf("-------|--------------|------------|----------\n");
    
    uint64_t baseline = h_latencies[0];
    
    for (int i = 0; i < num_offsets; i++) {
        double avg_cycles = (double)h_latencies[i] / TEST_ITERATIONS;
        double relative = (double)h_latencies[i] / baseline;
        
        printf("%6d | %12lu | %10.2f | %8.2fx\n",
               offsets[i], h_latencies[i], avg_cycles, relative);
    }
    
    printf("\nAnalysis:\n");
    printf("- Accesses within the same cache line should have similar latency\n");
    printf("- Latency increases when crossing cache line boundaries\n\n");
    
    cudaFree(d_data);
    cudaFree(d_latencies);
    cudaFree(d_offsets);
    free(h_data);
    free(h_latencies);
}

int main() {
    // 检查CUDA设备
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        return -1;
    }
    
    cudaSetDevice(0);
    print_device_info();
    
    // 执行两种测试
    test_sequential_access();
    test_offset_access();
    
    printf("Test completed. The cache line size can be determined from the\n");
    printf("patterns in the latency measurements above.\n");
    
    return 0;
}
