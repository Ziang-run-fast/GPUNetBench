#include <cstdio>
#include <cuda_runtime.h>

// 全局变量，用于测试的内存地址
__device__ int d_var;

// 测试延迟的kernel，仅使用一个线程
__global__ void latencyTestKernel(unsigned long long *d_cycles) {
    // 仅让线程0执行
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long start, end;
        
        // 读取开始时的时钟周期
        start = clock64();
        
        // 重复访问同一个内存地址多次，增加测量精度
        for (int i = 0; i < 1000000; ++i) {
            d_var++;  // 简单的读写操作
        }
        
        // 读取结束时的时钟周期
        end = clock64();
        
        // 计算总周期数（减去循环本身的开销）
        *d_cycles = end - start;
    }
}

int main() {
    unsigned long long *d_cycles, h_cycles;
    
    // 分配设备内存
    cudaMalloc((void**)&d_cycles, sizeof(unsigned long long));
    
    // 初始化全局变量
    cudaMemset(&d_var, 0, sizeof(int));
    
    // 启动kernel，仅使用1个线程
    latencyTestKernel<<<1, 1>>>(d_cycles);
    
    // 复制结果到主机
    cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    
    // 获取GPU时钟频率
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float clock_rate = prop.clockRate * 1000.0f;  // 转换为Hz
    
    // 计算单次访问的延迟（纳秒）
    float avg_latency = (h_cycles * 1e9f) / (clock_rate * 1000000.0f);
    
    printf("平均内存访问延迟: %.2f 纳秒\n", avg_latency);
    
    // 释放内存
    cudaFree(d_cycles);
    
    return 0;
}
