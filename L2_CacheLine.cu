#include <cstdio>
#include <cuda_runtime.h>


#define INTERATION 1000
// 全局变量，用于测试的内存地址
// __device__ int d_var;

// 测试延迟的kernel，仅使用一个线程
__global__ void latencyTestKernel(unsigned long long *d_cycles, int *d_var) {
    // 仅让线程0执行
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long start, end;

        int temp;
        *d_var = 1;

        
        // // 重复访问同一个内存地址多次，增加测量精度
        for (int i = 0; i < 1000; ++i) {
            *d_var = i;
            temp = *d_var;
        }
        // 读取开始时的时钟周期
        start = clock64();
        //printf("Start time is %llu\n", start);

        for (int i = 0; i < 1; ++i) {
            // *d_var = i;
            temp = __ldcg(d_var);
            // temp = *d_var;
        }

        // 读取结束时的时钟周期
        end = clock64();
        //printf("End time is %llu\n", end);
        
        // 计算总周期数（减去循环本身的开销）
        *d_cycles = end - start;

        printf("End time is %llu\n", *d_cycles);
        if(temp == 0){
            *d_var = 0;
        }
    }
}

int main() {
    unsigned long long *d_cycles, h_cycles;

    int dev=3; cudaSetDevice(dev);
    // 分配设备内存
    int *d_var;
    cudaMalloc((void**)&d_var, sizeof(int));
    cudaMalloc((void**)&d_cycles, sizeof(unsigned long long));
    
    // 初始化全局变量
    // cudaMemset(&d_var, dev, sizeof(int));
    
    // 启动kernel，仅使用1个线程
    latencyTestKernel<<<1, 1>>>(d_cycles,d_var);
    
    // 复制结果到主机
    cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    
    // 获取GPU时钟频率
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
        
    // 释放内存
    cudaFree(d_cycles);

    
    return 0;
}
