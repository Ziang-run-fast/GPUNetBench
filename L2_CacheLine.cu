#include <cstdio>
#include <cuda_runtime.h>

// 定义足够大的数组大小，确保能覆盖缓存线测试范围
#define ARRAY_SIZE 4096
// 预热迭代次数
#define WARMUP_ITERATIONS 1000
// 测量迭代次数
#define MEASURE_ITERATIONS 100000

// 测试缓存线大小的kernel
__global__ void cacheLineTestKernel(unsigned long long *d_cycles, int *d_array) {
    // 仅让线程0执行
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long start, end;
        int temp;
        
        // 初始化数组
        for (int i = 0; i < ARRAY_SIZE; ++i) {
            d_array[i] = i;
        }
        
        // 预热阶段：将数组加载到缓存
        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            temp = d_array[i % ARRAY_SIZE];
        }
        
        // 测试不同步长下的访问延迟，步长从1到64（单位：int元素，每个int占4字节）
        for (int stride = 1; stride <= 64; ++stride) {
            // 开始计时
            start = clock64();
            
            // 以当前步长访问数组
            for (int i = 0; i < MEASURE_ITERATIONS; ++i) {
                int index = (i * stride) % ARRAY_SIZE;
                temp = __ldcg(&d_array[index]);  // 使用L2缓存加载
            }
            
            // 结束计时
            end = clock64();
            
            // 存储当前步长的周期数
            d_cycles[stride] = end - start;
        }
        
        // 防止编译器优化
        if (temp == 0) {
            d_array[0] = 0;
        }
    }
}

int main() {
    unsigned long long *d_cycles, *h_cycles;
    int *d_array;
    int dev = 3;
    
    // 设置GPU设备
    cudaSetDevice(dev);
    
    // 分配设备内存
    cudaMalloc((void**)&d_array, ARRAY_SIZE * sizeof(int));
    cudaMalloc((void**)&d_cycles, (65) * sizeof(unsigned long long));  // 存储步长1到64的结果
    
    // 分配主机内存
    h_cycles = (unsigned long long*)malloc(65 * sizeof(unsigned long long));
    
    // 启动kernel，仅使用1个线程
    cacheLineTestKernel<<<1, 1>>>(d_cycles, d_array);
    cudaDeviceSynchronize();  // 确保kernel执行完成
    
    // 复制结果到主机
    cudaMemcpy(h_cycles, d_cycles, 65 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    
    // 获取GPU时钟频率
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    float clock_rate = prop.clockRate * 1000.0f;  // 转换为Hz
    
    // 输出结果
    printf("GPU设备: %d, 名称: %s\n", dev, prop.name);
    printf("步长(int元素)\t总周期数\t平均周期\t平均延迟(ns)\n");
    printf("---------------------------------------------------------\n");
    
    for (int stride = 1; stride <= 64; ++stride) {
        float avg_cycles = (float)h_cycles[stride] / MEASURE_ITERATIONS;
        float avg_ns = (avg_cycles * 1e9f) / clock_rate;
        printf("%d\t\t%llu\t\t%.2f\t\t%.2f\n", 
               stride, h_cycles[stride], avg_cycles, avg_ns);
    }
    
    // 释放内存
    cudaFree(d_cycles);
    cudaFree(d_array);
    free(h_cycles);
    
    return 0;
}
