#include <cstdio>
#include <cstdlib>
#include <ctime>

// 全局内存访问 kernel
__global__ void memoryAccessKernel(float *g_data, int stride, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dataSize = blockDim.x * gridDim.x * stride;
    
    // 重复访问以获得可测量的时间
    for (int i = 0; i < iterations; ++i) {
        int pos = idx * stride;
        // 确保访问在数组范围内
        if (pos < dataSize) {
            g_data[pos] = g_data[pos] * 0.5f + 1.0f;
        }
    }
}

// 计算带宽 (GB/s)
float calculateBandwidth(float time_ms, int dataSizeBytes, int iterations) {
    // 每个迭代完成一次读和一次写，共2次操作
    float totalBytes = 2.0f * dataSizeBytes * iterations;
    // 转换为GB/s (1 GB = 1e9 bytes)
    return (totalBytes / (1e9f)) / (time_ms / 1000.0f);
}

int main() {
    const int blockSize = 256;
    const int gridSize = 64;
    const int iterations = 10000;  // 迭代次数，确保测量时间足够长
    const int maxStride = 128;     // 最大步长，足够覆盖常见的cache line大小
    
    // 为每种步长分配内存
    float *d_data;
    int maxDataSize = blockSize * gridSize * maxStride;
    
    // 分配全局内存
    cudaMalloc((void**)&d_data, maxDataSize * sizeof(float));
    cudaMemset(d_data, 0, maxDataSize * sizeof(float));
    
    printf("Testing L2 Cache Line Size...\n");
    printf("Stride\tBandwidth (GB/s)\n");
    printf("-------------------------\n");
    
    // 测试不同的步长
    for (int stride = 1; stride <= maxStride; ++stride) {
        int dataSize = blockSize * gridSize * stride;
        
        // 启动kernel并计时
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start, 0);
        memoryAccessKernel<<<gridSize, blockSize>>>(d_data, stride, iterations);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        
        // 计算时间
        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        // 计算带宽
        float bandwidth = calculateBandwidth(time_ms, dataSize * sizeof(float), iterations);
        
        // 输出结果
        printf("%d\t%.2f\n", stride, bandwidth);
        
        // 清理
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // 释放内存
    cudaFree(d_data);
    
    printf("\nL2 Cache Line Size is where bandwidth drops significantly\n");
    
    return 0;
}
