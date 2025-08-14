#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef ITERATION
#define ITERATION 10000
#endif

// 获取SM ID的内联PTX
__device__ inline unsigned int get_smid() {
    unsigned int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

/*
 * 简单的L2通信内核
 * - 块0（写入方）：将数据写入全局内存（L2缓存）
 * - 块1（读取方）：从全局内存读取数据并测量延迟
 */
__global__ void simple_l2_kernel(int *global_buffer, 
                                 unsigned long long *results, 
                                 int buffer_size) {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    unsigned int my_smid = get_smid();
    
    // 同步标志：确保写入完成后再读取
    __shared__ int sync_flag;
    if (thread_id == 0) sync_flag = 0;
    __syncthreads();
    
    if (block_id == 0) {
        // ========== 写入方（块0）==========
        // 记录写入方的SM ID
        if (thread_id == 0) {
            results[0] = my_smid;  // 写入方SM ID
        }
        
        // 将数据写入全局内存（L2缓存）
        for (int i = thread_id; i < buffer_size; i += blockDim.x) {
            global_buffer[i] = my_smid + i;  // 写入唯一数据
        }
        
        // 确保写入完成并对其他块可见
        __syncthreads();
        __threadfence();
        
        // 设置完成标志
        if (thread_id == 0) {
            atomicExch(&global_buffer[buffer_size], 1);  // 使用buffer末尾作为标志
        }
        
    } else if (block_id == 1) {
        // ========== 读取方（块1）==========
        // 记录读取方的SM ID
        if (thread_id == 0) {
            results[1] = my_smid;  // 读取方SM ID
        }
        
        // 等待写入完成
        if (thread_id == 0) {
            while (atomicAdd(&global_buffer[buffer_size], 0) == 0) {
                // 自旋等待写入完成
            }
        }
        __syncthreads();
        
        // 开始延迟测量
        unsigned long long total_latency = 0;
        int access_count = 0;
        
        for (int rep = 0; rep < ITERATION; rep++) {
            int idx = thread_id;
            
            while (idx < buffer_size) {
                // 测量单次访问延迟
                unsigned long long start_time = clock();
                
                // 从L2缓存读取数据
                volatile int data = global_buffer[idx];
                
                unsigned long long end_time = clock();
                
                total_latency += (end_time - start_time);
                access_count++;
                
                idx += blockDim.x;
            }
        }
        
        // 计算平均延迟
        unsigned long long avg_latency = access_count ? (total_latency / access_count) : 0;
        
        // 存储每个线程的平均延迟
        results[2 + thread_id] = avg_latency;
    }
}

int main(int argc, char **argv) {
    // 参数解析
    int buffer_size = (argc > 1) ? atoi(argv[1]) : 1024 * 1024;  // 默认1M个int
    int block_size = (argc > 2) ? atoi(argv[2]) : 256;           // 默认256线程
    
    printf("L2 Cache Communication Latency Test\n");
    printf("Buffer size: %d ints (%.2f MB)\n", buffer_size, buffer_size * sizeof(int) / 1e6);
    printf("Block size: %d threads\n", block_size);
    
    // 设备属性
    int dev = 0;
    cudaDeviceProp prop;
    cudaSetDevice(dev);
    cudaGetDeviceProperties(&prop, dev);
    
    printf("GPU: %s\n", prop.name);
    printf("Clock Rate: %.2f GHz\n", prop.clockRate / 1e6);
    
    // 分配内存
    int *d_buffer;
    unsigned long long *d_results, *h_results;
    
    // 全局内存缓冲区（+1个int用作同步标志）
    cudaMalloc(&d_buffer, (buffer_size + 1) * sizeof(int));
    cudaMemset(d_buffer, 0, (buffer_size + 1) * sizeof(int));
    
    // 结果数组：[写入方SM_ID, 读取方SM_ID, 每线程延迟...]
    size_t results_size = (2 + block_size) * sizeof(unsigned long long);
    cudaMalloc(&d_results, results_size);
    cudaMemset(d_results, 0, results_size);
    
    h_results = (unsigned long long*)malloc(results_size);
    
    // 启动内核：2个块，每个块block_size个线程
    printf("\nLaunching kernel with 2 blocks...\n");
    
    simple_l2_kernel<<<2, block_size>>>(d_buffer, d_results, buffer_size);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // 复制结果
    cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost);
    
    // 分析结果
    unsigned long long writer_sm = h_results[0];
    unsigned long long reader_sm = h_results[1];
    
    printf("\nCommunication Results:\n");
    printf("Writer SM ID: %llu\n", writer_sm);
    printf("Reader SM ID: %llu\n", reader_sm);
    
    if (writer_sm == reader_sm) {
        printf("WARNING: Both blocks running on same SM (%llu)\n", writer_sm);
    } else {
        printf("SUCCESS: Communication between SM %llu -> SM %llu\n", writer_sm, reader_sm);
    }
    
    // 计算延迟统计
    unsigned long long total_latency = 0;
    unsigned long long min_latency = ULLONG_MAX;
    unsigned long long max_latency = 0;
    
    printf("\nPer-thread Latency (clock cycles):\n");
    for (int t = 0; t < block_size; t++) {
        unsigned long long lat = h_results[2 + t];
        total_latency += lat;
        if (lat < min_latency) min_latency = lat;
        if (lat > max_latency) max_latency = lat;
        
        if (t < 10) {  // 只显示前10个线程
            printf("Thread %2d: %llu cycles\n", t, lat);
        }
    }
    
    double avg_latency = (double)total_latency / block_size;
    
    printf("\nLatency Statistics:\n");
    printf("Average Latency: %.2f clock cycles\n", avg_latency);
    printf("Min Latency: %llu clock cycles\n", min_latency);
    printf("Max Latency: %llu clock cycles\n", max_latency);
    
    // 转换为纳秒（如果需要）
    double clock_period_ns = 1e9 / (prop.clockRate * 1000.0);
    printf("Average Latency: %.2f ns\n", avg_latency * clock_period_ns);
    
    // 清理内存
    cudaFree(d_buffer);
    cudaFree(d_results);
    free(h_results);
    
    return 0;
}
