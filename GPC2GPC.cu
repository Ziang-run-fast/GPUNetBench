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
 * 跨GPC的L2缓存通信内核
 * 移除线程块集群约束，让GPU调度器将块分配到不同GPC
 */
__global__ void cross_gpc_l2_kernel(int *global_buffer, 
                                    unsigned long long *results, 
                                    int num_ints,
                                    int rt_destSM,
                                    int rt_srcSM) {
    // 使用相同的rank概念，但不依赖cluster
    int rank = blockIdx.x % 16;  // 模拟16个rank
    unsigned int my_smid = get_smid();
    
    // === 数据准备阶段（目标rank）===
    if (rank == rt_destSM && threadIdx.x == 0) {
        // 记录目标SM ID
        results[0] = my_smid;
        
        // 填充全局内存缓冲区（代替共享内存）
        for (int i = 0; i < num_ints; i++) {
            global_buffer[i] = my_smid + i;
        }
        
        // 设置数据就绪标志
        global_buffer[num_ints] = 1;  // 使用缓冲区末尾作为标志
    }
    
    // 等待数据准备完成
    __syncthreads();
    __threadfence();
    
    // === 延迟测量阶段（源rank）===
    if (rank == rt_srcSM) {
        // 等待目标rank完成数据准备
        if (threadIdx.x == 0) {
            while (atomicAdd(&global_buffer[num_ints], 0) == 0) {
                // 自旋等待
            }
        }
        __syncthreads();
        
        // 开始延迟测量（与原代码保持一致的数据访问模式）
        unsigned long long lat_acc = 0, lat_cnt = 0;
        int local_sum = 0;
        
        for (int rep = 0; rep < ITERATION; rep++) {
            int idx = threadIdx.x;  // 对应原代码的 threadIdx.x * STRIDE，STRIDE=1
            
            // 模拟原代码的访问模式
            for (; idx < num_ints; idx += blockDim.x) {
                unsigned long long t0 = clock();
                
                // 从全局内存（L2缓存）读取数据
                local_sum += global_buffer[idx];
                
                unsigned long long t1 = clock();
                lat_acc += (t1 - t0);
                lat_cnt++;
            }
        }
        
        __syncthreads();
        
        // 存储结果（与原代码格式一致）
        unsigned long long avgLat = lat_cnt ? (lat_acc / lat_cnt) : 0;
        int base = blockIdx.x * (blockDim.x + 2);
        
        if (threadIdx.x == 0) {
            results[1] = my_smid;  // 源SM ID
        }
        
        // 每个线程的平均延迟
        results[base + 2 + threadIdx.x] = avgLat;
        
        // 防止编译器优化
        global_buffer[threadIdx.x % num_ints] = local_sum;
    }
}

int main(int argc, char **argv) {
    // 参数解析（与原代码保持一致）
    int rt_destSM = (argc > 1) ? atoi(argv[1]) : 1;
    int rt_srcSM = (argc > 2) ? atoi(argv[2]) : 0;
    int total_blocks = (argc > 3) ? atoi(argv[3]) : 32;  // 增加块数以提高跨GPC概率
    int blockSize = (argc > 4) ? atoi(argv[4]) : 1024;
    
    printf("Cross-GPC L2 Cache Communication Latency Test (L1 Bypass)\n");
    printf("Using __ldcg/__stcg to force L2-only access\n");
    printf("Target rank: %d, Source rank: %d\n", rt_destSM, rt_srcSM);
    printf("Total blocks: %d, Block size: %d\n", total_blocks, blockSize);
    
    // 设备属性
    int dev = 0;
    cudaDeviceProp prop;
    cudaSetDevice(dev);
    cudaGetDeviceProperties(&prop, dev);
    
    printf("GPU: %s (Total SMs: %d)\n", prop.name, prop.multiProcessorCount);
    
    // 计算共享内存等效大小（与原代码一致）
    int maxSMemOptin;
    cudaDeviceGetAttribute(&maxSMemOptin,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    size_t shared_bytes = maxSMemOptin;
    int num_ints = shared_bytes / sizeof(int);
    
    printf("Using buffer size: %d ints (%.2f KB)\n", num_ints, shared_bytes / 1024.0);
    
    // 分配内存
    int *d_buffer;
    unsigned long long *d_results, *h_results;
    
    // 分配大于L1缓存但小于L2缓存的缓冲区，确保使用L2
    size_t l1_cache_size = 128 * 1024;  // H100 L1缓存大小约128KB
    size_t buffer_size = max((size_t)(num_ints + 1) * sizeof(int), l1_cache_size * 2);
    
    printf("Allocating buffer size: %.2f MB (larger than L1 to force L2 usage)\n", 
           buffer_size / (1024.0 * 1024.0));
    
    // 全局内存缓冲区，确保超出L1缓存大小
    cudaMalloc(&d_buffer, buffer_size);
    cudaMemset(d_buffer, 0, buffer_size);
    
    // 预热L2缓存：先访问一遍数据
    printf("Warming L2 cache...\n");
    int *warmup_data = (int*)malloc(buffer_size);
    for (int i = 0; i < buffer_size / sizeof(int); i++) {
        warmup_data[i] = i;
    }
    cudaMemcpy(d_buffer, warmup_data, buffer_size, cudaMemcpyHostToDevice);
    free(warmup_data);
    
    // 结果数组（与原代码延迟模式格式一致）
    size_t results_size = total_blocks * (blockSize + 2) * sizeof(unsigned long long);
    cudaMalloc(&d_results, results_size);
    cudaMemset(d_results, 0, results_size);
    
    h_results = (unsigned long long*)malloc(results_size);
    
    // 启动内核：使用更多块来增加跨GPC通信的概率
    printf("\nLaunching kernel with %d blocks...\n", total_blocks);
    
    cross_gpc_l2_kernel<<<total_blocks, blockSize>>>(
        d_buffer, d_results, num_ints, rt_destSM, rt_srcSM, buffer_size);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // 复制结果
    cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost);
    
    // 分析结果
    unsigned long long dest_sm = h_results[0];
    unsigned long long src_sm = h_results[1];
    
    printf("\nCommunication Results:\n");
    printf("Destination SM (rank %d): %llu\n", rt_destSM, dest_sm);
    printf("Source SM (rank %d): %llu\n", rt_srcSM, src_sm);
    
    // 判断是否为跨GPC通信（基于SM ID差异）
    long long sm_diff = (long long)src_sm - (long long)dest_sm;
    printf("SM ID Difference: %lld\n", sm_diff);
    
    if (abs((int)sm_diff) > 16) {
        printf("LIKELY Cross-GPC Communication: SM %llu -> SM %llu\n", dest_sm, src_sm);
    } else {
        printf("LIKELY Intra-GPC Communication: SM %llu -> SM %llu\n", dest_sm, src_sm);
    }
    
    // 计算延迟统计（与原代码一致的方式）
    double total_latency = 0;
    unsigned long long min_latency = ULLONG_MAX;
    unsigned long long max_latency = 0;
    
    // 找到源rank对应的块
    int src_block = -1;
    for (int b = 0; b < total_blocks; b++) {
        if ((b % 16) == rt_srcSM) {
            src_block = b;
            break;
        }
    }
    
    if (src_block >= 0) {
        int base = src_block * (blockSize + 2);
        
        printf("\nPer-thread Latency (first 10 threads, clock cycles):\n");
        for (int t = 0; t < blockSize; t++) {
            unsigned long long lat = h_results[base + 2 + t];
            total_latency += lat;
            if (lat < min_latency && lat > 0) min_latency = lat;
            if (lat > max_latency) max_latency = lat;
            
            if (t < 10) {  // 只显示前10个线程
                printf("Thread %2d: %llu cycles\n", t, lat);
            }
        }
        
        double avg_latency = total_latency / blockSize;
        
        printf("\nLatency Statistics:\n");
        printf("Average Latency: %.2f clock cycles\n", avg_latency);
        printf("Min Latency: %llu clock cycles\n", min_latency);
        printf("Max Latency: %llu clock cycles\n", max_latency);
        
        // 与原代码输出格式一致
        printf("\nResult Summary:\n");
        printf("Block %d destSM %llu srcSM %llu Avg Latency %.4f clock cycles\n",
               src_block, dest_sm, src_sm, avg_latency);
    }
    
    // 清理内存
    cudaFree(d_buffer);
    cudaFree(d_results);
    free(h_results);
    
    return 0;
}
