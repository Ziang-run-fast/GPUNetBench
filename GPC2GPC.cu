#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

// 编译时配置
#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 16
#endif

#ifndef ILP_FACTOR
#define ILP_FACTOR 8
#endif

#ifndef ITERATION
#define ITERATION 10000
#endif

#ifndef STRIDE
#define STRIDE 1
#endif

// 选择测量模式
#if !defined(CALC_BW) && !defined(CALC_LATENCY)
#define CALC_BW
#endif

namespace cg = cooperative_groups;

// 获取SM ID的内联PTX
__device__ inline unsigned int get_smid() {
    unsigned int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// 跨GPC通信结构体
struct CrossGPCData {
    int* global_buffer;        // 全局内存缓冲区
    unsigned int* sm_mapping;  // SM到块的映射
    int* sync_flags;           // 同步标志
    unsigned long long* results; // 结果存储
};

/*
 * 原始的单GPC内核（保持不变，用于对比）
 */
__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1)
void single_gpc_kernel(unsigned long long *out,
                       int num_ints,
                       int rt_destSM,
                       int rt_srcSM) {
    extern __shared__ int sdata[];
    
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int rank = cluster.block_rank();
    int cluster_id = blockIdx.x / CLUSTER_SIZE;

    // === 预热 & 记录destSM ===
    if (rank == rt_destSM && threadIdx.x == 0) {
        unsigned int my_smid = get_smid();
#ifdef CALC_BW
        out[3 * cluster_id + 0] = my_smid;
#else
        out[cluster_id * (blockDim.x + 2) + 0] = my_smid;
#endif
        sdata[0] = my_smid;
        for (int i = 1; i < num_ints; i++)
            sdata[i] = my_smid + i;
    }

    int * __restrict__ ws = cluster.map_shared_rank(sdata, rt_destSM);
    int local_sum = 0;

#ifdef CALC_LATENCY
    unsigned long long lat_acc = 0, lat_cnt = 0;
#endif

    cluster.sync();

    // === 读取 & 记录srcSM + 计时 ===
    if (rank == rt_srcSM) {
#ifdef CALC_BW
        unsigned long long startCycles = clock64();
#endif

        for (int rep = 0; rep < ITERATION; rep++) {
            int idx = threadIdx.x * STRIDE;
            for (; idx + (ILP_FACTOR - 1) * blockDim.x * STRIDE < num_ints;
                  idx += blockDim.x * ILP_FACTOR * STRIDE) {
#ifdef CALC_LATENCY
                unsigned long long t0 = clock();
#endif
#pragma unroll
                for (int j = 0; j < ILP_FACTOR; j++)
                    local_sum += ws[idx + j * blockDim.x * STRIDE];
#ifdef CALC_LATENCY
                unsigned long long t1 = clock();
                lat_acc += (t1 - t0);
                lat_cnt++;
#endif
            }
            for (; idx < num_ints; idx += blockDim.x * STRIDE) {
#ifdef CALC_LATENCY
                unsigned long long t0 = clock();
#endif
                local_sum += ws[idx];
#ifdef CALC_LATENCY
                unsigned long long t1 = clock();
                lat_acc += (t1 - t0);
                lat_cnt++;
#endif
            }
        }

        __syncthreads();

#ifdef CALC_BW
        unsigned long long totalCycles = clock64() - startCycles;
        unsigned int my_smid = get_smid();
        out[3 * cluster_id + 1] = my_smid;
        out[3 * cluster_id + 2] = totalCycles;
#endif
#ifdef CALC_LATENCY
        unsigned long long avgLat = lat_cnt ? (lat_acc / lat_cnt) : 0;
        int base = cluster_id * (blockDim.x + 2);
        out[base + 1] = get_smid();
        out[base + 2 + threadIdx.x] = avgLat;
#endif
    }

    sdata[threadIdx.x] = local_sum;
    cluster.sync();
}

/*
 * 新的跨GPC通信内核
 */
__global__ void cross_gpc_kernel(CrossGPCData comm_data,
                                 int num_ints,
                                 int target_gpc,
                                 int comm_mode) {
    extern __shared__ int sdata[];
    
    unsigned int my_smid = get_smid();
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    
    // 确定当前块的角色：发送方(0)还是接收方(1)
    int my_role = (block_id % 2 == 0) ? 0 : 1;  // 偶数块=发送方，奇数块=接收方
    
    // 记录SM映射
    if (thread_id == 0) {
        comm_data.sm_mapping[block_id] = my_smid;
    }
    
    __syncthreads();
    __threadfence();  // 确保SM映射对所有块可见

    int local_sum = 0;
    
#ifdef CALC_LATENCY
    unsigned long long lat_acc = 0, lat_cnt = 0;
#endif

    if (my_role == 1) {  // 接收方：准备数据
        if (thread_id == 0) {
            // 在全局内存中准备数据
            int base_addr = block_id * num_ints;
            for (int i = 0; i < num_ints; i++) {
                comm_data.global_buffer[base_addr + i] = my_smid + i;
            }
            
            // 设置数据就绪标志
            comm_data.sync_flags[block_id] = 1;
        }
        
        // 在共享内存中也准备一份数据（用于对比）
        if (thread_id < num_ints) {
            sdata[thread_id] = my_smid + thread_id;
        }
    }
    
    __syncthreads();
    __threadfence();  // 确保数据对所有块可见

    if (my_role == 0) {  // 发送方：查找目标并读取数据
        // 寻找目标接收方块（简化：选择下一个奇数块）
        int target_block = (block_id % 2 == 0) ? block_id + 1 : block_id - 1;
        if (target_block >= gridDim.x) target_block = 1;  // 边界处理
        
        // 等待目标块准备就绪
        if (thread_id == 0) {
            while (atomicAdd(&comm_data.sync_flags[target_block], 0) == 0) {
                // 自旋等待
            }
        }
        __syncthreads();

#ifdef CALC_BW
        unsigned long long startCycles = clock64();
#endif

        // 开始通信测试
        for (int rep = 0; rep < ITERATION; rep++) {
            int base_addr = target_block * num_ints;
            int idx = thread_id * STRIDE;
            
            // ILP展开的主循环
            for (; idx + (ILP_FACTOR - 1) * blockDim.x * STRIDE < num_ints;
                  idx += blockDim.x * ILP_FACTOR * STRIDE) {
#ifdef CALC_LATENCY
                unsigned long long t0 = clock();
#endif
#pragma unroll
                for (int j = 0; j < ILP_FACTOR; j++) {
                    local_sum += comm_data.global_buffer[base_addr + idx + j * blockDim.x * STRIDE];
                }
#ifdef CALC_LATENCY
                unsigned long long t1 = clock();
                lat_acc += (t1 - t0);
                lat_cnt++;
#endif
            }
            
            // 处理剩余元素
            for (; idx < num_ints; idx += blockDim.x * STRIDE) {
#ifdef CALC_LATENCY
                unsigned long long t0 = clock();
#endif
                local_sum += comm_data.global_buffer[base_addr + idx];
#ifdef CALC_LATENCY
                unsigned long long t1 = clock();
                lat_acc += (t1 - t0);
                lat_cnt++;
#endif
            }
        }

        __syncthreads();

        // 记录结果
        if (thread_id == 0) {
#ifdef CALC_BW
            unsigned long long totalCycles = clock64() - startCycles;
            unsigned long long bytes = (unsigned long long)num_ints * sizeof(int) * ITERATION;
            
            // 存储结果：[srcSM, destSM, cycles, bytes]
            int result_idx = block_id * 4;
            comm_data.results[result_idx + 0] = my_smid;
            comm_data.results[result_idx + 1] = comm_data.sm_mapping[target_block];
            comm_data.results[result_idx + 2] = totalCycles;
            comm_data.results[result_idx + 3] = bytes;
#endif

#ifdef CALC_LATENCY
            unsigned long long avgLat = lat_cnt ? (lat_acc / lat_cnt) : 0;
            int result_idx = block_id * 3;
            comm_data.results[result_idx + 0] = my_smid;
            comm_data.results[result_idx + 1] = comm_data.sm_mapping[target_block];
            comm_data.results[result_idx + 2] = avgLat;
#endif
        }
    }
    
    // 防止编译器优化
    if (thread_id < num_ints) {
        sdata[thread_id] = local_sum;
    }
}

int main(int argc, char **argv) {
    // 解析参数
    int communication_mode = 0;  // 0=单GPC, 1=跨GPC
    int numClusters = 1;
    int blockSize = 1024;
    int rt_destSM = 1;
    int rt_srcSM = 0;

    if (argc > 1) communication_mode = atoi(argv[1]);
    if (argc > 2) numClusters = atoi(argv[2]);
    if (argc > 3) blockSize = atoi(argv[3]);
    if (argc > 4) rt_destSM = atoi(argv[4]);
    if (argc > 5) rt_srcSM = atoi(argv[5]);

    // 设备设置
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    printf("Running %s communication test\n", 
           communication_mode == 0 ? "Single-GPC" : "Cross-GPC");
    printf("GPU: %s (SM Count: %d)\n", prop.name, prop.multiProcessorCount);

    int maxSMemOptin;
    cudaDeviceGetAttribute(&maxSMemOptin,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    size_t shared_bytes = maxSMemOptin;
    int num_ints = shared_bytes / sizeof(int);

    if (communication_mode == 0) {
        // === 单GPC通信测试（原始代码）===
        int total_blocks = numClusters * CLUSTER_SIZE;
        size_t out_size;
        
#ifdef CALC_BW
        out_size = numClusters * 3 * sizeof(unsigned long long);
#else
        out_size = numClusters * (blockSize + 2) * sizeof(unsigned long long);
#endif

        unsigned long long *d_out, *h_out;
        h_out = (unsigned long long*)malloc(out_size);
        cudaMalloc(&d_out, out_size);

        // 设置内核属性
        cudaFuncSetAttribute(single_gpc_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        cudaFuncSetAttribute(single_gpc_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
        cudaFuncSetAttribute(single_gpc_kernel,
            cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

        // 启动内核
        single_gpc_kernel<<<total_blocks, blockSize, shared_bytes>>>(
            d_out, num_ints, rt_destSM, rt_srcSM);
        cudaDeviceSynchronize();

        // 复制结果
        cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost);

        double clkHz = prop.clockRate * 1e3;

#ifdef CALC_BW
        for (int i = 0; i < numClusters; i++) {
            unsigned long long destSM = h_out[3*i + 0];
            unsigned long long srcSM  = h_out[3*i + 1];
            unsigned long long cycles = h_out[3*i + 2];
            unsigned long long bytes  = (unsigned long long)num_ints * sizeof(int) * ITERATION;
            double bpc = (double)bytes / (double)cycles;
            double bw  = bpc * clkHz / 1e9;
            printf("Single-GPC: Cluster %d destSM %llu srcSM %llu Bandwidth %.4f GB/s\n",
                   i, destSM, srcSM, bw);
        }
#else
        for (int i = 0; i < numClusters; i++) {
            double average = 0;
            int base = i * (blockSize + 2);
            unsigned long long destSM = h_out[base + 0];
            unsigned long long srcSM  = h_out[base + 1];
            for (int j = 0; j < blockSize; j++) {
                unsigned long long avgLat = h_out[base + 2 + j];
                average += avgLat;
            }
            average /= blockSize;
            printf("Single-GPC: Cluster %d destSM %llu srcSM %llu Avg Latency %.4f cycles\n",
                   i, destSM, srcSM, average);
        }
#endif

        cudaFree(d_out);
        free(h_out);

    } else {
        // === 跨GPC通信测试 ===
        
        // 使用更多块来增加跨GPC通信的可能性
        int total_blocks = prop.multiProcessorCount / 2;  // 使用一半的SM
        if (total_blocks < 32) total_blocks = 32;  // 最少32个块
        
        printf("Launching %d blocks for cross-GPC communication\n", total_blocks);

        // 分配通信数据结构
        CrossGPCData h_comm_data, d_comm_data;
        
        size_t buffer_size = total_blocks * num_ints * sizeof(int);
        size_t mapping_size = total_blocks * sizeof(unsigned int);
        size_t flags_size = total_blocks * sizeof(int);
        
#ifdef CALC_BW
        size_t results_size = total_blocks * 4 * sizeof(unsigned long long);
#else
        size_t results_size = total_blocks * 3 * sizeof(unsigned long long);
#endif

        // 分配设备内存
        cudaMalloc(&d_comm_data.global_buffer, buffer_size);
        cudaMalloc(&d_comm_data.sm_mapping, mapping_size);
        cudaMalloc(&d_comm_data.sync_flags, flags_size);
        cudaMalloc(&d_comm_data.results, results_size);

        // 初始化
        cudaMemset(d_comm_data.global_buffer, 0, buffer_size);
        cudaMemset(d_comm_data.sm_mapping, 0, mapping_size);
        cudaMemset(d_comm_data.sync_flags, 0, flags_size);
        cudaMemset(d_comm_data.results, 0, results_size);

        // 分配主机内存用于结果
        unsigned long long* h_results = (unsigned long long*)malloc(results_size);
        unsigned int* h_mapping = (unsigned int*)malloc(mapping_size);

        // 启动跨GPC内核
        cross_gpc_kernel<<<total_blocks, blockSize, shared_bytes>>>(
            d_comm_data, num_ints, 1, communication_mode);
        cudaDeviceSynchronize();

        // 复制结果
        cudaMemcpy(h_results, d_comm_data.results, results_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_mapping, d_comm_data.sm_mapping, mapping_size, cudaMemcpyDeviceToHost);

        double clkHz = prop.clockRate * 1e3;

        // 分析结果
        printf("\nCross-GPC Communication Results:\n");
        printf("Block -> SM Mapping:\n");
        for (int i = 0; i < total_blocks && i < 20; i++) {  // 只显示前20个
            printf("Block %d -> SM %u\n", i, h_mapping[i]);
        }

        printf("\nCommunication Performance:\n");
        int valid_results = 0;
        
#ifdef CALC_BW
        for (int i = 0; i < total_blocks; i += 2) {  // 只有偶数块(发送方)有结果
            unsigned long long srcSM = h_results[i * 4 + 0];
            unsigned long long destSM = h_results[i * 4 + 1];
            unsigned long long cycles = h_results[i * 4 + 2];
            unsigned long long bytes = h_results[i * 4 + 3];
            
            if (cycles > 0 && bytes > 0) {
                double bpc = (double)bytes / (double)cycles;
                double bw = bpc * clkHz / 1e9;
                printf("Cross-GPC: Block %d srcSM %llu -> destSM %llu Bandwidth %.4f GB/s\n",
                       i, srcSM, destSM, bw);
                valid_results++;
            }
        }
#else
        for (int i = 0; i < total_blocks; i += 2) {
            unsigned long long srcSM = h_results[i * 3 + 0];
            unsigned long long destSM = h_results[i * 3 + 1];
            unsigned long long avgLat = h_results[i * 3 + 2];
            
            if (avgLat > 0) {
                printf("Cross-GPC: Block %d srcSM %llu -> destSM %llu Avg Latency %llu cycles\n",
                       i, srcSM, destSM, avgLat);
                valid_results++;
            }
        }
#endif

        printf("Total valid cross-GPC communication pairs: %d\n", valid_results);

        // 清理内存
        cudaFree(d_comm_data.global_buffer);
        cudaFree(d_comm_data.sm_mapping);
        cudaFree(d_comm_data.sync_flags);
        cudaFree(d_comm_data.results);
        free(h_results);
        free(h_mapping);
    }

    return 0;
}
