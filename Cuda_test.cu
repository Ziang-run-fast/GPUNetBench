#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA错误: " << cudaGetErrorString(error) \
                  << " 在文件 " << __FILE__ << " 第 " << __LINE__ << " 行" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

void printDeviceInfo(int deviceId) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    
    std::cout << "\n========== GPU设备 " << deviceId << " 信息 ==========\n";
    std::cout << "设备名称: " << prop.name << std::endl;
    std::cout << "计算能力: " << prop.major << "." << prop.minor << std::endl;
    
    std::cout << "\n--- 线程和Block限制 ---\n";
    std::cout << "每个block最大线程数: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "每个SM最大线程数: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个SM最大block数: " << prop.maxBlocksPerMultiProcessor << std::endl;
    
    std::cout << "\n--- Block维度限制 ---\n";
    std::cout << "Block最大维度 (x,y,z): (" 
              << prop.maxThreadsDim[0] << ", " 
              << prop.maxThreadsDim[1] << ", " 
              << prop.maxThreadsDim[2] << ")" << std::endl;
    
    std::cout << "\n--- Grid维度限制 ---\n";
    std::cout << "Grid最大维度 (x,y,z): (" 
              << prop.maxGridSize[0] << ", " 
              << prop.maxGridSize[1] << ", " 
              << prop.maxGridSize[2] << ")" << std::endl;
    
    std::cout << "\n--- 硬件资源 ---\n";
    std::cout << "SM数量: " << prop.multiProcessorCount << std::endl;
    std::cout << "全局内存: " << std::fixed << std::setprecision(1) 
              << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    std::cout << "L2缓存大小: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
    std::cout << "每个block共享内存: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "每个SM共享内存: " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
    std::cout << "每个block寄存器数: " << prop.regsPerBlock << std::endl;
    std::cout << "每个SM寄存器数: " << prop.regsPerMultiprocessor << std::endl;
    
    std::cout << "\n--- 性能特性 ---\n";
    std::cout << "Warp大小: " << prop.warpSize << std::endl;
    std::cout << "内存时钟频率: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "内存总线宽度: " << prop.memoryBusWidth << " bits" << std::endl;
    
    // 计算理论带宽
    float memBandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    std::cout << "理论内存带宽: " << std::fixed << std::setprecision(1) 
              << memBandwidth << " GB/s" << std::endl;
    
    std::cout << "\n--- 支持特性 ---\n";
    std::cout << "支持统一内存: " << (prop.managedMemory ? "是" : "否") << std::endl;
    std::cout << "支持并发内核: " << (prop.concurrentKernels ? "是" : "否") << std::endl;
    std::cout << "支持异步传输: " << (prop.asyncEngineCount > 0 ? "是" : "否") << std::endl;
    
    std::cout << "\n--- 占用率分析 ---\n";
    // 分析不同block大小的理论占用率
    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    std::cout << "Block大小\t每SM Block数\t占用率\n";
    std::cout << "--------------------------------\n";
    
    for (int blockSize : blockSizes) {
        if (blockSize <= prop.maxThreadsPerBlock) {
            int blocksPerSM = prop.maxThreadsPerMultiProcessor / blockSize;
            if (blocksPerSM > prop.maxBlocksPerMultiProcessor) {
                blocksPerSM = prop.maxBlocksPerMultiProcessor;
            }
            
            float occupancy = (float)(blocksPerSM * blockSize) / prop.maxThreadsPerMultiProcessor * 100;
            std::cout << blockSize << "\t\t" << blocksPerSM << "\t\t" 
                      << std::fixed << std::setprecision(1) << occupancy << "%\n";
        }
    }
}

// 简单的测试kernel
__global__ void testKernel() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // 简单的计算，避免被优化掉
    if (tid == 0) {
        printf("测试kernel在GPU上成功运行！\n");
    }
}

int main() {
    std::cout << "CUDA设备信息查询程序\n";
    std::cout << "==================\n";
    
    // 检查CUDA支持
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cerr << "错误: 未找到支持CUDA的设备！" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "检测到 " << deviceCount << " 个CUDA设备\n";
    
    // 打印所有设备的信息
    for (int i = 0; i < deviceCount; i++) {
        printDeviceInfo(i);
    }
    
    // 运行一个简单的测试kernel
    std::cout << "\n========== 运行测试 ==========\n";
    std::cout << "运行简单的测试kernel...\n";
    
    // 设置使用第一个设备
    CUDA_CHECK(cudaSetDevice(0));
    
    // 启动测试kernel
    dim3 blockSize(256);
    dim3 gridSize(1);
    
    testKernel<<<gridSize, blockSize>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "所有测试完成！\n";
    
    return 0;
}
