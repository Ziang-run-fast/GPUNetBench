#include <stdio.h>
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned get_smid() {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__global__ void collect_smid(unsigned *sm_ids) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    sm_ids[tid] = get_smid();
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int total_threads = prop.multiProcessorCount * 32; // 每个SM至少一个warp，保证全覆盖

    unsigned *d_smid, *h_smid;
    h_smid = (unsigned*)malloc(total_threads * sizeof(unsigned));
    cudaMalloc(&d_smid, total_threads * sizeof(unsigned));

    collect_smid<<<total_threads/32, 32>>>(d_smid);
    cudaMemcpy(h_smid, d_smid, total_threads * sizeof(unsigned), cudaMemcpyDeviceToHost);

    // 去重并打印
    bool seen[2048] = {0}; // 假设SM ID不超过2048
    printf("Available SM IDs:\n");
    for (int i = 0; i < total_threads; ++i) {
        if (!seen[h_smid[i]]) {
            seen[h_smid[i]] = true;
            printf("%u ", h_smid[i]);
        }
    }
    printf("\n");

    free(h_smid);
    cudaFree(d_smid);
    return 0;
}
