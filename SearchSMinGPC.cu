// discover_gpc_by_cluster.cu
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <set>
#include <queue>

namespace cg = cooperative_groups;

#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 16   // Hopper 常见 8/16，按需改
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

__device__ __forceinline__ unsigned get_smid() {
  unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid)); return smid;
}

// 把每个 block 的 SMID 记到按 cluster 编址的数组中
__global__ __cluster_dims__(CLUSTER_SIZE,1,1)
void dump_cluster_smids(int *smid_out) {
  cg::cluster_group cluster = cg::this_cluster();
  const int rank       = (int)cluster.block_rank();          // 0..CLUSTER_SIZE-1
  const int cluster_id = (int)blockIdx.x / CLUSTER_SIZE;     // 0..numClusters-1
  const int idx        = cluster_id * CLUSTER_SIZE + rank;
  if (threadIdx.x == 0) {
    smid_out[idx] = (int)get_smid();
  }
}

static void bfs_groups(const std::vector<std::vector<int>>& adj,
                       std::vector<int>& comp_id,
                       std::vector<std::vector<int>>& groups)
{
  const int N = (int)adj.size();
  comp_id.assign(N, -1);
  int cid = 0;
  for (int s=0; s<N; ++s) {
    if (adj[s].empty() || comp_id[s] != -1) continue;
    std::queue<int> q; q.push(s); comp_id[s] = cid;
    std::vector<int> comp{ s };
    while (!q.empty()) {
      int u = q.front(); q.pop();
      for (int v: adj[u]) if (comp_id[v] == -1) {
        comp_id[v] = cid; q.push(v); comp.push_back(v);
      }
    }
    std::sort(comp.begin(), comp.end());
    groups.push_back(std::move(comp));
    cid++;
  }
}

int main(int argc, char** argv) {
  int device = 0;
  cudaSetDevice(device);

  int numClusters = 32;  // 采样的 cluster 数，越大越容易覆盖全部 GPC/SM
  if (argc > 1) numClusters = atoi(argv[1]);

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  const int numSM = prop.multiProcessorCount;

  // 1) 启动 numClusters 个 cluster，每个 cluster 有 CLUSTER_SIZE 个 block
  const int totalBlocks = numClusters * CLUSTER_SIZE;

  // 设备端数组：每个 cluster 的每个 rank 记录一个 smid
  int *d_smids = nullptr;
  cudaMalloc(&d_smids, totalBlocks * sizeof(int));
  cudaMemset(d_smids, 0xFF, totalBlocks * sizeof(int)); // -1

  // 需要允许非可移植的 cluster 尺寸
  cudaFuncSetAttribute(dump_cluster_smids,
      cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

  dump_cluster_smids<<<totalBlocks, BLOCK_SIZE>>>(d_smids);
  cudaDeviceSynchronize();

  // 拷回主机
  std::vector<int> smids(totalBlocks, -1);
  cudaMemcpy(smids.data(), d_smids, totalBlocks * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_smids);

  // 2) 基于“同一 cluster 内出现过”的关系，构建无向图
  std::vector<std::vector<int>> adj(numSM);  // 0..numSM-1
  auto add_edge = [&](int a, int b){
    if (a<0 || b<0 || a>=numSM || b>=numSM || a==b) return;
    adj[a].push_back(b);
    adj[b].push_back(a);
  };

  for (int c = 0; c < numClusters; ++c) {
    // 收集该 cluster 的 SMID（去重）
    std::set<int> uniq;
    for (int r = 0; r < CLUSTER_SIZE; ++r) {
      int sm = smids[c*CLUSTER_SIZE + r];
      if (sm >= 0 && sm < numSM) uniq.insert(sm);
    }
    if (uniq.size() <= 1) continue;
    // 同一 cluster 内两两连边
    std::vector<int> v(uniq.begin(), uniq.end());
    for (size_t i=0;i<v.size();++i)
      for (size_t j=i+1;j<v.size();++j)
        add_edge(v[i], v[j]);
  }

  // 3) 做连通分量 → GPC 组
  std::vector<int> comp_id;
  std::vector<std::vector<int>> groups;
  bfs_groups(adj, comp_id, groups);

  // 输出结果
  printf("Detected %d SMs on device %s (cc %d.%d)\n",
         numSM, prop.name, prop.major, prop.minor);
  printf("Sampled %d clusters (cluster_size=%d)\n", numClusters, CLUSTER_SIZE);
  printf("== Inferred GPC groups (connected components) ==\n");
  // 按组大小排序打印
  std::sort(groups.begin(), groups.end(),
            [](const auto& a, const auto& b){ return a.size() > b.size(); });

  for (size_t gi=0; gi<groups.size(); ++gi) {
    printf("Group %zu (size=%zu): ", gi, groups[gi].size());
    for (size_t k=0;k<groups[gi].size();++k) {
      printf("%d%s", groups[gi][k], (k+1==groups[gi].size())? "" : " ");
    }
    printf("\n");
  }

  // 4) 为每个 SM 列出同 GPC 的 SM（便于后续直接查）
  printf("\n== Per-SM same-GPC lists ==\n");
  for (int sm=0; sm<numSM; ++sm) {
    if (comp_id[sm] < 0) {
      printf("SM %d: (unseen; increase numClusters)\n", sm);
      continue;
    }
    const auto& g = groups[ comp_id[sm] ];
    printf("SM %d: ", sm);
    for (int x: g) if (x != sm) printf("%d ", x);
    printf("\n");
  }

  return 0;
}
