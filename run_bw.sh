#!/bin/bash
set -euo pipefail

# 可根据需要改
TOTAL_SM=132
NUM_CLUSTERS=16       # 提高同 cluster 命中率（可调大）
BLOCK_SIZE=256        # 或 512/1024
BIN=./SM2SM           # 你的可执行文件（DSMEM探测版）

OUT_CSV="same_gpc_pairs.csv"
OUT_LOG="scan_same_gpc.log"
> "$OUT_CSV"
> "$OUT_LOG"

echo "dstSM,srcSM" >> "$OUT_CSV"

for ((dst=0; dst<TOTAL_SM; dst++)); do
  echo "=== Scanning dstSM=${dst} ===" | tee -a "$OUT_LOG"
  same_list=()
  for ((src=0; src<TOTAL_SM; src++)); do
    if [ $src -eq $dst ]; then continue; fi
    # 运行一次探测（程序会多次 attempt）
    line=$("$BIN" "$dst" "$src" "$NUM_CLUSTERS" "$BLOCK_SIZE")
    echo "$line" >> "$OUT_LOG"

    # 判断是否 SAME_GPC
    if echo "$line" | grep -q "SAME_GPC"; then
      same_list+=("$src")
      echo "${dst},${src}" >> "$OUT_CSV"
    fi
  done

  # 友好打印该 dstSM 的同 GPC 列表
  if [ ${#same_list[@]} -gt 0 ]; then
    echo "dstSM ${dst} SAME_GPC with: ${same_list[*]}" | tee -a "$OUT_LOG"
  else
    echo "dstSM ${dst} SAME_GPC with: (none)" | tee -a "$OUT_LOG"
  fi
done

echo "Done. Results:"
echo "  - raw log    -> $OUT_LOG"
echo "  - same pairs -> $OUT_CSV"
