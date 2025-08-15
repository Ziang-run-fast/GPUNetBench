#!/bin/bash

# 固定 blockSize
BLOCKSIZE=1024

# 总 SM 数
TOTAL_SM=132

# 保存结果文件
LOGFILE="result_SM2SM.log"
> "$LOGFILE"

for ((dstSM=0; dstSM<$TOTAL_SM; dstSM++)); do
    for ((srcSM=0; srcSM<$TOTAL_SM; srcSM++)); do
        if [ $dstSM -ne $srcSM ]; then
            echo "Running dstSM=$dstSM srcSM=$srcSM" | tee -a "$LOGFILE"
            ./SM2SM $dstSM $srcSM 1 $BLOCKSIZE | tee -a "$LOGFILE"
        fi
    done
done
