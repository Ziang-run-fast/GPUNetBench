#!/bin/bash

# 清空旧的日志文件
> result_BW.log

# 循环 blockSize = 32, 64, ..., 1024
for blockSize in $(seq 32 32 1024)
do
    echo "Running blockSize = $blockSize" | tee -a result_BW.log
    ./SM2SM 1 0 1 $blockSize | tee -a result_BW.log
    echo "" >> result_BW.log
done
