import re
import matplotlib.pyplot as plt

log_file = "result_BW.log"

# 读取日志文件
with open(log_file, "r") as f:
    log_data = f.read()

# 正则匹配 blockSize 和 Bandwidth
pattern = r"Running blockSize = (\d+)\s+Cluster.*Bandwidth ([\d\.]+) GB/s"
matches = re.findall(pattern, log_data)

# 转换为数值
block_sizes = [int(m[0]) for m in matches]
bandwidths = [float(m[1]) for m in matches]

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(block_sizes, bandwidths, color='blue', label='Measured Bandwidth')
plt.plot(block_sizes, bandwidths, color='orange', linestyle='--', alpha=0.7, label='Trend')

plt.title("Bandwidth vs BlockSize")
plt.xlabel("BlockSize")
plt.ylabel("Bandwidth (GB/s)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# 保存图片
plt.savefig("bandwidth_vs_blocksize.png", dpi=300)  # 高分辨率PNG
# plt.savefig("bandwidth_vs_blocksize.pdf")  # 如果要保存为PDF

plt.show()
