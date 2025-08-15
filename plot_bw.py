# plot_latency_dist.py
import re
import sys
import statistics as stats
import matplotlib.pyplot as plt

def parse_latencies(log_path: str):
    """
    从日志中提取 'Avg xxx.xx cycles/group' 的延迟数值，返回 float 列表。
    兼容形如：
    'Inter-GPC L2  destSM 0  srcSM 1  Avg 302.00 cycles/group'
    """
    pattern = re.compile(r"Avg\s+([0-9]+(?:\.[0-9]+)?)\s+cycles/group")
    values = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                values.append(float(m.group(1)))
    return values

def main():
    log_file = "result_SM2SM.log" if len(sys.argv) < 2 else sys.argv[1]
    vals = parse_latencies(log_file)
    if not vals:
        print(f"[WARN] No latency values found in {log_file}")
        return

    # 打印一些统计
    print(f"[INFO] parsed {len(vals)} latencies from {log_file}")
    print(f"  min = {min(vals):.2f} cycles/group")
    print(f"  p50 = {stats.median(vals):.2f} cycles/group")
    try:
        print(f"  mean = {stats.mean(vals):.2f} cycles/group")
        print(f"  stdev = {stats.pstdev(vals):.2f} cycles/group")
    except stats.StatisticsError:
        pass
    print(f"  max = {max(vals):.2f} cycles/group")

    # 画直方图（分布图）
    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins='auto', edgecolor='black')
    plt.title("Latency Distribution (cycles per group)")
    plt.xlabel("Latency (cycles/group)")
    plt.ylabel("Count")
    plt.tight_layout()

    out_png = "latency_distribution.png"
    plt.savefig(out_png, dpi=300)
    print(f"[INFO] saved figure -> {out_png}")

    # 如需在本地弹窗显示，取消下一行注释
    # plt.show()

if __name__ == "__main__":
    main()
