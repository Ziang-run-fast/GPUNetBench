# plot_latency_sorted.py
import re
import sys
import matplotlib.pyplot as plt

def parse_latencies(path: str):
    # 兼容任意空白：… Avg 302.00 cycles/group
    pat = re.compile(r"Avg\s+([0-9]+(?:\.[0-9]+)?)\s+cycles/group", re.IGNORECASE)
    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                vals.append(float(m.group(1)))
    return vals

def main():
    log = sys.argv[1] if len(sys.argv) > 1 else "result_SM2SM.log"
    vals = parse_latencies(log)
    if not vals:
        print(f"[WARN] no latencies found in {log}")
        return

    vals.sort()  # 从小到大排序
    x = list(range(1, len(vals) + 1))  # 排名/序号

    # 画散点图
    plt.figure(figsize=(8, 5))
    plt.scatter(x, vals)  # 默认配色，不额外指定颜色
    plt.title("Sorted Latencies (cycles per group)")
    plt.xlabel("Sample Rank (ascending)")
    plt.ylabel("Latency (cycles/group)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = "latency_sorted_scatter.png"
    plt.savefig(out, dpi=300)
    print(f"[INFO] parsed {len(vals)} samples, saved figure -> {out}")

if __name__ == "__main__":
    main()
