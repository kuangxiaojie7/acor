import json
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(path):
    # ... (这部分函数不变) ...
    steps, returns = [], []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("global_step") is not None and row.get("episode_return") is not None:
                    steps.append(row.get("global_step"))
                    returns.append(row.get("episode_return"))
    except FileNotFoundError:
        print(f"错误：找不到文件 {path}")
        return [], []
    return steps, returns

def smooth(y, window_size=50):
    # ... (这部分函数不变) ...
    if len(y) < window_size:
        return y
    window = np.ones(window_size) / window_size
    y_smooth = np.convolve(y, window, mode='valid')
    return y_smooth

# --- 加载数据 ---
steps_acor, ret_acor = load_metrics("runs/acor_2000_2gpu/metrics.jsonl")
# 下面这行不再需要，注释掉
steps_mappo, ret_mappo = load_metrics("runs/mappo_2000_2gpu/metrics.jsonl")
steps_mappo_light, ret_mappo_light = load_metrics("runs/mappo_light_2000_2gpu/metrics.jsonl")

# --- 绘制图形 ---
plt.figure(figsize=(10, 6))

ret_acor_smooth = smooth(ret_acor)
# 下面这行不再需要，注释掉
ret_mappo_smooth = smooth(ret_mappo)
ret_mappo_light_smooth = smooth(ret_mappo_light)

plt.plot(steps_acor[:len(ret_acor_smooth)], ret_acor_smooth, label="ACOR")
# 下面这行不再需要，注释掉
plt.plot(steps_mappo[:len(ret_mappo_smooth)], ret_mappo_smooth, label="MAPPO")
plt.plot(steps_mappo_light[:len(ret_mappo_light_smooth)], ret_mappo_light_smooth, label="MAPPO_LIGHT")


plt.xlabel("Global Step")
plt.ylabel("Smoothed Episode Return")
plt.legend()

plt.title("ACOR Performance on MPE simple_spread_v3")
plt.grid(True)
plt.tight_layout()
plt.savefig("acor_performance.png", dpi=300)
print("ACOR 性能图已保存为 acor_performance.png")