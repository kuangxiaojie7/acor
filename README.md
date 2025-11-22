# ACOR：自适应协同组织与鲁棒多智能体强化学习框架

本仓库实现了ACOR（Adaptive Collaborative Organization for Robust MARL）框架，融合模糊 Top-K 分组、基于行为的信任评估、动态领导者选举与层级共识，可在 PettingZoo 的 MPE `simple_spread_v3` 场景下运行，并支持 PyTorch DDP 多 GPU 加速。

---

## 核心组件
- `ACORPolicy`：负责分组、信任建模、领导者选举和层级信息传播的策略网络。
- `ACORTrainer`：PPO 风格训练器，包含信任记忆更新、AMP 混合精度、自动保存检查点等机制。
- `MAPPOTrainer`：参数共享的 MAPPO 基线，用于快速与 ACOR 做性能对比。
- `VectorMPE`：对 PettingZoo MPE 并行环境的高性能封装。
- `configs/*.yaml`：集中管理实验参数，可通过命令行覆盖。

---

## 如何成功运行

1. **准备 Python 与 CUDA**  
   - 建议 Python ≥ 3.10。  
   - 若使用 GPU，请先根据服务器 CUDA 版本安装对应 PyTorch，例如：  
     ```bash
     pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

2. **创建虚拟环境（推荐）**  
   ```bash
   python -m venv acor-venv
   source acor-venv/bin/activate        # Linux/macOS
   # 或
   acor-venv\Scripts\activate           # Windows
   ```

3. **安装项目依赖**  
   ```bash
   pip install -r requirements.txt
   ```
   若提示某些包冲突，请根据提示调整 PyTorch/CUDA 版本。

4. **单卡运行 ACOR**  
   ```bash
   python -m scripts.train_acor \
       --config configs/acor_spread.yaml \
       --output_dir runs \
       --run_name acor_single
   ```
   ```bash
   # 单行命令，1卡运行 ACOR
   python -m scripts.train_acor --config configs/acor_spread.yaml --output_dir runs --run_name acor_1gpu
   ```

5. **�������� MAPPO ����**  
   ```bash
   python -m scripts.train_mappo \\
       --config configs/mappo_spread.yaml \\
       --output_dir runs \\
       --run_name mappo_single
   ```
   ���� MAPPO-Light ��汾��
   ```bash
   python -m scripts.train_mappo_light \\
       --config configs/mappo_light.yaml \\
       --output_dir runs \\
       --run_name mappo_light_single
   ```

6. **多卡（DDP）运行**  
   将 `--nproc_per_node` 设为 GPU 数量，例如 4 卡：  
   ```bash
   torchrun --nproc_per_node=4 -m scripts.train_acor \
       --config configs/acor_spread.yaml \
       --output_dir runs \
       --run_name acor_ddp
   ```
   ```bash
   # 单行命令，4卡运行 ACOR
   torchrun --nproc_per_node=4 -m scripts.train_acor --config configs/acor_spread.yaml --output_dir runs --run_name acor_4gpu
   ```

   MAPPO 同理：  
   ```bash
   torchrun --nproc_per_node=4 -m scripts.train_mappo \
       --config configs/mappo_spread.yaml \
       --output_dir runs \
       --run_name mappo_ddp
   ```

7. **断点续训（Resume）**
   - 在 `configs/acor_spread.yaml` 的 `train` 段落中加入 `resume_from` 指向 checkpoint 路径  
     ```yaml
     train:
       resume_from: runs/acor_ddp/checkpoint_000500.pt
     ```
   - 也可以在命令行覆盖：  
     ```bash
     torchrun --nproc_per_node=4 -m scripts.train_acor \
         --config configs/acor_spread.yaml \
         --output_dir runs \
         --run_name acor_ddp \
         --train.resume_from=runs/acor_ddp/checkpoint_000500.pt
     ```

8. **确认运行成功**
   - �ն˳������ JSON ��־������ `global_step`��`episode_return` ���ֶΣ���  
   - `runs/<run_name>/metrics.jsonl` �ļ���������  
   - �������� `checkpoint_*.pt` �����ļ���
---

## 与基线对比及结果图绘制

1. **收集日志**  
   - ACOR 输出：`runs/acor_ddp/metrics.jsonl`  
   - MAPPO 输出：`runs/mappo_ddp/metrics.jsonl`

2. **读取并绘制学习曲线（示例）**  
   ```python
   import json
   import matplotlib.pyplot as plt

   def load_metrics(path):
       steps, returns = [], []
       with open(path, "r", encoding="utf-8") as f:
           for line in f:
               row = json.loads(line)
               steps.append(row.get("global_step", 0))
               returns.append(row.get("episode_return", 0.0))
       return steps, returns

   steps_acor, ret_acor = load_metrics("runs/acor_ddp/metrics.jsonl")
   steps_mappo, ret_mappo = load_metrics("runs/mappo_ddp/metrics.jsonl")

   plt.plot(steps_acor, ret_acor, label="ACOR")
   plt.plot(steps_mappo, ret_mappo, label="MAPPO")
   plt.xlabel("Global Step")
   plt.ylabel("Episode Return")
   plt.legend()
   plt.title("ACOR vs. MAPPO on MPE simple_spread_v3")
   plt.tight_layout()
   plt.savefig("comparison.png", dpi=200)
   plt.show()
   ```
   将脚本保存后运行，即可得到 ACOR 与 MAPPO 的性能对比图。如果需要查看其他指标（如 `policy_loss`、`fps`、`value_loss`），可类似读取对应键值。

3. **加载检查点继续训练或评估**  
   ```python
   import torch
   ckpt = torch.load("runs/acor_ddp/checkpoint_002000.pt", map_location="cpu")
   model_state = ckpt["model"]
   ```
   将 `model_state` 加载进 `ACORPolicy` 即可复现实验或进行推理。

---

## 扩展建议
- 根据需求修改 `configs/*.yaml`：可调整环境、并行环境数、模型宽度、学习率等超参数。
- 若要加入更多 SOTA 基线（如 IPPO、HAPPO、FACMAC 等），可参考 `MAPPOTrainer` 增加新的训练脚本，并复用统一的日志/绘图流程。
- 如需迁移至其他多智能体环境，可在 `acor/envs` 目录新增封装，实现相同的接口即可。

完成上述步骤后，即可顺利运行 ACOR，并生成与基线的性能对比图。祝实验顺利！







