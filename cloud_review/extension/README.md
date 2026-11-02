# WM-811K 开发集复现实验

本目录回答两个问题。在相同训练预算下，哪种方法能找回更多真实划痕，同时控制正常片总误报？在每批 1,000 张候选片只能检 200 张时，优先检哪些片能减少模拟损失？

**English quick start.** Open the [frozen training notebook](WM811K_05_Imbalance_Study.ipynb) in a hosted Colab GPU runtime. Save checkpoints to your own Drive if you need to resume. Run the decision and report commands below on the resulting 30 probability files. Training changes are compared against a fresh cross-entropy control on the same runtime. Threshold and inspection policies reuse the same scores. The development audit has been inspected before. The cost assumptions and queue batches are simulations.

## 1. 在托管 Colab 准备数据

打开[训练笔记本](WM811K_05_Imbalance_Study.ipynb)，选择免费可用的托管 GPU。笔记本使用 Python 3、NumPy、PyTorch、Git 和 KaggleHub，按公开版本 `qingyi/wm811k-wafer-map/versions/1` 获取 `LSWMD.pkl`，并将原仓库固定到 `aaa667c0421ece13e3a71fa616fb10287b4453ff`。免费 GPU 是否可用取决于当前配额。

希望跨会话续跑时，先在 Colab 增加一个设置单元并执行。原笔记本会检查下列目录是否存在。如果跳过设置，它会把结果写到临时盘 `/content/wm811k_training`，全部训练完成后提供私有证据 ZIP 下载。

```python
from google.colab import drive
from pathlib import Path
drive.mount("/content/drive")
Path("/content/drive/MyDrive/Colab Notebooks/WM811K_Extension_20261101").mkdir(
    parents=True, exist_ok=True
)
```

顺序运行笔记本单元即可完成训练。在运行完成前，可在日志中查看每轮所用时间。脚本先核对原始文件、开发集索引和 CNN 源文件的 SHA-256，再按字节偏移读取 54,355 张开发片并逐片核对图像哈希。两通道缓存采用 float32，约占 1.78 GB，保存在 Colab 临时盘。原始晶圆图可以留在云端数据源；Snowflake 的汇总复核只需要约定的结果表。类别占比应按本次分区的 `split_support.csv` 读取。

## 2. 复现五组训练

31,457 张训练片、11,307 张验证片和 11,591 张已查看的开发审计片按固定角色使用。同 lot 或图像哈希相同的晶圆处于同一连接组，跨角色核对 lot、连接组和图像哈希。网络只在训练片上更新权重。

每个种子（42、123、2026）均使用 medium CNN、批量 128、七轮 AdamW、学习率 0.001、权重衰减 0.0001。每轮抽取 31,457 次并更新 246 步。基线与各改进均在本次运行时重新训练。

| 训练臂 | 相对本轮基线的变化 |
| --- | --- |
| `cnn_ce` | 自然分布打乱，无权重交叉熵 |
| `cnn_weighted` | 仅改变损失，每类权重由训练计数的平方根倒数计算 |
| `cnn_sampled` | 仅改变抽样，同样的每类权重用于有放回抽取训练片 |
| `cnn_rot180` | 每次训练读取以 50% 概率将整张图旋转 180° |
| `cnn_focal` | 仅改用 focal loss，γ = 2，α = 1 |

原始图居中补成正方形，面积插值生成 64 × 64 的有效晶粒与失效晶粒覆盖图。旋转同时作用于这两个通道，保留晶圆边缘、缺陷与背景的相对位置。这个实验假设改变读图朝向后类别仍成立。如果生产线有固定缺口方向或使用设备坐标诊断方向性划痕，应先单独验证这个假设。验证和审计图使用原始朝向。

需要分次执行时，可用下列代码替换笔记本的训练单元。它复用前置单元已定义的 `repo` 和 `raw`。第一条只校验数据并生成缓存，第二条只训练一组，第三条接续其余组合。

```python
import hashlib, shutil, subprocess, sys
from pathlib import Path
ext = Path("/content/drive/MyDrive/Colab Notebooks/WM811K_Extension_20261101")
out = ext / "training" if ext.is_dir() else Path("/content/wm811k_training")
cmd = [sys.executable, "-u", "/content/train_imbalance.py",
       "--repo", str(repo), "--raw", str(raw), "--out", str(out),
       "--cache", "/content/wm811k_xcache", "--device", "cuda"]
subprocess.run(cmd + ["--prepare-only"], check=True)
subprocess.run(cmd + ["--arms", "cnn_ce", "--seeds", "42"], check=True)
subprocess.run(cmd, check=True)
```

每轮保存模型、优化器和抽样器状态。重新启动同一命令时，脚本核验输出和检查点，跳过已完成的模型。每个输出目录只交给一个进程写入。若 Python、PyTorch、NumPy、GPU 类型或协议签名改变，请使用新目录并成套重跑。训练完成的标志是 `run_status.json` 显示 15/15 且 `probabilities/manifest.json` 的 `complete` 为真。每个种子和训练臂各生成两张逐片概率表，带 `source_row`、真实类别及九类概率。

## 3. 冻结阈值与复检策略

笔记本自带的是训练阶段。继续评估时，把**本目录的扩展脚本**放到 Colab 的 `/content/wm811k_extension`。旧提交只提供冻结的数据索引、网络和历史概率。决策脚本依赖 NumPy 和 Matplotlib，汇总脚本只使用标准库。在一个 Colab Python 单元执行以下命令，`repo` 指向笔记本固定的仓库，`out` 指向同一次训练的输出。

```python
from pathlib import Path
import subprocess, sys
extension = Path("/content/wm811k_extension")
decision = out.parent / "decision_full_v1"
public = out.parent / "decision_public_v1"
subprocess.run([sys.executable, str(extension / "decision_study.py"),
                "--archive", str(repo / "results/probabilities"),
                "--new", str(out), "--out", str(decision)], check=True)
subprocess.run([sys.executable, str(extension / "render_evidence.py"),
                "--run", str(decision), "--out", str(public)], check=True)
```

这两步各自要求新的输出目录。只想复核已发布模型时，可以省略 `--new`，使用独立目录输出历史模型的比较。历史模式的结果不能作为五组训练实验的证据。

- **概率和模型选择**。程序按连接组对原验证集做确定性拆分。一部分对每个模型和种子拟合温度参数，另一部分选择阈值，并按三个种子的平均九类 Macro-F1 从五组 CNN 中选出下游模型。相同得分按预先声明的顺序选择，基线优先。所有选择写入 `frozen_validation_policy.json` 后才读取审计集概率进行评估。
- **划痕约束**。在选择分区逐个考察划痕阈值，与同一模型同一种子的默认分类比较。真实划痕命中数必须持平或增加，正常片被判为任意缺陷的数量和被判为划痕的数量都必须持平或减少。若无严格改善则保留原决定。在审计分区另报三个条件是否仍成立。
- **成本与容量**。模拟假设人工复检能检出缺陷，误送复检的正常片每张花费 5 美元。两种情景分别设漏检任一缺陷损失 1,000 美元，以及单独把漏检划痕损失提高到 3,000 美元。净收益等于各缺陷概率乘漏检损失的总和，减去正常概率乘 5 美元。以上均为假设参数。
- **两种容量指标**。验证上选择的容量阈值把当时的复检量控制在 20% 内，未来样本经过同一阈值时比例可能改变。实际队列模拟使用固定历史 RF71 入口，按 `1 − P(none) ≥ 0.5` 形成各方法共享的候选池，再比较随机、缺陷概率、预期净收益排序。每个满批固定分配 200 个名额，末批取该批人数的 20% 并向下取整。总体漏检成本也包含未通过入口的缺陷。

## 4. 读取结果与分享

先查看汇总目录内的 `SUMMARY.zh-CN.md` 或 `SUMMARY.md`，再解压查看 `per_class.csv.gz`、`confusion.csv.gz` 和 `scratch_guard.csv.gz`。阈值曲线、概率可靠性图与 `five_stages.csv.gz` 分别解释分类质量、校准、路由与检验容量。三个种子面对同一批晶圆，审计中的真实划痕仍然是同一组样本。独立生产效果需要新的、有标签的批次。

公开材料使用 `render_evidence.py` 生成的 `aggregate_evidence.zip` 或其解压后的双语摘要、汇总表、图和哈希清单。原始数据、缓存、检查点、逐片概率、复检名单及笔记本最后导出的训练证据 ZIP 保存在自己的存储中。原始/未标注数据分布与历史成本的深入排查是独立流程；`shift_followup.py` 的输入与阶段参数可由 `--help` 查看。现有数据可继续保留在原处，按需传递汇总证据。
