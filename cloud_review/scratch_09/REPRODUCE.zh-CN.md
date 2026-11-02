# 复现开发集划痕实验

## 数据与运行环境

本轮使用托管 Colab 的单张 T4。取 WM-811K 的 Kaggle 版本 `qingyi/wm811k-wafer-map/versions/1` 中的 `LSWMD.pkl`。用本目录的程序配合如下固定版本的历史依赖运行。把各变量替换为当前云端会话路径。

```bash
git clone https://github.com/Matthewsyr/reliable-wafer-classification.git /content/frozen-wafer
git -C /content/frozen-wafer checkout --detach 985c3117a20ea2abdfa9812f5a46d873cd089a47
REPO=/content/frozen-wafer
RAW=/path/to/LSWMD.pkl
RUN=/content/wafer09
mkdir -p "$RUN"
```

冻结版本提供双通道栅格化、CNN、开发集分组和 RF77 已存概率。脚本校验文件 SHA-256。数据加载器只解码开发集索引指定的晶圆。训练、阈值设计、选择和历史开发审计分别有 31,457、5,637、5,670 和 11,591 张有标签晶圆。批次和相同图像关联的组件保持在同一分区。

## 比较三种训练方法

```bash
python -u train09.py --repo "$REPO" --raw "$RAW" --cache "$RUN/cache" --out "$RUN/training" \
  --arms baseline real_repeat synthetic --seeds 42 123 2026
python -u pseudo09.py qualify --repo "$REPO" --out "$RUN/rf_gate"
```

三个训练臂使用相同的中型 CNN、自然抽样顺序、优化器、初始化和七轮训练。每轮均有 31,457 次样本曝光和 246 次梯度更新。A 使用全部原始标签。B 在 80 个正常片位置放入重复抽取的真实划痕。C 替换完全相同的位置，将训练划痕中真实存在的一段失效 die 平移到尺寸相近的真实训练正常片上。训练正常片在这里指原标签为正常，图中仍可含散布的失效 die。合成只改变晶圆内原本正常的 die，保留原有失效和边界。供体不能满足几何要求时会重新抽取。因此 C 对 B 衡量整套合成方法在同等曝光量下的增量。

## 按尺寸核验教师

RF77 固定为种子 42，使用原始九类投票。按训练集中长宽完全相同的真实正常片数量，分成 0、1–24 和至少 25 三层。设计批次中仅考察预测第一名为 Scratch 的样本，在 0.50、0.60、0.70、0.80、0.90 和 0.95 中选符合准入条件且接受量最大的阈值，同量取更高阈值。每层须有至少 12 张候选、5 个批次、90% 实测精度，单侧 95% Wilson 下界须达 75%。用独立的确认批次只验证一次，再要求逐次剔除一个批次所得的最低精度达到 80%。Wilson 区间以晶圆为单位，批次条件和留一分析补充其局限。

只有 `qualification.json` 的 `qualified_bands` 非空才需要隔离的未标注图、RF77 权重和私有来源索引。执行 `pseudo09.py select --help` 查看参数。该步骤逐票复核冻结教师，在预先隔离的 50,000 张中只对获准尺寸生成候选，按图哈希选最多 80 张，每批次最多两张。只有选样回执的 `pseudo_train` 为真，才用 `train09.py --arms student --pseudo "$RUN/rf_gate"` 加上前述相同输入参数运行学生训练。伪标签在交叉熵中权重为 0.25。学生在用于准入的两个开发子集上的指标都只可作描述。

## 配对评价与复核

```bash
python -u evaluate09.py --repo "$REPO" --training "$RUN/training" \
  --pseudo-decision "$RUN/rf_gate/select_receipt.json" --out "$RUN/evaluation"
python -u audit09.py --repo "$REPO" --training "$RUN/training" \
  --eval "$RUN/evaluation" --out "$RUN/audit09"
python -u transitions09.py --repo "$REPO" --training "$RUN/training" \
  --eval "$RUN/evaluation" --out "$RUN/transitions"
```

先在开发选择批次按九分类第一名计算真实划痕命中、正常总误报和正常误判划痕。必须在每个种子均守住三项，且真实划痕命中或正常总误报至少改善一项，才进入固定名额的成本模拟。另在历史开发审计批次上报告同一方案的结果。配对区间以完整批次及图像关联组件为重采样单位，为每个种子单独计算。复核还逐格对照历史 CNN CE 混淆矩阵、逐轮核验 B 和 C 的替换位置、抽查留档合成图中每个新增失效 die 的来源。

运行中产生的逐片预测、权重、原图示例、来源行和入选清单均存入私有证据包。公开结果只含聚合数值、运行参数和文件校验值。
