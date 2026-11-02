# 07 冻结复检回放

07 用原测试批次复算历史决策。分类器、温度、入池门槛、复检名额、同分次序和相对损失都已固定。原测试标签曾用于误差诊断，结论限于回顾性评估。[冻结规则及 SHA-256](FROZEN_RULES.md) · [04 成本与尺寸审计收据](../results/shift/cost_receipt.json) · [冻结参考版本](https://github.com/Matthewsyr/reliable-wafer-classification/tree/92d2002c2b76e2db2fec492bda72ea66605d5e35)

**准备输入。** 使用已有额度的托管 Colab CPU，系统内存至少 12 GB。在云端取得 Kaggle `qingyi/wm811k-wafer-map/versions/1` 的 `LSWMD.pkl`。两个脚本依赖 NumPy、pandas、PyTorch、joblib 和固定版本 scikit-learn 1.6.1。个人 Drive 中的 `WM811K_07_inputs.zip` 保存了以下私有输入，复制到临时运行时的 `/content/`。

| 包内位置 | 原始来源 |
|---|---|
| `inputs/rf_{original71,linearity77}_s{42,123,2026}.joblib` | 六份几何消融训练目录中的 `model.joblib` |
| `inputs/cnn_s{42,123,2026}.pt` | 前两个种子的 repair screen `bn_natural` 权重，种子 2026 的第二轮 screen `baseline` 权重 |
| `inputs/protected_final_test_manifest.csv.gz` | 原 stage 3 输出中的同名测试清单 |
| `inputs/tune_calibration_freeze.json`、`previous/cost/{protocol,tune_calibration_freeze}.json` | 04 的温度和成本冻结文件 |
| `inputs/lot_components.csv.gz` | 完整未标注审计的 `shift_followup/pool/`，保留批次间经同图连接的多跳关系 |

另外把原开发特征缓存 `development_features.npz` 和 `linearity_features.npy` 放到云端 `/content/`。前者来自 stage 2 基线，SHA-256 为 `61c98d4dc415687e20386cd180b48a3ab3f87099e54b31defbae6606329d745b`。后者来自几何消融，SHA-256 为 `93039e3cbeb64fb3ebacc730e150edf4e3ee980fefd7fb09857fee261c6c8865`。它们只在开发特征核验时使用。

**执行。** 在新的托管 Colab 笔记本运行下列 Bash 单元。原图只在云端读取。参考仓库固定到历史提交，07 脚本取自包含本指南的版本。已保留历史代码检出的用户可直接设置 `REF` 和 `CODE`。

```bash
%%bash
set -eu
cd /content
git clone -q https://github.com/Matthewsyr/reliable-wafer-classification.git project
CODE=/content/project/cloud_review/roadmap_07
REF=/content/wm07_ref
git -C project worktree add --detach "$REF" 92d2002c2b76e2db2fec492bda72ea66605d5e35
python -m pip -q install scikit-learn==1.6.1
echo '54a2a6269136af87939eb553e2805f18f8443bb2a91d9814066bf9a5dec4b421  WM811K_07_inputs.zip' | sha256sum -c -
unzip -q WM811K_07_inputs.zip -d private07
python - <<'PY'
import kagglehub
from pathlib import Path
p = Path(kagglehub.dataset_download('qingyi/wm811k-wafer-map/versions/1',path='LSWMD.pkl'))
Path('/content/wm07_raw_path').write_text(str(p if p.is_file() else p/'LSWMD.pkl'))
PY
RAW=$(cat /content/wm07_raw_path)
python -u "$CODE/diagnose_parity.py" --repo "$REF" --raw "$RAW" --inputs /content/private07/inputs \
  --geometry /content/linearity_features.npy --base /content/development_features.npz \
  --windows-lut "$CODE/windows_strip_trig.json" --extra 1024 --out /content/parity_report.json
python - <<'PY'
import json
from pathlib import Path
p=json.loads(Path('/content/parity_report.json').read_bytes())
assert p['n']==1090 and p['base']=={} and p['windows_angles']=={}
PY
python -u "$CODE/score_portable.py" --repo "$REF" --raw "$RAW" --inputs /content/private07/inputs \
  --windows-lut "$CODE/windows_strip_trig.json" --parity-report /content/parity_report.json \
  --out /content/wm07_scores
python -u "$CODE/replay_target.py" --decision "$CODE/decision.py" --repo "$REF" \
  --input /content/wm07_scores --cost /content/private07/previous/cost \
  --lot-components /content/private07/inputs/lot_components.csv.gz --out /content/wm07_replay
```

开发检查先核对 1,090 张晶圆。原 Windows 训练环境与云端 NumPy 对个别角度的三角函数末位取值不同，足以让几何特征的条带分箱和树的走向改变。第一次检查在测试打分前中止。生效的入口只固定原环境 36 个角度的正弦和余弦。这次 71 个基础特征和 6 个几何特征在 1,090 张上与训练缓存逐值相同，三个 RF77 的概率最大差为 `2.22e-16`。推理容差保持原值。[常数表](windows_strip_trig.json) · [兼容处理](strip_portability.py) · [实际运行入口](score_portable.py)

评分程序核验原文件、检查点和逐片身份，用开发集归档概率核对九个模型，再保存测试元数据、全量 RF71 初筛概率、候选片分数及 `score_receipt.json`。决策程序用真实标签结算各策略，生成 `run_receipt.json`。两个输出目录均须是新目录。将输出复制到自己的持久存储后释放会话。公开聚合位于 [results/](results/)，权重、逐片分数和名单留在私人存储。

没有原私有权重的读者可检查公开汇总、源代码及收据，也可用 `score_replay.py --self-test` 和 `replay_target.py --decision decision.py --self-test` 验证合成样本。重新训练相同架构会产生另一组权重，须作为独立复现实验记录。
### 核对结果与生成阅读报告

在同一个 Colab 终端运行以下命令。第一条从 04 的聚合计数重建开发审计表，第二条核验每一个队列的真实占用和成本，最后一条生成中文报告。

```bash
python "$CODE/build_relative_readout.py" --cost /content/private07/previous/cost --out /content/dev04_summary
python "$CODE/check_replay_aggregates.py" --results /content/wm07_replay
python "$CODE/render_closeout.py" --run /content/wm07_replay --support "$CODE" --repo "$REF" --out /content/REPORT.zh-CN.md
```
