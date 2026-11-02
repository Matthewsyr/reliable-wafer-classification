# 复现 04 的未标注审计与历史复检分层

**公开复现未标注池。** 在免费的托管 Colab 或 Kaggle CPU 中，以本仓库根目录为工作目录。准备公开数据 `qingyi/wm811k-wafer-map/versions/1` 的 `LSWMD.pkl`，将 `RAW` 和 `OUT` 分别设为原文件路径和自己的持久结果目录。依赖为 NumPy、pandas 和 psutil。扫描前需要至少 8 GiB 当前可用内存和超过 2 GiB 临时盘空间。默认运行上限为 25 分钟。

```sh
python cloud_review/extension/shift_followup.py --stage pool \
  --previous cloud_review/results/legacy_04 \
  --reference data-reference/development_reference.csv.gz \
  --indexer cloud_review/extension/shift_audit_v1.py \
  --raw "$RAW" --out "$OUT"
```

[旧版索引器](shift_audit_v1.py) 逐字节取自原 04 笔记本的 `%%writefile shift_audit.py` 单元，SHA-256 为 `786692eaea13c1b0a6d52d821e443e00f546735aa353a24d034caec94638b7e2`。它在解包前核对原文件 SHA-256，再以原始高宽和像素计算图像哈希。命令使用原云端保存的[旧版聚合摘要](../results/legacy_04/shift/shift_summary.json)。该摘要还记录运行时的空闲内存和软件版本，重新扫描产生的新摘要可能有不同的文件哈希。

重建后核对 `$OUT/pool/pool_summary.json` 的组件数、保护数和唯一图数。历史值为 34,610、8,848、434,498。候选清单包含原始行号、批次、组件、高宽及图像哈希，完整清单保存在运行者自己的目录。比较不同时刻生成的清单时，应比对解压后的记录，gzip 压缩头可能包含不同时间戳。[历史池统计](../results/shift/pool_summary.json) · [原产物校验值](../results/shift/pool_receipt.json)

**复现旧复检名单的成本分层。** 原项目的个人 Drive 已保存 `Colab Notebooks/WM811K_Cloud_Results/20261102T042635Z_bf1507/`。将 `PREVIOUS` 指向这个运行目录。程序从中读取下列文件。

- `cost/protocol.json` 和 `cost/tune_calibration_freeze.json` 定义旧相对成本以及种子 42 的 RF71 上游报警规则。
- `cost/gate.csv` 和 `cost/summary.csv` 用来核对各类报警数、复检数与损失总额。
- `cost/evidence/simulated_flagged_candidates/s{42,123,2026}/{RF71,RF77}/selected_rows.json.gz` 表示 3 个种子和 2 个模型的 6 份名单。程序取各自 `scratch_5x` 情景、`benefit_fixed` 排序的入选原始行号。

```sh
python cloud_review/extension/shift_followup.py --stage cost \
  --previous "$PREVIOUS" --reference data-reference/development_reference.csv.gz \
  --anchor-probs results/probabilities/rf_original71_s42_audit.csv.gz --out "$OUT"
python cloud_review/extension/shift_followup.py --stage decision --out "$OUT"
```

`decision` 读取同一 `OUT` 下通过收据核验的 `cost` 和 `pool`。公开仓库提供阶段统计、分层表和输入校验值；仅凭这些聚合数无法还原旧名单中每一张晶圆的身份。外部读者可以直接复算零正常尺寸覆盖层的误报率、比较 RF71/RF77 的复检数，或使用上述公开原始数据独立重建未标注池。模型性能和复检收益来自回顾性开发审计；正式泛化验证仍需新批次。[公开分层计数](../results/shift/audit_cost_strata.csv) · [历史成本收据](../results/shift/cost_receipt.json) · [尺寸覆盖与漏损](../results/shift/pool_support_and_cost.csv)
