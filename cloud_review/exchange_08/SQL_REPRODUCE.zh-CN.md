# 在 Snowflake 核对两套复检名单

SQL 的输入是两套冻结名单的并集，每个种子、每张晶圆各一行。A 按 RF77 的任意缺陷概率排序，B 按划痕漏损五倍情景的正净收益排序。两者在相同的上游候选中做决策。公开的 [查询模板](08_exchange_query.sql) 使用名为 `WM08_SELECTED_UNION` 的输入关系，包含下列列。

| 列 | 含义 |
|---|---|
| `seed, source_row, queue` | 训练种子、原始晶圆行号、虚拟报警队列编号 |
| `truth, train_normal_count` | 原始类别标签、31,457 张训练片中同一原始高宽的正常片数量 |
| `p_scratch, p_any` | 已冻结温度校准后的划痕概率、任意缺陷概率 |
| `in_a, in_b` | 两种策略的实际复检标志，取 0 或 1 |

**名单与指标。** 查询先核对两个名单的复检总数、并集身份唯一性及 50 个队列的容量，再以 `in_a <> in_b` 仅保留策略独选片。按训练正常片数量分为 0、1–24、至少 25 三档。查询同时给出真实类别组成、按尺寸汇总的结果、以及每档中按划痕概率分箱后的观测频率。每组都报告预测划痕概率和、均值、真实划痕比例和两者之差。概率过高的证据应看整个尺寸档或同一概率箱中的真实频率；按已知类别拆开的计数主要解释挤出了哪些缺陷。

假设复检收费 0.1、避免损失比例 0.9，普通缺陷漏损 1、划痕漏损 5，正常漏损 0。当两套策略的共同入选片保持相同，其复检收益和费用会在成对差值中抵消。按每千张全部来片结算时，完整账本还包含共同未复检和未报警的全部来片。此 SQL 返回交换组的模拟净收益，不能直接代替全来片总成本。

**执行。** 在托管 Colab 完成 `derive_exchange.py` 后，使用下列命令打包私有输入。

```bash
python make_snowflake_exchange.py build --run /content/exchange --out /content/new_sql
```

生成器将独选组汇总与另一份逐类汇总对账，然后把并集记录无损压缩生成只读查询。在 Snowsight 的试用工作区打开 `08_private_exchange_all.sql`，或依次打开生成的各个单种子查询，完整执行并下载 CSV。私有查询内含编码后可还原的逐片数据，应留在私人工作区。公开模板、生成脚本和仅含汇总的返回表可用于复核。对下载结果使用下面的命令；分种子运行时，将全部 CSV 都列在 `--actual` 后面。

```bash
python make_snowflake_exchange.py compare \
  --expected /content/new_sql/expected_snowflake_exchange.csv \
  --actual snowflake_result.csv --receipt snowflake_verified.json
```

比较按分组键连接，计数逐格完全相同，浮点值使用 `abs_tol=1e-8, rel_tol=1e-10`。查询由 Snowflake 对逐片并集重新做差和汇总；参考表由 Python 对同一份冻结决策单独核算。两者一致可以验证结算实现，仍需从未参与训练、预训练或选策略的新批次验证迁移后的收益。私有字节打包使用 Snowflake 文档支持的字符串解压和二进制转换，生成器将每份查询控制在 990,000 字节以内。[DECOMPRESS_STRING](https://docs.snowflake.com/en/sql-reference/functions/decompress_string) · [TO_BINARY](https://docs.snowflake.com/en/sql-reference/functions/to_binary) · [查询大小](https://docs.snowflake.com/en/user-guide/query-size-limits)

## 本次执行记录

先将公开仓库单独检出至 `4b2f0123ea429fa7b1b5e5e2139148dc789c3b2d`，在托管 Colab 运行 `python derive_exchange.py --repo PINNED_07 --inputs WM811K_07_inputs.zip --evidence WM811K_07_private_evidence.zip --out /content/exchange`。08 的派生脚本位于包含本次提交的另一工作目录。它对两份私有包先核哈希，再重建同一候选池及两个名单。参考汇总与逐片行号分开保存在 `public/`、`private/`。[派生收据](results/exchange_receipt.json)

三个种子分别使用一条只读查询。首次生成的 SQL 在真类名称映射的 `CASE` 分支之间误加逗号，Snowflake 在编译阶段报错。修复工具在验证原始查询 SHA-256 后只删除各查询中的八个多余逗号，输入概率和名单保持原样。现行生成器已修复该处并有合成回归检查。早期输入收据记录旧生成器版本，语法修复收据记录实际运行的三份修复后查询的 SHA-256。[原始打包收据](results/snowflake_input_receipt.json) · [修复收据](results/syntax_repair_receipt.json) · [修复工具](repair_sql_syntax.py)

浏览器阻止了直接下载结果 CSV，本次通过 Snowsight 结果表的复制功能取得带表头的原始制表符文本。三个种子分别有 76、71、72 行。这里保留原始字节，并按同一列顺序转换为一份 CSV 供比较。与托管参考表核对的 219 组中，计数无差异，浮点字段最大绝对差为 `4.992e-10` 以内。[42 原始 TSV](results/snowflake_s42.tsv) · [123 原始 TSV](results/snowflake_s123.tsv) · [2026 原始 TSV](results/snowflake_s2026.tsv) · [合并返回表](results/snowflake_exchange.csv) · [托管参考表](results/expected_snowflake_exchange.csv) · [逐字段校验](results/snowflake_verification.json)

在公开结果目录中复核汇总即可使用下列命令。

```bash
python make_snowflake_exchange.py compare --expected results/expected_snowflake_exchange.csv --actual results/snowflake_exchange.csv
```
