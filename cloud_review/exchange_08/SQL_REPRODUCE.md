# Verify the inspection-list exchange in Snowflake

The [SQL template](08_exchange_query.sql) expects a relation called `WM08_SELECTED_UNION`, with one row per selected wafer per seed. A is the RF77 any-defect policy and B the fivefold-Scratch, positive-net-benefit policy. Both rank the same upstream alerts.

| Columns | Definition |
|---|---|
| `seed, source_row, queue` | Model seed, original wafer row and simulated alert queue |
| `truth, train_normal_count` | True class and matching native size count among normals in the 31,457-wafer training set |
| `p_scratch, p_any` | Frozen calibrated probabilities of Scratch and any defect |
| `in_a, in_b` | Binary inspection decisions |

The query checks list totals, unique identities and all 50 queue capacities per seed. It filters to `in_a <> in_b` for exchange analysis. Group by matching normal training count (0, 1–24, ≥25), true class, and Scratch probability bins. Compare the sum and mean of predicted Scratch probabilities to observed Scratch frequency across each size band or bin. True-class cells identify cases lost or gained.

The simulated fee is 0.1 per review, avoidable loss fraction 0.9, ordinary defect loss 1, Scratch loss 5 and normal loss 0. Common decisions cancel in the paired cost difference. Whole-stream absolute costs also include all common unreviewed and unalerted arrivals. The SQL reports net benefit on the exchange subset.

## Reproduce from archived inputs

The full private replay requires the original inputs and frozen outputs. In hosted Colab, run the stage 08 scripts from their current directory and supply a separate stage 07 checkout pinned to `4b2f0123ea429fa7b1b5e5e2139148dc789c3b2d`:

```bash
python derive_exchange.py --repo PINNED_07 --inputs WM811K_07_inputs.zip \
  --evidence WM811K_07_private_evidence.zip --out /content/exchange
python make_snowflake_exchange.py build --run /content/exchange --out /content/new_sql
```

The generator verifies both archive hashes, reconciles the exclusive summary to an independent class aggregation and losslessly packs the per-wafer union into a read-only query. Execute `08_private_exchange_all.sql` or the generated per-seed queries in Snowsight. Keep these queries private because their encoded rows are recoverable. Download the CSV results and compare them with the separately produced expected aggregates:

```bash
python make_snowflake_exchange.py compare \
  --expected /content/new_sql/expected_snowflake_exchange.csv \
  --actual snowflake_result.csv --receipt snowflake_verified.json
```

Pass all per-seed results after `--actual` if executed separately. The comparison joins on group keys and checks all counts exactly. Floating comparisons use abs_tol=1e-8 and rel_tol=1e-10. Each generated query is capped at 990,000 bytes. [DECOMPRESS_STRING](https://docs.snowflake.com/en/sql-reference/functions/decompress_string) · [TO_BINARY](https://docs.snowflake.com/en/sql-reference/functions/to_binary) · [Query size limits](https://docs.snowflake.com/en/user-guide/query-size-limits)

## Original execution and public replay

The first SQL payload had extraneous commas between true-label CASE branches and failed to compile. The repair script verified its SHA-256 before removing eight commas per query. Probabilities and selection flags were unchanged. The current generator includes the fix and a synthetic regression check. The archived results were copied from the Snowsight grid as original tab-separated text. Counts matched across all 219 keyed groups, with maximum numeric error below 5e-10.

From this directory, replay the public verification directly:

```bash
python make_snowflake_exchange.py compare --expected results/expected_snowflake_exchange.csv --actual results/snowflake_exchange.csv
```

[Derivation receipt](results/exchange_receipt.json) · [Original packing receipt](results/snowflake_input_receipt.json) · [Syntax repair receipt](results/syntax_repair_receipt.json) · [Repair script](repair_sql_syntax.py) · [Seed 42](results/snowflake_s42.tsv) · [Seed 123](results/snowflake_s123.tsv) · [Seed 2026](results/snowflake_s2026.tsv) · [Merged output](results/snowflake_exchange.csv) · [Expected aggregates](results/expected_snowflake_exchange.csv) · [Comparison receipt](results/snowflake_verification.json)
