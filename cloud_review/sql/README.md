# Snowflake review-cost analysis

Hosted receipts from the 2026-11-01 Los Angeles run: `00_connection_check.sql` returned one row; 01 returned three passing checks; 02 returned fifteen passing scenario checks. The Drive handoff also contains 06, a seed-42, per-wafer Snowsight query generated inside the existing Colab runtime from its hash-verified scores. It ranks 3,642 real candidates in four virtual queues, compares both 728-wafer lists with the saved reference IDs, and checks the realized costs. The final query passed. See [Chinese readout](../RESULTS.zh-CN.md) for floating-point tie handling and the preserved first-attempt rounding failure. Files 03 and 04 are optional templates for a broader import of aggregated evidence; their execution is not claimed here.

The two short worksheets run as independent, read-only queries in Snowsight. Neither uses a staged file, creates an object or trains a model. Their `VALUES` rows come from the completed hosted Colab run `20261102T042635Z_bf1507`, evaluated on the previously examined development audit.

| Worksheet | Inputs and result |
|---|---|
| [01_validate_colab_summary.sql](01_validate_colab_summary.sql) | Six RF71/RF77 records for matched seeds. Reconstructs the flagged-pool and full-arrival costs under `scratch_5x`, checks the recorded paired differences and reconciles the change in reviewed Scratch counts. Expect three rows with `PASS`. |
| [02_fixed_list_sensitivity.sql](02_fixed_list_sensitivity.sql) | Twenty-seven paired class-level slot exchanges. Reprices the fixed 5× Scratch lists at loss multipliers 0, 1, 2, 5 and 10. Reports the gain per 1,000 candidates, the break-even multiplier and a reference check for each seed. Expect 15 rows with `PASS`. |

Use the existing trial-only Snowflake account if the account status and remaining allowance confirm this run cannot produce additional charges. A remaining promotional balance can coexist with on-demand billing after an upgrade. A trial that ends without upgrading is suspended. If trial-only billing cannot be verified, keep the existing hosted Colab results and the SQL for later. These worksheets require no subscription change, payment details, service integration, new warehouse or new account. [Snowflake trial terms](https://docs.snowflake.com/en/user-guide/admin-trial-account)

## Frozen experiment

The audit contains 11,591 wafers. The frozen RF71 seed-42 gate selects 3,642 candidates with temperature-calibrated defect probability at least 0.5, shared by all models. The original cloud code uses seeded keyed BLAKE2b ordering of lots and then wafers within lots, with a separate identity-based tiebreaker. It divides the common pool into queues of 1,000, 1,000, 1,000 and 642. Their capacities are 200, 200, 200 and 128.

The original replay has five policies. It includes no review, a common seeded random list, a list sorted by defect probability, a full-capacity list sorted by expected net benefit and a capped positive-net-benefit list. The expected benefit is the sum of calibrated class probabilities times class loss times the avoided fraction, less the inspection fee. Equal loss and equal efficacy give the same fixed ranking as defect probability. The three original scenarios assign one unit to ordinary defects, zero to normal wafers, and optionally five units to Scratch or Near-full. Every reviewed wafer costs 0.1 relative units and averts 90% of its true defect loss in the simulation. Cost per 1,000 arrivals includes the loss of unflagged defects under this fixed gate. Reported gains compare like populations, seeds, policies and price schedules.

## What the second worksheet measures

Under the original full-capacity 5× Scratch decision, RF77 gains 3, 2 and 2 net reviewed Scratch wafers relative to RF71 across the three seeds. The net gains in other reviewed defects are 61, 56 and 55. Repricing retains the exact same list memberships. For Scratch multiplier *m*, total RF71 cost minus RF77 cost is 0.9 × (net extra other defects + *m* × net extra Scratch) minus 0.1 × net extra reviews. The two fixed lists use the same capacity. The mathematical break-even multipliers are approximately −20.33, −28 and −27.5. Within nonnegative multipliers, these particular RF77 lists retain lower simulated cost for all three seeds.

The original `equal` scenario recomputed rankings from equal losses and produced equal RF71/RF77 point costs. The `m = 1` rows in worksheet 02 value the *different lists selected at m = 5*. They answer how fragile an existing decision is to an assumed price, while the original `equal` run measures the outcome of selecting a new list under that price. Each seed evaluates the same wafers. The reported savings are simulated relative loss units.

## Optional full-table workflow

Import four CSVs from the same cloud result's `cost/` folder. Only aggregate rows are required.

| File | Grain | Expected data rows |
|---|---|---:|
| `summary.csv` | model seed × model × pool × ranking scenario × policy | 270 |
| `per_class.csv` | summary row × true class | 2,430 |
| `slot_swaps.csv` | paired model comparison × pool × seed × ranking scenario × benefit policy × true class | 648 |
| `gate.csv` | true class × counts entering and excluded | 9 |

1. Use one verified trial-only session with an existing database and schema. Run [03_session_inputs.sql](03_session_inputs.sql). Its stage and tables are temporary.
2. Stage the four **uncompressed** CSVs under their exact names. In a supported Snowflake client in the same session, issue e.g. `PUT file:///absolute/path/per_class.csv @WM_REVIEW_TRIAL_STAGE AUTO_COMPRESS=FALSE` and repeat for the other files. A session in a separate client cannot address this temporary stage. For an existing authorized stage, adjust the four COPY source locations accordingly. The first two worksheets need no file upload. [Snowflake PUT](https://docs.snowflake.com/en/sql-reference/sql/put)
3. Execute [04_full_cost_analysis.sql](04_full_cost_analysis.sql) in that session. It reconstructs all 270 original policies' aggregate costs from truth counts and all 288 original/alternative fixed-list paired comparisons from slot exchanges. Require all eight initial result checks to report `PASS`. Then use the second result set for the 270 original operating points and the third for the RF77/RF71 sensitivity comparison.
4. Add a hypothetical price/efficacy/fee schedule to `WM_REVIEW_VALUES` for additional fixed-list analysis. Adjust the 288 comparison-count assertion for an expanded schedule count (72 original paired comparisons × number of valuations). Close the session when finished. Review capacity applies to each virtual queue in the source Colab replay. The SQL aggregate check verifies total reviewed counts and their capacity.

The original cloud notebook owns individual ranks, queue formation, calibration and lot-bootstrap intervals. The SQL independently reconstructs and stresses the cost of those decisions from the reported outcome counts. Rebuilding the ranking after changing a loss schedule requires the cloud notebook's per-wafer scores. Worksheet execution evidence, when available, is recorded separately from these source files.
