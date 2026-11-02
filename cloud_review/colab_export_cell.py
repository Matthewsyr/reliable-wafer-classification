"""Append to the completed hosted Colab session. Requires its existing work and dst."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np


def export_wafer_sql(work, dst):
    if not Path("/content").is_dir() or not any(k.startswith("COLAB_") for k in os.environ):
        raise RuntimeError("Use the existing hosted Colab runtime.")
    work, dst = Path(work).resolve(), Path(dst).resolve()
    cost = dst / "cost"
    protocol = json.loads((cost / "protocol.json").read_text())

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def csv_rows(path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def quoted(value):
        return "'" + str(value).replace("\\", "\\\\").replace("'", "''") + "'"

    def numeric_string(value):
        return quoted(format(float(value), ".17g"))

    score_file = work / "scored" / "probabilities_s42.npz"
    assert digest(score_file) == protocol["checked_input_digests"][score_file.name]
    assert digest(work / "decision.py") == protocol["decision_sha256"]
    sys.path.insert(0, str(work))
    import decision as d
    assert Path(d.__file__).resolve() == work / "decision.py"
    with np.load(score_file, allow_pickle=False) as archive:
        assert tuple(archive["class_order"]) == d.CLASS_ORDER
        gate = 1 - archive["p_rf"][:, -1] >= 0.5
        assert len(gate) == 11591 and gate.sum() == protocol["gate"]["rows_selected"] == 3642
        ids = archive["source_row"][gate].astype(str)
        lots = archive["lot"][gate].astype(str)
        truth = archive["y"][gate]
        probabilities = [archive[k][gate] for k in ("p_rf", "p_rf_geometry")]
    order, chunks, queues, _, _, ties = d._plans(ids, lots, protocol["queue_seed"])
    ids, lots, truth = ids[order], lots[order], truth[order]
    assert len(np.unique(ids)) == len(ids)
    tie_order = np.empty(len(ids), dtype=int)
    tie_order[np.lexsort((ids, ties))] = np.arange(1, len(ids) + 1)
    queue, cap = np.empty(len(ids), dtype=int), np.empty(len(ids), dtype=int)
    for (lo, hi), entry in zip(chunks, queues):
        queue[lo:hi], cap[lo:hi] = entry["queue"], entry["capacity"]
    assert [x["capacity"] for x in queues] == [200, 200, 200, 128]
    scenario = d.SCENARIOS["scratch_5x"]
    amounts, effects, harm = map(np.asarray, (scenario.missed_loss, scenario.effect, scenario.review_harm))
    assert np.array_equal(amounts, [1, 1, 1, 1, 1, 1, 1, 5, 0])
    assert np.array_equal(effects, [.9] * 8 + [0]) and not harm.any() and scenario.review_fee == .1
    pool, policy, case = "simulated_flagged_candidates", "benefit_fixed", "scratch_5x"
    expected, fields = {}, []
    summary = csv_rows(cost / "summary.csv")
    paired = csv_rows(cost / "paired_models.csv")
    for model, raw in zip(("RF71", "RF77"), probabilities):
        evidence_dir = cost / "evidence" / pool / "s42" / model
        check = json.loads((evidence_dir / "protocol.json").read_text())
        assert d._digest(ids, lots) == check["candidate_order_sha256"]
        assert hashlib.sha256(truth.astype("uint8").tobytes()).hexdigest() == check["ordered_labels_sha256"]
        with gzip.open(evidence_dir / "selected_rows.json.gz", "rt", encoding="utf-8") as stream:
            frozen = json.load(stream)[case][policy]
        assert len(frozen) == len(set(frozen)) == 728
        p = np.asarray(raw[order], dtype=np.float64)
        source_score = p @ (amounts * effects - harm) - scenario.review_fee
        membership = d._top(source_score, ids, ties, chunks)
        assert set(ids[membership]) == set(frozen)
        assert np.max(np.abs(.9 * ((1 - p[:, -1]) + 4 * p[:, 7]) - .1 - source_score)) < 1.e-12
        refs = [r for r in summary if (r["model_seed"], r["model"], r["pool_kind"], r["scenario"], r["policy"]) == ("42", model, pool, case, policy)]
        assert len(refs) == 1
        realized_cost = (amounts[truth] + membership * (.1 - amounts[truth] * effects[truth])).sum()
        reference = float(refs[0]["cost_per_1000"])
        assert abs(realized_cost * 1000 / len(ids) - reference) < 1.e-7
        fields.append((p[:, -1], p[:, 7], source_score, membership))
        expected[model] = reference
    pair = [r for r in paired if (r["pool_kind"], r["model_seed"], r["new_model"], r["base_model"], r["scenario"], r["policy"]) == (pool, "42", "RF77", "RF71", case, policy)]
    assert len(pair) == 1
    pair = pair[0]
    assert abs((expected["RF71"] - expected["RF77"]) - float(pair["paired_savings_per_1000"])) < 1.e-7
    raw_rows = []
    for i, source_id in enumerate(ids):
        row = [quoted(source_id), quoted(lots[i]), str(queue[i]), str(cap[i]), str(tie_order[i]), quoted(d.CLASS_ORDER[int(truth[i])])]
        for p0, ps, source_score, membership in fields:
            row.extend((numeric_string(p0[i]), numeric_string(ps[i]), numeric_string(source_score[i]), str(int(membership[i]))))
        raw_rows.append("(" + ",".join(row) + ")")
    sql = """-- WM-811K hosted Colab replay: seed 42, scratch_5x, benefit_fixed.
-- One read-only statement. Run only on a verified covered trial warehouse.
-- No images or training records are included. Every row is in the shared simulated gate.
-- The fresh two-probability formula is algebraically equal to the source nine-term float64
-- dot product. It can differ by rounding near ties. pure_* shows the independent rank.
-- replay_* breaks ties within SQL score bins (1e-9) using the original model's full
-- floating-point dot product, then the original label-blind tie order. Membership flags
-- are compared ONLY AFTER row_number. Costs below are illustrative relative loss units.
WITH raw AS (
  SELECT column1::VARCHAR AS source_row,column2::VARCHAR AS lot,column3::INT AS queue_id,
         column4::INT AS capacity,column5::INT AS tie_order,column6::VARCHAR AS truth,
         column7::DOUBLE AS rf71_none,column8::DOUBLE AS rf71_scratch,column9::DOUBLE AS rf71_dot,
         column10::INT AS frozen71,column11::DOUBLE AS rf77_none,column12::DOUBLE AS rf77_scratch,
         column13::DOUBLE AS rf77_dot,column14::INT AS frozen77
  FROM VALUES
""" + ",\n".join(raw_rows) + """
), scores AS (
 SELECT raw.*,
    0.9*((1.0-rf71_none)+4.0*rf71_scratch)-0.1 AS value71,
    0.9*((1.0-rf77_none)+4.0*rf77_scratch)-0.1 AS value77,
    CASE WHEN truth='none' THEN 0.0 WHEN truth='Scratch' THEN 5.0 ELSE 1.0 END AS loss_units
 FROM raw
), ranks AS (
 SELECT scores.*,
    ROW_NUMBER() OVER(PARTITION BY queue_id ORDER BY value71 DESC,tie_order) AS pure_rank71,
    ROW_NUMBER() OVER(PARTITION BY queue_id ORDER BY value77 DESC,tie_order) AS pure_rank77,
    ROW_NUMBER() OVER(PARTITION BY queue_id ORDER BY ROUND(value71,9) DESC,rf71_dot DESC,tie_order) AS replay_rank71,
    ROW_NUMBER() OVER(PARTITION BY queue_id ORDER BY ROUND(value77,9) DESC,rf77_dot DESC,tie_order) AS replay_rank77
 FROM scores
), promoted AS (
 SELECT ranks.*, IFF(replay_rank71<=capacity,1,0) AS picked71,
        IFF(replay_rank77<=capacity,1,0) AS picked77 FROM ranks
), per_wafer AS (
 SELECT promoted.*,loss_units + picked71*(0.1-0.9*loss_units) AS cost71,
        loss_units + picked77*(0.1-0.9*loss_units) AS cost77
 FROM promoted
), metrics AS (
 SELECT COUNT(*) AS candidates,COUNT(DISTINCT source_row) AS distinct_ids,
 SUM(picked71) AS slots71,SUM(picked77) AS slots77,
 SUM(IFF(picked71<>frozen71,1,0)) AS id_mismatches71,SUM(IFF(picked77<>frozen77,1,0)) AS id_mismatches77,
 SUM(IFF(IFF(pure_rank71<=capacity,1,0)<>frozen71,1,0)) AS pure_float_id_mismatches71,
 SUM(IFF(IFF(pure_rank77<=capacity,1,0)<>frozen77,1,0)) AS pure_float_id_mismatches77,
 GREATEST(MAX(ABS(value71-rf71_dot)),MAX(ABS(value77-rf77_dot))) AS max_dot_error,
 SUM(IFF(picked77=1 AND picked71=0,1,0)) AS entered,
 SUM(IFF(picked71=1 AND picked77=0,1,0)) AS displaced,
 SUM(IFF(truth='Scratch',picked77-picked71,0)) AS scratch_net,
 SUM(IFF(truth='none',picked77-picked71,0)) AS normal_net,
 -- Sum fixed-point costs first. AVG(NUMBER) can round before per-1000 scaling.
 1000*SUM(cost71)::DOUBLE/COUNT(*) AS rf71_cost_per_1000,
 1000*SUM(cost77)::DOUBLE/COUNT(*) AS rf77_cost_per_1000,
 1000*SUM(cost71-cost77)::DOUBLE/COUNT(*) AS savings_per_1000,
 1000*SUM(loss_units+IFF(pure_rank77<=capacity,0.1-0.9*loss_units,0))::DOUBLE/COUNT(*)
     AS pure_float_rf77_cost_per_1000,
 SUM(IFF(pure_rank77<=capacity AND truth='Scratch',1,0)) AS pure_float_rf77_scratches,
 SUM(IFF(picked77=1 AND truth='Scratch',1,0)) AS replay_rf77_scratches
 FROM per_wafer
)
-- To inspect individual real IDs replace the final SELECT with:
-- SELECT source_row,lot,truth,queue_id,capacity,picked71,picked77,value71,value77,
--   rf71_none,rf71_scratch,rf77_none,rf77_scratch,pure_rank71,pure_rank77,replay_rank71,replay_rank77
-- FROM per_wafer WHERE picked71<>picked77 ORDER BY queue_id,tie_order;
"""
    sql += f"""SELECT metrics.*,
  IFF(candidates=3642 AND distinct_ids=3642 AND slots71=728 AND slots77=728
      AND id_mismatches71=0 AND id_mismatches77=0 AND max_dot_error<1e-12
      AND entered={int(pair['entered'])} AND displaced={int(pair['displaced'])}
      AND ABS(rf71_cost_per_1000-{expected['RF71']:.17g})<1e-7
      AND ABS(rf77_cost_per_1000-{expected['RF77']:.17g})<1e-7
      AND ABS(savings_per_1000-{float(pair['paired_savings_per_1000']):.17g})<1e-7,
      'PASS','FAIL') AS replay_status FROM metrics;
"""
    encoded = sql.encode("utf-8")
    assert len(encoded) < 1_000_000, "SQL payload exceeds the expected bound"
    target = dst / "06_seed42_wafer_replay.sql"
    if target.exists():
        assert target.read_bytes() == encoded, "Refusing to overwrite different evidence"
    else:
        target.write_bytes(encoded)
    print(f"Bridge exported: {target.name}; {len(encoded)} bytes; 3,642 unique rows; SHA256={digest(target)}")
    print(f"Expected RF71={expected['RF71']:.6f}, RF77={expected['RF77']:.6f}, savings={float(pair['paired_savings_per_1000']):.6f} per 1,000.")
    return target


cloud_review_sql_path = export_wafer_sql(work, dst)
