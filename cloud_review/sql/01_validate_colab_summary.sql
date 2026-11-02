-- One read-only Snowflake SELECT. No stage, imports, persistent objects or writes.
-- Run 20261102T042635Z_bf1507; inputs from cost/{summary,per_class,paired_models,slot_swaps}.csv.
-- Frozen simulated_flagged_candidates, scratch_5x, benefit_fixed. Relative loss units.
-- Reconstruct total cost from reviewed class counts. The input costs are comparison targets only.
WITH inputs(seed,model,n,defects,scratches,reviewed,reviewed_defects,reviewed_scratches,
            upstream_loss,reported_per_1000,reported_per_1000_arrivals) AS (
  SELECT * FROM VALUES
    (42,'RF71',3642,3484,62,728,644,53,188.0,833.1685886875342,278.0087999309809),
    (42,'RF77',3642,3484,62,728,708,56,188.0,814.387699066447,272.10766974376673),
    (123,'RF71',3642,3484,62,728,652,54,188.0,830.2031850631521,277.07704253299977),
    (123,'RF77',3642,3484,62,728,710,56,188.0,813.89346512905,271.9523768441032),
    (2026,'RF71',3642,3484,62,728,651,53,188.0,831.4387699066447,277.4652747821586),
    (2026,'RF77',3642,3484,62,728,708,55,188.0,815.3761669412411,272.4182555430938)
), reference(seed,reported_gain) AS (
  SELECT * FROM VALUES (42,18.780889621087315), (123,16.30971993410215), (2026,16.062602965403627)
), scratch_swaps(seed,entered,displaced) AS (
  SELECT * FROM VALUES (42,3,0), (123,3,1), (2026,3,1)
), costs AS (
  SELECT *, defects + 4*scratches - 0.9*(reviewed_defects + 4*reviewed_scratches)
              + 0.1*reviewed AS reconstructed_units
  FROM inputs
), measures AS (
  SELECT *, 1000.0*reconstructed_units/n AS per_1000,
         1000.0*(reconstructed_units+upstream_loss)/11591 AS per_1000_arrivals
  FROM costs
)
SELECT a.seed, a.n AS candidates, a.reviewed AS reviews_each,
       ROUND(a.per_1000,3) AS rf71_candidate_cost, ROUND(b.per_1000,3) AS rf77_candidate_cost,
       ROUND(a.per_1000-b.per_1000,3) AS gain_per_1000_candidates,
       ROUND(a.per_1000_arrivals-b.per_1000_arrivals,3) AS gain_per_1000_all_arrivals,
       s.entered AS scratch_entered, s.displaced AS scratch_displaced,
       CASE WHEN a.n=3642 AND a.n=b.n AND a.reviewed=728 AND b.reviewed=728
             AND a.defects=b.defects AND a.scratches=b.scratches AND a.upstream_loss=b.upstream_loss
             AND ABS(a.per_1000-a.reported_per_1000)<0.000001
             AND ABS(b.per_1000-b.reported_per_1000)<0.000001
             AND ABS(a.per_1000_arrivals-a.reported_per_1000_arrivals)<0.000001
             AND ABS(b.per_1000_arrivals-b.reported_per_1000_arrivals)<0.000001
             AND ABS(a.per_1000-b.per_1000-r.reported_gain)<0.000001
             AND b.reviewed_scratches-a.reviewed_scratches=s.entered-s.displaced
            THEN 'PASS' ELSE 'FAIL' END AS reconciliation
FROM measures a JOIN measures b ON b.seed=a.seed AND b.model='RF77'
JOIN reference r ON r.seed=a.seed JOIN scratch_swaps s ON s.seed=a.seed
WHERE a.model='RF71' ORDER BY a.seed;
