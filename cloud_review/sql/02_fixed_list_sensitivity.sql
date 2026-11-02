-- Read-only Snowflake scenario analysis. Both inspection lists stay fixed at scratch_5x.
-- Relative cost. A positive gain means lower RF77 cost than RF71.
-- Source: Colab run 20261102T042635Z_bf1507, flagged candidates, benefit_fixed.
-- Input counts from slot_swaps.csv; group totals/5x gain from paired_models.csv and summary.csv.
WITH swap_counts(seed,label,entered,displaced) AS (
  SELECT * FROM VALUES
    (42,'Center',113,59),
    (42,'Donut',23,2),
    (42,'Edge-Loc',50,82),
    (42,'Edge-Ring',86,72),
    (42,'Loc',32,51),
    (42,'Near-full',0,0),
    (42,'Random',29,6),
    (42,'Scratch',3,0),
    (42,'none',3,67),
    (123,'Center',110,73),
    (123,'Donut',19,3),
    (123,'Edge-Loc',69,70),
    (123,'Edge-Ring',79,62),
    (123,'Loc',29,53),
    (123,'Near-full',1,0),
    (123,'Random',28,18),
    (123,'Scratch',3,1),
    (123,'none',2,60),
    (2026,'Center',122,62),
    (2026,'Donut',16,7),
    (2026,'Edge-Loc',45,62),
    (2026,'Edge-Ring',78,68),
    (2026,'Loc',26,47),
    (2026,'Near-full',4,0),
    (2026,'Random',26,16),
    (2026,'Scratch',3,1),
    (2026,'none',2,59)
), source_ref(seed,candidates,capacity,ref_entered,ref_displaced,ref_gain_5x) AS (
  SELECT * FROM VALUES (42,3642,728,339,339,18.780889621087315), (123,3642,728,340,340,16.30971993410215), (2026,3642,728,322,322,16.062602965403627)
), delta AS (
  SELECT seed, COUNT(*) AS labels, COUNT(DISTINCT label) AS distinct_labels,
         SUM(entered) AS entered, SUM(displaced) AS displaced,
         SUM(CASE WHEN label='Scratch' THEN entered-displaced ELSE 0 END) AS scratch_delta,
         SUM(CASE WHEN label NOT IN ('none','Scratch') THEN entered-displaced ELSE 0 END) AS other_defect_delta,
         SUM(CASE WHEN label='none' THEN entered-displaced ELSE 0 END) AS normal_delta
  FROM swap_counts GROUP BY seed
), coeff AS (
  SELECT d.*, r.candidates, r.capacity, r.ref_entered, r.ref_displaced, r.ref_gain_5x,
         0.9*d.scratch_delta AS gain_slope,
         0.9*d.other_defect_delta-0.1*(d.entered-d.displaced) AS gain_intercept
  FROM delta d JOIN source_ref r ON d.seed=r.seed
), scenario(scratch_multiplier) AS (SELECT * FROM VALUES (0),(1),(2),(5),(10))
SELECT c.seed, s.scratch_multiplier, c.candidates, c.capacity, c.scratch_delta,
       c.other_defect_delta, c.normal_delta,
       ROUND((c.gain_intercept+s.scratch_multiplier*c.gain_slope)*1000.0/c.candidates,6)
           AS fixed_list_gain_per_1000_candidates,
       ROUND(-c.gain_intercept/NULLIF(c.gain_slope,0),6) AS mathematical_break_even,
       CASE WHEN c.gain_slope=0 AND c.gain_intercept=0 THEN 'tie for all multipliers'
            WHEN c.gain_slope=0 AND c.gain_intercept>0 THEN 'RF77 lower cost for all multipliers'
            WHEN c.gain_slope=0 THEN 'RF71 lower cost for all multipliers'
            WHEN c.gain_slope>0 AND c.gain_intercept>0 THEN 'RF77 lower cost for all nonnegative multipliers'
            WHEN c.gain_slope<0 AND c.gain_intercept<0 THEN 'RF71 lower cost for all nonnegative multipliers'
            WHEN c.gain_slope>0 THEN 'RF77 lower cost above break-even'
            ELSE 'RF77 lower cost below break-even' END AS fixed_list_interpretation,
       CASE WHEN c.labels=9 AND c.distinct_labels=9 AND c.candidates=3642 AND c.capacity=728
             AND c.entered=c.ref_entered AND c.displaced=c.ref_displaced AND c.entered=c.displaced
             AND ABS((c.gain_intercept+5*c.gain_slope)*1000.0/c.candidates-c.ref_gain_5x)<0.000001
            THEN 'PASS' ELSE 'FAIL' END AS colab_swap_reconciliation
FROM coeff c CROSS JOIN scenario s ORDER BY c.seed,s.scratch_multiplier;
