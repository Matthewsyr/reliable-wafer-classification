-- Optional. Before running, stage the four files from the SAME completed Colab
-- cost/ directory in the session created by 03. File names below are literal.
COPY INTO WM_REVIEW_CLASS FROM @WM_REVIEW_TRIAL_STAGE FILES=('per_class.csv') ON_ERROR=ABORT_STATEMENT;
COPY INTO WM_REVIEW_GATE FROM @WM_REVIEW_TRIAL_STAGE FILES=('gate.csv') ON_ERROR=ABORT_STATEMENT;
COPY INTO WM_REVIEW_SWAPS FROM @WM_REVIEW_TRIAL_STAGE FILES=('slot_swaps.csv') ON_ERROR=ABORT_STATEMENT;
COPY INTO WM_REVIEW_REPORTED FROM @WM_REVIEW_TRIAL_STAGE FILES=('summary.csv') ON_ERROR=ABORT_STATEMENT;

-- ranking_scenario always identifies the original, already selected list.
-- valuation changes the price of outcomes while keeping that same list.
CREATE TEMPORARY TABLE WM_REVIEW_VALUES AS
SELECT column1 AS valuation, column2 AS scratch_loss, column3 AS near_full_loss,
       column4 AS other_defect_loss, column5 AS avoided_fraction, column6 AS inspection_fee
FROM VALUES
  ('equal',1,1,1,0.9,0.1), ('scratch_5x',5,1,1,0.9,0.1),
  ('near_full_5x',1,5,1,0.9,0.1), ('scratch_10x',10,1,1,0.9,0.1);

CREATE TEMPORARY VIEW WM_REVIEW_CLASS_COST AS
SELECT g.label, g.all_audit, g.simulated_flagged, g.excluded_before_ranking,
       v.valuation, v.avoided_fraction, v.inspection_fee,
       CASE g.label WHEN 'none' THEN 0 WHEN 'Scratch' THEN v.scratch_loss
            WHEN 'Near-full' THEN v.near_full_loss ELSE v.other_defect_loss END AS loss
FROM WM_REVIEW_GATE g CROSS JOIN WM_REVIEW_VALUES v;

CREATE TEMPORARY VIEW WM_REVIEW_REPRICED AS
WITH class_costs AS (
  SELECT c.model_seed,c.model,c.pool_kind,c.ranking_scenario,c.policy,v.valuation,
         c.wafer_support,c.reviewed,
         CASE WHEN c.label='none' THEN c.reviewed ELSE 0 END AS normal_reviews,
         CASE WHEN c.label='Scratch' THEN c.reviewed ELSE 0 END AS scratch_reviews,
         c.reviewed*v.inspection_fee AS fees,
         c.wafer_support*v.loss-c.reviewed*v.loss*v.avoided_fraction AS residual,
         CASE WHEN c.pool_kind='simulated_flagged_candidates'
              THEN v.excluded_before_ranking*v.loss ELSE 0 END AS upstream_loss,
         v.all_audit AS class_arrivals
  FROM WM_REVIEW_CLASS c JOIN WM_REVIEW_CLASS_COST v ON c.label=v.label
)
SELECT model_seed,model,pool_kind,ranking_scenario,policy,valuation,
       SUM(wafer_support) AS pool_size,SUM(class_arrivals) AS all_arrivals,
       SUM(reviewed) AS reviewed,SUM(normal_reviews) AS normal_reviews,
       SUM(scratch_reviews) AS scratch_reviews,SUM(fees) AS fees,
       SUM(residual) AS residual,SUM(upstream_loss) AS upstream_loss,
       SUM(fees+residual) AS pool_cost,
       1000.0*SUM(fees+residual)/NULLIF(SUM(wafer_support),0) AS cost_per_1000_pool,
       1000.0*SUM(fees+residual+upstream_loss)/NULLIF(SUM(class_arrivals),0) AS cost_per_1000_arrivals
FROM class_costs GROUP BY model_seed,model,pool_kind,ranking_scenario,policy,valuation;

CREATE TEMPORARY VIEW WM_REVIEW_MODEL_SWAPS AS
SELECT s.pool_kind,s.model_seed,s.new_model,s.baseline_model,s.ranking_scenario,s.policy,v.valuation,
       SUM(s.entered) AS slots_entered,SUM(s.displaced) AS slots_displaced,
       SUM(CASE WHEN s.label='Scratch' THEN s.entered ELSE 0 END) AS scratch_entered,
       SUM(CASE WHEN s.label='Scratch' THEN s.displaced ELSE 0 END) AS scratch_displaced,
       SUM((s.entered-s.displaced)*(v.avoided_fraction*v.loss-v.inspection_fee)) AS gain_units
FROM WM_REVIEW_SWAPS s JOIN WM_REVIEW_CLASS_COST v ON s.label=v.label
GROUP BY s.pool_kind,s.model_seed,s.new_model,s.baseline_model,s.ranking_scenario,s.policy,v.valuation;

-- Eight independent consistency checks. All must be PASS for this frozen run.
WITH matching AS (
  SELECT p.*,r.capacity, r.reviewed AS expected_reviews,
         r.cost_per_1000 AS expected_pool,r.cost_per_1000_simulated_arrivals AS expected_all
  FROM WM_REVIEW_REPRICED p JOIN WM_REVIEW_REPORTED r
   USING(model_seed,model,pool_kind,ranking_scenario,policy)
  WHERE p.ranking_scenario=p.valuation
), swap_check AS (
  SELECT s.gain_units,base.pool_cost-revised.pool_cost AS cost_delta
  FROM WM_REVIEW_MODEL_SWAPS s
  JOIN WM_REVIEW_REPRICED base ON base.model=s.baseline_model AND base.model_seed=s.model_seed
   AND base.valuation=s.valuation AND base.pool_kind=s.pool_kind
   AND base.policy=s.policy AND base.ranking_scenario=s.ranking_scenario
  JOIN WM_REVIEW_REPRICED revised ON revised.model=s.new_model AND revised.model_seed=s.model_seed
   AND revised.valuation=s.valuation AND revised.pool_kind=s.pool_kind
   AND revised.policy=s.policy AND revised.ranking_scenario=s.ranking_scenario
), checks(check_name,is_ok) AS (
  SELECT 'gate population', COUNT(*)=9 AND COUNT(DISTINCT label)=9
         AND SUM(all_audit)=11591 AND SUM(simulated_flagged)=3642
         AND SUM(ABS(all_audit-simulated_flagged-excluded_before_ranking))=0 FROM WM_REVIEW_GATE
  UNION ALL SELECT 'class coverage',COUNT(*)=2430 AND SUM(ABS(wafer_support-reviewed-not_reviewed))=0
                  AND MIN(reviewed)>=0 AND MIN(not_reviewed)>=0 FROM WM_REVIEW_CLASS
  UNION ALL SELECT 'unique class keys',COUNT(*)=2430 FROM (
    SELECT DISTINCT model_seed,model,pool_kind,ranking_scenario,policy,label FROM WM_REVIEW_CLASS)
  UNION ALL SELECT 'selected exchanges',COUNT(*)=648 AND SUM(ABS(entered-displaced-net_reviewed))=0
                  AND MIN(entered)>=0 AND MIN(displaced)>=0 FROM WM_REVIEW_SWAPS
  UNION ALL SELECT 'unique swap keys',COUNT(*)=648 FROM (
    SELECT DISTINCT pool_kind,model_seed,new_model,baseline_model,ranking_scenario,policy,label FROM WM_REVIEW_SWAPS)
  UNION ALL SELECT '270 frozen costs',COUNT(*)=270 AND MAX(ABS(cost_per_1000_pool-expected_pool))<0.000001
         AND MAX(ABS(cost_per_1000_arrivals-expected_all))<0.000001 FROM matching
  UNION ALL SELECT 'review budgets',COUNT(*)=270 AND MAX(reviewed-capacity)<=0
         AND MAX(ABS(reviewed-expected_reviews))=0 FROM matching
  UNION ALL SELECT 'paired gains from slot exchanges',COUNT(*)=288
         AND MAX(ABS(gain_units-cost_delta))<0.000001 FROM swap_check
)
SELECT check_name,CASE WHEN is_ok THEN 'PASS' ELSE 'FAIL' END AS status FROM checks ORDER BY check_name;

-- 270 original paired-cohort measurements and their simulated full-arrival costs.
SELECT * FROM WM_REVIEW_REPRICED WHERE ranking_scenario=valuation
ORDER BY pool_kind,ranking_scenario,model_seed,model,policy;

-- Same Colab scratch_5x lists under each alternative price schedule. No reranking.
SELECT m.model_seed,m.valuation,m.slots_entered,m.slots_displaced,m.scratch_entered,m.scratch_displaced,
       1000.0*m.gain_units/r.pool_size AS gain_per_1000_candidates,
       1000.0*m.gain_units/r.all_arrivals AS gain_per_1000_all_arrivals
FROM WM_REVIEW_MODEL_SWAPS m JOIN WM_REVIEW_REPRICED r
  ON r.model=m.new_model AND r.model_seed=m.model_seed AND r.pool_kind=m.pool_kind
 AND r.ranking_scenario=m.ranking_scenario AND r.policy=m.policy AND r.valuation=m.valuation
WHERE m.pool_kind='simulated_flagged_candidates' AND m.ranking_scenario='scratch_5x'
  AND m.policy='benefit_fixed' AND m.new_model='RF77' AND m.baseline_model='RF71'
ORDER BY m.model_seed,m.valuation;
