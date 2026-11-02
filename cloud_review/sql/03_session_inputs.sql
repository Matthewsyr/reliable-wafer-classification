-- Optional full-table workflow. Use an existing database, schema and personally
-- verified trial-only compute context. Keep the same Snowflake session through 04.
-- This file does not create, select, resize or resume a warehouse.
CREATE TEMPORARY STAGE WM_REVIEW_TRIAL_STAGE
  FILE_FORMAT=(TYPE=CSV SKIP_HEADER=1 FIELD_OPTIONALLY_ENCLOSED_BY='"'
               EMPTY_FIELD_AS_NULL=TRUE ERROR_ON_COLUMN_COUNT_MISMATCH=TRUE);

CREATE TEMPORARY TABLE WM_REVIEW_CLASS (
    model_seed INTEGER, model VARCHAR, pool_kind VARCHAR, ranking_scenario VARCHAR,
    policy VARCHAR, label VARCHAR, wafer_support INTEGER, reviewed INTEGER, not_reviewed INTEGER,
    review_coverage FLOAT, assumed_gross_avoided_loss FLOAT, remaining_loss FLOAT
);
CREATE TEMPORARY TABLE WM_REVIEW_GATE (
    label VARCHAR, all_audit INTEGER, simulated_flagged INTEGER,
    excluded_before_ranking INTEGER, coverage FLOAT
);
CREATE TEMPORARY TABLE WM_REVIEW_SWAPS (
    pool_kind VARCHAR, model_seed INTEGER, new_model VARCHAR, baseline_model VARCHAR,
    ranking_scenario VARCHAR, policy VARCHAR, label VARCHAR,
    entered INTEGER, displaced INTEGER, net_reviewed INTEGER, net_savings_units FLOAT
);
CREATE TEMPORARY TABLE WM_REVIEW_REPORTED (
    model_seed INTEGER, model VARCHAR, pool_kind VARCHAR, ranking_scenario VARCHAR,
    policy VARCHAR, wafer_rows INTEGER, lots INTEGER, reviewed INTEGER, capacity INTEGER,
    reviewed_per_1000 FLOAT, true_normal_reviewed INTEGER, true_defect_reviewed INTEGER,
    true_defect_not_reviewed INTEGER, reviewed_defect_fraction FLOAT, defect_coverage FLOAT,
    missed_loss_without_inspection FLOAT, gross_avoided_loss FLOAT,
    avoided_share_of_no_review_loss FLOAT, residual_loss FLOAT, review_fees FLOAT,
    review_harm FLOAT, total_cost FLOAT, cost_per_1000 FLOAT,
    savings_vs_no_review_per_1000 FLOAT, loss_from_excluded_candidates FLOAT,
    cost_per_1000_simulated_arrivals FLOAT
);
