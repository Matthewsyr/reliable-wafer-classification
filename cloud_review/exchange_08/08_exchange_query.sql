-- Read-only historical exchange audit. WM08_SELECTED_UNION contains each wafer selected
-- by either frozen policy, once per seed. The private generator can inline this source.
-- A = RF77 any-defect probability; B = RF77 positive benefit with Scratch loss = 5.
WITH ledger AS (
/* BEGIN INPUT */
    SELECT seed, source_row, queue, truth, train_normal_count, p_scratch, p_any, in_a, in_b
    FROM WM08_SELECTED_UNION
/* END INPUT */
), queue_counts AS (
    SELECT seed, queue, SUM(in_a) AS a, SUM(in_b) AS b
    FROM ledger GROUP BY seed, queue
), queue_audit AS (
    SELECT seed, COUNT(*) AS queues_checked,
           SUM(IFF(queue NOT BETWEEN 0 AND 49 OR a>IFF(queue=49,45,200)
                   OR b>IFF(queue=49,45,200),1,0)) AS queue_overages
    FROM queue_counts GROUP BY seed
), membership AS (
    SELECT seed, COUNT(*) AS union_rows, COUNT(DISTINCT source_row) AS union_ids,
           SUM(in_a) AS a_reviews, SUM(in_b) AS b_reviews,
           SUM(IFF(in_a=1 AND in_b=1,1,0)) AS common_reviews
    FROM ledger GROUP BY seed
), exclusive AS (
    SELECT seed, IFF(in_a=1,'A_only','B_only') AS side, truth, p_scratch, p_any,
           CASE WHEN train_normal_count=0 THEN '0'
                WHEN train_normal_count<25 THEN '1-24' ELSE '25+' END AS size_support,
           CASE WHEN p_scratch<0.01 THEN '0-.01' WHEN p_scratch<0.05 THEN '.01-.05'
                WHEN p_scratch<0.1 THEN '.05-.10' WHEN p_scratch<0.25 THEN '.10-.25'
                WHEN p_scratch<0.5 THEN '.25-.50' WHEN p_scratch<0.75 THEN '.50-.75'
                ELSE '.75-1' END AS scratch_bin
    FROM ledger WHERE in_a<>in_b
), grouped AS (
    SELECT seed, side,
           CASE WHEN GROUPING(size_support)=1 THEN 'total'
                WHEN GROUPING(truth)=0 THEN 'class'
                WHEN GROUPING(scratch_bin)=0 THEN 'reliability' ELSE 'size' END AS grain,
           IFF(GROUPING(size_support)=1,'all',size_support) AS size_band,
           IFF(GROUPING(truth)=1,'all',truth) AS true_class,
           IFF(GROUPING(scratch_bin)=1,'all',scratch_bin) AS probability_bin,
           COUNT(*) AS n, SUM(IFF(truth='Scratch',1,0)) AS scratch_n,
           SUM(IFF(truth='none',1,0)) AS normal_n,
           SUM(p_scratch) AS predicted_scratch_n, SUM(p_any) AS predicted_defect_n
    FROM exclusive
    GROUP BY GROUPING SETS ((seed,side,size_support,truth), (seed,side,size_support,scratch_bin),
                            (seed,side,size_support), (seed,side))
)
SELECT g.seed, g.grain, g.side, g.size_band, g.true_class, g.probability_bin,
       g.n, g.scratch_n, g.normal_n, g.n-g.normal_n-g.scratch_n AS other_defect_n,
       g.predicted_scratch_n, g.predicted_defect_n,
       g.predicted_scratch_n/g.n AS mean_p_scratch,
       g.scratch_n::DOUBLE/g.n AS observed_scratch_rate,
       (g.predicted_scratch_n-g.scratch_n)/g.n AS scratch_overprediction,
       0.9*(g.n-g.normal_n+4*g.scratch_n)-0.1*g.n AS realized_net_benefit,
       0.9*(g.predicted_defect_n+4*g.predicted_scratch_n)-0.1*g.n AS predicted_net_benefit,
       m.union_rows, m.union_ids, m.a_reviews, m.b_reviews, m.common_reviews,
       q.queues_checked, q.queue_overages
FROM grouped g JOIN membership m ON g.seed=m.seed JOIN queue_audit q ON g.seed=q.seed
ORDER BY g.seed, g.grain, g.side, g.size_band, g.true_class, g.probability_bin;
