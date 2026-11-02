-- Read-only check of this Snowsight SQL session. No stored credentials or user tables.
SELECT 'WM811K_CONNECTED' AS connection_check,
       CURRENT_ROLE() AS active_role,
       CURRENT_WAREHOUSE() AS selected_warehouse,
       CURRENT_TIMESTAMP() AS checked_at;
