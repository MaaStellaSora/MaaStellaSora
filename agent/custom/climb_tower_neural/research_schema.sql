PRAGMA foreign_keys = ON;
PRAGMA user_version = 2;

CREATE TABLE IF NOT EXISTS presets (
    preset_key TEXT PRIMARY KEY,
    target_vector_json TEXT NOT NULL, target_vector_complete INTEGER NOT NULL,
    content_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, physical_game_id TEXT NOT NULL,
    segment_index INTEGER NOT NULL, segment_count INTEGER NOT NULL,
    preset_key TEXT REFERENCES presets(preset_key), language TEXT,
    lifecycle_status TEXT NOT NULL CHECK(lifecycle_status IN ('completed','interrupted','open')),
    termination_reason TEXT, termination_basis TEXT NOT NULL,
    full_run_start_observed INTEGER NOT NULL, completion_observed INTEGER NOT NULL,
    quality_status TEXT NOT NULL CHECK(quality_status IN ('valid','partial','suspect','invalid'))
);
CREATE INDEX IF NOT EXISTS runs_preset ON runs(preset_key);
CREATE INDEX IF NOT EXISTS runs_physical_game ON runs(physical_game_id);
CREATE TABLE IF NOT EXISTS records (
    record_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), sequence INTEGER,
    event_type TEXT NOT NULL, basis TEXT NOT NULL,
    quality_status TEXT NOT NULL CHECK(quality_status IN ('valid','partial','suspect','invalid')),
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS records_timeline ON records(run_id,sequence);
CREATE INDEX IF NOT EXISTS records_event ON records(event_type,run_id);
CREATE TABLE IF NOT EXISTS quality_issues (
    issue_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
    record_id TEXT REFERENCES records(record_id), field_path TEXT NOT NULL, code TEXT NOT NULL,
    severity TEXT NOT NULL CHECK(severity IN ('partial','suspect','invalid')),
    evidence_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS issues_location ON quality_issues(run_id,record_id,field_path);
CREATE INDEX IF NOT EXISTS issues_record ON quality_issues(record_id);


-- A missing/uncertain action affects its following interval. It never changes
-- the quality of an earlier, independently confirmed prefix.
-- A complete candidate page is independently observed even when the enclosing
-- purchase/refresh/take later fails. Keep interval quality for note deltas.
DROP VIEW IF EXISTS record_issue_links;
CREATE VIEW record_issue_links AS
SELECT r.record_id,q.issue_id,q.code,q.severity,q.field_path
FROM records r JOIN quality_issues q ON q.run_id=r.run_id
 AND (q.record_id=r.record_id OR
      (json_extract(q.evidence_json,'$.affected_from_sequence') IS NOT NULL
       AND NOT (r.event_type IN ('offer_observed','offer_reobserved')
                AND q.code IN ('unconfirmed_action_tail','action_not_confirmed')
                AND COALESCE(json_extract(r.payload_json,'$.source'),'') IN ('drink','specified_drink','enhance')
                AND COALESCE(json_extract(r.payload_json,'$.origin_operation_id') =
                             json_extract(q.evidence_json,'$.operation_id'),0))
       AND r.sequence >= json_extract(q.evidence_json,'$.affected_from_sequence')
       AND r.sequence <= json_extract(q.evidence_json,'$.affected_through_sequence')));

CREATE VIEW IF NOT EXISTS record_quality AS
SELECT r.*,
 CASE WHEN r.quality_status='invalid' OR EXISTS(SELECT 1 FROM record_issue_links q WHERE q.record_id=r.record_id AND q.severity='invalid') THEN 'invalid'
      WHEN r.quality_status='suspect'
        OR (r.event_type='note_delta' AND json_array_length(json_extract(r.payload_json,'$.event_components'))>1
            AND (json_extract(r.payload_json,'$.source_type')<>'combined_interval'
                 OR json_extract(r.payload_json,'$.independently_observed')=1
                 OR json_extract(r.payload_json,'$.analysis_eligible')=1
                 OR json_extract(r.payload_json,'$.probability_eligible')=1))
        OR EXISTS(SELECT 1 FROM record_issue_links q WHERE q.record_id=r.record_id AND q.severity='suspect') THEN 'suspect'
      WHEN r.quality_status='partial' OR EXISTS(SELECT 1 FROM record_issue_links q WHERE q.record_id=r.record_id AND q.severity='partial') THEN 'partial'
      ELSE 'valid' END AS effective_quality_status
FROM records r;

CREATE VIEW IF NOT EXISTS potential_candidates AS
SELECT r.run_id,r.record_id,r.sequence,r.effective_quality_status AS quality_status,
 json_extract(r.payload_json,'$.offer_id') AS offer_id,
 json_extract(r.payload_json,'$.reward_chain_id') AS reward_chain_id,
 json_extract(r.payload_json,'$.refresh_index') AS refresh_index,
 json_extract(r.payload_json,'$.same_page_read_index') AS same_page_read_index,
 json_extract(r.payload_json,'$.source') AS source,
 json_extract(r.payload_json,'$.source_type') AS source_type,
 CAST(c.key AS INTEGER) AS candidate_position,
 json_extract(c.value,'$.potential_id') AS potential_id,
 json_extract(c.value,'$.role') AS role,json_extract(c.value,'$.character_id') AS character_id,
 json_extract(c.value,'$.name') AS name,json_extract(c.value,'$.rarity') AS rarity,
 json_extract(c.value,'$.level_before') AS level_before,
 json_extract(c.value,'$.level_after') AS offered_level_after,
 json_extract(c.value,'$.level_delta') AS offered_level_delta,
 json_extract(c.value,'$.identity_inferred') AS identity_inferred,
 json_extract(c.value,'$.recognition_status') AS recognition_status,c.value AS candidate_json
FROM record_quality r,json_each(r.payload_json,'$.candidates') c WHERE r.event_type='offer_observed';

CREATE VIEW IF NOT EXISTS potential_draw_validity AS
SELECT r.run_id,r.record_id,json_extract(r.payload_json,'$.offer_id') AS offer_id,r.effective_quality_status AS quality_status,
 CASE WHEN json_extract(r.payload_json,'$.same_page_read_index') IN (0,1)
       AND json_extract(r.payload_json,'$.source') IS NOT NULL
       AND json_extract(r.payload_json,'$.source') NOT IN ('','unknown')
       AND json_extract(r.payload_json,'$.candidate_read_complete')=1
       AND json_extract(r.payload_json,'$.candidates_complete')=1
       AND json_extract(r.payload_json,'$.owned_complete')=1
       AND COALESCE(json_extract(r.payload_json,'$.recognition_consistent'),1)=1
       AND COALESCE(json_array_length(json_extract(r.payload_json,'$.inventory_conflicts')),0)=0
       AND r.effective_quality_status='valid'
      THEN 1 ELSE 0 END AS usable_for_draw_statistics
FROM record_quality r WHERE r.event_type='offer_observed';

CREATE VIEW IF NOT EXISTS shop_slots AS
SELECT r.run_id,r.record_id,r.sequence,r.effective_quality_status AS quality_status,
 json_extract(r.payload_json,'$.shop_visit_id') AS shop_visit_id,
 json_extract(r.payload_json,'$.stock_scan_id') AS stock_scan_id,
 json_extract(r.payload_json,'$.shelf_generation_id') AS shelf_generation_id,
 json_extract(r.payload_json,'$.previous_shelf_generation_id') AS previous_shelf_generation_id,
 json_extract(r.payload_json,'$.repeated_scan') AS repeated_scan,
 json_extract(s.value,'$.slot') AS slot,json_extract(s.value,'$.item_instance_id') AS item_instance_id,
 json_extract(s.value,'$.item.internal_name') AS item_type,
 json_extract(s.value,'$.item.quantity') AS quantity,
 json_extract(s.value,'$.effective_price') AS effective_price,s.value AS slot_json
FROM record_quality r,json_each(r.payload_json,'$.slots') s WHERE r.event_type='shop_observed';

CREATE VIEW IF NOT EXISTS event_options AS
SELECT r.run_id,r.record_id,r.sequence,r.event_type,
 json_extract(r.payload_json,'$.event_id') AS local_event_id,
 json_extract(r.payload_json,'$.question') AS question,
 CAST(c.key AS INTEGER) AS option_position,json_extract(c.value,'$.text') AS text,
 json_extract(c.value,'$.consequence') AS consequence,c.value AS option_json
FROM records r,json_each(r.payload_json,'$.choices') c
WHERE r.event_type IN ('event_menu_observed','event_menu_reobserved');

CREATE VIEW IF NOT EXISTS event_actions AS
SELECT r.run_id,r.record_id,r.sequence,json_extract(r.payload_json,'$.event_id') AS local_event_id,
 json_extract(r.payload_json,'$.selected_text') AS selected_text,
 json_extract(r.payload_json,'$.displayed_consequence') AS displayed_consequence,
 json_extract(r.payload_json,'$.selected_ui_index') AS selected_ui_index,
 json_extract(r.payload_json,'$.selection_status') AS selection_status,
 EXISTS(SELECT 1 FROM records click WHERE click.run_id=r.run_id
  AND click.event_type='event_choice_click_observed'
  AND json_extract(click.payload_json,'$.event_choice_record_id')=r.record_id) AS click_observed,
 json_extract(r.payload_json,'$.event_mapping') AS event_mapping_json
FROM records r WHERE r.event_type='event_choice';

CREATE VIEW IF NOT EXISTS note_inventory AS
WITH inventory_records AS (
 SELECT r.*,COALESCE(json_extract(r.payload_json,'$.scope_melodies'),
                     json_extract(r.payload_json,'$.tracked_melodies')) AS scope_melodies
 FROM record_quality r WHERE r.event_type='note_inventory_observed'
)
SELECT r.run_id,r.record_id,r.sequence,r.effective_quality_status AS quality_status,
 json_extract(r.payload_json,'$.page') AS page,json_extract(r.payload_json,'$.floor') AS floor,
 n.key AS melody,n.value AS current,
 json_extract(r.payload_json,'$.sources.'||n.key) AS reading_source,
 json_extract(r.payload_json,'$.complete') AS scope_complete,
 r.scope_melodies,
 EXISTS(SELECT 1 FROM json_each(r.scope_melodies) s WHERE s.value=n.key) AS in_scope,
 EXISTS(SELECT 1 FROM json_each(r.payload_json,'$.tracked_melodies') t WHERE t.value=n.key) AS tracked,
 json_extract(r.payload_json,'$.inventory_revision') AS inventory_revision
FROM inventory_records r,json_each(r.payload_json,'$.counts') n;

CREATE VIEW IF NOT EXISTS note_deltas AS
WITH interval_records AS (
 SELECT r.*,COALESCE(json_extract(r.payload_json,'$.scope_melodies'),
                     json_extract(r.payload_json,'$.tracked_melodies')) AS scope_melodies,
  COALESCE(json_array_length(json_extract(r.payload_json,'$.event_components')),0)>1 AS multiple_events,
  (json_extract(r.payload_json,'$.source_type')='combined_interval'
   OR COALESCE(json_array_length(json_extract(r.payload_json,'$.event_components')),0)>1) AS combined_interval
 FROM record_quality r WHERE r.event_type='note_delta'
)
SELECT r.run_id,r.record_id,r.sequence,r.effective_quality_status AS quality_status,
 -- Read old event-labelled intervals as combined without rewriting their payload.
 CASE WHEN r.multiple_events AND json_extract(r.payload_json,'$.source_type')<>'combined_interval'
      THEN (SELECT 'combined_'||group_concat(json_extract(c.value,'$.operation_id'),'_')
            FROM json_each(r.payload_json,'$.event_components') c)
      ELSE json_extract(r.payload_json,'$.operation_id') END AS operation_id,
 CASE WHEN r.combined_interval THEN 'combined_interval'
      ELSE json_extract(r.payload_json,'$.source_type') END AS source_type,
 CASE WHEN r.multiple_events AND json_extract(r.payload_json,'$.source_type')<>'combined_interval'
      THEN NULL ELSE json_extract(r.payload_json,'$.floor') END AS floor,
 n.key AS melody,n.value AS delta,
 json_extract(r.payload_json,'$.before.'||n.key) AS before_count,
 json_extract(r.payload_json,'$.after.'||n.key) AS after_count,
 json_extract(r.payload_json,'$.complete') AS complete,
 json_extract(r.payload_json,'$.total') AS total,
 CASE WHEN r.multiple_events AND json_extract(r.payload_json,'$.source_type')<>'combined_interval'
      THEN 'consecutive_events_without_intermediate_capture'
      ELSE json_extract(r.payload_json,'$.interval_basis') END AS interval_basis,
 CASE WHEN json_array_length(json_extract(r.payload_json,'$.mixed_sources'))>=2
      THEN json_extract(r.payload_json,'$.mixed_sources')
      WHEN r.multiple_events THEN json_extract(r.payload_json,'$.event_components')
      ELSE json_extract(r.payload_json,'$.mixed_sources') END AS component_sources,
 CASE WHEN r.combined_interval THEN 0
      ELSE COALESCE(json_extract(r.payload_json,'$.independently_observed'),1) END AS independently_observed,
 CASE WHEN r.combined_interval THEN 0
      ELSE COALESCE(json_extract(r.payload_json,'$.analysis_eligible'),1) END AS analysis_eligible,
 CASE WHEN r.combined_interval THEN 0
      ELSE json_extract(r.payload_json,'$.probability_eligible') END AS probability_eligible,
 CASE WHEN r.combined_interval THEN 'combined_sources_not_separable'
      ELSE json_extract(r.payload_json,'$.analysis_exclusion_reason') END AS analysis_exclusion_reason,
 r.scope_melodies,
 EXISTS(SELECT 1 FROM json_each(r.scope_melodies) s WHERE s.value=n.key) AS in_scope,
 CASE WHEN json_extract(r.payload_json,'$.complete')=1
       AND n.value IS NOT NULL
       AND EXISTS(SELECT 1 FROM json_each(r.scope_melodies) s WHERE s.value=n.key)
       AND json_extract(r.payload_json,'$.floor_complete')=1
       AND json_extract(r.payload_json,'$.floor') IS NOT NULL
       AND json_extract(r.payload_json,'$.source_type') NOT IN ('','unknown')
       AND NOT r.combined_interval
       AND COALESCE(json_extract(r.payload_json,'$.independently_observed'),1)=1
       AND COALESCE(json_extract(r.payload_json,'$.analysis_eligible'),1)=1
       AND json_extract(r.payload_json,'$.probability_eligible')=1
       AND r.effective_quality_status='valid'
      THEN 1 ELSE 0 END AS usable_for_interval_statistics
FROM interval_records r,json_each(r.payload_json,'$.delta') n;

CREATE VIEW IF NOT EXISTS purchases AS
SELECT r.run_id,r.record_id,r.sequence,r.effective_quality_status AS quality_status,
 json_extract(r.payload_json,'$.operation_id') AS operation_id,
 json_extract(r.payload_json,'$.item.internal_name') AS item_type,
 json_extract(r.payload_json,'$.item.price') AS price,
 json_extract(r.payload_json,'$.coins_before') AS coins_before,
 json_extract(r.payload_json,'$.coins_after') AS coins_after,
 json_extract(r.payload_json,'$.purchase_debit') AS purchase_debit,
 json_extract(r.payload_json,'$.pipeline_coins_after') AS pipeline_coins_after,
 json_extract(r.payload_json,'$.confirmation_status') AS confirmation_status,
 json_extract(r.payload_json,'$.wallet_before_observation_id') AS before_wallet_record_id,
 json_extract(r.payload_json,'$.wallet_observation_id') AS after_wallet_record_id,
 json_extract(r.payload_json,'$.pipeline_wallet_observation_id') AS pipeline_wallet_record_id
FROM record_quality r WHERE r.event_type='purchase_result';

CREATE VIEW IF NOT EXISTS preset_targets AS
SELECT p.preset_key,json_extract(role.value,'$.role') AS role,
 json_extract(role.value,'$.character_id') AS character_id,
 json_extract(t.value,'$.potential_id') AS potential_id,
 json_extract(t.value,'$.rarity_class') AS rarity,
 json_extract(t.value,'$.target_level') AS target_level,
 json_extract(t.value,'$.catalog_id_namespace') AS catalog_id_namespace,
 json_extract(t.value,'$.region_validation_status') AS region_validation_status,
 json_extract(t.value,'$.cn_server_id_equivalence_verified') AS cn_server_id_equivalence_verified,t.value AS target_json
FROM presets p,json_each(p.content_json,'$.party') role,json_each(role.value,'$.targets') t
UNION ALL
SELECT p.preset_key,json_extract(t.value,'$.role'),json_extract(t.value,'$.character_id'),
 json_extract(t.value,'$.potential_id'),json_extract(t.value,'$.rarity_class'),
 json_extract(t.value,'$.target_level'),json_extract(t.value,'$.catalog_id_namespace'),
 json_extract(t.value,'$.region_validation_status'),json_extract(t.value,'$.cn_server_id_equivalence_verified'),t.value
FROM presets p,json_each(p.content_json,'$.recognition_catalog') t;

CREATE VIEW IF NOT EXISTS settlement_ui_totals AS
SELECT r.run_id,r.record_id,CAST(c.key AS INTEGER)+1 AS ui_position,NULL AS role,
 c.value AS displayed_potential_total,'ui_position_order' AS order_basis,
 r.effective_quality_status AS quality_status,
 json_extract(r.payload_json,'$.potential_counts_complete') AS potential_counts_complete,
 json_extract(r.payload_json,'$.potential_counts_conflicts') AS potential_counts_conflicts
FROM record_quality r,json_each(r.payload_json,'$.potential_counts') c WHERE r.event_type='settlement_observed';
