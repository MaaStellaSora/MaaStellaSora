PRAGMA foreign_keys = ON;
PRAGMA user_version = 1;

CREATE TABLE IF NOT EXISTS presets (
    preset_key TEXT PRIMARY KEY, source_preset_id TEXT,
    target_vector_json TEXT NOT NULL, target_vector_complete INTEGER NOT NULL,
    content_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, participant_id TEXT, session_id TEXT, task_id TEXT,
    preset_key TEXT REFERENCES presets(preset_key), language TEXT,
    started_at TEXT, ended_at TEXT,
    lifecycle_status TEXT NOT NULL CHECK(lifecycle_status IN ('completed','interrupted','open')),
    termination_reason TEXT, termination_basis TEXT NOT NULL,
    full_run_start_observed INTEGER NOT NULL, completion_observed INTEGER NOT NULL,
    quality_status TEXT NOT NULL CHECK(quality_status IN ('valid','partial','suspect','invalid')),
    collector_version TEXT, source_schema_version INTEGER,
    runtime_versions_json TEXT NOT NULL, model_versions_json TEXT NOT NULL,
    config_json TEXT NOT NULL, observation_kind TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_preset ON runs(preset_key);
CREATE TABLE IF NOT EXISTS records (
    record_id TEXT PRIMARY KEY, original_record_id TEXT NOT NULL, run_id TEXT NOT NULL REFERENCES runs(run_id),
    origin TEXT NOT NULL, sequence INTEGER, run_sequence INTEGER, timestamp TEXT,
    event_type TEXT NOT NULL, basis TEXT NOT NULL,
    quality_status TEXT NOT NULL CHECK(quality_status IN ('valid','partial','suspect','invalid')),
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS records_timeline ON records(run_id,sequence);
CREATE INDEX IF NOT EXISTS records_event ON records(event_type,run_id);
CREATE TABLE IF NOT EXISTS objects (
    object_id TEXT PRIMARY KEY, kind TEXT NOT NULL, basis TEXT NOT NULL, content_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS record_objects (
    record_id TEXT NOT NULL REFERENCES records(record_id), field_path TEXT NOT NULL,
    object_id TEXT NOT NULL REFERENCES objects(object_id), PRIMARY KEY(record_id,field_path,object_id)
);
CREATE INDEX IF NOT EXISTS record_objects_object ON record_objects(object_id);
CREATE TABLE IF NOT EXISTS readings (
    reading_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), original_record_id TEXT,
    timestamp TEXT, sequence INTEGER, node_name TEXT, method TEXT, read_id TEXT,
    coordinate_space TEXT NOT NULL, roi_json TEXT NOT NULL, status TEXT NOT NULL, content_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS readings_run ON readings(run_id);
CREATE TABLE IF NOT EXISTS record_readings (
    record_id TEXT NOT NULL REFERENCES records(record_id), reading_id TEXT NOT NULL REFERENCES readings(reading_id),
    field_path TEXT NOT NULL, PRIMARY KEY(record_id,reading_id,field_path)
);
CREATE INDEX IF NOT EXISTS record_readings_reading ON record_readings(reading_id);
CREATE TABLE IF NOT EXISTS quality_issues (
    issue_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
    record_id TEXT REFERENCES records(record_id), field_path TEXT NOT NULL, code TEXT NOT NULL,
    severity TEXT NOT NULL CHECK(severity IN ('partial','suspect','invalid')),
    message TEXT NOT NULL, evidence_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS issues_location ON quality_issues(run_id,record_id,field_path);
CREATE INDEX IF NOT EXISTS issues_record ON quality_issues(record_id);

DROP VIEW IF EXISTS record_issue_links;
DROP VIEW IF EXISTS record_quality;
DROP VIEW IF EXISTS potential_candidates;
DROP VIEW IF EXISTS potential_draw_validity;
DROP VIEW IF EXISTS shop_slots;
DROP VIEW IF EXISTS event_options;
DROP VIEW IF EXISTS event_actions;
DROP VIEW IF EXISTS event_option_evaluations;
DROP VIEW IF EXISTS note_inventory;
DROP VIEW IF EXISTS note_deltas;
DROP VIEW IF EXISTS decision_actions;
DROP VIEW IF EXISTS run_policies;
DROP VIEW IF EXISTS purchases;
DROP VIEW IF EXISTS preset_targets;
DROP VIEW IF EXISTS settlement_ui_totals;

-- A missing/uncertain action affects its following interval. It never changes
-- the quality of an earlier, independently confirmed prefix.
CREATE VIEW record_issue_links AS
SELECT r.record_id,q.issue_id,q.code,q.severity,q.field_path
FROM records r JOIN quality_issues q ON q.run_id=r.run_id
 AND (q.record_id=r.record_id OR
      (json_extract(q.evidence_json,'$.affected_from_sequence') IS NOT NULL
       AND r.sequence >= json_extract(q.evidence_json,'$.affected_from_sequence')
       AND r.sequence <= json_extract(q.evidence_json,'$.affected_through_sequence')));

CREATE VIEW record_quality AS
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
SELECT r.run_id,r.record_id,r.sequence,r.timestamp,r.effective_quality_status AS quality_status,
 json_extract(r.payload_json,'$.offer_id') AS offer_id,
 json_extract(r.payload_json,'$.reward_chain_id') AS reward_chain_id,
 json_extract(r.payload_json,'$.refresh_index') AS refresh_index,
 json_extract(r.payload_json,'$.same_page_read_index') AS same_page_read_index,
 json_extract(r.payload_json,'$.source') AS source,
 json_extract(r.payload_json,'$.source_type') AS source_type,
 json_extract(r.payload_json,'$.source_raw') AS source_raw,
 CAST(c.key AS INTEGER) AS candidate_position,
 json_extract(c.value,'$.potential_id') AS potential_id,
 json_extract(c.value,'$.role') AS role,json_extract(c.value,'$.character_id') AS character_id,
 json_extract(c.value,'$.name') AS name,json_extract(c.value,'$.canonical_rarity') AS rarity,
 json_extract(c.value,'$.target_level') AS target_level,
 json_extract(c.value,'$.level_before') AS level_before,
 json_extract(c.value,'$.offered_level_after') AS offered_level_after,
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
SELECT r.run_id,r.record_id,r.sequence,r.timestamp,r.effective_quality_status AS quality_status,
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
 json_extract(r.payload_json,'$.selected_box') AS selected_box,
 COALESCE(json_extract(r.payload_json,'$.selected_ui_index'),
          json_extract(r.payload_json,'$.model_mapping.selected_ui_index')) AS selected_ui_index,
 NULL AS game_event_id,NULL AS game_option_id,
 EXISTS(SELECT 1 FROM records click WHERE click.run_id=r.run_id AND click.event_type='event_choice_click_observed'
  AND (json_extract(click.payload_json,'$.event_choice_record_id')=r.record_id OR
       (json_type(click.payload_json,'$.event_choice_record_id') IS NULL
        AND json_extract(click.payload_json,'$.event_id')=json_extract(r.payload_json,'$.event_id')
        AND click.sequence>r.sequence
        AND NOT EXISTS(SELECT 1 FROM records newer WHERE newer.run_id=r.run_id AND newer.event_type='event_choice'
         AND json_extract(newer.payload_json,'$.event_id')=json_extract(r.payload_json,'$.event_id')
         AND newer.sequence>r.sequence AND newer.sequence<click.sequence)))) AS click_observed,
 json_extract(r.payload_json,'$.model_mapping') AS model_mapping_json,
 CASE WHEN json_extract(r.payload_json,'$.policy_mode')='native_fallback'
        OR json_extract(r.payload_json,'$.model_mapping.policy_mode')='native_fallback' THEN 'native_fallback'
      ELSE COALESCE(json_extract(r.payload_json,'$.model_mapping.policy_mode'),json_extract(r.payload_json,'$.policy_mode'),p.policy_mode) END AS policy_mode,
 CASE WHEN json_extract(r.payload_json,'$.policy_mode')='native_fallback'
        OR json_extract(r.payload_json,'$.model_mapping.policy_mode')='native_fallback' THEN NULL
      ELSE COALESCE(json_extract(r.payload_json,'$.model_mapping.model_id'),p.model_id) END AS model_id,
 CASE WHEN json_extract(r.payload_json,'$.policy_mode')='native_fallback'
        OR json_extract(r.payload_json,'$.model_mapping.policy_mode')='native_fallback' THEN NULL
      ELSE COALESCE(json_extract(r.payload_json,'$.model_mapping.event_policy_id'),p.event_policy_id) END AS event_policy_id,
 json_extract(r.payload_json,'$.model_mapping.decision_method') AS decision_method,
 json_extract(r.payload_json,'$.model_mapping.evaluation_method') AS evaluation_method,
 json_extract(r.payload_json,'$.model_mapping.evaluation_pool_order_basis') AS evaluation_pool_order_basis,
 json_extract(r.payload_json,'$.model_mapping.evaluation_samples') AS evaluation_samples,
 json_extract(r.payload_json,'$.model_mapping.evaluation_seed') AS evaluation_seed,
 json_extract(r.payload_json,'$.model_mapping.tie_break') AS tie_break,
 json_extract(r.payload_json,'$.model_mapping.full_visible_menu') AS full_visible_menu,
 json_extract(r.payload_json,'$.model_mapping.event') AS model_event_id,
 json_extract(r.payload_json,'$.model_mapping.option') AS model_option_id
FROM records r JOIN run_policies p ON p.run_id=r.run_id WHERE r.event_type='event_choice';

-- All visible candidates, including options outside the active model's domain.
CREATE VIEW event_option_evaluations AS
SELECT r.run_id,r.record_id,r.sequence,json_extract(r.payload_json,'$.event_id') AS local_event_id,
 CASE WHEN json_extract(r.payload_json,'$.policy_mode')='native_fallback'
        OR json_extract(r.payload_json,'$.model_mapping.policy_mode')='native_fallback' THEN 'native_fallback'
      ELSE COALESCE(json_extract(r.payload_json,'$.model_mapping.policy_mode'),json_extract(r.payload_json,'$.policy_mode'),p.policy_mode) END AS policy_mode,
 CASE WHEN json_extract(r.payload_json,'$.policy_mode')='native_fallback'
        OR json_extract(r.payload_json,'$.model_mapping.policy_mode')='native_fallback' THEN NULL
      ELSE COALESCE(json_extract(r.payload_json,'$.model_mapping.model_id'),p.model_id) END AS model_id,
 CASE WHEN json_extract(r.payload_json,'$.policy_mode')='native_fallback'
        OR json_extract(r.payload_json,'$.model_mapping.policy_mode')='native_fallback' THEN NULL
      ELSE COALESCE(json_extract(r.payload_json,'$.model_mapping.event_policy_id'),p.event_policy_id) END AS event_policy_id,
 json_extract(r.payload_json,'$.model_mapping.decision_method') AS decision_method,
 json_extract(r.payload_json,'$.model_mapping.evaluation_method') AS evaluation_method,
 json_extract(r.payload_json,'$.model_mapping.evaluation_pool_order_basis') AS evaluation_pool_order_basis,
 json_extract(r.payload_json,'$.model_mapping.evaluation_samples') AS evaluation_samples,
 json_extract(r.payload_json,'$.model_mapping.evaluation_seed') AS evaluation_seed,
 json_extract(r.payload_json,'$.model_mapping.tie_break') AS tie_break,
 json_extract(r.payload_json,'$.model_mapping.full_visible_menu') AS full_visible_menu,
 json_extract(v.value,'$.ui_index') AS ui_index,
 json_extract(v.value,'$.model_option_id') AS model_option_id,
 json_extract(v.value,'$.evaluation_status') AS evaluation_status,
 json_extract(v.value,'$.reason') AS reason,
 json_extract(v.value,'$.note_effect') AS note_effect_json,
 json_extract(v.value,'$.predicted_value') AS predicted_value,
 json_extract(v.value,'$.ui_index')=COALESCE(json_extract(r.payload_json,'$.selected_ui_index'),
          json_extract(r.payload_json,'$.model_mapping.selected_ui_index')) AS selected
FROM records r JOIN run_policies p ON p.run_id=r.run_id,
 json_each(r.payload_json,'$.model_mapping.visible_options') v WHERE r.event_type='event_choice';

CREATE VIEW IF NOT EXISTS note_inventory AS
WITH inventory_records AS (
 SELECT r.*,COALESCE(json_extract(r.payload_json,'$.scope_melodies'),
                     json_extract(r.payload_json,'$.tracked_melodies')) AS scope_melodies
 FROM record_quality r WHERE r.event_type='note_inventory_observed'
)
SELECT r.run_id,r.record_id,r.sequence,r.timestamp,r.effective_quality_status AS quality_status,
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
SELECT r.run_id,r.record_id,r.sequence,r.timestamp,r.effective_quality_status AS quality_status,
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

-- Initial model configuration stays separate from observed execution below.
CREATE VIEW run_policies AS
WITH execution_records AS (
 SELECT r.*,chosen.payload_json AS choice_json,
  -- Read the former route spelling without changing its original records.
  CASE WHEN json_extract(r.payload_json,'$.policy_mode')='m144' THEN 'model' ELSE
  COALESCE(json_extract(r.payload_json,'$.policy_mode'),
   CASE WHEN COALESCE(json_extract(u.model_versions_json,'$.model_id'),
                      json_extract(u.model_versions_json,'$.model_selection.model_id')) IS NOT NULL THEN 'model' END) END AS routing_policy
 FROM records r JOIN runs u USING(run_id)
 LEFT JOIN records chosen ON r.event_type='event_choice_click_observed' AND chosen.run_id=r.run_id
  AND chosen.record_id=COALESCE(json_extract(r.payload_json,'$.event_choice_record_id'),
   (SELECT c.record_id FROM records c WHERE c.run_id=r.run_id AND c.event_type='event_choice'
     AND json_extract(c.payload_json,'$.event_id')=json_extract(r.payload_json,'$.event_id')
     AND c.sequence<r.sequence ORDER BY c.sequence DESC LIMIT 1))
 WHERE r.event_type IN ('model_decision','selection_ui_transition_observed','event_choice_click_observed',
                        'purchase_result','shop_refresh_result','runtime_recovery')
), event_rules AS (
 SELECT *,CASE WHEN event_type='event_choice_click_observed' AND routing_policy='model'
   AND json_extract(choice_json,'$.model_mapping.decision_method') IN ('official_rule_fallback','official_answer_rule')
   AND json_extract(choice_json,'$.model_mapping.quiz_event_id') IS NOT NULL
   AND COALESCE(json_extract(choice_json,'$.model_mapping.full_visible_menu'),1)=1
   AND COALESCE(json_extract(choice_json,'$.model_mapping.model_history_incomplete'),0)=0
   AND NOT EXISTS(SELECT 1 FROM json_each(choice_json,'$.model_mapping.ocr_missing') WHERE value<>'coin')
  THEN 1 ELSE 0 END AS native_rule_action
 FROM execution_records
), classified AS (
 SELECT *,
  -- Earlier journals used one action for both a single page and a whole-run fallback.
  CASE WHEN event_type='runtime_recovery' AND json_extract(payload_json,'$.recovery_action')='use_native_fallback'
       AND COALESCE(json_extract(payload_json,'$.reason'),'') NOT IN ('route_position_unconfirmed','pool_order_unconfirmed')
       THEN 1 ELSE 0 END AS run_fallback,
  CASE WHEN event_type='event_choice_click_observed'
   OR (event_type IN ('selection_ui_transition_observed','purchase_result','shop_refresh_result')
       AND json_extract(payload_json,'$.operation_confirmed')=1) THEN 1 ELSE 0 END AS confirmed_action,
  CASE WHEN event_type='runtime_recovery' AND json_extract(payload_json,'$.recovery_action')='leave_shop'
             AND json_extract(payload_json,'$.continued')=1 THEN 1
       WHEN event_type='event_choice_click_observed' AND native_rule_action=0
        AND (routing_policy='native_fallback' OR json_extract(choice_json,'$.model_mapping.policy_mode')='native_fallback'
             OR json_extract(choice_json,'$.model_mapping.decision_method')='official_rule_fallback') THEN 1
       WHEN event_type IN ('selection_ui_transition_observed','purchase_result','shop_refresh_result')
             AND json_extract(payload_json,'$.operation_confirmed')=1 AND routing_policy='native_fallback' THEN 1
       ELSE 0 END AS fallback_action
 FROM event_rules
), execution_summary AS (
 SELECT run_id,SUM(event_type='model_decision') AS model_decision_count,
  SUM(confirmed_action=1 AND routing_policy='model' AND fallback_action=0) AS model_action_count,
  SUM(native_rule_action) AS native_rule_action_count,SUM(fallback_action) AS fallback_action_count,
  MAX(run_fallback) AS permanent_fallback_entered,
  MIN(CASE WHEN run_fallback THEN timestamp END) AS permanent_fallback_started_at
 FROM classified GROUP BY run_id
)
SELECT u.run_id,
 json_extract(model_versions_json,'$.policy_mode') AS policy_mode,
 COALESCE(json_extract(model_versions_json,'$.model_id'),
          json_extract(model_versions_json,'$.model_selection.model_id')) AS model_id,
 json_extract(model_versions_json,'$.event_policy_id') AS event_policy_id,
 json_extract(model_versions_json,'$.event_evaluation_method') AS event_evaluation_method,
 json_extract(model_versions_json,'$.checkpoint_sha256') AS checkpoint_sha256,
 json_extract(model_versions_json,'$.policy') AS action_selection,
 json_extract(model_versions_json,'$.objective') AS objective_json,
 json_extract(model_versions_json,'$.value_semantics') AS value_semantics,
 json_extract(model_versions_json,'$.state_schema_version') AS state_schema_version,
 json_extract(model_versions_json,'$.action_schema_version') AS action_schema_version,
 json_extract(model_versions_json,'$.feature_version') AS feature_version,
 json_extract(model_versions_json,'$.network_version') AS network_version,
 json_extract(model_versions_json,'$.rules_version') AS model_rules_version,
 json_extract(runtime_versions_json,'$.note_rules_sha256') AS observation_note_rules_sha256,
 COALESCE(e.model_decision_count,0) AS model_decision_count,
 COALESCE(e.model_action_count,0) AS model_action_count,
 COALESCE(e.native_rule_action_count,0) AS native_rule_action_count,
 COALESCE(e.fallback_action_count,0) AS fallback_action_count,
 COALESCE(e.permanent_fallback_entered,0) AS permanent_fallback_entered,
 e.permanent_fallback_started_at,
 CASE WHEN COALESCE(e.model_decision_count,0)+COALESCE(e.model_action_count,0)+COALESCE(e.fallback_action_count,0)=0 THEN 'unobserved'
      WHEN (e.fallback_action_count>0 OR e.permanent_fallback_entered=1)
       AND (e.model_decision_count>0 OR e.model_action_count>0) THEN 'mixed'
      WHEN e.fallback_action_count>0 THEN 'native_fallback'
      ELSE 'model' END AS execution_mode
FROM runs u LEFT JOIN execution_summary e ON e.run_id=u.run_id;

CREATE VIEW IF NOT EXISTS decision_actions AS
SELECT r.run_id,r.record_id,r.sequence,'model_inference' AS basis,
 p.policy_mode,p.model_id,p.objective_json,p.value_semantics,
 p.event_policy_id,
 json_extract(r.payload_json,'$.decision.evaluation_method') AS evaluation_method,
 json_extract(r.payload_json,'$.decision.evaluation_pool_order_basis') AS evaluation_pool_order_basis,
 json_extract(r.payload_json,'$.decision.evaluation_samples') AS evaluation_samples,
 json_extract(r.payload_json,'$.decision.evaluation_seed') AS evaluation_seed,
 json_extract(r.payload_json,'$.decision.tie_break') AS tie_break,
 json_extract(r.payload_json,'$.decision.decision_index') AS decision_index,
 json_extract(v.value,'$.action') AS action,
 json_extract(v.value,'$.action_record') AS action_record_json,
 COALESCE(json_extract(v.value,'$.predicted_value'),
          json_extract(v.value,'$.expected_remaining_ordinary_return')) AS predicted_value,
 json_extract(v.value,'$.expected_remaining_ordinary_return') AS predicted_remaining_return,
 json_extract(r.payload_json,'$.decision.action')=json_extract(v.value,'$.action') AS selected,
 json_extract(r.payload_json,'$.decision.public_state_object_id') AS model_state_object_id
FROM records r JOIN run_policies p ON p.run_id=r.run_id,
 json_each(r.payload_json,'$.decision.values') v WHERE r.event_type='model_decision';

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
