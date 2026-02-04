-- =============================================================================
-- 권장 인덱스 생성 스크립트 (SQLite)
-- - IF NOT EXISTS: 이미 인덱스가 있어도 오류 없이 건너뜀 (안전 재실행 가능)
-- - 실행: sqlite3 data/app.db < docs/create_indexes.sql
-- =============================================================================

-- stats: sido_code='00' 일 때 WHERE is_active=1 / 전체 ORDER BY용
CREATE INDEX IF NOT EXISTS idx_stats_is_active_order
ON stats(is_active, sido_code, category, name);

-- stats: 특정 시도 조회 시 WHERE sido_code=? [AND is_active=1] ORDER BY category, name
CREATE INDEX IF NOT EXISTS idx_stats_sido_active_cat
ON stats(sido_code, is_active, category, name);

-- virtual_population_db: 시도별 목록 WHERE sido_code=? ORDER BY added_at
CREATE INDEX IF NOT EXISTS idx_vpd_sido_added
ON virtual_population_db(sido_code, added_at);

-- virtual_population_db: 중복 체크 WHERE sido_code=? AND record_timestamp=? AND record_excel_path=?
CREATE INDEX IF NOT EXISTS idx_vpd_sido_ts_path
ON virtual_population_db(sido_code, record_timestamp, record_excel_path);
