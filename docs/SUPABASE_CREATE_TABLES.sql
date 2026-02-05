-- ============================================
-- Supabase(PostgreSQL) 테이블 생성 스크립트
-- ============================================
-- Supabase 대시보드 > SQL Editor에서 이 스크립트를 실행하세요.
-- 실행 후 앱에서 st.secrets에 SUPABASE_URL, SUPABASE_KEY를 설정하면 DB 연동이 됩니다.
--
-- 코드 기준 컬럼 매핑 (core/db.py):
--   stats                  : id, sido_code, category, name, url, is_active, created_at, updated_at
--   sido_axis_margin_stats : id, sido_code, axis_key, stat_id, created_at, updated_at
--   sido_templates         : sido_code(PK), filename, file_blob(base64 TEXT), created_at, updated_at
--   virtual_population_db   : id, sido_code, sido_name, record_timestamp, record_excel_path, data_json(JSON 문자열), added_at
-- ============================================

-- 1. sido_profiles (예비/확장용)
CREATE TABLE IF NOT EXISTS sido_profiles (
    id BIGSERIAL PRIMARY KEY,
    sido_code TEXT NOT NULL UNIQUE,
    sido_name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- 2. sido_axis_weights (예비/확장용)
CREATE TABLE IF NOT EXISTS sido_axis_weights (
    id BIGSERIAL PRIMARY KEY,
    sido_code TEXT NOT NULL UNIQUE,
    w_sigungu REAL NOT NULL DEFAULT 1.0,
    w_gender REAL NOT NULL DEFAULT 1.0,
    w_age REAL NOT NULL DEFAULT 1.0,
    w_econ REAL NOT NULL DEFAULT 1.0,
    w_income REAL NOT NULL DEFAULT 1.0,
    w_edu REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- 3. stats (시도별 통계 목록: db_list_stats_by_sido, db_upsert_stat, db_delete_stat_by_id, db_update_stat_by_id)
CREATE TABLE IF NOT EXISTS stats (
    id BIGSERIAL PRIMARY KEY,
    sido_code TEXT NOT NULL,
    category TEXT NOT NULL,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(sido_code, category, name)
);
CREATE INDEX IF NOT EXISTS idx_stats_is_active_order ON stats(is_active, sido_code, category, name);
CREATE INDEX IF NOT EXISTS idx_stats_sido_active_cat ON stats(sido_code, is_active, category, name);

-- 4. sido_weight_stats (예비/확장용)
CREATE TABLE IF NOT EXISTS sido_weight_stats (
    id BIGSERIAL PRIMARY KEY,
    sido_code TEXT NOT NULL,
    stat_id BIGINT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(sido_code, stat_id)
);

-- 5. sido_axis_margin_stats (축별 마진 통계: db_get_axis_margin_stats, db_upsert_axis_margin_stat)
CREATE TABLE IF NOT EXISTS sido_axis_margin_stats (
    id BIGSERIAL PRIMARY KEY,
    sido_code TEXT NOT NULL,
    axis_key TEXT NOT NULL,
    stat_id BIGINT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(sido_code, axis_key)
);

-- 6. sido_templates (시도별 템플릿 파일; file_blob는 base64 인코딩된 TEXT)
CREATE TABLE IF NOT EXISTS sido_templates (
    sido_code TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_blob TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- 7. virtual_population_db (가상인구 DB: 2차 대입결과별 data_json 저장)
CREATE TABLE IF NOT EXISTS virtual_population_db (
    id BIGSERIAL PRIMARY KEY,
    sido_code TEXT NOT NULL,
    sido_name TEXT NOT NULL,
    record_timestamp TEXT NOT NULL,
    record_excel_path TEXT NOT NULL,
    data_json TEXT NOT NULL,
    added_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_vpd_sido_added ON virtual_population_db(sido_code, added_at);
CREATE INDEX IF NOT EXISTS idx_vpd_sido_ts_path ON virtual_population_db(sido_code, record_timestamp, record_excel_path);

-- RLS 활성화 후 anon 역할로 읽기/쓰기 허용 (개발·단일 사용자용)
-- 이미 정책이 있으면 DROP 후 다시 생성 (스크립트 재실행 가능)
ALTER TABLE sido_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE sido_axis_weights ENABLE ROW LEVEL SECURITY;
ALTER TABLE stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE sido_weight_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE sido_axis_margin_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE sido_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE virtual_population_db ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow all for anon" ON sido_profiles;
CREATE POLICY "Allow all for anon" ON sido_profiles FOR ALL USING (true) WITH CHECK (true);
DROP POLICY IF EXISTS "Allow all for anon" ON sido_axis_weights;
CREATE POLICY "Allow all for anon" ON sido_axis_weights FOR ALL USING (true) WITH CHECK (true);
DROP POLICY IF EXISTS "Allow all for anon" ON stats;
CREATE POLICY "Allow all for anon" ON stats FOR ALL USING (true) WITH CHECK (true);
DROP POLICY IF EXISTS "Allow all for anon" ON sido_weight_stats;
CREATE POLICY "Allow all for anon" ON sido_weight_stats FOR ALL USING (true) WITH CHECK (true);
DROP POLICY IF EXISTS "Allow all for anon" ON sido_axis_margin_stats;
CREATE POLICY "Allow all for anon" ON sido_axis_margin_stats FOR ALL USING (true) WITH CHECK (true);
DROP POLICY IF EXISTS "Allow all for anon" ON sido_templates;
CREATE POLICY "Allow all for anon" ON sido_templates FOR ALL USING (true) WITH CHECK (true);
DROP POLICY IF EXISTS "Allow all for anon" ON virtual_population_db;
CREATE POLICY "Allow all for anon" ON virtual_population_db FOR ALL USING (true) WITH CHECK (true);
