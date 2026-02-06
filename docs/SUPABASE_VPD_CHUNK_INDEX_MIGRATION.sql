-- 기존 virtual_population_db 테이블에 chunk_index 컬럼 추가 (한 번만 실행)
-- Supabase SQL Editor에서 실행하세요.
ALTER TABLE virtual_population_db ADD COLUMN IF NOT EXISTS chunk_index INTEGER NOT NULL DEFAULT 0;
