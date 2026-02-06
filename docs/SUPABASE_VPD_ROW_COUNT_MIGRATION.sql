-- 가상인구 DB 시도별 통계를 data_json 없이 빠르게 조회하기 위해 row_count 컬럼 추가
-- Supabase SQL Editor에서 실행하세요.

ALTER TABLE virtual_population_db ADD COLUMN IF NOT EXISTS row_count INTEGER NOT NULL DEFAULT 0;

-- (선택) 기존 행에 대해 data_json 배열 길이로 row_count 보정. 데이터가 많으면 타임아웃될 수 있으므로 필요 시 소량씩 나눠 실행
-- UPDATE virtual_population_db SET row_count = json_array_length(data_json::json) WHERE row_count = 0 AND data_json IS NOT NULL AND data_json != '';
