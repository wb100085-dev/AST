-- ============================================
-- Supabase RLS 정책 적용 (테이블은 있는데 앱에서 조회가 안 될 때)
-- ============================================
-- Supabase 대시보드 → SQL Editor 에서 아래를 실행하세요.
-- 앱은 anon 키로 접속하므로, anon 역할에 SELECT/INSERT/UPDATE/DELETE 를 허용해야 합니다.
-- ============================================

-- 1) RLS가 켜져 있는지 확인 후, 정책 추가 (이미 있으면 오류 무시)
ALTER TABLE sido_axis_margin_stats ENABLE ROW LEVEL SECURITY;

-- 2) anon 역할으로 모든 행 읽기/쓰기 허용 (기존 정책과 이름이 겹치면 먼저 DROP)
DROP POLICY IF EXISTS "Allow all for anon" ON sido_axis_margin_stats;
CREATE POLICY "Allow all for anon" ON sido_axis_margin_stats
  FOR ALL
  USING (true)
  WITH CHECK (true);

-- 다른 테이블도 앱에서 안 보이면 동일하게 적용
-- ALTER TABLE stats ENABLE ROW LEVEL SECURITY;
-- DROP POLICY IF EXISTS "Allow all for anon" ON stats;
-- CREATE POLICY "Allow all for anon" ON stats FOR ALL USING (true) WITH CHECK (true);
-- (나머지 테이블도 필요 시 docs/SUPABASE_CREATE_TABLES.sql 참고)
