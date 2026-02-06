-- 페르소나·현시대 반영만 갱신하는 RPC (저장 속도 향상: 전체 data_json 대신 델타만 전송)
-- Supabase SQL Editor에서 한 번 실행하세요.
-- 실행 후 앱의 "기록을 DB에 추가" 시 전체 재저장 대신 페르소나/현시대 반영만 전송합니다.
-- RPC 미설치 시 자동으로 기존 방식(전체 갱신)으로 저장됩니다.
--
-- 동작: 이미 DB에 페르소나·현시대 반영이 있는 행은 건너뛰고, 비어 있는 행(새로 추가된 부분)만 갱신합니다.

CREATE OR REPLACE FUNCTION update_vpd_chunk_persona_reflection(p_chunk_id bigint, p_updates jsonb)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  current_json text;
  current_jsonb jsonb;
  arr jsonb;
  i int;
  upd jsonb;
  idx int;
  p_val text;
  r_val text;
  elem jsonb;
  p_existing text;
  r_existing text;
  need_update boolean;
BEGIN
  SELECT data_json INTO current_json FROM virtual_population_db WHERE id = p_chunk_id;
  IF current_json IS NULL OR current_json = '' THEN
    RETURN;
  END IF;
  current_jsonb := current_json::jsonb;
  IF jsonb_typeof(current_jsonb) != 'array' THEN
    RETURN;
  END IF;
  arr := current_jsonb;
  FOR i IN 0 .. jsonb_array_length(p_updates) - 1 LOOP
    upd := p_updates->i;
    idx := (upd->>'i')::int;
    p_val := upd->>'p';
    r_val := upd->>'r';
    IF idx >= 0 AND idx < jsonb_array_length(arr) THEN
      elem := arr->idx;
      IF elem IS NULL THEN
        elem := '{}'::jsonb;
      END IF;
      -- 이미 페르소나·현시대 반영이 있으면 건너뛰고, 없는(비어 있는) 행만 갱신
      p_existing := trim(COALESCE(elem->>'페르소나', ''));
      r_existing := trim(COALESCE(elem->>'현시대 반영', ''));
      need_update := (p_existing = '' OR r_existing = '');
      IF need_update THEN
        elem := jsonb_set(
          jsonb_set(
            COALESCE(elem, '{}'::jsonb),
            '{페르소나}', to_jsonb(COALESCE(p_val, '')::text)
          ),
          '{현시대 반영}', to_jsonb(COALESCE(r_val, '')::text)
        );
        arr := jsonb_set(arr, ARRAY[idx::text], elem);
      END IF;
    END IF;
  END LOOP;
  UPDATE virtual_population_db
  SET data_json = arr::text, row_count = jsonb_array_length(arr)
  WHERE id = p_chunk_id;
END;
$$;
