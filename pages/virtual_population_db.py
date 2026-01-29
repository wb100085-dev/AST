"""
가상인구 DB 페이지
2차 대입결과 누적 관리 및 페르소나 생성
"""
# Streamlit 자동 페이지 감지 방지: 이 파일은 app.py에서 직접 호출됨
import streamlit as st
import pandas as pd
import os
import time
from io import BytesIO
from utils.step2_records import list_step2_records
from utils.gemini_client import GeminiClient


def page_virtual_population_db():
    """가상인구 DB: 2차 대입결과 누적 관리"""
    # st를 전역 변수로 명시적으로 참조 (UnboundLocalError 방지)
    global st
    
    # app.py에서 필요한 함수 및 상수 import
    import sys
    import importlib
    app_module = sys.modules.get('app')
    if app_module is None:
        import app
        app_module = app
    
    SIDO_MASTER = app_module.SIDO_MASTER
    SIDO_LABEL_TO_CODE = app_module.SIDO_LABEL_TO_CODE
    SIDO_CODE_TO_NAME = app_module.SIDO_CODE_TO_NAME
    db_conn = app_module.db_conn
    
    st.header("가상인구 DB")
    
    st.markdown("---")
    
    # 지역별 선택
    sido_options = [f"{s['sido_name']} ({s['sido_code']})" for s in SIDO_MASTER]
    selected_sido_label = st.selectbox(
        "지역 선택",
        options=sido_options,
        index=sido_options.index(st.session_state.get("selected_sido_label", "경상북도 (37)")) if st.session_state.get("selected_sido_label", "경상북도 (37)") in sido_options else 0,
        key="vdb_sido_select"
    )
    selected_sido_code = SIDO_LABEL_TO_CODE.get(selected_sido_label, "37")
    selected_sido_name = SIDO_CODE_TO_NAME.get(selected_sido_code, "경상북도")
    
    st.markdown("---")
    
    # 2분할 레이아웃
    col_left, col_right = st.columns(2)
    
    with col_left:
        # 2차 대입결과 목록 불러오기
        st.subheader("2차 대입결과 목록")
        records = list_step2_records()
        
        # 선택한 지역의 기록만 필터링
        filtered_records = [r for r in records if r.get("sido_code") == selected_sido_code]
        
        if not filtered_records:
            st.info(f"{selected_sido_name} 지역의 2차 대입결과가 없습니다.")
        else:
            st.caption(f"총 {len(filtered_records)}건 (날짜·시간순)")
            
            # 선택된 기록들을 저장할 세션 상태 초기화
            if "vdb_selected_records" not in st.session_state:
                st.session_state.vdb_selected_records = []
            
            # 기록 목록 표시 및 선택
            selected_record_keys = []
            for idx, r in enumerate(filtered_records):
                ts = r.get("timestamp", "")
                sido_name = r.get("sido_name", "")
                rows = r.get("rows", 0)
                cols = r.get("columns_count", 0)
                excel_path = r.get("excel_path", "")
                added = r.get("added_columns", [])
                
                record_key = f"{ts}_{selected_sido_code}_{idx}"
                is_selected = st.checkbox(
                    f"{ts} | {sido_name} | {rows}명 | 추가 컬럼 {len(added)}개",
                    key=f"vdb_record_{record_key}",
                    value=record_key in st.session_state.vdb_selected_records
                )
                
                if is_selected and record_key not in st.session_state.vdb_selected_records:
                    st.session_state.vdb_selected_records.append(record_key)
                elif not is_selected and record_key in st.session_state.vdb_selected_records:
                    st.session_state.vdb_selected_records.remove(record_key)
                
                if is_selected:
                    selected_record_keys.append({
                        "key": record_key,
                        "record": r,
                        "excel_path": excel_path
                    })
            
            st.markdown("---")
            
            # 초기화 및 페르소나 버튼
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("초기화", type="secondary", use_container_width=True, key="vdb_reset"):
                    conn = db_conn()
                    cur = conn.cursor()
                    cur.execute("DELETE FROM virtual_population_db WHERE sido_code = ?", (selected_sido_code,))
                    conn.commit()
                    conn.close()
                    st.session_state.vdb_selected_records = []
                    st.success(f"{selected_sido_name} 지역의 가상인구 DB가 초기화되었습니다.")
                    st.rerun()
            
            with col_btn2:
                # 페르소나 생성 중지 플래그 초기화
                if "persona_generation_stop" not in st.session_state:
                    st.session_state.persona_generation_stop = False
                if "persona_generation_running" not in st.session_state:
                    st.session_state.persona_generation_running = False
                
                if not st.session_state.persona_generation_running:
                    if st.button("페르소나", type="primary", use_container_width=True, key="vdb_persona"):
                        st.session_state.persona_generation_running = True
                        st.session_state.persona_generation_stop = False
                        st.rerun()
                else:
                    # 생성 중일 때 중지 버튼 표시
                    if st.button("중지", type="secondary", use_container_width=True, key="vdb_persona_stop"):
                        st.session_state.persona_generation_stop = True
                        st.rerun()
                
                # 페르소나 생성 로직 실행
                if st.session_state.persona_generation_running and not st.session_state.persona_generation_stop:
                    # 페르소나 생성 기능
                    conn = db_conn()
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT id, data_json FROM virtual_population_db WHERE sido_code = ? ORDER BY added_at",
                        (selected_sido_code,)
                    )
                    db_rows = cur.fetchall()
                    
                    if not db_rows:
                        st.warning("가상인구 DB에 데이터가 없습니다. 먼저 데이터를 추가해주세요.")
                        st.session_state.persona_generation_running = False
                    else:
                        # 페르소나가 비어있는 행만 찾기
                        all_dfs = []
                        rows_to_update = []
                        
                        for row in db_rows:
                            try:
                                record_id = row[0]
                                data_json = row[1]
                                df = pd.read_json(data_json, orient="records")
                                
                                # 식별NO 제거
                                if "식별NO" in df.columns:
                                    df = df.drop(columns=["식별NO"])
                                
                                # 페르소나, 현시대 반영 컬럼 추가
                                if "페르소나" not in df.columns:
                                    df["페르소나"] = ""
                                if "현시대 반영" not in df.columns:
                                    df["현시대 반영"] = ""
                                
                                # 페르소나가 비어있는 행 찾기
                                empty_persona_mask = (df["페르소나"].isna()) | (df["페르소나"].astype(str).str.strip() == "")
                                
                                if empty_persona_mask.sum() > 0:
                                    rows_to_update.append({
                                        "id": record_id,
                                        "df": df,
                                        "empty_mask": empty_persona_mask
                                    })
                            except Exception as e:
                                st.warning(f"데이터 로드 실패: {e}")
                                continue
                        
                        if not rows_to_update:
                            st.info("모든 가상인물의 페르소나가 이미 생성되어 있습니다.")
                            st.session_state.persona_generation_running = False
                        else:
                            # Gemini 클라이언트 초기화
                            try:
                                # API 키 확인 (Streamlit Secrets 우선, 없으면 utils 파일)
                                api_key_found = False
                                api_key_source = ""
                                
                                # 1. Streamlit Secrets 확인 (배포 환경)
                                try:
                                    if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
                                        api_key_found = True
                                        api_key_source = "Streamlit Secrets"
                                except:
                                    pass
                                
                                # 2. utils 파일 확인 (로컬 환경)
                                if not api_key_found:
                                    try:
                                        from utils.gemini_key import GEMINI_API_KEY
                                        if GEMINI_API_KEY and GEMINI_API_KEY != "" and "여기에_" not in GEMINI_API_KEY:
                                            api_key_found = True
                                            api_key_source = "utils/gemini_key.py"
                                    except:
                                        pass
                                
                                # 3. 환경변수 확인
                                if not api_key_found:
                                    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                                        api_key_found = True
                                        api_key_source = "환경변수"
                                
                                if not api_key_found:
                                    # 배포 환경인지 확인
                                    is_cloud = False
                                    try:
                                        if hasattr(st, 'secrets'):
                                            is_cloud = True
                                    except:
                                        pass
                                    
                                    if is_cloud:
                                        st.error("⚠️ **API 키가 설정되지 않았습니다**")
                                        st.error("배포 환경에서 API 키를 설정해야 합니다:")
                                        st.markdown("""
                                        1. **Streamlit Cloud** 대시보드 접속
                                        2. 앱 선택 → **Settings** → **Secrets**
                                        3. 다음 형식으로 추가:
                                        ```
                                        GEMINI_API_KEY = "your-api-key-here"
                                        ```
                                        4. **Google AI Studio**에서 API 키 발급: https://aistudio.google.com/apikey
                                        5. 저장 후 앱이 자동으로 재배포됩니다
                                        """)
                                    else:
                                        st.error("⚠️ **API 키가 설정되지 않았습니다**")
                                        st.error("로컬 환경에서 다음 중 하나를 설정해주세요:")
                                        st.markdown("""
                                        1. `utils/gemini_key.py` 파일의 `GEMINI_API_KEY` 설정
                                        2. 환경변수 `GEMINI_API_KEY` 설정
                                        3. **Google AI Studio**에서 API 키 발급: https://aistudio.google.com/apikey
                                        """)
                                    
                                    st.session_state.persona_generation_running = False
                                    st.rerun()
                                
                                gemini_client = GeminiClient()
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # 모든 빈 페르소나 행을 수집 (record별로 그룹화)
                                all_empty_rows = []
                                for item in rows_to_update:
                                    df = item["df"]
                                    empty_mask = item["empty_mask"]
                                    record_id = item["id"]
                                    empty_rows = df[empty_mask]
                                    
                                    for idx, row in empty_rows.iterrows():
                                        all_empty_rows.append({
                                            "record_id": record_id,
                                            "df": df,
                                            "row_idx": idx,
                                            "row_data": row
                                        })
                                
                                total_rows = len(all_empty_rows)
                                processed = 0
                                batch_size = 100  # 100명 단위로 처리
                                
                                # 100명씩 배치로 처리
                                for batch_start in range(0, total_rows, batch_size):
                                    if st.session_state.persona_generation_stop:
                                        # 중지 시 현재까지 처리된 내용 저장
                                        break
                                    
                                    batch_end = min(batch_start + batch_size, total_rows)
                                    batch_rows = all_empty_rows[batch_start:batch_end]
                                    
                                    # 배치 내 모든 행의 정보 수집
                                    batch_prompts = []
                                    batch_indices = []
                                    
                                    for batch_item in batch_rows:
                                        row = batch_item["row_data"]
                                        # 가상인물의 특성 정보 수집
                                        persona_info = {}
                                        key_columns = ["거주지역", "성별", "연령", "경제활동", "교육정도", "월평균소득"]
                                        for col in key_columns:
                                            if col in row and pd.notna(row[col]):
                                                persona_info[col] = str(row[col])
                                        
                                        # 추가 특성 정보 (있는 경우)
                                        additional_cols = ["자녀유무", "배우자의 경제활동 상태", "주택점유형태", 
                                                          "생활수준(10점 만점)", "건강상태(10점 만점)", "대인관계(10점 만점)",
                                                          "안전정도(10점 만점)", "지역사회소속감(10점 만점)", "미래안정성(10점 만점)"]
                                        for col in additional_cols:
                                            if col in row and pd.notna(row[col]):
                                                persona_info[col] = str(row[col])
                                        
                                        prompt = f"""
다음 가상인물의 특성 정보를 바탕으로 이 인물의 페르소나를 200자 이내의 서술형으로 작성해주세요.
특성 정보를 종합하여 이 인물의 성격, 생활 방식, 가치관, 행동 패턴 등을 묘사해주세요.
완전한 문장으로 작성하고, 중간에 잘리지 않도록 주의해주세요.

특성 정보:
{chr(10).join([f"- {k}: {v}" for k, v in persona_info.items()])}

페르소나 (200자 이내, 완전한 문장으로 서술형):
"""
                                        batch_prompts.append(prompt)
                                        batch_indices.append(batch_item)
                                    
                                    # 배치 단위로 Gemini API 호출 (1명씩 처리하되 배치로 그룹화)
                                    batch_results = []
                                    record_updates = {}  # record_id별로 그룹화
                                    
                                    for i, (prompt, batch_item) in enumerate(zip(batch_prompts, batch_indices)):
                                        if st.session_state.persona_generation_stop:
                                            break
                                        
                                        current_person_num = batch_start + i + 1
                                        
                                        # 처리 시작 전 진행 상황 표시
                                        status_text.text(f"처리 중: {processed + 1}/{total_rows}명 (배치 {batch_start//batch_size + 1}/{total_rows//batch_size + (1 if total_rows % batch_size > 0 else 0)}, {current_person_num}번째)")
                                        time.sleep(0.05)  # UI 업데이트를 위한 짧은 대기
                                        
                                        persona_text = ""
                                        max_retries = 3
                                        retry_delay = 60
                                        
                                        for retry in range(max_retries):
                                            try:
                                                response = gemini_client._client.models.generate_content(
                                                    model=gemini_client._model,
                                                    contents=prompt,
                                                )
                                                persona_text = (response.text or "").strip()
                                                # 200자 제한 (완전한 문장 유지)
                                                if len(persona_text) > 200:
                                                    # 마지막 문장이 완전하지 않을 수 있으므로 마지막 마침표나 문장 부호를 찾아서 자름
                                                    truncated = persona_text[:200]
                                                    last_period = max(
                                                        truncated.rfind('.'), 
                                                        truncated.rfind('!'), 
                                                        truncated.rfind('?'),
                                                        truncated.rfind('다'),
                                                        truncated.rfind('요'),
                                                        truncated.rfind('음'),
                                                        truncated.rfind('니다')
                                                    )
                                                    if last_period > 150:  # 최소 150자 이상은 유지
                                                        persona_text = truncated[:last_period + 1].strip()
                                                    else:
                                                        persona_text = truncated.strip()
                                                
                                                # 성공 시 즉시 진행 상황 업데이트
                                                processed += 1
                                                progress_bar.progress(processed / total_rows)
                                                status_text.text(f"처리 완료: {processed}/{total_rows}명 (배치 {batch_start//batch_size + 1}/{total_rows//batch_size + (1 if total_rows % batch_size > 0 else 0)}, {current_person_num}번째)")
                                                time.sleep(0.05)  # UI 업데이트를 위한 짧은 대기
                                                break  # 성공 시 루프 종료
                                                
                                            except Exception as e:
                                                error_str = str(e)
                                                # 400 오류 (API 키 만료/유효하지 않음) 처리
                                                if "400" in error_str or "INVALID_ARGUMENT" in error_str or "API key expired" in error_str or "API_KEY_INVALID" in error_str:
                                                    # 배포 환경인지 확인
                                                    is_cloud = False
                                                    api_key_source_info = "알 수 없음"
                                                    try:
                                                        # Streamlit Cloud 또는 배포 환경 확인
                                                        if hasattr(st, 'secrets'):
                                                            is_cloud = True
                                                            # API 키 출처 확인
                                                            if hasattr(gemini_client, '_key_source'):
                                                                api_key_source_info = gemini_client._key_source
                                                            elif 'GEMINI_API_KEY' in st.secrets:
                                                                api_key_source_info = "Streamlit Secrets (확인됨)"
                                                            else:
                                                                api_key_source_info = "Streamlit Secrets (로드 실패 가능)"
                                                    except:
                                                        pass
                                                    
                                                    if is_cloud:
                                                        st.error("⚠️ **API 키 만료 오류**")
                                                        st.warning(f"**현재 사용 중인 API 키 출처**: {api_key_source_info}")
                                                        st.error("배포 환경에서 API 키가 만료되었습니다. 다음 단계를 따라주세요:")
                                                        st.markdown("""
                                                        1. **Google AI Studio**에서 새 API 키 발급: https://aistudio.google.com/apikey
                                                        2. **Streamlit Cloud** 대시보드 접속
                                                        3. 앱 선택 → **Settings** → **Secrets**
                                                        4. 기존 `GEMINI_API_KEY` 값을 **새로운 API 키**로 교체:
                                                        ```
                                                        GEMINI_API_KEY = "새로_발급받은_API_키"
                                                        ```
                                                        5. **저장** 후 앱이 자동으로 재배포됩니다 (몇 분 소요)
                                                        6. 재배포 완료 후 다시 시도해주세요
                                                        """)
                                                        st.info("💡 **팁**: Secrets에 저장한 후 앱이 재배포될 때까지 몇 분 기다려주세요. 재배포가 완료되면 새 API 키가 적용됩니다.")
                                                    else:
                                                        st.error("⚠️ **API 키 만료 오류**")
                                                        st.warning(f"**현재 사용 중인 API 키 출처**: {api_key_source_info}")
                                                        st.error("로컬 환경에서 API 키가 만료되었습니다.")
                                                        st.error("`utils/gemini_key.py` 파일의 `GEMINI_API_KEY`를 확인하고 Google AI Studio에서 새 키를 발급받아주세요.")
                                                        st.markdown("**Google AI Studio**: https://aistudio.google.com/apikey")
                                                    
                                                    st.error(f"**상세 오류**: {error_str[:300]}")
                                                    persona_text = ""
                                                    st.session_state.persona_generation_running = False
                                                    st.session_state.persona_generation_stop = True
                                                    break
                                                # 429 오류 (할당량 초과) 처리
                                                elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                                                    if retry < max_retries - 1:
                                                        if "retry in" in error_str.lower():
                                                            try:
                                                                import re
                                                                delay_match = re.search(r'retry in ([\d.]+)s', error_str.lower())
                                                                if delay_match:
                                                                    retry_delay = int(float(delay_match.group(1))) + 5
                                                            except:
                                                                pass
                                                        
                                                        status_text.text(f"API 할당량 초과. {retry_delay}초 후 재시도... (시도 {retry+1}/{max_retries}) - {current_person_num}번째 ({processed}/{total_rows}명)")
                                                        time.sleep(retry_delay)
                                                        continue
                                                    else:
                                                        st.warning(f"페르소나 생성 실패 ({current_person_num}번째): API 할당량 초과. 나중에 다시 시도해주세요.")
                                                        persona_text = ""
                                                        break
                                                else:
                                                    st.warning(f"페르소나 생성 실패 ({current_person_num}번째): {e}")
                                                    persona_text = ""
                                                    break
                                        
                                        batch_results.append(persona_text)
                                        
                                        # 각 개인 처리 후 즉시 DataFrame에 반영
                                        record_id = batch_item["record_id"]
                                        df = batch_item["df"]
                                        row_idx = batch_item["row_idx"]
                                        
                                        df.at[row_idx, "페르소나"] = persona_text
                                        
                                        if record_id not in record_updates:
                                            record_updates[record_id] = df
                                        
                                        # 10명마다 중간 저장 (진행 상황 보존)
                                        if (i + 1) % 10 == 0 or i == len(batch_prompts) - 1:
                                            for rid, update_df in record_updates.items():
                                                updated_json = update_df.to_json(orient="records", force_ascii=False)
                                                cur.execute(
                                                    "UPDATE virtual_population_db SET data_json = ? WHERE id = ?",
                                                    (updated_json, rid)
                                                )
                                            conn.commit()
                                    
                                    # 배치 처리 완료 후 최종 저장
                                    for record_id, df in record_updates.items():
                                        updated_json = df.to_json(orient="records", force_ascii=False)
                                        cur.execute(
                                            "UPDATE virtual_population_db SET data_json = ? WHERE id = ?",
                                            (updated_json, record_id)
                                        )
                                    conn.commit()  # 배치 단위로 커밋
                                    
                                    # 배치 간 짧은 대기 (API 부하 분산)
                                    if batch_end < total_rows and not st.session_state.persona_generation_stop:
                                        status_text.text(f"배치 {batch_start//batch_size + 1} 완료: {processed}/{total_rows}명 처리됨 (다음 배치 준비 중...)")
                                        time.sleep(0.5)
                                    else:
                                        status_text.text(f"배치 {batch_start//batch_size + 1} 완료: {processed}/{total_rows}명 처리됨")
                                
                                conn.close()
                                
                                progress_bar.empty()
                                status_text.empty()
                                
                                if st.session_state.persona_generation_stop:
                                    st.warning(f"페르소나 생성이 중지되었습니다. 지금까지 {processed}명의 가상인물에 페르소나가 생성되었습니다.")
                                else:
                                    st.success(f"페르소나 생성 완료: {processed}명의 가상인물에 페르소나가 생성되었습니다.")
                                
                                st.session_state.persona_generation_running = False
                                st.session_state.persona_generation_stop = False
                                st.rerun()
                                
                            except Exception as e:
                                conn.close()
                                st.error(f"페르소나 생성 중 오류 발생: {e}")
                                st.exception(e)
                                st.session_state.persona_generation_running = False
                                st.session_state.persona_generation_stop = False
            
            # 선택된 기록을 가상인구 DB에 누적
            if selected_record_keys:
                if st.button("선택한 기록을 DB에 추가", type="primary", use_container_width=True, key="vdb_add_records"):
                    conn = db_conn()
                    cur = conn.cursor()
                    added_count = 0
                    
                    for item in selected_record_keys:
                        record = item["record"]
                        excel_path = item["excel_path"]
                        
                        # 이미 추가된 기록인지 확인
                        cur.execute(
                            "SELECT id FROM virtual_population_db WHERE sido_code = ? AND record_timestamp = ? AND record_excel_path = ?",
                            (selected_sido_code, record.get("timestamp", ""), excel_path)
                        )
                        if cur.fetchone():
                            continue  # 이미 추가된 기록은 스킵
                        
                        # 기존 DB에서 같은 지역의 모든 데이터를 가져와서 페르소나 매핑 생성
                        cur.execute(
                            "SELECT data_json FROM virtual_population_db WHERE sido_code = ?",
                            (selected_sido_code,)
                        )
                        existing_personas = {}  # {가상이름: 페르소나} 매핑
                        existing_reflections = {}  # {가상이름: 현시대 반영} 매핑
                        
                        for existing_row in cur.fetchall():
                            try:
                                existing_data_json = existing_row[0]
                                existing_df = pd.read_json(existing_data_json, orient="records")
                                
                                # 가상이름 컬럼이 있으면 페르소나 매핑 생성
                                if "가상이름" in existing_df.columns and "페르소나" in existing_df.columns:
                                    for _, existing_person in existing_df.iterrows():
                                        name = str(existing_person.get("가상이름", "")).strip()
                                        persona = str(existing_person.get("페르소나", "")).strip()
                                        reflection = str(existing_person.get("현시대 반영", "")).strip()
                                        
                                        if name and name != "nan" and persona and persona != "nan" and persona != "":
                                            existing_personas[name] = persona
                                        if name and name != "nan" and reflection and reflection != "nan" and reflection != "":
                                            existing_reflections[name] = reflection
                            except:
                                pass
                        
                        # Excel 파일을 DataFrame으로 로드
                        try:
                            df = pd.read_excel(excel_path, engine="openpyxl")
                            # 식별NO 컬럼 제거
                            if "식별NO" in df.columns:
                                df = df.drop(columns=["식별NO"])
                            
                            # 미래안정성(10점 만점) 컬럼이 있으면 페르소나, 현시대 반영 컬럼 추가
                            if "미래안정성(10점 만점)" in df.columns:
                                # 페르소나 컬럼 추가 (비어있으면 빈 문자열)
                                if "페르소나" not in df.columns:
                                    df["페르소나"] = ""
                                # 현시대 반영 컬럼 추가 (비어있으면 빈 문자열)
                                if "현시대 반영" not in df.columns:
                                    df["현시대 반영"] = ""
                                
                                # 기존 페르소나 유지: 가상이름을 기준으로 기존 페르소나 매핑
                                if "가상이름" in df.columns:
                                    for idx, row in df.iterrows():
                                        name = str(row.get("가상이름", "")).strip()
                                        if name and name != "nan":
                                            # 기존 페르소나가 있으면 유지
                                            if name in existing_personas:
                                                df.at[idx, "페르소나"] = existing_personas[name]
                                            # 기존 현시대 반영이 있으면 유지
                                            if name in existing_reflections:
                                                df.at[idx, "현시대 반영"] = existing_reflections[name]
                                
                                # 컬럼 순서 조정: 미래안정성 다음에 페르소나, 현시대 반영 배치
                                cols = list(df.columns)
                                if "미래안정성(10점 만점)" in cols:
                                    idx = cols.index("미래안정성(10점 만점)")
                                    # 미래안정성, 페르소나, 현시대 반영을 제거
                                    cols = [c for c in cols if c not in ["미래안정성(10점 만점)", "페르소나", "현시대 반영"]]
                                    # 미래안정성 위치에 다시 삽입
                                    cols.insert(idx, "미래안정성(10점 만점)")
                                    cols.insert(idx + 1, "페르소나")
                                    cols.insert(idx + 2, "현시대 반영")
                                    df = df[cols]
                            else:
                                # 미래안정성 컬럼이 없으면 맨 뒤에 추가
                                if "페르소나" not in df.columns:
                                    df["페르소나"] = ""
                                if "현시대 반영" not in df.columns:
                                    df["현시대 반영"] = ""
                                
                                # 기존 페르소나 유지: 가상이름을 기준으로 기존 페르소나 매핑
                                if "가상이름" in df.columns:
                                    for idx, row in df.iterrows():
                                        name = str(row.get("가상이름", "")).strip()
                                        if name and name != "nan":
                                            # 기존 페르소나가 있으면 유지
                                            if name in existing_personas:
                                                df.at[idx, "페르소나"] = existing_personas[name]
                                            # 기존 현시대 반영이 있으면 유지
                                            if name in existing_reflections:
                                                df.at[idx, "현시대 반영"] = existing_reflections[name]
                            
                            # DataFrame을 JSON으로 변환
                            data_json = df.to_json(orient="records", force_ascii=False)
                            
                            # DB에 저장
                            cur.execute(
                                """
                                INSERT INTO virtual_population_db 
                                (sido_code, sido_name, record_timestamp, record_excel_path, data_json)
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                (
                                    selected_sido_code,
                                    selected_sido_name,
                                    record.get("timestamp", ""),
                                    excel_path,
                                    data_json
                                )
                            )
                            added_count += 1
                        except Exception as e:
                            st.warning(f"기록 추가 실패: {excel_path} - {e}")
                    
                    conn.commit()
                    conn.close()
                    
                    if added_count > 0:
                        st.success(f"{added_count}개의 기록이 가상인구 DB에 추가되었습니다.")
                        st.session_state.vdb_selected_records = []
                        st.rerun()
                    else:
                        st.info("추가할 새로운 기록이 없습니다. (이미 추가된 기록은 제외됩니다)")
            
            st.markdown("---")
            
            # 가상인구 DB 표시 (누적된 모든 데이터)
            st.subheader("가상인구 DB")
            
            # DB에서 누적된 데이터 불러오기
            conn = db_conn()
            cur = conn.cursor()
            cur.execute(
                "SELECT data_json FROM virtual_population_db WHERE sido_code = ? ORDER BY added_at",
                (selected_sido_code,)
            )
            db_rows = cur.fetchall()
            conn.close()
            
            if not db_rows:
                st.info("가상인구 DB에 데이터가 없습니다. 위에서 2차 대입결과를 선택하여 추가해주세요.")
            else:
                # 모든 데이터를 하나의 DataFrame으로 합치기
                all_dfs = []
                for row in db_rows:
                    try:
                        data_json = row[0]
                        df = pd.read_json(data_json, orient="records")
                        # 식별NO 컬럼 제거 (이미 저장된 데이터에도 있을 수 있으므로)
                        if "식별NO" in df.columns:
                            df = df.drop(columns=["식별NO"])
                        
                        # 페르소나, 현시대 반영 컬럼이 없으면 추가
                        if "페르소나" not in df.columns:
                            df["페르소나"] = ""
                        if "현시대 반영" not in df.columns:
                            df["현시대 반영"] = ""
                        
                        all_dfs.append(df)
                    except Exception as e:
                        st.warning(f"데이터 로드 실패: {e}")
                        continue
                
                if all_dfs:
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    
                    # 페르소나 컬럼의 데이터 타입 확인 및 보정
                    if "페르소나" in combined_df.columns:
                        # NaN이나 None을 빈 문자열로 변환
                        combined_df["페르소나"] = combined_df["페르소나"].fillna("")
                        # 숫자 타입이면 문자열로 변환
                        combined_df["페르소나"] = combined_df["페르소나"].astype(str)
                        # "nan" 문자열을 빈 문자열로 변환
                        combined_df["페르소나"] = combined_df["페르소나"].replace("nan", "")
                    
                    # 현시대 반영 컬럼도 동일하게 처리
                    if "현시대 반영" in combined_df.columns:
                        combined_df["현시대 반영"] = combined_df["현시대 반영"].fillna("")
                        combined_df["현시대 반영"] = combined_df["현시대 반영"].astype(str)
                        combined_df["현시대 반영"] = combined_df["현시대 반영"].replace("nan", "")
                    
                    # 컬럼 순서 조정: 미래안정성 다음에 페르소나, 현시대 반영 배치
                    if "미래안정성(10점 만점)" in combined_df.columns:
                        original_cols = list(combined_df.columns)
                        future_idx = original_cols.index("미래안정성(10점 만점)")
                        # 미래안정성 앞의 컬럼들
                        before_cols = [c for c in original_cols[:future_idx] if c not in ["페르소나", "현시대 반영"]]
                        # 미래안정성 뒤의 컬럼들
                        after_cols = [c for c in original_cols[future_idx+1:] if c not in ["페르소나", "현시대 반영"]]
                        # 재배치
                        new_cols = before_cols + ["미래안정성(10점 만점)", "페르소나", "현시대 반영"] + after_cols
                        combined_df = combined_df[new_cols]
                    
                    st.caption(f"총 {len(combined_df):,}명의 데이터 (모든 기록 누적)")
                    
                    # 페르소나 생성 통계 표시
                    if "페르소나" in combined_df.columns:
                        persona_count = combined_df["페르소나"].apply(lambda x: str(x).strip() != "" and str(x).strip() != "nan").sum()
                        st.caption(f"페르소나 생성 완료: {persona_count:,}명 / 전체 {len(combined_df):,}명")
                    
                    # 전체 데이터 표시 (미리보기 아님)
                    st.dataframe(combined_df, use_container_width=True, height=600)
                    
                    # 다운로드 버튼
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        out_buffer = BytesIO()
                        combined_df.to_excel(out_buffer, index=False, engine="openpyxl")
                        out_buffer.seek(0)
                        st.download_button(
                            "Excel 다운로드",
                            data=out_buffer.getvalue(),
                            file_name=f"{selected_sido_name}_virtual_population_db.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="vdb_dl_excel"
                        )
                    with col_dl2:
                        csv = combined_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "CSV 다운로드",
                            data=csv,
                            file_name=f"{selected_sido_name}_virtual_population_db.csv",
                            mime="text/csv",
                            key="vdb_dl_csv"
                        )
    
    with col_right:
        # 채팅창 영역
        st.subheader("가상인구와 대화")
        
        # 채팅 모드 선택 버튼
        col_chat1, col_chat2, col_chat3 = st.columns(3)
        with col_chat1:
            chat_mode_1to1 = st.button("1:1대화", key="chat_mode_1to1", use_container_width=True)
        with col_chat2:
            chat_mode_5to1 = st.button("5:1대화", key="chat_mode_5to1", use_container_width=True)
        with col_chat3:
            chat_mode_all = st.button("전체 학습", key="chat_mode_all", use_container_width=True)
        
        # 채팅 모드 설정
        if chat_mode_1to1:
            st.session_state.chat_mode = "1:1대화"
            st.session_state.selected_chat_person = None  # 선택 초기화
        elif chat_mode_5to1:
            st.session_state.chat_mode = "5:1대화"
            st.session_state.selected_chat_people = None  # 선택 초기화
        elif chat_mode_all:
            st.session_state.chat_mode = "전체 학습"
        
        if "chat_mode" not in st.session_state:
            st.session_state.chat_mode = "1:1대화"
        
        # 채팅 히스토리 초기화
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # 현재 모드 표시
        st.caption(f"현재 모드: {st.session_state.chat_mode}")
        st.markdown("---")
        
        # 전체 학습 모드: 텍스트 + PDF 업로드 및 현시대 반영 대입
        if st.session_state.chat_mode == "전체 학습":
            st.markdown("**전체 학습: 현시대 자료 입력**")
            st.caption("텍스트를 입력하거나 PDF를 업로드하면, Gemini가 분석하여 각 가상인구의 페르소나·특성에 맞게 관심분야 가중치를 적용해 '현시대 반영'을 100자 이내로 생성·대입합니다.")
            
            col_learn1, col_learn2 = st.columns([1, 1])
            with col_learn1:
                learn_text = st.text_area(
                    "현시대 관련 텍스트",
                    height=120,
                    placeholder="현시대 반영에 사용할 내용을 입력하세요 (뉴스, 정책, 트렌드 등)...",
                    key="learn_text"
                )
            with col_learn2:
                learn_pdf = st.file_uploader("PDF 파일 업로드", type=["pdf"], key="learn_pdf")
                st.caption("텍스트 입력 또는 PDF 업로드 후 아래 버튼을 누르세요.")
            
            # PDF 텍스트 추출 헬퍼
            def _extract_pdf_text(uploaded_file):
                if uploaded_file is None:
                    return ""
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(uploaded_file)
                    return "\n".join(page.extract_text() or "" for page in reader.pages)
                except Exception:
                    return ""
            
            # 중지 플래그 초기화
            if "learn_stop" not in st.session_state:
                st.session_state.learn_stop = False
            if "learn_running" not in st.session_state:
                st.session_state.learn_running = False
            
            col_btn1, col_btn2 = st.columns([1, 2])
            with col_btn1:
                if st.session_state.learn_running:
                    if st.button("⏹️ 중지", type="secondary", use_container_width=True, key="learn_stop_btn"):
                        st.session_state.learn_stop = True
                        st.rerun()
                else:
                    run_learn_btn = st.button("전체 학습 실행", type="primary", use_container_width=True, key="run_learn_btn")
            with col_btn2:
                st.caption("실행 시 가상인구 DB의 '현시대 반영' 컬럼이 갱신됩니다. (페르소나가 있는 가상인구만)")
            
            run_learn_btn = run_learn_btn if not st.session_state.learn_running else False
            if run_learn_btn:
                combined_content = (learn_text or "").strip()
                if learn_pdf is not None:
                    pdf_text = _extract_pdf_text(learn_pdf)
                    if pdf_text:
                        combined_content = (combined_content + "\n\n[PDF 내용]\n" + pdf_text).strip()
                if not combined_content:
                    st.warning("텍스트를 입력하거나 PDF 파일을 업로드해주세요.")
                else:
                    st.session_state.learn_running = True
                    st.session_state.learn_stop = False
                    st.session_state.learn_combined_content = combined_content
                    st.session_state.learn_sido_code = selected_sido_code
                    st.session_state.learn_total_done = 0  # 누적 처리 인원
                    st.rerun()
            
            # 전체 학습 실행 중일 때 — 10명씩 처리 후 rerun (무한 로딩 방지)
            LEARN_CHUNK_SIZE = 10
            if st.session_state.learn_running and not st.session_state.learn_stop:
                combined_content = st.session_state.get("learn_combined_content", "").strip()
                run_sido = st.session_state.get("learn_sido_code", selected_sido_code)
                if not combined_content:
                    st.warning("현시대 자료가 없습니다. 다시 텍스트/PDF를 입력한 뒤 실행해주세요.")
                    st.session_state.learn_running = False
                else:
                    conn = db_conn()
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT id, data_json FROM virtual_population_db WHERE sido_code = ? ORDER BY added_at",
                        (run_sido,)
                    )
                    db_rows = cur.fetchall()
                    if not db_rows:
                        st.warning("가상인구 DB에 데이터가 없습니다.")
                        st.session_state.learn_running = False
                    else:
                        try:
                            all_people_to_process = []
                            for rec_idx, (record_id, data_json) in enumerate(db_rows):
                                df = pd.read_json(data_json, orient="records")
                                if "현시대 반영" not in df.columns:
                                    df["현시대 반영"] = ""
                                if "페르소나" not in df.columns:
                                    df["페르소나"] = ""
                                for idx, row in df.iterrows():
                                    persona = str(row.get("페르소나", "")).strip()
                                    reflection = str(row.get("현시대 반영", "")).strip()
                                    if persona and persona != "" and persona != "nan" and (not reflection or reflection == "" or reflection == "nan"):
                                        all_people_to_process.append({"record_id": record_id, "df": df, "row_idx": idx, "row": row})
                            
                            total_people = len(all_people_to_process)
                            if total_people == 0:
                                total_done = st.session_state.get("learn_total_done", 0)
                                st.success(f"전체 학습 완료: 총 {total_done}명의 가상인구에 현시대 반영을 반영했습니다.")
                                st.session_state.learn_running = False
                                st.session_state.learn_stop = False
                                for k in ("learn_combined_content", "learn_sido_code", "learn_total_done"):
                                    st.session_state.pop(k, None)
                                conn.close()
                                st.rerun()
                            else:
                                chunk = all_people_to_process[:LEARN_CHUNK_SIZE]
                                progress_bar = st.progress(0)
                                status_placeholder = st.empty()
                                total_done = st.session_state.get("learn_total_done", 0)
                                status_placeholder.caption(f"처리 중: 이번에 {len(chunk)}명 처리 (누적 {total_done + len(chunk)}명 / 남은 대상 {total_people}명)")
                                gemini_client = GeminiClient()
                                processed_this_run = 0
                                for person_item in chunk:
                                    if st.session_state.learn_stop:
                                        break
                                    record_id = person_item["record_id"]
                                    df = person_item["df"]
                                    row_idx = person_item["row_idx"]
                                    row = person_item["row"]
                                    profile_parts = []
                                    for c in ["거주지역", "성별", "연령", "경제활동", "교육정도", "월평균소득", "가상이름"]:
                                        if c in row and pd.notna(row.get(c)):
                                            profile_parts.append(f"{c}: {row[c]}")
                                    persona = str(row.get("페르소나", "")).strip()
                                    if persona and persona != "" and persona != "nan":
                                        profile_parts.append(f"페르소나: {persona[:150]}")
                                    profile_str = "\n".join(profile_parts) if profile_parts else "특성 없음"
                                    prompt = f"""다음은 '현시대 자료'입니다.
---
{combined_content[:8000]}
---
아래 가상인물의 페르소나와 특성을 고려하여, 이 인물의 관심분야에 맞는 내용만 선택적으로 반영하세요.
관련도가 높으면 100자 이내로 한 문장으로 '현시대 반영' 문장을 작성하고, 관련도가 낮으면 빈 문자열만 반환하세요.
반드시 100자 이내, 한 문장으로만 출력하고 다른 설명은 하지 마세요.

가상인물 정보:
{profile_str}

현시대 반영 (100자 이내, 또는 관련 없으면 빈칸):"""
                                    try:
                                        resp = gemini_client._client.models.generate_content(model=gemini_client._model, contents=prompt)
                                        text = (resp.text or "").strip()
                                        if "현시대 반영" in text and ":" in text:
                                            text = text.split(":", 1)[-1].strip()
                                        if len(text) > 100:
                                            text = text[:97].rsplit(" ", 1)[0] if " " in text[:97] else text[:97]
                                            if len(text) > 100:
                                                text = text[:97] + "..."
                                        df.at[row_idx, "현시대 반영"] = text
                                    except Exception:
                                        pass
                                    new_json = df.to_json(orient="records", force_ascii=False)
                                    cur.execute("UPDATE virtual_population_db SET data_json = ? WHERE id = ?", (new_json, record_id))
                                    conn.commit()
                                    processed_this_run += 1
                                    progress_bar.progress(processed_this_run / len(chunk))
                                
                                conn.close()
                                progress_bar.empty()
                                status_placeholder.empty()
                                st.session_state.learn_total_done = total_done + processed_this_run
                                if st.session_state.learn_stop:
                                    st.warning(f"전체 학습이 중지되었습니다. 지금까지 {st.session_state.learn_total_done}명의 가상인구에 현시대 반영이 반영되었습니다.")
                                    st.session_state.learn_running = False
                                    st.session_state.learn_stop = False
                                    for k in ("learn_combined_content", "learn_sido_code", "learn_total_done"):
                                        st.session_state.pop(k, None)
                                    st.rerun()
                                else:
                                    st.rerun()
                        except Exception as e:
                            st.error(f"전체 학습 중 오류: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                            st.session_state.learn_running = False
                            st.session_state.learn_stop = False
                            for k in ("learn_combined_content", "learn_sido_code", "learn_total_done"):
                                st.session_state.pop(k, None)
            st.markdown("---")
        
        # 가상인구 선택 UI (1:1대화, 5:1대화 모드일 때만)
        if st.session_state.chat_mode in ["1:1대화", "5:1대화"]:
            # DB에서 가상인구 데이터 가져오기
            conn = db_conn()
            cur = conn.cursor()
            cur.execute(
                "SELECT data_json FROM virtual_population_db WHERE sido_code = ? ORDER BY added_at",
                (selected_sido_code,)
            )
            db_rows = cur.fetchall()
            conn.close()
            
            if db_rows:
                # 모든 데이터를 하나의 DataFrame으로 합치기
                all_dfs = []
                for row in db_rows:
                    try:
                        data_json = row[0]
                        df = pd.read_json(data_json, orient="records")
                        all_dfs.append(df)
                    except Exception:
                        continue
                
                if all_dfs:
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    
                    if st.session_state.chat_mode == "1:1대화":
                        # 1명 선택 (대분류-중분류-소분류 필터링)
                        st.markdown("**가상인구 선택:**")
                        
                        # 대분류: 연령대
                        age_groups = {}
                        for idx, person in combined_df.iterrows():
                            age = person.get('연령', 0)
                            try:
                                age_num = int(float(str(age)))
                                if age_num < 20:
                                    age_group = "10대"
                                elif age_num < 30:
                                    age_group = "20대"
                                elif age_num < 40:
                                    age_group = "30대"
                                elif age_num < 50:
                                    age_group = "40대"
                                elif age_num < 60:
                                    age_group = "50대"
                                elif age_num < 70:
                                    age_group = "60대"
                                elif age_num < 80:
                                    age_group = "70대"
                                else:
                                    age_group = "80대 이상"
                                
                                if age_group not in age_groups:
                                    age_groups[age_group] = []
                                age_groups[age_group].append((idx, person))
                            except:
                                if "미상" not in age_groups:
                                    age_groups["미상"] = []
                                age_groups["미상"].append((idx, person))
                        
                        # 대분류 선택
                        selected_age_group = st.selectbox(
                            "대분류: 연령대 선택",
                            options=sorted(age_groups.keys()),
                            key="chat_age_group_1to1"
                        )
                        
                        filtered_by_age = age_groups.get(selected_age_group, [])
                        
                        # 중분류: 성별
                        gender_groups = {}
                        for idx, person in filtered_by_age:
                            gender = str(person.get('성별', 'N/A')).strip()
                            if gender not in gender_groups:
                                gender_groups[gender] = []
                            gender_groups[gender].append((idx, person))
                        
                        # 중분류 선택
                        selected_gender = st.selectbox(
                            "중분류: 성별 선택",
                            options=sorted(gender_groups.keys()),
                            key="chat_gender_1to1"
                        )
                        
                        filtered_by_gender = gender_groups.get(selected_gender, [])
                        
                        # 소분류: 지역
                        region_groups = {}
                        for idx, person in filtered_by_gender:
                            region = str(person.get('거주지역', 'N/A')).strip()
                            if region not in region_groups:
                                region_groups[region] = []
                            region_groups[region].append((idx, person))
                        
                        # 소분류 선택
                        selected_region = st.selectbox(
                            "소분류: 지역 선택",
                            options=sorted(region_groups.keys()),
                            key="chat_region_1to1"
                        )
                        
                        final_filtered = region_groups.get(selected_region, [])
                        
                        # 최종 선택 (필터링된 목록에서)
                        if len(final_filtered) > 0:
                            person_options = []
                            for idx, person in final_filtered:
                                name = person.get('가상이름', f'인물 {idx+1}')
                                age = person.get('연령', 'N/A')
                                gender = person.get('성별', 'N/A')
                                region = person.get('거주지역', 'N/A')
                                person_label = f"{name} ({age}세, {gender}, {region})"
                                person_options.append((idx, person_label, person))
                            
                            selected_person_idx = st.selectbox(
                                f"가상인구 선택 (총 {len(person_options)}명):",
                                options=[opt[0] for opt in person_options],
                                format_func=lambda x: next(opt[1] for opt in person_options if opt[0] == x),
                                key="chat_select_person_1to1"
                            )
                            
                            selected_person = next(opt[2] for opt in person_options if opt[0] == selected_person_idx)
                            st.session_state.selected_chat_person = selected_person
                            
                            # 선택한 가상인구 정보 표시
                            with st.expander("선택한 가상인구 정보", expanded=False):
                                st.write(f"- 이름: {selected_person.get('가상이름', 'N/A')}")
                                st.write(f"- 거주지역: {selected_person.get('거주지역', 'N/A')}")
                                st.write(f"- 성별: {selected_person.get('성별', 'N/A')}")
                                st.write(f"- 연령: {selected_person.get('연령', 'N/A')}")
                                st.write(f"- 경제활동: {selected_person.get('경제활동', 'N/A')}")
                                st.write(f"- 교육정도: {selected_person.get('교육정도', 'N/A')}")
                                st.write(f"- 월평균소득: {selected_person.get('월평균소득', 'N/A')}")
                                st.write(f"- 페르소나: {selected_person.get('페르소나', 'N/A')}")
                        else:
                            st.warning("선택한 조건에 해당하는 가상인구가 없습니다.")
                            st.session_state.selected_chat_person = None
                    
                    elif st.session_state.chat_mode == "5:1대화":
                        # 5명 선택 (대분류-중분류-소분류 필터링)
                        st.markdown("**가상인구 선택 (5명):**")
                        
                        # 대분류: 연령대
                        age_groups = {}
                        for idx, person in combined_df.iterrows():
                            age = person.get('연령', 0)
                            try:
                                age_num = int(float(str(age)))
                                if age_num < 20:
                                    age_group = "10대"
                                elif age_num < 30:
                                    age_group = "20대"
                                elif age_num < 40:
                                    age_group = "30대"
                                elif age_num < 50:
                                    age_group = "40대"
                                elif age_num < 60:
                                    age_group = "50대"
                                elif age_num < 70:
                                    age_group = "60대"
                                elif age_num < 80:
                                    age_group = "70대"
                                else:
                                    age_group = "80대 이상"
                                
                                if age_group not in age_groups:
                                    age_groups[age_group] = []
                                age_groups[age_group].append((idx, person))
                            except:
                                if "미상" not in age_groups:
                                    age_groups["미상"] = []
                                age_groups["미상"].append((idx, person))
                        
                        # 대분류 선택
                        selected_age_group = st.selectbox(
                            "대분류: 연령대 선택",
                            options=sorted(age_groups.keys()),
                            key="chat_age_group_5to1"
                        )
                        
                        filtered_by_age = age_groups.get(selected_age_group, [])
                        
                        # 중분류: 성별
                        gender_groups = {}
                        for idx, person in filtered_by_age:
                            gender = str(person.get('성별', 'N/A')).strip()
                            if gender not in gender_groups:
                                gender_groups[gender] = []
                            gender_groups[gender].append((idx, person))
                        
                        # 중분류 선택
                        selected_gender = st.selectbox(
                            "중분류: 성별 선택",
                            options=sorted(gender_groups.keys()),
                            key="chat_gender_5to1"
                        )
                        
                        filtered_by_gender = gender_groups.get(selected_gender, [])
                        
                        # 소분류: 지역
                        region_groups = {}
                        for idx, person in filtered_by_gender:
                            region = str(person.get('거주지역', 'N/A')).strip()
                            if region not in region_groups:
                                region_groups[region] = []
                            region_groups[region].append((idx, person))
                        
                        # 소분류 선택
                        selected_region = st.selectbox(
                            "소분류: 지역 선택",
                            options=sorted(region_groups.keys()),
                            key="chat_region_5to1"
                        )
                        
                        final_filtered = region_groups.get(selected_region, [])
                        
                        # 최종 선택 (필터링된 목록에서)
                        if len(final_filtered) > 0:
                            person_options = []
                            for idx, person in final_filtered:
                                name = person.get('가상이름', f'인물 {idx+1}')
                                age = person.get('연령', 'N/A')
                                gender = person.get('성별', 'N/A')
                                region = person.get('거주지역', 'N/A')
                                person_label = f"{name} ({age}세, {gender}, {region})"
                                person_options.append((idx, person_label, person))
                            
                            selected_indices = st.multiselect(
                                f"가상인구 선택 (최대 5명, 총 {len(person_options)}명):",
                                options=[opt[0] for opt in person_options],
                                format_func=lambda x: next(opt[1] for opt in person_options if opt[0] == x),
                                key="chat_select_people_5to1",
                                max_selections=5
                            )
                            
                            if len(selected_indices) > 0:
                                selected_people = [next(opt[2] for opt in person_options if opt[0] == idx) for idx in selected_indices]
                                st.session_state.selected_chat_people = selected_people
                                
                                # 선택한 가상인구 정보 표시
                                with st.expander("선택한 가상인구 정보", expanded=False):
                                    for i, person in enumerate(selected_people):
                                        st.markdown(f"**인물 {i+1}:**")
                                        st.write(f"- 이름: {person.get('가상이름', 'N/A')}")
                                        st.write(f"- 거주지역: {person.get('거주지역', 'N/A')}")
                                        st.write(f"- 성별: {person.get('성별', 'N/A')}")
                                        st.write(f"- 연령: {person.get('연령', 'N/A')}")
                                        st.write(f"- 페르소나: {person.get('페르소나', 'N/A')}")
                                        st.markdown("---")
                            else:
                                st.session_state.selected_chat_people = None
                                st.info("가상인구를 선택해주세요.")
                        else:
                            st.warning("선택한 조건에 해당하는 가상인구가 없습니다.")
                            st.session_state.selected_chat_people = None
                    
                    st.markdown("---")
                else:
                    st.info("가상인구 DB에 데이터가 없습니다.")
            else:
                st.info("가상인구 DB에 데이터가 없습니다.")
        
        # 전체 학습 모드가 아닐 때만 채팅 영역(히스토리·입력) 표시 — 전체 학습에서는 "대화를 시작해보세요" 및 입력창 숨김
        if st.session_state.chat_mode != "전체 학습":
            # 채팅 말풍선 스타일 CSS
            st.markdown("""
        <style>
        .chat-container-wrapper {
            max-height: 500px;
            overflow-y: auto;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 12px;
            margin-bottom: 20px;
            border: 1px solid #e5e7eb;
        }
        .chat-message-wrapper {
            margin-bottom: 20px;
            display: block !important;
            width: 100% !important;
        }
        .chat-message-wrapper.user-msg {
            text-align: right !important;
        }
        .chat-message-wrapper.assistant-msg {
            text-align: left !important;
        }
        .chat-bubble {
            display: inline-block !important;
            max-width: 70% !important;
            padding: 12px 16px !important;
            border-radius: 18px !important;
            word-wrap: break-word !important;
            line-height: 1.5 !important;
            font-size: 14px !important;
            margin-top: 4px !important;
        }
        .chat-bubble.user-bubble {
            background-color: #4f46e5 !important;
            color: white !important;
            border-bottom-right-radius: 4px !important;
            text-align: left !important;
        }
        .chat-bubble.assistant-bubble {
            background-color: white !important;
            color: #1f2937 !important;
            border: 1px solid #e5e7eb !important;
            border-bottom-left-radius: 4px !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }
        .chat-label {
            font-size: 11px !important;
            color: #6b7280 !important;
            margin-bottom: 4px !important;
            font-weight: 600 !important;
            display: block !important;
        }
        </style>
        """, unsafe_allow_html=True)
            st.markdown('<div class="chat-container-wrapper">', unsafe_allow_html=True)
            if not st.session_state.chat_history:
                st.info("💬 대화를 시작해보세요!")
            else:
                import html
                # 가상인구 특성 정보 가져오기 (1:1대화 또는 5:1대화 모드)
                def get_person_label(chat_data):
                    """채팅 데이터에서 가상인구 레이블 생성"""
                    if "person_info" in chat_data:
                        person_info = chat_data["person_info"]
                        name = person_info.get('가상이름', '가상인구')
                        age = person_info.get('연령', 'N/A')
                        gender = person_info.get('성별', 'N/A')
                        region = person_info.get('거주지역', 'N/A')
                        return f"{name} ({age}세, {gender}, {region})"
                    elif "people_info" in chat_data:
                        people_info = chat_data["people_info"]
                        if len(people_info) == 1:
                            p = people_info[0]
                            name = p.get('가상이름', '가상인구')
                            age = p.get('연령', 'N/A')
                            gender = p.get('성별', 'N/A')
                            region = p.get('거주지역', 'N/A')
                            return f"{name} ({age}세, {gender}, {region})"
                        else:
                            return f"가상인구 {len(people_info)}명"
                    elif "total_count" in chat_data:
                        return f"전체 가상인구 ({chat_data['total_count']}명)"
                    return "가상인구"
                for chat in st.session_state.chat_history:
                    escaped_message = html.escape(str(chat['message']))
                    escaped_message = escaped_message.replace('\n', '<br>')
                    if chat["role"] == "user":
                        st.markdown(f"""
                    <div class="chat-message-wrapper user-msg">
                        <span class="chat-label">사용자</span>
                        <div class="chat-bubble user-bubble">{escaped_message}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    else:
                        person_label = get_person_label(chat)
                        st.markdown(f"""
                    <div class="chat-message-wrapper assistant-msg">
                        <span class="chat-label">{person_label}</span>
                        <div class="chat-bubble assistant-bubble">{escaped_message}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
            # 채팅 입력 (1:1대화, 5:1대화 모드에서만)
            # 채팅 입력 초기화 (세션 상태 관리)
            if "chat_input_value" not in st.session_state:
                st.session_state.chat_input_value = ""
            
            # 전송 버튼 활성화 조건 확인
            can_send = False
            if st.session_state.chat_mode == "1:1대화":
                can_send = st.session_state.get("selected_chat_person") is not None
            elif st.session_state.chat_mode == "5:1대화":
                selected_people = st.session_state.get("selected_chat_people")
                can_send = selected_people is not None and len(selected_people) > 0
            else:
                can_send = True
            
            # 전송 함수 정의 (user_input은 폼에서 전달받음 — 위젯 key 접근 금지)
            def send_message(msg_text: str):
                user_input = (msg_text or "").strip()
                if user_input and can_send:
                    # 입력창 초기화는 폼의 clear_on_submit=True로 자동 처리됨 (위젯 key 수정 금지)
                    
                    # 사용자 메시지 추가
                    st.session_state.chat_history.append({"role": "user", "message": user_input})
                    
                    # GeminiClient 재사용 (세션 상태에 저장)
                    if "chat_gemini_client" not in st.session_state:
                        st.session_state.chat_gemini_client = GeminiClient()
                    gemini_client = st.session_state.chat_gemini_client
                    
                    # 가상인구 응답 생성
                    try:
                        if st.session_state.chat_mode == "1:1대화":
                            # 선택한 가상인구 사용
                            selected_person = st.session_state.get("selected_chat_person")
                            if selected_person is None:
                                response_text = "가상인구를 선택해주세요."
                            else:
                                # 프롬프트 최적화 (페르소나 + 현시대 반영 반영)
                                persona_info = selected_person.get('페르소나', '')
                                if not persona_info or persona_info == 'N/A':
                                    persona_info = f"{selected_person.get('거주지역', '')} 거주, {selected_person.get('성별', '')}, {selected_person.get('연령', '')}세, {selected_person.get('경제활동', '')}, {selected_person.get('교육정도', '')}"
                                _ref = selected_person.get('현시대 반영', '') or ''
                                reflection = str(_ref).strip() if _ref is not None else ''
                                if not reflection or reflection in ('N/A', 'nan'):
                                    reflection = ""
                                prompt_extra = f"\n현시대 반영 (이 인물의 최근 현시대 관심/반응, 답변 시 반영할 것): {reflection}\n" if reflection else ""
                                prompt = f"""다음 가상인물의 페르소나: {persona_info}{prompt_extra}
    
    이 가상인물의 입장에서 사용자의 질문에 자연스럽고 현실적인 대화체로 답변해주세요. 현시대 반영이 주어졌다면 그 관심·반응도 답변에 반영해주세요.
    
    사용자 질문: {user_input}"""
                                
                                # 스트리밍 응답 사용 (말풍선 스타일)
                                response_text = ""
                                response_placeholder = st.empty()
                                
                                try:
                                    # 스트리밍 응답 생성
                                    stream = gemini_client._client.models.generate_content_stream(
                                        model=gemini_client._model,
                                        contents=prompt,
                                    )
                                    
                                    import html
                                    for chunk in stream:
                                        if chunk.text:
                                            response_text += chunk.text
                                            # HTML 특수 문자 이스케이프 및 줄바꿈 처리
                                            escaped_text = html.escape(response_text).replace('\n', '<br>')
                                            # 가상인구 특성 레이블 생성
                                            name = selected_person.get('가상이름', '가상인구')
                                            age = selected_person.get('연령', 'N/A')
                                            gender = selected_person.get('성별', 'N/A')
                                            region = selected_person.get('거주지역', 'N/A')
                                            person_label = f"{name} ({age}세, {gender}, {region})"
                                            
                                            # 말풍선 스타일로 실시간 표시
                                            response_placeholder.markdown(f"""
                                            <div class="chat-message-wrapper assistant-msg">
                                                <span class="chat-label">{person_label}</span>
                                                <div class="chat-bubble assistant-bubble">{escaped_text}</div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                except Exception as stream_error:
                                    # 스트리밍 실패 시 일반 방식으로 폴백
                                    response = gemini_client._client.models.generate_content(
                                        model=gemini_client._model,
                                        contents=prompt,
                                    )
                                    response_text = (response.text or "").strip()
                                    import html
                                    escaped_text = html.escape(response_text).replace('\n', '<br>')
                                    # 가상인구 특성 레이블 생성
                                    name = selected_person.get('가상이름', '가상인구')
                                    age = selected_person.get('연령', 'N/A')
                                    gender = selected_person.get('성별', 'N/A')
                                    region = selected_person.get('거주지역', 'N/A')
                                    person_label = f"{name} ({age}세, {gender}, {region})"
                                    
                                    response_placeholder.markdown(f"""
                                    <div class="chat-message-wrapper assistant-msg">
                                        <span class="chat-label">{person_label}</span>
                                        <div class="chat-bubble assistant-bubble">{escaped_text}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        elif st.session_state.chat_mode == "5:1대화":
                            # 선택한 가상인구들 사용
                            selected_people = st.session_state.get("selected_chat_people")
                            if selected_people is None or len(selected_people) == 0:
                                response_text = "가상인구를 선택해주세요."
                            else:
                                # 프롬프트 최적화 (페르소나 + 현시대 반영 반영)
                                personas = []
                                for i, person in enumerate(selected_people):
                                    persona_info = person.get('페르소나', '')
                                    if not persona_info or persona_info == 'N/A':
                                        persona_info = f"{person.get('거주지역', '')} 거주, {person.get('성별', '')}, {person.get('연령', '')}세"
                                    _ref = person.get('현시대 반영', '') or ''
                                    reflection = str(_ref).strip() if _ref is not None else ''
                                    if not reflection or reflection in ('N/A', 'nan'):
                                        reflection = ""
                                    if reflection:
                                        personas.append(f"인물 {i+1}: {persona_info} | 현시대 반영: {reflection}")
                                    else:
                                        personas.append(f"인물 {i+1}: {persona_info}")
                                
                                prompt = f"""다음 가상인물들이 함께 대화합니다:
    {chr(10).join(personas)}
    
    이 가상인물들이 각자의 관점에서 사용자의 질문에 답변해주세요. 각 인물의 특성과 현시대 반영(주어졌다면)에 맞는 다양한 의견을 제시해주세요.
    
    사용자 질문: {user_input}"""
                                
                                # 스트리밍 응답 사용 (말풍선 스타일)
                                response_text = ""
                                response_placeholder = st.empty()
                                
                                try:
                                    # 스트리밍 응답 생성
                                    stream = gemini_client._client.models.generate_content_stream(
                                        model=gemini_client._model,
                                        contents=prompt,
                                    )
                                    
                                    import html
                                    for chunk in stream:
                                        if chunk.text:
                                            response_text += chunk.text
                                            # HTML 특수 문자 이스케이프 및 줄바꿈 처리
                                            escaped_text = html.escape(response_text).replace('\n', '<br>')
                                            # 5:1대화 모드 레이블 생성
                                            if len(selected_people) == 1:
                                                p = selected_people[0]
                                                name = p.get('가상이름', '가상인구')
                                                age = p.get('연령', 'N/A')
                                                gender = p.get('성별', 'N/A')
                                                region = p.get('거주지역', 'N/A')
                                                person_label = f"{name} ({age}세, {gender}, {region})"
                                            else:
                                                person_label = f"가상인구 {len(selected_people)}명"
                                            
                                            # 말풍선 스타일로 실시간 표시
                                            response_placeholder.markdown(f"""
                                            <div class="chat-message-wrapper assistant-msg">
                                                <span class="chat-label">{person_label}</span>
                                                <div class="chat-bubble assistant-bubble">{escaped_text}</div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                except Exception as stream_error:
                                    # 스트리밍 실패 시 일반 방식으로 폴백
                                    response = gemini_client._client.models.generate_content(
                                        model=gemini_client._model,
                                        contents=prompt,
                                    )
                                    response_text = (response.text or "").strip()
                                    import html
                                    escaped_text = html.escape(response_text).replace('\n', '<br>')
                                    # 5:1대화 모드 레이블 생성
                                    if len(selected_people) == 1:
                                        p = selected_people[0]
                                        name = p.get('가상이름', '가상인구')
                                        age = p.get('연령', 'N/A')
                                        gender = p.get('성별', 'N/A')
                                        region = p.get('거주지역', 'N/A')
                                        person_label = f"{name} ({age}세, {gender}, {region})"
                                    else:
                                        person_label = f"가상인구 {len(selected_people)}명"
                                    
                                    response_placeholder.markdown(f"""
                                    <div class="chat-message-wrapper assistant-msg">
                                        <span class="chat-label">{person_label}</span>
                                        <div class="chat-bubble assistant-bubble">{escaped_text}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        else:  # 전체 학습
                            # DB에서 가상인구 데이터 가져오기
                            conn = db_conn()
                            cur = conn.cursor()
                            cur.execute(
                                "SELECT data_json FROM virtual_population_db WHERE sido_code = ? ORDER BY added_at",
                                (selected_sido_code,)
                            )
                            db_rows = cur.fetchall()
                            conn.close()
                            
                            if not db_rows:
                                response_text = "가상인구 DB에 데이터가 없습니다. 먼저 데이터를 추가해주세요."
                            else:
                                # 모든 데이터를 하나의 DataFrame으로 합치기
                                all_dfs = []
                                for row in db_rows:
                                    try:
                                        data_json = row[0]
                                        df = pd.read_json(data_json, orient="records")
                                        all_dfs.append(df)
                                    except Exception:
                                        continue
                                
                                if all_dfs:
                                    combined_df = pd.concat(all_dfs, ignore_index=True)
                                    # 프롬프트 최적화
                                    prompt = f"""전체 {len(combined_df)}명의 가상인구 데이터를 학습한 AI입니다.
    다양한 가상인구의 특성과 페르소나를 종합하여 사용자의 질문에 자연스럽고 현실적인 답변을 해주세요.
    
    사용자 질문: {user_input}"""
                                    
                                    # 스트리밍 응답 사용 (말풍선 스타일)
                                    response_text = ""
                                    response_placeholder = st.empty()
                                    
                                    try:
                                        # 스트리밍 응답 생성
                                        stream = gemini_client._client.models.generate_content_stream(
                                            model=gemini_client._model,
                                            contents=prompt,
                                        )
                                        
                                        for chunk in stream:
                                            if chunk.text:
                                                response_text += chunk.text
                                                # 말풍선 스타일로 실시간 표시
                                                import html
                                                escaped_response = html.escape(str(response_text)).replace('\n', '<br>')
                                                response_placeholder.markdown(f"""
                                                <div class="chat-message-wrapper assistant-msg">
                                                    <span class="chat-label">전체 가상인구 ({len(combined_df)}명)</span>
                                                    <div class="chat-bubble assistant-bubble">{escaped_response}</div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                    except Exception as stream_error:
                                        # 스트리밍 실패 시 일반 방식으로 폴백
                                        response = gemini_client._client.models.generate_content(
                                            model=gemini_client._model,
                                            contents=prompt,
                                        )
                                        response_text = (response.text or "").strip()
                                        import html
                                        escaped_response = html.escape(response_text).replace('\n', '<br>')
                                        response_placeholder.markdown(f"""
                                        <div class="chat-message-wrapper assistant-msg">
                                            <span class="chat-label">전체 가상인구 ({len(combined_df)}명)</span>
                                            <div class="chat-bubble assistant-bubble">{escaped_response}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    response_text = "가상인구 데이터를 불러올 수 없습니다."
                    except Exception as e:
                        response_text = f"응답 생성 중 오류 발생: {e}"
                    
                    # 응답이 생성된 경우에만 히스토리에 추가 (가상인구 정보 포함)
                    if response_text:
                        chat_entry = {"role": "assistant", "message": response_text}
                        
                        # 가상인구 정보 추가
                        if st.session_state.chat_mode == "1:1대화":
                            selected_person = st.session_state.get("selected_chat_person")
                            if selected_person is not None:
                                # Series를 dict로 변환
                                if hasattr(selected_person, 'to_dict'):
                                    chat_entry["person_info"] = selected_person.to_dict()
                                else:
                                    chat_entry["person_info"] = dict(selected_person)
                        elif st.session_state.chat_mode == "5:1대화":
                            selected_people = st.session_state.get("selected_chat_people")
                            if selected_people is not None:
                                chat_entry["people_info"] = [p.to_dict() if hasattr(p, 'to_dict') else dict(p) for p in selected_people]
                        else:  # 전체 학습
                            # 전체 학습 모드에서는 인원 수만 저장
                            conn = db_conn()
                            cur = conn.cursor()
                            cur.execute(
                                "SELECT data_json FROM virtual_population_db WHERE sido_code = ? ORDER BY added_at",
                                (selected_sido_code,)
                            )
                            db_rows = cur.fetchall()
                            conn.close()
                            
                            if db_rows:
                                all_dfs = []
                                for row in db_rows:
                                    try:
                                        data_json = row[0]
                                        df = pd.read_json(data_json, orient="records")
                                        all_dfs.append(df)
                                    except Exception:
                                        continue
                                
                                if all_dfs:
                                    combined_df = pd.concat(all_dfs, ignore_index=True)
                                    chat_entry["total_count"] = len(combined_df)
                        
                        st.session_state.chat_history.append(chat_entry)
                    st.rerun()
            
            # 채팅 입력 폼 (엔터 키로 전송 가능)
            with st.form(key="chat_form", clear_on_submit=True):
                user_input = st.text_input(
                    "메시지를 입력하세요:",
                    key="chat_input",
                    value=st.session_state.get("chat_input_value", "")
                )
                
                # 전송 버튼 (폼 내부에서 엔터 키로도 제출 가능)
                submitted = st.form_submit_button(
                    "전송",
                    disabled=not can_send,
                    use_container_width=True
                )
                
                # 폼 제출 시 메시지 전송 (user_input을 인자로 전달 — session_state.chat_input 미사용)
                if submitted:
                    if user_input and user_input.strip() and can_send:
                        send_message(user_input)
            
            if not can_send and st.session_state.chat_mode in ["1:1대화", "5:1대화"]:
                st.info("가상인구를 선택한 후 메시지를 보낼 수 있습니다.")
            
            # 채팅 히스토리 초기화 버튼
            if st.button("대화 초기화", key="chat_clear"):
                st.session_state.chat_history = []
                st.rerun()
