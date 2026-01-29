"""
설문조사 페이지 (Process 1)
"""
# Streamlit 자동 페이지 감지 방지: 이 파일은 app.py에서 직접 호출됨
import streamlit as st
import pandas as pd
import os
from pages.common import apply_common_styles, render_common_input_form
from utils.step2_records import list_step2_records, load_step2_record


def page_survey():
    """설문조사 페이지 (Process 1)"""
    apply_common_styles()
    
    # 가상인구 데이터 불러오기 섹션 (페이지 최상단에 표시)
    st.markdown("### 가상인구 데이터 불러오기")
    try:
        records = list_step2_records()
    except Exception as e:
        st.error(f"2차 대입 결과 목록을 불러오는 중 오류 발생: {e}")
        records = []
    
    if records:
        record_options = {}
        for r in records:
            ts = r.get("timestamp", "")
            sido_name = r.get("sido_name", "")
            rows = r.get("rows", 0)
            cols = r.get("columns_count", 0)
            label = f"{ts} | {sido_name} | {rows}명 | {cols}개 컬럼"
            record_options[label] = r
        
        selected_label = st.selectbox(
            "2차 대입 결과에서 가상인구 데이터를 선택하세요:",
            options=list(record_options.keys()),
            index=0 if not st.session_state.get("survey_selected_record_label") else 
                  (list(record_options.keys()).index(st.session_state.survey_selected_record_label) 
                   if st.session_state.survey_selected_record_label in record_options else 0),
            key="survey_record_select"
        )
        
        if selected_label and selected_label in record_options:
            selected_record = record_options[selected_label]
            excel_path = selected_record.get("excel_path", "")
            
            if excel_path and os.path.isfile(excel_path):
                try:
                    df = load_step2_record(excel_path)
                    st.session_state.survey_population_df = df
                    st.session_state.survey_selected_record_label = selected_label
                    st.success(f"✅ 가상인구 데이터를 불러왔습니다. (총 {len(df)}명, {len(df.columns)}개 컬럼)")
                    
                    with st.expander("불러온 데이터 미리보기"):
                        st.dataframe(df.head(20), use_container_width=True, height=300)
                        st.caption(f"전체 {len(df)}명 중 처음 20명 표시")
                except Exception as e:
                    st.error(f"데이터 로드 실패: {e}")
            else:
                st.warning("선택한 데이터 파일을 찾을 수 없습니다.")
    else:
        st.info("아직 2차 대입 결과가 없습니다. 가상인구 생성 후 2단계에서 통계를 대입하면 여기서 불러올 수 있습니다.")
        st.session_state.survey_population_df = None
    
    st.markdown("---")
    
    # 메인 타이틀
    st.markdown("""
    <div class="survey-header">
        <h1 style="font-size: 32px; font-weight: 800; color: #0f172a; margin-bottom: 12px;">
            새로운 설문조사 프로젝트 시작하기
        </h1>
        <p style="color: #64748b; font-size: 16px; line-height: 1.6;">
            제품과 니즈를 상세히 적어주실수록, AI가 더 정교한 설문 설계를 제안합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 좌우 분할 레이아웃
    left_col, right_col = st.columns([0.65, 0.35], gap="large")
    
    with left_col:
        is_definition_valid, is_needs_valid = render_common_input_form()
        is_all_valid = is_definition_valid and is_needs_valid
    
    with right_col:
        # AI 설문 설계 제안 카드
        if is_all_valid:
            card_class = "survey-card-indigo"
        else:
            card_class = "survey-card"
        
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 24px;">
            <span style="color: #4f46e5; font-weight: 700; font-size: 16px;">AI 설문 설계 제안</span>
        </div>
        """, unsafe_allow_html=True)
        
        if not is_all_valid:
            st.warning("필수 정보를 300자 이상 입력하시면 AI가 최적의 설문 설계를 제안합니다.")
        else:
            st.success("입력 정보가 충분합니다. 아래 버튼을 클릭하여 설문 설계를 시작하세요.")
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 가설 설정 및 설문 문항 생성 버튼
            if st.button("가설 설정", type="primary", use_container_width=True, key="survey_hypothesis_btn"):
                st.session_state.survey_hypothesis_set = True
            
            if st.session_state.get("survey_hypothesis_set", False):
                st.markdown("---")
                st.markdown("**설정된 가설:**")
                st.info("""
                **가설 1:** 타겟 유저는 가격 저항선이 높을 것으로 예상됩니다.
                
                **가설 2:** 경쟁사 대비 제품의 강점은 사용 편의성에 있을 것입니다.
                
                **가설 3:** 구매 결정에 가장 큰 영향을 미치는 요소는 브랜드 신뢰도입니다.
                """)
                
                if st.button("설문 문항 자동 생성", type="primary", use_container_width=True, key="survey_generate_btn"):
                    st.session_state.survey_questions_generated = True
            
            if st.session_state.get("survey_questions_generated", False):
                st.markdown("---")
                st.markdown("**생성된 설문 문항 (리커트 5점 척도):**")
                
                questions = [
                    {
                        "title": "가격 수용도",
                        "question": "이 제품의 가격이 적정하다고 생각하시나요?",
                        "scale": "1점(전혀 그렇지 않다) ~ 5점(매우 그렇다)"
                    },
                    {
                        "title": "사용 편의성",
                        "question": "이 제품의 사용 편의성에 만족하시나요?",
                        "scale": "1점(전혀 그렇지 않다) ~ 5점(매우 그렇다)"
                    },
                    {
                        "title": "브랜드 신뢰도",
                        "question": "이 브랜드를 신뢰할 수 있다고 생각하시나요?",
                        "scale": "1점(전혀 그렇지 않다) ~ 5점(매우 그렇다)"
                    }
                ]
                
                for i, q in enumerate(questions, 1):
                    with st.container():
                        st.markdown(f"""
                        <div style="background: white; padding: 20px; border-radius: 12px; margin-bottom: 16px; border: 1px solid #e0e7ff;">
                            <h4 style="color: #4f46e5; margin-bottom: 8px;">문항 {i}: {q['title']}</h4>
                            <p style="color: #0f172a; font-size: 14px; margin-bottom: 4px;">{q['question']}</p>
                            <p style="color: #64748b; font-size: 12px;">{q['scale']}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("데이터는 암호화되어 보호되며, 분석 완료 후 즉시 파기됩니다.")
