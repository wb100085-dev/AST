"""
심층면접 페이지 (Process 2 - FGI/IDI)
"""
# Streamlit 자동 페이지 감지 방지: 이 파일은 app.py에서 직접 호출됨
import streamlit as st
import pandas as pd
import os
from pages.common import apply_common_styles, render_common_input_form
from utils.step2_records import list_step2_records, load_step2_record


def page_interview():
    """심층면접 페이지 (Process 2 - FGI/IDI)"""
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
            index=0 if not st.session_state.get("interview_selected_record_label") else 
                  (list(record_options.keys()).index(st.session_state.interview_selected_record_label) 
                   if st.session_state.interview_selected_record_label in record_options else 0),
            key="interview_record_select"
        )
        
        if selected_label and selected_label in record_options:
            selected_record = record_options[selected_label]
            excel_path = selected_record.get("excel_path", "")
            
            if excel_path and os.path.isfile(excel_path):
                try:
                    df = load_step2_record(excel_path)
                    st.session_state.interview_population_df = df
                    st.session_state.interview_selected_record_label = selected_label
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
        st.session_state.interview_population_df = None
    
    st.markdown("---")
    
    # 메인 타이틀
    st.markdown("""
    <div class="survey-header">
        <h1 style="font-size: 32px; font-weight: 800; color: #0f172a; margin-bottom: 12px;">
            새로운 심층면접 프로젝트 시작하기
        </h1>
        <p style="color: #64748b; font-size: 16px; line-height: 1.6;">
            제품과 니즈를 상세히 적어주실수록, AI가 최적의 인터뷰 방식을 추천하고 질문 가이드를 생성합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 좌우 분할 레이아웃
    left_col, right_col = st.columns([0.65, 0.35], gap="large")
    
    with left_col:
        is_definition_valid, is_needs_valid = render_common_input_form()
        is_all_valid = is_definition_valid and is_needs_valid
    
    with right_col:
        # AI 인터뷰 설계 제안 카드
        if is_all_valid:
            card_class = "survey-card-indigo"
        else:
            card_class = "survey-card"
        
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 24px;">
            <span style="color: #4f46e5; font-weight: 700; font-size: 16px;">AI 인터뷰 설계 제안</span>
        </div>
        """, unsafe_allow_html=True)
        
        if not is_all_valid:
            st.warning("필수 정보를 300자 이상 입력하시면 AI가 최적의 인터뷰 방식을 제안합니다.")
        else:
            st.markdown("**인터뷰 방식 선택:**")
            
            # 인터뷰 방식 선택
            interview_type = st.radio(
                "인터뷰 방식",
                options=["FGI (Focus Group Interview)", "FGD (Focus Group Discussion)", "IDI (Individual Depth Interview)"],
                key="interview_type_radio",
                help="FGI: 모더레이터 1 : 소비자 N | FGD: 소비자 N (비개입) | IDI: 1:1 심층 인터뷰"
            )
            st.session_state.interview_type = interview_type
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # AI 추천 메시지
            if interview_type == "FGI (Focus Group Interview)":
                st.info("**AI 추천:** 입력하신 니즈에는 FGI 방식이 적합합니다. 그룹 토론을 통해 다양한 관점을 수집할 수 있습니다.")
            elif interview_type == "FGD (Focus Group Discussion)":
                st.info("**AI 추천:** 입력하신 니즈에는 FGD 방식이 적합합니다. 자연스러운 소비자 간 대화에서 인사이트를 도출할 수 있습니다.")
            else:
                st.info("**AI 추천:** 입력하신 니즈에는 IDI 방식이 적합합니다. 개인별 심층적인 의견을 수집할 수 있습니다.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 모더레이터 질문 가이드 생성 버튼
            if st.button("모더레이터 질문 가이드 생성", type="primary", use_container_width=True, key="interview_guide_btn"):
                st.session_state.interview_guide_generated = True
            
            if st.session_state.get("interview_guide_generated", False):
                st.markdown("---")
                st.markdown("**생성된 모더레이터 질문 가이드:**")
                
                guide_sections = [
                    {
                        "section": "오프닝 (5분)",
                        "questions": [
                            "오늘 참석해주셔서 감사합니다. 먼저 간단히 자기소개 부탁드립니다.",
                            "이 제품에 대해 들어보신 적이 있으신가요?"
                        ]
                    },
                    {
                        "section": "본론 - 제품 인식 (15분)",
                        "questions": [
                            "이 제품을 처음 접했을 때 어떤 생각이 들었나요?",
                            "경쟁 제품과 비교했을 때 어떤 차이점이 느껴지시나요?",
                            "가장 매력적인 점과 아쉬운 점은 무엇인가요?"
                        ]
                    },
                    {
                        "section": "본론 - 구매 의도 (15분)",
                        "questions": [
                            "실제로 구매를 고려해보신 적이 있으신가요?",
                            "구매 결정에 가장 큰 영향을 미치는 요소는 무엇인가요?",
                            "가격이 구매 결정에 어떤 영향을 미치나요?"
                        ]
                    },
                    {
                        "section": "마무리 (5분)",
                        "questions": [
                            "이 제품을 주변 사람들에게 추천하시겠나요?",
                            "개선되었으면 하는 점이 있다면 무엇인가요?"
                        ]
                    }
                ]
                
                for section in guide_sections:
                    st.markdown(f"**{section['section']}**")
                    for q in section['questions']:
                        st.markdown(f"- {q}")
                    st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("데이터는 암호화되어 보호되며, 분석 완료 후 즉시 파기됩니다.")
