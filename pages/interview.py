"""
심층면접 페이지 (Process 2 - FGI/IDI)
- 가상인구 DB(페르소나·현시대 반영된 데이터)를 지역별로 불러와 테이블로 표시
"""
import streamlit as st
from pages.common import apply_common_styles, render_common_input_form
from core.constants import SIDO_LABEL_TO_CODE, SIDO_CODE_TO_NAME
from core.db import get_virtual_population_db_combined_df
from core.session_cache import get_sido_master


def page_interview():
    """심층면접 페이지 (Process 2 - FGI/IDI)"""
    apply_common_styles()

    # 지역 선택 (가상인구 DB 페이지와 동일)
    sido_master = get_sido_master()
    sido_options = [f"{s['sido_name']} ({s['sido_code']})" for s in sido_master]
    selected_sido_label = st.selectbox(
        "지역 선택",
        options=sido_options,
        index=sido_options.index(st.session_state.get("interview_sido_label", "경상북도 (37)"))
        if st.session_state.get("interview_sido_label", "경상북도 (37)") in sido_options else 0,
        key="interview_sido_select",
    )
    selected_sido_code = SIDO_LABEL_TO_CODE.get(selected_sido_label, "37")
    selected_sido_name = SIDO_CODE_TO_NAME.get(selected_sido_code, "경상북도")
    st.session_state.interview_sido_label = selected_sido_label

    st.markdown("---")

    # 선택 지역의 가상인구 DB 데이터 (페르소나·현시대 반영된 모든 기록 누적)
    st.markdown("### 가상인구 DB")
    try:
        combined_df = get_virtual_population_db_combined_df(selected_sido_code)
    except Exception as e:
        st.error(f"가상인구 DB를 불러오는 중 오류 발생: {e}")
        combined_df = None

    if combined_df is not None and len(combined_df) > 0:
        total_n = len(combined_df)
        n_cols = len(combined_df.columns)
        if "페르소나" in combined_df.columns:
            persona_filled = combined_df["페르소나"].fillna("").astype(str).str.strip()
            persona_count = (persona_filled != "").sum()
        else:
            persona_count = 0

        st.caption(f"총 {total_n:,}명의 데이터 (모든 기록 누적)")
        st.caption(f"페르소나·현시대 반영된 목록: {persona_count}명 / 전체 {total_n:,}명")

        st.session_state.interview_population_df = combined_df
        st.success(f"✅ {selected_sido_name} 가상인구 DB에서 불러왔습니다. (총 {total_n:,}명, {n_cols}개 컬럼)")

        with st.expander("불러온 데이터 (클릭하여 전체 보기)", expanded=False):
            st.dataframe(combined_df, use_container_width=True, height=400)
            st.caption(f"전체 {total_n:,}명 데이터")
    else:
        st.info(f"{selected_sido_name} 지역에 페르소나·현시대 반영된 데이터가 없습니다. 가상인구 DB 페이지에서 2차 대입결과를 추가한 뒤 전체 학습(페르소나·현시대 반영)을 실행하면 여기서 불러올 수 있습니다.")
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
