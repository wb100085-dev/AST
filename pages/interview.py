"""
심층면접 페이지 (Process 2 - FGI/IDI)
- 가상인구 DB(페르소나·현시대 반영된 데이터)를 지역별로 불러와 테이블로 표시
"""
import streamlit as st
from pages.common import apply_common_styles, render_common_input_form
from core.constants import SIDO_LABEL_TO_CODE, SIDO_CODE_TO_NAME
from core.db import get_virtual_population_db_combined_df
from core.session_cache import get_sido_master
from utils.gemini_client import GeminiClient


def page_interview():
    """심층면접 페이지 (Process 2 - FGI/IDI)"""
    apply_common_styles()

    # 상단 내비게이션: 이전 페이지
    _, col_nav = st.columns([1, 0.2])
    with col_nav:
        if st.button("이전 페이지", key="interview_back_to_prev", type="secondary", use_container_width=True):
            st.switch_page("pages/survey.py")
    st.markdown("---")

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
        
        # 인터뷰 방식 상세 정보 표 (Markdown 표)
        st.markdown("**인터뷰 방식 안내**")
        st.markdown("""
| 방식 | 설명 | 특징 |
|---|---|---|
| **FGI** (Focus Group Interview) | 진행자(Moderator) 주도로 소수의 참여자(6~12인)를 대상으로 특정 주제에 대해 인터뷰를 진행하여 정보를 수집하는 방식 | - 진행자가 사전에 준비된 가이드라인에 따라 질문을 던지고 답변을 청취<br>- 다수의 의견을 효율적으로 수집 가능, 공통된 인식/패턴 파악 유리<br>- 진행자와 참여자 간의 문답이 주를 이룸 |
| **FGD** (Focus Group Discussion) | FGI와 유사하나, 진행자의 개입을 최소화하고 참여자 간의 자유로운 토론과 상호작용을 통해 데이터를 도출 | - 집단 역학(Group Dynamics) 활용, 사회적 규범/합의 형성 과정 관찰<br>- 참여자 간 의견 충돌/상호 보완을 통해 심층적 맥락 발견 가능<br>- 진행자는 촉진자(Facilitator) 역할 수행 |
| **IDI** (Individual Depth Interview) | 연구자와 대상자가 1:1로 대면하여 개인의 경험, 감정, 신념 등을 깊이 있게 탐구하는 방식 | - 타인의 시선 없이 민감한 주제/개인적 경험 청취 적합<br>- 무의식적 반응이나 복잡한 행동 동기 심층 파악 가능<br>- 정보의 밀도와 정확성이 매우 높음 |
        """)
        st.markdown("<br>", unsafe_allow_html=True)

        if not is_all_valid:
            st.warning("필수 정보를 300자 이상 입력하시면 AI가 최적의 인터뷰 방식을 제안합니다.")
            interview_options = ["FGI (Focus Group Interview)", "FGD (Focus Group Discussion)", "IDI (Individual Depth Interview)"]
            interview_type = st.radio(
                "인터뷰 방식",
                options=interview_options,
                key="interview_type_radio",
                help="FGI: 모더레이터 1 : 소비자 N | FGD: 소비자 N (비개입) | IDI: 1:1 심층 인터뷰"
            )
            st.session_state.interview_type = interview_type
        else:
            # AI 추천: definition, needs 기반으로 Gemini 호출 (입력이 바뀌면 재계산)
            definition = st.session_state.get("definition", "")
            needs = st.session_state.get("needs", "")
            input_hash = hash((definition, needs))
            if (
                st.session_state.get("interview_ai_rec_input_hash") != input_hash
                or "interview_ai_rec_method" not in st.session_state
            ):
                try:
                    client = GeminiClient()
                    recommended_method, reason = client.recommend_interview_method(definition, needs)
                    st.session_state.interview_ai_rec_input_hash = input_hash
                    st.session_state.interview_ai_rec_method = recommended_method
                    st.session_state.interview_ai_rec_reason = reason
                except Exception as e:
                    st.session_state.interview_ai_rec_input_hash = input_hash
                    st.session_state.interview_ai_rec_method = "FGI (Focus Group Interview)"
                    st.session_state.interview_ai_rec_reason = f"추천 생성 중 오류: {e}"
            recommended_method = st.session_state.interview_ai_rec_method
            reason = st.session_state.interview_ai_rec_reason
            st.success(f"**AI 추천: {recommended_method}** — {reason}")
            st.markdown("<br>", unsafe_allow_html=True)

            interview_options = ["FGI (Focus Group Interview)", "FGD (Focus Group Discussion)", "IDI (Individual Depth Interview)"]
            default_idx = interview_options.index(recommended_method) if recommended_method in interview_options else 0
            interview_type = st.radio(
                "인터뷰 방식 선택",
                options=interview_options,
                index=default_idx,
                key="interview_type_radio",
                help="FGI: 모더레이터 1 : 소비자 N | FGD: 소비자 N (비개입) | IDI: 1:1 심층 인터뷰"
            )
            st.session_state.interview_type = interview_type

        st.markdown("<br>", unsafe_allow_html=True)

        # 모더레이터 질문 가이드 생성 버튼 (Gemini 호출)
        if st.button("모더레이터 질문 가이드 생성", type="primary", use_container_width=True, key="interview_guide_btn"):
            definition = st.session_state.get("definition", "")
            needs = st.session_state.get("needs", "")
            itype = st.session_state.get("interview_type", "FGI (Focus Group Interview)")
            with st.spinner("AI가 질문 가이드를 생성하는 중입니다..."):
                try:
                    client = GeminiClient()
                    guide_markdown = client.generate_interview_questions(itype, definition, needs)
                    st.session_state.interview_guide_markdown = guide_markdown
                    st.session_state.interview_guide_generated = True
                except Exception as e:
                    st.error(f"질문 가이드 생성 중 오류가 발생했습니다: {e}")
                    st.session_state.interview_guide_markdown = ""
                    st.session_state.interview_guide_generated = False

        if st.session_state.get("interview_guide_generated", False):
            st.markdown("---")
            st.markdown("**생성된 모더레이터 질문 가이드:**")
            guide_markdown = st.session_state.get("interview_guide_markdown", "")
            if guide_markdown:
                st.markdown(guide_markdown)
            else:
                st.caption("(내용 없음)")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("데이터는 암호화되어 보호되며, 분석 완료 후 즉시 파기됩니다.")
