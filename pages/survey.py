"""
시장성 조사 설계 페이지
- AI 설계 제안 후 설문조사/심층면접을 같은 페이지 내 서브페이지로 전환
- 설문조사 서브: 설정된 가설 + 설문 문항 생성
- 심층면접 서브: 인터뷰 방식(FGI/FGD/IDI) + 모더레이터 질문 가이드
"""
import streamlit as st
from pages.common import apply_common_styles, render_definition_block, render_needs_block
from core.constants import SIDO_LABEL_TO_CODE, SIDO_CODE_TO_NAME
from core.db import get_virtual_population_db_combined_df
from core.session_cache import get_sido_master

# 서브페이지: ""(폼+AI 제안) | "설문조사" | "심층면접"
if "survey_design_subpage" not in st.session_state:
    st.session_state.survey_design_subpage = ""


def _get_survey_method_ai_suggestion(definition: str, needs: str) -> tuple[str | None, str]:
    """제품 정의·조사 니즈를 Gemini로 분석해 설문조사 vs 심층면접 제안. 반환: ('설문조사'|'심층면접'|None, 사유텍스트)."""
    try:
        from utils.gemini_client import GeminiClient
    except Exception:
        return None, "Gemini 클라이언트를 불러올 수 없습니다."
    prompt = f"""다음 두 가지 입력을 분석해서, 이 과제에 더 적합한 방법이 "설문조사"인지 "심층면접"인지 한 단어로만 판단해주세요.
판단 근거를 2~4문장으로 간단히 이어서 작성해주세요.

출력 형식 (따옴표 없이 그대로):
첫 줄: 설문조사 또는 심층면접 (반드시 이 두 단어 중 하나만)
둘째 줄: 빈 줄
셋째 줄부터: 판단 근거

[제품/서비스의 정의]
{definition[:3000]}

[조사의 목적과 니즈]
{needs[:3000]}
"""
    try:
        client = GeminiClient()
        resp = client._client.models.generate_content(
            model=client._model,
            contents=prompt,
        )
        text = (resp.text or "").strip()
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        method = None
        if lines:
            first = lines[0]
            if "설문조사" in first:
                method = "설문조사"
            elif "심층면접" in first:
                method = "심층면접"
        reason = "\n".join(lines[1:]) if len(lines) > 1 else text
        return method or None, reason or "판단 근거를 생성하지 못했습니다."
    except Exception as e:
        return None, f"AI 분석 중 오류: {str(e)}"


def _render_vpd_section():
    """가상인구 DB (지역별) 2:8 수평 블록"""
    st.markdown("---")
    st.markdown("### 가상인구 DB (지역별)")
    col_region, col_db = st.columns([0.2, 0.8], gap="large")
    with col_region:
        sido_master = get_sido_master()
        sido_options = [f"{s['sido_name']} ({s['sido_code']})" for s in sido_master]
        default_label = st.session_state.get("survey_sido_label", "경상북도 (37)")
        default_idx = sido_options.index(default_label) if default_label in sido_options else 0
        selected_sido_label = st.selectbox(
            "지역 선택",
            options=sido_options,
            index=default_idx,
            key="survey_sido_select",
        )
        selected_sido_code = SIDO_LABEL_TO_CODE.get(selected_sido_label, "37")
        selected_sido_name = SIDO_CODE_TO_NAME.get(selected_sido_code, "경상북도")
        st.session_state.survey_sido_label = selected_sido_label
    with col_db:
        try:
            combined_df = get_virtual_population_db_combined_df(selected_sido_code)
        except Exception as e:
            st.error(f"가상인구 DB를 불러오는 중 오류 발생: {e}")
            combined_df = None
        if combined_df is not None and len(combined_df) > 0:
            total_n = len(combined_df)
            n_cols = len(combined_df.columns)
            persona_count = (combined_df["페르소나"].fillna("").astype(str).str.strip() != "").sum() if "페르소나" in combined_df.columns else 0
            st.caption(f"총 {total_n:,}명 (페르소나·현시대 반영: {persona_count}명)")
            st.session_state.survey_population_df = combined_df
            st.success(f"✅ {selected_sido_name} 가상인구 DB에서 불러왔습니다. (총 {total_n:,}명, {n_cols}개 컬럼)")
            with st.expander("불러온 데이터 (클릭하여 전체 보기)", expanded=False):
                st.dataframe(combined_df, use_container_width=True, height=400)
                st.caption(f"전체 {total_n:,}명 데이터")
        else:
            st.info(f"{selected_sido_name} 지역에 페르소나·현시대 반영된 데이터가 없습니다. 가상인구 DB 페이지에서 2차 대입결과를 추가한 뒤 전체 학습(페르소나·현시대 반영)을 실행하면 여기서 불러올 수 있습니다.")
            st.session_state.survey_population_df = None


def _render_survey_subpage():
    """서브페이지: 설문조사 — 설정된 가설 + 설문 문항 생성"""
    if st.button("← 설계 선택으로", key="survey_back_to_design"):
        st.session_state.survey_design_subpage = ""
        st.rerun()
    st.markdown("---")
    st.markdown("**설문조사**")
    st.markdown("**설정된 가설:**")
    st.info("""
    **가설 1:** 타겟 유저는 가격 저항선이 높을 것으로 예상됩니다.

    **가설 2:** 경쟁사 대비 제품의 강점은 사용 편의성에 있을 것입니다.

    **가설 3:** 구매 결정에 가장 큰 영향을 미치는 요소는 브랜드 신뢰도입니다.
    """)
    if st.button("설문 문항 자동 생성", type="primary", use_container_width=True, key="survey_generate_btn"):
        st.session_state.survey_questions_generated = True
        st.rerun()

    if st.session_state.get("survey_questions_generated", False):
        st.markdown("---")
        st.markdown("**생성된 설문 문항 (리커트 5점 척도):**")
        questions = [
            {"title": "가격 수용도", "question": "이 제품의 가격이 적정하다고 생각하시나요?", "scale": "1점(전혀 그렇지 않다) ~ 5점(매우 그렇다)"},
            {"title": "사용 편의성", "question": "이 제품의 사용 편의성에 만족하시나요?", "scale": "1점(전혀 그렇지 않다) ~ 5점(매우 그렇다)"},
            {"title": "브랜드 신뢰도", "question": "이 브랜드를 신뢰할 수 있다고 생각하시나요?", "scale": "1점(전혀 그렇지 않다) ~ 5점(매우 그렇다)"},
        ]
        for i, q in enumerate(questions, 1):
            st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 12px; margin-bottom: 16px; border: 1px solid #e0e7ff;">
                <h4 style="color: #4f46e5; margin-bottom: 8px;">문항 {i}: {q['title']}</h4>
                <p style="color: #0f1a2d; font-size: 14px; margin-bottom: 4px;">{q['question']}</p>
                <p style="color: #64748b; font-size: 12px;">{q['scale']}</p>
            </div>
            """, unsafe_allow_html=True)


def _render_interview_subpage():
    """서브페이지: 심층면접 — 인터뷰 방식(FGI/FGD/IDI) + 모더레이터 질문 가이드"""
    if st.button("← 설계 선택으로", key="interview_back_to_design"):
        st.session_state.survey_design_subpage = ""
        st.rerun()
    st.markdown("---")
    st.markdown("""
    <div class="survey-header">
        <h1 style="font-size: 32px; font-weight: 800; color: #0f1a2d; margin-bottom: 12px;">
            새로운 심층면접 프로젝트 시작하기
        </h1>
        <p style="color: #64748b; font-size: 16px; line-height: 1.6;">
            제품과 니즈를 상세히 적어주실수록, AI가 최적의 인터뷰 방식을 추천하고 질문 가이드를 생성합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**인터뷰 방식 선택:**")
    interview_type = st.radio(
        "인터뷰 방식",
        options=["FGI (Focus Group Interview)", "FGD (Focus Group Discussion)", "IDI (Individual Depth Interview)"],
        key="interview_type_radio",
        help="FGI: 모더레이터 1 : 소비자 N | FGD: 소비자 N (비개입) | IDI: 1:1 심층 인터뷰"
    )
    st.session_state.interview_type = interview_type
    st.markdown("<br>", unsafe_allow_html=True)

    if interview_type == "FGI (Focus Group Interview)":
        st.info("**AI 추천:** 입력하신 니즈에는 FGI 방식이 적합합니다. 그룹 토론을 통해 다양한 관점을 수집할 수 있습니다.")
    elif interview_type == "FGD (Focus Group Discussion)":
        st.info("**AI 추천:** 입력하신 니즈에는 FGD 방식이 적합합니다. 자연스러운 소비자 간 대화에서 인사이트를 도출할 수 있습니다.")
    else:
        st.info("**AI 추천:** 입력하신 니즈에는 IDI 방식이 적합합니다. 개인별 심층적인 의견을 수집할 수 있습니다.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("모더레이터 질문 가이드 생성", type="primary", use_container_width=True, key="interview_guide_btn"):
        st.session_state.interview_guide_generated = True
        st.rerun()

    if st.session_state.get("interview_guide_generated", False):
        st.markdown("---")
        st.markdown("**생성된 모더레이터 질문 가이드:**")
        guide_sections = [
            {"section": "오프닝 (5분)", "questions": ["오늘 참석해주셔서 감사합니다. 먼저 간단히 자기소개 부탁드립니다.", "이 제품에 대해 들어보신 적이 있으신가요?"]},
            {"section": "본론 - 제품 인식 (15분)", "questions": ["이 제품을 처음 접했을 때 어떤 생각이 들었나요?", "경쟁 제품과 비교했을 때 어떤 차이점이 느껴지시나요?", "가장 매력적인 점과 아쉬운 점은 무엇인가요?"]},
            {"section": "본론 - 구매 의도 (15분)", "questions": ["실제로 구매를 고려해보신 적이 있으신가요?", "구매 결정에 가장 큰 영향을 미치는 요소는 무엇인가요?", "가격이 구매 결정에 어떤 영향을 미치나요?"]},
            {"section": "마무리 (5분)", "questions": ["이 제품을 주변 사람들에게 추천하시겠나요?", "개선되었으면 하는 점이 있다면 무엇인가요?"]},
        ]
        for section in guide_sections:
            st.markdown(f"**{section['section']}**")
            for q in section["questions"]:
                st.markdown(f"- {q}")
            st.markdown("<br>", unsafe_allow_html=True)
    st.caption("데이터는 암호화되어 보호되며, 분석 완료 후 즉시 파기됩니다.")


def page_survey():
    """시장성 조사 설계 페이지 — 폼 + AI 제안 또는 서브페이지(설문조사/심층면접)"""
    apply_common_styles()
    subpage = st.session_state.get("survey_design_subpage", "")

    # 서브페이지: 설문조사 → 설정된 가설 + 설문 문항 생성
    if subpage == "설문조사":
        _render_survey_subpage()
        _render_vpd_section()
        return

    # 서브페이지: 심층면접 → 인터뷰 방식 + 질문 가이드
    if subpage == "심층면접":
        _render_interview_subpage()
        _render_vpd_section()
        return

    # 기본: 폼 + AI 설계 제안 + 설문조사/심층면접 버튼 (같은 페이지 내 전환)
    st.markdown("""
    <div class="survey-header">
        <h1 style="font-size: 32px; font-weight: 800; color: #0f1a2d; margin-bottom: 12px;">
            새로운 설문조사 프로젝트 시작하기
        </h1>
        <p style="color: #64748b; font-size: 16px; line-height: 1.6;">
            제품과 니즈를 상세히 적어주실수록, AI가 더 정교한 설문 설계를 제안합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_def, col_needs = st.columns([0.5, 0.5], gap="large")
    with col_def:
        is_definition_valid = render_definition_block("_survey")
    with col_needs:
        is_needs_valid = render_needs_block("_survey")
    is_all_valid = is_definition_valid and is_needs_valid

    st.markdown("<br>", unsafe_allow_html=True)

    if "target" not in st.session_state:
        st.session_state.target = ""
    if "survey_target_pdf" not in st.session_state:
        st.session_state.survey_target_pdf = None

    st.markdown("### 타깃 (선택)")
    with st.expander("타깃 작성하기 (클릭하여 열기)", expanded=False):
        col_target_text, col_target_pdf = st.columns([0.5, 0.5], gap="large")
        with col_target_text:
            st.caption("텍스트로 타깃 설명")
            target = st.text_area(
                "희망 타깃",
                value=st.session_state.target,
                placeholder="특정 타깃이 있다면 적어주세요. (예: 30대 워킹맘)",
                height=120,
                key="survey_target_text"
            )
            st.session_state.target = target
        with col_target_pdf:
            st.caption("PDF 파일 업로드 (선택)")
            pdf_file = st.file_uploader(
                "PDF 업로드",
                type=["pdf"],
                key="survey_target_pdf_upload",
                label_visibility="collapsed"
            )
            if pdf_file is not None:
                st.session_state.survey_target_pdf = pdf_file
                st.success(f"업로드됨: {pdf_file.name}")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**AI 설문 설계 제안**")
    if not is_all_valid:
        st.warning("제품/서비스 정의와 조사의 목적·니즈를 각각 300자 이상 입력하시면 AI가 설문조사 vs 심층면접을 제안합니다.")
    else:
        if st.button("AI 설문 설계 제안", type="primary", use_container_width=True, key="survey_ai_design_btn"):
            with st.spinner("AI가 제품·조사 내용을 분석하고 있습니다…"):
                definition = st.session_state.get("definition", "")
                needs = st.session_state.get("needs", "")
                method, reason = _get_survey_method_ai_suggestion(definition, needs)
                st.session_state.survey_ai_suggestion_method = method
                st.session_state.survey_ai_suggestion_reason = reason
                st.session_state.survey_ai_done = True
            st.rerun()

    if st.session_state.get("survey_ai_done", False):
        ai_method = st.session_state.get("survey_ai_suggestion_method")
        ai_reason = st.session_state.get("survey_ai_suggestion_reason", "")
        col_ai, col_user = st.columns([0.5, 0.5], gap="large")
        with col_ai:
            st.markdown("**AI 제안**")
            st.info(ai_reason)
            if ai_method == "설문조사":
                if st.button("설문조사", type="primary", use_container_width=True, key="survey_btn_ai_survey"):
                    st.session_state.survey_design_subpage = "설문조사"
                    st.session_state.survey_method_chosen = "설문조사"
                    st.rerun()
            elif ai_method == "심층면접":
                if st.button("심층면접", type="primary", use_container_width=True, key="survey_btn_ai_interview"):
                    st.session_state.survey_design_subpage = "심층면접"
                    st.session_state.survey_method_chosen = "심층면접"
                    st.rerun()
            else:
                st.caption("AI가 설문조사/심층면접 중 하나를 제안하지 못했습니다. 오른쪽에서 직접 선택하세요.")
        with col_user:
            st.markdown("**사용자 선택**")
            st.caption("설문조사 또는 심층면접 중 진행할 방식을 선택하세요.")
            btn_survey, btn_interview = st.columns(2)
            with btn_survey:
                if st.button("설문조사", use_container_width=True, key="survey_btn_user_survey"):
                    st.session_state.survey_design_subpage = "설문조사"
                    st.session_state.survey_method_chosen = "설문조사"
                    st.rerun()
            with btn_interview:
                if st.button("심층면접", use_container_width=True, key="survey_btn_user_interview"):
                    st.session_state.survey_design_subpage = "심층면접"
                    st.session_state.survey_method_chosen = "심층면접"
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("데이터는 암호화되어 보호되며, 분석 완료 후 즉시 파기됩니다.")

    _render_vpd_section()
