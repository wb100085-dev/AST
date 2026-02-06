"""
시장성 조사 설계 페이지
- AI 설계 제안 후 설문조사/심층면접을 같은 페이지 내 서브페이지로 전환
- 설문조사 서브: 설정된 가설 + 설문 문항 생성
- 심층면접 서브: 인터뷰 방식(FGI/FGD/IDI) + 모더레이터 질문 가이드
"""
import re
import streamlit as st
import pandas as pd
from pages.common import apply_common_styles, render_definition_block, render_needs_block
from core.constants import SIDO_LABEL_TO_CODE, SIDO_CODE_TO_NAME, SIDO_MASTER, SIDO_TOTAL_POP
from core.session_cache import get_sido_master


def _render_panel_summary_and_charts(panel_df, key_prefix: str = ""):
    """불러온 가상인구 요약(메트릭) + 6축 그래프를 생성 페이지와 동일한 형식으로 렌더. key_prefix로 위젯 키 중복 방지."""
    import plotly.graph_objects as go

    if panel_df is None or panel_df.empty:
        st.info("가상인구를 불러온 후 요약과 그래프가 표시됩니다.")
        return

    # AI 설문 진행 옵션 선택 페이지용 6축 그래프 높이 (기준 300의 70% = 210)
    chart_height = 210
    k = key_prefix or "survey"
    df = panel_df
    n = len(df)

    # ----- 요약 지표 (생성 페이지와 동일 형식) -----
    st.markdown("#### 요약 지표")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("총 인구수", f"{n:,}명")
    with c2:
        if "성별" in df.columns:
            gender_counts = df["성별"].value_counts()
            male = gender_counts.get("남자", gender_counts.get("남", 0))
            st.metric("남성 비율", f"{(male / n * 100):.1f}%" if n else "N/A")
        else:
            st.metric("남성 비율", "N/A")
    with c3:
        if "연령" in df.columns:
            age_ser = pd.to_numeric(df["연령"], errors="coerce").dropna()
            st.metric("평균 연령", f"{age_ser.mean():.1f}세" if len(age_ser) else "N/A")
        else:
            st.metric("평균 연령", "N/A")
    with c4:
        st.metric("컬럼 수", len(df.columns))

    st.markdown("---")
    st.markdown("#### 인구통계 그래프")

    # 생성 페이지와 동일: 색상 적용, 연령은 인구 피라미드(종모양)
    def _draw_population_pyramid_survey(_df, key_suffix: str):
        """연령·성별 인구 피라미드 (생성 페이지와 동일 형식)."""
        if "성별" not in _df.columns or "연령" not in _df.columns:
            st.caption("성별 또는 연령 데이터가 없습니다.")
            return
        age_num = pd.to_numeric(_df["연령"], errors="coerce")
        _df_f = _df.loc[age_num >= 20].copy()
        _df_f["연령"] = age_num.loc[_df_f.index]
        male = _df_f[_df_f["성별"].astype(str).str.strip().isin(["남자", "남"])]
        female = _df_f[_df_f["성별"].astype(str).str.strip().isin(["여자", "여"])]
        male_counts = male["연령"].value_counts().sort_index()
        female_counts = female["연령"].value_counts().sort_index()
        all_ages = sorted(set(male_counts.index) | set(female_counts.index))
        all_ages = [a for a in all_ages if 20 <= a <= 120]
        male_vals = [-male_counts.get(a, 0) for a in all_ages]
        female_vals = [female_counts.get(a, 0) for a in all_ages]
        total = len(_df_f) or 1
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=all_ages, x=male_vals, name="남자", orientation="h",
            marker=dict(color="rgba(31, 119, 180, 0.8)"),
            customdata=[[abs(c), (abs(c) / total * 100)] for c in male_vals],
            hovertemplate="<b>남자</b><br>연령: %{y}세<br>인구수: %{customdata[0]:,}명<br>비율: %{customdata[1]:.2f}%<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            y=all_ages, x=female_vals, name="여자", orientation="h",
            marker=dict(color="rgba(255, 127, 14, 0.8)"),
            customdata=[[c, (c / total * 100)] for c in female_vals],
            hovertemplate="<b>여자</b><br>연령: %{y}세<br>인구수: %{customdata[0]:,}명<br>비율: %{customdata[1]:.2f}%<extra></extra>",
        ))
        max_abs = max(max(abs(m) for m in male_vals) if male_vals else 0, max(female_vals) if female_vals else 0) or 1
        step = max(1, int(max_abs / 10))
        tick_vals = list(range(-int(max_abs), int(max_abs) + step, step))
        fig.update_layout(
            xaxis=dict(title="인구수 (명)", tickvals=tick_vals, ticktext=[str(abs(t)) for t in tick_vals], range=[-max_abs * 1.1, max_abs * 1.1]),
            yaxis=dict(title="연령 (세)"),
            barmode="overlay",
            height=chart_height,
            margin=dict(l=50, r=50, t=20, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="closest",
        )
        st.plotly_chart(fig, use_container_width=True, key=key_suffix)

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.markdown("**거주지역**")
        if "거주지역" in df.columns:
            vc = df["거주지역"].value_counts()
            max_count = vc.max()
            colors = [f"rgba(31, 119, 180, {0.3 + 0.7 * (c / max_count)})" for c in vc.values]
            fig = go.Figure(data=[go.Bar(x=vc.index, y=vc.values, marker_color=colors, text=vc.values, textposition="auto")])
            fig.update_layout(height=chart_height, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="", yaxis_title="인구수")
            st.plotly_chart(fig, use_container_width=True, key=f"{k}_panel_region")
        else:
            st.caption("데이터 없음")
    with row1_c2:
        st.markdown("**성별**")
        if "성별" in df.columns:
            vc = df["성별"].value_counts()
            max_count = vc.max()
            colors = [f"rgba(255, 127, 14, {0.3 + 0.7 * (c / max_count)})" for c in vc.values]
            fig = go.Figure(data=[go.Bar(x=vc.index, y=vc.values, marker_color=colors, text=vc.values, textposition="auto")])
            fig.update_layout(height=chart_height, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="", yaxis_title="인구수")
            st.plotly_chart(fig, use_container_width=True, key=f"{k}_panel_gender")
        else:
            st.caption("데이터 없음")

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        st.markdown("**연령 (인구 피라미드)**")
        if "연령" in df.columns and "성별" in df.columns:
            _draw_population_pyramid_survey(df, f"{k}_panel_age")
        else:
            st.caption("데이터 없음")
    with row2_c2:
        st.markdown("**경제활동**")
        if "경제활동" in df.columns:
            vc = df["경제활동"].value_counts()
            max_count = vc.max()
            colors = [f"rgba(214, 39, 40, {0.3 + 0.7 * (c / max_count)})" for c in vc.values]
            fig = go.Figure(data=[go.Bar(x=vc.index, y=vc.values, marker_color=colors, text=vc.values, textposition="auto")])
            fig.update_layout(height=chart_height, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="", yaxis_title="인구수")
            st.plotly_chart(fig, use_container_width=True, key=f"{k}_panel_econ")
        else:
            st.caption("데이터 없음")

    row3_c1, row3_c2 = st.columns(2)
    with row3_c1:
        st.markdown("**교육정도**")
        if "교육정도" in df.columns:
            vc = df["교육정도"].value_counts()
            max_count = vc.max()
            colors = [f"rgba(148, 103, 189, {0.3 + 0.7 * (c / max_count)})" for c in vc.values]
            fig = go.Figure(data=[go.Bar(x=vc.index, y=vc.values, marker_color=colors, text=vc.values, textposition="auto")])
            fig.update_layout(height=chart_height, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="", yaxis_title="인구수")
            st.plotly_chart(fig, use_container_width=True, key=f"{k}_panel_edu")
        else:
            st.caption("데이터 없음")
    with row3_c2:
        st.markdown("**월평균소득**")
        if "월평균소득" in df.columns:
            vc = df["월평균소득"].value_counts()
            order = ["50만원미만", "50-100만원", "100-200만원", "200-300만원", "300-400만원", "400-500만원", "500-600만원", "600-700만원", "700-800만원", "800만원이상"]
            vc = vc.reindex([x for x in order if x in vc.index], fill_value=0)
            max_count = vc.max() if len(vc) else 1
            colors = [f"rgba(140, 86, 75, {0.3 + 0.7 * (c / max_count)})" for c in vc.values]
            fig = go.Figure(data=[go.Bar(x=vc.index, y=vc.values, marker_color=colors, text=vc.values, textposition="auto")])
            fig.update_layout(height=chart_height, margin=dict(l=0, r=0, t=30, b=80), xaxis_tickangle=-45, xaxis_title="", yaxis_title="인구수")
            st.plotly_chart(fig, use_container_width=True, key=f"{k}_panel_income")
        else:
            st.caption("데이터 없음")

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


def _generate_hypotheses_with_gemini(definition: str, needs: str) -> list[str]:
    """제품 정의·조사 니즈를 Gemini로 분석해 가설 3개 생성. 반환: [가설1, 가설2, 가설3]"""
    try:
        from utils.gemini_client import GeminiClient
    except Exception:
        return [
            "타겟 유저의 니즈와 제품 특성을 반영한 가설 1입니다.",
            "타겟 유저의 니즈와 제품 특성을 반영한 가설 2입니다.",
            "타겟 유저의 니즈와 제품 특성을 반영한 가설 3입니다.",
        ]
    prompt = f"""당신은 시장성 조사 설계 전문가입니다. 아래 [제품/서비스의 정의]와 [조사의 목적과 니즈]를 바탕으로, 다음 주의사항을 반드시 지키며 검증 가능한 가설을 정확히 3개 작성해주세요.

[주의사항 – 반드시 준수]

1. 검증 가능성 (Testable & Measurable)
- 측정 가능한 데이터로 변환: "고객들이 우리 제품을 좋아할 것이다"처럼 모호한 문장은 피하고, "30대 직장인 여성 100명 중 60% 이상이 1만 원대 가격에 구매 의향을 보일 것이다"처럼 수치화·측정 가능한 형태로 작성하세요.
- 데이터 확보 가능성: 설문·데이터 분석 등으로 수집 가능한 데이터로 검증할 수 있어야 합니다.

2. 구체적이고 명확한 표현 (Specific & Clear)
- 구체적인 대상과 변수: 누가(Target), 무엇을(Product/Feature), 어떻게(Action) 할 것인가를 포함하세요. 예: "20대 남성(대상)은 편리한 인터페이스(변수)를 갖춘 앱을 더 자주 이용할 것이다(결과)."
- 단순함 유지: 하나의 가설에는 하나의 핵심 아이디어만 담으세요.

3. 주관적 편향 경계 (Avoid Confirmation Bias)
- 답을 정해놓지 말고, "가설은 틀릴 수 있다"는 전제로 작성하세요.
- 필요 시 귀무가설(Null Hypothesis) 활용: "가격 인상은 매출에 영향을 주지 않는다"처럼 '변화가 없다'는 가설을 세워 기각하는 방식도 고려하세요.

4. 실현 가능성과 연관성 (Relevant to Business Problem)
- 가설은 현재 해결하고자 하는 비즈니스 핵심 문제와 직결되어야 합니다.
- 가장 중요하고 시급한 것부터 설정하고, 동시에 너무 많은 가설을 검증하려 하지 마세요.

5. 행동 중심의 구체적 설계 (XYZ 가설 모델)
- "X(타겟 고객)의 Y%(몇 퍼센트)는 Z(어떤 행동/구매)를 할 것이다" 형식을 활용하면 좋습니다.

[출력 형식] 가설 1:, 가설 2:, 가설 3: 로 시작하고 각각 한 줄에 하나씩만 작성하세요.

[제품/서비스의 정의]
{definition[:3000]}

[조사의 목적과 니즈]
{needs[:3000]}
"""
    try:
        client = GeminiClient()
        resp = client._client.models.generate_content(model=client._model, contents=prompt)
        text = (resp.text or "").strip()
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        hypotheses = []
        for line in lines:
            for prefix in ("가설 1:", "가설 2:", "가설 3:", "가설1:", "가설2:", "가설3:"):
                if line.startswith(prefix):
                    hypotheses.append(line[len(prefix):].strip() or line)
                    break
            else:
                if len(hypotheses) < 3 and line and not line.startswith("가설"):
                    hypotheses.append(line)
        if len(hypotheses) >= 3:
            return hypotheses[:3]
        if len(hypotheses) > 0:
            while len(hypotheses) < 3:
                hypotheses.append("(추가 가설)")
            return hypotheses[:3]
    except Exception:
        pass
    return [
        "타겟 유저의 니즈와 제품 특성을 반영한 가설 1입니다.",
        "타겟 유저의 니즈와 제품 특성을 반영한 가설 2입니다.",
        "타겟 유저의 니즈와 제품 특성을 반영한 가설 3입니다.",
    ]


def _generate_survey_questions_with_gemini(hypotheses: list[str]) -> list[dict]:
    """
    설정된 가설을 Gemini로 분석해 가설을 모두 증명할 수 있는 설문 문항 생성.
    전부 객관식만 생성 (주관식 없음). 15~25개.
    반환: [{"type": "객관식", "title": "", "question": "", "options": ["A","B",...]}, ...]
    """
    try:
        from utils.gemini_client import GeminiClient
    except Exception:
        return _default_survey_questions()
    hyp_text = "\n".join([f"가설 {i+1}: {h}" for i, h in enumerate(hypotheses[:3], 1)])
    prompt = f"""당신은 시장성 조사 설계 전문가입니다. 아래 설정된 가설 3개를 정밀하게 분석한 뒤, 다음 주의사항을 반드시 지키며 가설을 모두 검증할 수 있는 설문 문항을 체계적으로 작성해주세요.

[중요] 모든 문항은 반드시 객관식만 작성하세요. 주관식 문항은 사용하지 마세요. 각 문항마다 선택지(①~⑤)를 반드시 포함하세요.

반드시 15개 이상 25개 이하의 문항을 출력하세요.

[주의사항 – 설문 문항 작성 시 반드시 준수]

1. 문항 구성 및 내용 (Structure & Content)
- 명확한 목표 정의: 무엇을 알고 싶은지 명확히 하고, 타겟 고객·구매 요인·가격 저항선 등 핵심 정보에 집중하세요.
- 스크리닝(선별) 질문 포함: 조사 대상이 아닌 사람을 걸러내기 위해 인구통계학적 특성·특정 경험 유무를 묻는 스크리닝 질문을 설문 처음에 배치하세요.
- 이중 질문(Double-barreled) 금지: "가격과 품질에 만족하십니까?"처럼 두 가지를 한 문항으로 묻지 말고, 각각 별도 문항으로 물어보세요.
- 유도 질문 회피: 답변을 유도하는 비중립적 표현은 사용하지 마세요.

2. 문항 표현 및 어휘 (Wording & Language)
- 쉽고 명확한 용어: 전문 용어·업계 은어·약어를 피하고, 누구나 이해할 수 있는 일상어를 사용하세요.
- 단순하고 구체적인 문장: 질문은 짧고 명확하게 작성하여 오해가 없도록 하세요.
- 부정문·이중 부정 피하기: "...하지 않은 것은?" 같은 질문은 피하세요.

3. 답변 항목 설계 (Answer Options) — 모든 문항 객관식
- 상호 배타적·포괄적 보기: 보기가 서로 겹치지 않게 하고, 가능한 응답 범위를 모두 포함하세요.
- 균형 잡힌 척도: 만족도는 '매우 만족'~'매우 불만족'까지 균형 있게 배치하세요.
- '기타', '해당 없음' 포함: 응답자가 선택할 수 없는 상황을 피할 수 있도록 보완 문항을 넣으세요.
- 각 문항당 선택지 4~5개(또는 리커트 척도)를 반드시 작성하세요.

4. 설문 흐름 및 배열 (Flow & Order)
- 일반→구체: 가벼운 질문에서 시작해 점점 핵심·심층 질문으로 유도하세요.
- 민감한 질문은 후반부: 소득·개인 정보 등은 설문 후반에 배치하세요.

5. 기타 유의사항
- 응답자 행동 예측 질문 주의: "만약 …하면 구매하시겠습니까?" 식의 미래 행동 예측 질문은 과대평가될 수 있으므로, 과거 행동·현재 인식 위주로 질문하세요.

[작성 절차]
1) 각 가설을 분석하여 검증에 필요한 핵심 변인·하위 요인을 도출하세요.
2) 도출된 변인별로 측정할 문항을 배치하세요. 문항 간 내용 중복을 피하세요.
3) 모든 문항을 객관식으로 작성하고, 각 문항에 선택지(①~⑤)를 반드시 포함하세요.
4) 각 문항이 최소 하나의 가설 검증에 직접 기여하도록 하세요.

[문항 수 – 반드시 준수]
- 최소 15개 이상, 최대 25개 이하. 전부 객관식으로만 작성하세요.

[출력 형식] 아래 형식을 정확히 따르세요. 한 문항당 블록으로 구분하고, 제목에는 "문항 N" 또는 "N." 같은 번호를 붙이지 마세요.
- 제목·질문·선택지에 스크리닝 관련 부가 설명을 넣지 마세요. 순수한 제목·질문·선택지만 출력하세요.
---
유형: 객관식
제목: (문항 제목 한 줄, 번호 없이)
질문: (질문 내용)
선택지: ① (내용) ② (내용) ③ (내용) ④ (내용) ⑤ (해당 없음/기타 등)
---

[설정된 가설]
{hyp_text}
"""
    try:
        client = GeminiClient()
        resp = client._client.models.generate_content(model=client._model, contents=prompt)
        text = (resp.text or "").strip()
        questions = _parse_survey_questions_from_text(text)
        # (Test 버전: 검증 단계 생략하여 속도 우선)
        # 스크리닝/➡️ 등 부가 설명 제거
        questions = _strip_survey_question_annotations(questions)
        # 검토 후에도 15개 미만이면 기본 문항으로 보강
        if len(questions) < 15:
            default = _default_survey_questions()
            for j in range(15 - len(questions)):
                questions.append(default[j % len(default)])
        return questions[:25]
    except Exception:
        return _default_survey_questions()


def _questions_to_review_text(questions: list[dict]) -> str:
    """검토 요청용으로 설문 문항을 텍스트로 직렬화."""
    lines = []
    for i, q in enumerate(questions, 1):
        lines.append(f"--- 문항 {i} ---")
        lines.append(f"유형: {q.get('type', '객관식')}")
        lines.append(f"제목: {q.get('title', '')}")
        lines.append(f"질문: {q.get('question', '')}")
        opts = q.get("options") or []
        if opts:
            lines.append("선택지: " + " / ".join([f"① {o}" for o in opts]))
        else:
            lines.append("선택지: (없음)")
        lines.append("")
    return "\n".join(lines)


def _review_survey_questions_with_gemini(questions: list[dict]) -> list[dict]:
    """
    생성된 설문 문항을 Gemini로 검토하여 보기 누락·객관식에 보기 없음 등 이상을 보정.
    - 객관식인데 선택지가 비었거나 2개 미만이면 보기 보완
    - 선택지가 아예 없는 문항이 있으면 보기 추가
    - 형식 불일치·중복 등 정리 후 동일 형식으로 출력하도록 요청
    """
    if not questions:
        return questions
    try:
        from utils.gemini_client import GeminiClient
    except Exception:
        return questions
    review_input = _questions_to_review_text(questions)
    prompt = f"""당신은 설문조사 품질 검토 전문가입니다. 아래 설문 문항 목록을 검토해 다음 항목을 반드시 확인하고, 문제가 있으면 수정한 전체 설문을 동일한 출력 형식으로 작성해주세요.

[검토 항목]
1. 객관식 문항에 선택지(보기)가 반드시 2개 이상 있어야 합니다. 비었거나 1개뿐이면 적절한 보기(① ② ③ …)를 추가하세요.
2. 보기가 누락된 객관식 문항이 있으면 해당 문항의 보기를 보완하세요.
3. 예/아니오·유/무 등 이분형 질문은 반드시 양쪽 보기를 모두 포함하세요. (예: "현재 반려동물을 양육하고 계십니까?" → "예"와 "아니오" 둘 다 있어야 함. "아니오"만 있거나 "예"만 있으면 누락된 쪽을 반드시 추가하세요.)
4. 모든 객관식은 상호 배타적·포괄적 보기(예: 매우 그렇다 ~ 전혀 그렇지 않다, 또는 명확한 선택지)를 갖추세요. '기타', '해당 없음' 등이 필요하면 포함하세요.
5. 주관식은 선택지 없이 질문만 유지하세요.
6. 제목·질문·선택지가 명확히 구분되도록 하고, 출력 형식은 아래 "출력 형식"을 정확히 따르세요.
7. 제목·질문·선택지에 (➡️ ...), 스크리닝 시 활용 등 부가 설명이 있으면 제거하고 순수한 문항만 출력하세요.

[출력 형식] 수정이 필요한 문항만 고치고 나머지는 그대로 두되, 반드시 아래 형식으로 전체 문항을 출력하세요.
---
유형: 객관식
제목: (문항 제목)
질문: (질문 내용)
선택지: ① (내용) ② (내용) ③ (내용) ④ (내용) ⑤ (해당 없음 등)
---
유형: 주관식
제목: (문항 제목)
질문: (질문 내용)
---

[현재 설문 문항]
{review_input}
"""
    try:
        client = GeminiClient()
        resp = client._client.models.generate_content(model=client._model, contents=prompt)
        text = (resp.text or "").strip()
        reviewed = _parse_survey_questions_from_text(text)
        if reviewed:
            return reviewed
    except Exception:
        pass
    return questions


def _strip_survey_question_annotations(questions: list[dict]) -> list[dict]:
    """제목·질문·선택지에서 (➡️ ...), 스크리닝 관련 부가 설명 제거."""
    if not questions:
        return questions
    # (➡️ ...) 또는 스크리닝 언급 괄호 블록 제거
    ann = re.compile(r"\s*\(➡️[^)]*\)|\s*\([^)]*스크리닝[^)]*\)", re.IGNORECASE)
    out = []
    for q in questions:
        nq = dict(q)
        for key in ("title", "question"):
            if nq.get(key):
                nq[key] = ann.sub("", nq[key]).strip()
        if nq.get("options"):
            nq["options"] = [ann.sub("", o).strip() for o in nq["options"] if ann.sub("", o).strip()]
        out.append(nq)
    return out


def _parse_survey_questions_from_text(text: str) -> list[dict]:
    """Gemini 출력 텍스트에서 설문 문항 파싱."""
    import re
    questions = []
    blocks = re.split(r"---+\s*", text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        q = {"type": "객관식", "title": "", "question": "", "options": []}
        # 주관식 블록이어도 모두 객관식으로 통일(선택지 없으면 기본 보기 부여)
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        for line in lines:
            if line.startswith("제목:") or line.startswith("제목 :"):
                q["title"] = line.split(":", 1)[-1].strip()
            elif line.startswith("질문:") or line.startswith("질문 :"):
                q["question"] = line.split(":", 1)[-1].strip()
            elif "선택지" in line:
                rest = line.split(":", 1)[-1].strip() if ":" in line else line
                for sep in ["①", "②", "③", "④", "⑤", "⑥", "1.", "2.", "3.", "4.", "5."]:
                    rest = rest.replace(sep, "\n")
                opts = [o.strip() for o in rest.split("\n") if o.strip() and len(o.strip()) > 1]
                q["options"] = opts[:6]
        if q["question"]:
            if not q["options"]:
                q["options"] = ["매우 그렇다", "그렇다", "보통", "그렇지 않다", "해당 없음"]
            questions.append(q)
    if len(questions) > 25:
        questions = questions[:25]
    if len(questions) < 15:
        # 15개 미만이면 기본 문항으로 보강
        default = _default_survey_questions()
        needed = 15 - len(questions)
        for j in range(needed):
            questions.append(default[j % len(default)])
    if not questions:
        return _default_survey_questions()
    return questions


def _default_survey_questions() -> list[dict]:
    """기본 설문 문항 (Gemini 실패 시). 전부 객관식."""
    return [
        {"type": "객관식", "title": "가격 수용도", "question": "이 제품의 가격이 적정하다고 생각하시나요?", "options": ["전혀 그렇지 않다", "그렇지 않다", "보통", "그렇다", "매우 그렇다"]},
        {"type": "객관식", "title": "사용 편의성", "question": "이 제품의 사용 편의성에 만족하시나요?", "options": ["전혀 그렇지 않다", "그렇지 않다", "보통", "그렇다", "매우 그렇다"]},
        {"type": "객관식", "title": "브랜드 신뢰도", "question": "이 브랜드를 신뢰할 수 있다고 생각하시나요?", "options": ["전혀 그렇지 않다", "그렇지 않다", "보통", "그렇다", "매우 그렇다"]},
        {"type": "객관식", "title": "개선 의견", "question": "이 제품에서 개선되었으면 하는 점이 있다고 생각하시나요?", "options": ["매우 그렇다", "그렇다", "보통", "그렇지 않다", "해당 없음"]},
    ]


def _build_respondent_profile(row: pd.Series, compact: bool = False) -> str:
    """패널 한 행에서 페르소나·현시대 반영·통계 요약 문자열 생성. compact=True면 설문 속도용 짧은 한~두 줄."""
    stat_cols = ["거주지역", "성별", "연령", "경제활동", "교육정도", "월평균소득", "가상이름"]
    stats = []
    for c in stat_cols:
        if c in row.index and pd.notna(row.get(c)) and str(row.get(c)).strip():
            stats.append(f"{c}: {row.get(c)}")
    # 연령 → 연령대 보조 (설문 '연령대' 문항 반영용)
    age_val = row.get("연령")
    if age_val is not None and str(age_val).strip():
        try:
            age_num = int(float(str(age_val).strip()))
            if age_num < 20:
                age_band = "20대 미만"
            elif age_num < 30:
                age_band = "20대"
            elif age_num < 40:
                age_band = "30대"
            elif age_num < 50:
                age_band = "40대"
            elif age_num < 60:
                age_band = "50대"
            elif age_num < 70:
                age_band = "60대"
            else:
                age_band = "70대 이상"
            if not any("연령대" in s for s in stats):
                stats.append(f"연령대: {age_band}")
        except (ValueError, TypeError):
            pass
    stats_str = ", ".join(stats) if stats else "특성 없음"
    if compact:
        persona = row.get("페르소나") or row.get("persona")
        ref = row.get("현시대 반영")
        _p = str(persona).strip() if pd.notna(persona) and str(persona).strip() else ""
        _r = str(ref).strip() if pd.notna(ref) and str(ref).strip() else ""
        p = (_p[:200] + "…") if len(_p) > 200 else _p
        r = (_r[:200] + "…") if len(_r) > 200 else _r
        skip_keys = {"페르소나", "persona", "현시대 반영"} | set(stat_cols)
        other_parts = []
        for c in row.index:
            if c in skip_keys or pd.isna(row.get(c)) or str(row.get(c)).strip() == "":
                continue
            other_parts.append(f"{c}={row.get(c)}")
            if len(other_parts) >= 12:
                break
        other_str = ", ".join(other_parts[:12]) if other_parts else ""
        parts = []
        if p:
            parts.append(f"[가장 중요] 페르소나(이 인물의 성격·가치관·입장): {p}")
        if r:
            parts.append(f"[가장 중요] 현시대 반영(현재 시대·트렌드 반영): {r}")
        parts.append(f"인구통계(연령·성별·소득 등 문항에 이 값 사용): {stats_str}")
        if other_str:
            parts.append(f"기타 특성·통계(해당 문항이 있으면 이 값에 맞는 선택지 사용): {other_str[:350]}")
        return " | ".join(parts)
    parts = []
    persona = row.get("페르소나") or row.get("persona")
    if pd.notna(persona) and str(persona).strip():
        parts.append(f"페르소나: {str(persona).strip()[:400]}")
    ref = row.get("현시대 반영")
    if pd.notna(ref) and str(ref).strip():
        parts.append(f"현시대 반영: {str(ref).strip()[:300]}")
    if stats:
        parts.append("통계: " + stats_str)
    return "\n".join(parts) if parts else "특성 정보 없음"


def _generate_one_respondent_answers_gemini(panel_row: pd.Series, questions: list[dict]) -> list[str] | None:
    """한 명의 가상인구에 대해 Gemini로 설문 응답 생성. (Test 버전: 속도 우선) 객관식=선택지 번호만, 주관식=50자 내외."""
    try:
        from utils.gemini_client import GeminiClient
    except Exception:
        return None
    profile = _build_respondent_profile(panel_row, compact=True)
    q_lines = []
    for i, q in enumerate(questions[:25], 1):
        typ = q.get("type", "객관식")
        title = q.get("title", "")
        question = q.get("question", "")
        opts = q.get("options") or []
        if typ == "객관식" and opts:
            opts_str = " / ".join([f"{j}:{o}" for j, o in enumerate(opts, 1)])
            q_lines.append(f"[문항{i}] {title}\n질문: {question}\n선택지(번호만 입력): {opts_str}")
        else:
            opts_str = " / ".join([f"{j}:{o}" for j, o in enumerate(opts or ["해당 없음"], 1)])
            q_lines.append(f"[문항{i}] {title}\n질문: {question}\n선택지(번호만 입력): {opts_str}")
    nq = min(len(questions), 25)
    prompt = f"""당신은 아래 프로필의 가상 응답자입니다. 모든 문항은 객관식이므로 선택지 번호(1,2,3,...) 하나만 입력하세요. 다음 우선순위를 지키세요.

{profile}

[답변 우선순위 – 반드시 준수]
1) **페르소나·현시대 반영(가장 중요)**  
   의견·태도·만족도·선호·의도 등 심리/행동 문항은 위의 "페르소나"와 "현시대 반영"에 맞는 선택지를 고르세요.

2) **인구통계 문항**  
   연령·연령대·성별·소득·교육·경제활동 등을 묻는 문항은 프로필의 "인구통계" 값에 해당하는 선택지 번호를 입력하세요.

3) **인구통계 외 특징·통계(컬럼)**  
   문항이 이 인물의 다른 특성(가구구성, 만족도, 소비경험, 자녀 유무, 직업·생활 관련 등)을 묻는다면 프로필의 "기타 특성·통계"에 있는 해당 컬럼 값에 맞는 선택지를 고르세요.

[출력 형식] Q1부터 Q{nq}까지 빠짐없이 한 줄에 Qn: (숫자 하나) 형태로만 출력하세요.
Q1: (선택지 번호)
Q2: (선택지 번호)
...
Q{nq}: (선택지 번호)

[설문 문항]
{chr(10).join(q_lines)}
"""
    try:
        client = GeminiClient()
        try:
            from google.genai import types
            config = types.GenerateContentConfig(max_output_tokens=1500, temperature=0.2)
            resp = client._client.models.generate_content(model=client._model, contents=prompt, config=config)
        except Exception:
            resp = client._client.models.generate_content(model=client._model, contents=prompt)
        text = (resp.text or "").strip()
        block = text.split("---답변---", 1)[-1].strip() if "---답변---" in text else text
        # Q1: 2 형태로 된 줄을 모두 추출 (잘린 응답·형식 차이 대비)
        q_answer_map = {}
        for line in block.split("\n"):
            line = line.strip()
            m = re.match(r"^Q\s*(\d+)\s*[:\.]\s*(.+)$", line, re.IGNORECASE)
            if m:
                q_num = int(m.group(1))
                q_answer_map[q_num] = m.group(2).strip()
        answers = []
        for i in range(1, nq + 1):
            raw = q_answer_map.get(i, "")
            if i > len(questions):
                answers.append(raw[:500] if raw else "")
                continue
            q = questions[i - 1]
            typ = q.get("type", "객관식")
            opts = q.get("options") or []
            if typ == "객관식" and opts:
                num_match = re.match(r"^(\d+)", (raw or "").replace("①", "1").replace("②", "2").replace("③", "3").replace("④", "4").replace("⑤", "5"))
                if num_match:
                    k = int(num_match.group(1))
                    if 1 <= k <= len(opts):
                        answers.append(opts[k - 1])
                    else:
                        answers.append(opts[0] if opts else raw[:200])
                elif raw.strip() in opts:
                    answers.append(raw.strip())
                else:
                    answers.append(opts[0] if opts else (raw[:200] or ""))
            else:
                answers.append((raw[:500] if raw else "(주관식 응답)"))
        return answers if len(answers) == nq else None
    except Exception:
        return None


def _generate_survey_responses_from_panel(panel_df: pd.DataFrame, questions: list[dict], random_state: int = 42):
    """패널(가상인구)별로 Gemini를 활용해 설문 응답 생성. (속도: 병렬 호출 + 짧은 프로필) 실패 시 랜덤 폴백."""
    import random
    from concurrent.futures import ThreadPoolExecutor, as_completed
    if panel_df is None or panel_df.empty or not questions:
        return None, []
    rng = random.Random(random_state)
    n = len(panel_df)
    cols = ["respondent_id"] + [f"Q{i+1}" for i in range(min(len(questions), 25))]
    max_workers = min(5, n)
    results_by_idx = [None] * n
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_one_respondent_answers_gemini, panel_df.iloc[idx], questions): idx
            for idx in range(n)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results_by_idx[idx] = fut.result()
            except Exception:
                results_by_idx[idx] = None
    rows = []
    records = []
    for idx in range(n):
        answers = results_by_idx[idx]
        if answers is None:
            answers = []
            for q in questions[:25]:
                typ = q.get("type", "객관식")
                opts = q.get("options") or []
                if typ == "객관식" and opts:
                    answers.append(rng.choice(opts))
                else:
                    answers.append("(주관식 응답)")
        pad = len(cols) - 1 - len(answers)
        row = [idx + 1] + list(answers)[: len(cols) - 1] + ([""] * pad if pad > 0 else [])
        rec = {"respondent_id": idx + 1}
        for i, a in enumerate(row[1:], 1):
            rec[f"Q{i}"] = rec[f"문항{i}"] = a
        rows.append(row)
        records.append(rec)
    response_df = pd.DataFrame(rows, columns=cols)
    return response_df, records


def _compute_survey_results(questions: list[dict], response_df: pd.DataFrame) -> list[dict]:
    """문항별 응답자 수, 비율, 평균점수(리커트 1~5 가정) 집계."""
    if response_df is None or response_df.empty or not questions:
        return []
    results = []
    for i, q in enumerate(questions[:25]):
        col = f"Q{i+1}"
        if col not in response_df.columns:
            continue
        ser = response_df[col].dropna()
        n = len(ser)
        vc = ser.value_counts()
        total = n or 1
        dist = [{"선택지": k, "응답자수": int(v), "비율(%)": round(v / total * 100, 1)} for k, v in vc.items()]
        # 평균점수: 리커트형(1~5)이면 1~5 매핑 후 평균, 아니면 NaN
        opts = q.get("options") or []
        if len(opts) >= 3 and n:
            try:
                order = list(range(1, len(opts) + 1))
                score_map = {o: order[j] for j, o in enumerate(opts)}
                scores = ser.map(score_map).dropna()
                mean_score = float(scores.mean()) if len(scores) else None
            except Exception:
                mean_score = None
        else:
            mean_score = None
        results.append({
            "문항번호": i + 1,
            "제목": q.get("title", ""),
            "질문": q.get("question", ""),
            "유형": q.get("type", "객관식"),
            "응답자수": n,
            "분포": dist,
            "평균점수": mean_score,
        })
    return results


def _get_open_ended_key_points(response_df: pd.DataFrame, question_num: int, question_title: str, question_text: str) -> list[dict]:
    """주관식 문항의 응답 목록을 분석하여 요점 5가지 반환. [{"번호": 1, "요점": "..."}, ...]"""
    if response_df is None or response_df.empty:
        return []
    col = f"Q{question_num}"
    if col not in response_df.columns:
        return []
    texts = response_df[col].dropna().astype(str).tolist()
    real = [t.strip() for t in texts if t.strip() and "(주관식 응답)" not in t and len(t.strip()) > 2]
    if not real:
        return [{"번호": i, "요점": "(수집된 응답 없음)"} for i in range(1, 6)]
    try:
        from utils.gemini_client import GeminiClient
    except Exception:
        # 폴백: 앞 5개 응답을 요점으로
        return [{"번호": i, "요점": real[i - 1][:200]} for i in range(1, min(6, len(real) + 1))]
    sample = "\n".join(real[:50])
    prompt = f"""다음은 설문 문항 "{question_title}"에 대한 응답자들의 주관식 답변입니다.
이 답변들을 분석하여 핵심 요점을 정확히 5가지로 요약해주세요. 각 요점은 한 문장으로 작성하세요.

[설문 질문]
{question_text[:300]}

[응답 목록]
{sample[:4000]}

[출력 형식] 반드시 아래처럼 번호와 요점만 출력하세요.
1. (첫 번째 요점)
2. (두 번째 요점)
3. (세 번째 요점)
4. (네 번째 요점)
5. (다섯 번째 요점)
"""
    try:
        client = GeminiClient()
        resp = client._client.models.generate_content(model=client._model, contents=prompt)
        text = (resp.text or "").strip()
        points = []
        for i in range(1, 6):
            for sep in [f"{i}.", f"{i})", f"{i}、"]:
                if sep in text:
                    rest = text.split(sep, 1)[-1].strip()
                    # 다음 번호(또는 줄끝)까지가 이 요점
                    next_idx = len(rest)
                    for j in range(1, 6):
                        if j == i:
                            continue
                        for s in [f"\n{j}.", f"\n{j})", f"\n{j}、"]:
                            idx = rest.find(s)
                            if idx >= 0:
                                next_idx = min(next_idx, idx)
                    line = rest[:next_idx].strip().split("\n")[0].strip()[:300]
                    points.append({"번호": i, "요점": line or f"(요점 {i})"})
                    break
            else:
                points.append({"번호": i, "요점": f"(요점 {i})"})
        return points[:5]
    except Exception:
        return [{"번호": i, "요점": real[i - 1][:200] if i <= len(real) else "(요점 없음)"} for i in range(1, 6)]


def _survey_overview_text(n_respondents: int, selected_sido_label: str, hypotheses: list) -> str:
    """조사 개요: 목적·대상(표본·추출방법)·기간·분석방법론."""
    from datetime import datetime
    purpose = "설정된 가설 검증 및 시장성·수요 파악"
    if hypotheses:
        purpose = "다음 가설 검증: " + " | ".join((h or "").strip()[:60] for h in (hypotheses or [])[:3] if (h or "").strip())
    return f"""
**조사 목적**  
{purpose}

**조사 대상**  
- 표본 크기: **{n_respondents}명**  
- 추출 방법: AI 가상인구(가상 패널) DB에서 해당 지역 내 무작위 추출(난수 시드 고정)

**조사 기간**  
- 분석 기준일: {datetime.now().strftime("%Y-%m-%d")} (AI 설문 응답 생성 시점)

**분석 방법론**  
- 문항별 빈도·비율·평균점수(리커트 척도) 기술 통계  
- 가설별 검증 및 타깃그룹 비교는 AI 분석(Gemini)으로 보완  
- 시각화: 막대·파이 차트로 응답 분포 표시
"""


def _render_question_result_chart(r: dict, key_suffix: str, chart_type: str = "pie"):
    """문항별 응답 분포를 원형(파이) 차트로 표시. 응답자 수와 비율을 함께 표기."""
    import plotly.graph_objects as go
    dist = r.get("분포") or []
    if not dist:
        return
    total = sum(d.get("응답자수", 0) for d in dist) or 1
    labels = [f"{d.get('선택지', '')} ({d.get('응답자수', 0)}명, {d.get('비율(%)', 0)}%)" for d in dist]
    values = [d.get("응답자수", 0) for d in dist]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4, textinfo="label+percent")])
    fig.update_layout(height=280, margin=dict(t=20, b=20), showlegend=True, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True, key=key_suffix)


def _summarize_open_ended_responses(response_df: pd.DataFrame, questions: list[dict]) -> dict:
    """주관식 문항별 핵심 응답 TOP 5(응답 내용) 반환. 그래프 없음."""
    if response_df is None or response_df.empty or not questions:
        return {}
    out = {}
    for i, q in enumerate(questions[:25]):
        if q.get("type") != "주관식":
            continue
        col = f"Q{i+1}"
        if col not in response_df.columns:
            continue
        texts = response_df[col].dropna().astype(str).tolist()
        real_texts = [t.strip() for t in texts if t.strip() and "(주관식 응답)" not in t and len(t.strip()) > 2]
        if not real_texts:
            out[i + 1] = {"제목": q.get("title", ""), "응답목록": [], "안내": "수집된 주관식 응답이 없거나 플레이스홀더입니다. 실제 응답 수집 시 핵심 응답 TOP 5로 요약할 수 있습니다."}
            continue
        # 중복 제거 후 순서 유지, 최대 5개
        seen = set()
        top5 = []
        for t in real_texts:
            if t not in seen and len(top5) < 5:
                seen.add(t)
                top5.append(t)
        out[i + 1] = {"제목": q.get("title", ""), "응답목록": top5, "안내": None}
    return out


def _generate_survey_report_analysis(hypotheses: list, questions: list[dict], results: list[dict], response_df: pd.DataFrame) -> dict:
    """상세분석·결과 및 전략 텍스트를 Gemini로 생성."""
    try:
        from utils.gemini_client import GeminiClient
    except Exception:
        return {"상세분석": "", "결과및전략": ""}
    results_text = []
    for r in results[:15]:
        results_text.append(f"문항{r['문항번호']} ({r['제목']}): 응답자수 {r['응답자수']}, 분포 {r['분포'][:5]}, 평균점수 {r.get('평균점수')}")
    prompt = f"""당신은 시장조사 보고서 전문가입니다. 아래 정보를 바탕으로 **상세분석**과 **결과 및 전략** 두 섹션의 초안을 작성해주세요.

[가설]
{chr(10).join(f"- {h}" for h in (hypotheses or [])[:3])}

[설문 문항 요약]
{chr(10).join(q.get("title", "") + ": " + (q.get("question", "") or "")[:80] for q in (questions or [])[:10])}

[설문결과 요약]
{chr(10).join(results_text[:12])}

[요청]
1. **상세분석**: 가설별 검증 분석, 타깃그룹 비교, 구매장벽 및 기대효익 분석을 2~3문단으로 작성하세요.
2. **결과 및 전략**: 가설별 결과, 종합 결론, 핵심 시사점, 구체적 실행방안을 2~3문단으로 작성하세요.
각 섹션을 "## 상세분석" / "## 결과 및 전략" 제목으로 구분하고, 한국어로 작성하세요."""
    try:
        client = GeminiClient()
        resp = client._client.models.generate_content(model=client._model, contents=prompt)
        text = (resp.text or "").strip()
        detail = ""
        strategy = ""
        if "## 상세분석" in text:
            parts = text.split("## 상세분석", 1)[-1].split("## 결과 및 전략", 1)
            detail = (parts[0].strip() or "")[:3000]
            if len(parts) > 1:
                strategy = (parts[1].strip() or "")[:3000]
        else:
            detail = text[:2000]
        return {"상세분석": detail, "결과및전략": strategy}
    except Exception:
        return {"상세분석": "(분석 생성 중 오류가 발생했습니다.)", "결과및전략": ""}


def _find_korean_font_path():
    """한글을 지원하는 TTF 폰트 경로 탐색 (Windows / Linux / 프로젝트 fonts)."""
    import os
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = []
    if os.name == "nt":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        candidates.append(os.path.join(windir, "Fonts", "malgun.ttf"))
        candidates.append(os.path.join(windir, "Fonts", "gulim.ttc"))
    candidates.append(os.path.join(_root, "fonts", "NanumGothic.ttf"))
    candidates.append(os.path.join(_root, "fonts", "NotoSansKR-Regular.ttf"))
    candidates.append("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
    candidates.append("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _build_survey_report_pdf_bytes(
    overview_text: str,
    questions: list,
    results_by_q: list,
    report_analysis: dict,
    open_ended: dict,
    n_respondents: int,
    selected_sido_label: str,
) -> bytes:
    """설문 응답 결과 보고서를 PDF 바이트로 생성 (reportlab 사용, 한글 폰트 지원)."""
    import io
    from reportlab.lib.pagesizes import A4

    def _escape(s: str) -> str:
        if not s:
            return ""
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
    from reportlab.lib import colors

    # 한글 폰트 등록
    pdf_font_name = "Helvetica"
    font_path = _find_korean_font_path()
    if font_path:
        try:
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            pdf_font_name = "KoreanReportFont"
            pdfmetrics.registerFont(TTFont(pdf_font_name, font_path))
        except Exception:
            pdf_font_name = "Helvetica"

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name="ReportTitle", parent=styles["Heading1"], fontSize=16, spaceAfter=12, fontName=pdf_font_name)
    heading_style = ParagraphStyle(name="ReportHeading2", parent=styles["Heading2"], fontName=pdf_font_name)
    heading3_style = ParagraphStyle(name="ReportHeading3", parent=styles["Heading3"], fontName=pdf_font_name)
    body_style = ParagraphStyle(name="ReportBody", parent=styles["Normal"], fontName=pdf_font_name)

    story = []
    story.append(Paragraph("시장성 조사 설계 — 설문 응답 결과 보고서", title_style))
    story.append(Spacer(1, 8*mm))

    story.append(Paragraph("1. 설문지 — 조사 개요 및 전체 설문 문항", heading_style))
    story.append(Spacer(1, 4*mm))
    for line in (overview_text or "").strip().split("\n"):
        line = line.strip()
        if line:
            line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
            story.append(Paragraph(line, body_style))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("전체 설문 문항", heading3_style))
    for i, q in enumerate((questions or [])[:25], 1):
        typ = q.get("type", "객관식")
        title = _escape((q.get("title", "") or "").strip()[:200])
        question = _escape((q.get("question", "") or "").strip()[:500])
        options = q.get("options") or []
        opts_text = " / ".join([f"{j+1}. {_escape(str(o))}" for j, o in enumerate(options[:10])]) if options else "(주관식)"
        story.append(Paragraph(f"<b>문항 {i}</b> [{typ}] {title}", body_style))
        story.append(Paragraph(question, body_style))
        story.append(Paragraph(opts_text, body_style))
        story.append(Spacer(1, 2*mm))
    story.append(PageBreak())

    story.append(Paragraph("2. 설문결과 — 핵심 지표", heading_style))
    story.append(Spacer(1, 4*mm))
    for r in (results_by_q or [])[:25]:
        story.append(Paragraph(f"<b>문항 {r.get('문항번호', '')}</b> {_escape(r.get('제목', ''))}", body_style))
        story.append(Paragraph(_escape((r.get("질문", "") or "")[:300]), body_style))
        dist = r.get("분포") or []
        if dist:
            table_data = [["선택지", "응답자수", "비율(%)"]]
            for d in dist[:15]:
                table_data.append([
                    str(d.get("선택지", ""))[:50],
                    str(d.get("응답자수", "")),
                    str(d.get("비율(%)", "")),
                ])
            t = Table(table_data)
            t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), pdf_font_name), ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
            story.append(t)
        if r.get("평균점수") is not None:
            story.append(Paragraph(f"평균점수: {r['평균점수']}", body_style))
        story.append(Spacer(1, 4*mm))
    story.append(PageBreak())

    story.append(Paragraph("3. 상세분석", heading_style))
    story.append(Spacer(1, 4*mm))
    detail = (report_analysis or {}).get("상세분석") or "(내용 없음)"
    for para in (detail or "").strip().split("\n\n")[:30]:
        if para.strip():
            story.append(Paragraph(_escape(para.strip()[:1500]), body_style))
            story.append(Spacer(1, 2*mm))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("4. 결론 및 시사점", heading_style))
    story.append(Spacer(1, 4*mm))
    strategy = (report_analysis or {}).get("결과및전략") or "(내용 없음)"
    for para in (strategy or "").strip().split("\n\n")[:20]:
        if para.strip():
            story.append(Paragraph(_escape(para.strip()[:1500]), body_style))
            story.append(Spacer(1, 2*mm))
    if open_ended:
        story.append(PageBreak())
        story.append(Paragraph("5. 주관식 응답 — 핵심 응답 TOP 5", heading_style))
        story.append(Spacer(1, 4*mm))
        for q_num, data in list(open_ended.items())[:10]:
            story.append(Paragraph(f"<b>문항 {q_num}</b> {_escape(data.get('제목', ''))}", body_style))
            if data.get("응답목록"):
                for rank, text in enumerate((data["응답목록"] or [])[:5], 1):
                    story.append(Paragraph(f"{rank}. {_escape(str(text)[:300])}", body_style))
            story.append(Spacer(1, 2*mm))

    doc.build(story)
    return buf.getvalue()


def _extract_text_from_pdf(pdf_file) -> str:
    """PDF 파일에서 텍스트 추출 (pypdf 사용)."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_file)
        parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        return "\n".join(parts) if parts else ""
    except Exception:
        return ""


def _parse_survey_pdf_with_gemini(pdf_text: str) -> list[dict]:
    """PDF에서 추출한 텍스트를 Gemini로 분석해 설문 문항 리스트로 변환."""
    if not (pdf_text and pdf_text.strip()):
        return []
    try:
        from utils.gemini_client import GeminiClient
    except Exception:
        return []
    prompt = f"""당신은 설문조사 문서 분석 전문가입니다. 아래는 설문조사 PDF에서 추출한 텍스트입니다. 
이 텍스트에서 설문 문항(제목, 질문, 객관식/주관식, 선택지)을 모두 추출하여 아래 출력 형식으로 정리해주세요.
- 문항 번호·불릿 등은 제거하고 제목과 질문만 남기세요.
- 객관식은 선택지(보기)를 ① ② ③ 형식으로 구분하세요.
- 최대 25개 문항까지 추출하세요.

[출력 형식] 반드시 아래 형식으로만 출력하세요.
---
유형: 객관식
제목: (문항 제목)
질문: (질문 내용)
선택지: ① (내용) ② (내용) ③ (내용) ...
---
유형: 주관식
제목: (문항 제목)
질문: (질문 내용)
---

[PDF 추출 텍스트]
{pdf_text[:12000]}
"""
    try:
        client = GeminiClient()
        resp = client._client.models.generate_content(model=client._model, contents=prompt)
        text = (resp.text or "").strip()
        questions = _parse_survey_questions_from_text(text)
        if questions:
            questions = _review_survey_questions_with_gemini(questions)
        return questions[:25] if questions else []
    except Exception:
        return []


def _render_survey_subpage():
    """서브페이지: 설문조사 — 설정된 가설 + 설문 문항 생성"""
    # 시장성 조사 설계 모든 페이지: 버튼 그림자 제거 (stButton)
    st.markdown("""
    <style>
    .stButton > button, .stButton > button:hover { box-shadow: none !important; }
    div[data-testid="stVerticalBlock"] > div:first-child .stButton > button,
    div[data-testid="column"]:nth-of-type(2) .stButton > button { min-height: 2.25rem !important; height: 2.25rem !important; padding: 0.25rem 1rem !important; font-size: 0.9rem !important; }
    /* 왼쪽(가설)·오른쪽(설문 문항) 컬럼 버튼: 고정 width/height 제거, 텍스트에 맞게 자동 크기 */
    div[data-testid="column"]:nth-of-type(3) .stButton > button,
    div[data-testid="column"]:nth-of-type(4) .stButton > button {
        width: auto !important; min-width: fit-content !important;
        height: auto !important; min-height: 2.25rem !important;
        padding: 10px 20px !important;
        white-space: nowrap !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
    }
    /* 설문조사 헤더: 타이틀-구분선-버튼 한 줄, Ghost 스타일, 버튼 그림자 제거·오른쪽 정렬 */
    .survey-header-row { display: flex; align-items: center; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.5rem; margin-bottom: 0.75rem; }
    .survey-header-title { font-size: 1.25rem; font-weight: 700; color: #0f172a; }
    .survey-header-line { flex: 1; min-width: 16px; border-bottom: 1px solid #e2e8f0; align-self: flex-end; margin-left: 0.5rem; }
    div[data-testid="stHorizontalBlock"]:has(.survey-header-row) > div:first-child .survey-header-row { margin-bottom: 0; }
    div[data-testid="stHorizontalBlock"]:has(.survey-header-row) > div:nth-child(2) { border-bottom: 1px solid #e2e8f0; padding-bottom: 0.5rem; margin-bottom: 0.75rem; display: flex; align-items: center; justify-content: flex-end; margin-left: auto; }
    div[data-testid="stHorizontalBlock"]:has(.survey-header-row) > div:nth-child(2) .stButton > button { min-height: 1.75rem !important; height: 1.75rem !important; padding: 0.2rem 0.6rem !important; font-size: 0.8rem !important; background: transparent !important; border: 1px solid #cbd5e1 !important; color: #475569 !important; box-shadow: none !important; border-radius: 0.375rem !important; }
    div[data-testid="stHorizontalBlock"]:has(.survey-header-row) > div:nth-child(2) .stButton > button:hover { background: #f1f5f9 !important; border-color: #94a3b8 !important; }
    /* 설문조사 페이지 전체 버튼 그림자 제거 */
    div[data-testid="column"] .stButton > button { box-shadow: none !important; }
    </style>
    """, unsafe_allow_html=True)
    # 상단 행: 타이틀 --------- [설계 선택으로] (선 위에 버튼, Ghost 스타일)
    is_ai_survey = st.session_state.get("survey_ai_flow_page") == "ai_survey"
    is_ai_conduct = st.session_state.get("survey_ai_flow_page") == "ai_survey_conduct"
    if is_ai_conduct:
        cols = st.columns([8, 1])  # 설문 응답 결과: 이전페이지 오른쪽 끝
    elif is_ai_survey:
        cols = st.columns([8, 1])  # AI 옵션 선택: 이전페이지 오른쪽 끝
    else:
        cols = st.columns([8, 1])  # 설정된 가설: 이전페이지 오른쪽 끝
    with cols[0]:
        st.markdown("""
        <div class="survey-header-row">
        <span class="survey-header-title">설문조사</span>
        <span class="survey-header-line"></span>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        if is_ai_conduct:
            # 설문 응답 결과 페이지: 이전페이지 (← 설계 선택으로와 동일 Ghost 스타일)
            if st.button("이전페이지", key="survey_back_to_ai_options", type="secondary"):
                st.session_state.survey_ai_flow_page = "ai_survey"
                st.rerun()
        elif is_ai_survey:
            if st.button("이전페이지", key="survey_back_to_questions", type="secondary"):
                st.session_state.survey_ai_flow_page = ""
                st.rerun()
        else:
            if st.button("이전페이지", key="survey_back_to_design", type="secondary"):
                st.session_state.survey_design_subpage = ""
                st.session_state.pop("survey_hypotheses", None)
                st.session_state.pop("survey_ai_flow_page", None)
                st.rerun()

    # 설문 응답 결과 페이지: 6개 섹션 보고서
    if is_ai_conduct:
        st.markdown("### 설문 응답 결과")
        questions = st.session_state.get("survey_questions", _default_survey_questions())
        response_df = st.session_state.get("survey_response_df")
        response_records = st.session_state.get("survey_response_records", [])
        if not response_records and (response_df is None or response_df.empty):
            st.warning("응답 데이터가 없습니다. **이전페이지**에서 **AI 설문 진행**을 다시 실행해 주세요.")
        results_by_q = st.session_state.get("survey_results_by_question", [])
        report_analysis = st.session_state.get("survey_report_analysis", {"상세분석": "", "결과및전략": ""})
        panel_df = st.session_state.get("survey_ai_panel_df")
        n_respondents = len(response_records) if response_records else 0
        selected_sido_label = st.session_state.get("survey_ai_sido_label", "")

        hypotheses = st.session_state.get("survey_hypotheses", [])[:3]

        # 1. 설문지 — 조사 개요(목적·대상·기간·분석방법론) + 전체 설문 문항
        with st.expander("**1. 설문지 — 조사 개요 및 전체 설문 문항**", expanded=True):
            st.markdown("#### 조사 개요")
            st.markdown(_survey_overview_text(n_respondents, selected_sido_label, hypotheses))
            st.markdown("---")
            st.markdown("#### 전체 설문 문항")
            for i, q in enumerate(questions[:25], 1):
                typ = q.get("type", "객관식")
                title = (q.get("title", "") or "").strip()
                question = (q.get("question", "") or "").strip()
                options = q.get("options", [])
                opts_text = " / ".join([f"① {o}" for o in options]) if options else "(주관식)"
                st.markdown(f"**문항 {i}** [{typ}] {title}\n\n{question}\n\n{opts_text}")

        # 2. 응답자 특성 분석 — 성별·연령·지역 등 인구통계학적 배경, 편향성 점검
        with st.expander("**2. 응답자 특성 분석 — 성별·연령·지역 등 인구통계학적 배경**", expanded=True):
            if panel_df is not None and not panel_df.empty:
                st.caption("응답자의 인구통계학적 특성을 파악하여 응답의 편향성을 점검하고, 집단별 특성 해석에 활용합니다.")
                _render_panel_summary_and_charts(panel_df, key_prefix="respondent_char")
            else:
                st.info("패널(응답자) 정보가 없습니다.")

        # 3. 설문결과(핵심 지표) — 객관식: 표+그래프, 주관식: 응답 분석 요점 5가지 표만
        with st.expander("**3. 설문결과 — 핵심 지표 분석(빈도·비율·평균) 및 시각화**", expanded=True):
            for r in results_by_q:
                st.markdown(f"**문항 {r['문항번호']}** {r['제목']}")
                st.caption(r["질문"][:100] + ("…" if len(r["질문"]) > 100 else ""))
                if r.get("유형") == "주관식":
                    # 주관식: 그래프 없이 표만 — 패널 응답 분석 요점 5가지
                    key_points = _get_open_ended_key_points(response_df, r["문항번호"], r["제목"], r["질문"])
                    if key_points:
                        st.dataframe(pd.DataFrame(key_points), use_container_width=True, hide_index=True)
                else:
                    # 객관식: 표 + 파이 차트
                    df_dist = pd.DataFrame(r["분포"])
                    if not df_dist.empty:
                        tab_left, tab_right = st.columns([1, 1])
                        with tab_left:
                            st.dataframe(df_dist, use_container_width=True, hide_index=True)
                        with tab_right:
                            _render_question_result_chart(r, f"result_pie_{r['문항번호']}", chart_type="pie")
                st.markdown("---")
            st.markdown("#### 통계적 유의성 검토")
            st.caption(
                "본 조사는 AI 가상인구 패널 기반 시뮬레이션으로, 기술 통계(빈도·비율·평균)를 제시합니다. "
                "실제 표본조사와 달리 추론 통계(χ² 검정, t검정 등)는 생략되었으며, "
                "표본 크기가 작을 경우 통계적 유의성에 한계가 있을 수 있음을 참고하시기 바랍니다."
            )

        # 4. 상세분석
        with st.expander("**4. 상세분석 — 가설별 검증·타깃그룹 비교·구매장벽·기대효익**", expanded=True):
            st.markdown(report_analysis.get("상세분석") or "(상세분석 내용이 없습니다. AI 설문 진행을 다시 실행해 주세요.)")

        # 5. 결론 및 시사점(결과 및 전략)
        with st.expander("**5. 결론 및 시사점 — 가설별 결과·종합 결론·핵심 시사점·실행방안**", expanded=True):
            st.markdown(
                "데이터 분석 결과를 토대로 당면한 문제의 원인을 진단하고, "
                "향후 의사결정에 반영할 구체적인 개선 방향을 제안합니다."
            )
            st.markdown(report_analysis.get("결과및전략") or "(결과 및 전략 내용이 없습니다.)")

        # 주관식 응답 정제 — 핵심 응답 TOP 5 (그래프 없음)
        open_ended = _summarize_open_ended_responses(response_df, questions)
        if open_ended:
            with st.expander("**6. 주관식 응답 — 핵심 응답 TOP 5**", expanded=False):
                st.caption("주관식 문항별 대표 응답 내용을 최대 5개까지 표시합니다. (그래프 없음)")
                for q_num, data in open_ended.items():
                    st.markdown(f"**문항 {q_num}** {data['제목']}")
                    if data.get("안내"):
                        st.info(data["안내"])
                    elif data.get("응답목록"):
                        for rank, text in enumerate(data["응답목록"], 1):
                            st.markdown(f"{rank}. {text}")
                    st.markdown("---")

        # 설문응답원본 (주관식 섹션 유무에 따라 6 또는 7번)
        with st.expander("**6. 설문응답원본 — 응답자 전체 설문 응답 데이터**" if not open_ended else "**7. 설문응답원본 — 응답자 전체 설문 응답 데이터**", expanded=False):
            if response_df is not None and not response_df.empty:
                st.dataframe(response_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "CSV 다운로드",
                    data=response_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="survey_responses.csv",
                    mime="text/csv",
                    key="survey_response_csv",
                )
            else:
                st.info("응답 원본 데이터가 없습니다.")

        st.markdown("---")
        st.markdown("#### 최종 보고서 저장")
        try:
            overview_text = _survey_overview_text(n_respondents, selected_sido_label, hypotheses or [])
            pdf_bytes = _build_survey_report_pdf_bytes(
                overview_text=overview_text,
                questions=questions,
                results_by_q=results_by_q,
                report_analysis=report_analysis,
                open_ended=open_ended,
                n_respondents=n_respondents,
                selected_sido_label=selected_sido_label,
            )
            st.download_button(
                "PDF 파일로 저장",
                data=pdf_bytes,
                file_name="survey_report.pdf",
                mime="application/pdf",
                key="survey_pdf_download",
            )
        except Exception as e:
            st.caption(f"PDF 생성 중 오류: {e}")
        return

    # 페이지 내 전환: AI 설문 진행 옵션 선택 (2분할: 왼쪽 지역/패널/지도, 오른쪽 가상인구 요약+그래프+버튼)
    if is_ai_survey:
        st.markdown("### AI 설문 진행 옵션 선택")
        st.caption("**지역**과 **패널 수**를 선택한 뒤, 오른쪽에서 불러온 가상인구를 확인하고 **AI 설문 진행**을 누르세요.")

        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown("**지역·패널**")
            sido_master = get_sido_master()
            sido_options = [f"{s['sido_name']} ({s['sido_code']})" for s in sido_master]
            default_label = st.session_state.get("survey_ai_sido_label", "경상북도 (37)")
            default_idx = sido_options.index(default_label) if default_label in sido_options else 0
            selected_sido_label = st.selectbox(
                "지역 선택",
                options=sido_options,
                index=default_idx,
                key="survey_ai_sido_select",
            )
            st.session_state.survey_ai_sido_label = selected_sido_label
            panel_count = st.number_input(
                "패널 수",
                min_value=1,
                max_value=1000,
                value=st.session_state.get("survey_ai_panel_count", 10),
                key="survey_ai_panel_count_input",
            )
            st.session_state.survey_ai_panel_count = panel_count
            selected_sido_code = SIDO_LABEL_TO_CODE.get(selected_sido_label, "37")

            try:
                from core.db import get_virtual_population_db_combined_df
                combined_df = get_virtual_population_db_combined_df(selected_sido_code)
                if combined_df is not None and len(combined_df) > 0:
                    n = min(int(panel_count), len(combined_df))
                    panel_df = combined_df.sample(n=n, random_state=42) if len(combined_df) >= n else combined_df.head(n)
                    st.session_state.survey_ai_panel_df = panel_df
                    st.success(f"**{n}명** 불러옴 (페르소나·현시대 반영)")
                else:
                    st.session_state.pop("survey_ai_panel_df", None)
                    st.warning("해당 지역에 가상인구가 없습니다. 가상인구 DB에서 2차 대입 후 전체 학습을 실행하세요.")
            except Exception as e:
                st.session_state.pop("survey_ai_panel_df", None)
                st.warning(f"가상인구 DB 조회 오류: {e}")

            st.markdown("**지도**")
            try:
                from core.db import get_sido_vdb_stats
                from pages.virtual_population_db import _build_korea_choropleth_figure
                try:
                    vdb_stats = get_sido_vdb_stats()
                except Exception:
                    vdb_stats = {}
                region_stats = {}
                for s in SIDO_MASTER:
                    if s.get("sido_code") == "00":
                        continue
                    code = s["sido_code"]
                    region_stats[code] = {
                        "sido_name": s.get("sido_name", ""),
                        "vdb_count": vdb_stats.get(code, {}).get("vdb_count", 0),
                        "total_pop": SIDO_TOTAL_POP.get(code),
                    }
                fig = _build_korea_choropleth_figure(selected_sido_code, region_stats)
                st.plotly_chart(fig, key="survey_ai_sido_map", use_container_width=True)
            except Exception:
                st.caption("지도를 불러올 수 없습니다.")

        with col_right:
            st.markdown("**불러온 가상인구 정보**")
            panel_df = st.session_state.get("survey_ai_panel_df")
            _render_panel_summary_and_charts(panel_df)
            st.markdown("---")
            if panel_df is not None and len(panel_df) > 0:
                if st.button("AI 설문 진행", type="primary", use_container_width=True, key="survey_ai_conduct_btn"):
                    questions = st.session_state.get("survey_questions", _default_survey_questions())
                    with st.spinner("패널 설문 응답을 생성하고 있습니다…"):
                        response_df, response_records = _generate_survey_responses_from_panel(panel_df, questions)
                        st.session_state.survey_response_df = response_df
                        st.session_state.survey_response_records = response_records
                        if response_df is not None and not response_df.empty:
                            results = _compute_survey_results(questions, response_df)
                            st.session_state.survey_results_by_question = results
                            hypotheses = st.session_state.get("survey_hypotheses", [])[:3]
                            analysis = _generate_survey_report_analysis(hypotheses, questions, results, response_df)
                            st.session_state.survey_report_analysis = analysis
                        else:
                            st.session_state.survey_results_by_question = []
                            st.session_state.survey_report_analysis = {"상세분석": "", "결과및전략": ""}
                        # 설문 응답 결과를 분석 페이지(컨조인트/PSM/Bass/A·B 테스트)에서 사용할 수 있도록 저장
                        import os
                        survey_latest_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "survey_latest")
                        try:
                            os.makedirs(survey_latest_dir, exist_ok=True)
                            panel_path = os.path.join(survey_latest_dir, "panel.csv")
                            response_path = os.path.join(survey_latest_dir, "responses.csv")
                            panel_df.to_csv(panel_path, index=False, encoding="utf-8-sig")
                            response_df.to_csv(response_path, index=False, encoding="utf-8-sig")
                        except Exception:
                            pass
                    st.session_state.survey_ai_flow_page = "ai_survey_conduct"
                    st.rerun()
            else:
                st.button("AI 설문 진행", type="primary", use_container_width=True, key="survey_ai_conduct_btn", disabled=True)
                st.caption("지역·패널을 선택해 가상인구를 먼저 불러오세요.")
        return

    # 가설: 앞페이지 정의·니즈로 Gemini가 3개 생성 (최초 1회)
    if "survey_hypotheses" not in st.session_state:
        definition = st.session_state.get("definition", "") or ""
        needs = st.session_state.get("needs", "") or ""
        if definition.strip() and needs.strip():
            with st.spinner("가설을 생성하고 있습니다…"):
                st.session_state.survey_hypotheses = _generate_hypotheses_with_gemini(definition, needs)
            st.rerun()
        else:
            st.session_state.survey_hypotheses = [
                "제품/서비스 정의와 조사 니즈를 입력한 뒤 설계 선택에서 설문조사를 선택해주세요.",
                "(가설 2)",
                "(가설 3)",
            ]

    hypotheses = list(st.session_state.get("survey_hypotheses", []))  # 복사본으로 3개 유지
    while len(hypotheses) < 3:
        hypotheses.append("")

    # 수평 2분할: 왼쪽 설정된 가설 + 버튼, 오른쪽 설문 문항
    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:
        st.markdown("**설정된 가설**")
        editing_hyp = st.session_state.get("survey_editing_hypotheses", False)
        if editing_hyp:
            new_h = []
            for i in range(3):
                val = st.text_area(f"가설 {i+1}", value=hypotheses[i], key=f"survey_hyp_edit_{i}", height=80)
                new_h.append(val.strip() or hypotheses[i])
            if st.button("가설 저장", type="primary", key="survey_hyp_save"):
                st.session_state.survey_hypotheses = new_h[:3]
                st.session_state.survey_editing_hypotheses = False
                st.rerun()
            if st.button("취소", key="survey_hyp_cancel"):
                st.session_state.survey_editing_hypotheses = False
                st.rerun()
        else:
            for i, h in enumerate(hypotheses[:3], 1):
                if h:
                    st.markdown(f"**가설 {i}:** {h}")
            if st.button("가설 수정", key="survey_hyp_edit_btn"):
                st.session_state.survey_editing_hypotheses = True
                st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("설문 문항 자동 생성", type="primary", use_container_width=True, key="survey_generate_btn"):
            with st.spinner("가설을 분석하여 설문 문항을 생성하고 있습니다…"):
                st.session_state.survey_questions = _generate_survey_questions_with_gemini(
                    st.session_state.get("survey_hypotheses", [])[:3]
                )
                st.session_state.survey_questions_generated = True
                st.session_state.pop("survey_questions_editing", None)
            st.rerun()

    with col_right:
        st.markdown("**설문 문항**")
        if st.session_state.get("survey_questions_generated", False):
            questions = st.session_state.get("survey_questions", _default_survey_questions())
            editing_q = st.session_state.get("survey_questions_editing", False)
            if editing_q:
                new_questions = []
                for i, q in enumerate(questions):
                    with st.expander(f"문항 {i+1} 수정: {q.get('title', '')[:20]}", expanded=(i == 0)):
                        t = st.selectbox("유형", ["객관식", "주관식"], index=0 if q.get("type") == "객관식" else 1, key=f"q_type_{i}")
                        title = st.text_input("제목", value=q.get("title", ""), key=f"q_title_{i}")
                        question = st.text_area("질문", value=q.get("question", ""), key=f"q_question_{i}")
                        opts = q.get("options") or []
                        if t == "객관식":
                            opts_str = st.text_area("선택지 (한 줄에 하나씩)", value="\n".join(opts), key=f"q_opts_{i}")
                            opts = [ln.strip() for ln in opts_str.split("\n") if ln.strip()]
                        new_questions.append({"type": t, "title": title, "question": question, "options": opts if t == "객관식" else []})
                if st.button("설문 문항 저장", type="primary", key="survey_q_save"):
                    st.session_state.survey_questions = new_questions[:25]
                    st.session_state.survey_questions_editing = False
                    st.rerun()
                if st.button("취소", key="survey_q_cancel"):
                    st.session_state.survey_questions_editing = False
                    st.rerun()
            else:
                for i, q in enumerate(questions[:25], 1):
                    typ = q.get("type", "객관식")
                    title = (q.get("title", "") or "").strip()
                    question = (q.get("question", "") or "").strip()
                    options = q.get("options", [])
                    # 제목에서 앞쪽 "문항 N", "N.", "N)" 등 중복 번호 제거
                    title = re.sub(r"^(문항\s*\d+\s*[:\.\)]\s*|\d+[\.\)]\s*)", "", title, flags=re.IGNORECASE).strip() or title
                    nums = "①②③④⑤⑥"
                    opts_text = " / ".join([f"{nums[j] if j < len(nums) else j+1}. {o}" for j, o in enumerate(options)]) if options else ""
                    # [객관식]은 표시하지 않고, [주관식]만 질문 끝에 표시
                    question_display = f"{question} [주관식]" if typ == "주관식" else question
                    st.markdown(f"""
                    <div style="background: white; padding: 16px; border-radius: 12px; margin-bottom: 12px; border: 1px solid #e0e7ff;">
                        <h4 style="color: #4f46e5; margin-bottom: 6px; font-size: 14px;">문항 {i}: {title}</h4>
                        <p style="color: #0f1a2d; font-size: 13px; margin-bottom: 4px;">{question_display}</p>
                        <p style="color: #64748b; font-size: 11px;">{opts_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                # 수정 | 설문문항 업로드(가로 2배) | 패널 지정
                col_edit, col_upload, col_panel, _ = st.columns([2, 4, 2, 4])
                with col_edit:
                    if st.button("수정", key="survey_questions_edit_btn"):
                        st.session_state.survey_questions_editing = True
                        st.rerun()
                with col_upload:
                    if st.button("설문문항 업로드", key="survey_pdf_upload_btn"):
                        st.session_state.survey_show_pdf_upload = True
                        st.rerun()
                with col_panel:
                    if st.button("패널 지정", type="primary", key="survey_ai_flow_btn"):
                        st.session_state.survey_ai_flow_page = "ai_survey"
                        st.rerun()
                if st.session_state.get("survey_show_pdf_upload"):
                    st.caption("설문문항 PDF를 올리면 문항으로 반영됩니다.")
                    pdf_file = st.file_uploader("PDF 파일 선택", type=["pdf"], key="survey_pdf_upload")
                    if pdf_file is not None:
                        last_processed = st.session_state.get("survey_pdf_processed_name")
                        if last_processed != pdf_file.name:
                            with st.spinner("PDF 분석 중…"):
                                raw_text = _extract_text_from_pdf(pdf_file)
                                parsed = _parse_survey_pdf_with_gemini(raw_text)
                                if parsed:
                                    st.session_state.survey_questions = parsed[:25]
                                    st.session_state.survey_questions_generated = True
                                    st.session_state.survey_pdf_processed_name = pdf_file.name
                                    st.session_state.survey_show_pdf_upload = False
                                    st.session_state.pop("survey_questions_editing", None)
                                    st.rerun()
                                else:
                                    st.session_state.survey_pdf_processed_name = pdf_file.name
                                    st.warning("PDF에서 설문 문항을 추출하지 못했습니다.")
        else:
            st.info("왼쪽에서 **설문 문항 자동 생성**을 누르면 여기에 설문 문항이 표시됩니다.")


def _render_interview_subpage():
    """서브페이지: 심층면접 — 인터뷰 방식(FGI/FGD/IDI) + 모더레이터 질문 가이드"""
    # 설계 선택 버튼: 타이틀-구분선-버튼 한 줄 (설문조사와 동일 헤더 스타일)
    st.markdown("""
    <style>
    .interview-header-row { display: flex; align-items: center; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.5rem; margin-bottom: 0.75rem; }
    .interview-header-title { font-size: 1.25rem; font-weight: 700; color: #0f172a; }
    .interview-header-line { flex: 1; min-width: 16px; border-bottom: 1px solid #e2e8f0; align-self: flex-end; margin-left: 0.5rem; }
    div[data-testid="stHorizontalBlock"]:has(.interview-header-row) > div:nth-child(2) { border-bottom: 1px solid #e2e8f0; padding-bottom: 0.5rem; margin-bottom: 0.75rem; display: flex; align-items: center; justify-content: flex-end; }
    div[data-testid="stHorizontalBlock"]:has(.interview-header-row) > div:nth-child(2) .stButton > button { min-height: 1.75rem !important; height: 1.75rem !important; padding: 0.2rem 0.6rem !important; font-size: 0.8rem !important; background: transparent !important; border: 1px solid #cbd5e1 !important; color: #475569 !important; box-shadow: none !important; border-radius: 0.375rem !important; }
    div[data-testid="stHorizontalBlock"]:has(.interview-header-row) > div:nth-child(2) .stButton > button:hover { background: #f1f5f9 !important; border-color: #94a3b8 !important; }
    </style>
    """, unsafe_allow_html=True)
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown("""
        <div class="interview-header-row">
        <span class="interview-header-title">심층면접</span>
        <span class="interview-header-line"></span>
        </div>
        """, unsafe_allow_html=True)
    with col_btn:
        if st.button("← 설계 선택으로", key="interview_back_to_design", type="secondary"):
            st.session_state.survey_design_subpage = ""
            st.rerun()

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
    # 시장성 조사 설계 모든 페이지: 버튼 그림자 제거
    st.markdown("""<style>.stButton > button, .stButton > button:hover { box-shadow: none !important; }</style>""", unsafe_allow_html=True)
    subpage = st.session_state.get("survey_design_subpage", "")

    # 서브페이지: 설문조사 → 설정된 가설 + 설문 문항 생성
    if subpage == "설문조사":
        _render_survey_subpage()
        return

    # 서브페이지: 심층면접 → 인터뷰 방식 + 질문 가이드
    if subpage == "심층면접":
        _render_interview_subpage()
        return

    # 기본: 폼 + AI 설계 제안 + 설문조사/심층면접 버튼 (같은 페이지 내 전환)
    st.markdown("""
    <div class="survey-header">
        <h1 style="font-size: 32px; font-weight: 800; color: #0f1a2d; margin-bottom: 12px;">
            새로운 시장성 조사 설계 시작하기
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
