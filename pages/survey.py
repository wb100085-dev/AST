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
from core.constants import SIDO_LABEL_TO_CODE, SIDO_CODE_TO_NAME
from core.session_cache import get_sido_master


def _render_panel_summary_and_charts(panel_df, key_prefix: str = ""):
    """불러온 가상인구 요약(메트릭) + 6축 그래프를 생성 페이지와 동일한 형식으로 렌더. key_prefix로 위젯 키 중복 방지."""
    import plotly.graph_objects as go

    if panel_df is None or panel_df.empty:
        st.info("가상인구를 불러온 후 요약과 그래프가 표시됩니다.")
        return

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
    st.markdown("#### 6축 분포 그래프")

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
            height=300,
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
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="", yaxis_title="인구수")
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
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="", yaxis_title="인구수")
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
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="", yaxis_title="인구수")
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
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="", yaxis_title="인구수")
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
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=80), xaxis_tickangle=-45, xaxis_title="", yaxis_title="인구수")
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
    최대 25개: 객관식 약 20개, 주관식 약 5개.
    반환: [{"type": "객관식"|"주관식", "title": "", "question": "", "options": ["A","B",...] 또는 []}, ...]
    """
    try:
        from utils.gemini_client import GeminiClient
    except Exception:
        return _default_survey_questions()
    hyp_text = "\n".join([f"가설 {i+1}: {h}" for i, h in enumerate(hypotheses[:3], 1)])
    prompt = f"""당신은 시장성 조사 설계 전문가입니다. 아래 설정된 가설 3개를 정밀하게 분석한 뒤, 다음 주의사항을 반드시 지키며 가설을 모두 검증할 수 있는 설문 문항을 체계적으로 작성해주세요.

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

3. 답변 항목 설계 (Answer Options)
- 상호 배타적·포괄적 보기: 보기가 서로 겹치지 않게(Mutually Exclusive) 하고, 가능한 응답 범위를 모두 포함(Collectively Exhaustive)하세요. (예: 20-29세, 30-39세처럼 구간이 겹치지 않게)
- 균형 잡힌 척도: 만족도는 '매우 만족'~'매우 불만족'까지 균형 있게 배치하세요.
- '기타', '해당 없음' 포함: 응답자가 선택할 수 없는 상황을 피할 수 있도록 보완 문항을 넣으세요.

4. 설문 흐름 및 배열 (Flow & Order)
- 일반→구체: 가벼운 질문에서 시작해 점점 핵심·심층 질문으로 유도하세요.
- 민감한 질문은 후반부: 소득·개인 정보 등은 설문 후반에 배치하세요.
- 적절한 길이: 핵심 질문만 포함해 짧게 유지하세요.

5. 기타 유의사항
- 응답자 행동 예측 질문 주의: "만약 …하면 구매하시겠습니까?" 식의 미래 행동 예측 질문은 과대평가될 수 있으므로, 과거 행동·현재 인식 위주로 질문하세요.

[작성 절차]
1) 각 가설을 분석하여 검증에 필요한 핵심 변인·하위 요인을 도출하세요.
2) 도출된 변인별로 측정할 문항을 배치하세요. 문항 간 내용 중복을 피하세요.
3) 객관식은 리커트 척도 또는 명확한 선택지 4~5개, 주관식은 심층 의견이 드러나도록 한 문장으로 작성하세요.
4) 각 문항이 최소 하나의 가설 검증에 직접 기여하도록 하세요.

[문항 구성]
- 총 최대 25개: 객관식 약 20개, 주관식 약 5개. 처음에 스크리닝 문항을 포함하세요.
- 객관식: 제목·질문·선택지(①~⑤)를 명확히 구분. 선택지에 '기타', '해당 없음' 등 보완 항목 포함.
- 주관식: 제목·질문만 작성. 응답자가 자유 서술할 수 있는 형태로.

[출력 형식] 아래 형식을 정확히 따르세요. 한 문항당 블록으로 구분하고, 제목에는 "문항 N" 또는 "N." 같은 번호를 붙이지 마세요.
---
유형: 객관식
제목: (문항 제목 한 줄, 번호 없이)
질문: (질문 내용)
선택지: ① (내용) ② (내용) ③ (내용) ④ (내용) ⑤ (해당 없음/기타 등)
---
유형: 주관식
제목: (문항 제목 한 줄, 번호 없이)
질문: (질문 내용)
---

[설정된 가설]
{hyp_text}
"""
    try:
        client = GeminiClient()
        resp = client._client.models.generate_content(model=client._model, contents=prompt)
        text = (resp.text or "").strip()
        questions = _parse_survey_questions_from_text(text)
        # 최종 검토: 보기 누락·객관식 보기 없음 등 이상 여부를 Gemini로 한 번 더 확인 후 보정
        questions = _review_survey_questions_with_gemini(questions)
        return questions
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
3. 모든 객관식은 상호 배타적·포괄적 보기(예: 매우 그렇다 ~ 전혀 그렇지 않다, 또는 명확한 선택지)를 갖추세요. '기타', '해당 없음' 등이 필요하면 포함하세요.
4. 주관식은 선택지 없이 질문만 유지하세요.
5. 제목·질문·선택지가 명확히 구분되도록 하고, 출력 형식은 아래 "출력 형식"을 정확히 따르세요.

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
        if "유형: 주관식" in block or "유형:주관식" in block:
            q["type"] = "주관식"
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
            questions.append(q)
    if len(questions) > 25:
        questions = questions[:25]
    if not questions:
        return _default_survey_questions()
    return questions


def _default_survey_questions() -> list[dict]:
    """기본 설문 문항 (Gemini 실패 시)."""
    return [
        {"type": "객관식", "title": "가격 수용도", "question": "이 제품의 가격이 적정하다고 생각하시나요?", "options": ["전혀 그렇지 않다", "그렇지 않다", "보통", "그렇다", "매우 그렇다"]},
        {"type": "객관식", "title": "사용 편의성", "question": "이 제품의 사용 편의성에 만족하시나요?", "options": ["전혀 그렇지 않다", "그렇지 않다", "보통", "그렇다", "매우 그렇다"]},
        {"type": "객관식", "title": "브랜드 신뢰도", "question": "이 브랜드를 신뢰할 수 있다고 생각하시나요?", "options": ["전혀 그렇지 않다", "그렇지 않다", "보통", "그렇다", "매우 그렇다"]},
        {"type": "주관식", "title": "개선 의견", "question": "이 제품에서 개선되었으면 하는 점이 있다면 자유롭게 적어주세요.", "options": []},
    ]


def _render_survey_subpage():
    """서브페이지: 설문조사 — 설정된 가설 + 설문 문항 생성"""
    # 설계 선택·설문 문항으로 버튼: 사이드바 메뉴와 동일 높이 (2.25rem)
    st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"] > div:first-child .stButton > button,
    div[data-testid="column"]:nth-of-type(2) .stButton > button,
    div[data-testid="column"]:nth-of-type(3) .stButton > button { min-height: 2.25rem !important; height: 2.25rem !important; padding: 0.25rem 1rem !important; font-size: 0.9rem !important; }
    /* 왼쪽 컬럼(설정된 가설): 가설 수정·설문 문항 자동 생성 버튼 높이 = 사이드바 */
    div[data-testid="column"]:nth-of-type(3) .stButton > button { min-height: 2.25rem !important; height: 2.25rem !important; padding: 0.25rem 1rem !important; font-size: 0.9rem !important; }
    /* 오른쪽 컬럼(설문 문항): 수정·패널 지정 버튼 높이 = 사이드바 */
    div[data-testid="column"]:nth-of-type(4) .stButton > button { min-height: 2.25rem !important; height: 2.25rem !important; padding: 0.25rem 1rem !important; font-size: 0.9rem !important; }
    </style>
    """, unsafe_allow_html=True)
    # 상단 행: [설문조사] [← 설계 선택] [← 설문 문항으로] (AI 옵션 시) [← AI 옵션 선택으로] (진행 페이지 시)
    is_ai_survey = st.session_state.get("survey_ai_flow_page") == "ai_survey"
    is_ai_conduct = st.session_state.get("survey_ai_flow_page") == "ai_survey_conduct"
    if is_ai_conduct:
        cols = st.columns([2, 1, 1, 1])
    elif is_ai_survey:
        cols = st.columns([3, 1, 1])
    else:
        cols = st.columns([4, 1])
    with cols[0]:
        st.markdown("### 설문조사")
    with cols[1]:
        if st.button("← 설계 선택으로", key="survey_back_to_design"):
            st.session_state.survey_design_subpage = ""
            st.session_state.pop("survey_hypotheses", None)
            st.session_state.pop("survey_ai_flow_page", None)
            st.rerun()
    idx = 2
    if is_ai_survey or is_ai_conduct:
        with cols[idx]:
            if st.button("← 설문 문항으로", key="survey_back_to_questions"):
                st.session_state.survey_ai_flow_page = ""
                st.rerun()
        idx += 1
    if is_ai_conduct:
        with cols[idx]:
            if st.button("← AI 옵션 선택으로", key="survey_back_to_ai_options"):
                st.session_state.survey_ai_flow_page = "ai_survey"
                st.rerun()
    st.markdown("---")

    # 새 페이지: AI 설문 진행 — 설문 문항 + AI 설문 옵션(지역/패널/지도·요약·그래프) 함께 표시
    if is_ai_conduct:
        st.markdown("### AI 설문 진행 — 설문 문항 · 옵션")
        main_left, main_right = st.columns([1, 1], gap="large")

        with main_left:
            st.markdown("**설문 문항**")
            questions = st.session_state.get("survey_questions", _default_survey_questions())
            for i, q in enumerate(questions[:25], 1):
                typ = q.get("type", "객관식")
                title = (q.get("title", "") or "").strip()
                question = (q.get("question", "") or "").strip()
                options = q.get("options", [])
                title = re.sub(r"^(문항\s*\d+\s*[:\.\)]\s*|\d+[\.\)]\s*)", "", title, flags=re.IGNORECASE).strip() or title
                nums = "①②③④⑤⑥"
                opts_text = " / ".join([f"{nums[j] if j < len(nums) else j+1}. {o}" for j, o in enumerate(options)]) if options else ""
                question_display = f"{question} [주관식]" if typ == "주관식" else question
                st.markdown(f"""
                <div style="background: #f8fafc; padding: 12px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #e2e8f0;">
                    <strong style="color: #4f46e5;">문항 {i}: {title}</strong><br>
                    <span style="font-size: 13px;">{question_display}</span><br>
                    <span style="color: #64748b; font-size: 11px;">{opts_text}</span>
                </div>
                """, unsafe_allow_html=True)

        with main_right:
            st.markdown("**AI 설문 진행 옵션**")
            opt_inner_left, opt_inner_right = st.columns([1, 1], gap="medium")
            with opt_inner_left:
                selected_sido_label = st.session_state.get("survey_ai_sido_label", "경상북도 (37)")
                panel_count = st.session_state.get("survey_ai_panel_count", 10)
                st.metric("지역", selected_sido_label)
                st.metric("패널 수", f"{panel_count}명")
                try:
                    from pages.virtual_population_db import _render_korea_map
                    _render_korea_map(SIDO_LABEL_TO_CODE.get(selected_sido_label, "37"))
                except Exception:
                    st.caption("지도 로드 실패")
            with opt_inner_right:
                panel_df = st.session_state.get("survey_ai_panel_df")
                if panel_df is not None and not panel_df.empty:
                    n = len(panel_df)
                    st.metric("불러온 가상인구", f"{n:,}명")
                    _render_panel_summary_and_charts(panel_df, key_prefix="conduct")
                else:
                    st.info("가상인구 정보가 없습니다. AI 옵션 선택에서 다시 불러오세요.")
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
                from pages.virtual_population_db import _render_korea_map
                _render_korea_map(selected_sido_code)
            except Exception:
                st.caption("지도를 불러올 수 없습니다.")

        with col_right:
            st.markdown("**불러온 가상인구 정보**")
            panel_df = st.session_state.get("survey_ai_panel_df")
            _render_panel_summary_and_charts(panel_df)
            st.markdown("---")
            if panel_df is not None and len(panel_df) > 0:
                if st.button("AI 설문 진행", type="primary", use_container_width=True, key="survey_ai_conduct_btn"):
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
                col_edit, col_spacer, col_panel = st.columns([1, 2, 1])
                with col_edit:
                    if st.button("수정", key="survey_questions_edit_btn"):
                        st.session_state.survey_questions_editing = True
                        st.rerun()
                with col_panel:
                    if st.button("패널 지정", type="primary", key="survey_ai_flow_btn"):
                        st.session_state.survey_ai_flow_page = "ai_survey"
                        st.rerun()
        else:
            st.info("왼쪽에서 **설문 문항 자동 생성**을 누르면 여기에 설문 문항이 표시됩니다.")


def _render_interview_subpage():
    """서브페이지: 심층면접 — 인터뷰 방식(FGI/FGD/IDI) + 모더레이터 질문 가이드"""
    # 설계 선택 버튼: 사이드바 메뉴와 동일 높이 (2.25rem)
    st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"] > div:first-child .stButton > button { min-height: 2.25rem !important; height: 2.25rem !important; padding: 0.25rem 1rem !important; font-size: 0.9rem !important; }
    </style>
    """, unsafe_allow_html=True)
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown("### 심층면접")
    with col_btn:
        if st.button("← 설계 선택으로", key="interview_back_to_design"):
            st.session_state.survey_design_subpage = ""
            st.rerun()
    st.markdown("---")

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
