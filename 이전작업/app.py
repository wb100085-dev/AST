import streamlit as st
import pandas as pd
from datetime import datetime
import time
from io import BytesIO

# 내부 모듈
from utils.ipf_generator import IPFGenerator
from utils.kosis_client import KOSISClient
from utils.validator import Validator

# -----------------------------
# 0) 공통 설정/세션 초기화
# -----------------------------
st.set_page_config(
    page_title="가상인구 생성 시스템",
    page_icon="🧩",
    layout="wide"
)

if "page" not in st.session_state:
    st.session_state.page = "generate"  # generate | results | manage

if "generated_df" not in st.session_state:
    st.session_state.generated_df = None

if "validation_result" not in st.session_state:
    st.session_state.validation_result = None

if "history" not in st.session_state:
    st.session_state.history = []  # 이력(세션 기반)

if "kosis_sources" not in st.session_state:
    st.session_state.kosis_sources = []  # 통계 URL 목록(세션 기반)

if "last_params" not in st.session_state:
    st.session_state.last_params = {}

# -----------------------------
# 1) 컬럼명 한글화
# -----------------------------
COLUMN_KO = {
    "id": "ID",
    "name": "이름",
    "city": "거주지역",
    "gender": "성별",
    "age": "연령",
    "economic_activity": "경제활동",
    "spouse_economic_activity": "배우자경제활동",
    "education": "교육수준",
    "income": "월평균소득",
    "income_group": "소득그룹",
    "age_group": "연령대",
    "ipf_score": "IPF점수",
    "medical_satisfaction": "의료시설_만족도",
    "transport_satisfaction": "대중교통_만족도",
    "welfare_satisfaction": "사회복지_만족도",
    "government_satisfaction": "행정서비스_만족도",
}

def rename_columns_to_korean(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_KO)

# -----------------------------
# 2) 시도(17) 목록 + 지도(Plotly choropleth)
#    - 프로토타입: 지도는 시각화(hover), 선택은 multiselect로 확정
# -----------------------------
SIDO_17 = [
    "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시",
    "세종특별자치시", "경기도", "강원특별자치도", "충청북도", "충청남도",
    "전북특별자치도", "전라남도", "경상북도", "경상남도", "제주특별자치도"
]

# Plotly 한국 시도 choropleth는 geojson이 필요함
# 프로토타입 단계에서는 (1) 지도 "대체 시각화(막대+hover)" 또는
# (2) geojson 파일 확보 후 choropleth 구성 중 택1이 현실적임
# 여기서는 (A) 즉시 구현을 위해 "지도 영역에 시각화 + hover 강조"를 구현하고,
# geojson 확보 시 choropleth로 교체할 수 있게 구조를 열어둠.

def render_korea_map_placeholder(selected_sido: list[str]):
    """
    지도 대체 시각화(프로토타입 안정형)
    - 시도 목록을 지도 패널에 시각적으로 배치하고 hover/강조 제공
    - 실제 선택은 multiselect로 확정
    """
    import plotly.express as px

    df = pd.DataFrame({
        "시도": SIDO_17,
        "선택": [1 if s in selected_sido else 0 for s in SIDO_17],
        "가중치": [10 if s in selected_sido else 1 for s in SIDO_17],
    })

    fig = px.bar(
        df,
        x="가중치",
        y="시도",
        orientation="h",
        color="선택",
        color_continuous_scale=["#2b2b2b", "#22c55e"],
        height=520,
        title="대한민국 시도 선택(hover 가능, 선택 상태 강조)",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=45, b=10),
        yaxis_title=None,
        xaxis_title=None,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 3) 상단 내비게이션(탭 느낌)
# -----------------------------
def top_nav():
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("🧩 생성", use_container_width=True):
            st.session_state.page = "generate"
    with c2:
        if st.button("📌 결과", use_container_width=True):
            st.session_state.page = "results"
    with c3:
        if st.button("🗂️ 데이터관리", use_container_width=True):
            st.session_state.page = "manage"

st.title("🧩 가상인구 생성 시스템")
st.caption("Streamlit 프로토타입 | 시도 다중선택 + IPF 기반 생성 + 검증/다운로드")
top_nav()
st.markdown("---")

# =============================
# PAGE 1) 생성
# =============================
if st.session_state.page == "generate":
    st.subheader("생성 화면")

    left, right = st.columns([1.15, 0.85], gap="large")

    # -------------------------
    # 좌측: 지도 + 시도 선택
    # -------------------------
    with left:
        st.markdown("#### [대한민국 시도 선택](#)")
        st.write("시도 단위 다중 선택 후 우측 설정과 함께 생성 실행까지 진행 가능")
        st.markdown("- hover 시 강조 시각화 제공\n- 선택은 다중 선택 가능\n- 선택 결과가 생성 파라미터로 반영")

        selected_sido = st.multiselect(
            "시도 다중 선택",
            options=SIDO_17,
            default=st.session_state.last_params.get("selected_sido", ["경상북도"]),
        )

        if len(selected_sido) == 0:
            st.warning("최소 1개 시도 선택 필요")
        render_korea_map_placeholder(selected_sido)

    # -------------------------
    # 우측: 생성 설정 패널
    # -------------------------
    with right:
        st.markdown("#### [가상인구 생성 설정](#)")
        st.write("인구수, 가중치, 생성 컬럼을 설정 후 생성 실행 가능")
        st.markdown("- 가중치 합 100% 제약 적용\n- 생성 컬럼 선택 기반으로 결과 컬럼 구성\n- 생성 실행 시 결과 화면 자동 전환")

        population_size = st.number_input(
            "생성할 가상인구 수",
            min_value=100,
            max_value=100000,
            value=int(st.session_state.last_params.get("population_size", 1000)),
            step=100
        )

        st.markdown("**IPF 가중치(합 100%)**")
        w_income = st.slider("소득(%)", 0, 100, int(st.session_state.last_params.get("w_income", 50)), 5)
        w_age = st.slider("연령(%)", 0, 100, int(st.session_state.last_params.get("w_age", 20)), 5)
        w_edu = st.slider("교육(%)", 0, 100, int(st.session_state.last_params.get("w_edu", 15)), 5)
        w_gender = st.slider("성별(%)", 0, 100, int(st.session_state.last_params.get("w_gender", 10)), 5)
        w_rand = st.slider("무작위(%)", 0, 100, int(st.session_state.last_params.get("w_rand", 5)), 5)

        total_w = w_income + w_age + w_edu + w_gender + w_rand
        if total_w == 100:
            st.success(f"가중치 합계 100% 충족")
        else:
            st.error(f"가중치 합계 {total_w}%로 100% 필요")

        st.markdown("**생성할 만족도 컬럼**")
        gen_med = st.checkbox("의료시설 만족도", value=True)
        gen_trn = st.checkbox("대중교통 만족도", value=True)
        gen_wel = st.checkbox("사회복지 만족도", value=True)
        gen_gov = st.checkbox("행정서비스 만족도", value=False)

        can_run = (total_w == 100) and (len(selected_sido) > 0)

        if st.button("🚀 생성 실행", type="primary", use_container_width=True, disabled=not can_run):
            # 파라미터 저장
            st.session_state.last_params = {
                "selected_sido": selected_sido,
                "population_size": population_size,
                "w_income": w_income, "w_age": w_age, "w_edu": w_edu, "w_gender": w_gender, "w_rand": w_rand
            }

            weights = {
                "income": w_income / 100,
                "age": w_age / 100,
                "education": w_edu / 100,
                "gender": w_gender / 100,
                "random": w_rand / 100
            }

            with st.spinner("생성 및 검증 수행 중"):
                t0 = time.time()

                generator = IPFGenerator(weights)
                # region 파라미터는 기존 generator가 단일 region 문자열을 받는 구조임
                # 다중 시도는 생성 데이터의 거주지역을 선택 시도 중에서 랜덤 부여하는 방식으로 반영함
                # 이를 위해 우선 region="전체"로 두고, city/region 필드는 후처리로 반영
                people = generator.generate_base_population(population_size, region="전체")

                # 거주지역을 선택 시도 중 랜덤 할당
                import random
                for p in people:
                    p["city"] = random.choice(selected_sido)

                kosis_client = KOSISClient()
                columns_config = []

                if gen_med:
                    columns_config.append({
                        "name": "medical_satisfaction",
                        "levels": kosis_client.get_levels_list(),
                        "distribution": kosis_client.get_distribution_list()
                    })
                if gen_trn:
                    columns_config.append({
                        "name": "transport_satisfaction",
                        "levels": ["매우 불만족", "불만족", "보통", "만족", "매우 만족"],
                        "distribution": [0.03, 0.18, 0.48, 0.26, 0.05]
                    })
                if gen_wel:
                    columns_config.append({
                        "name": "welfare_satisfaction",
                        "levels": ["전혀 그렇지 않다", "그렇지 않은 편이다", "보통이다", "그런 편이다", "매우 그렇다", "모름/무응답"],
                        "distribution": [0.05, 0.15, 0.45, 0.25, 0.08, 0.02]
                    })
                if gen_gov:
                    columns_config.append({
                        "name": "government_satisfaction",
                        "levels": ["매우 불만족", "불만족", "보통", "만족", "매우 만족"],
                        "distribution": [0.023, 0.142, 0.640, 0.181, 0.013]
                    })

                df = generator.generate_multiple_columns(people, columns_config)
                df = rename_columns_to_korean(df)

                # 검증(의료시설_만족도 존재 시 해당 컬럼으로)
                validator = Validator()
                validation = None
                if "의료시설_만족도" in df.columns:
                    data = df["의료시설_만족도"].tolist()
                    validation = validator.validate_all(
                        data=data,
                        people=df.to_dict("records"),
                        target_levels=kosis_client.get_levels_list(),
                        target_dist=kosis_client.get_distribution_list()
                    )

                t1 = time.time()

                st.session_state.generated_df = df
                st.session_state.validation_result = validation

                # 이력 저장(세션 기반)
                st.session_state.history.append({
                    "생성일시": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "시도선택": ", ".join(selected_sido),
                    "인구수": len(df),
                    "생성컬럼": ", ".join([c["name"] for c in columns_config]) if columns_config else "없음",
                    "가중치": weights,
                    "검증통과": (validation.get("all_passed") if validation else None),
                    "소요시간(초)": round(t1 - t0, 2)
                })

            # ★ 자동 이동: 결과 화면으로 상태 전환
            st.session_state.page = "results"
            st.rerun()

# =============================
# PAGE 2) 결과
# =============================
elif st.session_state.page == "results":
    st.subheader("결과 화면")

    df = st.session_state.generated_df
    validation = st.session_state.validation_result

    if df is None:
        st.info("생성된 결과가 없음. '생성' 화면에서 먼저 생성 실행 필요")
    else:
        # 요약
        p = st.session_state.last_params
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            st.markdown("#### [생성 요약](#)")
            st.write(f"생성 인구수 {len(df):,}명으로 생성 완료 상태임")
            st.markdown("- 선택 시도: " + ", ".join(p.get("selected_sido", [])))
            st.markdown(f"- 인구수: {p.get('population_size', len(df)):,}")
            st.markdown(f"- 생성 컬럼 수: {len(df.columns)}개")

        with c2:
            st.markdown("#### [가중치 요약](#)")
            st.write("IPF 가중치 설정값 반영 상태임")
            st.markdown(f"- 소득 {p.get('w_income')}%")
            st.markdown(f"- 연령 {p.get('w_age')}%")
            st.markdown(f"- 교육 {p.get('w_edu')}%")

        with c3:
            st.markdown("#### [다운로드](#)")
            st.write("전체 데이터 파일로 즉시 저장 가능 상태임")
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="가상인구")
            st.download_button(
                "📥 Excel 다운로드",
                data=output.getvalue(),
                file_name=f"가상인구_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            csv_data = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "📥 CSV 다운로드",
                data=csv_data,
                file_name=f"가상인구_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("---")

        # 미리보기
        st.markdown("#### [생성 결과 미리보기](#)")
        st.write(f"전체 {len(df):,}명 중 일부 행 표시, 전체는 다운로드로 확인 가능")
        st.markdown("- 기본적으로 50행 표시\n- 필요 시 전체 표시 옵션 제공\n- 대용량(>1만)에서 전체 표시는 지연 가능")
        show_all = st.checkbox("전체 데이터 표시(대용량 시 지연)", value=False)
        st.dataframe(df if show_all else df.head(50), use_container_width=True, height=520)

        st.markdown("---")

        # 검증 결과
        st.markdown("#### [검증 결과](#)")
        if validation is None:
            st.warning("검증 결과가 없음(의료시설_만족도 미생성 또는 검증 스킵)")
        else:
            passed = validation.get("all_passed", False)
            st.success("전체 검증 통과 상태임") if passed else st.warning("일부 검증 이슈 가능 상태임")

            # 표 1개 포함(요구사항 충족): 검증 요약 표
            v_con = validation.get("consecutive", {})
            v_dst = validation.get("distribution", {})
            v_cst = validation.get("consistency", {})

            summary_tbl = pd.DataFrame([
                {"검증항목": "연속 패턴", "핵심지표": "최대 연속", "결과": v_con.get("max_consecutive"), "판정": "PASS" if v_con.get("passed") else "FAIL"},
                {"검증항목": "분포 정확도", "핵심지표": "평균 오차(%p)", "결과": v_dst.get("avg_error"), "판정": "PASS" if v_dst.get("passed") else "FAIL"},
                {"검증항목": "논리 개연성", "핵심지표": "모순 비율(%)", "결과": v_cst.get("percentage"), "판정": "PASS" if v_cst.get("passed") else "WARN"},
            ])
            st.dataframe(summary_tbl, use_container_width=True)

            with st.expander("분포 정확도 상세 보기"):
                details = v_dst.get("details", [])
                if details:
                    st.dataframe(pd.DataFrame(details), use_container_width=True)
                else:
                    st.info("상세 데이터 없음")

# =============================
# PAGE 3) 데이터 관리
# =============================
elif st.session_state.page == "manage":
    st.subheader("데이터 관리 화면")

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown("#### [생성 이력](#)")
        st.write("세션 기반 이력 관리이며, 재실행 시 초기화될 수 있음")
        st.markdown("- 생성일시/시도/인구수/가중치/검증통과 여부 기록\n- 향후 SQLite 저장으로 영속화 가능\n- 프로젝트명/메모 필드 추가 가능")

        if len(st.session_state.history) == 0:
            st.info("이력 없음")
        else:
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df, use_container_width=True, height=520)

    with right:
        st.markdown("#### [KOSIS 통계 URL 관리](#)")
        st.write("통계 소스 목록을 세션에 저장하는 형태임")
        st.markdown("- URL 추가/목록화\n- 향후 API 파싱 결과(분포/레벨) 저장 가능\n- 컬럼별 매핑까지 확장 가능")

        url_name = st.text_input("통계명", placeholder="예: 의료시설 만족도(2024)")
        url_value = st.text_input("KOSIS URL", placeholder="https://kosis.kr/openapi/...")
        if st.button("➕ URL 추가", use_container_width=True):
            if url_name and url_value:
                st.session_state.kosis_sources.append({
                    "통계명": url_name,
                    "URL": url_value,
                    "등록일시": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("추가 완료")
                st.rerun()
            else:
                st.error("통계명과 URL 모두 필요")

        st.markdown("---")
        if len(st.session_state.kosis_sources) == 0:
            st.info("등록된 통계 URL 없음")
        else:
            src_df = pd.DataFrame(st.session_state.kosis_sources)
            st.dataframe(src_df, use_container_width=True, height=360)
