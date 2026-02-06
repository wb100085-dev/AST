"""
AI Social Twin - 가상인구 생성 및 조사 설계 애플리케이션
"""
from __future__ import annotations

import hashlib
import os
import re
import traceback
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from io import BytesIO
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

# 캐시 유효시간: 24시간 (초)
CACHE_TTL_SECONDS = 24 * 3600  # 86400


def _hash_dataframe(df: pd.DataFrame) -> str:
    """st.cache_data용 DataFrame 해시. 동일 데이터면 동일 해시."""
    return hashlib.md5(df.to_json(orient="split").encode()).hexdigest()


# 타입 검사용 (실제 로딩은 main()에서 지연 로딩)
if TYPE_CHECKING:
    from google import genai
    from utils.kosis_client import KosisClient
    from utils.ipf_generator import generate_base_population
    from utils.gemini_client import GeminiClient
    from utils.step2_records import STEP2_RECORDS_DIR, list_step2_records, save_step2_record

from core.constants import (
    APP_TITLE,
    AUTOSAVE_PATH,
    EXPORT_SHEET_NAME,
    EXPORT_COLUMNS,
    STEP2_COLUMN_RENAME,
    DEFAULT_WEIGHTS_SCORE,
    SIDO_MASTER,
    SIDO_CODE,
    SIDO_NAME,
    SIDO_LABELS,
    SIDO_LABEL_TO_CODE,
    SIDO_CODE_TO_NAME,
    AXIS_KEYS,
    AXIS_LABELS,
)
from core.db import (
    db_init,
    db_upsert_stat,
    db_delete_stat_by_id,
    db_update_stat_by_id,
    db_upsert_axis_margin_stat,
    db_upsert_template,
    build_stats_template_xlsx_kr,
    build_stats_export_xlsx_kr,
    get_export_filename,
    import_stats_from_excel_kr,
    get_sido_vdb_stats,
)
from core.session_cache import (
    get_cached_db_list_stats,
    get_cached_db_axis_margin_stats,
    get_cached_db_six_axis_stat_ids,
    invalidate_db_stats_cache,
    invalidate_db_axis_margin_cache,
)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_cached_kosis_json(url: str) -> list:
    """KOSIS API JSON 결과를 24시간 캐시. fetch_json 대체용."""
    import requests
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        data = data.get("data", []) if isinstance(data, dict) else []
    return data


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def cached_generate_base_population(
    n: int,
    selected_sigungu_json: str,
    weights_6axis_json: str,
    sigungu_pool_json: str,
    seed: int,
    margins_axis_json: str,
    apply_ipf_flag: bool,
):
    """generate_base_population 결과를 24시간 캐시. 동일 인자면 재계산 생략."""
    from utils.ipf_generator import generate_base_population
    selected_sigungu = json.loads(selected_sigungu_json) if selected_sigungu_json else []
    weights_6axis = json.loads(weights_6axis_json) if weights_6axis_json else {}
    sigungu_pool = json.loads(sigungu_pool_json) if sigungu_pool_json else []
    margins_axis = json.loads(margins_axis_json) if margins_axis_json else {}
    return generate_base_population(
        n=n,
        selected_sigungu=selected_sigungu,
        weights_6axis=weights_6axis,
        sigungu_pool=sigungu_pool,
        seed=seed,
        margins_axis=margins_axis,
        apply_ipf_flag=apply_ipf_flag,
    )


def _apply_step2_column_rename(df: pd.DataFrame) -> pd.DataFrame:
    """2단계 결과 컬럼명을 출력용으로 변경 (존재하는 컬럼만)."""
    rename = {k: v for k, v in STEP2_COLUMN_RENAME.items() if k in df.columns}
    return df.rename(columns=rename) if rename else df


@st.cache_data(ttl=CACHE_TTL_SECONDS, hash_funcs={pd.DataFrame: _hash_dataframe})
def _build_excel_bytes_for_download(df: pd.DataFrame, _is_step2: bool) -> bytes:
    """다운로드 탭에서 요청 시에만 Excel 바이트 생성 (캐시됨)."""
    buf = BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf.getvalue()


def _apply_step2_logical_consistency(df: pd.DataFrame) -> None:
    """2단계 결과 개연성: 조건에 맞지 않으면 해당 셀 비움 또는 값 조정 (inplace)."""
    # 1. 소득 없음 → 소득 만족도·소비생활만족도 비움
    if "소득 여부" in df.columns:
        no_income = df["소득 여부"].astype(str).str.strip().str.lower().isin(["아니오", "없음", "없다", "0", "no", ""])
        for col in ["소득 만족도", "소비생활만족도"]:
            if col in df.columns:
                df.loc[no_income, col] = ""
    
    # 2. 자녀 없음 → 공교육비·사교육비 비움 (가상인구 DB 컬럼명 기준)
    if "학생 및 미취학 자녀 유무" in df.columns:
        no_child = df["학생 및 미취학 자녀 유무"].astype(str).str.strip().isin(["무", "없음", "없다"])
        for col in ["공교육비", "사교육비"]:
            if col in df.columns:
                df.loc[no_child, col] = ""
    
    # 3. 자원봉사 없음 → 방식·시간 비움
    if "자원봉사 활동 여부" in df.columns:
        no_vol = df["자원봉사 활동 여부"].astype(str).str.strip().isin(["없다", "없음"])
        for col in ["자원봉사 활동 방식", "지난 1년 동안 자원봉사 활동 시간"]:
            if col in df.columns:
                df.loc[no_vol, col] = ""
    
    # 4. 기부 없음 → 기부 방식·기부금액 비움
    if "기부 여부" in df.columns:
        no_donation = df["기부 여부"].astype(str).str.strip().isin(["없다", "없음"])
        for col in ["기부 방식", "기부금액(만원)"]:
            if col in df.columns:
                df.loc[no_donation, col] = ""
    
    # 5. 준법 잘 지키면 → 법을 지키지 않는 이유 비움
    if "자신의 평소 준법수준" in df.columns and "평소 법을 지키지 않는 주된 이유" in df.columns:
        obey = df["자신의 평소 준법수준"].astype(str).str.strip().str.contains("잘 지킨다", na=False)
        df.loc[obey, "평소 법을 지키지 않는 주된 이유"] = ""
    
    # ==========================================
    # 인구통계학적 일관성 및 논리성 규칙 추가
    # ==========================================
    
    # 6. 연령-경제활동 불일치: 75세 초과는 유급 경제활동 불가
    if "연령" in df.columns and "경제활동" in df.columns and "종사상 지위" in df.columns:
        try:
            age_numeric = pd.to_numeric(df["연령"], errors='coerce')
            elderly = age_numeric > 75
            econ_active = df["경제활동"].astype(str).str.strip() == "경제활동"
            # 75세 초과 + 경제활동인 경우, 종사상 지위를 무급 또는 비경제활동으로 조정
            invalid_elderly = elderly & econ_active
            if invalid_elderly.sum() > 0:
                # 종사상 지위가 유급인 경우 무급으로 변경
                invalid_indices = df[invalid_elderly].index
                paid_status = df.loc[invalid_indices, "종사상 지위"].astype(str).str.strip().str.contains("유급|상용|임시|일용", na=False, regex=True)
                paid_indices = invalid_indices[paid_status]
                if len(paid_indices) > 0:
                    df.loc[paid_indices, "종사상 지위"] = "무급"
                # 또는 비경제활동으로 변경 (50% 확률)
                np.random.seed(42)
                to_inactive_mask = pd.Series(np.random.rand(len(df)) < 0.5, index=df.index)
                to_inactive = invalid_elderly & to_inactive_mask
                if to_inactive.sum() > 0:
                    df.loc[to_inactive, "경제활동"] = "비경제활동"
        except Exception:
            pass
    
    # 7. 거주기간 논리 오류
    if "연령" in df.columns:
        try:
            age_numeric = pd.to_numeric(df["연령"], errors='coerce')
            # 7-1. 30세 미만이 20년 이상 거주 불가
            if "시도 거주기간" in df.columns:
                young = age_numeric < 30
                long_residence = df["시도 거주기간"].astype(str).str.contains("20년|30년|40년|50년", na=False, regex=True)
                invalid_residence = young & long_residence
                if invalid_residence.sum() > 0:
                    # 최대 거주기간을 연령에 맞게 조정 (예: 25세면 최대 25년)
                    for idx in df[invalid_residence].index:
                        age_val = age_numeric.iloc[idx]
                        if pd.notna(age_val) and age_val < 30:
                            max_years = int(age_val)
                            if max_years < 5:
                                df.loc[idx, "시도 거주기간"] = "5년 미만"
                            elif max_years < 10:
                                df.loc[idx, "시도 거주기간"] = "5-10년"
                            elif max_years < 20:
                                df.loc[idx, "시도 거주기간"] = "10-20년"
                            else:
                                df.loc[idx, "시도 거주기간"] = "20년 이상"
            
            # 7-2. 시군구 거주기간 > 시도 거주기간 불가
            if "시도 거주기간" in df.columns and "시군구 거주기간" in df.columns:
                # 거주기간을 숫자로 변환하여 비교
                def parse_residence_years(s):
                    if pd.isna(s) or s == "":
                        return 0
                    s_str = str(s)
                    if "5년 미만" in s_str:
                        return 2.5
                    elif "5-10년" in s_str:
                        return 7.5
                    elif "10-20년" in s_str:
                        return 15
                    elif "20년" in s_str or "30년" in s_str or "40년" in s_str or "50년" in s_str:
                        return 25
                    return 0
                
                sido_years = df["시도 거주기간"].apply(parse_residence_years)
                sigungu_years = df["시군구 거주기간"].apply(parse_residence_years)
                invalid_order = sigungu_years > sido_years
                if invalid_order.sum() > 0:
                    # 시군구 거주기간을 시도 거주기간 이하로 조정
                    for idx in df[invalid_order].index:
                        sido_val = sido_years.iloc[idx]
                        if sido_val <= 2.5:
                            df.loc[idx, "시군구 거주기간"] = "5년 미만"
                        elif sido_val <= 7.5:
                            df.loc[idx, "시군구 거주기간"] = "5-10년"
                        elif sido_val <= 15:
                            df.loc[idx, "시군구 거주기간"] = "10-20년"
                        else:
                            df.loc[idx, "시군구 거주기간"] = df.loc[idx, "시도 거주기간"]
        except Exception:
            pass
    
    # 8. 경제활동-소득 불일치: 비경제활동자(자녀 없음)는 고소득 불가
    if "경제활동" in df.columns and "월평균소득" in df.columns and "학생 및 미취학 자녀 유무" in df.columns:
        try:
            inactive = df["경제활동"].astype(str).str.strip() == "비경제활동"
            no_child = df["학생 및 미취학 자녀 유무"].astype(str).str.strip().isin(["무", "없음", "없다"])
            high_income = df["월평균소득"].astype(str).str.contains("200|300|400|500|600|700|800", na=False, regex=True)
            invalid_income = inactive & no_child & high_income
            if invalid_income.sum() > 0:
                # 고소득을 낮은 소득으로 조정
                df.loc[invalid_income, "월평균소득"] = np.random.choice(
                    ["50만원미만", "50-100만원", "100-200만원"],
                    size=invalid_income.sum(),
                    p=[0.3, 0.4, 0.3]
                )
        except Exception:
            pass
    
    # 9. 교육-연령 부적합: 22세 미만은 대졸 이상 불가
    if "연령" in df.columns and "교육정도" in df.columns:
        try:
            age_numeric = pd.to_numeric(df["연령"], errors='coerce')
            young = age_numeric < 22
            high_edu = df["교육정도"].astype(str).str.contains("대졸|대학|대학원", na=False, regex=True)
            invalid_edu = young & high_edu
            if invalid_edu.sum() > 0:
                # 대졸 이상을 고졸 이하로 조정
                df.loc[invalid_edu, "교육정도"] = np.random.choice(
                    ["중졸이하", "고졸"],
                    size=invalid_edu.sum(),
                    p=[0.3, 0.7]
                )
        except Exception:
            pass
    
    # 10. 배우자-자녀 연계 이상: 25세 미만은 배우자+자녀 동시 보유 불가
    if "연령" in df.columns:
        try:
            age_numeric = pd.to_numeric(df["연령"], errors='coerce')
            very_young = age_numeric < 25
            
            # 배우자 보유 여부 확인 (배우자의 경제활동 상태 컬럼 사용)
            has_spouse = pd.Series([False] * len(df), index=df.index)
            spouse_cols = [col for col in df.columns if "배우자" in col and ("경제활동" in col or "상태" in col)]
            if spouse_cols:
                spouse_col = spouse_cols[0]
                has_spouse = df[spouse_col].astype(str).str.strip().isin(["경제활동", "비경제활동", "유", "있음", "있다", "있습니다"])
            
            # 자녀 보유 여부 확인 (가상인구 DB 컬럼명 기준)
            has_child = pd.Series([False] * len(df), index=df.index)
            if "학생 및 미취학 자녀 유무" in df.columns:
                has_child = df["학생 및 미취학 자녀 유무"].astype(str).str.strip().isin(["유", "있음", "있다", "있습니다"])
            
            invalid_family = very_young & has_spouse & has_child
            if invalid_family.sum() > 0:
                # 배우자 또는 자녀 중 하나만 유지 (50% 확률로 배우자 제거)
                np.random.seed(42)
                remove_spouse_mask = pd.Series(np.random.rand(len(df)) < 0.5, index=df.index)
                remove_spouse = invalid_family & remove_spouse_mask
                
                if spouse_cols and remove_spouse.sum() > 0:
                    df.loc[remove_spouse, spouse_cols[0]] = "무"
                
                # 나머지는 자녀 제거
                remove_child = invalid_family & ~remove_spouse
                if "학생 및 미취학 자녀 유무" in df.columns and remove_child.sum() > 0:
                    df.loc[remove_child, "학생 및 미취학 자녀 유무"] = "무"
        except Exception:
            pass
    
    # 11. 주거 안정성: 주택점유형태와 향후거주의사 연계
    if "주택점유형태" in df.columns and "향후 10년 거주 희망의사" in df.columns:
        try:
            # 자가인데 이사 희망하는 경우는 가능하지만, 전세/월세인데 계속 거주 희망은 논리적
            owned = df["주택점유형태"].astype(str).str.contains("자가|소유", na=False, regex=True)
            will_move = df["향후 10년 거주 희망의사"].astype(str).str.contains("이사|이주|이동", na=False, regex=True)
            # 전세/월세인데 계속 거주 희망하는 경우는 논리적으로 맞지 않으므로 이사 희망으로 변경
            rental = df["주택점유형태"].astype(str).str.contains("전세|월세|임대", na=False, regex=True)
            will_stay = df["향후 10년 거주 희망의사"].astype(str).str.contains("계속|유지|그대로", na=False, regex=True)
            invalid_stay = rental & will_stay
            if invalid_stay.any():
                df.loc[invalid_stay, "향후 10년 거주 희망의사"] = "이사 희망"
        except Exception:
            pass
    
    # 12. 경제활동 연계성: 비경제활동자는 종사상 지위, 직업, 근로만족도 비움
    if "경제활동" in df.columns:
        inactive = df["경제활동"].astype(str).str.strip() == "비경제활동"
        for col in ["종사상 지위", "하는 일의 종류(직업 종분류)", "근로만족도"]:
            if col in df.columns:
                df.loc[inactive, col] = ""
    
    # 13. 복지 수혜 적격성: 연령-소득-자녀-복지만족도 연계
    if "연령" in df.columns and "월평균소득" in df.columns and "학생 및 미취학 자녀 유무" in df.columns:
        try:
            age_numeric = pd.to_numeric(df["연령"], errors='coerce')
            # 13-1. 자녀 없으면 임신·출산·육아 복지 만족도 비움
            if "임신·출산·육아에 대한 복지 만족도" in df.columns:
                no_child = df["학생 및 미취학 자녀 유무"].astype(str).str.strip().isin(["무", "없음", "없다"])
                df.loc[no_child, "임신·출산·육아에 대한 복지 만족도"] = ""
            
            # 13-2. 고소득자(300만원 이상)는 저소득층 복지 만족도 비움 (일반적으로)
            if "저소득층 등 취약계층에 대한 복지 만족도" in df.columns:
                high_income = df["월평균소득"].astype(str).str.contains("300|400|500|600|700|800", na=False, regex=True)
                # 단, 고령자(65세 이상)는 예외로 둠
                not_elderly = age_numeric < 65
                invalid_welfare = high_income & not_elderly
                df.loc[invalid_welfare, "저소득층 등 취약계층에 대한 복지 만족도"] = ""
        except Exception:
            pass
    
    # 14. 교육-직업 적합성: 교육수준과 직업종류의 적합성 (기본적인 검증)
    if "교육정도" in df.columns and "하는 일의 종류(직업 종분류)" in df.columns:
        try:
            # 고졸 이하인데 전문직/관리직은 비현실적일 수 있음 (선택적 규칙)
            low_edu = df["교육정도"].astype(str).str.contains("중졸|고졸", na=False, regex=True)
            professional_job = df["하는 일의 종류(직업 종분류)"].astype(str).str.contains("전문|관리|경영|의사|변호사|회계사", na=False, regex=True)
            # 고졸 이하인데 전문직/관리직인 경우 직업을 비움 (완화된 규칙)
            invalid_edu_job = low_edu & professional_job
            if invalid_edu_job.any():
                df.loc[invalid_edu_job, "하는 일의 종류(직업 종분류)"] = ""
        except Exception:
            pass


@st.cache_data(ttl=CACHE_TTL_SECONDS, hash_funcs={pd.DataFrame: _hash_dataframe})
def _apply_step2_logical_consistency_cached(df: pd.DataFrame) -> pd.DataFrame:
    """2단계 논리 일관성 적용 결과 반환 (캐시됨). 동일 df 입력 시 재계산 생략."""
    out = df.copy()
    _apply_step2_logical_consistency(out)
    return out


# -----------------------------
# 4. UI Utilities
# -----------------------------
def group_by_category(stats: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for s in stats:
        out.setdefault(s["category"], []).append(s)
    return out


_SESSION_DEFAULTS = {
    "app_started": False,
    "generated_df": None,
    "report": None,
    "sigungu_list": ["전체"],
    "selected_categories": [],
    "selected_stats": [],
    "last_error": None,
    "selected_sido_label": "경상북도 (37)",
}


def ensure_session_state():
    for key, default in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


# -----------------------------
# 5. KOSIS Helpers
# -----------------------------

def load_sigungu_options(sido_code: str, kosis_client: KosisClient) -> List[str]:
    """
    KOSIS 인구 테이블에서 시군구 목록 추출
    """
    try:
        # kosis_client가 리스트로 잘못 전달된 경우 방어 코드
        if isinstance(kosis_client, list):
            st.warning("kosis_client 타입 오류")
            return []
        
        url = (
            f"https://kosis.kr/openapi/statisticsData.do?"
            f"method=getList&apiKey=YOUR_KEY&format=json&jsonVD=Y&"
            f"userStatsId=...&objL1={sido_code}"
        )
        data = get_cached_kosis_json(url)
        sigungu_list = kosis_client.extract_sigungu_list_from_population_table(
            data, sido_prefix=SIDO_CODE_TO_NAME.get(sido_code, "")
        )
        return sigungu_list
    except Exception as e:
        st.warning(f"시군구 목록 로드 실패: {e}")
        return []



def page_data_management():
    """데이터 관리 페이지"""
    st.title("데이터 관리")

    sido_label = st.selectbox(
        "시도 선택",
        options=SIDO_LABELS,
        key="data_mgmt_sido",
    )
    sido_code = SIDO_LABEL_TO_CODE[sido_label]
    sido_name = SIDO_CODE_TO_NAME[sido_code]

    st.markdown("---")
    st.subheader("통계 목록")

    # 통계 목록 업로드/다운로드
    col1, col2 = st.columns(2)
    with col1:
        if st.button("템플릿 다운로드", key="download_stats_template"):
            template_bytes = build_stats_template_xlsx_kr(sido_code)
            st.download_button(
                "통계목록 템플릿.xlsx 다운로드",
                data=template_bytes,
                file_name=f"{sido_name}_통계목록_템플릿.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        if st.button("활성화 통계 내보내기", key="export_active_stats"):
            export_bytes = build_stats_export_xlsx_kr(sido_code)
            st.download_button(
                "통계목록.xlsx 다운로드",
                data=export_bytes,
                file_name=f"{sido_name}_통계목록_활성화.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with col2:
        uploaded_file = st.file_uploader(
            "통계 목록 업로드(.xlsx)",
            type=["xlsx"],
            key="upload_stats_list",
        )
        if uploaded_file:
            file_bytes = uploaded_file.read()
            result = import_stats_from_excel_kr(sido_code, file_bytes)
            if not result.get("ok", True):
                st.error(result.get("error", "업로드 실패"))
            else:
                st.success(
                    f"통계 목록 업로드 완료\n"
                    f"- 반영: {result.get('반영건수', 0)}건\n"
                    f"- 스킵: {result.get('스킵건수', 0)}건\n"
                    f"- 오류: {result.get('오류건수', 0)}건"
                )
                if result.get("오류상세"):
                    with st.expander("오류 상세"):
                        st.json(result["오류상세"])
                st.rerun()

    # 통계 목록 표시
    st.markdown("---")
    all_stats = get_cached_db_list_stats(sido_code)
    if not all_stats:
        st.info("등록된 통계가 없습니다.")
    else:
        df_stats = pd.DataFrame(all_stats)
        df_stats["is_active"] = df_stats["is_active"].map({1: "Y", 0: "N"})
        st.dataframe(df_stats, use_container_width=True)

    # 6축 고정 마진 통계 소스 설정 (저장 후에도 최신값 표시되도록 DB 직접 조회)
    st.markdown("---")
    st.subheader("6축 고정 마진 통계 소스")
    st.markdown("각 축의 목표 마진을 제공할 통계를 선택한 뒤 **「6축 설정 저장」** 버튼을 누르세요.")

    active_stats = [s for s in all_stats if s["is_active"] == 1]
    if not active_stats:
        st.info("활성화된 통계가 없습니다.")
    else:
        # id를 int로 통일해 Supabase 반환값(문자열 등)과 비교 오류 방지
        stat_options = {int(s["id"]): f"[{s['category']}] {s['name']}" for s in active_stats}
        option_list = [None] + list(stat_options.keys())
        axis_list = [
            ("sigungu", "거주지역"),
            ("gender", "성별"),
            ("age", "연령"),
            ("econ", "경제활동"),
            ("income", "소득"),
            ("edu", "교육"),
        ]

        # 6축 설정 한 번의 쿼리로 조회 (세션 캐시, 저장 시 무효화로 최신값 반영)
        from core.session_cache import get_cached_db_all_axis_margin_stats
        axis_margin_by_key = get_cached_db_all_axis_margin_stats(sido_code)
        selections = {}
        load_failed_count = 0
        for axis_key, axis_label in axis_list:
            current_stat = axis_margin_by_key.get(axis_key)
            current_id = None
            if current_stat and current_stat.get("stat_id") is not None:
                try:
                    current_id = int(current_stat["stat_id"])
                except (TypeError, ValueError):
                    current_id = None
            else:
                load_failed_count += 1
            if current_id is not None and current_id not in stat_options:
                current_id = None
            default_idx = 0 if current_id is None else (option_list.index(current_id) if current_id in option_list else 0)

            selected_id = st.selectbox(
                f"{axis_label} ({axis_key})",
                options=option_list,
                format_func=lambda x, opts=stat_options: "선택 안 함" if x is None else opts.get(x, "?"),
                index=default_idx,
                key=f"axis_margin_{axis_key}",
            )
            selections[axis_key] = selected_id

        # DB에는 있는데 6축이 전부 조회되지 않으면 RLS(권한) 문제일 수 있음
        if load_failed_count >= 6:
            st.info(
                "💡 **6축 설정이 DB에 있는데도 표시되지 않나요?** "
                "Supabase 대시보드 → **SQL Editor**에서 프로젝트의 **docs/SUPABASE_RLS_정책_적용.sql** 내용을 실행하세요. "
                "anon 역할에 SELECT가 허용되어야 앱에서 조회됩니다."
            )

        if st.button("6축 설정 저장", type="primary", key="save_six_axis"):
            updated = 0
            for axis_key, axis_label in axis_list:
                sid = selections.get(axis_key)
                if sid is not None:
                    try:
                        db_upsert_axis_margin_stat(sido_code, axis_key, int(sid))
                        updated += 1
                    except Exception as e:
                        st.error(f"{axis_label} 저장 실패: {e}")
            if updated:
                invalidate_db_axis_margin_cache(sido_code)
                st.success(f"6축 설정 {updated}건 저장되었습니다. 새로고침 시 유지됩니다.")
                st.rerun()

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def convert_kosis_to_distribution_cached(kosis_data_json: str, axis_key: str) -> Tuple[list, list]:
    """KOSIS 데이터 변환 결과를 캐시. 동일 (데이터, 축)이면 재계산 생략."""
    kosis_data = json.loads(kosis_data_json) if kosis_data_json else []
    return _convert_kosis_to_distribution_impl(kosis_data, axis_key)


def _convert_kosis_to_distribution_impl(kosis_data, axis_key: str) -> Tuple[list, list]:
    """KOSIS 데이터를 확률 분포로 변환 (labels, probabilities). 순수 계산만 수행."""
    if not kosis_data:
        return [], []
    if isinstance(kosis_data, dict):
        kosis_data = kosis_data.get("data", []) if "data" in kosis_data else [kosis_data]
    if not isinstance(kosis_data, list):
        return [], []
    labels = []
    values = []
    if axis_key == "sigungu":
        seen = set()
        for row in kosis_data:
            if isinstance(row, dict):
                label = row.get("C1_NM", "").strip()
                val = row.get("DT", "0")
                if label and label not in ["소계", "합계", "Total", "경상북도"] and label not in seen:
                    try:
                        values.append(float(val))
                        labels.append(label)
                        seen.add(label)
                    except Exception:
                        pass
    elif axis_key == "gender":
        gender_map = {"남자": 0, "여자": 0}
        for row in kosis_data:
            if isinstance(row, dict):
                label = row.get("C2_NM", "").strip()
                val = row.get("DT", "0")
                try:
                    val_float = float(val)
                    if "남자" in label or label == "남":
                        gender_map["남자"] += val_float
                    elif "여자" in label or label == "여":
                        gender_map["여자"] += val_float
                except (ValueError, TypeError):
                    pass
        for gender in ["남자", "여자"]:
            if gender_map[gender] > 0:
                labels.append(gender)
                values.append(gender_map[gender])
    elif axis_key == "age":
        age_map = {}
        for row in kosis_data:
            if not isinstance(row, dict):
                continue
            age_str = (row.get("C3_NM") or "").strip()
            try:
                val = float(str(row.get("DT", "0")).replace(",", "").strip() or 0)
            except (ValueError, TypeError):
                val = 0
            if not age_str or age_str == "계":
                continue
            range_match = re.search(r"(\d+)\s*[-~]\s*(\d+)", age_str)
            if range_match:
                low = int(range_match.group(1))
                high = int(range_match.group(2))
                if low > high:
                    low, high = high, low
                low = max(20, low)
                high = min(120, high)
                if low <= high:
                    count = high - low + 1
                    per_age = val / count
                    for a in range(low, high + 1):
                        age_map[a] = age_map.get(a, 0) + per_age
            else:
                single_match = re.search(r"(\d+)", age_str)
                if single_match:
                    age_num = int(single_match.group(1))
                    if 20 <= age_num <= 120:
                        age_map[age_num] = age_map.get(age_num, 0) + val
        for age_num in sorted(age_map.keys()):
            labels.append(age_num)
            values.append(age_map[age_num])
    elif axis_key == "econ":
        econ_map = {}
        for row in kosis_data:
            if isinstance(row, dict):
                label = row.get("C2_NM", "").strip()
                val = row.get("DT", "0").strip()
                if val in ("-", "", None):
                    continue
                try:
                    val_float = float(val)
                except (ValueError, TypeError):
                    continue
                if "하였다" in label or "일하였음" in label or "취업" in label or "경제활동" in label or "일했음" in label:
                    econ_map["경제활동"] = econ_map.get("경제활동", 0) + val_float
                elif "하지 않았다" in label or "일하지 않았음" in label or "구직" in label or "실업" in label or "비경제" in label:
                    econ_map["비경제활동"] = econ_map.get("비경제활동", 0) + val_float
                else:
                    econ_map["비경제활동"] = econ_map.get("비경제활동", 0) + val_float
        for econ_type, total_val in econ_map.items():
            labels.append(econ_type)
            values.append(total_val)
    elif axis_key == "edu":
        edu_map = {"중졸이하": 0, "고졸": 0, "대졸이상": 0}
        for row in kosis_data:
            if isinstance(row, dict):
                label = ""
                for field in ["C2_NM", "C3_NM", "C4_NM", "C5_NM"]:
                    val_field = str(row.get(field, "")).strip()
                    if val_field and val_field not in ("계", "전체", "소계"):
                        if any(k in val_field for k in ["초졸", "중졸", "고졸", "대졸", "대학", "무학", "초등", "전문대", "석사", "박사"]):
                            label = val_field
                            break
                if not label:
                    for field in ["C3_NM", "C4_NM", "C5_NM"]:
                        val_field = str(row.get(field, "")).strip()
                        if val_field and val_field not in ("계", "전체", "소계"):
                            label = val_field
                            break
                if not label:
                    continue
                dt_val = str(row.get("DT", "0")).strip()
                if dt_val in ("-", "", None):
                    continue
                try:
                    val_float = float(dt_val)
                except (ValueError, TypeError):
                    continue
                if "초졸" in label or "무학" in label or "초등" in label or "중졸" in label:
                    edu_map["중졸이하"] += val_float
                elif "고졸" in label or "고등" in label:
                    edu_map["고졸"] += val_float
                elif "대학" in label or "대졸" in label or "전문대" in label or "석사" in label or "박사" in label:
                    edu_map["대졸이상"] += val_float
                elif val_float > 0:
                    edu_map["중졸이하"] += val_float
        if sum(edu_map.values()) == 0:
            edu_map = {"중졸이하": 25.0, "고졸": 40.0, "대졸이상": 35.0}
        for edu_level, total_val in edu_map.items():
            if total_val > 0:
                labels.append(edu_level)
                values.append(total_val)
    elif axis_key == "income":
        income_map = {
            "50만원미만": 0, "50-100만원": 0, "100-200만원": 0, "200-300만원": 0,
            "300-400만원": 0, "400-500만원": 0, "500-600만원": 0, "600-700만원": 0,
            "700-800만원": 0, "800만원이상": 0,
        }
        for row in kosis_data:
            if isinstance(row, dict):
                label = row.get("C2_NM", "").strip()
                val = row.get("DT", "0")
                try:
                    val_float = float(val)
                    if "50만원미만" in label or ("50" in label and "미만" in label and "100" not in label):
                        income_map["50만원미만"] += val_float
                    elif "50~100" in label or "50-100" in label or ("50만원" in label and "100만원" in label):
                        income_map["50-100만원"] += val_float
                    elif "100~200" in label or "100-200" in label:
                        income_map["100-200만원"] += val_float
                    elif "200~300" in label or "200-300" in label:
                        income_map["200-300만원"] += val_float
                    elif "300~400" in label or "300-400" in label:
                        income_map["300-400만원"] += val_float
                    elif "400~500" in label or "400-500" in label:
                        income_map["400-500만원"] += val_float
                    elif "500~600" in label or "500-600" in label:
                        income_map["500-600만원"] += val_float
                    elif "600~700" in label or "600-700" in label:
                        income_map["600-700만원"] += val_float
                    elif "700~800" in label or "700-800" in label:
                        income_map["700-800만원"] += val_float
                    elif "800만원" in label and ("이상" in label or "초과" in label):
                        income_map["800만원이상"] += val_float
                except Exception:
                    pass
        for income_range, total_val in income_map.items():
            if total_val > 0:
                labels.append(income_range)
                values.append(total_val)
    if labels and values:
        total = sum(values)
        probabilities = [v / total for v in values] if total > 0 else [1.0 / len(values)] * len(values)
        return labels, probabilities
    return [], []


def convert_kosis_to_distribution(kosis_data, axis_key: str):
    """
    KOSIS 데이터를 확률 분포로 변환 (labels, probabilities)
    axis_key: "sigungu", "gender", "age", "econ", "income", "edu"
    로그 기록 후 _convert_kosis_to_distribution_impl 호출.
    """
    try:
        labels, probabilities = _convert_kosis_to_distribution_impl(kosis_data, axis_key)
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stage": f"convert_kosis_to_distribution({axis_key})",
            "axis_key": axis_key,
            "status": "success" if labels else "error",
            "label_count": len(labels),
        }
        if not labels:
            log_entry["error"] = "No valid labels/values extracted"
        if "work_logs" not in st.session_state:
            st.session_state.work_logs = []
        st.session_state.work_logs.append(log_entry)
        return labels, probabilities
    except Exception as e:
        log_entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "stage": f"convert_kosis_to_distribution({axis_key})", "axis_key": axis_key, "status": "exception", "error": str(e)}
        if "work_logs" not in st.session_state:
            st.session_state.work_logs = []
        st.session_state.work_logs.append(log_entry)
        return [], []


# -----------------------------
# 6. Chart Helper Functions (캐싱으로 탭 전환 시 재계산 방지)
# -----------------------------
@st.cache_data(ttl=CACHE_TTL_SECONDS, hash_funcs={pd.DataFrame: _hash_dataframe})
def _build_population_pyramid_figure(df: pd.DataFrame):
    """인구 피라미드용 Plotly Figure 생성 (캐시됨). 데이터 없으면 None."""
    import plotly.graph_objects as go

    if "성별" not in df.columns or "연령" not in df.columns:
        return None
    df_filtered = df[df["연령"] >= 20].copy()
    male_data = df_filtered[df_filtered["성별"].isin(["남자", "남"])].copy()
    female_data = df_filtered[df_filtered["성별"].isin(["여자", "여"])].copy()
    male_age_counts = male_data["연령"].value_counts().sort_index()
    female_age_counts = female_data["연령"].value_counts().sort_index()
    all_ages = sorted(set(male_age_counts.index) | set(female_age_counts.index))
    all_ages = [age for age in all_ages if 20 <= age <= 120]
    male_counts = [-male_age_counts.get(age, 0) for age in all_ages]
    female_counts = [female_age_counts.get(age, 0) for age in all_ages]
    total_population = len(df_filtered)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=all_ages, x=male_counts, name="남자", orientation="h",
        marker=dict(color="rgba(31, 119, 180, 0.8)"),
        hovertemplate="<b>남자</b><br>연령: %{y}세<br>인구수: %{customdata[0]:,}명<br>비율: %{customdata[1]:.2f}%<extra></extra>",
        customdata=[[abs(c), (abs(c) / total_population * 100) if total_population > 0 else 0] for c in male_counts],
    ))
    fig.add_trace(go.Bar(
        y=all_ages, x=female_counts, name="여자", orientation="h",
        marker=dict(color="rgba(255, 127, 14, 0.8)"),
        hovertemplate="<b>여자</b><br>연령: %{y}세<br>인구수: %{customdata[0]:,}명<br>비율: %{customdata[1]:.2f}%<extra></extra>",
        customdata=[[c, (c / total_population * 100) if total_population > 0 else 0] for c in female_counts],
    ))
    max_abs_value = max(max(abs(m) for m in male_counts), max(female_counts)) if male_counts or female_counts else 1
    step = max(1, int(max_abs_value / 10))
    tick_vals = list(range(-int(max_abs_value), int(max_abs_value) + step, step))
    tick_texts = [str(abs(i)) for i in tick_vals]
    fig.update_layout(
        xaxis=dict(title="인구수 (명)", tickvals=tick_vals, ticktext=tick_texts, range=[-max_abs_value * 1.1, max_abs_value * 1.1]),
        yaxis=dict(title="연령 (세)"),
        barmode="overlay",
        height=300,
        margin=dict(l=50, r=50, t=20, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )
    return fig


@st.cache_data(ttl=CACHE_TTL_SECONDS, hash_funcs={pd.DataFrame: _hash_dataframe})
def _build_chart_figures(df: pd.DataFrame, step2_columns_tuple: Tuple[str, ...], step2_only: bool) -> Dict[str, Any]:
    """6축 + 2단계 추가 통계용 Plotly Figure들 생성 (캐시됨). 반환: axes dict + step2 list."""
    import plotly.graph_objects as go

    out = {"region": None, "gender": None, "pyramid": None, "econ": None, "edu": None, "income": None, "step2": []}
    step2_columns = list(step2_columns_tuple)

    if not step2_only:
        if "거주지역" in df.columns:
            region_counts = df["거주지역"].value_counts()
            max_count = region_counts.max()
            colors = [f"rgba(31, 119, 180, {0.3 + 0.7 * (c / max_count)})" for c in region_counts.values]
            fig = go.Figure(data=[go.Bar(x=region_counts.index, y=region_counts.values, marker_color=colors, text=region_counts.values, textposition="auto")])
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="", yaxis_title="인구수")
            out["region"] = fig
        if "성별" in df.columns:
            gender_counts = df["성별"].value_counts()
            max_count = gender_counts.max()
            colors = [f"rgba(255, 127, 14, {0.3 + 0.7 * (c / max_count)})" for c in gender_counts.values]
            fig = go.Figure(data=[go.Bar(x=gender_counts.index, y=gender_counts.values, marker_color=colors, text=gender_counts.values, textposition="auto")])
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="", yaxis_title="인구수")
            out["gender"] = fig
        out["pyramid"] = _build_population_pyramid_figure(df)
        if "경제활동" in df.columns:
            econ_counts = df["경제활동"].value_counts()
            max_count = econ_counts.max()
            colors = [f"rgba(214, 39, 40, {0.3 + 0.7 * (c / max_count)})" for c in econ_counts.values]
            fig = go.Figure(data=[go.Bar(x=econ_counts.index, y=econ_counts.values, marker_color=colors, text=econ_counts.values, textposition="auto")])
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="", yaxis_title="인구수")
            out["econ"] = fig
        if "교육정도" in df.columns:
            edu_counts = df["교육정도"].value_counts()
            max_count = edu_counts.max()
            colors = [f"rgba(148, 103, 189, {0.3 + 0.7 * (c / max_count)})" for c in edu_counts.values]
            fig = go.Figure(data=[go.Bar(x=edu_counts.index, y=edu_counts.values, marker_color=colors, text=edu_counts.values, textposition="auto")])
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="", yaxis_title="인구수")
            out["edu"] = fig
        if "월평균소득" in df.columns:
            income_counts = df["월평균소득"].value_counts()
            income_order = ["50만원미만", "50-100만원", "100-200만원", "200-300만원", "300-400만원", "400-500만원", "500-600만원", "600-700만원", "700-800만원", "800만원이상"]
            income_sorted = income_counts.reindex([i for i in income_order if i in income_counts.index], fill_value=0)
            max_count = income_sorted.max() if len(income_sorted) > 0 else 1
            colors = [f"rgba(140, 86, 75, {0.3 + 0.7 * (c / max_count)})" for c in income_sorted.values]
            fig = go.Figure(data=[go.Bar(x=income_sorted.index, y=income_sorted.values, marker_color=colors, text=income_sorted.values, textposition="auto")])
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="", yaxis_title="인구수")
            out["income"] = fig

    for idx, col_name in enumerate(step2_columns):
        if col_name not in df.columns:
            out["step2"].append((col_name, None))
            continue
        counts = df[col_name].value_counts()
        if counts.empty or len(counts) == 0:
            out["step2"].append((col_name, None))
            continue
        fig = go.Figure(data=[
            go.Bar(x=counts.index.astype(str), y=counts.values, text=counts.values, textposition="auto", marker_color="rgba(44, 160, 44, 0.7)")
        ])
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=5, b=80), xaxis_title="", yaxis_title="인구수", xaxis_tickangle=-45)
        out["step2"].append((col_name, fig))
    return out


def draw_population_pyramid(df: pd.DataFrame):
    """인구 피라미드 렌더링. Figure는 캐시된 _build_population_pyramid_figure 사용."""
    fig = _build_population_pyramid_figure(df)
    if fig is None:
        st.info("성별 또는 연령 데이터가 없습니다.")
        return
    st.plotly_chart(fig, use_container_width=True, key="pyramid_main")


def draw_charts(df: pd.DataFrame, step2_columns: Optional[List[str]] = None, step2_only: bool = False):
    """6축 그래프 + 2단계 추가 통계 분포 렌더링. Figure는 캐시된 _build_chart_figures 사용."""
    step2_cols = step2_columns or []
    figures = _build_chart_figures(df, tuple(step2_cols), step2_only)

    if not step2_only:
        st.markdown("### 인구통계 그래프")
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        row3_col1, row3_col2 = st.columns(2)

        with row1_col1:
            st.markdown("**거주지역**")
            if figures["region"] is not None:
                st.plotly_chart(figures["region"], use_container_width=True, key="chart_region")
            else:
                st.info("거주지역 데이터가 없습니다.")

        with row1_col2:
            st.markdown("**성별**")
            if figures["gender"] is not None:
                st.plotly_chart(figures["gender"], use_container_width=True, key="chart_gender")
            else:
                st.info("성별 데이터가 없습니다.")

        with row2_col1:
            st.markdown("**연령 (인구 피라미드)**")
            if figures["pyramid"] is not None:
                st.plotly_chart(figures["pyramid"], use_container_width=True, key="pyramid_main")
            else:
                st.info("연령 또는 성별 데이터가 없습니다.")

        with row2_col2:
            st.markdown("**경제활동**")
            if figures["econ"] is not None:
                st.plotly_chart(figures["econ"], use_container_width=True, key="chart_econ")
            else:
                st.info("경제활동 데이터가 없습니다.")

        with row3_col1:
            st.markdown("**교육정도**")
            if figures["edu"] is not None:
                st.plotly_chart(figures["edu"], use_container_width=True, key="chart_edu")
            else:
                st.info("교육정도 데이터가 없습니다.")

        with row3_col2:
            st.markdown("**월평균소득**")
            if figures["income"] is not None:
                st.plotly_chart(figures["income"], use_container_width=True, key="chart_income")
            else:
                st.info("월평균소득 데이터가 없습니다.")

    if step2_cols:
        if not step2_only:
            st.markdown("---")
            st.markdown("### 2단계 추가 통계 분포")
        for i in range(0, len(step2_cols), 2):
            cols = st.columns(2)
            for j, col_widget in enumerate(cols):
                idx = i + j
                if idx >= len(step2_cols):
                    break
                col_name, fig = figures["step2"][idx]
                with col_widget:
                    st.markdown(f"**{col_name}**")
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_step2_{idx}")
                    else:
                        st.caption("데이터 없음")


def _get_result_df():
    """step2_df / generated_df / step1_df 중 사용할 DataFrame 반환. DataFrame은 or 연산 불가하므로 명시적 선택."""
    for key in ("step2_df", "generated_df", "step1_df"):
        v = st.session_state.get(key)
        if v is not None and (not hasattr(v, "empty") or not v.empty):
            return v
    return None


@st.fragment
def _fragment_draw_charts(step2_only: bool = False):
    """그래프 탭 전용 fragment — 이 블록 내 상호작용 시 전체가 아닌 이 부분만 갱신."""
    df = _get_result_df()
    if df is None:
        st.info("표시할 데이터가 없습니다.")
        return
    step2_cols = None
    if step2_only:
        step2_cols = [c for c in (st.session_state.get("step2_added_columns") or []) if c in df.columns]
    draw_charts(df, step2_columns=step2_cols, step2_only=step2_only)


def apply_step2_row_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    2단계 통계 대입 후 행 방향 논리 일관성 정리.
    - 경제활동=비경제활동 이면 직장/직업/근로만족도 관련 컬럼은 비움.
    - 만 20세 미만이면 배우자 경제활동 상태를 '무'로 통일.
    """
    out = df.copy()
    econ_col = "경제활동"
    if econ_col not in out.columns:
        return out
    non_econ_mask = out[econ_col].astype(str).str.strip().isin(("비경제활동", "비경제 활동"))
    employment_related_columns = [
        "종사상 지위",
        "직장명(산업 대분류)",
        "하는 일의 종류(직업 종분류)",
        "하는일 만족도",
        "임금/가구소득 만족도",
        "근로시간 만족도",
        "근무환경 만족도",
        "근무 여건 전반적인 만족도",
    ]
    for col in employment_related_columns:
        if col in out.columns:
            out.loc[non_econ_mask, col] = ""
    age_col = "연령"
    spouse_econ_col = "배우자의 경제활동 상태"
    if age_col in out.columns and spouse_econ_col in out.columns:
        import re
        def _age_to_num(a):
            if a is None or (isinstance(a, float) and pd.isna(a)):
                return None
            s = str(a).strip()
            for part in re.findall(r"\d+", s):
                return int(part)
            return None
        ages = out[age_col].map(_age_to_num)
        minor_mask = ages.notna() & (ages < 20)
        out.loc[minor_mask, spouse_econ_col] = "무"
    return out


def calculate_mape(df: pd.DataFrame, margins_axis: Dict[str, Dict]) -> Tuple[float, Dict[str, Dict]]:
    """
    MAPE(Mean Absolute Percentage Error) 계산
    
    Args:
        df: 생성된 가상인구 데이터프레임
        margins_axis: KOSIS 통계 기반 목표 분포
        
    Returns:
        (평균 MAPE, 변수별 상세 오차 리포트)
    """
    axis_name_map = {
        'sigungu': '거주지역',
        'gender': '성별',
        'age': '연령',
        'econ': '경제활동',
        'income': '월평균소득',
        'edu': '교육정도'
    }
    
    mape_list = []
    error_report = {}
    
    for axis_key, axis_data in margins_axis.items():
        col_name = axis_name_map.get(axis_key)
        
        if not col_name or col_name not in df.columns:
            continue
        
        labels = axis_data.get('labels', [])
        target_probs = axis_data.get('p', axis_data.get('probs', []))
        
        if not labels or not target_probs:
            continue
        
        # 실제 분포 계산
        actual_counts = df[col_name].value_counts()
        total = len(df)
        
        axis_errors = []
        axis_details = []
        
        for label, target_prob in zip(labels, target_probs):
            actual_count = actual_counts.get(label, 0)
            actual_prob = actual_count / total if total > 0 else 0
            
            # 절대 오차율 계산 (비율 단위)
            abs_error = abs(actual_prob - target_prob)
            
            # MAPE 계산 (target_prob가 0이 아닌 경우만)
            if target_prob > 0:
                mape = (abs_error / target_prob) * 100
            else:
                mape = abs_error * 100 if abs_error > 0 else 0
            
            axis_errors.append(abs_error)
            axis_details.append({
                'label': str(label),
                'target_pct': target_prob * 100,
                'actual_pct': actual_prob * 100,
                'abs_error_pct': abs_error * 100,
                'mape': mape
            })
        
        # 축별 평균 절대 오차율 (비율 단위)
        axis_mae = np.mean(axis_errors) if axis_errors else 0
        mape_list.append(axis_mae)
        
        error_report[axis_key] = {
            'axis_name': col_name,
            'mae': axis_mae,
            'mae_pct': axis_mae * 100,
            'details': axis_details
        }
    
    # 전체 평균 절대 오차율 (비율 단위)
    avg_mape = np.mean(mape_list) if mape_list else 0
    
    return avg_mape, error_report


def get_step2_target_distributions(
    kosis_client: KosisClient,
    kosis_data: List[Dict[str, Any]],
    is_residence: bool,
    columns: List[str],
    is_children_student: bool = False,
    is_pet: bool = False,
    is_dwelling: bool = False,
    is_parents_survival_cohabitation: bool = False,
    is_parents_expense_provider: bool = False,
    is_housing_satisfaction: bool = False,
    is_spouse_economic: bool = False,
    is_employment_status: bool = False,
    is_industry_major: bool = False,
    is_job_class: bool = False,
    is_work_satisfaction: bool = False,
    is_pet_cost: bool = False,
    is_income_consumption_satisfaction: bool = False,
    is_education_cost: bool = False,
    is_other_region_consumption: bool = False,
    is_preset: bool = False,
    stat_name: str = "",
) -> List[Tuple[str, List[Any], List[float], Any]]:
    """
    2단계 통계용 KOSIS 데이터에서 컬럼별 목표 분포 추출.
    반환: [(컬럼명, labels, target_p, condition), ...]
    target_p는 비율 0~1. condition이 (cond_col, cond_val)이면 해당 조건으로 필터한 인원 기준 검증.
    """
    from collections import defaultdict
    out: List[Tuple[str, List[Any], List[float], Any]] = []
    if not kosis_data or not columns:
        return out
    if is_preset and stat_name:
        return kosis_client.get_preset_target_distributions(stat_name, kosis_data, list(columns))
    if is_children_student and len(columns) >= 2:
        dist_has, dist_count = kosis_client.parse_children_student_kosis(kosis_data)
        if dist_has:
            labels_has = list(kosis_client.CHILDREN_HAS_LABELS)
            p_has = [dist_has.get(l, 0.0) for l in labels_has]
            out.append((columns[0], labels_has, p_has, None))
        if dist_count:
            labels_count = list(kosis_client.CHILDREN_COUNT_LABELS)
            p_count = [dist_count.get(l, 0.0) for l in labels_count]
            out.append((columns[1], labels_count, p_count, (columns[0], "있다")))
    elif is_pet and len(columns) >= 2:
        dist_has, dist_type = kosis_client.parse_pet_kosis(kosis_data)
        if dist_has:
            labels_has = list(kosis_client.PET_HAS_LABELS)
            p_has = [dist_has.get(l, 0.0) for l in labels_has]
            out.append((columns[0], labels_has, p_has, None))
        if dist_type:
            labels_type = list(dist_type.keys())
            p_type = list(dist_type.values())
            out.append((columns[1], labels_type, p_type, (columns[0], "예")))
    elif is_dwelling and len(columns) >= 2:
        dist_dwelling, dist_occupancy = kosis_client.parse_dwelling_kosis(kosis_data)
        if dist_dwelling:
            labels_dw = list(dist_dwelling.keys())
            p_dw = list(dist_dwelling.values())
            out.append((columns[0], labels_dw, p_dw, None))
        if dist_occupancy:
            labels_occ = list(dist_occupancy.keys())
            p_occ = list(dist_occupancy.values())
            out.append((columns[1], labels_occ, p_occ, None))
    elif is_parents_survival_cohabitation and len(columns) >= 2:
        dist_survival, dist_cohabitation = kosis_client.parse_parents_survival_cohabitation_kosis(kosis_data)
        if dist_survival:
            labels_survival = list(dist_survival.keys())
            p_survival = list(dist_survival.values())
            out.append((columns[0], labels_survival, p_survival, None))
        if dist_cohabitation:
            labels_cohabitation = list(dist_cohabitation.keys())
            p_cohabitation = list(dist_cohabitation.values())
            out.append((columns[1], labels_cohabitation, p_cohabitation, None))
    elif is_parents_expense_provider and len(columns) >= 1:
        dist = kosis_client.parse_parents_expense_provider_kosis(kosis_data)
        if dist:
            labels = list(dist.keys())
            p = list(dist.values())
            out.append((columns[0], labels, p, None))
    elif is_housing_satisfaction and len(columns) >= 3:
        d0, d1, d2 = kosis_client.parse_housing_satisfaction_kosis(kosis_data)
        for col_name, dist in zip(columns[:3], [d0, d1, d2]):
            if dist:
                labels = list(dist.keys())
                p = list(dist.values())
                out.append((col_name, labels, p, None))
    elif is_spouse_economic and len(columns) >= 1:
        dist = kosis_client.parse_spouse_economic_kosis(kosis_data)
        if dist:
            labels = list(dist.keys())
            p = list(dist.values())
            out.append((columns[0], labels, p, None))
    elif is_employment_status and len(columns) >= 1:
        dist = kosis_client.parse_employment_status_kosis(kosis_data)
        if dist:
            labels = list(dist.keys())
            p = list(dist.values())
            out.append((columns[0], labels, p, None))
    elif is_industry_major and len(columns) >= 1:
        dist = kosis_client.parse_industry_major_kosis(kosis_data)
        if dist:
            labels = list(dist.keys())
            p = list(dist.values())
            out.append((columns[0], labels, p, None))
    elif is_job_class and len(columns) >= 1:
        dist = kosis_client.parse_job_class_kosis(kosis_data)
        if dist:
            labels = list(dist.keys())
            p = list(dist.values())
            out.append((columns[0], labels, p, None))
    elif is_work_satisfaction and len(columns) >= 5:
        d1, d2, d3, d4, d5 = kosis_client.parse_work_satisfaction_kosis(kosis_data)
        for col_name, dist in zip(columns[:5], [d1, d2, d3, d4, d5]):
            if dist:
                labels = list(dist.keys())
                p = list(dist.values())
                out.append((col_name, labels, p, None))
    elif is_pet_cost and len(columns) >= 1:
        # 반려동물 양육비용: 숫자(원) 컬럼 - 목표 분포 검증 생략
        pass
    elif is_income_consumption_satisfaction and len(columns) >= 3:
        d1, d2, d3 = kosis_client.parse_income_consumption_satisfaction_kosis(kosis_data)
        for col_name, dist in zip(columns[:3], [d1, d2, d3]):
            if dist:
                labels = list(dist.keys())
                p = list(dist.values())
                out.append((col_name, labels, p, None))
    elif is_education_cost and len(columns) >= 2:
        pass
    elif is_other_region_consumption and len(columns) >= 4:
        pass
    elif is_residence and len(columns) >= 3:
        dist_sido, dist_sigungu, dist_intent = kosis_client.parse_residence_duration_kosis(kosis_data)
        for col_name, dist in zip(columns[:3], [dist_sido, dist_sigungu, dist_intent]):
            if dist:
                labels = list(dist.keys())
                p = list(dist.values())
                out.append((col_name, labels, p, None))
    else:
        # 일반 통계: C2_NM 또는 C1_NM 기준 DT 합계로 비율 계산
        agg: Dict[str, float] = defaultdict(float)
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            lab = str(r.get("C2_NM") or r.get("C1_NM") or "").strip()
            if not lab or lab in ("계", "합계", "소계", "Total", "평균"):
                continue
            try:
                v = float(str(r.get("DT", "") or "").replace(",", "").strip() or 0)
            except Exception:
                continue
            agg[lab] += v
        total = sum(agg.values())
        if total > 0 and columns:
            labels = list(agg.keys())
            p = [agg[l] / total for l in labels]
            out.append((columns[0], labels, p, None))
    return out


def build_step2_error_report(
    df: pd.DataFrame,
    step2_validation_info: List[Dict[str, Any]],
    kosis_client: KosisClient,
) -> Dict[str, Dict]:
    """2단계 검증용: KOSIS 목표 분포와 생성 데이터 비교해 1단계와 동일한 error_report 형식으로 반환."""
    error_report: Dict[str, Dict] = {}
    for item in step2_validation_info:
        url = item.get("url", "")
        columns = item.get("columns", [])
        is_residence = item.get("is_residence", False)
        is_children_student = item.get("is_children_student", False)
        is_pet = item.get("is_pet", False)
        is_dwelling = item.get("is_dwelling", False)
        is_parents_survival_cohabitation = item.get("is_parents_survival_cohabitation", False)
        is_parents_expense_provider = item.get("is_parents_expense_provider", False)
        is_housing_satisfaction = item.get("is_housing_satisfaction", False)
        is_spouse_economic = item.get("is_spouse_economic", False)
        is_employment_status = item.get("is_employment_status", False)
        is_industry_major = item.get("is_industry_major", False)
        is_job_class = item.get("is_job_class", False)
        is_work_satisfaction = item.get("is_work_satisfaction", False)
        is_pet_cost = item.get("is_pet_cost", False)
        is_income_consumption_satisfaction = item.get("is_income_consumption_satisfaction", False)
        is_education_cost = item.get("is_education_cost", False)
        is_other_region_consumption = item.get("is_other_region_consumption", False)
        is_preset = item.get("is_preset", False)
        stat_name = item.get("stat_name", "")
        if not url or not columns:
            continue
        try:
            kosis_data = get_cached_kosis_json(url)
        except Exception:
            continue
        if not isinstance(kosis_data, list):
            continue
        targets = get_step2_target_distributions(
            kosis_client, kosis_data, is_residence, columns,
            is_children_student=is_children_student, is_pet=is_pet, is_dwelling=is_dwelling,
            is_parents_survival_cohabitation=is_parents_survival_cohabitation,
            is_parents_expense_provider=is_parents_expense_provider,
            is_housing_satisfaction=is_housing_satisfaction,
            is_spouse_economic=is_spouse_economic,
            is_employment_status=is_employment_status,
            is_industry_major=is_industry_major,
            is_job_class=is_job_class,
            is_work_satisfaction=is_work_satisfaction,
            is_pet_cost=is_pet_cost,
            is_income_consumption_satisfaction=is_income_consumption_satisfaction,
            is_education_cost=is_education_cost,
            is_other_region_consumption=is_other_region_consumption,
            is_preset=is_preset,
            stat_name=stat_name,
        )
        # 학생 및 미취학자녀수: 실제 저장값(유/무, 0·1·2·3) → KOSIS 라벨(있다/없다, 1명/2명/3명이상) 매핑용
        display_to_has = {"유": "있다", "무": "없다"}
        num_to_count = {1: "1명", 2: "2명", 3: "3명이상", "1": "1명", "2": "2명", "3": "3명이상"}
        for target in targets:
            if len(target) >= 4:
                col_name, labels, target_p, condition = target[0], target[1], target[2], target[3]
            else:
                col_name, labels, target_p = target[0], target[1], target[2]
                condition = None
            if col_name not in df.columns or not labels or not target_p:
                continue
            if condition is not None and isinstance(condition, (list, tuple)) and len(condition) >= 2:
                cond_col, cond_val = condition[0], condition[1]
                # G열 검증: 조건값이 '있다'면 실제 데이터는 '유'로 저장됨
                filter_val = "유" if (is_children_student and str(cond_val).strip() == "있다") else str(cond_val).strip()
                sub = df[df[cond_col].astype(str).str.strip() == filter_val]
                if is_children_student:
                    # G열 실제값 1,2,3 → 1명,2명,3명이상으로 매핑해 집계
                    mapped = sub[col_name].astype(object).map(num_to_count)
                    actual_counts = mapped.value_counts().reindex(list(labels), fill_value=0)
                elif is_pet:
                    # L열: 반려동물 종류별 집계, 누락 라벨은 0으로
                    actual_counts = sub[col_name].value_counts().reindex(list(labels), fill_value=0)
                else:
                    actual_counts = sub[col_name].value_counts()
                total = len(sub)
            else:
                if is_children_student:
                    # F열 실제값 유/무 → 있다/없다로 매핑해 집계
                    mapped = df[col_name].astype(str).str.strip().map(display_to_has)
                    actual_counts = mapped.value_counts()
                else:
                    actual_counts = df[col_name].value_counts()
                total = len(df)
            axis_errors = []
            axis_details = []
            for label, target_prob in zip(labels, target_p):
                actual_count = actual_counts.get(label, 0)
                actual_prob = actual_count / total if total > 0 else 0
                abs_error = abs(actual_prob - target_prob)
                mape = (abs_error / target_prob) * 100 if target_prob > 0 else (abs_error * 100 if abs_error > 0 else 0)
                axis_errors.append(abs_error)
                axis_details.append({
                    "label": str(label),
                    "target_pct": target_prob * 100,
                    "actual_pct": actual_prob * 100,
                    "abs_error_pct": abs_error * 100,
                    "mape": mape,
                })
            axis_mae = np.mean(axis_errors) if axis_errors else 0
            key = col_name
            error_report[key] = {
                "axis_name": col_name,
                "mae_pct": axis_mae * 100,
                "details": axis_details,
            }
    return error_report


def render_validation_tab(df: pd.DataFrame, margins_axis: Dict[str, Dict] = None, 
                          error_report: Dict[str, Dict] = None):
    """검증 탭 렌더링 - KOSIS 통계와 부합도 % 검증"""
    st.markdown("### 통계 대비 검증")
    
    if not margins_axis and not error_report:
        st.warning("KOSIS 통계 정보가 없습니다.")
        return
    
    # error_report가 있으면 상세 리포트 사용, 없으면 계산
    if error_report:
        total_errors = []
        
        for axis_key, report_data in error_report.items():
            axis_name = report_data.get('axis_name', axis_key)
            axis_mae_pct = report_data.get('mae_pct', 0)
            details = report_data.get('details', [])
            
            st.markdown(f"#### {axis_name}")
            
            # 상세 비교 테이블 생성
            comparison_data = []
            for detail in details:
                comparison_data.append({
                    "항목": detail.get('label', ''),
                    "KOSIS(%)": f"{detail.get('target_pct', 0):.2f}",
                    "생성(%)": f"{detail.get('actual_pct', 0):.2f}",
                    "절대오차(%)": f"{detail.get('abs_error_pct', 0):.2f}",
                    "부합도(%)": f"{100 - detail.get('abs_error_pct', 0):.2f}",
                    "MAPE(%)": f"{detail.get('mape', 0):.2f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, height=200)
            
            # 평균 오차율 및 부합도
            total_errors.append(axis_mae_pct)
            match_rate = max(0, 100 - axis_mae_pct)
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("평균 오차율", f"{axis_mae_pct:.2f}%")
            with col_stat2:
                st.metric("부합도", f"{match_rate:.2f}%")
            with col_stat3:
                if axis_mae_pct < 1:
                    st.success("매우 우수")
                elif axis_mae_pct < 3:
                    st.success("우수")
                elif axis_mae_pct < 5:
                    st.info("양호")
                else:
                    st.warning("개선 필요")
            
            st.markdown("---")
        
        # 전체 평균 부합도
        if total_errors:
            overall_avg_error = sum(total_errors) / len(total_errors)
            overall_match_rate = max(0, 100 - overall_avg_error)
            
            st.markdown("### 전체 통계 부합도")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("전체 평균 오차율", f"{overall_avg_error:.2f}%")
            with col2:
                st.metric("전체 부합도", f"{overall_match_rate:.2f}%")
            with col3:
                if overall_avg_error < 2:
                    grade = "S (완벽)"
                elif overall_avg_error < 3:
                    grade = "A (우수)"
                elif overall_avg_error < 5:
                    grade = "B (양호)"
                else:
                    grade = "C (보통)"
                st.metric("등급", grade)
    
    elif margins_axis:
        # 기존 방식으로 계산 (하위 호환성)
        total_errors = []
        
        for axis_key, axis_data in margins_axis.items():
            axis_name_map = {
                "sigungu": "거주지역",
                "gender": "성별",
                "age": "연령",
                "econ": "경제활동",
                "income": "월평균소득",
                "edu": "교육정도"
            }
            
            axis_name = axis_name_map.get(axis_key, axis_key)
            
            if axis_name not in df.columns:
                continue
            
            st.markdown(f"#### {axis_name}")
            
            # KOSIS 통계 분포
            kosis_labels = axis_data.get("labels", [])
            kosis_probs = axis_data.get("p", [])
            
            # 생성된 데이터 분포
            generated_counts = df[axis_name].value_counts()
            generated_probs = (generated_counts / len(df)).to_dict()
            
            # 비교 테이블 생성
            comparison_data = []
            axis_total_error = 0
            
            for i, label in enumerate(kosis_labels):
                kosis_pct = kosis_probs[i] * 100 if i < len(kosis_probs) else 0
                generated_pct = generated_probs.get(label, 0) * 100
                error = abs(kosis_pct - generated_pct)
                match_rate = max(0, 100 - error)
                axis_total_error += error
                
                comparison_data.append({
                    "항목": str(label),
                    "KOSIS(%)": f"{kosis_pct:.2f}",
                    "생성(%)": f"{generated_pct:.2f}",
                    "오차(%)": f"{error:.2f}",
                    "부합도(%)": f"{match_rate:.2f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, height=200)
            
            # 평균 오차율 및 부합도
            avg_error = axis_total_error / len(kosis_labels) if len(kosis_labels) > 0 else 0
            avg_match_rate = max(0, 100 - avg_error)
            total_errors.append(avg_error)
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("평균 오차율", f"{avg_error:.2f}%")
            with col_stat2:
                st.metric("부합도", f"{avg_match_rate:.2f}%")
            with col_stat3:
                if avg_error < 1:
                    st.success("매우 우수")
                elif avg_error < 3:
                    st.success("우수")
                elif avg_error < 5:
                    st.info("양호")
                else:
                    st.warning("개선 필요")
            
            st.markdown("---")
        
        # 전체 평균 부합도
        if total_errors:
            overall_avg_error = sum(total_errors) / len(total_errors)
            overall_match_rate = max(0, 100 - overall_avg_error)
            
            st.markdown("### 전체 통계 부합도")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("전체 평균 오차율", f"{overall_avg_error:.2f}%")
            with col2:
                st.metric("전체 부합도", f"{overall_match_rate:.2f}%")
            with col3:
                if overall_avg_error < 2:
                    grade = "S (완벽)"
                elif overall_avg_error < 3:
                    grade = "A (우수)"
                elif overall_avg_error < 5:
                    grade = "B (양호)"
                else:
                    grade = "C (보통)"
                st.metric("등급", grade)


# -----------------------------
# 7. Pages: 생성(좌 옵션 / 우 결과)
# -----------------------------
def page_generate():
    st.title("가상인구 생성")
    
    # ========== 오토세이브 로딩 로직 ==========
    # 세션 상태에 step1_df가 없는데 오토세이브 파일이 존재하면 자동 복구
    if "step1_df" not in st.session_state or st.session_state.get("step1_df") is None:
        if os.path.exists(AUTOSAVE_PATH):
            try:
                step1_df = pd.read_pickle(AUTOSAVE_PATH)
                st.session_state["step1_df"] = step1_df
                st.session_state["step1_completed"] = True
                # 관련 메타데이터도 복구 시도 (파일이 있으면)
                metadata_path = AUTOSAVE_PATH.replace(".pkl", "_metadata.pkl")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                        for key, value in metadata.items():
                            if key not in st.session_state or st.session_state.get(key) is None:
                                st.session_state[key] = value
                        # ✅ margins_axis 명시적 복원 (검증 탭에서 사용)
                        if "margins_axis" in metadata:
                            margins_axis_restored = metadata["margins_axis"]
                            st.session_state["step1_margins_axis"] = margins_axis_restored
                            st.session_state["generated_margins_axis"] = margins_axis_restored
                st.info("이전 생성 결과를 자동으로 불러왔습니다.")
            except Exception as e:
                st.warning(f"오토세이브 파일 로드 실패: {e}")

    # 좌우 2단 레이아웃 (0.35:0.65 비율)
    col_left, col_right = st.columns([0.35, 0.65])

    # ========== 좌측: 생성 옵션 ==========
    with col_left:
        st.subheader("생성 옵션")
        
        # 초기화 버튼
        if st.button("결과 초기화", type="secondary", use_container_width=True):
            # 세션 상태 초기화
            if "step1_df" in st.session_state:
                del st.session_state["step1_df"]
            if "generated_df" in st.session_state:
                del st.session_state["generated_df"]
            if "generated_excel" in st.session_state:
                del st.session_state["generated_excel"]
            st.session_state["step1_completed"] = False
            # 오토세이브 파일 삭제
            if os.path.exists(AUTOSAVE_PATH):
                try:
                    os.remove(AUTOSAVE_PATH)
                except:
                    pass
            metadata_path = AUTOSAVE_PATH.replace(".pkl", "_metadata.pkl")
            if os.path.exists(metadata_path):
                try:
                    os.remove(metadata_path)
                except:
                    pass
            st.success("초기화 완료")
            st.rerun()
        
        st.markdown("---")

        # 지도: 가상인구 DB와 동일 Choropleth, 지도 클릭 시 시도 선택과 양방향 연동
        from pages.virtual_population_db import _build_korea_choropleth_figure
        _gen_sido_default = f"{list(SIDO_CODE.values())[0]} ({list(SIDO_CODE.keys())[0]})"
        # 지도 클릭 선택 처리 → gen_sido 갱신 (양방향 바인딩)
        _map_state = st.session_state.get("gen_sido_map")
        _sel = None
        if _map_state is not None:
            _sel = _map_state.get("selection") if isinstance(_map_state, dict) else getattr(_map_state, "selection", None)
        _pts = []
        if _sel is not None:
            _pts = _sel.get("points", []) if isinstance(_sel, dict) else (getattr(_sel, "points", None) or [])
        if not _pts and isinstance(_sel, dict) and _sel.get("locations"):
            _pts = [{"location": loc} if isinstance(loc, (str, int, float)) else loc for loc in (_sel.get("locations") or [])]
        if _pts:
            _p0 = _pts[0] if isinstance(_pts[0], dict) else (getattr(_pts[0], "__dict__", None) or {})
            _cd = _p0.get("customdata") or _p0.get("customData")
            _loc_id = _p0.get("location")
            _code = None
            if _cd and (isinstance(_cd, (list, tuple)) and len(_cd) > 0):
                _code = str(_cd[0])
            elif isinstance(_cd, (str, int, float)):
                _code = str(_cd)
            if not _code and _loc_id is not None:
                _code = str(_loc_id)
            if not _code:
                _pi = _p0.get("point_index") or _p0.get("pointIndex")
                if _pi is not None:
                    _sidos_ordered = [s["sido_code"] for s in SIDO_MASTER if s["sido_code"] != "00"]
                    if 0 <= _pi < len(_sidos_ordered):
                        _code = str(_sidos_ordered[_pi])
            if _code and _code in SIDO_CODE_TO_NAME:
                st.session_state["gen_sido"] = f"{SIDO_CODE_TO_NAME[_code]} ({_code})"
        _gen_sido_label = st.session_state.get("gen_sido", _gen_sido_default)
        _gen_sido_code = _gen_sido_label.split("(")[-1].rstrip(")").strip()
        _region_stats = get_sido_vdb_stats()
        _gen_fig = _build_korea_choropleth_figure(_gen_sido_code, _region_stats)
        st.plotly_chart(_gen_fig, key="gen_sido_map", use_container_width=True, on_select="rerun", selection_mode="points")

        # 1) 시도 선택
        selected_label = st.selectbox(
            "시도 선택",
            options=[f"{v} ({k})" for k, v in SIDO_CODE.items()],
            key="gen_sido",
        )
        sido_code = selected_label.split("(")[-1].rstrip(")")
        sido_name = SIDO_CODE[sido_code]

        # 2) 생성 인구수
        n = st.number_input(
            "생성 인구수",
            min_value=10,
            max_value=100000,
            value=1000,
            step=100,
        )

        # 3) 6축 가중치 (클릭하여 펼친 뒤 슬라이더로 변경 가능)
        with st.expander("**6축 가중치** (클릭하여 펼치고 조정)", expanded=False):
            w_sigungu = st.slider("시군구", 0.0, 5.0, 1.0, key="w_sigungu")
            w_gender = st.slider("성별", 0.0, 5.0, 1.0, key="w_gender")
            w_age = st.slider("연령", 0.0, 5.0, 1.0, key="w_age")
            w_econ = st.slider("경제활동", 0.0, 5.0, 1.0, key="w_econ")
            w_income = st.slider("소득", 0.0, 5.0, 1.0, key="w_income")
            w_edu = st.slider("교육정도", 0.0, 5.0, 1.0, key="w_edu")
            st.caption("가중치를 높이면 해당 축 분포가 KOSIS 통계에 더 가깝게 반영됩니다.")

        # 4) 통계 목표 활성화 (UI만 유지)
        st.markdown("**통계 목표 활성화**")
        active_stats = [s for s in get_cached_db_list_stats(sido_code) if s["is_active"] == 1]
        if not active_stats:
            st.info("활성화된 통계가 없습니다.")
        else:
            stat_options = {s["id"]: f"[{s['category']}] {s['name']}" for s in active_stats}
            st.multiselect(
                "목표로 할 통계 선택 (선택 사항)",
                options=list(stat_options.keys()),
                format_func=lambda x: stat_options[x],
            )

        # 5) 생성 버튼
        if st.button("가상인구 생성", type="primary", key="btn_gen_pop"):
            import io
            import contextlib
            import traceback
            from datetime import datetime

            log_buf = io.StringIO()
            def log(msg: str):
                log_buf.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
                log_buf.flush()

            try:
                log("버튼 클릭됨 – 생성 시작")
                st.session_state["step1_debug_log"] = log_buf.getvalue()

                with st.spinner("KOSIS 통계 수집 및 인구 생성 중… (1~2분 소요될 수 있음)"):
                    with contextlib.redirect_stdout(log_buf):
                        print("[1단계] 생성 시작 (stdout 캡처 중)")
                        # KOSIS 클라이언트 초기화
                        log("KOSIS 클라이언트 초기화 중...")
                        kosis = KosisClient(use_gemini=False)
                        print("[1단계] KOSIS 클라이언트 준비 완료")

                        # 6축 마진 통계 소스에서 KOSIS 데이터 가져와서 확률 분포로 변환 (통계 목록 1회만 조회)
                        with st.spinner("KOSIS 통계 데이터 가져오는 중..."):
                            axis_keys = ["sigungu", "gender", "age", "econ", "income", "edu"]
                            margins_axis = {}
                            all_stats = get_cached_db_list_stats(sido_code)
                            for axis_key in axis_keys:
                                margin_stat = get_cached_db_axis_margin_stats(sido_code, axis_key)
                                if margin_stat and margin_stat.get("stat_id"):
                                    stat_info = next((s for s in all_stats if s["id"] == margin_stat["stat_id"]), None)
                                    if stat_info:
                                        st.info(f"{axis_key} <- [{stat_info['category']}] {stat_info['name']}")
                                        try:
                                            kosis_data = get_cached_kosis_json(stat_info["url"])
                                            labels, probs = convert_kosis_to_distribution_cached(
                                                json.dumps(kosis_data, sort_keys=True, default=str), axis_key
                                            )
                                            if labels and probs:
                                                margins_axis[axis_key] = {"labels": labels, "p": probs}
                                                st.success(f"{axis_key}: {len(labels)}개 항목 ({sum(probs):.2f} 확률 합)")
                                            else:
                                                st.warning(f"{axis_key}: KOSIS 데이터 변환 실패 (균등 분포 사용)")
                                        except Exception as e:
                                            st.warning(f"{axis_key}: KOSIS 데이터 가져오기 실패: {e}")
                            if len(margins_axis) < 6:
                                st.warning(f"KOSIS 통계 기반: {len(margins_axis)}/6 (나머지는 기본값)")
                            print(f"[1단계] 마진 수집 완료: {list(margins_axis.keys())}")

                        # 1단계: KOSIS 통계 기반 6축 인구 생성
                        print(f"[1단계] generate_base_population 호출 직전 (n={int(n)})")
                        with st.spinner(f"1단계: KOSIS 통계 기반 {int(n)}명 생성 중..."):
                            # sigungu_pool 생성 (margins_axis에서 거주지역 목록 추출)
                            sigungu_pool = []
                            if margins_axis and "sigungu" in margins_axis:
                                sigungu_pool = margins_axis["sigungu"].get("labels", [])
                            import random
                            # 매 실행마다 다른 이름·6축이 나오도록 랜덤 시드 사용 (중복 가상인물 방지)
                            base_seed = random.randint(0, 2**31 - 1)
                            seed = base_seed
                            base_df = cached_generate_base_population(
                                n=int(n),
                                selected_sigungu_json=json.dumps([], sort_keys=True),
                                weights_6axis_json=json.dumps({
                                    'sigungu': w_sigungu, 'gender': w_gender, 'age': w_age,
                                    'econ': w_econ, 'income': w_income, 'edu': w_edu,
                                }, sort_keys=True),
                                sigungu_pool_json=json.dumps(sigungu_pool, sort_keys=True),
                                seed=seed,
                                margins_axis_json=json.dumps(margins_axis if margins_axis else {}, sort_keys=True, default=str),
                                apply_ipf_flag=True,
                            )

                        if base_df is None or base_df.empty:
                            st.error("기본 인구 생성 실패")
                            st.stop()

                        # 오차율 계산 (1회만, Blocking 반복 제거)
                        avg_mae_pct = 0.0
                        if margins_axis:
                            avg_mae, error_report = calculate_mape(base_df, margins_axis)
                            avg_mae_pct = avg_mae * 100

                        if margins_axis and avg_mae_pct < 5.0:
                            st.success(f"KOSIS 통계 기반 {len(base_df)}명 생성 완료 (평균 오차율: {avg_mae_pct:.2f}%)")
                        else:
                            st.success(f"KOSIS 통계 기반 {len(base_df)}명 생성 완료")
                            if margins_axis and avg_mae_pct >= 5.0:
                                st.warning(
                                    f"오차율이 다소 높습니다({avg_mae_pct:.2f}%). "
                                    "더 정교한 결과를 원하시면 **다시 생성** 버튼을 눌러주세요."
                                )

                    # Excel은 다운로드 탭에서 요청 시 캐시된 함수로 생성(지연 변환)
                    st.session_state["generated_excel"] = None
                    st.session_state["generated_df"] = base_df
                    st.session_state["step1_df"] = base_df  # step1_df도 저장 (오토세이브용)
                    st.session_state["step1_completed"] = True
                    st.session_state["generated_n"] = n
                    st.session_state["generated_sido_code"] = sido_code
                    st.session_state["generated_sido_name"] = sido_name
                    st.session_state["generated_weights"] = {
                        "sigungu": w_sigungu,
                        "gender": w_gender,
                        "age": w_age,
                        "econ": w_econ,
                        "income": w_income,
                        "edu": w_edu,
                    }
                    # ✅ margins_axis 저장 (검증 탭에서 사용)
                    st.session_state["step1_margins_axis"] = margins_axis if margins_axis else {}
                    st.session_state["generated_margins_axis"] = margins_axis if margins_axis else {}
                    st.session_state["generated_report"] = f"KOSIS 통계 기반 {len(base_df)}명 생성 완료 ({len(margins_axis)}/6 축 반영)"
                    
                    # ✅ 오토세이브 저장 (1단계 완료 시 즉시 저장)
                    try:
                        base_df.to_pickle(AUTOSAVE_PATH)
                        # 메타데이터도 함께 저장
                        metadata = {
                            "step1_completed": True,
                            "generated_n": n,
                            "generated_sido_code": sido_code,
                            "generated_sido_name": sido_name,
                            "generated_weights": {
                                "sigungu": w_sigungu,
                                "gender": w_gender,
                                "age": w_age,
                                "econ": w_econ,
                                "income": w_income,
                                "edu": w_edu,
                            },
                            "margins_axis": margins_axis if margins_axis else {},  # ✅ margins_axis 저장
                            "generated_report": f"KOSIS 통계 기반 {len(base_df)}명 생성 완료 ({len(margins_axis)}/6 축 반영)",
                        }
                        metadata_path = AUTOSAVE_PATH.replace(".pkl", "_metadata.pkl")
                        with open(metadata_path, 'wb') as f:
                            pickle.dump(metadata, f)
                    except Exception as e:
                        st.warning(f"오토세이브 저장 실패: {e}")

                    # ✅ 1단계 생성 완료 시 작업 기록에 6축 정보 저장
                    from datetime import datetime
                    if "work_logs" not in st.session_state:
                        st.session_state.work_logs = []
                    
                    # 6축 정보 추출
                    axis_info = {}
                    for axis_key in ["sigungu", "gender", "age", "econ", "income", "edu"]:
                        if axis_key in margins_axis:
                            axis_data = margins_axis[axis_key]
                            labels = axis_data.get("labels", [])
                            probs = axis_data.get("p", axis_data.get("probs", []))
                            axis_info[axis_key] = {
                                "labels": labels[:10] if len(labels) > 10 else labels,  # 처음 10개만
                                "label_count": len(labels),
                                "probabilities_sample": probs[:10] if len(probs) > 10 else probs
                            }
                    
                    generation_log = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "stage": "1단계 가상인구 생성 완료",
                        "status": "success",
                        "sido_code": sido_code,
                        "sido_name": sido_name,
                        "population_size": len(base_df),
                        "target_size": n,
                        "weights_6axis": {
                            "sigungu": w_sigungu,
                            "gender": w_gender,
                            "age": w_age,
                            "econ": w_econ,
                            "income": w_income,
                            "edu": w_edu,
                        },
                        "axis_info": axis_info,
                        "axis_count": len(margins_axis)
                    }
                    st.session_state.work_logs.append(generation_log)
                    
                    st.success("1단계 완료: KOSIS 통계 기반 생성.")
                    st.balloons()
                    st.rerun()

            except Exception as e:
                log_buf.write("\n--- 예외 ---\n")
                log_buf.write(traceback.format_exc())
                st.error(f"생성 실패: {e}")
                st.code(traceback.format_exc())
            finally:
                st.session_state["step1_debug_log"] = log_buf.getvalue()

    # ========== 우측: 생성 결과 대시보드 ==========
    with col_right:
        st.subheader("생성 결과 대시보드")
        
        # step2_df 우선 확인 (2단계 완료 시), 없으면 step1_df 또는 generated_df 확인
        df = st.session_state.get("step2_df")
        is_step2 = df is not None
        
        if df is None:
            df = st.session_state.get("step1_df")
        if df is None:
            df = st.session_state.get("generated_df")
        
        if df is None:
            st.info("👈 왼쪽 패널에서 설정을 마치고 [생성] 버튼을 눌러주세요.")
            return
        
        # 메타데이터 가져오기
        n = st.session_state.get("generated_n", len(df))
        sido_name = st.session_state.get("generated_sido_name", "알 수 없음")
        weights = st.session_state.get("generated_weights", {})
        report = st.session_state.get("generated_report", "")
        excel_bytes = st.session_state.get("generated_excel")
        
        # 요약 지표 (한 줄에 표시)
        st.markdown("### 요약 지표")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 인구수", f"{len(df):,}명")
        with col2:
            if "성별" in df.columns:
                gender_counts = df["성별"].value_counts()
                male_count = gender_counts.get("남자", gender_counts.get("남", 0))
                male_ratio = (male_count / len(df) * 100) if len(df) > 0 else 0
                st.metric("남성 비율", f"{male_ratio:.1f}%")
            else:
                st.metric("남성 비율", "N/A")
        with col3:
            if "연령" in df.columns:
                avg_age = df["연령"].mean()
                st.metric("평균 연령", f"{avg_age:.1f}세")
            else:
                st.metric("평균 연령", "N/A")
        with col4:
            st.metric("총 컬럼 수", len(df.columns))
        
        st.markdown("---")
        
        # 데이터 미리보기 (2단계 완료 시 대입된 통계 포함)
        if is_step2:
            st.markdown("### 데이터 미리보기 (2단계 완료: 추가 통계 포함)")
        else:
            st.markdown("### 데이터 미리보기")
        
        # 데이터프레임 컬럼 순서 그대로 표시 (기본 컬럼 + 추가 통계 오른쪽 열)
        df_preview = df
        st.dataframe(df_preview.head(100), height=300, use_container_width=True)
        
        st.markdown("---")
        
        # ✅ 2단계: 다른 통계 대입 버튼 (항상 표시 - 반복 대입 가능)
        st.markdown("### 2단계: 다른 통계 대입")
        col_step2_1, col_step2_2 = st.columns([3, 1])
        with col_step2_1:
            if is_step2:
                st.info("추가 KOSIS 통계를 반복적으로 대입할 수 있습니다. (이미 대입된 통계 포함)")
            else:
                st.info("1단계에서 생성된 가상인구에 추가 KOSIS 통계를 대입할 수 있습니다.")
        with col_step2_2:
            if st.button("다른 통계 대입", type="primary", use_container_width=True):
                st.session_state["show_step2_dialog"] = True
                st.rerun()
        
        # 2단계 다이얼로그 표시
        if st.session_state.get("show_step2_dialog", False):
            with st.expander("2단계: 다른 통계 선택 및 대입", expanded=True):
                sido_code = st.session_state.get("generated_sido_code", "")
                if not sido_code:
                    st.error("먼저 1단계 가상인구를 생성해주세요.")
                else:
                    # 활성화된 통계 목록 가져오기 (6축 마진에 쓰인 통계는 제외 — 이미 1단계에서 반영됨)
                    all_stats = get_cached_db_list_stats(sido_code)
                    active_stats = [s for s in all_stats if s.get("is_active", 0) == 1]
                    six_axis_stat_ids = get_cached_db_six_axis_stat_ids(sido_code)
                    stats_for_step2 = [s for s in active_stats if s["id"] not in six_axis_stat_ids]
                    
                    if not active_stats:
                        st.info("활성화된 통계가 없습니다. 데이터 관리 탭에서 통계를 활성화해주세요.")
                    elif not stats_for_step2:
                        st.info("2단계에 대입할 통계가 없습니다. (6축에 사용 중인 통계를 제외한 나머지 활성 통계가 없습니다.)")
                    else:
                        # 통계 선택: 6축 제외한 활성 통계만 표시, 기본값은 전체 선택
                        stat_options = {s["id"]: f"[{s['category']}] {s['name']}" for s in stats_for_step2}
                        all_stat_ids = list(stat_options.keys())
                        
                        if six_axis_stat_ids:
                            st.caption("6축 마진에 사용 중인 통계는 목록에서 제외됩니다 (이미 1단계에 반영됨).")
                        
                        # 전체 선택 체크박스 (기본 True — 데이터 관리의 모든 통계 반영)
                        select_all = st.checkbox("전체 선택 (6축 제외 나머지 모두 대입)", key="step2_select_all", value=True)
                        
                        # multiselect 기본값: 전체 선택이면 6축 제외 전부
                        default_selection = all_stat_ids if select_all else []
                        
                        selected_stat_ids = st.multiselect(
                            "대입할 통계 선택 (여러 개 선택 가능)",
                            options=all_stat_ids,
                            default=default_selection,
                            format_func=lambda x: stat_options[x],
                            key="step2_stat_selection"
                        )
                        
                        # 전체 선택 체크박스 변경 시 자동 반영
                        if select_all and len(selected_stat_ids) != len(all_stat_ids):
                            selected_stat_ids = all_stat_ids
                            st.rerun()
                        elif not select_all and len(selected_stat_ids) == len(all_stat_ids):
                            pass
                        
                        col_apply, col_cancel = st.columns(2)
                        with col_apply:
                            if st.button("통계 대입 실행", type="primary", use_container_width=True):
                                if not selected_stat_ids:
                                    st.warning("통계를 선택해주세요.")
                                else:
                                    with st.spinner("통계 대입 중..."):
                                        try:
                                            kosis = KosisClient(use_gemini=True)
                                            # step2_df가 있으면 그것을 기반으로, 없으면 현재 df 사용
                                            step2_df = st.session_state.get("step2_df")
                                            base_df_for_step2 = step2_df if step2_df is not None else df
                                            result_df = base_df_for_step2.copy()
                                            
                                            # 통계별 다열 대입 시 기본 컬럼명 사용 (추가 통계는 Excel 오른쪽에 열로 이어 붙임)
                                            residence_duration_columns_by_stat = {}
                                            # 통계 대입 로그 초기화
                                            if "stat_assignment_logs" not in st.session_state:
                                                st.session_state.stat_assignment_logs = []
                                            # 2단계 검증용: 통계별 URL·컬럼·거주기간 여부 저장
                                            step2_validation_info = []
                                            
                                            for stat_id in selected_stat_ids:
                                                stat_info = next((s for s in active_stats if s["id"] == stat_id), None)
                                                if stat_info:
                                                    from datetime import datetime
                                                    log_entry = {
                                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                        "stat_id": stat_id,
                                                        "category": stat_info["category"],
                                                        "stat_name": stat_info["name"],
                                                        "url": stat_info.get("url", ""),
                                                        "status": "processing"
                                                    }
                                                    
                                                    try:
                                                        kosis_data = get_cached_kosis_json(stat_info["url"])
                                                        kosis_data_count = len(kosis_data) if isinstance(kosis_data, list) else 0
                                                        log_entry["kosis_data_count"] = kosis_data_count
                                                        
                                                        # KOSIS 데이터 샘플 저장 (최대 5개)
                                                        if isinstance(kosis_data, list) and len(kosis_data) > 0:
                                                            sample_size = min(5, len(kosis_data))
                                                            log_entry["kosis_data_sample"] = kosis_data[:sample_size]
                                                            if len(kosis_data) > 0:
                                                                first_item = kosis_data[0]
                                                                if isinstance(first_item, dict):
                                                                    log_entry["kosis_data_fields"] = list(first_item.keys())
                                                        else:
                                                            log_entry["kosis_data_sample"] = []
                                                            log_entry["kosis_data_fields"] = []
                                                        
                                                        # 학생 및 미취학자녀수: 2열(F=자녀 유무, G=자녀 수) 전용 로직
                                                        use_children_student = (
                                                            "학생" in (stat_info.get("name") or "")
                                                            and "미취학" in (stat_info.get("name") or "")
                                                        )
                                                        # 반려동물 현황: 2열(K=유무, L=종류) 전용 로직
                                                        use_pet = "반려동물" in (stat_info.get("name") or "") and "현황" in (stat_info.get("name") or "")
                                                        # 거처 종류 및 점유 형태: 2열(거처 종류, 주택 점유 형태) 전용 로직
                                                        use_dwelling = (
                                                            "거처" in (stat_info.get("name") or "")
                                                            and "점유" in (stat_info.get("name") or "")
                                                        )
                                                        # 부모님 생존여부 및 동거여부: 2열(생존여부, 부모님 동거 여부) 전용 로직
                                                        use_parents_survival_cohabitation = (
                                                            "부모님" in (stat_info.get("name") or "")
                                                            and "생존" in (stat_info.get("name") or "")
                                                            and "동거" in (stat_info.get("name") or "")
                                                        )
                                                        # 부모님 생활비 주 제공자: 단일 컬럼 전용 로직
                                                        use_parents_expense_provider = (
                                                            "부모님" in (stat_info.get("name") or "")
                                                            and "생활비" in (stat_info.get("name") or "")
                                                            and "주 제공자" in (stat_info.get("name") or "")
                                                        )
                                                        # 현재 거주주택 만족도: 3열
                                                        use_housing_satisfaction = (
                                                            "거주" in (stat_info.get("name") or "")
                                                            and "만족도" in (stat_info.get("name") or "")
                                                            and ("주택" in (stat_info.get("name") or "") or "주거" in (stat_info.get("name") or ""))
                                                        )
                                                        # 배우자의 경제활동 상태: 1열 유/무
                                                        use_spouse_economic = (
                                                            "배우자" in (stat_info.get("name") or "")
                                                            and "경제활동" in (stat_info.get("name") or "")
                                                        )
                                                        # 종사상 지위: 1열
                                                        use_employment_status = "종사상 지위" in (stat_info.get("name") or "")
                                                        # 직장명(산업 대분류): 1열
                                                        use_industry_major = "직장명" in (stat_info.get("name") or "") and "산업" in (stat_info.get("name") or "") and "대분류" in (stat_info.get("name") or "")
                                                        # 하는 일의 종류(직업 종분류): 1열
                                                        use_job_class = "하는 일" in (stat_info.get("name") or "") and "직업" in (stat_info.get("name") or "")
                                                        # 취업자 근로여건 만족도: 5열
                                                        use_work_satisfaction = "취업자" in (stat_info.get("name") or "") and "근로여건" in (stat_info.get("name") or "") and "만족도" in (stat_info.get("name") or "")
                                                        # 반려동물 양육비용: 1열 숫자(원)
                                                        use_pet_cost = "반려동물" in (stat_info.get("name") or "") and "양육비용" in (stat_info.get("name") or "")
                                                        # 소득 및 소비생활 만족도: 3열 (소득 여부, 소득 만족도, 소비생활만족도)
                                                        use_income_consumption_satisfaction = (
                                                            "소득" in (stat_info.get("name") or "")
                                                            and "소비생활" in (stat_info.get("name") or "")
                                                            and "만족도" in (stat_info.get("name") or "")
                                                        )
                                                        # 월평균 공교육 및 사교육비: 2열 (공교육비, 사교육비) 만원
                                                        use_education_cost = (
                                                            "공교육" in (stat_info.get("name") or "")
                                                            and "사교육" in (stat_info.get("name") or "")
                                                        )
                                                        # 타지역 소비: 4열
                                                        use_other_region_consumption = (
                                                            "타지역" in (stat_info.get("name") or "")
                                                            and "소비" in (stat_info.get("name") or "")
                                                        )
                                                        # 프리셋 통계 21종 (거주지역 대중교통, 의료기관, 의료시설 만족도 등)
                                                        _sn = (stat_info.get("name") or "")
                                                        use_public_transport_satisfaction = "거주지역 대중교통 만족도" in _sn or ("대중교통" in _sn and "만족도" in _sn)
                                                        use_medical_facility_main = "의료기관 주 이용시설" in _sn or ("의료기관" in _sn and "이용시설" in _sn)
                                                        use_medical_satisfaction = "의료시설 만족도" in _sn or ("의료시설" in _sn and "만족도" in _sn)
                                                        use_welfare_satisfaction = (
                                                            "지역의 사회복지 서비스 만족도" in _sn
                                                            or ("임신" in _sn and "복지" in _sn)
                                                            or ("저소득층" in _sn and "복지" in _sn)
                                                            or ("사회복지" in _sn and "만족도" in _sn)
                                                        )
                                                        use_provincial_satisfaction = "도정만족도" in _sn or "도정정책" in _sn or ("도정" in _sn and "만족도" in _sn) or "행정서비스" in _sn
                                                        use_social_communication = "사회적관계별 소통정도" in _sn or ("사회적" in _sn and "소통" in _sn)
                                                        use_trust_people = "일반인에 대한 신뢰" in _sn or ("일반인" in _sn and "신뢰" in _sn)
                                                        use_subjective_class = "주관적 귀속계층" in _sn or ("주관적" in _sn and "귀속" in _sn)
                                                        use_volunteer = ("자원봉사활동" in _sn or "자원봉사 활동" in _sn or "자원봉사" in _sn) and ("여부" in _sn or "여부및" in _sn or "시간" in _sn)
                                                        use_donation = "후원금 금액" in _sn or "후원금" in _sn or ("기부" in _sn and ("여부" in _sn or "금액" in _sn or "방식" in _sn))
                                                        use_regional_belonging = "지역소속감" in _sn or ("지역" in _sn and "소속감" in _sn) or "동네 소속감" in _sn or "시군 소속감" in _sn
                                                        use_safety_eval = "안전환경에 대한 평가" in _sn or ("안전환경" in _sn and "평가" in _sn) or ("안전" in _sn and "환경" in _sn and "평가" in _sn)
                                                        use_crime_fear = "일상생활 범죄피해 두려움" in _sn or ("일상생활" in _sn and "범죄" in _sn and "두려움" in _sn)
                                                        use_daily_fear = "일상생활에서 두려움" in _sn or ("일상생활" in _sn and "두려움" in _sn and "밤" in _sn)
                                                        use_law_abiding = "자신의 평소 준법수준" in _sn or ("준법" in _sn and "수준" in _sn) or "평소 준법" in _sn
                                                        use_environment_feel = "환경체감도" in _sn or ("환경" in _sn and "체감도" in _sn) or "대기환경" in _sn or "수질환경" in _sn
                                                        use_time_pressure = "생활시간 압박" in _sn or ("생활시간" in _sn and "압박" in _sn)
                                                        use_leisure_satisfaction = (
                                                            ("여가활동 만족도" in _sn and "불만족 이유" in _sn)
                                                            or ("여가활동" in _sn and "불만족" in _sn)
                                                            or "여가활동 만족도 및 불만족 이유" in _sn
                                                        )
                                                        use_culture_attendance = ("문화예술행사" in _sn and "관람" in _sn) or "문화예술행사 관람" in _sn or ("문화예술" in _sn and "관람" in _sn)
                                                        use_life_satisfaction = "삶에 대한 만족감과 정서경험" in _sn or ("삶에 대한" in _sn and "만족감" in _sn) or ("만족감" in _sn and "정서" in _sn)
                                                        use_happiness_level = "행복수준" in _sn or ("행복" in _sn and "수준" in _sn)
                                                        two_cols = residence_duration_columns_by_stat.get(
                                                            (str(stat_info.get("category", "")).strip(), str(stat_info.get("name", "")).strip()), []
                                                        )
                                                        two_names = [r3 for _, r3 in two_cols][:2]
                                                        use_residence_duration = (
                                                            "거주기간" in (stat_info.get("name") or "")
                                                            and "정주의사" in (stat_info.get("name") or "")
                                                        )
                                                        three_cols = residence_duration_columns_by_stat.get(
                                                            (str(stat_info.get("category", "")).strip(), str(stat_info.get("name", "")).strip()), []
                                                        )
                                                        three_names = [r3 for _, r3 in three_cols]
                                                        # 학생 및 미취학자녀수: 가상인구 DB 컬럼명 기준
                                                        if use_children_student:
                                                            col_f, col_g = (two_names[0], two_names[1]) if len(two_names) >= 2 else ("학생 및 미취학 자녀 유무", "학생 및 미취학 자녀 수")
                                                            result_df, success = kosis.assign_children_student_columns(
                                                                result_df,
                                                                kosis_data,
                                                                column_names=(col_f, col_g),
                                                                seed=42,
                                                            )
                                                        # 반려동물 현황: K·L 2열 (템플릿에 없으면 기본 컬럼명)
                                                        elif use_pet:
                                                            col_k, col_l = (two_names[0], two_names[1]) if len(two_names) >= 2 else ("반려동물유무", "반려동물종류")
                                                            result_df, success = kosis.assign_pet_columns(
                                                                result_df,
                                                                kosis_data,
                                                                column_names=(col_k, col_l),
                                                                seed=42,
                                                            )
                                                        # 거처 종류 및 점유 형태: 2열 (거처 종류, 주택 점유 형태)
                                                        elif use_dwelling:
                                                            col_dw, col_occ = (two_names[0], two_names[1]) if len(two_names) >= 2 else ("거처 종류", "주택 점유 형태")
                                                            result_df, success = kosis.assign_dwelling_columns(
                                                                result_df,
                                                                kosis_data,
                                                                column_names=(col_dw, col_occ),
                                                                seed=42,
                                                            )
                                                        # 부모님 생존여부 및 동거여부: 2열(부모님 생존 여부, 부모님 동거 여부)
                                                        elif use_parents_survival_cohabitation:
                                                            col_survival, col_cohabitation = (two_names[0], two_names[1]) if len(two_names) >= 2 else ("부모님 생존 여부", "부모님 동거 여부")
                                                            result_df, success = kosis.assign_parents_survival_cohabitation_columns(
                                                                result_df,
                                                                kosis_data,
                                                                column_names=(col_survival, col_cohabitation),
                                                                seed=42,
                                                            )
                                                        # 부모님 생활비 주 제공자: 단일 컬럼 (부모님 생존 여부가 해당없음인 행은 해당없음으로 일관 처리)
                                                        elif use_parents_expense_provider:
                                                            col_expense = "부모님 생활비 주 제공자"
                                                            col_survival_for_expense = "부모님 생존 여부"
                                                            if col_survival_for_expense not in result_df.columns:
                                                                col_survival_for_expense = next((c for c in result_df.columns if "생존" in str(c)), None)
                                                            result_df, success = kosis.assign_parents_expense_provider_column(
                                                                result_df,
                                                                kosis_data,
                                                                column_name=col_expense,
                                                                survival_column=col_survival_for_expense,
                                                                seed=42,
                                                            )
                                                        # 현재 거주주택 만족도: 3열
                                                        elif use_housing_satisfaction:
                                                            col_sat1, col_sat2, col_sat3 = (
                                                                "현재 거주 주택 만족도",
                                                                "현재 상하수도, 도시가스 도로 등 기반시설 만족도",
                                                                "주거지역내 주차장이용 만족도",
                                                            )
                                                            result_df, success = kosis.assign_housing_satisfaction_columns(
                                                                result_df,
                                                                kosis_data,
                                                                column_names=(col_sat1, col_sat2, col_sat3),
                                                                seed=42,
                                                            )
                                                        # 배우자의 경제활동 상태: 1열 유/무
                                                        elif use_spouse_economic:
                                                            col_spouse = "배우자의 경제활동 상태"
                                                            result_df, success = kosis.assign_spouse_economic_column(
                                                                result_df,
                                                                kosis_data,
                                                                column_name=col_spouse,
                                                                seed=42,
                                                            )
                                                        # 종사상 지위: 1열
                                                        elif use_employment_status:
                                                            col_emp = "종사상 지위"
                                                            result_df, success = kosis.assign_employment_status_column(
                                                                result_df,
                                                                kosis_data,
                                                                column_name=col_emp,
                                                                seed=42,
                                                            )
                                                        # 직장명(산업 대분류): 1열
                                                        elif use_industry_major:
                                                            col_ind = "직장명(산업 대분류)"
                                                            result_df, success = kosis.assign_industry_major_column(
                                                                result_df,
                                                                kosis_data,
                                                                column_name=col_ind,
                                                                seed=42,
                                                            )
                                                        # 하는 일의 종류(직업 종분류): 1열
                                                        elif use_job_class:
                                                            col_job = "하는 일의 종류(직업 종분류)"
                                                            result_df, success = kosis.assign_job_class_column(
                                                                result_df,
                                                                kosis_data,
                                                                column_name=col_job,
                                                                seed=42,
                                                            )
                                                        # 취업자 근로여건 만족도: 5열 (가상인구 DB 컬럼명 기준)
                                                        elif use_work_satisfaction:
                                                            col_ws = ("하는일 만족도", "임금/가구소득 만족도", "근로시간 만족도", "근무환경 만족도", "근무 여건 전반적인 만족도")
                                                            result_df, success = kosis.assign_work_satisfaction_columns(
                                                                result_df,
                                                                kosis_data,
                                                                column_names=col_ws,
                                                                seed=42,
                                                            )
                                                        # 반려동물 양육비용: 1열 원 단위
                                                        elif use_pet_cost:
                                                            col_pet_cost = "반려동물 양육비용"
                                                            result_df, success = kosis.assign_pet_cost_column(
                                                                result_df,
                                                                kosis_data,
                                                                column_name=col_pet_cost,
                                                                seed=42,
                                                            )
                                                        elif use_income_consumption_satisfaction:
                                                            col_ics = ("소득 여부", "소득 만족도", "소비생활만족도")
                                                            result_df, success = kosis.assign_income_consumption_satisfaction_columns(
                                                                result_df,
                                                                kosis_data,
                                                                column_names=col_ics,
                                                                seed=42,
                                                            )
                                                        elif use_education_cost:
                                                            col_edu = ("공교육비", "사교육비")
                                                            result_df, success = kosis.assign_education_cost_columns(
                                                                result_df,
                                                                kosis_data,
                                                                column_names=col_edu,
                                                                seed=42,
                                                            )
                                                        elif use_other_region_consumption:
                                                            col_other = (
                                                                "경북 외 소비 경험 여부",
                                                                "경북 외 주요 소비지역",
                                                                "경북 외 주요 소비 상품 및 서비스(1순위)",
                                                                "경북 외 주요 소비 상품 및 서비스(2순위)",
                                                            )
                                                            result_df, success = kosis.assign_other_region_consumption_columns(
                                                                result_df,
                                                                kosis_data,
                                                                column_names=col_other,
                                                                seed=42,
                                                            )
                                                        elif use_public_transport_satisfaction:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("시내버스/마을버스 만족도", "시외/고속버스 만족도", "택시 만족도", "기타(기차,선박)만족도"), seed=42)
                                                        elif use_medical_facility_main:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("의료기관 주 이용시설",), seed=42)
                                                        elif use_medical_satisfaction:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("의료시설 만족도",), seed=42)
                                                        elif use_welfare_satisfaction:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("임신·출산·육아에 대한 복지 만족도", "저소득층 등 취약계층에 대한 복지 만족도"), seed=42)
                                                        elif use_provincial_satisfaction:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("도정정책 만족도", "행정서비스 만족도"), seed=42)
                                                        elif use_social_communication:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("사회적관계별 소통정도",), seed=42)
                                                        elif use_trust_people:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("일반인에 대한 신뢰",), seed=42)
                                                        elif use_subjective_class:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("주관적 귀속계층",), seed=42)
                                                        elif use_volunteer:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("자원봉사 활동 여부", "자원봉사 활동 방식", "지난 1년 동안 자원봉사 활동 시간"), seed=42)
                                                        elif use_donation:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("기부 여부", "기부 방식", "기부금액(만원)"), seed=42)
                                                        elif use_regional_belonging:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("동네 소속감", "시군 소속감", "경상북도 소속감"), seed=42)
                                                        elif use_safety_eval:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=(
                                                                    "(안전환경)어둡고 후미진 곳이 많다", "(안전환경)주변에 쓰레기가 아무렇게 버려져 있고 지저분 하다",
                                                                    "(안전환경)주변에 방치된 차나 빈 건물이 많다", "(안전환경)무리 지어 다니는 불량 청소년이 많다",
                                                                    "(안전환경)기초질서를 지키지 않는 사람이 많다",
                                                                    "(안전환경)큰소리로 다투거나 싸우는 사람들을 자주 볼 수 있다"), seed=42)
                                                        elif use_crime_fear:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("(일상생활 범죄피해 두려움)나자신", "(일상생활 범죄피해 두려움)배우자(애인)", "(일상생활 범죄피해 두려움)자녀", "(일상생활 범죄피해 두려움)부모"), seed=42)
                                                        elif use_daily_fear:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("(일상생활에서 두려움)밤에 혼자 집에 있을 때", "(일상생활에서 두려움)밤에 혼자 지역(동네)의 골목길을 걸을때"), seed=42)
                                                        elif use_law_abiding:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("자신의 평소 준법수준", "평소 법을 지키지 않는 주된 이유"), seed=42)
                                                        elif use_environment_feel:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("대기환경 체감도", "수질환경 체감도", "토양환경 체감도", "소음/진동환경 체감도", "녹지환경 체감도"), seed=42)
                                                        elif use_time_pressure:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("평일 생활시간 압박", "주말 생활시간 압박"), seed=42)
                                                        elif use_leisure_satisfaction:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("문화여가시설 만족도", "전반적인 여가활동 만족도", "여가활동 불만족 이유"), seed=42)
                                                        elif use_culture_attendance:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("문화예술행사 관람 여부", "문화예술행사 관람 분야"), seed=42)
                                                        elif use_life_satisfaction:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=(
                                                                    "삶에 대한 전반적 만족감(10점 만점)", "살고있는 지역의 전반적 만족감(10점 만점)",
                                                                    "어제 행복 정도(10점 만점)", "어제 걱정 정도(10점 만점)"), seed=42)
                                                        elif use_happiness_level:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=(
                                                                    "생활수준(10점 만점)", "건강상태(10점 만점)", "성취도(10점 만점)", "대인관계(10점 만점)",
                                                                    "안전정도(10점 만점)", "지역사회소속감(10점 만점)", "미래안정성(10점 만점)"), seed=42)
                                                        elif use_residence_duration:
                                                            default_names = ("시도 거주기간", "시군구 거주기간", "향후 10년 거주 희망의사")
                                                            if len(three_names) >= 3:
                                                                col1, col2, col3 = three_names[0], three_names[1], three_names[2]
                                                            elif len(three_names) == 2:
                                                                col1, col2, col3 = three_names[0], three_names[1], default_names[2]
                                                            elif len(three_names) == 1:
                                                                col1, col2, col3 = three_names[0], default_names[1], default_names[2]
                                                            else:
                                                                col1, col2, col3 = default_names[0], default_names[1], default_names[2]
                                                            result_df, success = kosis.assign_residence_duration_columns(
                                                                result_df,
                                                                kosis_data,
                                                                column_names=(col1, col2, col3),
                                                                seed=42,
                                                            )
                                                        else:
                                                            result_df, success = kosis.assign_stat_columns_to_population(
                                                                result_df,
                                                                kosis_data,
                                                                category=stat_info["category"],
                                                                stat_name=stat_info["name"],
                                                                url=stat_info["url"]
                                                            )
                                                        
                                                        if success:
                                                            log_entry["status"] = "success"
                                                            log_entry["message"] = "통계 대입 완료"
                                                            st.success(f"[{stat_info['category']}] {stat_info['name']} 대입 완료")
                                                            # 검증 탭에서 KOSIS 대비 검증용 정보 저장
                                                            if use_children_student:
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": [col_f, col_g],
                                                                    "is_residence": False,
                                                                    "is_children_student": True,
                                                                })
                                                            elif use_pet:
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": [col_k, col_l],
                                                                    "is_residence": False,
                                                                    "is_pet": True,
                                                                })
                                                            elif use_dwelling:
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": [col_dw, col_occ],
                                                                    "is_residence": False,
                                                                    "is_dwelling": True,
                                                                })
                                                            elif use_parents_survival_cohabitation:
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": [col_survival, col_cohabitation],
                                                                    "is_residence": False,
                                                                    "is_parents_survival_cohabitation": True,
                                                                })
                                                            elif use_parents_expense_provider:
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": [col_expense],
                                                                    "is_residence": False,
                                                                    "is_parents_expense_provider": True,
                                                                })
                                                            elif use_housing_satisfaction:
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": [col_sat1, col_sat2, col_sat3],
                                                                    "is_residence": False,
                                                                    "is_housing_satisfaction": True,
                                                                })
                                                            elif use_spouse_economic:
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": [col_spouse],
                                                                    "is_residence": False,
                                                                    "is_spouse_economic": True,
                                                                })
                                                            elif use_employment_status:
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": [col_emp],
                                                                    "is_residence": False,
                                                                    "is_employment_status": True,
                                                                })
                                                            elif use_industry_major:
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": [col_ind],
                                                                    "is_residence": False,
                                                                    "is_industry_major": True,
                                                                })
                                                            elif use_job_class:
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": [col_job],
                                                                    "is_residence": False,
                                                                    "is_job_class": True,
                                                                })
                                                            elif use_work_satisfaction:
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": list(col_ws),
                                                                    "is_residence": False,
                                                                    "is_work_satisfaction": True,
                                                                })
                                                            elif use_pet_cost:
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": [col_pet_cost],
                                                                    "is_residence": False,
                                                                    "is_pet_cost": True,
                                                                })
                                                            elif use_income_consumption_satisfaction:
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": list(col_ics),
                                                                    "is_residence": False,
                                                                    "is_income_consumption_satisfaction": True,
                                                                })
                                                            elif use_education_cost:
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": list(col_edu),
                                                                    "is_residence": False,
                                                                    "is_education_cost": True,
                                                                })
                                                            elif use_other_region_consumption:
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": list(col_other),
                                                                    "is_residence": False,
                                                                    "is_other_region_consumption": True,
                                                                })
                                                            elif use_public_transport_satisfaction:
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": ["시내버스/마을버스 만족도", "시외/고속버스 만족도", "택시 만족도", "기타(기차,선박)만족도"],
                                                                    "is_residence": False,
                                                                    "is_preset": True,
                                                                })
                                                            elif use_medical_facility_main:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["의료기관 주 이용시설"], "is_residence": False, "is_preset": True})
                                                            elif use_medical_satisfaction:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["의료시설 만족도"], "is_residence": False, "is_preset": True})
                                                            elif use_welfare_satisfaction:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["임신·출산·육아에 대한 복지 만족도", "저소득층 등 취약계층에 대한 복지 만족도"], "is_residence": False, "is_preset": True})
                                                            elif use_provincial_satisfaction:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["도정정책 만족도", "행정서비스 만족도"], "is_residence": False, "is_preset": True})
                                                            elif use_social_communication:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["사회적관계별 소통정도"], "is_residence": False, "is_preset": True})
                                                            elif use_trust_people:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["일반인에 대한 신뢰"], "is_residence": False, "is_preset": True})
                                                            elif use_subjective_class:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["주관적 귀속계층"], "is_residence": False, "is_preset": True})
                                                            elif use_volunteer:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["자원봉사 활동 여부", "자원봉사 활동 방식", "지난 1년 동안 자원봉사 활동 시간"], "is_residence": False, "is_preset": True})
                                                            elif use_donation:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["기부 여부", "기부 방식", "기부금액(만원)"], "is_residence": False, "is_preset": True})
                                                            elif use_regional_belonging:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["동네 소속감", "시군 소속감", "경상북도 소속감"], "is_residence": False, "is_preset": True})
                                                            elif use_safety_eval:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["(안전환경)어둡고 후미진 곳이 많다", "(안전환경)주변에 쓰레기가 아무렇게 버려져 있고 지저분 하다", "(안전환경)주변에 방치된 차나 빈 건물이 많다", "(안전환경)무리 지어 다니는 불량 청소년이 많다", "(안전환경)기초질서를 지키지 않는 사람이 많다", "(안전환경)큰소리로 다투거나 싸우는 사람들을 자주 볼 수 있다"], "is_residence": False, "is_preset": True})
                                                            elif use_crime_fear:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["(일상생활 범죄피해 두려움)나자신", "(일상생활 범죄피해 두려움)배우자(애인)", "(일상생활 범죄피해 두려움)자녀", "(일상생활 범죄피해 두려움)부모"], "is_residence": False, "is_preset": True})
                                                            elif use_daily_fear:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["(일상생활에서 두려움)밤에 혼자 집에 있을 때", "(일상생활에서 두려움)밤에 혼자 지역(동네)의 골목길을 걸을때"], "is_residence": False, "is_preset": True})
                                                            elif use_law_abiding:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["자신의 평소 준법수준", "평소 법을 지키지 않는 주된 이유"], "is_residence": False, "is_preset": True})
                                                            elif use_environment_feel:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["대기환경 체감도", "수질환경 체감도", "토양환경 체감도", "소음/진동환경 체감도", "녹지환경 체감도"], "is_residence": False, "is_preset": True})
                                                            elif use_time_pressure:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["평일 생활시간 압박", "주말 생활시간 압박"], "is_residence": False, "is_preset": True})
                                                            elif use_leisure_satisfaction:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["문화여가시설 만족도", "전반적인 여가활동 만족도", "여가활동 불만족 이유"], "is_residence": False, "is_preset": True})
                                                            elif use_culture_attendance:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["문화예술행사 관람 여부", "문화예술행사 관람 분야"], "is_residence": False, "is_preset": True})
                                                            elif use_life_satisfaction:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["삶에 대한 전반적 만족감(10점 만점)", "살고있는 지역의 전반적 만족감(10점 만점)", "어제 행복 정도(10점 만점)", "어제 걱정 정도(10점 만점)"], "is_residence": False, "is_preset": True})
                                                            elif use_happiness_level:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["생활수준(10점 만점)", "건강상태(10점 만점)", "성취도(10점 만점)", "대인관계(10점 만점)", "안전정도(10점 만점)", "지역사회소속감(10점 만점)", "미래안정성(10점 만점)"], "is_residence": False, "is_preset": True})
                                                            elif use_residence_duration:
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": [col1, col2, col3],
                                                                    "is_residence": True,
                                                                })
                                                            else:
                                                                _slug_col = f"kosis_{kosis._extract_category_code(stat_info.get('category', ''))}__{kosis._slug(stat_info.get('name', ''))}"
                                                                step2_validation_info.append({
                                                                    "stat_name": stat_info.get("name", ""),
                                                                    "url": stat_info.get("url", ""),
                                                                    "columns": [_slug_col],
                                                                    "is_residence": False,
                                                                })
                                                        else:
                                                            log_entry["status"] = "warning"
                                                            log_entry["message"] = "대입 실패 (기본값 사용)"
                                                            st.warning(f"[{stat_info['category']}] {stat_info['name']} 대입 실패 (기본값 사용)")
                                                    except Exception as e:
                                                        log_entry["status"] = "error"
                                                        log_entry["message"] = str(e)
                                                        log_entry["error"] = str(e)
                                                        import traceback
                                                        log_entry["traceback"] = traceback.format_exc()
                                                        st.error(f"[{stat_info['category']}] {stat_info['name']} 대입 중 에러: {e}")
                                                    
                                                    st.session_state.stat_assignment_logs.append(log_entry)
                                            
                                            # 행 방향 논리 일관성 정리 (비경제활동 → 직장/직업/근로만족도 비움, 미성년 → 배우자 경제활동 무)
                                            result_df = apply_step2_row_consistency(result_df)
                                            
                                            # 개연성 적용 (캐시된 함수로 1회 계산 결과 재사용)
                                            result_df = _apply_step2_logical_consistency_cached(result_df)
                                            # 결과 저장 (내부 컬럼명 유지하여 검증 등에서 사용)
                                            st.session_state["step2_df"] = result_df
                                            st.session_state["generated_df"] = result_df
                                            st.session_state["show_step2_dialog"] = False
                                            
                                            # 추가된 통계 정보 저장
                                            step1_base_columns = ['식별NO', '가상이름', '거주지역', '성별', '연령', '경제활동', '교육정도', '월평균소득']
                                            added_columns = [col for col in result_df.columns if col not in step1_base_columns]
                                            st.session_state["step2_added_columns"] = added_columns
                                            st.session_state["step2_validation_info"] = step2_validation_info
                                            
                                            # ✅ 2단계 대입 완료 시 작업 기록에 추가된 통계 정보 저장
                                            from datetime import datetime
                                            if "work_logs" not in st.session_state:
                                                st.session_state.work_logs = []
                                            
                                            # 대입된 통계 정보 추출
                                            assigned_stats = []
                                            for stat_id in selected_stat_ids:
                                                stat_info = next((s for s in active_stats if s["id"] == stat_id), None)
                                                if stat_info:
                                                    assigned_stats.append({
                                                        "id": stat_info["id"],
                                                        "category": stat_info["category"],
                                                        "name": stat_info["name"],
                                                        "url": stat_info.get("url", "")
                                                    })
                                            
                                            step2_log = {
                                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                "stage": "2단계 통계 대입 완료",
                                                "status": "success",
                                                "sido_code": sido_code,
                                                "sido_name": st.session_state.get("generated_sido_name", ""),
                                                "population_size": len(result_df),
                                                "assigned_statistics": assigned_stats,
                                                "added_columns": added_columns,
                                                "added_column_count": len(added_columns)
                                            }
                                            st.session_state.work_logs.append(step2_log)
                                            result_df_export = _apply_step2_column_rename(result_df.copy())
                                            added_columns_export = [c for c in result_df_export.columns if c not in step1_base_columns]
                                            save_step2_record(result_df_export, sido_code, st.session_state.get("generated_sido_name", ""), added_columns_export)
                                            # Excel 생성: 출력용 컬럼명으로 저장
                                            try:
                                                import io
                                                out_buffer = io.BytesIO()
                                                result_df_export.to_excel(out_buffer, index=False, engine="openpyxl")
                                                out_buffer.seek(0)
                                                st.session_state["generated_excel"] = out_buffer.getvalue()
                                            except Exception as e:
                                                st.warning(f"Excel 저장 실패: {e}")
                                                import traceback
                                                st.code(traceback.format_exc())
                                            st.success("2단계 완료: 통계 대입 완료.")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"통계 대입 실패: {e}")
                                            import traceback
                                            st.code(traceback.format_exc())
                        
                        with col_cancel:
                            if st.button("취소", use_container_width=True):
                                st.session_state["show_step2_dialog"] = False
                                st.rerun()
        
        st.markdown("---")
        # 결과 뷰용 세션 저장 (fragment 격리 실행 시에도 동일 데이터 사용)
        st.session_state["_rv_df"] = df
        st.session_state["_rv_is_step2"] = is_step2
        st.session_state["_rv_sido_name"] = sido_name
        st.session_state["_rv_n"] = n
        st.session_state["_rv_weights"] = weights or {}
        st.session_state["_rv_report"] = report or ""
        _fragment_result_tabs()


@st.fragment
def _fragment_result_tabs():
    """요약·그래프·검증·데이터·다운로드 탭을 fragment로 렌더. 탭 내 상호작용 시 이 블록만 갱신."""
    df = st.session_state.get("_rv_df")
    is_step2 = st.session_state.get("_rv_is_step2", False)
    sido_name = st.session_state.get("_rv_sido_name", "알 수 없음")
    n = st.session_state.get("_rv_n", 0)
    weights = st.session_state.get("_rv_weights") or {}
    report = st.session_state.get("_rv_report", "")
    if df is None or df.empty:
        st.warning("표시할 데이터가 없습니다.")
        return
    tabs = st.tabs(["요약", "그래프", "검증", "데이터", "다운로드"])
    # [탭 0] 요약
    with tabs[0]:
        if is_step2:
            # 2단계: 추가 통계 데이터만 요약 (1단계 항목 제거)
            st.markdown("### 2단계 추가 통계 요약")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("지역", sido_name)
            with col2:
                st.metric("총 인구수", f"{len(df):,}명")
            with col3:
                st.metric("추가 통계 컬럼 수", len(st.session_state.get("step2_added_columns") or []))
            added_cols = st.session_state.get("step2_added_columns") or []
            added_cols = [c for c in added_cols if c in df.columns]
            if added_cols:
                st.markdown("---")
                for col in added_cols:
                    vc = df[col].value_counts()
                    total = len(df)
                    top3 = vc.head(3)
                    summary_parts = [f"{k}: {v}명({v/total*100:.1f}%)" for k, v in top3.items() if str(k).strip()]
                    if summary_parts:
                        st.markdown(f"**{col}**")
                        st.write(", ".join(summary_parts))
            else:
                st.caption("추가된 통계 컬럼이 없습니다.")
        else:
            # 1단계: 기존 요약
            st.metric("지역", sido_name)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("생성 인구수", f"{n:,}명")
            with col2:
                st.metric("실제 생성", f"{len(df):,}명")
            if "성별" in df.columns:
                gender_counts = df["성별"].value_counts()
                total = len(df)
                gender_ratio = {k: (v/total*100) for k, v in gender_counts.items()}
                col3, col4 = st.columns(2)
                with col3:
                    st.metric("남자 비율", f"{gender_ratio.get('남자', gender_ratio.get('남', 0)):.1f}%")
                with col4:
                    st.metric("여자 비율", f"{gender_ratio.get('여자', gender_ratio.get('여', 0)):.1f}%")
            if "연령" in df.columns:
                age_counts = df["연령"].value_counts()
                total = len(df)
                top_ages = age_counts.head(2)
                col5, col6 = st.columns(2)
                for idx, (age, count) in enumerate(top_ages.items()):
                    with (col5 if idx == 0 else col6):
                        st.metric(f"주요 연령대 {idx+1}위", f"{age}세", f"{(count/total*100):.1f}%")
            st.markdown("---")
            st.markdown("**6축 가중치**")
            for k, v in weights.items():
                st.write(f"- {k}: {v}")
            if report:
                st.markdown("**생성 결과**")
                st.info(report)
    # [탭 1] 그래프 — fragment로 분리해 그래프 탭 내 상호작용 시 부분 갱신
    with tabs[1]:
        if is_step2:
            step2_cols = [c for c in (st.session_state.get("step2_added_columns") or []) if c in df.columns]
            if step2_cols:
                st.markdown("### 2단계 추가 통계 분포")
                _fragment_draw_charts(step2_only=True)
            else:
                st.info("추가된 통계 컬럼이 없습니다.")
        else:
            _fragment_draw_charts(step2_only=False)
    # [탭 2] 검증 — 2단계 완료 시 KOSIS 대비 검증(1단계와 동일 방식), 아니면 1단계 6축 검증
    with tabs[2]:
        if is_step2:
            step2_val_info = st.session_state.get("step2_validation_info") or []
            if step2_val_info:
                st.markdown("### 2단계 추가 통계 · KOSIS 대비 검증")
                kosis_for_val = KosisClient(use_gemini=False)
                step2_error_report = build_step2_error_report(df, step2_val_info, kosis_for_val)
                if step2_error_report:
                    render_validation_tab(df, margins_axis=None, error_report=step2_error_report)
                else:
                    st.warning("KOSIS 데이터를 불러와 검증 정보를 만들 수 없습니다. URL 또는 통계 구성을 확인해주세요.")
            else:
                st.info("2단계에서 대입된 통계가 없거나 검증 정보가 없습니다.")
        else:
            margins_axis = st.session_state.get("step1_margins_axis") or st.session_state.get("generated_margins_axis")
            error_report = st.session_state.get("step1_error_report") or st.session_state.get("generated_error_report")
            if not error_report and margins_axis:
                avg_mae, error_report = calculate_mape(df, margins_axis)
                st.session_state["step1_error_report"] = error_report
                st.session_state["step1_final_mae"] = avg_mae
            render_validation_tab(df, margins_axis, error_report)
    # [탭 3] 데이터 (2단계 완료 시 추가된 통계 포함)
    with tabs[3]:
        if is_step2:
            st.markdown("### 생성된 데이터 미리보기 (2단계 완료: 추가 통계 포함)")
        else:
            st.markdown("### 생성된 데이터 미리보기")
        df_preview = _apply_step2_column_rename(df.copy()) if is_step2 else df
        st.dataframe(df_preview.head(100), height=400, use_container_width=True)
    # [탭 4] 다운로드 (탭 진입 시에만 캐시된 함수로 Excel 변환)
    with tabs[4]:
        st.markdown("### 다운로드")
        df_export = _apply_step2_column_rename(df.copy()) if is_step2 else df
        try:
            final_excel_bytes = _build_excel_bytes_for_download(df_export, is_step2)
        except Exception as e:
            st.warning(f"Excel 생성 실패: {e}")
            final_excel_bytes = None
        if final_excel_bytes:
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                file_suffix = "_step2_final" if is_step2 else "_step1"
                st.download_button(
                    "Excel 다운로드",
                    data=final_excel_bytes,
                    file_name=f"{sido_name}_synthetic_population{file_suffix}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with col_dl2:
                csv = df_export.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "CSV 다운로드",
                    data=csv,
                    file_name=f"{sido_name}_synthetic_population{file_suffix}.csv",
                    mime="text/csv",
                )
        else:
            st.info("Excel 파일이 생성되지 않았습니다.")


def page_step2_results():
    """2차 대입 결과: 날짜/시간별 기록 조회, 데이터 보기, 삭제(서버 파일까지 삭제). 여러 건 선택 후 일괄 삭제 가능. 페이지네이션(10건/페이지)."""
    from utils.step2_records import list_step2_records, delete_step2_record
    st.header("2차 대입 결과")
    records = list_step2_records()
    if not records:
        st.info("아직 2차 대입 결과가 없습니다. 가상인구 생성 후 2단계에서 통계를 대입하면 여기에 저장됩니다.")
        return
    # 페이지네이션: 한 페이지당 10건
    PER_PAGE = 10
    total_pages = max(1, (len(records) + PER_PAGE - 1) // PER_PAGE)
    if "step2_page" not in st.session_state:
        st.session_state["step2_page"] = 0
    current_page = min(max(0, st.session_state["step2_page"]), total_pages - 1)
    st.session_state["step2_page"] = current_page
    start = current_page * PER_PAGE
    end = min(start + PER_PAGE, len(records))
    page_records = records[start:end]

    st.caption(f"총 {len(records)}건 (날짜·시간순). 삭제 시 서버의 Excel·메타 파일이 함께 삭제됩니다.")
    st.markdown("**삭제할 항목을 체크한 뒤 아래 [선택한 항목 삭제] 버튼을 누르면 한 번에 삭제됩니다.**")
    for i, r in enumerate(page_records):
        idx = start + i
        ts = r.get("timestamp", "")
        sido_name = r.get("sido_name", "")
        rows = r.get("rows", 0)
        excel_path = r.get("excel_path", "")
        added = r.get("added_columns", [])
        row_label = f"{ts} | {sido_name} | {rows}명 | 추가 컬럼 {len(added)}개"
        with st.expander(row_label):
            st.checkbox("이 항목 삭제에 포함", key=f"step2_del_cb_{idx}")
            st.caption(f"추가된 컬럼: {', '.join(added[:8])}{' ...' if len(added) > 8 else ''}")
            # 지연 로딩: 버튼 클릭 시에만 엑셀 로드 (페이지 마비 방지)
            preview_key = f"step2_show_preview_{idx}"
            df_cache_key = f"step2_preview_df_{idx}"
            if st.button("데이터 미리보기", key=f"step2_preview_btn_{idx}", type="secondary"):
                st.session_state[preview_key] = True
            if st.session_state.get(preview_key):
                if df_cache_key not in st.session_state:
                    try:
                        st.session_state[df_cache_key] = pd.read_excel(excel_path, engine="openpyxl")
                    except Exception as e:
                        st.warning(f"데이터 로드 실패: {e}")
                if df_cache_key in st.session_state:
                    st.dataframe(st.session_state[df_cache_key].head(100), use_container_width=True, height=300)
                col_dl, col_del = st.columns([1, 1])
                with col_dl:
                    with open(excel_path, "rb") as f:
                        st.download_button("Excel 다운로드", data=f.read(), file_name=os.path.basename(excel_path), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_{ts}_{r.get('sido_code','')}_{idx}")
                with col_del:
                    if st.button("이 항목만 삭제", key=f"del_step2_{ts}_{r.get('sido_code','')}_{idx}", type="secondary"):
                        if delete_step2_record(excel_path):
                            st.success("해당 2차 대입 결과와 서버 파일을 삭제했습니다.")
                        else:
                            st.error("삭제에 실패했습니다.")
                        st.rerun()

    # 페이지 네비게이션
    if total_pages > 1:
        st.markdown("---")
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("← 이전", key="step2_prev_page", disabled=(current_page == 0)):
                st.session_state["step2_page"] = current_page - 1
                st.rerun()
        with col_info:
            st.caption(f"**{start + 1}–{end}** / {len(records)}건 (페이지 {current_page + 1}/{total_pages})")
        with col_next:
            if st.button("다음 →", key="step2_next_page", disabled=(current_page >= total_pages - 1)):
                st.session_state["step2_page"] = current_page + 1
                st.rerun()

    # 선택한 항목 일괄 삭제
    selected_paths = []
    for idx in range(len(records)):
        if st.session_state.get(f"step2_del_cb_{idx}", False):
            path = records[idx].get("excel_path", "")
            if path and path not in selected_paths:
                selected_paths.append(path)
    if selected_paths:
        if st.button("선택한 항목 삭제", type="primary", key="step2_bulk_delete"):
            success = 0
            fail = 0
            for path in selected_paths:
                if delete_step2_record(path):
                    success += 1
                else:
                    fail += 1
            if success:
                st.success(f"선택한 {success}건을 삭제했습니다." + (f" ({fail}건 실패)" if fail else ""))
            if fail:
                st.error(f"{fail}건 삭제에 실패했습니다.")
            st.rerun()
    else:
        st.caption("삭제할 항목을 위에서 체크하면 [선택한 항목 삭제] 버튼이 나타납니다.")




@st.fragment
def page_stat_assignment_log():
    """통계 대입 로그 페이지 - 2단계 통계 대입 시 발생하는 상세 로그 (fragment: 로그 영역만 갱신)"""
    st.header("통계 대입 로그")
    
    # 세션에서 통계 대입 로그 가져오기
    if "stat_assignment_logs" not in st.session_state:
        st.session_state.stat_assignment_logs = []
    
    if len(st.session_state.stat_assignment_logs) == 0:
        st.info("아직 통계 대입 로그가 없습니다. 2단계 통계 대입을 실행하면 로그가 기록됩니다.")
        return
    
    st.success(f"총 {len(st.session_state.stat_assignment_logs)}개의 통계 대입 로그")
    
    # 통계별로 그룹화
    stats_summary = {}
    for log in st.session_state.stat_assignment_logs:
        stat_key = f"{log.get('category', 'N/A')} - {log.get('stat_name', 'N/A')}"
        if stat_key not in stats_summary:
            stats_summary[stat_key] = {
                "total": 0,
                "success": 0,
                "warning": 0,
                "error": 0,
                "logs": []
            }
        stats_summary[stat_key]["total"] += 1
        stats_summary[stat_key]["logs"].append(log)
        status = log.get("status", "unknown")
        if status == "success":
            stats_summary[stat_key]["success"] += 1
        elif status == "warning":
            stats_summary[stat_key]["warning"] += 1
        elif status == "error":
            stats_summary[stat_key]["error"] += 1
    
    # 요약 표시
    st.markdown("### 통계별 요약")
    summary_data = []
    for stat_key, summary in stats_summary.items():
        summary_data.append({
            "통계명": stat_key,
            "총 시도": summary["total"],
            "성공": summary["success"],
            "경고": summary["warning"],
            "에러": summary["error"],
            "성공률": f"{(summary['success'] / summary['total'] * 100):.1f}%" if summary["total"] > 0 else "0%"
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 최신순으로 상세 로그 표시
    st.markdown("### 상세 로그 (최신순)")
    for idx, log in enumerate(reversed(st.session_state.stat_assignment_logs[-50:])):
        timestamp = log.get("timestamp", "N/A")
        category = log.get("category", "N/A")
        stat_name = log.get("stat_name", "N/A")
        status = log.get("status", "unknown")
        
        # 상태별 접두어 및 색상
        if status == "success":
            prefix = "[성공]"
            color = "green"
        elif status == "warning":
            prefix = "[경고]"
            color = "orange"
        elif status == "error":
            prefix = "[에러]"
            color = "red"
        else:
            prefix = "[대기]"
            color = "gray"
        
        with st.expander(f"{prefix} {timestamp} - [{category}] {stat_name}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**기본 정보**")
                st.write(f"- 카테고리: {category}")
                st.write(f"- 통계명: {stat_name}")
                st.write(f"- 상태: {status}")
                st.write(f"- 시각: {timestamp}")
            
            with col2:
                st.markdown("**데이터 정보**")
                kosis_data_count = log.get("kosis_data_count", 0)
                st.write(f"- KOSIS 데이터 건수: {kosis_data_count:,}건")
                url = log.get("url", "")
                if url:
                    st.write(f"- URL: {url[:100]}..." if len(url) > 100 else f"- URL: {url}")
                
                # KOSIS 데이터 필드 정보
                kosis_data_fields = log.get("kosis_data_fields", [])
                if kosis_data_fields:
                    st.write(f"- 데이터 필드: {', '.join(kosis_data_fields[:10])}" + ("..." if len(kosis_data_fields) > 10 else ""))
            
            # KOSIS 데이터 샘플 표시
            kosis_data_sample = log.get("kosis_data_sample", [])
            if kosis_data_sample:
                st.markdown("---")
                st.markdown("**가져온 KOSIS 데이터 샘플**")
                st.caption(f"전체 {kosis_data_count:,}건 중 처음 {len(kosis_data_sample)}건 표시")
                
                # 샘플 데이터를 DataFrame으로 변환하여 표시
                try:
                    sample_df = pd.DataFrame(kosis_data_sample)
                    st.dataframe(sample_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    # DataFrame 변환 실패 시 JSON으로 표시
                    st.json(kosis_data_sample)
            
            message = log.get("message", "")
            if message:
                st.markdown("---")
                st.markdown(f"**메시지:** {message}")
            
            # 에러 정보 표시
            if status == "error":
                st.markdown("---")
                st.markdown("**에러 정보**")
                error = log.get("error", "")
                if error:
                    st.error(f"에러: {error}")
                
                traceback_info = log.get("traceback", "")
                if traceback_info:
                    with st.expander("상세 에러 추적", expanded=False):
                        st.code(traceback_info, language="python")
            
            # 전체 로그 JSON (디버깅용)
            with st.expander("전체 로그 데이터 (JSON)", expanded=False):
                st.json(log)
    
    # 로그 초기화 버튼 (fragment 내부이므로 클릭 시 이 fragment만 갱신)
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("로그 삭제", key="stat_assignment_log_delete"):
            st.session_state.stat_assignment_logs = []


def page_guide():
    st.header("사용 가이드")
    st.markdown("""
    ### 사용 순서
    1. **데이터 관리 탭**: 템플릿 업로드 및 6축 마진 소스 설정
    2. **생성 탭**: 생성 옵션 설정 후 생성 버튼 클릭
    3. **2차 대입 결과 탭**: 2단계 통계 대입 결과 기록 조회
    """)


# -----------------------------
# 8. Pages: 설문 진행 관련 함수들
# -----------------------------
# 공통 UI 컴포넌트: pages/common.py
# 설문조사 페이지: pages/survey.py
# 심층면접 페이지: pages/interview.py


def page_survey_form_builder():
    # Custom CSS 스타일링 (Indigo 테마)
    st.markdown("""
    <style>
    /* 전체 배경 */
    .stApp {
        background-color: #FDFDFF;
    }
    
    /* 텍스트 영역 스타일 */
    .stTextArea > div > div > textarea {
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        padding: 20px;
        font-size: 14px;
        transition: all 0.3s;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #4f46e5;
        box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.1);
        outline: none;
    }
    
    /* 입력 필드 스타일 */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 12px;
        font-size: 14px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4f46e5;
        outline: none;
    }
    
    /* 버튼 스타일 (그림자 제거) */
    .stButton > button {
        border-radius: 24px;
        font-weight: 800;
        font-size: 18px;
        padding: 20px 30px;
        transition: all 0.3s;
        box-shadow: none !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: none !important;
    }
    
    /* 카드 스타일 */
    .survey-card {
        background: white;
        border-radius: 24px;
        padding: 32px;
        border: 1px solid #e0e7ff;
    }
    
    .survey-card-indigo {
        background: #eef2ff;
        border: 1px solid #c7d2fe;
    }
    
    /* 배지 스타일 */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 700;
    }
    
    .badge-indigo {
        background: #eef2ff;
        color: #4f46e5;
    }
    
    /* 헤더 스타일 */
    .survey-header {
        margin-bottom: 24px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 세션 상태 초기화
    if "survey_definition" not in st.session_state:
        st.session_state.survey_definition = ""
    if "survey_needs" not in st.session_state:
        st.session_state.survey_needs = ""
    if "survey_target" not in st.session_state:
        st.session_state.survey_target = ""
    if "survey_website" not in st.session_state:
        st.session_state.survey_website = ""
    if "survey_custom_mode" not in st.session_state:
        st.session_state.survey_custom_mode = False
    
    # 헤더 제거됨 (설문조사/심층면접 페이지는 pages/survey.py, pages/interview.py에서 처리)
    
    # 메인 타이틀
    st.markdown("""
    <div class="survey-header">
        <h1 style="font-size: 32px; font-weight: 800; color: #0f172a; margin-bottom: 12px;">
            새로운 시장성 조사 설계 시작하기
        </h1>
        <p style="color: #64748b; font-size: 16px; line-height: 1.6;">
            제품과 니즈를 상세히 적어주실수록, AI 가상패널이 더 정교한 인사이트를 도출합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 2단 분할 레이아웃
    left_col, right_col = st.columns([0.65, 0.35], gap="large")
    
    with left_col:
        # 필수 입력 1: 제품 정의
        st.markdown("### 제품/서비스의 정의 <span style='color: #ef4444;'>*</span>", unsafe_allow_html=True)
        
        definition_length = len(st.session_state.survey_definition)
        is_definition_valid = definition_length >= 300
        
        col_def_label, col_def_count = st.columns([3, 1])
        with col_def_count:
            if is_definition_valid:
                st.markdown(f"<span style='color: #10b981; font-size: 12px; font-weight: 600;'>{definition_length} / 300자 이상</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color: #94a3b8; font-size: 12px; font-weight: 600;'>{definition_length} / 300자 이상</span>", unsafe_allow_html=True)
        
        definition = st.text_area(
            "제품/서비스 정의",
            value=st.session_state.survey_definition,
            placeholder="제품의 핵심 기능, 가치, 시장 내 위치 등을 상세히 작성해주세요.",
            height=200,
            key="survey_definition_input",
            label_visibility="collapsed"
        )
        st.session_state.survey_definition = definition
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 필수 입력 2: 조사의 니즈
        st.markdown("### 조사의 목적과 니즈 <span style='color: #ef4444;'>*</span>", unsafe_allow_html=True)
        
        needs_length = len(st.session_state.survey_needs)
        is_needs_valid = needs_length >= 300
        
        col_needs_label, col_needs_count = st.columns([3, 1])
        with col_needs_count:
            if is_needs_valid:
                st.markdown(f"<span style='color: #10b981; font-size: 12px; font-weight: 600;'>{needs_length} / 300자 이상</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color: #94a3b8; font-size: 12px; font-weight: 600;'>{needs_length} / 300자 이상</span>", unsafe_allow_html=True)
        
        needs = st.text_area(
            "조사의 목적과 니즈",
            value=st.session_state.survey_needs,
            placeholder="이번 조사를 통해 무엇을 알고 싶으신가요? (예: 타겟 유저의 가격 저항선, 경쟁사 대비 강점 등)",
            height=200,
            key="survey_needs_input",
            label_visibility="collapsed"
        )
        st.session_state.survey_needs = needs
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 선택 입력 섹션
        with st.expander("추가 정보 (선택)", expanded=False):
            col_target, col_website = st.columns(2)
            with col_target:
                st.text_input(
                    "희망 타깃",
                    value=st.session_state.survey_target,
                    placeholder="특정 타깃이 있다면 적어주세요. (예: 30대 워킹맘)",
                    key="survey_target_input"
                )
            with col_website:
                st.text_input(
                    "홈페이지 주소",
                    value=st.session_state.survey_website,
                    placeholder="https://",
                    key="survey_website_input"
                )
            
            st.file_uploader(
                "참고자료 업로드",
                type=["pdf", "jpg", "jpeg", "png", "ppt", "pptx"],
                key="survey_file_upload",
                help="PDF, JPG, PPT 형식 지원"
            )
    
    with right_col:
        # AI 최적화 설계 제안 카드
        is_all_valid = is_definition_valid and is_needs_valid
        
        if is_all_valid:
            card_class = "survey-card-indigo"
        else:
            card_class = "survey-card"
        
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 24px;">
            <span style="color: #4f46e5; font-weight: 700; font-size: 16px;">AI 최적화 설계 제안</span>
        </div>
        """, unsafe_allow_html=True)
        
        if not is_all_valid:
            st.warning("필수 정보를 300자 이상 입력하시면 AI가 최적의 조사 설계를 제안합니다.")
        else:
            # 권장 조사 방식
            col_rec1, col_rec2 = st.columns([1, 1])
            with col_rec1:
                st.markdown("**권장 조사 방식**")
            with col_rec2:
                st.markdown('<span class="badge badge-indigo">질적 조사 (Talk)</span>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 최적 페르소나 그룹
            col_persona1, col_persona2 = st.columns([1, 1])
            with col_persona1:
                st.markdown("**최적 페르소나 그룹**")
            with col_persona2:
                st.markdown("**2,500명 (다변량 추출)**")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # AI 코멘트
            st.info("""
            **AI 코멘트**
            
            입력하신 니즈를 분석한 결과, 구체적인 구매 방해 요소를 파악하기 위해 
            **수천 명의 가상 패널과의 심층 토론(Talk)**이 가장 효과적일 것으로 예측됩니다.
            """)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 커스텀 모드 토글
            col_toggle1, col_toggle2 = st.columns([2, 1])
            with col_toggle1:
                st.markdown("<span style='font-size: 12px; color: #94a3b8;'>조사 구체 계획이 있으신가요?</span>", unsafe_allow_html=True)
            with col_toggle2:
                custom_mode = st.toggle(
                    "맞춤형(Custom) 모드",
                    value=st.session_state.survey_custom_mode,
                    key="survey_custom_toggle"
                )
                st.session_state.survey_custom_mode = custom_mode
            
            if custom_mode:
                st.markdown("---")
                st.markdown("**조사 방식 변경**")
                survey_type = st.radio(
                    "조사 방식",
                    options=["Talk", "Survey"],
                    key="survey_type_radio",
                    horizontal=True
                )
                st.session_state.survey_type = survey_type
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**표본 수 조정**")
                sample_size = st.slider(
                    "표본 수",
                    min_value=100,
                    max_value=10000,
                    value=2500,
                    step=100,
                    key="survey_sample_slider"
                )
                st.session_state.survey_sample_size = sample_size
                st.caption(f"최소 100명 ~ 최대 10,000명 (현재: {sample_size:,}명)")
            else:
                st.session_state.survey_type = "Talk"
                st.session_state.survey_sample_size = 2500
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 조사 시작하기 버튼
        if is_all_valid:
            if st.button("조사 시작하기", type="primary", use_container_width=True, key="survey_start_button"):
                survey_type = st.session_state.get("survey_type", "Talk")
                sample_size = st.session_state.get("survey_sample_size", 2500)
                target = st.session_state.survey_target if st.session_state.survey_target else "전체"
                
                st.success("조사 프로젝트가 시작되었습니다.")
                st.info(f"""
                **설정된 조사 정보:**
                - 조사 방식: {survey_type}
                - 표본 수: {sample_size:,}명
                - 타깃: {target}
                """)
        else:
            st.button("조사 시작하기", disabled=True, use_container_width=True, key="survey_start_button_disabled")
            st.caption("필수 정보를 300자 이상 입력해주세요.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 정보 안내
        st.caption("데이터는 암호화되어 보호되며, 분석 완료 후 즉시 파기됩니다.")


def page_survey_results():
    st.subheader("설문 결과")
    st.info("설문 결과 기능은 추후 확장 범위")


# -----------------------------
# 9. Main UI (상위 폴더 구조)
# -----------------------------
def render_landing():
    """랜딩 페이지: 로고 이미지 + 시작하기 버튼, 사이드바 숨김"""
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    header[data-testid="stHeader"] { display: none; }
    </style>
    """, unsafe_allow_html=True)
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")
    if os.path.isfile(logo_path):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(logo_path, use_container_width=True)
    else:
        st.markdown('<p style="text-align: center; margin-top: 4rem;">Social Simulation</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("시작하기", type="primary", use_container_width=True, key="landing_start"):
            st.session_state.app_started = True
            st.rerun()


def _ensure_generate_modules() -> None:
    """가상인구 생성 탭 전용 무거운 모듈 로드 (해당 탭 진입 시에만 실행)."""
    if "KosisClient" in globals() and globals().get("KosisClient") is not None:
        return
    try:
        from google import genai
        from utils.kosis_client import KosisClient
        from utils.ipf_generator import generate_base_population
        from utils.gemini_client import GeminiClient
        from utils.step2_records import STEP2_RECORDS_DIR, list_step2_records, save_step2_record
        globals()["genai"] = genai
        globals()["KosisClient"] = KosisClient
        globals()["generate_base_population"] = generate_base_population
        globals()["GeminiClient"] = GeminiClient
        globals()["STEP2_RECORDS_DIR"] = STEP2_RECORDS_DIR
        globals()["list_step2_records"] = list_step2_records
        globals()["save_step2_record"] = save_step2_record
    except Exception:
        raise


def _run_page_vdb():
    from pages.virtual_population_db import page_virtual_population_db
    st.title(APP_TITLE)
    page_virtual_population_db()


def _run_page_generate():
    st.title(APP_TITLE)
    try:
        _ensure_generate_modules()
    except Exception as e:
        st.error("가상인구 생성 모듈 로드 실패: " + str(e))
        st.code(traceback.format_exc())
    else:
        gen_tabs = st.tabs(["데이터 관리", "생성", "2차 대입 결과", "통계 대입 로그"])
        with gen_tabs[0]:
            page_data_management()
        with gen_tabs[1]:
            page_generate()
        with gen_tabs[2]:
            page_step2_results()
        with gen_tabs[3]:
            page_stat_assignment_log()


def _run_page_survey():
    from pages.survey import page_survey
    st.title(APP_TITLE)
    page_survey()


def _run_page_conjoint():
    from pages.result_analysis_conjoint import page_conjoint_analysis
    st.title(APP_TITLE)
    page_conjoint_analysis()


def _run_page_psm():
    from pages.result_analysis_psm import page_psm
    st.title(APP_TITLE)
    page_psm()


def _run_page_bass():
    from pages.result_analysis_bass import page_bass
    st.title(APP_TITLE)
    page_bass()


def _run_page_statcheck():
    from pages.result_analysis_statcheck import page_statcheck
    st.title(APP_TITLE)
    page_statcheck()


def _run_page_bg_removal():
    try:
        from pages.utils_background_removal import page_photo_background_removal
    except Exception as e:
        page_photo_background_removal = None
        _bg_err = e
    st.title(APP_TITLE)
    if page_photo_background_removal is not None:
        page_photo_background_removal()
    else:
        st.markdown("## 사진 배경제거")
        st.warning("이 페이지는 JavaScript/Streamlit 모듈로 구성됩니다. 현재 `pages/utils_background_removal.py` 가 Python 모듈이 아닌 경우 동작하지 않습니다.")
        st.caption(str(_bg_err))


def _run_page_clothing():
    try:
        from pages.utils_clothing_change import page_photo_clothing_change
    except Exception as e:
        page_photo_clothing_change = None
        _cloth_err = e
    st.title(APP_TITLE)
    if page_photo_clothing_change is not None:
        page_photo_clothing_change()
    else:
        st.markdown("## 사진 옷 변경")
        st.warning("이 페이지는 JavaScript/Streamlit 모듈로 구성됩니다. 현재 `pages/utils_clothing_change.py` 가 Python 모듈이 아닌 경우 동작하지 않습니다.")
        st.caption(str(_cloth_err))


def main():
    # set_page_config는 run.py에서 이미 1회 호출됨. 여기서 다시 호출하면 Streamlit Cloud 등에서 "can only be called once" 오류로 로딩 실패할 수 있음.
    # st.set_page_config(page_title=APP_TITLE, layout="wide")
    
    # 페이지 전환 시 이전 콘텐츠 잔상(ghosting) 방지 (st.navigation 메뉴는 사이드바에 그대로 표시)
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] main .block-container { opacity: 1 !important; }
    </style>
    """, unsafe_allow_html=True)

    # DB 초기화: Supabase 연결 검증 (세션당 1회 성공 시만 플래그 설정)
    if not st.session_state.get("_db_initialized", False):
        with st.spinner("준비 중…"):
            try:
                db_init()
                st.session_state.pop("db_init_error", None)
                st.session_state["_db_initialized"] = True
            except Exception as e:
                st.session_state["db_init_error"] = str(e)
    ensure_session_state()

    # 엄격한 단일 컨테이너: 모든 메인 UI는 이 플레이스홀더 안에서만 렌더 (잔상 방지)
    if "_main_placeholder" not in st.session_state:
        st.session_state["_main_placeholder"] = st.empty()
    main_container = st.session_state["_main_placeholder"]

    if not st.session_state.get("app_started", False):
        main_container.empty()
        with main_container.container():
            render_landing()
            if st.session_state.get("db_init_error"):
                st.error("Supabase 설정을 확인해주세요. " + st.session_state["db_init_error"])
        return

    # 페이지 전환 시 컨테이너는 각 _run_page_* 내부에서 empty() 후 채움
    # st.navigation: 페이지 전환 시 st.rerun() 없이 전환되어 깜빡임·지연 최소화
    page_vdb = st.Page(_run_page_vdb, title="가상인구 DB", default=True)
    page_gen = st.Page(_run_page_generate, title="가상인구 생성")
    page_survey = st.Page(_run_page_survey, title="시장성 조사 설계")
    page_conjoint = st.Page(_run_page_conjoint, title="[선호도 분석]컨조인트 분석")
    page_psm = st.Page(_run_page_psm, title="[가격 수용성]PSM")
    page_bass = st.Page(_run_page_bass, title="[시장 확산 예측]Bass 확산 모델")
    page_statcheck = st.Page(_run_page_statcheck, title="[가설 검증]A/B 테스트 검증")
    page_bg = st.Page(_run_page_bg_removal, title="사진 배경제거")
    page_cloth = st.Page(_run_page_clothing, title="사진 옷 변경")

    nav = st.navigation({
        "AI Social Twin": [page_vdb, page_gen, page_survey],
        "Result analysis": [page_conjoint, page_psm, page_bass, page_statcheck],
        "Utils": [page_bg, page_cloth],
    })
    nav.run()


if __name__ == "__main__":
    import streamlit as _st
    _st.set_page_config(page_title=APP_TITLE, layout="wide")
    try:
        main()
    except Exception as e:
        import streamlit as _st
        _st.error("앱 로드 중 오류가 발생했습니다.")
        _st.code(str(e))
        import traceback
        _st.code(traceback.format_exc())
