"""
AI Social Twin - 가상인구 생성 및 조사 설계 애플리케이션
"""
import os
import sqlite3
import traceback
import pickle
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO
from datetime import datetime
from google import genai

import pandas as pd
import numpy as np
import streamlit as st

from utils.kosis_client import KosisClient
from utils.ipf_generator import generate_base_population
from utils.gemini_client import GeminiClient


APP_TITLE = "AI Social Twin"

DB_PATH = os.path.join("data", "app.db")
os.makedirs("data", exist_ok=True)

# 오토세이브 경로
AUTOSAVE_PATH = os.path.join("data", "temp_step1_autosave.pkl")

EXPORT_SHEET_NAME = "통계목록"
EXPORT_COLUMNS = ["카테고리", "통계명", "URL", "활성여부"]

DEFAULT_WEIGHTS_SCORE = {
    "income": 0.50,
    "age": 0.20,
    "education": 0.15,
    "gender": 0.10,
    "random": 0.05,
}

# -----------------------------
# 0. 상수/마스터 데이터
# -----------------------------
SIDO_MASTER = [
    {"sido_name": "전국", "sido_code": "00"},
    {"sido_name": "서울특별시", "sido_code": "11"},
    {"sido_name": "부산광역시", "sido_code": "21"},
    {"sido_name": "대구광역시", "sido_code": "22"},
    {"sido_name": "인천광역시", "sido_code": "23"},
    {"sido_name": "광주광역시", "sido_code": "24"},
    {"sido_name": "대전광역시", "sido_code": "25"},
    {"sido_name": "울산광역시", "sido_code": "26"},
    {"sido_name": "세종특별자치시", "sido_code": "29"},
    {"sido_name": "경기도", "sido_code": "31"},
    {"sido_name": "강원도", "sido_code": "32"},
    {"sido_name": "충청북도", "sido_code": "33"},
    {"sido_name": "충청남도", "sido_code": "34"},
    {"sido_name": "전라북도", "sido_code": "35"},
    {"sido_name": "전라남도", "sido_code": "36"},
    {"sido_name": "경상북도", "sido_code": "37"},
    {"sido_name": "경상남도", "sido_code": "38"},
    {"sido_name": "제주특별자치도", "sido_code": "39"},
]

# SIDO 관련 딕셔너리/리스트
SIDO_CODE = {item["sido_code"]: item["sido_name"] for item in SIDO_MASTER}
SIDO_NAME = {item["sido_code"]: item["sido_name"] for item in SIDO_MASTER}
SIDO_LABELS = [f"{x['sido_name']} ({x['sido_code']})" for x in SIDO_MASTER]
SIDO_LABEL_TO_CODE = {f"{x['sido_name']} ({x['sido_code']})": x["sido_code"] for x in SIDO_MASTER}
SIDO_CODE_TO_NAME = {x["sido_code"]: x["sido_name"] for x in SIDO_MASTER}


# -----------------------------
# 1. DB Layer
# -----------------------------
def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_init():
    conn = db_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sido_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sido_code TEXT NOT NULL,
            sido_name TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_sido_profiles_code
        ON sido_profiles(sido_code)
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sido_axis_weights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sido_code TEXT NOT NULL,
            w_sigungu REAL NOT NULL DEFAULT 1.0,
            w_gender REAL NOT NULL DEFAULT 1.0,
            w_age REAL NOT NULL DEFAULT 1.0,
            w_econ REAL NOT NULL DEFAULT 1.0,
            w_income REAL NOT NULL DEFAULT 1.0,
            w_edu REAL NOT NULL DEFAULT 1.0,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(sido_code)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sido_code TEXT NOT NULL,
            category TEXT NOT NULL,
            name TEXT NOT NULL,
            url TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_stats_sido_category_name
        ON stats(sido_code, category, name)
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sido_weight_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sido_code TEXT NOT NULL,
            stat_id INTEGER NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(sido_code, stat_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sido_axis_margin_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sido_code TEXT NOT NULL,
            axis_key TEXT NOT NULL,
            stat_id INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(sido_code, axis_key)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sido_templates (
            sido_code TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_blob BLOB NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
        """
    )

    conn.commit()
    conn.close()


# -----------------------------
# 2. DB Helpers: Stats
# -----------------------------
def db_list_stats_by_sido(sido_code: str, active_only: bool = False) -> List[Dict[str, Any]]:
    conn = db_conn()
    cur = conn.cursor()
    if sido_code == "00":
        if active_only:
            cur.execute("SELECT * FROM stats WHERE is_active=1 ORDER BY sido_code, category, name")
        else:
            cur.execute("SELECT * FROM stats ORDER BY sido_code, category, name")
    else:
        if active_only:
            cur.execute(
                "SELECT * FROM stats WHERE sido_code=? AND is_active=1 ORDER BY category, name",
                (sido_code,),
            )
        else:
            cur.execute("SELECT * FROM stats WHERE sido_code=? ORDER BY category, name", (sido_code,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def db_upsert_stat(sido_code: str, category: str, name: str, url: str, is_active: int = 1):
    conn = db_conn()
    cur = conn.cursor()

    category = str(category).strip()
    name = str(name).strip()
    url = str(url).strip()
    is_active = 1 if int(is_active) != 0 else 0

    cur.execute(
        """
        INSERT INTO stats (sido_code, category, name, url, is_active)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(sido_code, category, name) DO UPDATE SET
            url=excluded.url,
            is_active=excluded.is_active,
            updated_at=datetime('now')
        """,
        (sido_code, category, name, url, is_active),
    )
    conn.commit()
    conn.close()


def db_delete_stat_by_id(stat_id: int):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM stats WHERE id=?", (int(stat_id),))
    conn.commit()
    conn.close()


def db_update_stat_by_id(stat_id: int, category: str, name: str, url: str, is_active: int):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE stats
        SET category=?, name=?, url=?, is_active=?, updated_at=datetime('now')
        WHERE id=?
        """,
        (category.strip(), name.strip(), url.strip(), int(is_active), int(stat_id)),
    )
    conn.commit()
    conn.close()


AXIS_KEYS = ["sigungu", "gender", "age", "econ", "income", "edu"]
AXIS_LABELS = {
    "sigungu": "시군(거주지역)",
    "gender": "성별",
    "age": "연령",
    "econ": "경제활동",
    "income": "소득",
    "edu": "교육정도",
}


def db_get_axis_margin_stats(sido_code: str, axis_key: str) -> Optional[Dict[str, Any]]:
    """특정 시도/축의 마진 통계 소스 조회"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, sido_code, axis_key, stat_id, created_at, updated_at
        FROM sido_axis_margin_stats
        WHERE sido_code = ? AND axis_key = ?
        """,
        (sido_code, axis_key),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def db_get_six_axis_stat_ids(sido_code: str) -> set:
    """6축 마진에 사용 중인 통계 ID 집합 (2단계에서 제외할 대상)"""
    axis_keys = ["sigungu", "gender", "age", "econ", "income", "edu"]
    ids = set()
    for axis_key in axis_keys:
        row = db_get_axis_margin_stats(sido_code, axis_key)
        if row and row.get("stat_id"):
            ids.add(int(row["stat_id"]))
    return ids



def db_upsert_axis_margin_stat(sido_code: str, axis_key: str, stat_id: int):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO sido_axis_margin_stats (sido_code, axis_key, stat_id)
        VALUES (?, ?, ?)
        ON CONFLICT(sido_code, axis_key) DO UPDATE SET
            stat_id=excluded.stat_id,
            updated_at=datetime('now')
        """,
        (sido_code, axis_key, int(stat_id)),
    )
    conn.commit()
    conn.close()

# -----------------------------
# 2-1. DB Helpers: Template
# -----------------------------
def db_upsert_template(sido_code: str, filename: str, file_bytes: bytes):
    """템플릿 저장/업데이트"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO sido_templates (sido_code, filename, file_blob)
        VALUES (?, ?, ?)
        ON CONFLICT(sido_code) DO UPDATE SET
            filename = excluded.filename,
            file_blob = excluded.file_blob,
            updated_at = datetime('now')
        """,
        (sido_code, filename, file_bytes),
    )
    conn.commit()
    conn.close()


def db_get_template(sido_code: str) -> Optional[Dict[str, Any]]:
    """템플릿 조회"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT filename, file_blob, updated_at
        FROM sido_templates
        WHERE sido_code = ?
        """,
        (sido_code,),
    )
    row = cur.fetchone()
    conn.close()
    
    if row:
        return {
            "filename": row[0],
            "bytes": row[1],
            "updated_at": row[2],
        }
    return None


# -----------------------------
# 3. Excel Round-trip
# -----------------------------
def build_stats_template_xlsx_kr() -> bytes:
    df = pd.DataFrame(
        [
            {
                "카테고리": "01. 인구·기본 인적 특성 통계항목",
                "통계명": "시군 및 읍면동별 내·외국인 인구현황",
                "URL": "https://kosis.kr/openapi/statisticsData.do?method=getList&apiKey=YOUR_KEY&format=json&jsonVD=Y&userStatsId=...",
                "활성여부": 1,
            }
        ],
        columns=EXPORT_COLUMNS,
    )
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=EXPORT_SHEET_NAME)
    return bio.getvalue()


def build_stats_export_xlsx_kr(sido_code: str) -> bytes:
    stats = db_list_stats_by_sido(sido_code=sido_code, active_only=False)
    df = pd.DataFrame(stats)
    if df.empty:
        out_df = pd.DataFrame(columns=EXPORT_COLUMNS)
    else:
        out_df = pd.DataFrame(
            {
                "카테고리": df["category"].astype(str),
                "통계명": df["name"].astype(str),
                "URL": df["url"].astype(str),
                "활성여부": df["is_active"].astype(int),
            }
        )

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name=EXPORT_SHEET_NAME)
    return bio.getvalue()


def get_export_filename(prefix: str) -> str:
    d = datetime.utcnow().strftime("%Y%m%d")
    return f"{prefix}_통계목록_EXPORT_{d}.xlsx"


def import_stats_from_excel_kr(sido_code: str, file_bytes: bytes) -> Dict[str, Any]:
    xdf = pd.read_excel(BytesIO(file_bytes), sheet_name=EXPORT_SHEET_NAME)
    xdf.columns = [str(c).strip() for c in xdf.columns]

    required = {"카테고리", "통계명", "URL"}
    missing = required - set(xdf.columns)
    if missing:
        return {"ok": False, "error": f"필수 컬럼 누락: {sorted(list(missing))} (시트명 '{EXPORT_SHEET_NAME}' 확인 필요)"}

    if "활성여부" not in xdf.columns:
        xdf["활성여부"] = 1
    xdf["활성여부"] = xdf["활성여부"].fillna(1)

    reflected, skipped, errors = 0, 0, []
    for i, row in xdf.iterrows():
        try:
            category = str(row.get("카테고리", "")).strip()
            name = str(row.get("통계명", "")).strip()
            url = str(row.get("URL", "")).strip()
            if not category or not name or not url:
                skipped += 1
                continue

            act = row.get("활성여부", 1)
            try:
                act = int(float(act))
            except Exception:
                act = 1
            act = 1 if act != 0 else 0

            db_upsert_stat(sido_code, category, name, url, act)
            reflected += 1
        except Exception as e:
            errors.append({"행": int(i) + 2, "오류": str(e)})

    return {"ok": True, "반영건수": reflected, "스킵건수": skipped, "오류건수": len(errors), "오류상세": errors, "총행수": len(xdf)}


# -----------------------------
# 4. UI Utilities
# -----------------------------
def group_by_category(stats: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for s in stats:
        out.setdefault(s["category"], []).append(s)
    return out


def ensure_session_state():
    if "generated_df" not in st.session_state:
        st.session_state.generated_df = None
    if "report" not in st.session_state:
        st.session_state.report = None
    if "sigungu_list" not in st.session_state:
        st.session_state.sigungu_list = ["전체"]
    if "selected_categories" not in st.session_state:
        st.session_state.selected_categories = []
    if "selected_stats" not in st.session_state:
        st.session_state.selected_stats = []
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "selected_sido_label" not in st.session_state:
        st.session_state.selected_sido_label = "경상북도 (37)"


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
        data = kosis_client.fetch_json(url)
        sigungu_list = kosis_client.extract_sigungu_list_from_population_table(
            data, sido_prefix=SIDO_CODE_TO_NAME.get(sido_code, "")
        )
        return sigungu_list
    except Exception as e:
        st.warning(f"시군구 목록 로드 실패: {e}")
        return []



def page_data_management():
    """데이터 관리 페이지"""
    st.title("📊 데이터 관리")

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
        if st.button("📥 템플릿 다운로드", key="download_stats_template"):
            template_bytes = build_stats_template_xlsx_kr(sido_code)
            st.download_button(
                "통계목록 템플릿.xlsx 다운로드",
                data=template_bytes,
                file_name=f"{sido_name}_통계목록_템플릿.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        if st.button("📤 활성화 통계 내보내기", key="export_active_stats"):
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
            st.success(
                f"✅ 통계 목록 업로드 완료\n"
                f"- 신규: {result['inserted']}\n"
                f"- 업데이트: {result['updated']}\n"
                f"- 오류: {result['errors']}"
            )
            st.rerun()

    # 통계 목록 표시
    st.markdown("---")
    all_stats = db_list_stats_by_sido(sido_code)
    if not all_stats:
        st.info("등록된 통계가 없습니다.")
    else:
        df_stats = pd.DataFrame(all_stats)
        df_stats["is_active"] = df_stats["is_active"].map({1: "✅", 0: "❌"})
        st.dataframe(df_stats, use_container_width=True)

    # 6축 고정 마진 통계 소스 설정
    st.markdown("---")
    st.subheader("6축 고정 마진 통계 소스")
    st.markdown("각 축의 목표 마진을 제공할 통계를 선택하세요.")

    active_stats = [s for s in all_stats if s["is_active"] == 1]
    if not active_stats:
        st.info("활성화된 통계가 없습니다.")
    else:
        stat_options = {s["id"]: f"[{s['category']}] {s['name']}" for s in active_stats}

        for axis_key, axis_label in [
            ("sigungu", "거주지역"),
            ("gender", "성별"),
            ("age", "연령"),
            ("econ", "경제활동"),
            ("income", "소득"),
            ("edu", "교육"),
        ]:
            current_stat = db_get_axis_margin_stats(sido_code, axis_key)
            current_id = current_stat["stat_id"] if current_stat else None

            selected_id = st.selectbox(
                f"{axis_label} ({axis_key})",
                options=[None] + list(stat_options.keys()),
                format_func=lambda x: "선택 안 함" if x is None else stat_options[x],
                index=0 if current_id is None else list(stat_options.keys()).index(current_id) + 1 if current_id in stat_options else 0,
                key=f"axis_margin_{axis_key}",
            )

            if selected_id and selected_id != current_id:
                db_upsert_axis_margin_stat(sido_code, axis_key, selected_id)
                st.success(f"✅ {axis_label} 마진 통계 업데이트: {stat_options[selected_id]}")
                st.rerun()

def convert_kosis_to_distribution(kosis_data, axis_key: str):
    """
    KOSIS 데이터를 확률 분포로 변환 (labels, probabilities)
    
    axis_key: "sigungu", "gender", "age", "econ", "income", "edu"
    """
    from datetime import datetime
    
    # 로그 기록 초기화
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage": f"convert_kosis_to_distribution({axis_key})",
        "input_data_type": str(type(kosis_data)),
        "input_sample": str(kosis_data)[:500] if kosis_data else "None",
        "axis_key": axis_key,
        "status": "started"
    }
    
    try:
        if not kosis_data:
            log_entry["status"] = "error"
            log_entry["error"] = "kosis_data is empty"
            if "work_logs" not in st.session_state:
                st.session_state.work_logs = []
            st.session_state.work_logs.append(log_entry)
            return [], []
        
        # 리스트 형태로 변환
        if isinstance(kosis_data, dict):
            if "data" in kosis_data:
                kosis_data = kosis_data["data"]
            else:
                kosis_data = [kosis_data]
        
        if not isinstance(kosis_data, list):
            log_entry["status"] = "error"
            log_entry["error"] = f"Invalid data type: {type(kosis_data)}"
            st.session_state.work_logs.append(log_entry)
            return [], []
        
        labels = []
        values = []
        
        # === 축별 처리 ===
        if axis_key == "sigungu":
            # ✅ C1_NM(지역명) 사용
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
                        except:
                            pass
        
        elif axis_key == "gender":
            # ✅ C2_NM에서 성별 추출 (순서 보장: 남자, 여자)
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
            
            # ✅ 순서 보장: 항상 남자, 여자 순서로
            for gender in ["남자", "여자"]:
                if gender_map[gender] > 0:
                    labels.append(gender)
                    values.append(gender_map[gender])
        
        elif axis_key == "age":
            # ✅ C3_NM에서 연령 추출
            import re
            age_map = {}
            
            for row in kosis_data:
                if isinstance(row, dict):
                    age_str = row.get("C3_NM", "").strip()
                    val = row.get("DT", "0")
                    
                    if age_str and age_str != "계":
                        match = re.search(r'(\d+)', age_str)
                        if match:
                            try:
                                age_num = int(match.group(1))
                                # 20세 미만은 생성하지 않음
                                if 20 <= age_num <= 120:
                                    age_map[age_num] = age_map.get(age_num, 0) + float(val)
                            except:
                                pass
            
            for age_num in sorted(age_map.keys()):
                labels.append(age_num)
                values.append(age_map[age_num])
        
        elif axis_key == "econ":
            # ✅ "하였다" → "경제활동"
            econ_map = {}
            for row in kosis_data:
                if isinstance(row, dict):
                    label = row.get("C2_NM", "").strip()
                    val = row.get("DT", "0").strip()
                    
                    # '-' 값 처리
                    if val == "-" or val == "" or val is None:
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
                        # 명시되지 않은 경우 비경제활동으로 분류
                        econ_map["비경제활동"] = econ_map.get("비경제활동", 0) + val_float
            
            for econ_type, total_val in econ_map.items():
                labels.append(econ_type)
                values.append(total_val)
        
        elif axis_key == "edu":
            # ✅ 교육정도 처리 (C2_NM, C3_NM, C4_NM, C5_NM 등 모든 필드에서 찾기)
            edu_map = {"중졸이하": 0, "고졸": 0, "대졸이상": 0}
            
            for row in kosis_data:
                if isinstance(row, dict):
                    # 모든 가능한 필드에서 교육정도 찾기 (C2_NM부터 C5_NM까지)
                    label = ""
                    for field in ["C2_NM", "C3_NM", "C4_NM", "C5_NM"]:
                        val_field = row.get(field, "").strip()
                        if val_field and val_field != "계" and val_field != "전체" and val_field != "소계":
                            # 교육정도 관련 키워드가 있는지 확인
                            if any(keyword in val_field for keyword in ["초졸", "중졸", "고졸", "대졸", "대학", "무학", "초등", "전문대", "석사", "박사"]):
                                label = val_field
                                break
                    
                    # label을 찾지 못했지만 다른 필드에 값이 있으면 그것도 시도
                    if not label:
                        for field in ["C3_NM", "C4_NM", "C5_NM"]:
                            val_field = row.get(field, "").strip()
                            if val_field and val_field != "계" and val_field != "전체" and val_field != "소계":
                                label = val_field
                                break
                    
                    if not label:
                        continue
                    
                    dt_val = row.get("DT", "0").strip()
                    
                    # '-' 처리
                    if dt_val == "-" or dt_val == "" or dt_val is None:
                        continue
                    
                    try:
                        val_float = float(dt_val)
                    except (ValueError, TypeError):
                        continue
                    
                    # ✅ 매핑
                    if "초졸" in label or "무학" in label or "초등" in label or "중졸" in label:
                        edu_map["중졸이하"] += val_float
                    elif "고졸" in label or "고등" in label:
                        edu_map["고졸"] += val_float
                    elif "대학" in label or "대졸" in label or "전문대" in label or "석사" in label or "박사" in label:
                        edu_map["대졸이상"] += val_float
                    else:
                        # 명시되지 않은 경우에도 값이 있으면 중졸이하로 분류 (안전한 기본값)
                        if val_float > 0:
                            edu_map["중졸이하"] += val_float
            
            # 최소한 하나의 값이 있어야 함
            if sum(edu_map.values()) == 0:
                # 기본값 사용
                edu_map = {"중졸이하": 25.0, "고졸": 40.0, "대졸이상": 35.0}
            
            for edu_level, total_val in edu_map.items():
                if total_val > 0:
                    labels.append(edu_level)
                    values.append(total_val)
        
        elif axis_key == "income":
            # ✅ 소득 구간 정확한 매핑 (50만원미만 추가)
            income_map = {
                "50만원미만": 0,
                "50-100만원": 0,
                "100-200만원": 0,
                "200-300만원": 0,
                "300-400만원": 0,
                "400-500만원": 0,
                "500-600만원": 0,
                "600-700만원": 0,
                "700-800만원": 0,
                "800만원이상": 0,
            }
            
            for row in kosis_data:
                if isinstance(row, dict):
                    label = row.get("C2_NM", "").strip()
                    val = row.get("DT", "0")
                    
                    try:
                        val_float = float(val)
                        
                        # ✅ 정확한 구간 매핑
                        if "50만원미만" in label or ("50" in label and "미만" in label and "100" not in label):
                            income_map["50만원미만"] += val_float
                        elif ("50~100" in label or "50-100" in label) or ("50만원" in label and "100만원" in label):
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
                    except:
                        pass
            
            for income_range, total_val in income_map.items():
                if total_val > 0:
                    labels.append(income_range)
                    values.append(total_val)
        
        # === 확률 정규화 ===
        if len(labels) > 0 and len(values) > 0:
            total = sum(values)
            if total > 0:
                probabilities = [v / total for v in values]
            else:
                probabilities = [1.0 / len(values)] * len(values)
            
            log_entry["status"] = "success"
            log_entry["output_labels"] = str(labels)[:200]
            log_entry["output_probs"] = str(probabilities)[:200]
            log_entry["label_count"] = len(labels)
            
            if "work_logs" not in st.session_state:
                st.session_state.work_logs = []
            st.session_state.work_logs.append(log_entry)
            
            return labels, probabilities
        
        else:
            log_entry["status"] = "error"
            log_entry["error"] = "No valid labels/values extracted"
            st.session_state.work_logs.append(log_entry)
            return [], []
    
    except Exception as e:
        log_entry["status"] = "exception"
        log_entry["error"] = str(e)
        log_entry["traceback"] = traceback.format_exc()
        
        if "work_logs" not in st.session_state:
            st.session_state.work_logs = []
        st.session_state.work_logs.append(log_entry)
        
        return [], []



# -----------------------------
# 6. Chart Helper Functions
# -----------------------------
def draw_population_pyramid(df: pd.DataFrame):
    """인구 피라미드 그래프 (성별·연령별 인구 구조) - 20세 이상만 표시"""
    import plotly.graph_objects as go
    
    if "성별" not in df.columns or "연령" not in df.columns:
        st.info("성별 또는 연령 데이터가 없습니다.")
        return
    
    # 20세 미만 데이터 필터링
    df_filtered = df[df["연령"] >= 20].copy()
    
    # 데이터 전처리: 성별별 연령대 카운트
    male_data = df_filtered[df_filtered["성별"].isin(["남자", "남"])].copy()
    female_data = df_filtered[df_filtered["성별"].isin(["여자", "여"])].copy()
    
    # 연령대별 카운트
    male_age_counts = male_data["연령"].value_counts().sort_index()
    female_age_counts = female_data["연령"].value_counts().sort_index()
    
    # 모든 연령대 통합 (20세 이상만)
    all_ages = sorted(set(male_age_counts.index) | set(female_age_counts.index))
    all_ages = [age for age in all_ages if 20 <= age <= 120]
    
    # 각 연령대별 남자/여자 인구수 (남자는 음수로 변환)
    male_counts = [-male_age_counts.get(age, 0) for age in all_ages]
    female_counts = [female_age_counts.get(age, 0) for age in all_ages]
    
    # 총 인구수 계산 (비율 계산용) - 필터링된 데이터 기준
    total_population = len(df_filtered)
    
    # 그래프 생성
    fig = go.Figure()
    
    # 남자 (왼쪽, 음수)
    fig.add_trace(go.Bar(
        y=all_ages,
        x=male_counts,
        name='남자',
        orientation='h',
        marker=dict(color='rgba(31, 119, 180, 0.8)'),
        hovertemplate='<b>남자</b><br>' +
                      '연령: %{y}세<br>' +
                      '인구수: %{customdata[0]:,}명<br>' +
                      '비율: %{customdata[1]:.2f}%<extra></extra>',
        customdata=[[abs(count), (abs(count) / total_population * 100) if total_population > 0 else 0] 
                    for count in male_counts]
    ))
    
    # 여자 (오른쪽, 양수)
    fig.add_trace(go.Bar(
        y=all_ages,
        x=female_counts,
        name='여자',
        orientation='h',
        marker=dict(color='rgba(255, 127, 14, 0.8)'),
        hovertemplate='<b>여자</b><br>' +
                      '연령: %{y}세<br>' +
                      '인구수: %{customdata[0]:,}명<br>' +
                      '비율: %{customdata[1]:.2f}%<extra></extra>',
        customdata=[[count, (count / total_population * 100) if total_population > 0 else 0] 
                    for count in female_counts]
    ))
    
    # 레이아웃 설정
    max_abs_value = max(max(abs(m) for m in male_counts), max(female_counts)) if male_counts or female_counts else 1
    
    # X축 설정: 절대값으로 표시
    tick_vals = []
    tick_texts = []
    step = max(1, int(max_abs_value / 10))
    for i in range(-int(max_abs_value), int(max_abs_value) + step, step):
        tick_vals.append(i)
        tick_texts.append(str(abs(i)))
    
    fig.update_layout(
        xaxis=dict(
            title="인구수 (명)",
            tickvals=tick_vals,
            ticktext=tick_texts,
            range=[-max_abs_value * 1.1, max_abs_value * 1.1]
        ),
        yaxis=dict(
            title="연령 (세)"
            # autorange='reversed' 제거: 아래가 젊은 나이, 위가 많은 나이
        ),
        barmode='overlay',
        height=300,  # 2x3 그리드에 맞게 높이 조정
        margin=dict(l=50, r=50, t=20, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True, key="pyramid_main")


def draw_charts(df: pd.DataFrame, step2_columns: Optional[List[str]] = None, step2_only: bool = False):
    """6축 그래프 + 2단계 추가 통계 분포 렌더링. step2_only=True면 추가 통계만 표시."""
    import plotly.graph_objects as go
    
    step2_columns = step2_columns or []
    
    # 2단계 전용 모드가 아니면 6축 그래프 표시
    if not step2_only:
        st.markdown("### 📊 6축 분포 그래프")
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        row3_col1, row3_col2 = st.columns(2)
        
        # 1행: 거주지역, 성별
        with row1_col1:
            st.markdown("**거주지역**")
            if "거주지역" in df.columns:
                region_counts = df["거주지역"].value_counts()
                max_count = region_counts.max()
                colors = [f'rgba(31, 119, 180, {0.3 + 0.7 * (count / max_count)})' 
                        for count in region_counts.values]
                
                fig = go.Figure(data=[
                    go.Bar(x=region_counts.index, y=region_counts.values, 
                        marker_color=colors, text=region_counts.values, textposition='auto')
                ])
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_title="",
                    yaxis_title="인구수"
                )
                st.plotly_chart(fig, use_container_width=True, key="chart_region")
            else:
                st.info("거주지역 데이터가 없습니다.")
        
        with row1_col2:
            st.markdown("**성별**")
            if "성별" in df.columns:
                gender_counts = df["성별"].value_counts()
                max_count = gender_counts.max()
                colors = ['rgba(255, 127, 14, {})'.format(0.3 + 0.7 * (count / max_count)) 
                        for count in gender_counts.values]
                
                fig = go.Figure(data=[
                    go.Bar(x=gender_counts.index, y=gender_counts.values, 
                        marker_color=colors, text=gender_counts.values, textposition='auto')
                ])
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_title="",
                    yaxis_title="인구수"
                )
                st.plotly_chart(fig, use_container_width=True, key="chart_gender")
            else:
                st.info("성별 데이터가 없습니다.")
        
        # 2행: 연령 (인구 피라미드), 경제활동
        with row2_col1:
            st.markdown("**연령 (인구 피라미드)**")
            if "연령" in df.columns and "성별" in df.columns:
                draw_population_pyramid(df)
            else:
                st.info("연령 또는 성별 데이터가 없습니다.")
        
        with row2_col2:
            st.markdown("**경제활동**")
            if "경제활동" in df.columns:
                econ_counts = df["경제활동"].value_counts()
                max_count = econ_counts.max()
                colors = ['rgba(214, 39, 40, {})'.format(0.3 + 0.7 * (count / max_count)) 
                        for count in econ_counts.values]
                
                fig = go.Figure(data=[
                    go.Bar(x=econ_counts.index, y=econ_counts.values, 
                        marker_color=colors, text=econ_counts.values, textposition='auto')
                ])
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_title="",
                    yaxis_title="인구수"
                )
                st.plotly_chart(fig, use_container_width=True, key="chart_econ")
            else:
                st.info("경제활동 데이터가 없습니다.")
        
        # 3행: 교육정도, 월평균소득
        with row3_col1:
            st.markdown("**교육정도**")
            if "교육정도" in df.columns:
                edu_counts = df["교육정도"].value_counts()
                max_count = edu_counts.max()
                colors = ['rgba(148, 103, 189, {})'.format(0.3 + 0.7 * (count / max_count)) 
                        for count in edu_counts.values]
                
                fig = go.Figure(data=[
                    go.Bar(x=edu_counts.index, y=edu_counts.values, 
                        marker_color=colors, text=edu_counts.values, textposition='auto')
                ])
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_title="",
                    yaxis_title="인구수"
                )
                st.plotly_chart(fig, use_container_width=True, key="chart_edu")
            else:
                st.info("교육정도 데이터가 없습니다.")
        
        with row3_col2:
            st.markdown("**월평균소득**")
            if "월평균소득" in df.columns:
                income_counts = df["월평균소득"].value_counts()
                income_order = ["50만원미만", "50-100만원", "100-200만원", "200-300만원", 
                            "300-400만원", "400-500만원", "500-600만원", "600-700만원", 
                            "700-800만원", "800만원이상"]
                income_sorted = income_counts.reindex([i for i in income_order if i in income_counts.index], fill_value=0)
                max_count = income_sorted.max() if len(income_sorted) > 0 else 1
                colors = ['rgba(140, 86, 75, {})'.format(0.3 + 0.7 * (count / max_count)) 
                        for count in income_sorted.values]
                
                fig = go.Figure(data=[
                    go.Bar(x=income_sorted.index, y=income_sorted.values, 
                        marker_color=colors, text=income_sorted.values, textposition='auto')
                ])
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_title="",
                    yaxis_title="인구수"
                )
                st.plotly_chart(fig, use_container_width=True, key="chart_income")
            else:
                st.info("월평균소득 데이터가 없습니다.")
    
    # 2단계 추가 통계 분포 (해당 데이터 기반)
    if step2_columns:
        if not step2_only:
            st.markdown("---")
        if not step2_only:
            st.markdown("### 📊 2단계 추가 통계 분포")
        # 2열 그리드로 추가 통계 막대 그래프
        for i in range(0, len(step2_columns), 2):
            cols = st.columns(2)
            for j, col_widget in enumerate(cols):
                idx = i + j
                if idx >= len(step2_columns):
                    break
                col_name = step2_columns[idx]
                with col_widget:
                    st.markdown(f"**{col_name}**")
                    counts = df[col_name].value_counts()
                    if counts.empty or len(counts) == 0:
                        st.caption("데이터 없음")
                        continue
                    fig = go.Figure(data=[
                        go.Bar(x=counts.index.astype(str), y=counts.values,
                               text=counts.values, textposition="auto",
                               marker_color="rgba(44, 160, 44, 0.7)")
                    ])
                    fig.update_layout(
                        height=280,
                        margin=dict(l=0, r=0, t=5, b=80),
                        xaxis_title="",
                        yaxis_title="인구수",
                        xaxis_tickangle=-45,
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_step2_{idx}")


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
        "하는일",
        "임금/가구소득",
        "근로시간",
        "근무환경",
        "전반적인 만족도",
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
            kosis_data = kosis_client.fetch_json(url)
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
    st.markdown("### 📊 통계 대비 검증")
    
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
                    st.success("✅ 매우 우수")
                elif axis_mae_pct < 3:
                    st.success("✅ 우수")
                elif axis_mae_pct < 5:
                    st.info("ℹ️ 양호")
                else:
                    st.warning("⚠️ 개선 필요")
            
            st.markdown("---")
        
        # 전체 평균 부합도
        if total_errors:
            overall_avg_error = sum(total_errors) / len(total_errors)
            overall_match_rate = max(0, 100 - overall_avg_error)
            
            st.markdown("### 🎯 전체 통계 부합도")
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
                    st.success("✅ 매우 우수")
                elif avg_error < 3:
                    st.success("✅ 우수")
                elif avg_error < 5:
                    st.info("ℹ️ 양호")
                else:
                    st.warning("⚠️ 개선 필요")
            
            st.markdown("---")
        
        # 전체 평균 부합도
        if total_errors:
            overall_avg_error = sum(total_errors) / len(total_errors)
            overall_match_rate = max(0, 100 - overall_avg_error)
            
            st.markdown("### 🎯 전체 통계 부합도")
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
    st.title("🏘️ 가상인구 생성")
    
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
                st.info("💾 이전 생성 결과를 자동으로 불러왔습니다.")
            except Exception as e:
                st.warning(f"⚠️ 오토세이브 파일 로드 실패: {e}")

    # 좌우 2단 레이아웃 (0.35:0.65 비율)
    col_left, col_right = st.columns([0.35, 0.65])

    # ========== 좌측: 생성 옵션 ==========
    with col_left:
        st.subheader("🛠️ 생성 옵션")
        
        # 초기화 버튼
        if st.button("🗑️ 결과 초기화", type="secondary", use_container_width=True):
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
            st.success("✅ 초기화 완료")
            st.rerun()
        
        st.markdown("---")

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

        # 3) 6축 가중치
        st.markdown("**6축 가중치**")
        w_sigungu = st.slider("시군구", 0.0, 5.0, 1.0, key="w_sigungu")
        w_gender = st.slider("성별", 0.0, 5.0, 1.0, key="w_gender")
        w_age = st.slider("연령", 0.0, 5.0, 1.0, key="w_age")
        w_econ = st.slider("경제활동", 0.0, 5.0, 1.0, key="w_econ")
        w_income = st.slider("소득", 0.0, 5.0, 1.0, key="w_income")
        w_edu = st.slider("교육정도", 0.0, 5.0, 1.0, key="w_edu")

        # 4) 통계 목표 활성화 (UI만 유지)
        st.markdown("**통계 목표 활성화**")
        active_stats = [s for s in db_list_stats_by_sido(sido_code) if s["is_active"] == 1]
        if not active_stats:
            st.info("활성화된 통계가 없습니다.")
        else:
            stat_options = {s["id"]: f"[{s['category']}] {s['name']}" for s in active_stats}
            st.multiselect(
                "목표로 할 통계 선택 (선택 사항)",
                options=list(stat_options.keys()),
                format_func=lambda x: stat_options[x],
            )

        # 🔍 마지막 실행 로그 (디버깅용)
        with st.expander("🔍 마지막 실행 로그 (디버깅용)", expanded=False):
            _log = st.session_state.get("step1_debug_log") or "(아직 실행 없음)"
            st.code(_log, language="text", line_numbers=False) if _log != "(아직 실행 없음)" else st.text(_log)

        # 5) 생성 버튼
        if st.button("🚀 가상인구 생성", type="primary", key="btn_gen_pop"):
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

                        # 6축 마진 통계 소스에서 KOSIS 데이터 가져와서 확률 분포로 변환
                        with st.spinner("📊 KOSIS 통계 데이터 가져오는 중..."):
                            axis_keys = ["sigungu", "gender", "age", "econ", "income", "edu"]
                            margins_axis = {}
                            for axis_key in axis_keys:
                                margin_stat = db_get_axis_margin_stats(sido_code, axis_key)
                                if margin_stat and margin_stat.get("stat_id"):
                                    all_stats = db_list_stats_by_sido(sido_code)
                                    stat_info = next((s for s in all_stats if s["id"] == margin_stat["stat_id"]), None)
                                    if stat_info:
                                        st.info(f"📊 {axis_key} ← [{stat_info['category']}] {stat_info['name']}")
                                        try:
                                            kosis_data = kosis.fetch_json(stat_info["url"])
                                            labels, probs = convert_kosis_to_distribution(kosis_data, axis_key)
                                            if labels and probs:
                                                margins_axis[axis_key] = {"labels": labels, "p": probs}
                                                st.success(f"✅ {axis_key}: {len(labels)}개 항목 ({sum(probs):.2f} 확률 합)")
                                            else:
                                                st.warning(f"⚠️ {axis_key}: KOSIS 데이터 변환 실패 (균등 분포 사용)")
                                        except Exception as e:
                                            st.warning(f"⚠️ {axis_key}: KOSIS 데이터 가져오기 실패: {e}")
                            if len(margins_axis) < 6:
                                st.warning(f"⚠️ KOSIS 통계 기반: {len(margins_axis)}/6 (나머지는 기본값)")
                            print(f"[1단계] 마진 수집 완료: {list(margins_axis.keys())}")

                        # 1단계: KOSIS 통계 기반 6축 인구 생성
                        print(f"[1단계] generate_base_population 호출 직전 (n={int(n)})")
                        with st.spinner(f"1단계: KOSIS 통계 기반 {int(n)}명 생성 중..."):
                            # sigungu_pool 생성 (margins_axis에서 거주지역 목록 추출)
                            sigungu_pool = []
                            if margins_axis and "sigungu" in margins_axis:
                                sigungu_pool = margins_axis["sigungu"].get("labels", [])
                            import random
                            seed = 42
                            base_df = generate_base_population(
                                n=int(n),
                                selected_sigungu=[],
                                weights_6axis={
                                    'sigungu': w_sigungu,
                                    'gender': w_gender,
                                    'age': w_age,
                                    'econ': w_econ,
                                    'income': w_income,
                                    'edu': w_edu,
                                },
                                sigungu_pool=sigungu_pool,
                                seed=seed,
                                margins_axis=margins_axis if margins_axis else {},
                            )

                        if base_df is None or base_df.empty:
                            st.error("기본 인구 생성 실패")
                            st.stop()

                        # ✅ 오차율 검증 및 재생성 로직
                        max_retries = 5
                        retry_count = 0
                        avg_mae = 100.0  # 초기값
                        
                        while retry_count < max_retries:
                            # 오차율 계산
                            if margins_axis:
                                avg_mae, error_report = calculate_mape(base_df, margins_axis)
                                avg_mae_pct = avg_mae * 100
                                
                                if avg_mae_pct < 5.0:
                                    # 오차율이 5.0% 미만이면 통과
                                    break
                                else:
                                    # 오차율이 5.0% 이상이면 재생성
                                    retry_count += 1
                                    if retry_count < max_retries:
                                        st.warning(f"⚠️ 평균 오차율 {avg_mae_pct:.2f}% (목표: 5.0% 미만) - {retry_count}회차 재생성 중...")
                                        # seed 변경하여 재생성
                                        seed = 42 + retry_count
                                        base_df = generate_base_population(
                                            n=int(n),
                                            selected_sigungu=[],
                                            weights_6axis={
                                                'sigungu': w_sigungu,
                                                'gender': w_gender,
                                                'age': w_age,
                                                'econ': w_econ,
                                                'income': w_income,
                                                'edu': w_edu,
                                            },
                                            sigungu_pool=sigungu_pool,
                                            seed=seed,
                                            margins_axis=margins_axis if margins_axis else {},
                                        )
                                    else:
                                        st.warning(f"⚠️ 최대 재시도 횟수({max_retries}회) 도달. 현재 오차율: {avg_mae_pct:.2f}%")
                            else:
                                # margins_axis가 없으면 검증 스킵
                                break
                        
                        if margins_axis and avg_mae * 100 < 5.0:
                            st.success(f"✅ KOSIS 통계 기반 {len(base_df)}명 생성 완료 (평균 오차율: {avg_mae * 100:.2f}%)")
                        else:
                            st.success(f"✅ KOSIS 통계 기반 {len(base_df)}명 생성 완료")

                    # Excel 생성: 기본 컬럼 + 추후 2단계에서 추가 통계는 오른쪽에 열로 이어 붙임
                    out_buffer = io.BytesIO()
                    base_df.to_excel(out_buffer, index=False, engine="openpyxl")
                    out_buffer.seek(0)
                    xbytes = out_buffer.getvalue()

                    # 세션에 저장
                    st.session_state["generated_excel"] = xbytes
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
                        st.warning(f"⚠️ 오토세이브 저장 실패: {e}")

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
                    
                    st.success(f"✅ 1단계 완료: KOSIS 통계 기반 생성!")
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
        st.subheader("📊 생성 결과 대시보드")
        
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
        st.markdown("### 📈 요약 지표")
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
            st.markdown("### 📋 데이터 미리보기 (2단계 완료: 추가 통계 포함)")
        else:
            st.markdown("### 📋 데이터 미리보기")
        
        # 데이터프레임 컬럼 순서 그대로 표시 (기본 컬럼 + 추가 통계 오른쪽 열)
        df_preview = df
        st.dataframe(df_preview.head(100), height=300, use_container_width=True)
        
        st.markdown("---")
        
        # ✅ 2단계: 다른 통계 대입 버튼 (항상 표시 - 반복 대입 가능)
        st.markdown("### 🔄 2단계: 다른 통계 대입")
        col_step2_1, col_step2_2 = st.columns([3, 1])
        with col_step2_1:
            if is_step2:
                st.info("💡 추가 KOSIS 통계를 반복적으로 대입할 수 있습니다. (이미 대입된 통계 포함)")
            else:
                st.info("💡 1단계에서 생성된 가상인구에 추가 KOSIS 통계를 대입할 수 있습니다.")
        with col_step2_2:
            if st.button("📊 다른 통계 대입", type="primary", use_container_width=True):
                st.session_state["show_step2_dialog"] = True
                st.rerun()
        
        # 2단계 다이얼로그 표시
        if st.session_state.get("show_step2_dialog", False):
            with st.expander("📊 2단계: 다른 통계 선택 및 대입", expanded=True):
                sido_code = st.session_state.get("generated_sido_code", "")
                if not sido_code:
                    st.error("먼저 1단계 가상인구를 생성해주세요.")
                else:
                    # 활성화된 통계 목록 가져오기 (6축 마진에 쓰인 통계는 제외 — 이미 1단계에서 반영됨)
                    all_stats = db_list_stats_by_sido(sido_code)
                    active_stats = [s for s in all_stats if s.get("is_active", 0) == 1]
                    six_axis_stat_ids = db_get_six_axis_stat_ids(sido_code)
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
                            st.caption("💡 6축 마진에 사용 중인 통계는 목록에서 제외됩니다 (이미 1단계에 반영됨).")
                        
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
                            if st.button("✅ 통계 대입 실행", type="primary", use_container_width=True):
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
                                                        kosis_data = kosis.fetch_json(stat_info["url"])
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
                                                        use_public_transport_satisfaction = "거주지역 대중교통 만족도" in (stat_info.get("name") or "")
                                                        use_medical_facility_main = "의료기관 주 이용시설" in (stat_info.get("name") or "")
                                                        use_medical_satisfaction = "의료시설 만족도" in (stat_info.get("name") or "")
                                                        use_welfare_satisfaction = "지역의 사회복지 서비스 만족도" in (stat_info.get("name") or "")
                                                        use_provincial_satisfaction = "도정만족도" in (stat_info.get("name") or "")
                                                        use_social_communication = "사회적관계별 소통정도" in (stat_info.get("name") or "")
                                                        use_trust_people = "일반인에 대한 신뢰" in (stat_info.get("name") or "")
                                                        use_subjective_class = "주관적 귀속계층" in (stat_info.get("name") or "")
                                                        use_volunteer = "자원봉사활동" in (stat_info.get("name") or "") and "여부" in (stat_info.get("name") or "")
                                                        use_donation = "후원금 금액" in (stat_info.get("name") or "")
                                                        use_regional_belonging = "지역소속감" in (stat_info.get("name") or "")
                                                        use_safety_eval = "안전환경에 대한 평가" in (stat_info.get("name") or "")
                                                        use_crime_fear = "일상생활 범죄피해 두려움" in (stat_info.get("name") or "")
                                                        use_daily_fear = "일상생활에서 두려움" in (stat_info.get("name") or "")
                                                        use_law_abiding = "자신의 평소 준법수준" in (stat_info.get("name") or "")
                                                        use_environment_feel = "환경체감도" in (stat_info.get("name") or "")
                                                        use_time_pressure = "생활시간 압박" in (stat_info.get("name") or "")
                                                        use_leisure_satisfaction = "여가활동 만족도" in (stat_info.get("name") or "") and "불만족 이유" in (stat_info.get("name") or "")
                                                        use_culture_attendance = "문화예술행사" in (stat_info.get("name") or "") and "관람" in (stat_info.get("name") or "")
                                                        use_life_satisfaction = "삶에 대한 만족감과 정서경험" in (stat_info.get("name") or "")
                                                        use_happiness_level = "행복수준" in (stat_info.get("name") or "")
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
                                                        # 학생 및 미취학자녀수: 템플릿에 F·G 2열이 없어도 기본 컬럼명으로 2열 대입
                                                        if use_children_student:
                                                            col_f, col_g = (two_names[0], two_names[1]) if len(two_names) >= 2 else ("자녀유무", "총학생수")
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
                                                        # 부모님 생존여부 및 동거여부: 2열(생존여부, 부모님 동거 여부)
                                                        elif use_parents_survival_cohabitation:
                                                            col_survival, col_cohabitation = (two_names[0], two_names[1]) if len(two_names) >= 2 else ("생존여부", "부모님 동거 여부")
                                                            result_df, success = kosis.assign_parents_survival_cohabitation_columns(
                                                                result_df,
                                                                kosis_data,
                                                                column_names=(col_survival, col_cohabitation),
                                                                seed=42,
                                                            )
                                                        # 부모님 생활비 주 제공자: 단일 컬럼
                                                        elif use_parents_expense_provider:
                                                            col_expense = "부모님 생활비 주 제공자"
                                                            result_df, success = kosis.assign_parents_expense_provider_column(
                                                                result_df,
                                                                kosis_data,
                                                                column_name=col_expense,
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
                                                        # 취업자 근로여건 만족도: 5열
                                                        elif use_work_satisfaction:
                                                            col_ws = ("하는일", "임금/가구소득", "근로시간", "근무환경", "전반적인 만족도")
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
                                                                "소비 경험 여부",
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
                                                                column_names=("임신·출산·육아에 대한 복지", "저소득층 등 취약계층에 대한 복지"), seed=42)
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
                                                                column_names=("동네", "시군", "경상북도"), seed=42)
                                                        elif use_safety_eval:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=(
                                                                    "어둡고 후미진 곳이 많다", "주변에 쓰레기가 아무렇게 버려져 있고 지저분 하다",
                                                                    "주변에 방치된 차나 빈 건물이 많다", "무리 지어 다니는 불량 청소년이 많다",
                                                                    "기초질서(무단횡단, 불법 주정차 등)를 지키지 않는 사람이 많다",
                                                                    "큰소리로 다투거나 싸우는 사람들을 자주 볼 수 있다"), seed=42)
                                                        elif use_crime_fear:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("나자신", "배우자(애인)", "자녀", "부모"), seed=42)
                                                        elif use_daily_fear:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("밤에 혼자 집에 있을 때", "밤에 혼자 지역(동네)의 골목길을 걸을때"), seed=42)
                                                        elif use_law_abiding:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("자신의 평소 준법수준", "평소 법을 지키지 않는 주된 이유"), seed=42)
                                                        elif use_environment_feel:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("대기", "수질", "토양", "소음/진동", "녹지환경"), seed=42)
                                                        elif use_time_pressure:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("평일 생활시간 압박", "주말 생활시간 압박"), seed=42)
                                                        elif use_leisure_satisfaction:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("문화여가시설 만족도", "전반적인 여가활동 만족도", "불만족 이유"), seed=42)
                                                        elif use_culture_attendance:
                                                            result_df, success = kosis.assign_preset_stat_columns(
                                                                result_df, kosis_data, stat_name=stat_info["name"],
                                                                column_names=("관람 여부", "관람 분야"), seed=42)
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
                                                            st.success(f"✅ [{stat_info['category']}] {stat_info['name']} 대입 완료")
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
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["임신·출산·육아에 대한 복지", "저소득층 등 취약계층에 대한 복지"], "is_residence": False, "is_preset": True})
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
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["동네", "시군", "경상북도"], "is_residence": False, "is_preset": True})
                                                            elif use_safety_eval:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["어둡고 후미진 곳이 많다", "주변에 쓰레기가 아무렇게 버려져 있고 지저분 하다", "주변에 방치된 차나 빈 건물이 많다", "무리 지어 다니는 불량 청소년이 많다", "기초질서(무단횡단, 불법 주정차 등)를 지키지 않는 사람이 많다", "큰소리로 다투거나 싸우는 사람들을 자주 볼 수 있다"], "is_residence": False, "is_preset": True})
                                                            elif use_crime_fear:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["나자신", "배우자(애인)", "자녀", "부모"], "is_residence": False, "is_preset": True})
                                                            elif use_daily_fear:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["밤에 혼자 집에 있을 때", "밤에 혼자 지역(동네)의 골목길을 걸을때"], "is_residence": False, "is_preset": True})
                                                            elif use_law_abiding:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["자신의 평소 준법수준", "평소 법을 지키지 않는 주된 이유"], "is_residence": False, "is_preset": True})
                                                            elif use_environment_feel:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["대기", "수질", "토양", "소음/진동", "녹지환경"], "is_residence": False, "is_preset": True})
                                                            elif use_time_pressure:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["평일 생활시간 압박", "주말 생활시간 압박"], "is_residence": False, "is_preset": True})
                                                            elif use_leisure_satisfaction:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["문화여가시설 만족도", "전반적인 여가활동 만족도", "불만족 이유"], "is_residence": False, "is_preset": True})
                                                            elif use_culture_attendance:
                                                                step2_validation_info.append({"stat_name": stat_info.get("name", ""), "url": stat_info.get("url", ""), "columns": ["관람 여부", "관람 분야"], "is_residence": False, "is_preset": True})
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
                                                            st.warning(f"⚠️ [{stat_info['category']}] {stat_info['name']} 대입 실패 (기본값 사용)")
                                                    except Exception as e:
                                                        log_entry["status"] = "error"
                                                        log_entry["message"] = str(e)
                                                        log_entry["error"] = str(e)
                                                        import traceback
                                                        log_entry["traceback"] = traceback.format_exc()
                                                        st.error(f"❌ [{stat_info['category']}] {stat_info['name']} 대입 중 에러: {e}")
                                                    
                                                    st.session_state.stat_assignment_logs.append(log_entry)
                                            
                                            # 행 방향 논리 일관성 정리 (비경제활동 → 직장/직업/근로만족도 비움, 미성년 → 배우자 경제활동 무)
                                            result_df = apply_step2_row_consistency(result_df)
                                            
                                            # 결과 저장 (컬럼 순서: 기본 6축 + 추가 통계가 오른쪽에 열로 이어짐)
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
                                            
                                            # Excel 생성: 기본 컬럼 + 추가 통계를 오른쪽에 열로 이어서 저장
                                            try:
                                                import io
                                                out_buffer = io.BytesIO()
                                                result_df.to_excel(out_buffer, index=False, engine="openpyxl")
                                                out_buffer.seek(0)
                                                st.session_state["generated_excel"] = out_buffer.getvalue()
                                            except Exception as e:
                                                st.warning(f"Excel 저장 실패: {e}")
                                                import traceback
                                                st.code(traceback.format_exc())
                                            
                                            st.success("✅ 2단계 완료: 통계 대입 완료!")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"통계 대입 실패: {e}")
                                            import traceback
                                            st.code(traceback.format_exc())
                        
                        with col_cancel:
                            if st.button("❌ 취소", use_container_width=True):
                                st.session_state["show_step2_dialog"] = False
                                st.rerun()
        
        st.markdown("---")
        
        # 탭 구성: 요약, 그래프, 검증, 데이터, 다운로드 (2단계 완료 시에도 동일, df가 step2_df를 포함)
        tabs = st.tabs(["요약", "그래프", "검증", "데이터", "다운로드"])
        
        # [탭 0] 요약 — 2단계 완료 시 추가 통계만, 아니면 1단계 요약
        with tabs[0]:
            if is_step2:
                # 2단계: 추가 통계 데이터만 요약 (1단계 항목 제거)
                st.markdown("### 📈 2단계 추가 통계 요약")
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
        
        # [탭 1] 그래프 — 2단계 완료 시 추가 통계만, 아니면 6축
        with tabs[1]:
            if is_step2:
                step2_cols = [c for c in (st.session_state.get("step2_added_columns") or []) if c in df.columns]
                if step2_cols:
                    st.markdown("### 📊 2단계 추가 통계 분포")
                    draw_charts(df, step2_columns=step2_cols, step2_only=True)
                else:
                    st.info("추가된 통계 컬럼이 없습니다.")
            else:
                draw_charts(df, step2_columns=None)
        
        # [탭 2] 검증 — 2단계 완료 시 KOSIS 대비 검증(1단계와 동일 방식), 아니면 1단계 6축 검증
        with tabs[2]:
            if is_step2:
                step2_val_info = st.session_state.get("step2_validation_info") or []
                if step2_val_info:
                    st.markdown("### 📊 2단계 추가 통계 · KOSIS 대비 검증")
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
        tab_data_idx = 3
        with tabs[tab_data_idx]:
            if is_step2:
                st.markdown("### 📋 생성된 데이터 미리보기 (2단계 완료: 추가 통계 포함)")
            else:
                st.markdown("### 📋 생성된 데이터 미리보기")
            
            # 데이터프레임 컬럼 순서 그대로 표시 (기본 컬럼 + 추가 통계가 오른쪽에 열로 이어짐)
            df_preview = df
            st.dataframe(df_preview.head(100), height=400, use_container_width=True)
        
        # [탭 4] 다운로드
        tab_download_idx = 4
        with tabs[tab_download_idx]:
            st.markdown("### 📥 다운로드")
            
            # Excel: 데이터프레임 컬럼 순서대로 1행 헤더 + 데이터 (추가 통계는 오른쪽에 열로 이어짐)
            final_excel_bytes = None
            try:
                import io
                buf = io.BytesIO()
                df.to_excel(buf, index=False, engine="openpyxl")
                buf.seek(0)
                final_excel_bytes = buf.getvalue()
                st.session_state["generated_excel"] = final_excel_bytes
            except Exception as e:
                st.warning(f"Excel 생성 실패: {e}")
                final_excel_bytes = st.session_state.get("generated_excel")
            
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
                    csv = df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "CSV 다운로드",
                        data=csv,
                        file_name=f"{sido_name}_synthetic_population{file_suffix}.csv",
                        mime="text/csv",
                    )
            else:
                st.info("Excel 파일이 생성되지 않았습니다.")


def page_work_log():
    """작업 기록 페이지 - 디버깅 및 검증용"""
    st.header("📋 작업 기록")
    
    # 세션에서 로그 가져오기
    if "work_logs" not in st.session_state:
        st.session_state.work_logs = []
    
    if len(st.session_state.work_logs) == 0:
        st.info("아직 기록된 작업이 없습니다. 생성 버튼을 클릭하면 기록이 남습니다.")
        return
    
    st.success(f"총 {len(st.session_state.work_logs)}개의 작업 기록")
    
    # 최신순으로 표시
    for idx, log in enumerate(reversed(st.session_state.work_logs[-20:])):
        timestamp = log.get("timestamp", "N/A")
        stage = log.get("stage", "Unknown")
        status = log.get("status", "unknown")
        
        # 상태별 이모지
        if status == "success":
            emoji = "✅"
        elif status == "error":
            emoji = "❌"
        elif status == "exception":
            emoji = "🚨"
        else:
            emoji = "⏳"
        
        with st.expander(f"{emoji} {timestamp} - {stage}"):
            st.json(log)
    
    # 로그 초기화 버튼
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🗑️ 로그 삭제", key="work_log_delete"):
            st.session_state.work_logs = []
            st.rerun()


def page_stat_assignment_log():
    """통계 대입 로그 페이지 - 2단계 통계 대입 시 발생하는 상세 로그"""
    st.header("📊 통계 대입 로그")
    
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
    st.markdown("### 📈 통계별 요약")
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
    st.markdown("### 📋 상세 로그 (최신순)")
    for idx, log in enumerate(reversed(st.session_state.stat_assignment_logs[-50:])):
        timestamp = log.get("timestamp", "N/A")
        category = log.get("category", "N/A")
        stat_name = log.get("stat_name", "N/A")
        status = log.get("status", "unknown")
        
        # 상태별 이모지 및 색상
        if status == "success":
            emoji = "✅"
            color = "green"
        elif status == "warning":
            emoji = "⚠️"
            color = "orange"
        elif status == "error":
            emoji = "❌"
            color = "red"
        else:
            emoji = "⏳"
            color = "gray"
        
        with st.expander(f"{emoji} {timestamp} - [{category}] {stat_name}", expanded=False):
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
                st.markdown("**📊 가져온 KOSIS 데이터 샘플**")
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
    
    # 로그 초기화 버튼
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🗑️ 로그 삭제", key="stat_assignment_log_delete"):
            st.session_state.stat_assignment_logs = []
            st.rerun()


def page_guide():
    st.header("ℹ️ 사용 가이드")
    st.markdown("""
    ### 사용 순서
    1. **데이터 관리 탭**: 템플릿 업로드 및 6축 마진 소스 설정
    2. **생성 탭**: 생성 옵션 설정 후 생성 버튼 클릭
    3. **작업 기록 탭**: 디버깅 및 검증용 로그 확인
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
    
    /* 버튼 스타일 */
    .stButton > button {
        border-radius: 24px;
        font-weight: 800;
        font-size: 18px;
        padding: 20px 30px;
        transition: all 0.3s;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(79, 70, 229, 0.3);
    }
    
    /* 카드 스타일 */
    .survey-card {
        background: white;
        border-radius: 24px;
        padding: 32px;
        border: 1px solid #e0e7ff;
        box-shadow: 0 20px 25px -5px rgba(79, 70, 229, 0.1);
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
    
    # 헤더
    col_header1, col_header2 = st.columns([1, 4])
    with col_header1:
        st.markdown("""
        <div style="width: 40px; height: 40px; background: #4f46e5; border-radius: 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3);">
            <span style="color: white; font-size: 20px;">✨</span>
        </div>
        """, unsafe_allow_html=True)
    with col_header2:
        st.markdown("""
        <h1 style="font-size: 28px; font-weight: 800; color: #0f172a; margin: 0; padding-top: 8px;">
            AI Lab. Insight
        </h1>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 메인 타이틀
    st.markdown("""
    <div class="survey-header">
        <h1 style="font-size: 32px; font-weight: 800; color: #0f172a; margin-bottom: 12px;">
            새로운 조사 프로젝트 시작하기
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
        st.markdown("### 📝 제품/서비스의 정의 <span style='color: #ef4444;'>*</span>", unsafe_allow_html=True)
        
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
        st.markdown("### 🎯 조사의 목적과 니즈 <span style='color: #ef4444;'>*</span>", unsafe_allow_html=True)
        
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
        with st.expander("⚙️ 추가 정보 (선택)", expanded=False):
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
            <span style="font-size: 20px;">✨</span>
            <span style="color: #4f46e5; font-weight: 700; font-size: 16px;">AI 최적화 설계 제안</span>
        </div>
        """, unsafe_allow_html=True)
        
        if not is_all_valid:
            st.warning("⚠️ 필수 정보를 300자 이상 입력하시면 AI가 최적의 조사 설계를 제안합니다.")
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
            💡 **AI 코멘트**
            
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
            if st.button("🚀 조사 시작하기", type="primary", use_container_width=True, key="survey_start_button"):
                survey_type = st.session_state.get("survey_type", "Talk")
                sample_size = st.session_state.get("survey_sample_size", 2500)
                target = st.session_state.survey_target if st.session_state.survey_target else "전체"
                
                st.success("✅ 조사 프로젝트가 시작되었습니다!")
                st.info(f"""
                **설정된 조사 정보:**
                - 조사 방식: {survey_type}
                - 표본 수: {sample_size:,}명
                - 타깃: {target}
                """)
        else:
            st.button("🚀 조사 시작하기", disabled=True, use_container_width=True, key="survey_start_button_disabled")
            st.caption("필수 정보를 300자 이상 입력해주세요.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 정보 안내
        st.caption("ℹ️ 데이터는 암호화되어 보호되며, 분석 완료 후 즉시 파기됩니다.")


def page_survey_results():
    st.subheader("설문 결과")
    st.info("설문 결과 기능은 추후 확장 범위")


# -----------------------------
# 9. Main UI (상위 폴더 구조)
# -----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    db_init()
    ensure_session_state()

    with st.sidebar:
        st.markdown("## 내비게이션")
        st.markdown("**AI Social Twin**")
        st.caption("가상인구 생성, 설문조사, 심층면접")
        st.markdown("**Utils**")
        st.caption("사진 배경제거, 사진 옷 변경")
        st.markdown("**기타**")
        st.caption("생성 결과 목록, Conjoint Analysis")
        st.markdown("---")
        all_options = [
            "🧬 가상인구 생성",
            "📂 생성 결과 목록 (Results Archive)",
            "📝 설문조사 (Survey)",
            "💬 심층면접 (Interview)",
            "🖼️ 사진 배경제거",
            "👔 사진 옷 변경",
            "📊 Conjoint Analysis Pro",
        ]
        selected = st.radio("페이지 선택", options=all_options, index=0, key="nav_main", label_visibility="collapsed")

    if selected == "🧬 가상인구 생성":
        sub = st.tabs(["데이터 관리", "생성", "📋 작업 기록", "📊 통계 대입 로그"])
        with sub[0]:
            page_data_management()
        with sub[1]:
            page_generate()
        with sub[2]:
            page_work_log()
        with sub[3]:
            page_stat_assignment_log()

    elif selected == "📂 생성 결과 목록 (Results Archive)":
        from pages.results_archive import page_results_archive
        page_results_archive()

    elif selected == "📝 설문조사 (Survey)":
        from pages.survey import page_survey
        page_survey()

    elif selected == "💬 심층면접 (Interview)":
        from pages.interview import page_interview
        page_interview()

    elif selected == "🖼️ 사진 배경제거":
        try:
            from pages.utils_background_removal import page_photo_background_removal
            page_photo_background_removal()
        except Exception as e:
            st.markdown("## 🖼️ 사진 배경제거")
            st.warning("이 페이지는 JavaScript/Streamlit 모듈로 구성됩니다. 현재 `pages/utils_background_removal.py` 가 Python 모듈이 아닌 경우 동작하지 않습니다.")
            st.caption(str(e))

    elif selected == "👔 사진 옷 변경":
        try:
            from pages.utils_clothing_change import page_photo_clothing_change
            page_photo_clothing_change()
        except Exception as e:
            st.markdown("## 👔 사진 옷 변경")
            st.warning("이 페이지는 JavaScript/Streamlit 모듈로 구성됩니다. 현재 `pages/utils_clothing_change.py` 가 Python 모듈이 아닌 경우 동작하지 않습니다.")
            st.caption(str(e))

    elif selected == "📊 Conjoint Analysis Pro":
        st.markdown("## 📊 Conjoint Analysis Pro")
        st.info(
            "**Conjoint Analysis Pro**는 React 기반 프론트엔드 앱입니다. "
            "이 기능을 사용하려면 별도의 React 개발 서버에서 실행해 주세요."
        )
        st.markdown("""
        - `pages/Conjoint Analysis Pro.py` 는 React/TypeScript 소스입니다.
        - React 앱으로 실행: 해당 프로젝트 폴더에서 `npm install` 후 `npm start` 로 실행할 수 있습니다.
        """)
        st.caption("콘조인트 분석 · OLS 회귀 기반 마케팅 인텔리전스 도구")


if __name__ == "__main__":
    main()
