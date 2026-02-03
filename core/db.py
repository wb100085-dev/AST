"""
AI Social Twin - DB 레이어 및 Excel round-trip
app.py에서 분리
"""
import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
from io import BytesIO

import pandas as pd
import streamlit as st

from core.constants import (
    DB_PATH,
    EXPORT_SHEET_NAME,
    EXPORT_COLUMNS,
)


def db_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
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

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS virtual_population_db (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sido_code TEXT NOT NULL,
            sido_name TEXT NOT NULL,
            record_timestamp TEXT NOT NULL,
            record_excel_path TEXT NOT NULL,
            data_json TEXT NOT NULL,
            added_at TEXT DEFAULT (datetime('now'))
        )
        """
    )

    # === 성능: stats 조회(WHERE/ORDER BY)용 인덱스 ===
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_stats_is_active_order
        ON stats(is_active, sido_code, category, name)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_stats_sido_active_cat
        ON stats(sido_code, is_active, category, name)
        """
    )

    # === 성능: virtual_population_db 조회(WHERE/ORDER BY)용 인덱스 (pages에서 사용) ===
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_vpd_sido_added
        ON virtual_population_db(sido_code, added_at)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_vpd_sido_ts_path
        ON virtual_population_db(sido_code, record_timestamp, record_excel_path)
        """
    )

    conn.commit()
    conn.close()


@st.cache_data(ttl=60)
def db_list_stats_by_sido(sido_code: str, active_only: bool = False) -> List[Dict[str, Any]]:
    conn = db_conn()
    cur = conn.cursor()
    # 모든 분기는 동등 조건(=)만 사용 → 인덱스 활용 가능. LIKE/함수 래핑 없음.
    if sido_code == "00":
        if active_only:
            # idx_stats_is_active_order (is_active, sido_code, category, name)
            cur.execute("SELECT * FROM stats WHERE is_active=1 ORDER BY sido_code, category, name")
        else:
            # ux_stats_sido_category_name (sido_code, category, name)
            cur.execute("SELECT * FROM stats ORDER BY sido_code, category, name")
    else:
        if active_only:
            # idx_stats_sido_active_cat (sido_code, is_active, category, name)
            cur.execute(
                "SELECT * FROM stats WHERE sido_code=? AND is_active=1 ORDER BY category, name",
                (sido_code,),
            )
        else:
            # ux_stats_sido_category_name (sido_code, category, name)
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


@st.cache_data(ttl=60)
def db_get_axis_margin_stats(sido_code: str, axis_key: str) -> Optional[Dict[str, Any]]:
    """특정 시도/축의 마진 통계 소스 조회"""
    conn = db_conn()
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


@st.cache_data(ttl=60)
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


def db_upsert_template(sido_code: str, filename: str, file_bytes: bytes):
    """템플릿 저장/업데이트"""
    conn = db_conn()
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


@st.cache_data(ttl=60)
def db_get_template(sido_code: str) -> Optional[Dict[str, Any]]:
    """템플릿 조회"""
    conn = db_conn()
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
            "filename": row["filename"],
            "bytes": row["file_blob"],
            "updated_at": row["updated_at"],
        }
    return None


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

    try:
        from core.session_cache import invalidate_db_stats_cache
        invalidate_db_stats_cache(sido_code)
    except Exception:
        pass
    return {"ok": True, "반영건수": reflected, "스킵건수": skipped, "오류건수": len(errors), "오류상세": errors, "총행수": len(xdf)}


def _vpd_record_has_persona_and_contemporary(data_json: bytes | str) -> bool:
    """data_json에 페르소나·현시대 반영이 한 건 이상 있는지 여부."""
    try:
        data = pd.read_json(data_json, orient="records") if isinstance(data_json, str) else pd.read_json(BytesIO(data_json), orient="records")
        if data.empty:
            return False
        for col in ("페르소나", "현시대 반영"):
            if col not in data.columns:
                return False
        persona = data["페르소나"].fillna("").astype(str).str.strip()
        cont = data["현시대 반영"].fillna("").astype(str).str.strip()
        return ((persona != "") & (cont != "")).any()
    except Exception:
        return False


def list_virtual_population_db_records(sido_code: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    가상인구 DB에서 페르소나·현시대 반영이 된 목록만 반환.
    sido_code가 있으면 해당 시도만, 없으면 전부.
    """
    conn = db_conn()
    cur = conn.cursor()
    if sido_code:
        cur.execute(
            "SELECT id, sido_code, sido_name, record_timestamp, record_excel_path, data_json FROM virtual_population_db WHERE sido_code = ? ORDER BY added_at",
            (sido_code,),
        )
    else:
        cur.execute(
            "SELECT id, sido_code, sido_name, record_timestamp, record_excel_path, data_json FROM virtual_population_db ORDER BY added_at",
        )
    rows = cur.fetchall()
    conn.close()

    out = []
    for row in rows:
        id_, sc, sido_name, record_ts, excel_path, data_json = row[0], row[1], row[2], row[3], row[4], row[5]
        if not _vpd_record_has_persona_and_contemporary(data_json):
            continue
        try:
            df = pd.read_json(data_json, orient="records")
            out.append({
                "id": id_,
                "sido_code": sc,
                "sido_name": sido_name,
                "record_timestamp": record_ts,
                "record_excel_path": excel_path or "",
                "rows": len(df),
                "columns_count": len(df.columns),
            })
        except Exception:
            continue
    return out


def get_virtual_population_db_record_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """가상인구 DB에서 id로 한 건 조회. DataFrame과 메타 반환."""
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, sido_code, sido_name, record_timestamp, record_excel_path, data_json FROM virtual_population_db WHERE id = ?",
        (int(record_id),),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    try:
        df = pd.read_json(row[5], orient="records")
        return {
            "id": row[0],
            "sido_code": row[1],
            "sido_name": row[2],
            "record_timestamp": row[3],
            "record_excel_path": row[4],
            "df": df,
            "rows": len(df),
            "columns_count": len(df.columns),
        }
    except Exception:
        return None


def get_virtual_population_db_combined_df(sido_code: str):
    """
    해당 시도의 가상인구 DB 중 페르소나·현시대 반영된 기록을 모두 합쳐 하나의 DataFrame으로 반환.
    없으면 None.
    """
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, sido_code, sido_name, record_timestamp, data_json FROM virtual_population_db WHERE sido_code = ? ORDER BY added_at",
        (sido_code,),
    )
    rows = cur.fetchall()
    conn.close()

    dfs = []
    for row in rows:
        data_json = row[4]
        if not _vpd_record_has_persona_and_contemporary(data_json):
            continue
        try:
            df = pd.read_json(data_json, orient="records")
            if "식별NO" in df.columns:
                df = df.drop(columns=["식별NO"])
            if "페르소나" not in df.columns:
                df["페르소나"] = ""
            if "현시대 반영" not in df.columns:
                df["현시대 반영"] = ""
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    # 행 단위: 페르소나·현시대 반영이 모두 채워진 행만 포함
    if "페르소나" in combined.columns and "현시대 반영" in combined.columns:
        persona_ok = combined["페르소나"].fillna("").astype(str).str.strip() != ""
        cont_ok = combined["현시대 반영"].fillna("").astype(str).str.strip() != ""
        combined = combined.loc[persona_ok & cont_ok].reset_index(drop=True)
    if combined.empty:
        return None
    return combined


def get_sido_vdb_stats() -> Dict[str, Dict[str, Any]]:
    """
    시도별 가상인구 DB 통계. 전국(00) 제외.
    반환: { sido_code: { "sido_name", "vdb_count" } }
    vdb_count = 해당 시도 virtual_population_db의 모든 기록에서의 총 인원 수(행 수 합계).
    """
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT sido_code, sido_name, data_json FROM virtual_population_db"
    )
    rows = cur.fetchall()
    conn.close()

    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sido_code = row[0]
        sido_name = row[1]
        if sido_code == "00":
            continue
        try:
            df = pd.read_json(row[2], orient="records")
            n = len(df)
        except Exception:
            n = 0
        if sido_code not in out:
            out[sido_code] = {"sido_name": sido_name, "vdb_count": 0}
        out[sido_code]["vdb_count"] = out[sido_code]["vdb_count"] + n
    return out
