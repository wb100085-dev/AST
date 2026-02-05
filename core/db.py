"""
AI Social Twin - DB 레이어 (Supabase PostgreSQL)
로컬/배포 환경 간 데이터 공유를 위해 Supabase 사용.
"""
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO

import pandas as pd
import streamlit as st

from core.constants import EXPORT_SHEET_NAME, EXPORT_COLUMNS
from core.supabase_client import get_supabase


def db_conn():
    """Supabase 사용 시 호환용. get_supabase() 반환 (cursor 없음)."""
    return get_supabase()


def delete_virtual_population_db_by_sido(sido_code: str) -> bool:
    """해당 시도 가상인구 DB 전체 삭제(초기화)."""
    try:
        sb = get_supabase()
        sb.table("virtual_population_db").delete().eq("sido_code", sido_code).execute()
        return True
    except Exception:
        return False


def db_init():
    """Supabase 연결만 검증(즉시 연결 시도). 테이블은 docs/SUPABASE_CREATE_TABLES.sql 로 생성."""
    get_supabase()


@st.cache_data(ttl=60)
def db_list_stats_by_sido(sido_code: str, active_only: bool = False) -> List[Dict[str, Any]]:
    sb = get_supabase()
    q = sb.table("stats").select("*")
    if sido_code != "00":
        q = q.eq("sido_code", sido_code)
    if active_only:
        q = q.eq("is_active", 1)
    q = q.order("sido_code").order("category").order("name")
    r = q.execute()
    rows = r.data or []
    return [dict(x) for x in rows]


def db_upsert_stat(sido_code: str, category: str, name: str, url: str, is_active: int = 1):
    sb = get_supabase()
    category = str(category).strip()
    name = str(name).strip()
    url = str(url).strip()
    is_active = 1 if int(is_active) != 0 else 0
    existing = sb.table("stats").select("id").eq("sido_code", sido_code).eq("category", category).eq("name", name).limit(1).execute()
    if existing.data and len(existing.data) > 0:
        sb.table("stats").update({"url": url, "is_active": is_active, "updated_at": datetime.utcnow().isoformat()}).eq("id", existing.data[0]["id"]).execute()
    else:
        sb.table("stats").insert({"sido_code": sido_code, "category": category, "name": name, "url": url, "is_active": is_active}).execute()


def db_delete_stat_by_id(stat_id: int):
    get_supabase().table("stats").delete().eq("id", int(stat_id)).execute()


def db_update_stat_by_id(stat_id: int, category: str, name: str, url: str, is_active: int):
    get_supabase().table("stats").update({
        "category": category.strip(),
        "name": name.strip(),
        "url": url.strip(),
        "is_active": int(is_active),
        "updated_at": datetime.utcnow().isoformat(),
    }).eq("id", int(stat_id)).execute()


def db_get_axis_margin_stats(sido_code: str, axis_key: str) -> Optional[Dict[str, Any]]:
    """6축 마진 설정 조회. 캐시 없이 매번 DB 조회 (저장 후 즉시 반영)."""
    r = get_supabase().table("sido_axis_margin_stats").select("*").eq("sido_code", sido_code).eq("axis_key", axis_key).limit(1).execute()
    if r.data and len(r.data) > 0:
        return dict(r.data[0])
    return None


def db_get_six_axis_stat_ids(sido_code: str) -> set:
    axis_keys = ["sigungu", "gender", "age", "econ", "income", "edu"]
    ids = set()
    for axis_key in axis_keys:
        row = db_get_axis_margin_stats(sido_code, axis_key)
        if row and row.get("stat_id") is not None:
            ids.add(int(row["stat_id"]))
    return ids


def db_upsert_axis_margin_stat(sido_code: str, axis_key: str, stat_id: int):
    """6축 마진 통계 소스 저장. 실패 시 예외 발생."""
    sb = get_supabase()
    existing = sb.table("sido_axis_margin_stats").select("id").eq("sido_code", sido_code).eq("axis_key", axis_key).limit(1).execute()
    payload = {"sido_code": str(sido_code), "axis_key": str(axis_key), "stat_id": int(stat_id)}
    if existing.data and len(existing.data) > 0:
        sb.table("sido_axis_margin_stats").update({**payload, "updated_at": datetime.utcnow().isoformat()}).eq("id", existing.data[0]["id"]).execute()
    else:
        sb.table("sido_axis_margin_stats").insert(payload).execute()


def db_upsert_template(sido_code: str, filename: str, file_bytes: bytes):
    """템플릿 저장/업데이트. file_blob는 base64 TEXT로 저장."""
    b64 = base64.b64encode(file_bytes).decode("ascii")
    sb = get_supabase()
    existing = sb.table("sido_templates").select("sido_code").eq("sido_code", sido_code).limit(1).execute()
    row = {"sido_code": sido_code, "filename": filename, "file_blob": b64}
    if existing.data and len(existing.data) > 0:
        sb.table("sido_templates").update({"filename": filename, "file_blob": b64, "updated_at": datetime.utcnow().isoformat()}).eq("sido_code", sido_code).execute()
    else:
        sb.table("sido_templates").insert(row).execute()


@st.cache_data(ttl=60)
def db_get_template(sido_code: str) -> Optional[Dict[str, Any]]:
    r = get_supabase().table("sido_templates").select("filename, file_blob, updated_at").eq("sido_code", sido_code).limit(1).execute()
    if not r.data or len(r.data) == 0:
        return None
    row = r.data[0]
    try:
        file_bytes = base64.b64decode(row["file_blob"])
    except Exception:
        return None
    return {"filename": row["filename"], "bytes": file_bytes, "updated_at": row["updated_at"]}


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


def delete_virtual_population_db_record(record_id: int) -> bool:
    try:
        r = get_supabase().table("virtual_population_db").delete().eq("id", int(record_id)).execute()
        return True
    except Exception:
        return False


def list_virtual_population_db_records_all(sido_code: str) -> List[Dict[str, Any]]:
    r = get_supabase().table("virtual_population_db").select("id, sido_code, sido_name, record_timestamp, record_excel_path, data_json").eq("sido_code", sido_code).order("added_at").execute()
    out = []
    for row in (r.data or []):
        try:
            df = pd.read_json(row["data_json"], orient="records")
            n_rows = len(df)
        except Exception:
            n_rows = 0
        out.append({
            "id": row["id"],
            "sido_code": row["sido_code"],
            "sido_name": row["sido_name"],
            "record_timestamp": row["record_timestamp"],
            "record_excel_path": row.get("record_excel_path") or "",
            "rows": n_rows,
        })
    return out


def list_virtual_population_db_data_jsons(sido_code: str) -> List[Tuple[int, str]]:
    """해당 시도 가상인구 DB 레코드의 (id, data_json) 목록 반환 (added_at 순)."""
    r = get_supabase().table("virtual_population_db").select("id, data_json").eq("sido_code", sido_code).order("added_at").execute()
    return [(row["id"], row["data_json"]) for row in (r.data or [])]


@st.cache_data(ttl=60, show_spinner=False)
def list_virtual_population_db_records(sido_code: Optional[str] = None) -> List[Dict[str, Any]]:
    sb = get_supabase()
    q = sb.table("virtual_population_db").select("id, sido_code, sido_name, record_timestamp, record_excel_path, data_json")
    if sido_code:
        q = q.eq("sido_code", sido_code)
    q = q.order("added_at")
    r = q.execute()
    out = []
    for row in (r.data or []):
        if not _vpd_record_has_persona_and_contemporary(row["data_json"]):
            continue
        try:
            df = pd.read_json(row["data_json"], orient="records")
            out.append({
                "id": row["id"],
                "sido_code": row["sido_code"],
                "sido_name": row["sido_name"],
                "record_timestamp": row["record_timestamp"],
                "record_excel_path": row.get("record_excel_path") or "",
                "rows": len(df),
                "columns_count": len(df.columns),
            })
        except Exception:
            continue
    return out


@st.cache_data(ttl=60, show_spinner=False)
def get_virtual_population_db_record_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    r = get_supabase().table("virtual_population_db").select("*").eq("id", int(record_id)).limit(1).execute()
    if not r.data or len(r.data) == 0:
        return None
    row = r.data[0]
    try:
        df = pd.read_json(row["data_json"], orient="records")
        return {
            "id": row["id"],
            "sido_code": row["sido_code"],
            "sido_name": row["sido_name"],
            "record_timestamp": row["record_timestamp"],
            "record_excel_path": row.get("record_excel_path"),
            "df": df.copy(),
            "rows": len(df),
            "columns_count": len(df.columns),
        }
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def get_virtual_population_db_combined_df(sido_code: str):
    r = get_supabase().table("virtual_population_db").select("id, sido_code, sido_name, record_timestamp, data_json").eq("sido_code", sido_code).order("added_at").execute()
    dfs = []
    for row in (r.data or []):
        if not _vpd_record_has_persona_and_contemporary(row["data_json"]):
            continue
        try:
            df = pd.read_json(row["data_json"], orient="records")
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
    if "페르소나" in combined.columns and "현시대 반영" in combined.columns:
        persona_ok = combined["페르소나"].fillna("").astype(str).str.strip() != ""
        cont_ok = combined["현시대 반영"].fillna("").astype(str).str.strip() != ""
        combined = combined.loc[persona_ok & cont_ok].reset_index(drop=True)
    if combined.empty:
        return None
    return combined.copy()


@st.cache_data(ttl=60, show_spinner=False)
def get_sido_vdb_stats() -> Dict[str, Dict[str, Any]]:
    r = get_supabase().table("virtual_population_db").select("sido_code, sido_name, data_json").execute()
    out = {}
    for row in (r.data or []):
        sido_code = row["sido_code"]
        if sido_code == "00":
            continue
        try:
            df = pd.read_json(row["data_json"], orient="records")
            n = len(df)
        except Exception:
            n = 0
        if sido_code not in out:
            out[sido_code] = {"sido_name": row["sido_name"], "vdb_count": 0}
        out[sido_code]["vdb_count"] = out[sido_code]["vdb_count"] + n
    return out


def insert_virtual_population_db(sido_code: str, sido_name: str, record_timestamp: str, record_excel_path: str, data_json: str) -> Optional[int]:
    """가상인구 DB 한 건 INSERT. 반환: id 또는 None."""
    try:
        r = get_supabase().table("virtual_population_db").insert({
            "sido_code": sido_code,
            "sido_name": sido_name,
            "record_timestamp": record_timestamp,
            "record_excel_path": record_excel_path,
            "data_json": data_json,
        }).execute()
        if r.data and len(r.data) > 0:
            return r.data[0].get("id")
    except Exception:
        pass
    return None


def update_virtual_population_db_data_json(record_id: int, data_json: str):
    """virtual_population_db 레코드의 data_json만 UPDATE."""
    get_supabase().table("virtual_population_db").update({"data_json": data_json}).eq("id", int(record_id)).execute()


def get_virtual_population_db_data_json(record_id: int) -> Optional[str]:
    """virtual_population_db 레코드의 data_json만 조회."""
    r = get_supabase().table("virtual_population_db").select("data_json").eq("id", int(record_id)).limit(1).execute()
    if r.data and len(r.data) > 0:
        return r.data[0].get("data_json")
    return None


def find_virtual_population_db_id(sido_code: str, record_timestamp: str, record_excel_path: str) -> Optional[int]:
    """sido_code, record_timestamp, record_excel_path 로 id 조회."""
    r = get_supabase().table("virtual_population_db").select("id").eq("sido_code", sido_code).eq("record_timestamp", record_timestamp).eq("record_excel_path", record_excel_path).limit(1).execute()
    if r.data and len(r.data) > 0:
        return r.data[0].get("id")
    return None
