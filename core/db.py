"""
AI Social Twin - DB 레이어 (Supabase PostgreSQL)
로컬/배포 환경 간 데이터 공유를 위해 Supabase 사용.
대용량 data_json은 statement timeout 방지를 위해 청크(chunk) 단위로 나누어 INSERT.
"""
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable
from io import BytesIO

import pandas as pd
import streamlit as st

from core.constants import EXPORT_SHEET_NAME, EXPORT_COLUMNS
from core.supabase_client import get_supabase

# 청크 단위 배치 크기 (한 번에 INSERT할 행 수)
VPD_CHUNK_SIZE = 500
# SELECT 시 statement timeout 방지를 위한 페이지 크기
VPD_FETCH_PAGE_SIZE = 50  # 메타만 조회 시 (data_json 미포함)
VPD_FETCH_PAGE_SIZE_HEAVY = 10  # data_json 포함 조회 시 (한 번에 10행만)


def _vpd_fetch_all_rows_paginated(
    columns: str,
    sido_code: Optional[str] = None,
    page_size: int = VPD_FETCH_PAGE_SIZE,
) -> List[Dict[str, Any]]:
    """virtual_population_db를 페이지 단위로 조회해 전체 행 리스트 반환 (statement timeout 방지)."""
    sb = get_supabase()
    all_rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        q = sb.table("virtual_population_db").select(columns)
        if sido_code is not None:
            q = q.eq("sido_code", sido_code).order("added_at").order("chunk_index")
        else:
            q = q.order("sido_code").order("added_at").order("chunk_index")
        q = q.range(offset, offset + page_size - 1)
        r = q.execute()
        data = r.data or []
        if not data:
            break
        all_rows.extend(data)
        if len(data) < page_size:
            break
        offset += page_size
    return all_rows


def _vpd_merge_chunks(rows: List[Dict[str, Any]]) -> Tuple[str, int]:
    """chunk_index 순으로 정렬 후 data_json 배열들을 병합. (merged_json_string, total_rows) 반환."""
    if not rows:
        return "[]", 0
    sorted_rows = sorted(rows, key=lambda r: (r.get("added_at") or "", r.get("chunk_index", 0)))
    merged = []
    for r in sorted_rows:
        raw = (r.get("data_json") or "").strip()
        if not raw:
            continue
        try:
            part = json.loads(raw)
            if isinstance(part, list):
                merged.extend(part)
            else:
                merged.append(part)
        except Exception:
            continue
    return json.dumps(merged, ensure_ascii=False), len(merged)


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
    """통계 목록 조회. 네트워크 부담 감소를 위해 필요한 컬럼만 select."""
    sb = get_supabase()
    q = sb.table("stats").select("id, sido_code, category, name, url, is_active")
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


def db_get_all_axis_margin_stats(sido_code: str) -> Dict[str, Dict[str, Any]]:
    """해당 시도의 6축 마진 설정을 한 번의 쿼리로 모두 조회. axis_key -> row 딕셔너리 반환."""
    r = get_supabase().table("sido_axis_margin_stats").select("id, sido_code, axis_key, stat_id, created_at, updated_at").eq("sido_code", sido_code).execute()
    rows = r.data or []
    return {str(row["axis_key"]): dict(row) for row in rows}


def db_get_six_axis_stat_ids(sido_code: str) -> set:
    """6축 마진 설정에서 stat_id 집합 반환. 한 번의 쿼리로 조회."""
    axis_map = db_get_all_axis_margin_stats(sido_code)
    ids = set()
    for row in axis_map.values():
        if row.get("stat_id") is not None:
            try:
                ids.add(int(row["stat_id"]))
            except (TypeError, ValueError):
                pass
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
        raw = data_json.decode("utf-8") if isinstance(data_json, bytes) else (data_json or "")
        data = pd.read_json(raw, orient="records")
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


def _vpd_get_chunks_for_record(sido_code: str, record_timestamp: str, record_excel_path: str) -> List[Dict[str, Any]]:
    """동일 (sido_code, record_timestamp, record_excel_path) 의 모든 청크 행 조회 (chunk_index 순)."""
    r = get_supabase().table("virtual_population_db").select("id, sido_code, sido_name, record_timestamp, record_excel_path, chunk_index, added_at, data_json").eq("sido_code", sido_code).eq("record_timestamp", record_timestamp).eq("record_excel_path", record_excel_path).order("added_at").order("chunk_index").execute()
    return list(r.data or [])


def _vpd_get_chunk_ids_for_record(sido_code: str, record_timestamp: str, record_excel_path: str) -> List[Dict[str, Any]]:
    """동일 레코드의 청크 id·chunk_index만 경량 조회 (data_json 미포함)."""
    r = get_supabase().table("virtual_population_db").select("id, chunk_index").eq("sido_code", sido_code).eq("record_timestamp", record_timestamp).eq("record_excel_path", record_excel_path).order("chunk_index").execute()
    return sorted(list(r.data or []), key=lambda x: x.get("chunk_index", 0))


def delete_virtual_population_db_by_record(sido_code: str, record_timestamp: str, record_excel_path: str) -> bool:
    """해당 논리적 레코드(동일 record_timestamp + record_excel_path)의 모든 청크 삭제."""
    try:
        get_supabase().table("virtual_population_db").delete().eq("sido_code", sido_code).eq("record_timestamp", record_timestamp).eq("record_excel_path", record_excel_path).execute()
        return True
    except Exception:
        return False


def delete_virtual_population_db_record(record_id: int) -> bool:
    """id로 한 행 조회 후, 동일 논리적 레코드(모든 청크) 삭제."""
    try:
        r = get_supabase().table("virtual_population_db").select("sido_code, record_timestamp, record_excel_path").eq("id", int(record_id)).limit(1).execute()
        if not r.data or len(r.data) == 0:
            return False
        row = r.data[0]
        return delete_virtual_population_db_by_record(row["sido_code"], row["record_timestamp"], row["record_excel_path"])
    except Exception:
        return False


def _vpd_group_metadata_only(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """data_json 없이 (record_timestamp, record_excel_path) 별로 그룹만. merged_json/row_count 없음 (경량 조회용)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in (rows or []):
        key = (r.get("record_timestamp"), r.get("record_excel_path") or "")
        groups[key].append(r)
    out = []
    for key, group_rows in groups.items():
        first = group_rows[0]
        sorted_rows = sorted(group_rows, key=lambda x: (x.get("added_at") or "", x.get("chunk_index", 0)))
        chunk_ids = [x.get("id") for x in sorted_rows if x.get("id") is not None]
        out.append({
            "id": first.get("id"),
            "chunk_ids": chunk_ids,
            "sido_code": first.get("sido_code"),
            "sido_name": first.get("sido_name"),
            "record_timestamp": key[0],
            "record_excel_path": key[1],
            "chunk_count": len(group_rows),
        })
    return out


def _vpd_group_and_merge(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """(record_timestamp, record_excel_path) 별로 그룹 후 청크 병합. 각 그룹당 한 요소: id(첫 청크), chunk_ids, merged_json, row_count."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in (rows or []):
        key = (r.get("record_timestamp"), r.get("record_excel_path") or "")
        groups[key].append(r)
    out = []
    for key, group_rows in groups.items():
        merged_json, row_count = _vpd_merge_chunks(group_rows)
        first = group_rows[0]
        sorted_rows = sorted(group_rows, key=lambda x: (x.get("added_at") or "", x.get("chunk_index", 0)))
        chunk_ids = [x.get("id") for x in sorted_rows if x.get("id") is not None]
        first_id = first.get("id")
        out.append({
            "id": first_id,
            "chunk_ids": chunk_ids,
            "sido_code": first.get("sido_code"),
            "sido_name": first.get("sido_name"),
            "record_timestamp": key[0],
            "record_excel_path": key[1],
            "merged_json": merged_json,
            "row_count": row_count,
        })
    return out


def list_virtual_population_db_metadata(sido_code: str) -> List[Dict[str, Any]]:
    """해당 시도 가상인구 DB 레코드 메타만 조회 (data_json 제외, timeout 방지). 표·다운로드용 전체 데이터는 버튼으로 별도 로드."""
    rows = _vpd_fetch_all_rows_paginated("id, sido_code, sido_name, record_timestamp, record_excel_path, chunk_index, added_at", sido_code=sido_code)
    return _vpd_group_metadata_only(rows)


def list_virtual_population_db_records_all(sido_code: str) -> List[Dict[str, Any]]:
    rows = _vpd_fetch_all_rows_paginated("id, sido_code, sido_name, record_timestamp, record_excel_path, chunk_index, added_at, data_json", sido_code=sido_code, page_size=VPD_FETCH_PAGE_SIZE_HEAVY)
    grouped = _vpd_group_and_merge(rows)
    return [
        {
            "id": g["id"],
            "chunk_ids": g["chunk_ids"],
            "sido_code": g["sido_code"],
            "sido_name": g["sido_name"],
            "record_timestamp": g["record_timestamp"],
            "record_excel_path": g["record_excel_path"],
            "rows": g["row_count"],
        }
        for g in grouped
    ]


def list_virtual_population_db_data_jsons(sido_code: str) -> List[Tuple[int, str]]:
    """해당 시도 가상인구 DB 레코드의 (id, data_json) 목록 반환 (added_at 순). 청크 병합된 JSON 문자열. 소량 페이지로 timeout 방지."""
    rows = _vpd_fetch_all_rows_paginated("id, record_timestamp, record_excel_path, chunk_index, added_at, data_json", sido_code=sido_code, page_size=VPD_FETCH_PAGE_SIZE_HEAVY)
    grouped = _vpd_group_and_merge(rows)
    return [(g["id"], g["merged_json"]) for g in grouped]


@st.cache_data(ttl=60, show_spinner=False)
def list_virtual_population_db_records(sido_code: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = _vpd_fetch_all_rows_paginated("id, sido_code, sido_name, record_timestamp, record_excel_path, chunk_index, added_at, data_json", sido_code=sido_code, page_size=VPD_FETCH_PAGE_SIZE_HEAVY)
    grouped = _vpd_group_and_merge(rows)
    out = []
    for g in grouped:
        if not _vpd_record_has_persona_and_contemporary(g["merged_json"]):
            continue
        try:
            df = pd.read_json(g["merged_json"], orient="records")
            out.append({
                "id": g["id"],
                "chunk_ids": g["chunk_ids"],
                "sido_code": g["sido_code"],
                "sido_name": g["sido_name"],
                "record_timestamp": g["record_timestamp"],
                "record_excel_path": g["record_excel_path"],
                "rows": len(df),
                "columns_count": len(df.columns),
            })
        except Exception:
            continue
    return out


@st.cache_data(ttl=60, show_spinner=False)
def get_virtual_population_db_record_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    r = get_supabase().table("virtual_population_db").select("sido_code, sido_name, record_timestamp, record_excel_path").eq("id", int(record_id)).limit(1).execute()
    if not r.data or len(r.data) == 0:
        return None
    row = r.data[0]
    chunks = _vpd_get_chunks_for_record(row["sido_code"], row["record_timestamp"], row["record_excel_path"])
    if not chunks:
        return None
    merged_json, row_count = _vpd_merge_chunks(chunks)
    try:
        df = pd.read_json(merged_json, orient="records")
        return {
            "id": chunks[0]["id"],
            "sido_code": row["sido_code"],
            "sido_name": row["sido_name"],
            "record_timestamp": row["record_timestamp"],
            "record_excel_path": row["record_excel_path"],
            "df": df.copy(),
            "rows": len(df),
            "columns_count": len(df.columns),
        }
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def get_virtual_population_db_combined_df(sido_code: str):
    rows = _vpd_fetch_all_rows_paginated("id, sido_code, sido_name, record_timestamp, record_excel_path, chunk_index, added_at, data_json", sido_code=sido_code, page_size=VPD_FETCH_PAGE_SIZE_HEAVY)
    grouped = _vpd_group_and_merge(rows)
    dfs = []
    for g in grouped:
        if not _vpd_record_has_persona_and_contemporary(g["merged_json"]):
            continue
        try:
            df = pd.read_json(g["merged_json"], orient="records")
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


def _get_sido_vdb_stats_light() -> Dict[str, Dict[str, Any]]:
    """row_count 컬럼만 조회해 시도별 합산 (경량). row_count 미도입 시 실패하면 None 반환."""
    try:
        rows = _vpd_fetch_all_rows_paginated("sido_code, sido_name, row_count", sido_code=None, page_size=VPD_FETCH_PAGE_SIZE)
    except Exception:
        return None
    out: Dict[str, Dict[str, Any]] = {}
    for r in (rows or []):
        if r.get("row_count") is None and "row_count" not in r:
            return None
        sido_code = r.get("sido_code")
        if not sido_code or sido_code == "00":
            continue
        cnt = r.get("row_count")
        try:
            cnt = int(cnt) if cnt is not None else 0
        except (TypeError, ValueError):
            cnt = 0
        if sido_code not in out:
            out[sido_code] = {"sido_name": r.get("sido_name") or "", "vdb_count": 0}
        out[sido_code]["vdb_count"] = out[sido_code]["vdb_count"] + cnt
    return out


@st.cache_data(ttl=60, show_spinner=False)
def get_sido_vdb_stats() -> Dict[str, Dict[str, Any]]:
    """시도별 가상인구 인원 수. row_count 컬럼 있으면 경량 조회, 없으면 기존 data_json 집계(무거움)로 폴백."""
    light = _get_sido_vdb_stats_light()
    if light is not None:
        return light
    # row_count 미도입 시: 기존 방식 (전국 data_json 조회 — 느림)
    rows = _vpd_fetch_all_rows_paginated("id, sido_code, sido_name, record_timestamp, record_excel_path, chunk_index, added_at, data_json", sido_code=None, page_size=VPD_FETCH_PAGE_SIZE_HEAVY)
    grouped = _vpd_group_and_merge(rows)
    out = {}
    for g in grouped:
        sido_code = g["sido_code"]
        if sido_code == "00":
            continue
        if sido_code not in out:
            out[sido_code] = {"sido_name": g["sido_name"], "vdb_count": 0}
        out[sido_code]["vdb_count"] = out[sido_code]["vdb_count"] + g["row_count"]
    return out


def insert_virtual_population_db(sido_code: str, sido_name: str, record_timestamp: str, record_excel_path: str, data_json: str) -> Optional[int]:
    """가상인구 DB 한 건 INSERT (단일 청크, chunk_index=0). 성공 시 id 반환."""
    try:
        parsed = json.loads(data_json)
        row_count = len(parsed) if isinstance(parsed, list) else 0
    except Exception:
        row_count = 0
    r = get_supabase().table("virtual_population_db").insert({
        "sido_code": str(sido_code),
        "sido_name": str(sido_name),
        "record_timestamp": str(record_timestamp),
        "record_excel_path": str(record_excel_path),
        "chunk_index": 0,
        "data_json": data_json,
        "row_count": row_count,
    }).execute()
    if r.data and len(r.data) > 0:
        return r.data[0].get("id")
    return None


def insert_virtual_population_db_chunked(
    sido_code: str,
    sido_name: str,
    record_timestamp: str,
    record_excel_path: str,
    data_json: str,
    chunk_size: int = VPD_CHUNK_SIZE,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Optional[int]:
    """전체 data_json을 청크 단위로 나누어 여러 번 INSERT. progress_callback(done, total) 호출. 성공 시 첫 청크 id 반환."""
    try:
        rows = json.loads(data_json)
    except Exception as e:
        raise ValueError(f"data_json 파싱 실패: {e}") from e
    if not isinstance(rows, list):
        raise ValueError("data_json은 JSON 배열이어야 합니다.")
    total = len(rows)
    if total == 0:
        return None
    first_id: Optional[int] = None
    inserted = 0
    for start in range(0, total, chunk_size):
        chunk = rows[start : start + chunk_size]
        chunk_json = json.dumps(chunk, ensure_ascii=False)
        try:
            r = get_supabase().table("virtual_population_db").insert({
                "sido_code": str(sido_code),
                "sido_name": str(sido_name),
                "record_timestamp": str(record_timestamp),
                "record_excel_path": str(record_excel_path),
                "chunk_index": start // chunk_size,
                "data_json": chunk_json,
                "row_count": len(chunk),
            }).execute()
            if r.data and len(r.data) > 0 and first_id is None:
                first_id = r.data[0].get("id")
            inserted += len(chunk)
        except Exception as e:
            if progress_callback:
                progress_callback(inserted, total)
            raise RuntimeError(f"청크 INSERT 실패 (offset={start}, size={len(chunk)}): {e}") from e
        if progress_callback:
            progress_callback(min(inserted, total), total)
    return first_id


def update_virtual_population_db_data_json(record_id: int, data_json: str):
    """virtual_population_db 레코드의 해당 행만 data_json·row_count UPDATE (단일 행)."""
    try:
        parsed = json.loads(data_json)
        row_count = len(parsed) if isinstance(parsed, list) else 0
    except Exception:
        row_count = 0
    get_supabase().table("virtual_population_db").update({"data_json": data_json, "row_count": row_count}).eq("id", int(record_id)).execute()


def update_virtual_population_db_record_persona_reflection_only(record_id: int, merged_json: str) -> bool:
    """페르소나·현시대 반영만 기존 청크에 반영 (전체 재저장 대신 델타만 전송 → 저장 속도 향상).
    RPC update_vpd_chunk_persona_reflection 사용. 성공 시 True, RPC 미설치 등으로 실패 시 False."""
    try:
        r = get_supabase().table("virtual_population_db").select("sido_code, record_timestamp, record_excel_path").eq("id", int(record_id)).limit(1).execute()
        if not r.data or len(r.data) == 0:
            return False
        row = r.data[0]
        chunks = _vpd_get_chunk_ids_for_record(row["sido_code"], row["record_timestamp"], row["record_excel_path"])
        if not chunks:
            return False
        rows_list = json.loads(merged_json)
        if not isinstance(rows_list, list):
            return False
        sb = get_supabase()
        for c in chunks:
            chunk_id = c["id"]
            chunk_index = int(c.get("chunk_index", 0))
            start = chunk_index * VPD_CHUNK_SIZE
            end = min(start + VPD_CHUNK_SIZE, len(rows_list))
            slice_rows = rows_list[start:end]
            updates = []
            for i, rrow in enumerate(slice_rows):
                if not isinstance(rrow, dict):
                    continue
                p_val = rrow.get("페르소나")
                r_val = rrow.get("현시대 반영")
                if p_val is None:
                    p_val = ""
                if r_val is None:
                    r_val = ""
                updates.append({"i": i, "p": str(p_val), "r": str(r_val)})
            if updates:
                sb.rpc("update_vpd_chunk_persona_reflection", {"p_chunk_id": chunk_id, "p_updates": updates}).execute()
        return True
    except Exception:
        return False


def update_virtual_population_db_record_merged(record_id: int, merged_json: str):
    """논리 레코드(첫 청크 id)와 병합된 전체 JSON을 받아, 해당 레코드의 모든 청크 행을 올바르게 갱신.
    캐시는 (첫 청크 id, 병합 JSON)을 갖고 DB는 여러 청크 행으로 나뉘어 있으므로, 병합 JSON을 청크로 나눠 각 행을 UPDATE."""
    r = get_supabase().table("virtual_population_db").select("sido_code, sido_name, record_timestamp, record_excel_path").eq("id", int(record_id)).limit(1).execute()
    if not r.data or len(r.data) == 0:
        raise ValueError(f"record_id {record_id} not found")
    row = r.data[0]
    sido_code = row["sido_code"]
    sido_name = row["sido_name"]
    record_timestamp = row["record_timestamp"]
    record_excel_path = row["record_excel_path"]
    db_chunks = _vpd_get_chunks_for_record(sido_code, record_timestamp, record_excel_path)
    if not db_chunks:
        raise ValueError(f"no chunks for record_id {record_id}")
    try:
        rows_list = json.loads(merged_json)
    except Exception as e:
        raise ValueError(f"merged_json parse error: {e}") from e
    if not isinstance(rows_list, list):
        raise ValueError("merged_json must be a JSON array")
    sorted_db = sorted(db_chunks, key=lambda x: (x.get("added_at") or "", x.get("chunk_index", 0)))
    new_chunks = []
    for start in range(0, len(rows_list), VPD_CHUNK_SIZE):
        chunk = rows_list[start : start + VPD_CHUNK_SIZE]
        new_chunks.append((json.dumps(chunk, ensure_ascii=False), len(chunk)))
    sb = get_supabase()
    for i, db_row in enumerate(sorted_db):
        chunk_id = db_row["id"]
        if i < len(new_chunks):
            chunk_json, rc = new_chunks[i]
            sb.table("virtual_population_db").update({"data_json": chunk_json, "row_count": rc}).eq("id", chunk_id).execute()
        else:
            sb.table("virtual_population_db").update({"data_json": "[]", "row_count": 0}).eq("id", chunk_id).execute()
    if len(new_chunks) > len(sorted_db):
        for i in range(len(sorted_db), len(new_chunks)):
            chunk_json, rc = new_chunks[i]
            sb.table("virtual_population_db").insert({
                "sido_code": sido_code,
                "sido_name": sido_name,
                "record_timestamp": record_timestamp,
                "record_excel_path": record_excel_path,
                "chunk_index": i,
                "data_json": chunk_json,
                "row_count": rc,
            }).execute()


def get_virtual_population_db_data_json(record_id: int) -> Optional[str]:
    """해당 레코드(모든 청크)의 data_json 병합 후 반환."""
    r = get_supabase().table("virtual_population_db").select("sido_code, record_timestamp, record_excel_path").eq("id", int(record_id)).limit(1).execute()
    if not r.data or len(r.data) == 0:
        return None
    row = r.data[0]
    chunks = _vpd_get_chunks_for_record(row["sido_code"], row["record_timestamp"], row["record_excel_path"])
    if not chunks:
        return None
    merged_json, _ = _vpd_merge_chunks(chunks)
    return merged_json


def find_virtual_population_db_id(sido_code: str, record_timestamp: str, record_excel_path: str) -> Optional[int]:
    """sido_code, record_timestamp, record_excel_path 로 id 조회 (첫 청크 id)."""
    r = get_supabase().table("virtual_population_db").select("id").eq("sido_code", sido_code).eq("record_timestamp", record_timestamp).eq("record_excel_path", record_excel_path).order("chunk_index").limit(1).execute()
    if r.data and len(r.data) > 0:
        return r.data[0].get("id")
    return None
