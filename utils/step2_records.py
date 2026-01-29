# utils/step2_records.py
"""
2차 대입 결과 관리 유틸리티
"""
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

# 앱 루트 기준 data 경로 (배포 환경 대응)
_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP2_RECORDS_DIR = os.path.join(_APP_ROOT, "data", "step2_records")


def save_step2_record(result_df: pd.DataFrame, sido_code: str, sido_name: str, added_columns: List[str]) -> Optional[str]:
    """2차 대입 결과를 날짜/시간별로 저장. 저장된 파일 경로 반환."""
    try:
        os.makedirs(STEP2_RECORDS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"{ts}_{sido_code}"
        excel_path = os.path.join(STEP2_RECORDS_DIR, f"{base_name}.xlsx")
        meta_path = os.path.join(STEP2_RECORDS_DIR, f"{base_name}.json")
        result_df.to_excel(excel_path, index=False, engine="openpyxl")
        meta = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sido_code": sido_code,
            "sido_name": sido_name,
            "rows": len(result_df),
            "added_columns": added_columns,
            "columns_count": len(result_df.columns),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return excel_path
    except Exception:
        return None


def _list_step2_records_impl() -> List[Dict[str, Any]]:
    """저장된 2차 대입 결과 목록 (최신순) - 내부 구현."""
    if not os.path.isdir(STEP2_RECORDS_DIR):
        return []
    records = []
    for f in os.listdir(STEP2_RECORDS_DIR):
        if not f.endswith(".json"):
            continue
        meta_path = os.path.join(STEP2_RECORDS_DIR, f)
        excel_path = os.path.join(STEP2_RECORDS_DIR, f.replace(".json", ".xlsx"))
        if not os.path.isfile(excel_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
            meta["excel_path"] = excel_path
            records.append(meta)
        except Exception:
            continue
    records.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return records


_cached_list_fn = None  # st.cache_data 래퍼 (한 번만 생성)


def list_step2_records() -> List[Dict[str, Any]]:
    """저장된 2차 대입 결과 목록 (최신순). Streamlit 사용 시 30초 캐시 적용."""
    global _cached_list_fn
    try:
        import streamlit as st
        if _cached_list_fn is None:
            _cached_list_fn = st.cache_data(ttl=30)(_list_step2_records_impl)
        return _cached_list_fn()
    except Exception:
        return _list_step2_records_impl()


def load_step2_record(excel_path: str) -> pd.DataFrame:
    """2차 대입 결과 Excel 파일을 DataFrame으로 로드."""
    try:
        return pd.read_excel(excel_path, engine="openpyxl")
    except Exception as e:
        raise Exception(f"데이터 로드 실패: {e}")
