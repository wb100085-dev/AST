"""
공통 데이터 세션 캐시: SIDO_MASTER, DB 조회 결과를 st.session_state에 저장해
페이지 이동 시 재조회를 줄임. core/ 모듈만 의존.
"""
from typing import List, Dict, Any, Optional, Set

import streamlit as st


def _get_cached(key: str):
    """캐시 값 반환. 키가 없으면 None (미로드). 키가 있으면 저장값 반환 (None일 수 있음)."""
    return st.session_state.get(key)


def get_sido_master() -> List[Dict[str, str]]:
    """SIDO_MASTER를 세션에 캐시해 반환. 없으면 core.constants에서 로드 후 저장."""
    key = "_cache_sido_master"
    cached = _get_cached(key)
    if cached is not None:
        return cached
    from core.constants import SIDO_MASTER
    st.session_state[key] = SIDO_MASTER
    return SIDO_MASTER


def get_cached_db_list_stats(sido_code: str, active_only: bool = False) -> List[Dict[str, Any]]:
    """db_list_stats_by_sido 결과를 세션에 캐시. 있으면 반환, 없으면 DB 조회 후 저장."""
    key = f"_cache_db_list_stats_{sido_code}_{active_only}"
    cached = _get_cached(key)
    if cached is not None:
        return cached
    from core.db import db_list_stats_by_sido
    result = db_list_stats_by_sido(sido_code=sido_code, active_only=active_only)
    st.session_state[key] = result
    return result


def get_cached_db_axis_margin_stats(sido_code: str, axis_key: str) -> Optional[Dict[str, Any]]:
    """db_get_axis_margin_stats 결과를 세션에 캐시. (None 결과도 캐시해 재조회 방지)"""
    key = f"_cache_db_axis_margin_{sido_code}_{axis_key}"
    if key in st.session_state:
        return st.session_state[key]
    from core.db import db_get_axis_margin_stats
    result = db_get_axis_margin_stats(sido_code=sido_code, axis_key=axis_key)
    st.session_state[key] = result
    return result


def get_cached_db_all_axis_margin_stats(sido_code: str) -> Dict[str, Dict[str, Any]]:
    """해당 시도 6축 마진 설정을 한 번의 쿼리로 조회 후 세션에 캐시(ttl 개념은 세션 유지 동안). 저장 직후 무효화로 최신값 반영."""
    key = f"_cache_db_all_axis_margin_{sido_code}"
    cached = _get_cached(key)
    if cached is not None:
        return cached
    from core.db import db_get_all_axis_margin_stats
    result = db_get_all_axis_margin_stats(sido_code)
    st.session_state[key] = result
    return result


def get_cached_db_six_axis_stat_ids(sido_code: str) -> Set[int]:
    """db_get_six_axis_stat_ids 결과를 세션에 캐시."""
    key = f"_cache_db_six_axis_stat_ids_{sido_code}"
    cached = _get_cached(key)
    if cached is not None:
        return cached
    from core.db import db_get_six_axis_stat_ids
    result = db_get_six_axis_stat_ids(sido_code=sido_code)
    st.session_state[key] = result
    return result


def get_cached_db_template(sido_code: str) -> Optional[Dict[str, Any]]:
    """db_get_template 결과를 세션에 캐시. (None 결과도 캐시)"""
    key = f"_cache_db_template_{sido_code}"
    if key in st.session_state:
        return st.session_state[key]
    from core.db import db_get_template
    result = db_get_template(sido_code=sido_code)
    st.session_state[key] = result
    return result


def invalidate_db_stats_cache(sido_code: Optional[str] = None):
    """stats 관련 캐시 무효화. sido_code가 있으면 해당 시도만, 없으면 전체."""
    to_del = [k for k in list(st.session_state.keys()) if k.startswith("_cache_db_list_stats_")]
    if sido_code is not None:
        to_del = [k for k in to_del if k.startswith(f"_cache_db_list_stats_{sido_code}_")]
    for k in to_del:
        st.session_state.pop(k, None)


def invalidate_db_axis_margin_cache(sido_code: Optional[str] = None):
    """axis_margin / six_axis / all_axis_margin 세션 캐시 무효화 (6축 저장 직후 최신값 반영)."""
    for prefix in ("_cache_db_axis_margin_", "_cache_db_all_axis_margin_", "_cache_db_six_axis_stat_ids_"):
        to_del = [k for k in list(st.session_state.keys()) if k.startswith(prefix)]
        if sido_code is not None:
            to_del = [k for k in to_del if k.startswith(f"{prefix}{sido_code}_") or k == f"{prefix}{sido_code}"]
        for k in to_del:
            st.session_state.pop(k, None)


def invalidate_db_template_cache(sido_code: Optional[str] = None):
    """template 캐시 무효화."""
    to_del = [k for k in list(st.session_state.keys()) if k.startswith("_cache_db_template_")]
    if sido_code is not None:
        to_del = [k for k in to_del if k == f"_cache_db_template_{sido_code}"]
    for k in to_del:
        st.session_state.pop(k, None)
