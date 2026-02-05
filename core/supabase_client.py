"""
Supabase 클라이언트 (st.secrets 또는 환경변수)
로컬: .streamlit/secrets.toml 에 SUPABASE_URL, SUPABASE_KEY 설정.
배포: Streamlit Cloud > Manage app > Secrets 에 동일 키 설정.
"""
import os
from typing import Optional, Tuple

import streamlit as st


def _get_supabase_credentials() -> Tuple[Optional[str], Optional[str]]:
    """st.secrets 우선, 없으면 환경변수에서 SUPABASE_URL, SUPABASE_KEY 반환."""
    url: Optional[str] = None
    key: Optional[str] = None
    try:
        if hasattr(st, "secrets") and st.secrets is not None:
            # Top-level (secrets.toml: SUPABASE_URL = "...")
            url = st.secrets.get("SUPABASE_URL")
            key = st.secrets.get("SUPABASE_KEY")
            if not url or not key:
                # Nested (secrets.toml: [supabase] url = "..." key = "...")
                supabase = st.secrets.get("supabase") or {}
                url = url or supabase.get("SUPABASE_URL") or supabase.get("url")
                key = key or supabase.get("SUPABASE_KEY") or supabase.get("key")
    except Exception:
        pass
    if not url or not key:
        url = url or os.environ.get("SUPABASE_URL")
        key = key or os.environ.get("SUPABASE_KEY")
    return (url, key)


@st.cache_resource
def get_supabase():
    """
    Supabase 클라이언트 반환. 연결 실패 시 RuntimeError 발생.
    """
    url, key = _get_supabase_credentials()
    if not url or not key or "your-" in str(key).lower() or "xxxxx" in str(url).lower():
        raise RuntimeError(
            "Supabase 설정이 없습니다. "
            "st.secrets 또는 환경변수에 SUPABASE_URL, SUPABASE_KEY를 설정하세요."
        )
    from supabase import create_client
    return create_client(url.strip(), key.strip())
