# utils/gemini_key.py
# - API 키는 Git에 올리지 말고, Streamlit Secrets 또는 환경변수로 설정하세요.
# - 배포(Streamlit Cloud): Manage app > Secrets 에서 GEMINI_API_KEY 설정
# - 로컬: .streamlit/secrets.toml 에 GEMINI_API_KEY = "your-key" 추가

import os

_GEMINI_API_KEY = ""

try:
    import streamlit as st
    if hasattr(st, "secrets") and st.secrets:
        _GEMINI_API_KEY = (st.secrets.get("GEMINI_API_KEY") or st.secrets.get("gemini_api_key")) or ""
except Exception:
    pass

if not _GEMINI_API_KEY:
    _GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""

GEMINI_API_KEY = _GEMINI_API_KEY
