# utils/gemini_key.py
# - API 키는 Git에 올리지 말고, .env 또는 환경변수로 설정하세요.
# - 로컬: 프로젝트 루트에 .env 파일 생성 후 GEMINI_API_KEY=your-key 추가 (python-dotenv로 로드)
# - 배포(Streamlit Cloud): Manage app > Secrets 에서 GEMINI_API_KEY 설정

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""

if not _GEMINI_API_KEY:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            _GEMINI_API_KEY = (st.secrets.get("GEMINI_API_KEY") or st.secrets.get("gemini_api_key")) or ""
    except Exception:
        pass

GEMINI_API_KEY = _GEMINI_API_KEY
