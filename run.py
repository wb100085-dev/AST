"""
Streamlit Cloud / 배포용 진입 스크립트.
실행: streamlit run run.py
- 한 번의 요청에서 앱을 로드해, Cloud에서 두 번째 요청 타임아웃을 피합니다.
"""
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import streamlit as st

st.set_page_config(page_title="AI Social Twin", layout="wide")

# 첫 방문에만 로딩 안내 표시 (rerun 없이 같은 요청 안에서 앱 로드 → Cloud 타임아웃 완화)
if "run_loading_started" not in st.session_state:
    st.session_state.run_loading_started = False
if not st.session_state.run_loading_started:
    st.info("앱을 불러오는 중입니다… 잠시만 기다려 주세요.")
    st.session_state.run_loading_started = True

try:
    from app import main
    main()
except Exception as e:
    st.error("앱 로드 중 오류가 발생했습니다.")
    st.code(str(e))
    import traceback
    with st.expander("상세 traceback", expanded=True):
        st.code(traceback.format_exc())
    st.caption("Streamlit Cloud에서는 Manage app > Logs 에서 서버 로그를 확인할 수 있습니다.")
