"""
Streamlit Cloud / 배포용 진입 스크립트.
- 첫 응답을 빨리 보내서 "무한 로딩"처럼 보이지 않도록, 먼저 "로딩 중"을 띄운 뒤 앱을 불러옵니다.
실행: streamlit run run.py
"""
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import streamlit as st

st.set_page_config(page_title="AI Social Twin", layout="wide")

# 첫 방문 시 "로딩 중"만 먼저 보여주고 rerun → 다음 요청에서 무거운 app 로드 (인터넷에서 끊기지 않도록)
if "run_loading_started" not in st.session_state:
    st.session_state.run_loading_started = False

if not st.session_state.run_loading_started:
    st.info("앱을 불러오는 중입니다… 잠시만 기다려 주세요.")
    st.session_state.run_loading_started = True
    st.rerun()

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
