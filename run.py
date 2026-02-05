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

# 첫 화면에서 로딩 안내를 잠깐 보여준 뒤 제거하고 앱을 로드 (Cloud 타임아웃 완화)
if "run_loading_started" not in st.session_state:
    st.session_state.run_loading_started = False
if not st.session_state.run_loading_started:
    load_ph = st.empty()
    with load_ph.container():
        st.info("앱을 불러오는 중입니다… 잠시만 기다려 주세요.")
    st.session_state.run_loading_started = True
    load_ph.empty()  # 메시지 제거 후 앱 로드 → 랜딩/스피너/에러가 보이도록

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
