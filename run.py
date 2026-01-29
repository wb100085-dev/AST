"""
Streamlit Cloud / 배포용 진입 스크립트.
import 오류가 나도 화면에 오류 메시지가 보이도록 합니다.
실행: streamlit run run.py
"""
import streamlit as st

st.set_page_config(page_title="AI Social Twin", layout="wide")

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
