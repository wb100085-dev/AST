"""
공통 UI 컴포넌트 및 스타일
설문조사와 심층면접 페이지에서 공통으로 사용하는 UI 요소들
"""
import streamlit as st


def apply_common_styles():
    """React Tailwind CSS 스타일을 Streamlit에 적용하는 공통 CSS"""
    st.markdown("""
    <style>
    /* 전체 배경 */
    .stApp {
        background-color: #FDFDFF;
    }
    
    /* 텍스트 영역 스타일 */
    .stTextArea > div > div > textarea {
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        padding: 20px;
        font-size: 14px;
        transition: all 0.3s;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #4f46e5;
        box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.1);
        outline: none;
    }
    
    /* 입력 필드 스타일 */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 12px;
        font-size: 14px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4f46e5;
        outline: none;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        border-radius: 24px;
        font-weight: 800;
        font-size: 18px;
        padding: 20px 30px;
        transition: all 0.3s;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(79, 70, 229, 0.3);
    }
    
    /* 카드 스타일 */
    .survey-card {
        background: white;
        border-radius: 24px;
        padding: 32px;
        border: 1px solid #e0e7ff;
        box-shadow: 0 20px 25px -5px rgba(79, 70, 229, 0.1);
    }
    
    .survey-card-indigo {
        background: #eef2ff;
        border: 1px solid #c7d2fe;
    }
    
    /* 배지 스타일 */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 700;
    }
    
    .badge-indigo {
        background: #eef2ff;
        color: #4f46e5;
    }
    
    /* 헤더 스타일 */
    .survey-header {
        margin-bottom: 24px;
    }
    </style>
    """, unsafe_allow_html=True)


def render_common_input_form():
    """공통 입력 폼 (제품 정의, 니즈, 타깃) - 좌측 패널용"""
    # 세션 상태 초기화
    if "definition" not in st.session_state:
        st.session_state.definition = ""
    if "needs" not in st.session_state:
        st.session_state.needs = ""
    if "target" not in st.session_state:
        st.session_state.target = ""
    
    # 필수 입력 1: 제품 정의
    st.markdown("### 제품/서비스의 정의 <span style='color: #ef4444;'>*</span>", unsafe_allow_html=True)
    
    definition_length = len(st.session_state.definition)
    is_definition_valid = definition_length >= 300
    
    col_def_label, col_def_count = st.columns([3, 1])
    with col_def_count:
        if is_definition_valid:
            st.markdown(f"<span style='color: #10b981; font-size: 12px; font-weight: 600;'>{definition_length} / 300자 이상</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color: #94a3b8; font-size: 12px; font-weight: 600;'>{definition_length} / 300자 이상</span>", unsafe_allow_html=True)
    
    definition = st.text_area(
        "제품/서비스 정의",
        value=st.session_state.definition,
        placeholder="제품의 핵심 기능, 가치, 시장 내 위치 등을 상세히 작성해주세요.",
        height=200,
        key="definition_input",
        label_visibility="collapsed"
    )
    st.session_state.definition = definition
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 필수 입력 2: 조사의 니즈
    st.markdown("### 조사의 목적과 니즈 <span style='color: #ef4444;'>*</span>", unsafe_allow_html=True)
    
    needs_length = len(st.session_state.needs)
    is_needs_valid = needs_length >= 300
    
    col_needs_label, col_needs_count = st.columns([3, 1])
    with col_needs_count:
        if is_needs_valid:
            st.markdown(f"<span style='color: #10b981; font-size: 12px; font-weight: 600;'>{needs_length} / 300자 이상</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color: #94a3b8; font-size: 12px; font-weight: 600;'>{needs_length} / 300자 이상</span>", unsafe_allow_html=True)
    
    needs = st.text_area(
        "조사의 목적과 니즈",
        value=st.session_state.needs,
        placeholder="이번 조사를 통해 무엇을 알고 싶으신가요? (예: 타겟 유저의 가격 저항선, 경쟁사 대비 강점 등)",
        height=200,
        key="needs_input",
        label_visibility="collapsed"
    )
    st.session_state.needs = needs
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 선택 입력: 타깃
    st.markdown("### 타깃 (선택)")
    target = st.text_input(
        "희망 타깃",
        value=st.session_state.target,
        placeholder="특정 타깃이 있다면 적어주세요. (예: 30대 워킹맘)",
        key="target_input"
    )
    st.session_state.target = target
    
    return is_definition_valid, is_needs_valid


def render_page_header(title: str, icon: str = "AI"):
    """페이지 헤더 렌더링 (icon: 헤더 박스 안 텍스트, 이모지 없이 'AI' 등 권장)"""
    col_header1, col_header2 = st.columns([1, 4])
    with col_header1:
        st.markdown(f"""
        <div style="width: 40px; height: 40px; background: #4f46e5; border-radius: 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3);">
            <span style="color: white; font-size: 14px; font-weight: 700;">{icon}</span>
        </div>
        """, unsafe_allow_html=True)
    with col_header2:
        st.markdown(f"""
        <h1 style="font-size: 28px; font-weight: 800; color: #0f172a; margin: 0; padding-top: 8px;">
            {title}
        </h1>
        """, unsafe_allow_html=True)
    st.markdown("---")
