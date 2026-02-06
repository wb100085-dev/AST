"""
[선호도 분석]컨조인트 분석 (Conjoint Analysis)
- 무거운 라이브러리(plotly, pandas, numpy, statsmodels)는 페이지 진입 시 지연 로딩
"""
import streamlit as st
import os
from core.ui_components import show_package_required_error, render_step2_data_selector

try:
    import statsmodels
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

def page_conjoint_analysis():
    """컨조인트 분석 페이지"""
    if not show_package_required_error(HAS_STATSMODELS, "statsmodels", "statsmodels"):
        return
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import itertools
    import json
    import statsmodels.api as sm

    use_real_data, real_data_df = render_step2_data_selector("conjoint", show_preview_row_count=True)

    st.markdown("---")
    
    # 나머지 Conjoint Analysis 코드는 기존 파일에서 가져옴
    # 페이지 설정 및 스타일
    st.markdown("""
        <style>
        .main {
            background-color: #f8fafc;
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 3em;
            background-color: #4f46e5;
            color: white;
            font-weight: bold;
        }
        .stMetric {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .card {
            background-color: white;
            padding: 24px;
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        h1, h2, h3 {
            color: #1e293b;
        }
        </style>
        """, unsafe_allow_html=True)

    # 초기 속성 설정
    if 'attributes' not in st.session_state:
        st.session_state.attributes = {
            '가격': ['저가 (무료 배송)', '중가 (표준)', '고가 (프리미엄)'],
            '디자인': ['모던 미니멀', '클래식 레트로'],
            '성능': ['전문가급 (고출력)', '일반용 (저소음)']
        }

    def generate_data(attributes, n_respondents=100):
        """제품 프로파일 생성 및 가상 응답 시뮬레이션"""
        keys = attributes.keys()
        values = attributes.values()
        profiles = [dict(zip(keys, v)) for v in itertools.product(*values)]
        df_profiles = pd.DataFrame(profiles)
        df_profiles['profile_id'] = range(1, len(df_profiles) + 1)
        
        # 가상 응답 생성
        np.random.seed(42)
        responses = []
        for user_id in range(1, n_respondents + 1):
            base_score = 10
            user_scores = []
            for _, p in df_profiles.iterrows():
                score = base_score
                for attr, levels in attributes.items():
                    lvl_idx = levels.index(p[attr])
                    # 앞쪽 수준일수록 선호도가 높도록 임의 가중치 부여
                    score += (len(levels) - lvl_idx) * np.random.uniform(0.5, 2.0)
                score += np.random.normal(0, 2)
                user_scores.append(score)
            
            ranks = pd.Series(user_scores).rank(ascending=False).astype(int)
            for idx, rank in enumerate(ranks):
                responses.append({
                    'user_id': user_id,
                    'profile_id': df_profiles.iloc[idx]['profile_id'],
                    'rank': rank
                })
        
        df_responses = pd.DataFrame(responses)
        df_total = pd.merge(df_responses, df_profiles, on='profile_id')
        df_total['preference_score'] = len(df_profiles) - df_total['rank'] + 1
        return df_profiles, df_total

    def run_analysis(attributes, df_total):
        """OLS 회귀분석 수행"""
        if not HAS_STATSMODELS:
            st.error("❌ `statsmodels`가 필요합니다. 설치 후 페이지를 새로고침해주세요.")
            st.stop()
        keys = list(attributes.keys())
        X = pd.get_dummies(df_total[keys], drop_first=True, dtype=int)
        X = sm.add_constant(X)
        y = df_total['preference_score']
        model = sm.OLS(y, X).fit()
        return model

    # 사이드바 - 속성 및 수준 구성
    with st.sidebar:
        st.title("⚙️ 속성 및 수준 설정")
        st.info("여기서 제품의 속성과 상세 옵션을 변경할 수 있습니다.")
        
        new_attr_name = st.text_input("➕ 새 속성 이름", placeholder="예: 배터리")
        if st.button("속성 추가"):
            if new_attr_name and new_attr_name not in st.session_state.attributes:
                st.session_state.attributes[new_attr_name] = ["옵션 1", "옵션 2"]
                st.rerun()

        st.divider()
        
        # 각 속성별 편집
        for attr, levels in list(st.session_state.attributes.items()):
            with st.expander(f"📌 {attr}"):
                levels_str = st.text_area(f"수준 목록 (줄바꿈으로 구분)", value="\n".join(levels), key=f"area_{attr}")
                new_levels = [l.strip() for l in levels_str.split("\n") if l.strip()]
                
                if new_levels != levels:
                    st.session_state.attributes[attr] = new_levels
                    
                if st.button(f"🗑️ {attr} 삭제", key=f"del_{attr}"):
                    if len(st.session_state.attributes) > 1:
                        del st.session_state.attributes[attr]
                        st.rerun()
                    else:
                        st.error("최소 1개의 속성은 유지해야 합니다.")

    # 메인 대시보드 UI
    st.title("🚀 Conjoint Analysis Pro")
    if use_real_data and real_data_df is not None:
        st.markdown(f"**AI 설문 응답 결과 활용** ({len(real_data_df)}명) - 고급 마케팅 선호도 분석 엔진")
    else:
        st.markdown("가상 인구 100명의 데이터를 활용한 고급 마케팅 선호도 분석 엔진")

    # 데이터 생성 및 분석 실행
    if use_real_data and real_data_df is not None:
        st.info("💡 AI 설문 응답 결과를 사용 중입니다. 분석에 사용할 속성(컬럼)을 선택하세요.")
        
        available_columns = list(real_data_df.columns)
        selected_columns = st.multiselect(
            "분석에 사용할 속성(컬럼) 선택:",
            options=available_columns,
            default=available_columns[:min(5, len(available_columns))] if len(available_columns) > 0 else [],
            key="conjoint_selected_columns"
        )
        
        if selected_columns:
            temp_attributes = {}
            for col in selected_columns:
                unique_values = real_data_df[col].dropna().unique().tolist()
                if len(unique_values) > 0:
                    if len(unique_values) > 10:
                        value_counts = real_data_df[col].value_counts()
                        unique_values = value_counts.head(10).index.tolist()
                    temp_attributes[col] = [str(v) for v in unique_values[:10]]
            
            if temp_attributes:
                df_profiles, df_total = generate_data(temp_attributes, n_respondents=min(len(real_data_df), 100))
                model = run_analysis(temp_attributes, df_total)
            else:
                st.warning("선택한 컬럼에서 유효한 데이터를 찾을 수 없습니다. 가상 데이터를 사용합니다.")
                df_profiles, df_total = generate_data(st.session_state.attributes)
                model = run_analysis(st.session_state.attributes, df_total)
        else:
            st.warning("분석할 속성을 선택해주세요. 가상 데이터를 사용합니다.")
            df_profiles, df_total = generate_data(st.session_state.attributes)
            model = run_analysis(st.session_state.attributes, df_total)
    else:
        df_profiles, df_total = generate_data(st.session_state.attributes)
        model = run_analysis(st.session_state.attributes, df_total)

    tab1, tab2, tab3 = st.tabs(["📋 데이터 개요", "📊 분석 결과", "🧪 시뮬레이터"])

    # --- Tab 1: 데이터 개요 ---
    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("총 응답자", "100 명")
        col2.metric("제품 조합(Profiles)", f"{len(df_profiles)} 개")
        col3.metric("총 관측치", f"{len(df_total)} 건")
        
        st.subheader("현재 제품 속성 구조")
        attr_df = pd.DataFrame([
            {"속성": k, "수준(Levels)": ", ".join(v)} 
            for k, v in st.session_state.attributes.items()
        ])
        st.table(attr_df)

    # --- Tab 2: 분석 결과 ---
    with tab2:
        st.subheader("💡 속성별 상대적 중요도 (Relative Importance)")
        
        results = model.params
        importance = {}
        utility_list = []
        
        for attr, levels in st.session_state.attributes.items():
            attr_coeffs = [0]
            utility_list.append({"Attribute": attr, "Level": levels[0], "Utility": 0})
            
            for lvl in levels[1:]:
                col_name = f"{attr}_{lvl}"
                val = results.get(col_name, 0)
                attr_coeffs.append(val)
                utility_list.append({"Attribute": attr, "Level": lvl, "Utility": val})
                
            importance[attr] = max(attr_coeffs) - min(attr_coeffs)
        
        total_imp = sum(importance.values())
        imp_df = pd.DataFrame([
            {"Attribute": k, "Importance": (v / total_imp) * 100} 
            for k, v in importance.items()
        ]).sort_values("Importance", ascending=True)
        
        fig_imp = px.bar(imp_df, x="Importance", y="Attribute", orientation='h',
                         color="Importance", color_continuous_scale="Blues",
                         text_auto='.1f')
        fig_imp.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.divider()
        
        st.subheader("📈 부분 가치 (Part-worth Utilities)")
        util_df = pd.DataFrame(utility_list)
        fig_util = px.bar(util_df, x="Level", y="Utility", color="Attribute",
                          barmode="group", text_auto='.2f')
        fig_util.update_layout(height=500)
        st.plotly_chart(fig_util, use_container_width=True)

    # --- Tab 3: 시뮬레이터 ---
    with tab3:
        st.subheader("🔎 제품 시장 선호도 시뮬레이션")
        st.write("다양한 옵션을 조합하여 가상 시장에서의 예상 선호도를 확인하세요.")
        
        sim_cols = st.columns(len(st.session_state.attributes))
        selected_config = {}
        
        for i, (attr, levels) in enumerate(st.session_state.attributes.items()):
            selected_config[attr] = sim_cols[i].selectbox(f"{attr} 선택", levels)
        
        # 예상 점수 계산
        score = model.params['const']
        for attr, lvl in selected_config.items():
            col_name = f"{attr}_{lvl}"
            score += model.params.get(col_name, 0)
        
        # 최대/최소 점수 계산
        max_score = model.params['const']
        for attr, levels in st.session_state.attributes.items():
            coeffs = [0] + [model.params.get(f"{attr}_{l}", 0) for l in levels[1:]]
            max_score += max(coeffs)
        
        min_score = model.params['const']
        for attr, levels in st.session_state.attributes.items():
            coeffs = [0] + [model.params.get(f"{attr}_{l}", 0) for l in levels[1:]]
            min_score += min(coeffs)
            
        preference_idx = (score - min_score) / (max_score - min_score) * 100 if (max_score != min_score) else 100
        
        # 게이지 차트
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = preference_idx,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "예상 선호 지수 (%)", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4f46e5"},
                'steps': [
                    {'range': [0, 40], 'color': "#fee2e2"},
                    {'range': [40, 70], 'color': "#fef3c7"},
                    {'range': [70, 100], 'color': "#dcfce7"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if preference_idx > 80:
            st.success("🌟 **강력 추천 조합!** 시장 점유율 확보 가능성이 매우 높은 조합입니다.")
        elif preference_idx > 50:
            st.info("✅ **안정적인 조합입니다.** 평균 이상의 선호도를 보입니다.")
        else:
            st.warning("⚠️ **개선이 필요합니다.** 타 속성 조합을 고려해보세요.")
