"""
[가격 수용성]PSM (Price Sensitivity Meter)
- 무거운 라이브러리(plotly, pandas, numpy, scipy)는 페이지 진입 시 지연 로딩
"""
import streamlit as st
import os
from core.ui_components import show_package_required_error, render_step2_data_selector

try:
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def page_psm():
    """PSM 분석 페이지"""
    if not show_package_required_error(HAS_SCIPY, "scipy", "scipy"):
        return
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("[가격 수용성]PSM (Price Sensitivity Meter)")
    st.markdown("**Van Westendorp 가격 민감도 분석** - 소비자의 가격 수용 범위를 측정합니다.")

    use_real_data, real_data_df = render_step2_data_selector("psm")

    st.markdown("---")
    
    # 데이터 입력 방식 선택 (설문 응답 결과가 있으면 해당 옵션 추가)
    psm_mode_options = ["가상 데이터 생성", "직접 입력", "CSV 파일 업로드"]
    if use_real_data and real_data_df is not None and len(real_data_df) > 0:
        psm_mode_options = ["AI 설문 응답 결과 사용"] + psm_mode_options
    data_mode = st.radio(
        "데이터 입력 방식",
        options=psm_mode_options,
        key="psm_data_mode"
    )
    
    # 1. 가상 데이터셋 생성 함수
    def generate_sample_data(n=100, base_price=50000):
        """가상 PSM 데이터 생성 (Too Cheap, Cheap, Expensive, Too Expensive)"""
        np.random.seed(42)
        
        # 현실적인 가격 응답 분포 생성 (tc < c < e < te 순서 보장)
        too_cheap = np.random.normal(base_price * 0.5, 5000, n)
        cheap = too_cheap + np.random.normal(base_price * 0.2, 3000, n)
        expensive = cheap + np.random.normal(base_price * 0.3, 4000, n)
        too_expensive = expensive + np.random.normal(base_price * 0.4, 6000, n)
        
        return pd.DataFrame({
            'too_cheap': np.maximum(too_cheap, 0),
            'cheap': np.maximum(cheap, 0),
            'expensive': np.maximum(expensive, 0),
            'too_expensive': np.maximum(too_expensive, 0)
        })
    
    # 데이터 준비
    psm_data = None
    
    if data_mode == "AI 설문 응답 결과 사용" and use_real_data and real_data_df is not None:
        st.info("💡 불러온 AI 설문 응답 결과로 PSM(가격 민감도) 분석을 진행합니다.")
        with st.expander("데이터 미리보기"):
            st.dataframe(real_data_df.head(20), use_container_width=True, height=300)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            too_cheap_col = st.selectbox("너무 쌈 컬럼", options=real_data_df.columns.tolist(), key="psm_survey_tc_col")
        with col2:
            cheap_col = st.selectbox("쌈 컬럼", options=real_data_df.columns.tolist(), key="psm_survey_c_col")
        with col3:
            expensive_col = st.selectbox("비쌈 컬럼", options=real_data_df.columns.tolist(), key="psm_survey_e_col")
        with col4:
            too_expensive_col = st.selectbox("너무 비쌈 컬럼", options=real_data_df.columns.tolist(), key="psm_survey_te_col")
        if st.button("설문 응답 데이터로 PSM 분석 적용", key="psm_apply_survey"):
            psm_data = pd.DataFrame({
                'too_cheap': pd.to_numeric(real_data_df[too_cheap_col], errors='coerce'),
                'cheap': pd.to_numeric(real_data_df[cheap_col], errors='coerce'),
                'expensive': pd.to_numeric(real_data_df[expensive_col], errors='coerce'),
                'too_expensive': pd.to_numeric(real_data_df[too_expensive_col], errors='coerce')
            }).dropna()
            if len(psm_data) > 0:
                st.session_state.psm_data = psm_data
                st.success(f"✅ 설문 응답 {len(psm_data)}명 데이터로 PSM 분석을 적용했습니다.")
            else:
                st.error("매핑한 컬럼에 유효한 숫자 데이터가 없습니다.")
        if 'psm_data' in st.session_state and st.session_state.psm_data is not None:
            psm_data = st.session_state.psm_data
    
    elif data_mode == "가상 데이터 생성":
        col1, col2 = st.columns(2)
        with col1:
            n_respondents = st.number_input("응답자 수", min_value=10, max_value=10000, value=100, step=10, key="psm_n_respondents")
        with col2:
            base_price = st.number_input("기준 가격 (원)", min_value=1000, max_value=10000000, value=50000, step=1000, key="psm_base_price")
        
        if st.button("가상 데이터 생성", key="psm_generate"):
            psm_data = generate_sample_data(n_respondents, base_price)
            st.session_state.psm_data = psm_data
            st.success(f"✅ {n_respondents}명의 가상 응답 데이터가 생성되었습니다.")
    
    elif data_mode == "직접 입력":
        st.info("💡 각 응답자별로 4가지 가격(너무 쌈, 쌈, 비쌈, 너무 비쌈)을 입력해주세요.")
        
        num_rows = st.number_input("응답자 수", min_value=1, max_value=1000, value=10, key="psm_num_rows")
        
        input_data = []
        for i in range(num_rows):
            with st.expander(f"응답자 {i+1}", expanded=(i < 3)):
                cols = st.columns(4)
                with cols[0]:
                    too_cheap = st.number_input("너무 쌈", min_value=0.0, value=25000.0, key=f"tc_{i}")
                with cols[1]:
                    cheap = st.number_input("쌈", min_value=0.0, value=35000.0, key=f"c_{i}")
                with cols[2]:
                    expensive = st.number_input("비쌈", min_value=0.0, value=50000.0, key=f"e_{i}")
                with cols[3]:
                    too_expensive = st.number_input("너무 비쌈", min_value=0.0, value=70000.0, key=f"te_{i}")
                
                input_data.append({
                    'too_cheap': too_cheap,
                    'cheap': cheap,
                    'expensive': expensive,
                    'too_expensive': too_expensive
                })
        
        if st.button("데이터 적용", key="psm_apply_manual"):
            psm_data = pd.DataFrame(input_data)
            st.session_state.psm_data = psm_data
            st.success(f"✅ {len(psm_data)}명의 응답 데이터가 적용되었습니다.")
    
    elif data_mode == "CSV 파일 업로드":
        uploaded_file = st.file_uploader("CSV 파일 업로드", type=['csv'], key="psm_upload")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head(10))
                
                # 컬럼 매핑
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    too_cheap_col = st.selectbox("너무 쌈 컬럼", options=df.columns.tolist(), key="psm_tc_col")
                with col2:
                    cheap_col = st.selectbox("쌈 컬럼", options=df.columns.tolist(), key="psm_c_col")
                with col3:
                    expensive_col = st.selectbox("비쌈 컬럼", options=df.columns.tolist(), key="psm_e_col")
                with col4:
                    too_expensive_col = st.selectbox("너무 비쌈 컬럼", options=df.columns.tolist(), key="psm_te_col")
                
                if st.button("데이터 적용", key="psm_apply_csv"):
                    psm_data = pd.DataFrame({
                        'too_cheap': pd.to_numeric(df[too_cheap_col], errors='coerce'),
                        'cheap': pd.to_numeric(df[cheap_col], errors='coerce'),
                        'expensive': pd.to_numeric(df[expensive_col], errors='coerce'),
                        'too_expensive': pd.to_numeric(df[too_expensive_col], errors='coerce')
                    }).dropna()
                    st.session_state.psm_data = psm_data
                    st.success(f"✅ {len(psm_data)}명의 응답 데이터가 적용되었습니다.")
            except Exception as e:
                st.error(f"파일 읽기 실패: {e}")
    
    # 세션 상태에서 데이터 가져오기
    if psm_data is None and 'psm_data' in st.session_state:
        psm_data = st.session_state.psm_data
    
    # PSM 분석 함수
    def analyze_psm(df):
        """PSM 분석 수행"""
        # 가격 범위 설정
        min_p = df.min().min() * 0.8
        max_p = df.max().max() * 1.2
        price_range = np.linspace(min_p, max_p, 200)
        
        n = len(df)
        
        # 누적 백분율 계산
        # '비쌈' 계열: 해당 가격 이하인 응답 비율 (정방향 누적)
        expensive_pct = [sum(df['expensive'] <= p) / n * 100 for p in price_range]
        too_expensive_pct = [sum(df['too_expensive'] <= p) / n * 100 for p in price_range]
        
        # '쌈' 계열: 해당 가격 이상인 응답 비율 (역방향 누적)
        cheap_inverted_pct = [sum(df['cheap'] >= p) / n * 100 for p in price_range]
        too_cheap_inverted_pct = [sum(df['too_cheap'] >= p) / n * 100 for p in price_range]
        
        psm_df = pd.DataFrame({
            'price': price_range,
            'too_cheap_inv': too_cheap_inverted_pct,
            'cheap_inv': cheap_inverted_pct,
            'expensive': expensive_pct,
            'too_expensive': too_expensive_pct
        })
        
        # 교차점 탐색 함수 (선형 보간)
        def find_intersection(y1_name, y2_name):
            diff = psm_df[y1_name] - psm_df[y2_name]
            for i in range(len(diff) - 1):
                if diff[i] * diff[i+1] <= 0:
                    # 선형 보간법 적용
                    x1, x2 = psm_df['price'].iloc[i], psm_df['price'].iloc[i+1]
                    y1a, y1b = psm_df[y1_name].iloc[i], psm_df[y1_name].iloc[i+1]
                    y2a, y2b = psm_df[y2_name].iloc[i], psm_df[y2_name].iloc[i+1]
                    
                    denom = (y1b - y1a) - (y2b - y2a)
                    if abs(denom) < 1e-6:
                        return x1, y1a
                    t = (y2a - y1a) / denom
                    intersect_x = x1 + t * (x2 - x1)
                    intersect_y = y1a + t * (y1b - y1a)
                    return intersect_x, intersect_y
            return None, None
        
        # 핵심 지표 산출
        opp_x, opp_y = find_intersection('too_cheap_inv', 'too_expensive')  # Optimal Price Point
        ipp_x, ipp_y = find_intersection('cheap_inv', 'expensive')        # Indifference Price Point
        pmc_x, pmc_y = find_intersection('too_cheap_inv', 'expensive')    # Point of Marginal Cheapness
        pme_x, pme_y = find_intersection('cheap_inv', 'too_expensive')    # Point of Marginal Expensiveness
        
        return psm_df, {
            'OPP': (opp_x, opp_y),
            'IPP': (ipp_x, ipp_y),
            'PMC': (pmc_x, pmc_y),
            'PME': (pme_x, pme_y)
        }
    
    # 분석 실행
    if psm_data is not None and len(psm_data) > 0:
        st.markdown("---")
        st.subheader("📊 PSM 분석 결과")
        
        # 데이터 유효성 검사
        invalid_rows = []
        for idx, row in psm_data.iterrows():
            if not (row['too_cheap'] < row['cheap'] < row['expensive'] < row['too_expensive']):
                invalid_rows.append(idx)
        
        if invalid_rows:
            st.warning(f"⚠️ {len(invalid_rows)}개의 응답이 논리적 순서(너무 쌈 < 쌈 < 비쌈 < 너무 비쌈)를 따르지 않습니다. 해당 행은 분석에서 제외됩니다.")
            psm_data = psm_data.drop(invalid_rows).reset_index(drop=True)
        
        if len(psm_data) > 0:
            try:
                results_df, points = analyze_psm(psm_data)
                
                # 결과 요약 카드
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if points['OPP'][0]:
                        st.metric("최적 가격점 (OPP)", f"{int(points['OPP'][0]):,}원", 
                                 help="Too Cheap과 Too Expensive의 교차점")
                    else:
                        st.metric("최적 가격점 (OPP)", "N/A")
                
                with col2:
                    if points['IPP'][0]:
                        st.metric("무관심 가격점 (IPP)", f"{int(points['IPP'][0]):,}원",
                                 help="Cheap과 Expensive의 교차점")
                    else:
                        st.metric("무관심 가격점 (IPP)", "N/A")
                
                with col3:
                    if points['PMC'][0] and points['PME'][0]:
                        st.metric("가격 수용 범위 (RA)", 
                                 f"{int(points['PMC'][0]):,}원\n~ {int(points['PME'][0]):,}원",
                                 help="Point of Marginal Cheapness ~ Point of Marginal Expensiveness")
                    else:
                        st.metric("가격 수용 범위 (RA)", "N/A")
                
                with col4:
                    st.metric("응답자 수", f"{len(psm_data)}명")
                
                # 시각화
                fig = go.Figure()
                
                # 곡선 그리기
                fig.add_trace(go.Scatter(
                    x=results_df['price'],
                    y=results_df['too_cheap_inv'],
                    mode='lines',
                    name='Too Cheap (Inverted)',
                    line=dict(color='#10b981', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=results_df['price'],
                    y=results_df['cheap_inv'],
                    mode='lines',
                    name='Cheap (Inverted)',
                    line=dict(color='#3b82f6', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=results_df['price'],
                    y=results_df['expensive'],
                    mode='lines',
                    name='Expensive',
                    line=dict(color='#f59e0b', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=results_df['price'],
                    y=results_df['too_expensive'],
                    mode='lines',
                    name='Too Expensive',
                    line=dict(color='#ef4444', width=2)
                ))
                
                # 수용 범위 강조
                if points['PMC'][0] and points['PME'][0]:
                    fig.add_vrect(
                        x0=points['PMC'][0],
                        x1=points['PME'][0],
                        fillcolor="gray",
                        opacity=0.1,
                        layer="below",
                        line_width=0,
                        annotation_text="Range of Acceptability (RA)",
                        annotation_position="top left"
                    )
                
                # 교차점 표시
                for name, (x, y) in points.items():
                    if x:
                        fig.add_trace(go.Scatter(
                            x=[x],
                            y=[y],
                            mode='markers+text',
                            name=name,
                            marker=dict(size=12, color='red', symbol='diamond'),
                            text=[name],
                            textposition="top center",
                            showlegend=False
                        ))
                
                fig.update_layout(
                    title="Van Westendorp Price Sensitivity Meter (PSM) Analysis",
                    xaxis_title="가격 (원)",
                    yaxis_title="응답자 비율 (%)",
                    height=600,
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 상세 결과 테이블
                with st.expander("📋 상세 분석 결과"):
                    results_table = []
                    for name, (x, y) in points.items():
                        if x:
                            results_table.append({
                                '지표': name,
                                '가격 (원)': f"{int(x):,}",
                                '응답자 비율 (%)': f"{y:.2f}",
                                '설명': {
                                    'OPP': '최적 가격점: Too Cheap과 Too Expensive의 교차점',
                                    'IPP': '무관심 가격점: Cheap과 Expensive의 교차점',
                                    'PMC': '한계 저가점: Too Cheap과 Expensive의 교차점',
                                    'PME': '한계 고가점: Cheap과 Too Expensive의 교차점'
                                }.get(name, '')
                            })
                    
                    if results_table:
                        st.table(pd.DataFrame(results_table))
                
                # 데이터 다운로드
                st.markdown("---")
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    csv_results = results_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        "분석 결과 CSV 다운로드",
                        data=csv_results,
                        file_name="psm_analysis_results.csv",
                        mime="text/csv"
                    )
                with col_dl2:
                    csv_data = psm_data.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        "원본 데이터 CSV 다운로드",
                        data=csv_data,
                        file_name="psm_raw_data.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("👆 위에서 데이터를 입력하거나 생성해주세요.")
