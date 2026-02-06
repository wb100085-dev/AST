"""
[시장 확산 예측]Bass 확산 모델 (Bass Diffusion Model)
- 무거운 라이브러리(plotly, pandas, numpy, scipy)는 페이지 진입 시 지연 로딩
"""
import streamlit as st
import os
from core.ui_components import show_package_required_error, render_step2_data_selector

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def page_bass():
    """Bass 확산 모델 페이지"""
    if not show_package_required_error(HAS_SCIPY, "scipy", "scipy"):
        return
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("[시장 확산 예측]Bass 확산 모델 (Bass Diffusion Model)")
    st.markdown("**Bass 확산 모델** - 신제품/서비스의 시장 확산 패턴을 예측합니다.")

    use_real_data, real_data_df = render_step2_data_selector("bass")

    st.markdown("---")
    
    # 데이터 입력 방식 선택 (설문 응답 결과가 있으면 해당 옵션 추가)
    bass_mode_options = ["가상 데이터 생성", "직접 입력", "CSV 파일 업로드"]
    if use_real_data and real_data_df is not None and len(real_data_df) > 0:
        bass_mode_options = ["AI 설문 응답 결과 사용"] + bass_mode_options
    data_mode = st.radio(
        "데이터 입력 방식",
        options=bass_mode_options,
        key="bass_data_mode"
    )
    
    # Bass 확산 모델 함수
    def bass_model(t, p, q, m):
        """
        Bass 확산 모델
        F(t) = m * (1 - exp(-(p+q)*t)) / (1 + (q/p)*exp(-(p+q)*t))
        
        Parameters:
        - p: 혁신 계수 (Innovation coefficient)
        - q: 모방 계수 (Imitation coefficient)
        - m: 잠재 시장 크기 (Market potential)
        - t: 시간
        """
        if p + q == 0:
            return np.zeros_like(t)
        exp_term = np.exp(-(p + q) * t)
        numerator = 1 - exp_term
        denominator = 1 + (q / p) * exp_term
        return m * numerator / denominator
    
    def bass_cumulative_adoptions(t, p, q, m):
        """누적 채택자 수"""
        return bass_model(t, p, q, m)
    
    def bass_new_adoptions(t, p, q, m):
        """신규 채택자 수 (시간당)"""
        if p + q == 0:
            return np.zeros_like(t)
        exp_term = np.exp(-(p + q) * t)
        numerator = (p + q) ** 2 * exp_term
        denominator = (1 + (q / p) * exp_term) ** 2
        return m * p * numerator / denominator
    
    # 데이터 준비
    bass_data = None
    
    if data_mode == "AI 설문 응답 결과 사용" and use_real_data and real_data_df is not None:
        st.info("💡 불러온 AI 설문 응답 결과로 Bass 확산 모델 분석을 진행합니다. 기간·누적 채택자 수에 해당하는 컬럼을 매핑하세요.")
        with st.expander("데이터 미리보기"):
            st.dataframe(real_data_df.head(20), use_container_width=True, height=300)
        col1, col2 = st.columns(2)
        with col1:
            period_col = st.selectbox("기간 컬럼", options=real_data_df.columns.tolist(), key="bass_survey_period_col")
        with col2:
            cumulative_col = st.selectbox("누적 채택자 수 컬럼", options=real_data_df.columns.tolist(), key="bass_survey_cumulative_col")
        if st.button("설문 응답 데이터로 Bass 분석 적용", key="bass_apply_survey"):
            bass_data = pd.DataFrame({
                'period': pd.to_numeric(real_data_df[period_col], errors='coerce'),
                'cumulative_adoptions': pd.to_numeric(real_data_df[cumulative_col], errors='coerce')
            }).dropna()
            bass_data = bass_data.sort_values('period').reset_index(drop=True)
            if len(bass_data) >= 3:
                bass_data['new_adoptions'] = np.diff(np.concatenate([[0], bass_data['cumulative_adoptions'].values]))
                st.session_state.bass_data = bass_data
                st.success(f"✅ 설문 응답 데이터 {len(bass_data)}개 기간으로 Bass 분석을 적용했습니다.")
            else:
                st.error("유효한 데이터가 부족합니다. 기간·누적 채택자 수 컬럼을 확인하세요. (최소 3개 기간 필요)")
        if 'bass_data' in st.session_state and st.session_state.bass_data is not None:
            bass_data = st.session_state.bass_data
    
    elif data_mode == "가상 데이터 생성":
        col1, col2, col3 = st.columns(3)
        with col1:
            market_potential = st.number_input("잠재 시장 크기 (m)", min_value=100, max_value=10000000, value=10000, step=100, key="bass_m")
        with col2:
            innovation_coef = st.number_input("혁신 계수 (p)", min_value=0.001, max_value=1.0, value=0.03, step=0.001, format="%.3f", key="bass_p")
        with col3:
            imitation_coef = st.number_input("모방 계수 (q)", min_value=0.001, max_value=1.0, value=0.38, step=0.001, format="%.3f", key="bass_q")
        
        time_periods = st.number_input("예측 기간 (개월)", min_value=12, max_value=120, value=60, step=6, key="bass_periods")
        
        if st.button("가상 데이터 생성", key="bass_generate"):
            # 실제 데이터 생성 (노이즈 추가)
            np.random.seed(42)
            time_points = np.arange(0, time_periods)
            true_cumulative = bass_cumulative_adoptions(time_points, innovation_coef, imitation_coef, market_potential)
            # 노이즈 추가 (5% 표준편차)
            noise = np.random.normal(0, true_cumulative * 0.05, len(time_points))
            observed_cumulative = np.maximum(true_cumulative + noise, 0)
            
            bass_data = pd.DataFrame({
                'period': time_points,
                'cumulative_adoptions': observed_cumulative,
                'new_adoptions': np.diff(np.concatenate([[0], observed_cumulative]))
            })
            st.session_state.bass_data = bass_data
            st.success(f"✅ {time_periods}개월의 가상 확산 데이터가 생성되었습니다.")
    
    elif data_mode == "직접 입력":
        st.info("💡 각 기간별 누적 채택자 수를 입력해주세요.")
        
        num_periods = st.number_input("기간 수", min_value=3, max_value=120, value=12, key="bass_num_periods")
        
        input_data = []
        for i in range(num_periods):
            cumulative = st.number_input(f"기간 {i+1} 누적 채택자 수", min_value=0.0, value=float(i * 100), key=f"bass_cum_{i}")
            input_data.append({
                'period': i,
                'cumulative_adoptions': cumulative
            })
        
        if st.button("데이터 적용", key="bass_apply_manual"):
            bass_data = pd.DataFrame(input_data)
            bass_data['new_adoptions'] = np.diff(np.concatenate([[0], bass_data['cumulative_adoptions']]))
            st.session_state.bass_data = bass_data
            st.success(f"✅ {len(bass_data)}개 기간의 데이터가 적용되었습니다.")
    
    elif data_mode == "CSV 파일 업로드":
        uploaded_file = st.file_uploader("CSV 파일 업로드", type=['csv'], key="bass_upload")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head(10))
                
                # 컬럼 매핑
                col1, col2 = st.columns(2)
                with col1:
                    period_col = st.selectbox("기간 컬럼", options=df.columns.tolist(), key="bass_period_col")
                with col2:
                    cumulative_col = st.selectbox("누적 채택자 수 컬럼", options=df.columns.tolist(), key="bass_cumulative_col")
                
                if st.button("데이터 적용", key="bass_apply_csv"):
                    bass_data = pd.DataFrame({
                        'period': pd.to_numeric(df[period_col], errors='coerce'),
                        'cumulative_adoptions': pd.to_numeric(df[cumulative_col], errors='coerce')
                    }).dropna()
                    bass_data = bass_data.sort_values('period').reset_index(drop=True)
                    bass_data['new_adoptions'] = np.diff(np.concatenate([[0], bass_data['cumulative_adoptions']]))
                    st.session_state.bass_data = bass_data
                    st.success(f"✅ {len(bass_data)}개 기간의 데이터가 적용되었습니다.")
            except Exception as e:
                st.error(f"파일 읽기 실패: {e}")
    
    # 세션 상태에서 데이터 가져오기
    if bass_data is None and 'bass_data' in st.session_state:
        bass_data = st.session_state.bass_data
    
    # 분석 실행
    if bass_data is not None and len(bass_data) > 0:
        st.markdown("---")
        st.subheader("📊 Bass 확산 모델 분석 결과")
        
        try:
            # 모델 파라미터 추정
            time_data = bass_data['period'].values
            cumulative_data = bass_data['cumulative_adoptions'].values
            
            # 초기 추정값
            m_initial = cumulative_data[-1] * 1.5  # 최종값의 1.5배를 잠재 시장으로 추정
            p_initial = 0.01
            q_initial = 0.3
            
            # 곡선 피팅
            try:
                popt, pcov = curve_fit(
                    bass_cumulative_adoptions,
                    time_data,
                    cumulative_data,
                    p0=[p_initial, q_initial, m_initial],
                    bounds=([0.001, 0.001, cumulative_data[-1]], [1.0, 1.0, cumulative_data[-1] * 10]),
                    maxfev=10000
                )
                p_estimated, q_estimated, m_estimated = popt
                
                # 표준 오차 계산
                perr = np.sqrt(np.diag(pcov))
                p_err, q_err, m_err = perr
                
            except Exception as e:
                st.warning(f"자동 파라미터 추정 실패: {e}. 기본값을 사용합니다.")
                p_estimated, q_estimated, m_estimated = p_initial, q_initial, m_initial
                p_err, q_err, m_err = 0, 0, 0
            
            # 결과 요약 카드
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("혁신 계수 (p)", f"{p_estimated:.4f}", 
                         delta=f"±{p_err:.4f}" if p_err > 0 else None,
                         help="혁신자(Innovators)의 영향력")
            
            with col2:
                st.metric("모방 계수 (q)", f"{q_estimated:.4f}",
                         delta=f"±{q_err:.4f}" if q_err > 0 else None,
                         help="모방자(Imitators)의 영향력")
            
            with col3:
                st.metric("잠재 시장 크기 (m)", f"{int(m_estimated):,}명",
                         delta=f"±{int(m_err):,}" if m_err > 0 else None,
                         help="최종 채택 가능한 총 시장 크기")
            
            with col4:
                peak_time = np.log(q_estimated / p_estimated) / (p_estimated + q_estimated) if p_estimated > 0 and q_estimated > 0 else 0
                st.metric("최대 성장 시점", f"{int(peak_time):.0f}개월",
                         help="신규 채택이 최대가 되는 시점")
            
            # 예측 기간 설정
            forecast_periods = st.slider("예측 기간 (개월)", min_value=len(time_data), max_value=120, 
                                        value=min(len(time_data) * 2, 120), key="bass_forecast")
            
            # 예측 데이터 생성
            forecast_time = np.arange(0, forecast_periods)
            forecast_cumulative = bass_cumulative_adoptions(forecast_time, p_estimated, q_estimated, m_estimated)
            forecast_new = bass_new_adoptions(forecast_time, p_estimated, q_estimated, m_estimated)
            
            # 시각화
            tab1, tab2, tab3 = st.tabs(["📈 누적 채택자", "📊 신규 채택자", "📋 상세 결과"])
            
            with tab1:
                fig_cum = go.Figure()
                
                # 실제 데이터
                fig_cum.add_trace(go.Scatter(
                    x=time_data,
                    y=cumulative_data,
                    mode='markers+lines',
                    name='실제 데이터',
                    line=dict(color='#3b82f6', width=2),
                    marker=dict(size=8)
                ))
                
                # 예측 데이터
                fig_cum.add_trace(go.Scatter(
                    x=forecast_time,
                    y=forecast_cumulative,
                    mode='lines',
                    name='Bass 모델 예측',
                    line=dict(color='#10b981', width=2, dash='dash')
                ))
                
                # 현재 시점 표시
                if len(time_data) < forecast_periods:
                    fig_cum.add_vline(
                        x=len(time_data) - 1,
                        line_dash="dot",
                        line_color="gray",
                        annotation_text="현재 시점"
                    )
                
                fig_cum.update_layout(
                    title="누적 채택자 수 예측",
                    xaxis_title="기간 (개월)",
                    yaxis_title="누적 채택자 수",
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_cum, use_container_width=True)
            
            with tab2:
                fig_new = go.Figure()
                
                # 실제 신규 채택자
                if 'new_adoptions' in bass_data.columns:
                    fig_new.add_trace(go.Scatter(
                        x=time_data,
                        y=bass_data['new_adoptions'].values,
                        mode='markers+lines',
                        name='실제 신규 채택',
                        line=dict(color='#3b82f6', width=2),
                        marker=dict(size=8)
                    ))
                
                # 예측 신규 채택자
                fig_new.add_trace(go.Scatter(
                    x=forecast_time,
                    y=forecast_new,
                    mode='lines',
                    name='예측 신규 채택',
                    line=dict(color='#10b981', width=2, dash='dash')
                ))
                
                # 최대 성장 시점 표시
                if peak_time > 0 and peak_time < forecast_periods:
                    peak_new = bass_new_adoptions(np.array([peak_time]), p_estimated, q_estimated, m_estimated)[0]
                    fig_new.add_trace(go.Scatter(
                        x=[peak_time],
                        y=[peak_new],
                        mode='markers',
                        name='최대 성장 시점',
                        marker=dict(size=15, color='red', symbol='diamond')
                    ))
                
                fig_new.update_layout(
                    title="신규 채택자 수 예측",
                    xaxis_title="기간 (개월)",
                    yaxis_title="신규 채택자 수",
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_new, use_container_width=True)
            
            with tab3:
                # 상세 결과 테이블
                results_df = pd.DataFrame({
                    '기간': forecast_time,
                    '누적 채택자': forecast_cumulative.astype(int),
                    '신규 채택자': forecast_new.astype(int),
                    '채택률 (%)': (forecast_cumulative / m_estimated * 100).round(2)
                })
                
                st.dataframe(results_df, use_container_width=True, height=400)
                
                # 주요 지표
                st.markdown("### 주요 지표")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    - **혁신 계수 (p)**: {p_estimated:.4f} ± {p_err:.4f}
                    - **모방 계수 (q)**: {q_estimated:.4f} ± {q_err:.4f}
                    - **잠재 시장 (m)**: {int(m_estimated):,}명 ± {int(m_err):,}명
                    """)
                with col2:
                    st.markdown(f"""
                    - **최대 성장 시점**: {int(peak_time)}개월
                    - **최대 신규 채택**: {int(bass_new_adoptions(np.array([peak_time]), p_estimated, q_estimated, m_estimated)[0]):,}명/월
                    - **현재 채택률**: {(cumulative_data[-1] / m_estimated * 100):.2f}%
                    """)
            
            # 데이터 다운로드
            st.markdown("---")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                csv_results = results_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "예측 결과 CSV 다운로드",
                    data=csv_results,
                    file_name="bass_forecast_results.csv",
                    mime="text/csv"
                )
            with col_dl2:
                csv_data = bass_data.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "원본 데이터 CSV 다운로드",
                    data=csv_data,
                    file_name="bass_raw_data.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("👆 위에서 데이터를 입력하거나 생성해주세요.")
