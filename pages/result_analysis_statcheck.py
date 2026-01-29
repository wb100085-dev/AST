"""
[가설 검증]A/B 테스트 검증 (Statistical Significance)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
import os
from utils.step2_records import list_step2_records, load_step2_record

def page_statcheck():
    """A/B 테스트 검증 페이지"""
    # scipy 확인
    if not HAS_SCIPY:
        st.error("""
        ⚠️ **필수 패키지 누락**
        
        `scipy` 모듈이 설치되어 있지 않습니다. 다음 명령으로 설치해주세요:
        
        ```bash
        pip install scipy
        ```
        
        또는 가상환경이 활성화된 상태에서:
        
        ```bash
        .\\venv\\Scripts\\Activate.ps1
        pip install scipy
        ```
        
        설치 후 페이지를 새로고침해주세요.
        """)
        return
    
    st.title("[가설 검증]A/B 테스트 검증 (Statistical Significance)")
    st.markdown("**A/B 테스트 통계적 유의성 검증** - 두 그룹 간 차이의 통계적 유의성을 검증합니다.")
    
    # 가상인구 데이터 불러오기 섹션
    st.markdown("### 가상인구 데이터 불러오기")
    records = list_step2_records()

    use_real_data = False
    real_data_df = None

    if records:
        record_options = {}
        for r in records:
            ts = r.get("timestamp", "")
            sido_name = r.get("sido_name", "")
            rows = r.get("rows", 0)
            cols = r.get("columns_count", 0)
            label = f"{ts} | {sido_name} | {rows}명 | {cols}개 컬럼"
            record_options[label] = r
        
        use_real_data_option = st.checkbox("2차 대입 결과 데이터 사용", value=False, key="statcheck_use_real_data")
        
        if use_real_data_option:
            selected_label = st.selectbox(
                "2차 대입 결과에서 가상인구 데이터를 선택하세요:",
                options=list(record_options.keys()),
                index=0 if not st.session_state.get("statcheck_selected_record_label") else 
                      (list(record_options.keys()).index(st.session_state.statcheck_selected_record_label) 
                       if st.session_state.statcheck_selected_record_label in record_options else 0),
                key="statcheck_record_select"
            )
            
            if selected_label and selected_label in record_options:
                selected_record = record_options[selected_label]
                excel_path = selected_record.get("excel_path", "")
                
                if excel_path and os.path.isfile(excel_path):
                    try:
                        real_data_df = load_step2_record(excel_path)
                        st.session_state.statcheck_population_df = real_data_df
                        st.session_state.statcheck_selected_record_label = selected_label
                        use_real_data = True
                        st.success(f"✅ 가상인구 데이터를 불러왔습니다. (총 {len(real_data_df)}명, {len(real_data_df.columns)}개 컬럼)")
                        
                        with st.expander("불러온 데이터 미리보기"):
                            st.dataframe(real_data_df.head(20), use_container_width=True, height=300)
                    except Exception as e:
                        st.error(f"데이터 로드 실패: {e}")
        else:
            st.session_state.statcheck_population_df = None
            st.info("직접 입력 방식을 사용합니다. 2차 대입 결과를 사용하려면 위 체크박스를 선택하세요.")
    else:
        st.info("아직 2차 대입 결과가 없습니다. 가상인구 생성 후 2단계에서 통계를 대입하면 여기서 불러올 수 있습니다.")

    st.markdown("---")
    
    # 데이터 입력 방식
    if use_real_data and real_data_df is not None:
        st.subheader("📊 데이터에서 그룹 선택")
        st.info("💡 불러온 데이터에서 A/B 테스트 그룹을 선택하세요.")
        
        # 그룹 컬럼 선택
        group_col = st.selectbox(
            "그룹 구분 컬럼 선택",
            options=real_data_df.columns.tolist(),
            key="statcheck_group_col"
        )
        
        # 측정 지표 컬럼 선택
        metric_col = st.selectbox(
            "측정 지표 컬럼 선택 (전환율, 클릭률 등)",
            options=real_data_df.columns.tolist(),
            key="statcheck_metric_col"
        )
        
        if st.button("그룹별 통계 계산", key="statcheck_calc_from_data"):
            try:
                groups = real_data_df[group_col].unique()
                if len(groups) >= 2:
                    group_a_name = groups[0]
                    group_b_name = groups[1]
                    
                    group_a_data = real_data_df[real_data_df[group_col] == group_a_name][metric_col]
                    group_b_data = real_data_df[real_data_df[group_col] == group_b_name][metric_col]
                    
                    # 이진 데이터인지 확인 (0/1 또는 True/False)
                    if group_a_data.nunique() <= 2 and group_b_data.nunique() <= 2:
                        # 전환율 계산
                        n_a = len(group_a_data)
                        n_b = len(group_b_data)
                        c_a = group_a_data.sum()
                        c_b = group_b_data.sum()
                        
                        st.session_state.statcheck_group_a = {'name': str(group_a_name), 'sampleSize': n_a, 'conversions': int(c_a)}
                        st.session_state.statcheck_group_b = {'name': str(group_b_name), 'sampleSize': n_b, 'conversions': int(c_b)}
                        st.success(f"✅ 그룹 A: {group_a_name} ({n_a}명, {c_a}건), 그룹 B: {group_b_name} ({n_b}명, {c_b}건)")
                    else:
                        # 연속형 데이터
                        st.session_state.statcheck_group_a = {'name': str(group_a_name), 'data': group_a_data.tolist()}
                        st.session_state.statcheck_group_b = {'name': str(group_b_name), 'data': group_b_data.tolist()}
                        st.success(f"✅ 그룹 A: {group_a_name} ({len(group_a_data)}명), 그룹 B: {group_b_name} ({len(group_b_data)}명)")
                else:
                    st.error("그룹이 2개 이상 필요합니다.")
            except Exception as e:
                st.error(f"계산 실패: {e}")
    else:
        st.subheader("📝 직접 입력")
    
    # 그룹 A 입력
    st.markdown("### 그룹 A (대조군/Control)")
    col1, col2 = st.columns(2)
    with col1:
        group_a_name = st.text_input("그룹 이름", value="대조군 (A)", key="statcheck_a_name")
        if 'statcheck_group_a' in st.session_state and 'sampleSize' in st.session_state.statcheck_group_a:
            n_a = st.number_input("표본 크기 (n)", min_value=1, max_value=10000000, 
                                 value=st.session_state.statcheck_group_a.get('sampleSize', 1000), key="statcheck_a_n")
            c_a = st.number_input("전환 수 (c)", min_value=0, max_value=n_a,
                                 value=st.session_state.statcheck_group_a.get('conversions', 50), key="statcheck_a_c")
        else:
            n_a = st.number_input("표본 크기 (n)", min_value=1, max_value=10000000, value=1000, key="statcheck_a_n")
            c_a = st.number_input("전환 수 (c)", min_value=0, max_value=n_a, value=50, key="statcheck_a_c")
    
    with col2:
        p_a = (c_a / n_a * 100) if n_a > 0 else 0
        st.metric("전환율", f"{p_a:.2f}%", help=f"{c_a} / {n_a}")
    
    # 그룹 B 입력
    st.markdown("### 그룹 B (실험군/Test)")
    col1, col2 = st.columns(2)
    with col1:
        group_b_name = st.text_input("그룹 이름", value="실험군 (B)", key="statcheck_b_name")
        if 'statcheck_group_b' in st.session_state and 'sampleSize' in st.session_state.statcheck_group_b:
            n_b = st.number_input("표본 크기 (n)", min_value=1, max_value=10000000,
                                 value=st.session_state.statcheck_group_b.get('sampleSize', 1000), key="statcheck_b_n")
            c_b = st.number_input("전환 수 (c)", min_value=0, max_value=n_b,
                                 value=st.session_state.statcheck_group_b.get('conversions', 75), key="statcheck_b_c")
        else:
            n_b = st.number_input("표본 크기 (n)", min_value=1, max_value=10000000, value=1000, key="statcheck_b_n")
            c_b = st.number_input("전환 수 (c)", min_value=0, max_value=n_b, value=75, key="statcheck_b_c")
    
    with col2:
        p_b = (c_b / n_b * 100) if n_b > 0 else 0
        st.metric("전환율", f"{p_b:.2f}%", help=f"{c_b} / {n_b}")
    
    # 통계 검정 함수
    def calculate_z_test(n_a, c_a, n_b, c_b):
        """Z-검정 수행 (비율 비교)"""
        p_a = c_a / n_a if n_a > 0 else 0
        p_b = c_b / n_b if n_b > 0 else 0
        
        # 통합 비율
        p_pooled = (c_a + c_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0
        
        # 표준 오차
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b)) if p_pooled > 0 else 0
        
        # Z-점수
        z_score = (p_b - p_a) / se if se > 0 else 0
        
        # 양측 p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # 신뢰구간 (95%)
        diff = p_b - p_a
        margin = 1.96 * se
        ci_lower = diff - margin
        ci_upper = diff + margin
        
        # Uplift
        uplift = ((p_b - p_a) / p_a * 100) if p_a > 0 else 0
        
        return {
            'p_a': p_a,
            'p_b': p_b,
            'z_score': z_score,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'uplift': uplift,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'se': se
        }
    
    # 분석 실행
    if n_a > 0 and n_b > 0:
        st.markdown("---")
        st.subheader("📊 통계적 유의성 검증 결과")
        
        results = calculate_z_test(n_a, c_a, n_b, c_b)
        
        # 결과 요약 카드
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            color = "normal" if results['is_significant'] else "off"
            st.metric(
                "통계적 유의성",
                "✅ 유의함" if results['is_significant'] else "❌ 유의하지 않음",
                delta=f"p = {results['p_value']:.4f}",
                delta_color=color
            )
        
        with col2:
            trend = "up" if results['uplift'] > 0 else "down" if results['uplift'] < 0 else "neutral"
            st.metric(
                "전환율 향상도 (Uplift)",
                f"{results['uplift']:.2f}%",
                help=f"그룹 B가 그룹 A보다 {abs(results['uplift']):.2f}% {'높음' if results['uplift'] > 0 else '낮음'}"
            )
        
        with col3:
            st.metric(
                "Z-Score",
                f"{results['z_score']:.4f}",
                help="표준화된 검정 통계량"
            )
        
        with col4:
            st.metric(
                "P-Value",
                f"{results['p_value']:.6f}",
                help="유의확률 (α=0.05 기준)"
            )
        
        # 시각화
        tab1, tab2, tab3 = st.tabs(["📈 전환율 비교", "📊 신뢰구간", "📋 상세 결과"])
        
        with tab1:
            # 전환율 비교 차트
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=[group_a_name, group_b_name],
                y=[results['p_a'] * 100, results['p_b'] * 100],
                name='전환율',
                marker_color=['#3b82f6', '#10b981'],
                text=[f"{results['p_a']*100:.2f}%", f"{results['p_b']*100:.2f}%"],
                textposition='outside'
            ))
            
            # 오차 막대 (신뢰구간)
            fig.add_trace(go.Scatter(
                x=[group_a_name, group_b_name],
                y=[results['p_a'] * 100, results['p_b'] * 100],
                error_y=dict(
                    type='data',
                    array=[1.96 * np.sqrt(results['p_a'] * (1 - results['p_a']) / n_a) * 100,
                           1.96 * np.sqrt(results['p_b'] * (1 - results['p_b']) / n_b) * 100],
                    visible=True
                ),
                mode='markers',
                marker=dict(size=0),
                showlegend=False
            ))
            
            fig.update_layout(
                title="전환율 비교",
                xaxis_title="그룹",
                yaxis_title="전환율 (%)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # 신뢰구간 차트
            fig_ci = go.Figure()
            
            diff = results['p_b'] - results['p_a']
            fig_ci.add_trace(go.Scatter(
                x=[diff * 100],
                y=[0],
                mode='markers',
                marker=dict(size=20, color='blue'),
                name='차이',
                text=[f"{diff*100:.2f}%"],
                textposition='top center'
            ))
            
            # 신뢰구간
            fig_ci.add_trace(go.Scatter(
                x=[results['ci_lower'] * 100, results['ci_upper'] * 100],
                y=[0, 0],
                mode='lines',
                line=dict(width=10, color='gray'),
                name='95% 신뢰구간'
            ))
            
            # 0선 표시
            fig_ci.add_vline(
                x=0,
                line_dash="dash",
                line_color="red",
                annotation_text="차이 없음"
            )
            
            fig_ci.update_layout(
                title="전환율 차이의 95% 신뢰구간",
                xaxis_title="차이 (%)",
                yaxis_title="",
                height=300,
                yaxis=dict(showticklabels=False, range=[-0.5, 0.5])
            )
            st.plotly_chart(fig_ci, use_container_width=True)
        
        with tab3:
            # 상세 결과 테이블
            st.markdown("### 상세 통계 결과")
            
            results_table = pd.DataFrame({
                '지표': [
                    '그룹 A 전환율',
                    '그룹 B 전환율',
                    '전환율 차이',
                    'Uplift',
                    'Z-Score',
                    'P-Value',
                    '95% 신뢰구간 하한',
                    '95% 신뢰구간 상한',
                    '통계적 유의성'
                ],
                '값': [
                    f"{results['p_a']*100:.4f}%",
                    f"{results['p_b']*100:.4f}%",
                    f"{(results['p_b']-results['p_a'])*100:.4f}%",
                    f"{results['uplift']:.2f}%",
                    f"{results['z_score']:.4f}",
                    f"{results['p_value']:.6f}",
                    f"{results['ci_lower']*100:.4f}%",
                    f"{results['ci_upper']*100:.4f}%",
                    "✅ 유의함" if results['is_significant'] else "❌ 유의하지 않음"
                ]
            })
            st.table(results_table)
            
            # 해석
            st.markdown("### 결과 해석")
            if results['is_significant']:
                if results['uplift'] > 0:
                    st.success(f"""
                    ✅ **통계적으로 유의한 차이가 있습니다.**
                    
                    - 그룹 B({group_b_name})가 그룹 A({group_a_name})보다 **{results['uplift']:.2f}%** 높은 전환율을 보입니다.
                    - 이 차이는 우연이 아닌 실제 효과로 보입니다 (p = {results['p_value']:.6f} < 0.05).
                    - 95% 신뢰구간: [{results['ci_lower']*100:.2f}%, {results['ci_upper']*100:.2f}%]
                    """)
                else:
                    st.warning(f"""
                    ⚠️ **통계적으로 유의한 차이가 있습니다.**
                    
                    - 그룹 B({group_b_name})가 그룹 A({group_a_name})보다 **{abs(results['uplift']):.2f}%** 낮은 전환율을 보입니다.
                    - 이 차이는 우연이 아닌 실제 효과로 보입니다 (p = {results['p_value']:.6f} < 0.05).
                    """)
            else:
                st.info(f"""
                ℹ️ **통계적으로 유의한 차이가 없습니다.**
                
                - 두 그룹 간의 전환율 차이는 우연일 가능성이 높습니다 (p = {results['p_value']:.6f} ≥ 0.05).
                - 더 많은 표본을 모으거나 실험을 재설계하는 것을 고려해보세요.
                """)
        
        # 데이터 다운로드
        st.markdown("---")
        results_summary = pd.DataFrame({
            '그룹': [group_a_name, group_b_name],
            '표본 크기': [n_a, n_b],
            '전환 수': [c_a, c_b],
            '전환율 (%)': [results['p_a']*100, results['p_b']*100],
            'Z-Score': [results['z_score'], results['z_score']],
            'P-Value': [results['p_value'], results['p_value']],
            'Uplift (%)': [0, results['uplift']],
            '통계적 유의성': ['', '✅ 유의함' if results['is_significant'] else '❌ 유의하지 않음']
        })
        
        csv_results = results_summary.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "분석 결과 CSV 다운로드",
            data=csv_results,
            file_name="ab_test_results.csv",
            mime="text/csv"
        )
