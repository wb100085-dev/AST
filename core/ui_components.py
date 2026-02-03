"""
분석 페이지(result_analysis_*) 공통 UI: 2차 대입 데이터 선택 체크박스·에러 메시지
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from utils.step2_records import list_step2_records, load_step2_record


def show_package_required_error(
    has_package: bool,
    package_display_name: str,
    pip_install_name: str,
) -> bool:
    """
    필수 패키지 누락 시 에러 메시지를 출력하고 False 반환.
    패키지가 있으면 True 반환 (호출 페이지는 계속 진행).

    사용 예:
        if not show_package_required_error(HAS_SCIPY, "scipy", "scipy"):
            return
    """
    if has_package:
        return True
    st.error(f"""
⚠️ **필수 패키지 누락**

`{package_display_name}` 모듈이 설치되어 있지 않습니다. 다음 명령으로 설치해주세요:

```bash
pip install {pip_install_name}
```

또는 가상환경이 활성화된 상태에서:

```bash
.\\venv\\Scripts\\Activate.ps1
pip install {pip_install_name}
```

설치 후 페이지를 새로고침해주세요.
""")
    return False


def render_step2_data_selector(
    session_key_prefix: str,
    info_when_no_records: str = "아직 2차 대입 결과가 없습니다. 가상인구 생성 후 2단계에서 통계를 대입하면 여기서 불러올 수 있습니다.",
    info_when_unchecked: str = "가상 데이터를 사용합니다. 2차 대입 결과를 사용하려면 위 체크박스를 선택하세요.",
    show_preview: bool = True,
    preview_caption: Optional[str] = None,
    show_preview_row_count: bool = False,
) -> Tuple[bool, Optional[pd.DataFrame]]:
    """
    2차 대입 결과 '데이터 선택 체크박스' + selectbox + 로드 결과를 한 번에 렌더링.
    반환: (use_real_data, real_data_df 또는 None)

    session_key_prefix: 스트림릿 key 접두사 (예: "bass", "psm", "conjoint", "statcheck")
    """
    st.markdown("### 가상인구 데이터 불러오기")
    records = list_step2_records()

    use_real_data = False
    real_data_df: Optional[pd.DataFrame] = None

    if not records:
        st.info(info_when_no_records)
        return False, None

    record_options: dict = {}
    for r in records:
        ts = r.get("timestamp", "")
        sido_name = r.get("sido_name", "")
        rows = r.get("rows", 0)
        cols = r.get("columns_count", 0)
        label = f"{ts} | {sido_name} | {rows}명 | {cols}개 컬럼"
        record_options[label] = r

    use_real_data_option = st.checkbox(
        "2차 대입 결과 데이터 사용",
        value=False,
        key=f"{session_key_prefix}_use_real_data",
    )

    if use_real_data_option:
        selected_label_key = f"{session_key_prefix}_selected_record_label"
        keys_list = list(record_options.keys())
        default_index = 0
        if st.session_state.get(selected_label_key) and st.session_state[selected_label_key] in record_options:
            default_index = keys_list.index(st.session_state[selected_label_key])

        selected_label = st.selectbox(
            "2차 대입 결과에서 가상인구 데이터를 선택하세요:",
            options=keys_list,
            index=default_index,
            key=f"{session_key_prefix}_record_select",
        )

        if selected_label and selected_label in record_options:
            selected_record = record_options[selected_label]
            excel_path = selected_record.get("excel_path", "")

            if not excel_path or not os.path.isfile(excel_path):
                st.warning("선택한 데이터 파일을 찾을 수 없습니다.")
            else:
                try:
                    real_data_df = load_step2_record(excel_path)
                    st.session_state[f"{session_key_prefix}_population_df"] = real_data_df
                    st.session_state[selected_label_key] = selected_label
                    use_real_data = True
                    st.success(
                        f"✅ 가상인구 데이터를 불러왔습니다. (총 {len(real_data_df)}명, {len(real_data_df.columns)}개 컬럼)"
                    )
                    if show_preview:
                        with st.expander("불러온 데이터 미리보기"):
                            st.dataframe(real_data_df.head(20), use_container_width=True, height=300)
                            if preview_caption:
                                st.caption(preview_caption)
                            elif show_preview_row_count:
                                st.caption(f"전체 {len(real_data_df)}명 중 처음 20명 표시")
                except Exception as e:
                    st.error(f"데이터 로드 실패: {e}")
    else:
        st.session_state[f"{session_key_prefix}_population_df"] = None
        st.info(info_when_unchecked)

    return use_real_data, real_data_df
