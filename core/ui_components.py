"""
분석 페이지(result_analysis_*) 공통 UI: 시장성 조사 설문 응답 결과 데이터 선택
(기존 2차 대입 결과 대신 설문 응답에 참여한 가상인구 데이터 사용)
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import pandas as pd
import streamlit as st


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


def _survey_latest_panel_path() -> Optional[str]:
    """시장성 조사 설문 응답 결과(가상인구 패널) 저장 경로. 없으면 None."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "data", "survey_latest", "panel.csv")
    return path if os.path.isfile(path) else None


def render_survey_response_data_selector(
    session_key_prefix: str,
    info_when_no_records: str = "시장성 조사 설계에서 **AI 설문 진행**을 실행한 뒤, 여기서 설문 응답에 참여한 가상인구 데이터를 불러올 수 있습니다.",
    info_when_unchecked: str = "가상 데이터를 사용합니다. 설문 응답 결과 데이터를 사용하려면 위 체크박스를 선택하세요.",
    show_preview: bool = True,
    preview_caption: Optional[str] = None,
    show_preview_row_count: bool = False,
) -> Tuple[bool, Optional[pd.DataFrame]]:
    """
    시장성 조사 설문 응답 결과 데이터(응답 참여 가상인구) 선택 UI.
    반환: (use_real_data, real_data_df 또는 None)
    """
    st.markdown("### AI설문 응답 결과 불러오기")
    panel_path = _survey_latest_panel_path()
    use_real_data = False
    real_data_df: Optional[pd.DataFrame] = None

    if not panel_path:
        st.info(info_when_no_records)
        return False, None

    use_real_data_option = st.checkbox(
        "시장성 조사 설문 응답 결과 데이터 사용",
        value=False,
        key=f"{session_key_prefix}_use_survey_data",
    )

    if use_real_data_option:
        try:
            real_data_df = pd.read_csv(panel_path, encoding="utf-8-sig")
            st.session_state[f"{session_key_prefix}_population_df"] = real_data_df
            use_real_data = True
            st.success(
                f"✅ 설문 응답에 참여한 가상인구 데이터를 불러왔습니다. (총 {len(real_data_df)}명, {len(real_data_df.columns)}개 컬럼)"
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


def render_step2_data_selector(
    session_key_prefix: str,
    info_when_no_records: str = "시장성 조사 설계에서 **AI 설문 진행**을 실행한 뒤, 여기서 설문 응답 데이터를 불러올 수 있습니다.",
    info_when_unchecked: str = "가상 데이터를 사용합니다. 설문 응답 결과 데이터를 사용하려면 위 체크박스를 선택하세요.",
    show_preview: bool = True,
    preview_caption: Optional[str] = None,
    show_preview_row_count: bool = False,
) -> Tuple[bool, Optional[pd.DataFrame]]:
    """
    분석 페이지용 데이터 선택. 시장성 조사 설문 응답 결과(가상인구 패널) 사용.
    반환: (use_real_data, real_data_df 또는 None)
    """
    return render_survey_response_data_selector(
        session_key_prefix=session_key_prefix,
        info_when_no_records=info_when_no_records,
        info_when_unchecked=info_when_unchecked,
        show_preview=show_preview,
        preview_caption=preview_caption,
        show_preview_row_count=show_preview_row_count,
    )
