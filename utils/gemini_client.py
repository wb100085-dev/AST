# utils/gemini_client.py
import os
import json
from typing import Any, Dict, Optional

# 키 해석: gemini_key 모듈이 load_dotenv() 후 os.getenv()로 로드함
try:
    from utils.gemini_key import GEMINI_API_KEY as _ENV_KEY
except Exception:
    _ENV_KEY = None

# Streamlit Secrets 지원 (배포 환경용)
def _get_streamlit_secret_key():
    """Streamlit Secrets에서 API 키 가져오기"""
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            if 'GEMINI_API_KEY' in st.secrets:
                return st.secrets['GEMINI_API_KEY']
            if hasattr(st.secrets, 'get') and 'secrets' in st.secrets:
                if 'GEMINI_API_KEY' in st.secrets['secrets']:
                    return st.secrets['secrets']['GEMINI_API_KEY']
    except Exception:
        pass
    return None


class GeminiClient:
    """
    - google-genai SDK 기반 래퍼
    - 통계마다 다른 KOSIS 구조를 Gemini가 해석하도록 '룰 추출'을 담당
    """

    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        # api_key 우선순위: 인자 > Streamlit Secrets > .env/환경변수 (gemini_key에서 load_dotenv 후 os.getenv)
        key_source = None
        key = None

        if api_key:
            key = api_key
            key_source = "인자로 전달됨"
        elif _get_streamlit_secret_key():
            key = _get_streamlit_secret_key()
            key_source = "Streamlit Secrets"
        elif os.getenv("GEMINI_API_KEY"):
            key = os.getenv("GEMINI_API_KEY")
            key_source = "환경변수 GEMINI_API_KEY"
        elif os.getenv("GOOGLE_API_KEY"):
            key = os.getenv("GOOGLE_API_KEY")
            key_source = "환경변수 GOOGLE_API_KEY"
        elif _ENV_KEY:
            key = _ENV_KEY
            key_source = ".env 또는 환경변수"

        if not key or "여기에_" in str(key):
            raise RuntimeError(
                "Gemini API Key 미설정 상태임. "
                "다음 중 하나를 설정해주세요:\n"
                "1. .env 파일: GEMINI_API_KEY=your-key (로컬, python-dotenv로 로드)\n"
                "2. 환경변수: GEMINI_API_KEY 또는 GOOGLE_API_KEY\n"
                "3. Streamlit Secrets: st.secrets['GEMINI_API_KEY'] (배포 환경)"
            )
        
        # API 키 출처 저장 (디버깅용)
        self._key_source = key_source

        # SDK import는 여기서 수행(환경 문제일 때 스택이 명확해짐)
        # 공식 문서: from google import genai [Source](https://github.com/googleapis/python-genai)
        from google import genai  # noqa: F401

        self._genai = genai
        self._client = genai.Client(api_key=key)  # [Source](https://github.com/googleapis/python-genai)
        self._model = model

    def extract_kosis_assignment_rule(
        self,
        *,
        category_code: str,
        stat_name: str,
        kosis_url: str,
        kosis_data_sample: Any,
        axis_keys_kr: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        목적:
        - KOSIS 원천 구조가 통계마다 다르므로, Gemini가 '개인(6축) -> 어떤 차원 매칭 -> 어떤 값/범주 샘플링' 규칙을 JSON으로 반환하도록 강제

        반환 JSON 예시(형태만 예시, 통계별로 달라질 수 있음):
        {
          "mode": "categorical_sampling" | "numeric_lookup" | "score_sampling",
          "dimensions": {
             "sigungu": {"source": "...", "mapping": {...}},
             "gender": {"source": "...", "mapping": {...}},
             ...
          },
          "value_field_candidates": ["DT", "DATA", ...],
          "label_fields": ["ITM_NM","C1_NM",...],
          "notes": "..."
        }
        """

        prompt = f"""
너는 KOSIS 통계 JSON을 분석해 가상개인 데이터프레임에 값을 대입하는 규칙을 설계하는 분석가임.
목표는 '통계마다 구조가 다름'을 전제로 자동 규칙을 만든 뒤 코드가 그 규칙대로 개인별 값을 넣는 것임.

[입력 정보]
- category_code: {category_code}
- stat_name: {stat_name}
- kosis_url: {kosis_url}
- 6축(개인 속성) 한글 키: {json.dumps(axis_keys_kr, ensure_ascii=False)}
- kosis_data_sample: 아래 JSON 샘플 참고

[요구]
1) 이 통계가 개인별로 어떤 값 생성이 적절한지 mode를 결정
   - categorical_sampling: 범주형 응답(예: 만족/불만족, 의사 등) 확률분포 샘플링
   - numeric_lookup: 조건에 맞는 수치(비율/명/금액 등) 조회 후 대입
   - score_sampling: 점수형으로 변환 후 샘플링
2) KOSIS JSON에서 차원/라벨 필드 후보를 식별
3) 6축(거주지역, 성별, 연령, 경제활동, 교육, 월평균소득)을 KOSIS 차원에 매칭하는 방법을 제시
4) 반드시 JSON만 출력 (설명 문장 출력 금지)

[kosis_data_sample]
{json.dumps(kosis_data_sample, ensure_ascii=False)[:12000]}
"""

        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )

        text = (resp.text or "").strip()
        # 모델이 JSON 외 텍스트를 섞는 경우를 대비한 최소한의 방어
        try:
            return json.loads(text)
        except Exception:
            # JSON 덩어리만 추출 시도
            l = text.find("{")
            r = text.rfind("}")
            if l >= 0 and r > l:
                return json.loads(text[l : r + 1])
            raise RuntimeError(f"Gemini rule JSON 파싱 실패. raw={text[:500]}")
