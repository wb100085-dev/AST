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

    INTERVIEW_METHODS = [
        "FGI (Focus Group Interview)",
        "FGD (Focus Group Discussion)",
        "IDI (Individual Depth Interview)",
    ]

    def recommend_interview_method(self, definition: str, needs: str) -> tuple[str, str]:
        """
        제품/서비스 정의(definition)와 조사 목적·니즈(needs)를 바탕으로
        FGI / FGD / IDI 중 가장 적합한 인터뷰 방식을 추천하고 이유를 반환.
        반환: (방식_전체문자열, 이유_한줄)
        """
        definition = (definition or "").strip()
        needs = (needs or "").strip()
        if not definition or not needs:
            return self.INTERVIEW_METHODS[0], "입력 정보가 부족해 기본값(FGI)을 추천합니다."

        prompt = f"""다음은 조사 의뢰자가 적은 '제품/서비스 정의'와 '조사의 목적과 니즈'입니다.
이 내용에 가장 적합한 심층면접 방식을 아래 세 가지 중 **정확히 하나**만 골라 주세요.

[방식 종류]
- FGI (Focus Group Interview): 진행자 주도로 소수(6~12명) 대상 그룹 인터뷰. 가이드라인에 따른 질문·청취, 공통 인식/패턴 파악에 유리.
- FGD (Focus Group Discussion): FGI와 유사하나 진행자 개입 최소화, 참여자 간 자유 토론·상호작용으로 데이터 도출. 집단 역학, 의견 충돌/보완 관찰에 유리.
- IDI (Individual Depth Interview): 연구자와 대상자 1:1 대면. 민감한 주제·개인 경험, 무의식적 반응·복잡한 동기 심층 탐구에 적합.

[입력]
제품/서비스 정의:
{definition[:2000]}

조사의 목적과 니즈:
{needs[:2000]}

[출력 형식] 반드시 아래 두 줄만 출력하세요. 다른 설명 금지.
첫 줄: FGI (Focus Group Interview) 또는 FGD (Focus Group Discussion) 또는 IDI (Individual Depth Interview) (위 세 문장 중 정확히 하나)
둘째 줄: 추천 이유를 한 문장(100자 이내)으로만 작성
"""

        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        text = (resp.text or "").strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        chosen = self.INTERVIEW_METHODS[0]
        reason = "입력 내용을 바탕으로 해당 방식을 추천합니다."
        for ln in lines:
            for m in self.INTERVIEW_METHODS:
                if m in ln or m.split(" ")[0] in ln:
                    chosen = m
                    break
        if len(lines) >= 2:
            reason = lines[1][:200]
        elif len(lines) == 1 and "추천" not in lines[0] and chosen != lines[0]:
            reason = lines[0][:200]
        return chosen, reason

    def generate_interview_questions(
        self,
        interview_type: str,
        definition: str,
        needs: str,
    ) -> str:
        """
        선택한 인터뷰 방식(FGI/FGD/IDI)에 따른 질문 방식(Questioning Style)을 반영하여
        제품 정의·시장성 조사 목적·니즈를 검토한 맞춤형 모더레이터 질문 가이드를 생성.
        """
        definition = (definition or "").strip()
        needs = (needs or "").strip()
        interview_type = (interview_type or "").strip() or "FGI (Focus Group Interview)"

        method_style = ""
        if "FGI" in interview_type:
            method_style = """
[FGI 전용 질문 방식 – 정보 수집을 위한 구조적 질의 (Structured Inquiry)]
- 사전 설계된 가이드라인을 엄격히 준수하고, 특정 주제에 대한 명확한 응답을 위해 **구체적·직접적 질문**을 사용하세요.
- 모더레이터는 '답변 수집가' 역할로, 논점 일탈을 막고 효율적으로 정보를 추출하는 데 집중합니다.
- **직접적 질문(Direct Questioning)**: "이 기능의 장점은 무엇인가?"처럼 답변 범위를 명확히 한정하여 혼란을 최소화합니다.
- **순차적 진행(Sequential Flow)**: 참여자 전원의 의견을 고르게 청취하기 위해 개별 지명 질문을 활용하고 발언권을 통제합니다.
- **확인 및 요약**: 모호한 답변에는 "방금 말씀하신 내용은 A라는 의미인가?"처럼 재확인하여 데이터 정확성을 확보합니다.
"""
        elif "FGD" in interview_type:
            method_style = """
[FGD 전용 질문 방식 – 상호작용 촉발을 위한 자극제 제시 (Stimulation for Interaction)]
- 구체적 답변 요구보다 **참여자 간 대화·논쟁을 유발하는 포괄적 화두나 시나리오**를 제시하세요.
- 모더레이터는 '촉진자'로서 질문자보다 토론의 심판·관찰자 위치에서 **개입을 최소화**합니다.
- **연결형 질문(Linking)**: "A님의 의견에 대해 B님은 어떻게 생각하나요?"처럼 참여자 간 관계를 잇는 질문을 주로 사용합니다.
- **투사법 활용(Projective Technique)**: "만약 이 브랜드가 사람이라면 어떤 성격일까요?"처럼 우회적 질문으로 잠재 인식을 표출시킵니다.
- **갈등 조장 및 심화**: 의견이 대립할 때 봉합하지 말고 "왜 의견 차이가 발생하는지"를 질문하여 집단 역학을 관찰합니다.
"""
        else:
            method_style = """
[IDI 전용 질문 방식 – 본질 규명을 위한 심층 탐구 (Deep Probing)]
- 대상자 답변에 따라 **유연하게 후속 질문을 이어가는 래더링(Laddering)**으로 표면적 행동 이면의 근본 동기를 파헤칩니다.
- 모더레이터는 '상담가'처럼 라포(Rapport)를 형성하고, 심리적 방어 기제를 해제시키는 질문을 구사합니다.
- **속성-혜택-가치 연결(Laddering)**: "왜 그 점이 중요한가?"를 반복 질문하여 제품 속성을 개인의 핵심 가치관과 연결합니다.
- **유연한 추적(Flexible Tracing)**: 준비된 질문 순서에 얽매이지 않고, 대상자의 의식 흐름(Flow)을 따라가며 돌발 인사이트를 포착합니다.
- **침묵의 활용**: 답변을 재촉하지 않고 침묵을 견디거나 비언어적 신호를 관찰하여 무의식적 반응을 이끌어냅니다.
"""

        prompt = f"""당신은 질적 연구 및 심층면접 설계 전문가입니다. **질문 방식(Questioning Style)**은 데이터의 성격을 결정하는 핵심 통제 변수이므로, 방법론별 구조적 특성에 따라 질문의 개방성·개입 빈도·탐구 깊이를 차별화해야 합니다.

아래 **제품 정의**와 **시장성 조사 목적·니즈**를 검토한 뒤, 선택한 인터뷰 방식에 특화된 **모더레이터 질문 가이드**를 작성해 주세요.

[입력 정보]
- 인터뷰 방식: {interview_type}
- 제품/서비스 정의: {definition[:1500]}
- 조사의 목적과 니즈(시장성 조사): {needs[:1500]}
{method_style}

[공통 요구사항]
1. 질문 개수는 **20개 내외**로 작성하세요.
2. 논리적 흐름으로 **섹션**을 구분하세요: 도입(오프닝) → 본론(상세 탐구, 조사 목적에 맞게 소제목 사용) → 마무리.
3. 위에서 제시한 해당 방식의 질문 스타일을 반드시 반영하여, **그 방식만의 톤과 기법**이 드러나도록 질문을 구체적으로 작성하세요.
4. 출력은 **Markdown 형식**만 사용하세요. 각 섹션은 ## 또는 ### 제목으로, 질문은 번호 목록(1. 2. ...) 또는 - 로 나열하세요.
5. 다른 설명 문단 없이, 질문 가이드 본문만 출력하세요.

[출력 예시 형식]
## 도입 (오프닝)
1. (방식에 맞는 오프닝 질문)
2. ...

## 본론 - (소제목)
...

## 마무리
...
"""

        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        return (resp.text or "").strip()
