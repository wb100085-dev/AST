# Streamlit Cloud에서 앱이 안 열릴 때

## 0. "로딩만 계속되는" 경우

- **Main file path**를 반드시 **`run.py`**로 두세요. `run.py`는 먼저 "앱을 불러오는 중…"을 띄운 뒤 앱을 로드해, 첫 응답이 빨리 나가도록 되어 있습니다.
- **Run command**: `streamlit run run.py --server.headless true` (또는 기본값 `streamlit run run.py`)

## 1. 메인 파일을 `run.py`로 설정

Streamlit Cloud 대시보드에서:

1. **Manage app** (앱 관리) 클릭
2. **Settings** 또는 **General** 에서 **Main file path** 확인
3. **Main file path** 를 `run.py` 로 설정
4. **Run command** 가 `streamlit run run.py` 인지 확인 (보통 자동)

`run.py` 는 import 오류가 나도 화면에 오류 메시지를 띄우므로, 원인 파악이 쉽습니다.

## 2. 로그로 오류 확인

1. **Manage app** > **Logs**
2. 빨간색 에러 메시지 확인 (ModuleNotFoundError, ImportError 등)
3. 필요한 패키지는 `requirements.txt` 에 버전과 함께 넣어 두세요.

## 3. Secrets (API 키 등) — 필수

- **Manage app** > **Secrets** 에서 설정
- **GEMINI_API_KEY** 를 반드시 설정해야 AI 기능(가상인구 대화, 페르소나 생성 등)이 동작합니다.
- 형식 예:
  ```
  GEMINI_API_KEY = "여기에_Google_AI_Studio에서_발급한_키"
  ```
- 로컬의 `.streamlit/secrets.toml` 내용을 그대로 붙여 넣거나, 필요한 키만 추가

## 4. 여전히 안 되면

- **Main file path** 를 `app.py` 로 두고 **Run command** 를 `streamlit run app.py` 로 사용해도 됩니다.
- 이때 import 오류가 나면 Streamlit 기본 오류 화면만 보일 수 있으니, **Logs** 에서 메시지를 확인하세요.
