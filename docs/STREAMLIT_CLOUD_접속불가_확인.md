# Streamlit Cloud 접속 불가 시 확인 사항

## 1. 먼저 확인할 것

1. **Manage app > Logs**  
   - 빨간색 에러(ImportError, ModuleNotFoundError, RuntimeError 등)가 있는지 확인하세요.  
   - 타임아웃·연결 끊김 메시지가 있는지 확인하세요.

2. **Main file path**  
   - **`run.py`** 로 설정되어 있는지 확인하세요.  
   - Run command: `streamlit run run.py` (또는 `streamlit run run.py --server.headless true`).

3. **Secrets**  
   - GEMINI_API_KEY 등 필요한 API 키가 설정되어 있으면, 키 오류로 인한 크래시 가능성이 줄어듭니다.  
   - 키가 없어도 앱 자체는 뜨고, 해당 기능만 실패하는 경우가 많습니다.

## 2. 접속 불가/타임아웃이 날 수 있는 원인

| 원인 | 확인 방법 | 대응 |
|------|-----------|------|
| **Cold start 지연** | 로그에서 앱 시작까지 시간 확인 | 재배포 후 1~2분 기다린 뒤 다시 접속 |
| **두 번째 요청 타임아웃** | 이전 run.py가 첫 요청에서 rerun 후, 두 번째 요청에서 app 로드 시 타임아웃 | run.py를 **한 번의 요청에서 앱 로드**하도록 수정 (rerun 제거) |
| **의존성 설치 실패** | Logs에 `pip install` 또는 패키지 이름 관련 에러 | requirements.txt 패키지·버전 점검, 필요 시 버전 완화 |
| **import/실행 시 예외** | Logs에 `ModuleNotFoundError`, `ImportError`, `FileNotFoundError` 등 | 로그에 나온 파일/모듈 기준으로 코드·경로 수정 |
| **Streamlit/Cloud 장애** | [Streamlit Status](https://status.streamlit.io/) 또는 포럼 공지 | Cloud 복구 후 재시도 |

## 3. 로그에서 자주 나오는 오류 예시

- `ModuleNotFoundError: No module named 'core'`  
  → 프로젝트 루트에서 실행되는지 확인 (Main file path가 `run.py`이고, 리포 루트가 맞는지).
- `sqlite3.OperationalError: unable to open database file`  
  → 앱이 이미 쓰기 가능 경로(예: /tmp)를 사용하도록 수정되어 있는지 확인.
- `Connection timeout` / `AxiosError: timeout exceeded`  
  → Cold start 또는 첫 응답 지연. run.py에서 rerun 제거 후 한 번에 로드하도록 변경해 보세요.

## 4. 로컬에서 배포와 비슷하게 테스트

```bash
streamlit run run.py --server.headless true
```

위처럼 실행한 뒤 브라우저에서 접속해, 로그에 에러가 없는지 확인하세요.
