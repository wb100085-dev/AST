# 프로젝트 리팩토링 가이드

현재 프로젝트 구조 분석 결과를 바탕으로, **코드 중복**, **성능 저하 요인**, **비대한 파일 모듈화**, **불필요한 의존성 제거**에 대한 가이드를 정리했습니다.

---

## 1. 프로젝트 구조 요약

| 경로 | 역할 | 비고 |
|------|------|------|
| `app.py` | 메인 앱, DB, 상수, 페이지, 생성/검증/차트 로직 | **약 4,055줄** – 과도하게 비대 |
| `run.py` | 배포용 진입점 | 적절 |
| `pages/*.py` | 가상인구 DB, 설문/심층면접, 결과 분석(Bass/PSM/Conjoint/StatCheck), 유틸(배경·옷) | `virtual_population_db.py` 1,600줄 이상, result_analysis_* 중복 다수 |
| `utils/*.py` | KOSIS, IPF, Gemini, step2_records, gemini_key | 적절한 분리 |

---

## 2. 코드 중복이 심한 구간

### 2.1 result_analysis_* 페이지 – “2차 대입 결과 선택” UI

**위치**: `pages/result_analysis_bass.py`, `result_analysis_psm.py`, `result_analysis_conjoint.py`, `result_analysis_statcheck.py`

**중복 내용**  
- `list_step2_records()` 호출 후 `record_options` 딕셔너리 생성  
- `"2차 대입 결과 데이터 사용"` 체크박스 + `selectbox`로 기록 선택  
- `load_step2_record(excel_path)` 호출 후 세션에 저장  
- 레이블 포맷: `f"{ts} | {sido_name} | {rows}명 | {cols}개 컬럼"`  

**제안**  
- `pages/common.py` 또는 새 `pages/step2_selector.py`에 **공통 위젯 함수** 추가  
  - 예: `render_step2_record_selector(session_key_prefix: str) -> Optional[pd.DataFrame]`  
- 각 result_analysis 페이지는 이 함수를 호출해 DataFrame만 받아 사용  

---

### 2.2 scipy 누락 시 동일한 에러 메시지

**위치**: `result_analysis_bass.py`, `result_analysis_psm.py` 등

**중복 내용**  
- `HAS_SCIPY = False`일 때 거의 동일한 `st.error("⚠️ 필수 패키지 누락... pip install scipy ...")` 블록  

**제안**  
- `pages/common.py`에 `ensure_scipy()` 또는 `check_scipy_and_show_error() -> bool` 같은 함수로 한 번만 정의 후, 각 페이지에서 호출  

---

### 2.3 virtual_population_db – app 모듈에서 상수/함수 가져오기

**위치**: `pages/virtual_population_db.py` (상단)

**중복/결합**  
- 런타임에 `sys.modules['app']`에서 `SIDO_MASTER`, `SIDO_LABEL_TO_CODE`, `SIDO_CODE_TO_NAME`, `db_conn` 참조  
- 시도·코드·이름 변환 로직이 app과 페이지 양쪽에 흩어져 있음  

**제안**  
- 시도 관련 상수/유틸을 **한 곳**로 모음 (아래 “상수/DB 분리” 참고)  
- `virtual_population_db.py`는 해당 모듈만 import (예: `from core.constants import SIDO_MASTER, ...`)

---

### 2.4 ensure_session_state – 반복적인 if 블록

**위치**: `app.py` 769~794줄

**중복 형태**  
- `if "키" not in st.session_state: st.session_state.키 = 기본값` 패턴이 10개 이상 반복  

**제안**  
- 딕셔너리로 기본값을 정의한 뒤 한 번에 초기화  
  - 예: `DEFAULT_SESSION = {"app_started": False, "nav_selected": "가상인구 DB", ...}`  
  - `for k, v in DEFAULT_SESSION.items(): st.session_state.setdefault(k, v)`  

---

## 3. 성능 저하를 유발하는 비효율적 로직

### 3.1 app.py 최상단 무거운 import

**위치**: `app.py` 1~17줄

**문제**  
- `pandas`, `numpy`, `streamlit`을 무조건 로드  
- `from app import main` 시점에 4,000줄 이상 파싱·실행  

**현재 완화**  
- `KosisClient`, `generate_base_population`, `GeminiClient`, `step2_records`는 `TYPE_CHECKING`/지연 로딩으로 처리됨  

**추가 제안**  
- `main()` 진입 직후 필요한 페이지/기능별로만 import (이미 일부 `_ensure_generate_modules()` 등으로 시도 중)  
- DB·상수·세션 초기화만 하는 “가벼운 진입점”을 유지하고, 생성/검증/차트 등은 해당 탭 선택 시 로드  

---

### 3.2 page_generate 내 반복되는 DB/파일 접근

**위치**: `app.py` 2054~3336줄 부근 (page_generate)

**문제**  
- 한 함수 안에서 `db_list_stats_by_sido(sido_code)` 등이 여러 번 호출될 수 있음  
- 오토세이브 경로 존재 여부·파일 읽기 등이 분산 호출  

**제안**  
- 생성 옵션 블록에서 한 번만 DB 조회 후 세션에 캐시 (예: `st.session_state.gen_active_stats`)  
- 오토세이브 로드/저장을 작은 함수로 분리 (`load_autosave()`, `save_autosave()`)하고, 호출 횟수 최소화  

---

### 3.3 list_step2_records 반복 호출

**위치**: `page_step2_results`, `virtual_population_db`, result_analysis_* 전반

**문제**  
- 같은 런타임에 여러 페이지/위젯에서 `list_step2_records()`를 각각 호출  
- 이미 `utils/step2_records.py`에 TTL 캐시가 있다면 유지하고, 없으면 30초~1분 TTL 캐시 도입 권장  

**확인**  
- `list_step2_records`가 데코레이터/캐시로 감싸져 있는지 확인 후, 없으면 `functools.lru_cache` 또는 간단한 메모리 캐시 적용  

---

### 3.4 대량 DataFrame 연산 시 st.rerun() 과다

**위치**: `page_generate` 내 생성 버튼 처리, 2단계 대입 등

**문제**  
- 긴 루프 안에서 매번 `st.rerun()`을 호출하면 전체 스크립트가 반복 실행되어 초기화 비용이 큼  

**제안**  
- “10명 단위로 처리 후 rerun” 같은 방식은 유지하되, **진행률만 갱신**하고 rerun 횟수는 줄이는 방향 검토  
- 가능하면 `st.progress` + `st.empty()`로 같은 run 내에서 업데이트  

---

## 4. 비대한 파일 모듈화 제안

### 4.1 app.py 분리 (우선순위 높음)

| 추출 대상 | app.py 라인 대역 (대략) | 제안 모듈 | 비고 |
|-----------|--------------------------|------------|------|
| 상수 (SIDO_MASTER, STEP2_COLUMN_RENAME 등) | 40~76, 342~368 | `core/constants.py` 또는 `config/sido.py` | 다른 모듈에서 단일 import |
| DB 레이어 (db_conn, db_init, db_* 전부) | 374~760 | `core/db.py` 또는 `db/sqlite_layer.py` | app은 `from core.db import db_conn, db_init, ...` |
| 세션 초기화 (ensure_session_state, load_sigungu_options) | 762~816 | `core/session.py` 또는 app 내 `session_utils` | 상수·DB에 의존하므로 분리 시 import 경로만 정리 |
| KOSIS 변환·차트·검증 (convert_kosis_to_distribution, draw_*, apply_step2_*, calculate_mape, get_step2_*, build_step2_error_report, render_validation_tab) | 918~2052 | `core/generate_helpers.py` 또는 `features/generation/validation.py` 등 | page_generate 전용으로 묶어서 분리 |
| page_generate | 2054~3335 | `pages/page_generate.py` (또는 `features/generation/page.py`) | 상수·DB·세션·generate_helpers만 import |
| page_step2_results, page_stat_assignment_log | 3336~3525 | `pages/step2_and_log.py` 또는 app에 유지 | 코드량이 크지 않으면 app에 두고, 나중에 분리 |
| page_guide, page_survey_form_builder, page_survey_results | 3508~3840 | `pages/guide_and_survey.py` 또는 기능별 파일 | 설문 관련이 크면 survey 전용 모듈로 |
| render_landing, main, _ensure_generate_modules | 3841~끝 | app.py에 유지 | main은 짧게 유지하고 위에서 분리한 페이지/모듈만 호출 |

**진입점 유지**  
- `run.py` → `app.main()` → `app.py`는 “라우팅 + set_page_config + render_landing/main UI”만 담당  
- 실제 로직은 `from core.db import ...`, `from pages.page_generate import page_generate` 등으로 가져와서 호출  

---

### 4.2 pages/virtual_population_db.py

**현재**: 1,600줄 이상, 페르소나 생성·전체 학습·대화 등이 한 파일에 있음  

**제안**  
- “2차 대입 결과 목록·선택” 블록 → `pages/virtual_population_db_list.py` 또는 공통 `step2_selector` 재사용  
- “페르소나 생성 / 전체 학습” 블록 → `pages/virtual_population_db_persona.py`  
- “1:1 대화 / 5:1 대화” 블록 → `pages/virtual_population_db_chat.py`  
- `page_virtual_population_db()`는 위 모듈들을 import해 탭/버튼에 따라 호출만 하도록 축소  

---

### 4.3 가독성 보조 규칙

- 한 파일은 **400~600줄** 이내를 목표로 (넘어가면 위처럼 기능 단위로 잘라 내기)  
- 한 함수는 **80줄** 이내 권장 (길어지면 “옵션 구성 / 실행 / 결과 표시” 등으로 나누기)  
- `page_generate`는 “옵션 UI / 생성 실행 / 결과·다운로드 UI”로 3개 함수로 쪼개도 됨  

---

## 5. 불필요한 의존성 제거

### 5.1 requirements.txt에서 제거 검토

| 패키지 | 사용처 검색 결과 | 제안 |
|--------|------------------|------|
| `google-cloud-storage` | 프로젝트 내 .py에서 미사용 | **제거** (클라우드 동기화 제거됨) |
| `boto3` | 프로젝트 내 .py에서 미사용 | **제거** (S3 등 미사용) |
| `google-api-python-client` | 사용처 확인 필요 | KOSIS/Drive 등에서 쓰이면 유지, 아니면 제거 |

**방법**  
- `requirements.txt`에서 해당 줄 삭제 후, 로컬·배포 환경에서 `pip install -r requirements.txt` 및 앱 실행·주요 기능 테스트  

---

### 5.2 import 정리

- **app.py**: `core.db`, `core.constants` 분리 후 해당 모듈에서만 `sqlite3`, `os` 등 사용하고, app.py는 `streamlit` + 라우팅 위주로  
- **pages 쪽**: `from app import SIDO_MASTER, db_conn` 대신 `from core.constants import SIDO_MASTER`, `from core.db import db_conn` 로 변경해 app에 대한 직접 의존 제거  

---

## 6. 리팩토링 순서 제안 (실행 가이드)

1. **의존성 정리**  
   - `google-cloud-storage`, `boto3` 제거 후 테스트  

2. **상수 분리**  
   - `core/constants.py` 생성, SIDO_MASTER·STEP2_COLUMN_RENAME·APP_TITLE 등 이동  
   - app.py 및 `virtual_population_db.py`에서 `from core.constants import ...` 로 변경  

3. **DB 레이어 분리**  
   - `core/db.py` 생성, `db_conn`, `db_init`, `db_*` 전부 이동  
   - app.py와 virtual_population_db는 `from core.db import db_conn, db_init, ...` 만 사용  

4. **공통 UI (2차 대입 선택)**  
   - `pages/common.py` 또는 `step2_selector.py`에 `render_step2_record_selector(session_key_prefix)` 추가  
   - result_analysis_bass/psm/conjoint/statcheck에서 해당 블록 제거 후 공통 함수 호출로 교체  

5. **ensure_session_state 리팩터**  
   - 기본값 딕셔너리 + `setdefault` 루프로 변경  

6. **page_generate 분리**  
   - `core/generate_helpers.py` (또는 `features/generation/helpers.py`) 에 KOSIS 변환·차트·검증 함수 이동  
   - `pages/page_generate.py` 생성 후 `page_generate` 함수 이동, app.py에서는 import만 해서 호출  

7. **virtual_population_db 분할**  
   - 목록 / 페르소나 / 대화 블록을 각각 파일로 나누고, `page_virtual_population_db()` 에서만 조합  

8. **캐시 확인**  
   - `list_step2_records`에 TTL 캐시가 없으면 추가해, 동일 run 내 반복 호출 감소  

---

## 7. 요약

| 구분 | 내용 |
|------|------|
| **중복** | result_analysis_* 의 “2차 대입 결과 선택” UI, scipy 에러 메시지, virtual_population_db의 app 의존, ensure_session_state 반복 if |
| **성능** | app 상단 무거운 import, page_generate 내 반복 DB/파일 접근, list_step2_records 중복 호출, st.rerun() 과다 사용 |
| **비대 파일** | app.py(~4,055줄) → 상수/DB/헬퍼/페이지 분리, virtual_population_db(~1,600줄) → 목록·페르소나·대화 분리 |
| **불필요 의존성** | google-cloud-storage, boto3 제거 검토, app 직접 참조를 core 모듈 참조로 교체 |

이 가이드를 단계별로 적용하면 가독성·유지보수성·성능이 개선되고, 이후 기능 추가 시에도 모듈 단위로 작업하기 쉬워집니다.
