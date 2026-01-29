# 프로젝트 전체 검토 보고서

**프로젝트명**: AI Social Twin - 가상인구 생성기  
**검토일**: 2025-01-27  
**검토 범위**: 전체 코드베이스

---

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [아키텍처 분석](#아키텍처-분석)
3. [발견된 이슈](#발견된-이슈)
4. [개선 제안](#개선-제안)
5. [코드 품질 평가](#코드-품질-평가)

---

## 프로젝트 개요

### 주요 기능
- **가상인구 생성**: KOSIS 통계 기반 IPF 알고리즘을 사용한 현실적인 가상인구 생성
- **데이터 관리**: 시도별 통계 목록 및 템플릿 관리
- **통계 연동**: KOSIS API를 통한 실시간 통계 데이터 수집
- **Excel 템플릿 지원**: 고정 형식 Excel 템플릿에 데이터 매핑 및 채우기

### 기술 스택
- **프레임워크**: Streamlit
- **데이터베이스**: SQLite
- **데이터 처리**: Pandas, NumPy
- **통계 API**: KOSIS (한국통계청)
- **AI**: Google Gemini API (통계 구조 분석용)
- **Excel 처리**: openpyxl

---

## 아키텍처 분석

### 디렉토리 구조
```
virtual_population_generator/
├── app.py                    # 메인 Streamlit 애플리케이션 (1,648줄)
├── requirements.txt          # 의존성 목록 (⚠️ 비어있음)
├── data/                     # 데이터 저장소
│   ├── app.db               # SQLite 데이터베이스
│   └── 37_synthetic_population.xlsx
├── streamlit/
│   └── config.toml          # Streamlit 설정
└── utils/                   # 유틸리티 모듈
    ├── __init__.py
    ├── ipf_generator.py      # IPF 알고리즘 및 가상인구 생성 (582줄)
    ├── kosis_client.py      # KOSIS API 클라이언트
    ├── gemini_client.py     # Gemini API 클라이언트
    ├── template_exporter.py # Excel 템플릿 내보내기
    └── validator.py         # 데이터 검증
```

### 주요 모듈

#### 1. `app.py` - 메인 애플리케이션
- **역할**: Streamlit UI 및 비즈니스 로직 통합
- **주요 기능**:
  - 데이터 관리 페이지 (`page_data_management`)
  - 가상인구 생성 페이지 (`page_generate`)
  - 작업 기록 페이지 (`page_work_log`)
  - DB 초기화 및 관리
  - KOSIS 데이터를 확률 분포로 변환 (`convert_kosis_to_distribution`)

#### 2. `utils/ipf_generator.py` - IPF 알고리즘
- **역할**: Iterative Proportional Fitting 알고리즘 구현
- **주요 기능**:
  - 6축(거주지역, 성별, 연령, 경제활동, 교육정도, 소득) 기반 인구 생성
  - 현실적 제약 조건 적용
  - 중복 제거 및 다양성 확보

#### 3. `utils/kosis_client.py` - KOSIS 클라이언트
- **역할**: KOSIS API 통신 및 데이터 처리
- **주요 기능**:
  - JSON 데이터 페칭 (캐싱 지원)
  - 시군구 목록 추출
  - Gemini를 통한 통계 구조 분석 및 매핑 규칙 생성

#### 4. `utils/gemini_client.py` - Gemini 클라이언트
- **역할**: Google Gemini API를 통한 통계 구조 분석
- **주요 기능**:
  - KOSIS 데이터 구조 분석
  - 6축 매핑 규칙 자동 생성

---

## 발견된 이슈

### 🔴 Critical (즉시 수정 필요)

#### 1. 함수 시그니처 불일치
**위치**: `app.py:1062` vs `utils/ipf_generator.py:362`

**문제**:
```python
# app.py에서 호출
generate_base_population(
    n=int(n),
    selected_sigungu=None,  # ❌ None 전달
    weights_6axis={...},
    margins_axis=margins_axis if margins_axis else None,
    # sigungu_pool, seed 누락
)

# ipf_generator.py 정의
def generate_base_population(
    n: int,
    selected_sigungu: List[str],  # ❌ List[str] 기대
    weights_6axis: dict,
    sigungu_pool: List[str],       # ❌ 필수 파라미터 누락
    seed: int,                      # ❌ 필수 파라미터 누락
    margins_axis: Dict[str, Dict],
    apply_ipf_flag: bool = True
) -> pd.DataFrame:
```

**영향**: 런타임 에러 발생 가능성

**해결 방안**:
- `app.py`에서 올바른 파라미터 전달
- 또는 `ipf_generator.py`의 함수 시그니처를 호출부에 맞게 수정

#### 2. 데이터베이스 스키마 누락
**위치**: `app.py:292-332` (사용) vs `app.py:78-168` (정의)

**문제**:
- `sido_templates` 테이블이 `db_init()` 함수에 정의되지 않음
- 하지만 `db_upsert_template()`, `db_get_template()` 함수에서 사용됨

**영향**: 템플릿 저장/조회 기능이 작동하지 않음

**해결 방안**:
```sql
CREATE TABLE IF NOT EXISTS sido_templates (
    sido_code TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_blob BLOB NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
)
```

#### 3. requirements.txt 비어있음
**위치**: `requirements.txt`

**문제**: 프로젝트 의존성 목록이 없음

**영향**: 새 환경에서 설치 불가능

**해결 방안**: 필요한 패키지 목록 작성

---

### 🟡 Warning (개선 권장)

#### 4. 중복 Import
**위치**: `app.py:5, 9`

```python
from typing import List, Dict, Any, Optional  # 5번째 줄
from typing import Dict, List, Tuple, Optional, Any  # 9번째 줄
```

**해결 방안**: 중복 제거 및 통합

#### 5. 하드코딩된 값들
**위치**: 여러 곳

- `app.py:444`: 기본 시도 코드 `"경상북도 (37)"`
- `app.py:461-465`: KOSIS URL 하드코딩
- `utils/ipf_generator.py`: 기본 확률 분포 하드코딩

**해결 방안**: 설정 파일 또는 환경변수로 이동

#### 6. 에러 처리 부족
**위치**: `app.py:1062-1074`

**문제**: `generate_base_population` 호출 시 예외 처리 미흡

**해결 방안**: try-except 블록 추가 및 사용자 친화적 에러 메시지

#### 7. 타입 힌트 불일치
**위치**: `utils/ipf_generator.py:184`

**문제**: `apply_ipf` 함수에서 `margins_axis`의 `probs` 키를 사용하지만, 실제로는 `p` 키 사용

```python
# 209번째 줄
probs = margin_data.get('probs', [])  # ❌ 'probs' 사용

# 하지만 app.py:1044에서는
margins_axis[axis_key] = {
    "labels": labels,
    "p": probs  # ✅ 'p' 사용
}
```

**해결 방안**: 키 이름 통일 (`p` 또는 `probs`)

#### 8. 로깅 시스템 부재
**위치**: 전체 프로젝트

**문제**: `print()` 문을 사용한 로깅, 파일 로깅 없음

**해결 방안**: Python `logging` 모듈 도입

---

### 🟢 Info (선택적 개선)

#### 9. 문서화 부족
- README.md 파일 없음
- 함수 docstring 일부 누락
- API 문서 없음

#### 10. 테스트 코드 없음
- 단위 테스트 없음
- 통합 테스트 없음

#### 11. 코드 중복
- `app copy.py`, `utils/ipf_generator copy.py` 파일 존재
- 정리 필요

#### 12. 성능 최적화 여지
- `app.py:599-923`: `convert_kosis_to_distribution` 함수가 매우 길고 복잡
- 함수 분리 및 최적화 필요

---

## 개선 제안

### 우선순위 1: Critical 이슈 수정

1. **함수 시그니처 통일**
   ```python
   # app.py 수정
   base_df = generate_base_population(
       n=int(n),
       selected_sigungu=[],  # 빈 리스트로 변경
       weights_6axis={...},
       sigungu_pool=[],  # 추가
       seed=42,  # 추가
       margins_axis=margins_axis if margins_axis else {},
   )
   ```

2. **DB 스키마 추가**
   ```python
   # db_init() 함수에 추가
   cur.execute("""
       CREATE TABLE IF NOT EXISTS sido_templates (
           sido_code TEXT PRIMARY KEY,
           filename TEXT NOT NULL,
           file_blob BLOB NOT NULL,
           created_at TEXT DEFAULT (datetime('now')),
           updated_at TEXT DEFAULT (datetime('now'))
       )
   """)
   ```

3. **requirements.txt 작성**
   ```
   streamlit>=1.53.0
   pandas>=2.3.0
   numpy>=2.2.0
   openpyxl>=3.1.0
   requests>=2.32.0
   google-genai>=1.60.0
   plotly>=6.5.0
   rapidfuzz>=3.14.0
   ```

### 우선순위 2: 코드 품질 개선

1. **타입 힌트 통일**: `probs` vs `p` 키 통일
2. **에러 처리 강화**: 모든 외부 API 호출에 try-except 추가
3. **로깅 시스템 도입**: `logging` 모듈 사용
4. **설정 파일 분리**: 하드코딩된 값들을 `config.py`로 이동

### 우선순위 3: 문서화 및 테스트

1. **README.md 작성**: 프로젝트 소개, 설치 방법, 사용법
2. **단위 테스트 추가**: 주요 함수들에 대한 테스트
3. **API 문서화**: 주요 함수 docstring 보완

---

## 코드 품질 평가

### 강점 ✅

1. **명확한 모듈 분리**: 기능별로 잘 분리됨
2. **현실적 제약 조건**: `apply_realistic_constraints` 함수로 현실성 확보
3. **IPF 알고리즘 구현**: 통계 기반 인구 생성 로직이 잘 구현됨
4. **사용자 친화적 UI**: Streamlit을 활용한 직관적인 인터페이스
5. **캐싱 시스템**: KOSIS 클라이언트에 캐싱 구현

### 개선 필요 영역 ⚠️

1. **에러 처리**: 외부 API 호출 시 예외 처리 부족
2. **타입 안정성**: 타입 힌트 불일치 및 누락
3. **테스트 커버리지**: 테스트 코드 전무
4. **문서화**: README 및 API 문서 부족
5. **코드 중복**: 일부 로직 중복 및 불필요한 파일

### 전체 평가

**점수**: 7/10

- **기능성**: 8/10 - 핵심 기능은 잘 구현됨
- **안정성**: 6/10 - Critical 이슈 존재
- **유지보수성**: 7/10 - 모듈 분리는 좋으나 문서화 부족
- **성능**: 7/10 - 대체로 양호하나 최적화 여지 있음
- **코드 품질**: 7/10 - 전반적으로 양호하나 일부 개선 필요

---

## 결론

프로젝트는 **가상인구 생성**이라는 복잡한 문제를 잘 해결하고 있으며, IPF 알고리즘과 현실적 제약 조건을 통해 현실적인 데이터를 생성하는 구조가 잘 설계되어 있습니다.

다만, **Critical 이슈 3개**를 즉시 수정해야 하며, 코드 품질 개선과 문서화를 통해 프로덕션 준비도를 높일 수 있습니다.

**즉시 조치 필요**:
1. ✅ 함수 시그니처 불일치 수정
2. ✅ DB 스키마 추가
3. ✅ requirements.txt 작성

이 3가지를 수정하면 프로젝트가 정상적으로 동작할 것입니다.
