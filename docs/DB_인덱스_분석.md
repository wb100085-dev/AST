# core/db.py 쿼리 분석 및 인덱스 권장안

## 1. 쿼리 분석 요약

### 1.1 stats 테이블

| 함수 | 쿼리 패턴 | 데이터량/빈도 |
|------|-----------|----------------|
| `db_list_stats_by_sido` | `WHERE is_active=1 ORDER BY sido_code, category, name` | 전체 스캔 + 정렬 (sido_code='00') |
| `db_list_stats_by_sido` | `ORDER BY sido_code, category, name` | 전체 스캔 + 정렬 (sido_code='00') |
| `db_list_stats_by_sido` | `WHERE sido_code=? AND is_active=1 ORDER BY category, name` | 시도별 필터 + 정렬 |
| `db_list_stats_by_sido` | `WHERE sido_code=? ORDER BY category, name` | 시도별 필터 + 정렬 |
| `db_upsert_stat` | `ON CONFLICT(sido_code, category, name)` | UNIQUE 인덱스 사용 |
| `db_delete_stat_by_id` | `WHERE id=?` | PK 사용 |
| `db_update_stat_by_id` | `WHERE id=?` | PK 사용 |

- **기존 인덱스**: `ux_stats_sido_category_name (sido_code, category, name)` — UNIQUE
- **문제점**:
  - `sido_code='00'`일 때 `WHERE is_active=1` 또는 무조건 전체 행 스캔 후 `ORDER BY sido_code, category, name` → `is_active` 조건 시 인덱스 미활용, 정렬 비용 발생 가능
  - `sido_code=? AND is_active=1`일 때 `is_active` 필터를 위해 추가 행 스캔 필요

### 1.2 sido_axis_margin_stats 테이블

| 함수 | 쿼리 패턴 | 데이터량/빈도 |
|------|-----------|----------------|
| `db_get_axis_margin_stats` | `WHERE sido_code=? AND axis_key=?` | 1건 조회 |
| `db_get_six_axis_stat_ids` | 위 함수를 6회 호출 (axis_key 6개) | 6회 조회 |

- **기존**: 테이블 제약 `UNIQUE(sido_code, axis_key)` → SQLite는 UNIQUE 제약에 자동 인덱스 생성
- **결론**: 별도 인덱스 추가 불필요 (UNIQUE가 조회에 이미 활용됨)

### 1.3 sido_templates 테이블

| 함수 | 쿼리 패턴 |
|------|-----------|
| `db_get_template` | `WHERE sido_code=?` |
| `db_upsert_template` | `ON CONFLICT(sido_code)` |

- **기존**: `sido_code` PRIMARY KEY
- **결론**: 별도 인덱스 불필요

### 1.4 virtual_population_db 테이블 (db_init에서 생성, pages에서 조회)

- **core/db.py에는 조회 함수 없음.** `pages/virtual_population_db.py`에서 아래 패턴으로 자주 사용됨.
- `WHERE sido_code=? ORDER BY added_at` — 시도별 목록 조회 + 추가 시각 순 정렬
- `WHERE sido_code=? AND record_timestamp=? AND record_excel_path=?` — 중복 체크
- `WHERE id=?` — UPDATE/DELETE (PK)
- `WHERE sido_code=?` — DELETE 시도 전체

---

## 2. 권장 인덱스 리스트 (core/db.py 기준)

### stats 테이블

| 인덱스 이름 | 컬럼 | 필요 이유 |
|-------------|------|-----------|
| `idx_stats_is_active_order` | `(is_active, sido_code, category, name)` | **sido_code='00'** 일 때 `WHERE is_active=1 ORDER BY sido_code, category, name` / `ORDER BY sido_code, category, name` 에서 **인덱스 범위 스캔 + 정렬 생략**. `is_active` 로 먼저 걸러서 활성만 읽고, 나머지 컬럼 순서로 정렬된 결과를 바로 반환. |
| `idx_stats_sido_active_cat` | `(sido_code, is_active, category, name)` | **특정 시도** 조회 시 `WHERE sido_code=? [AND is_active=1] ORDER BY category, name` 에서 **시도·활성 여부 필터와 정렬을 한 인덱스로 처리**. 기존 UNIQUE `(sido_code, category, name)` 만으로는 `is_active=1` 필터 시 불필요 행 스캔이 늘어남. 이 복합 인덱스로 시도+활성 여부까지 인덱스에서 걸러 정렬 비용과 스캔량을 줄임. |

### sido_axis_margin_stats / sido_templates

- **추가 인덱스 없음.** UNIQUE(sido_code, axis_key), PK(sido_code) 로 이미 조회·충돌 처리에 충분.

### virtual_population_db (db_init에서 생성 시 추가 권장)

| 인덱스 이름 | 컬럼 | 필요 이유 |
|-------------|------|-----------|
| `idx_vpd_sido_added` | `(sido_code, added_at)` | **`WHERE sido_code=? ORDER BY added_at`** 가 pages에서 반복 사용됨. 시도별로 `added_at` 순 정렬 목록 조회 시 **시도 필터 + 정렬을 한 인덱스로 처리**하여 풀 스캔·파일정렬을 피함. |
| `idx_vpd_sido_ts_path` | `(sido_code, record_timestamp, record_excel_path)` | **`WHERE sido_code=? AND record_timestamp=? AND record_excel_path=?`** 중복 체크 쿼리용. 복합 유니크는 아니지만, 동일 (시도, 타임스탬프, 경로) 조합 조회 시 **인덱스 룩업으로 빠르게 1건 판단** 가능. |

---

## 3. 요약

- **core/db.py** 에서 성능 이득이 큰 부분은 **stats** 의 `db_list_stats_by_sido` (전체/시도별 + is_active + ORDER BY) 이므로, 위 두 개의 **stats** 인덱스를 추가하는 것을 권장합니다.
- **sido_axis_margin_stats**, **sido_templates** 는 기존 제약만으로 충분하므로 인덱스 추가 생략.
- **virtual_population_db** 는 core/db.py에는 조회 로직이 없지만, `db_init()` 에서 테이블을 생성하므로 같은 초기화 단계에서 `idx_vpd_sido_added`, `idx_vpd_sido_ts_path` 를 생성해 두면 pages 쿼리 성능에 도움이 됩니다.

---

## 4. 쿼리-인덱스 매핑 및 검토 결과 (인덱스 활용 여부)

### 4.1 검토 요약

- **LIKE / 비조건**: core/db.py 조회 함수에는 `LIKE`, `LOWER(col)=?`, `CAST(col AS ...)` 등 인덱스를 타지 못하게 만드는 조건이 **없음**. 모든 WHERE는 동등 조건(`= ?` 또는 `= 1`)만 사용.
- **결론**: 현재 쿼리는 인덱스를 활용할 수 있는 형태이며, **추가 쿼리 수정 없이** 새로 만든 인덱스가 효율적으로 사용 가능함.

### 4.2 stats — db_list_stats_by_sido

| 분기 | 쿼리 | 사용 인덱스 | 설명 |
|------|------|-------------|------|
| sido_code='00', active_only=True | `WHERE is_active=1 ORDER BY sido_code, category, name` | **idx_stats_is_active_order** | `is_active=1` 범위 스캔 후 인덱스 순서가 (sido_code, category, name)와 일치 → 정렬 생략. |
| sido_code='00', active_only=False | `ORDER BY sido_code, category, name` | **ux_stats_sido_category_name** | 전체 스캔 시 인덱스 순서가 ORDER BY와 동일 → 정렬 생략. |
| sido_code≠'00', active_only=True | `WHERE sido_code=? AND is_active=1 ORDER BY category, name` | **idx_stats_sido_active_cat** | (sido_code, is_active) 동등 후 (category, name) 순서로 반환 → 정렬 생략. |
| sido_code≠'00', active_only=False | `WHERE sido_code=? ORDER BY category, name` | **ux_stats_sido_category_name** | (sido_code) 동등 후 (category, name) 순서로 반환 → 정렬 생략. |

### 4.3 기타 조회 함수

| 함수 | 쿼리 | 사용 인덱스 |
|------|------|-------------|
| db_get_axis_margin_stats | `WHERE sido_code=? AND axis_key=?` | UNIQUE(sido_code, axis_key) |
| db_get_template | `WHERE sido_code=?` | PRIMARY KEY(sido_code) |
| db_delete_stat_by_id / db_update_stat_by_id | `WHERE id=?` | PRIMARY KEY(id) |

### 4.4 검증 방법

실제 인덱스 사용 여부는 SQLite에서 아래로 확인할 수 있음.

```sql
EXPLAIN QUERY PLAN SELECT * FROM stats WHERE is_active=1 ORDER BY sido_code, category, name;
EXPLAIN QUERY PLAN SELECT * FROM stats WHERE sido_code=? ORDER BY category, name;
```

출력에 `SEARCH TABLE stats USING INDEX idx_...` 또는 `USING INDEX ux_...` 가 나오면 해당 인덱스 사용 중.
