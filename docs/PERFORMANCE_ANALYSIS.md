# 성능 저하 원인 분석 및 대응

## 1. 핵심 원인: `get_sido_vdb_stats()` (가상인구 DB 페이지)

### 현상
- **가상인구 DB** 페이지 진입/새로고침 시 전체 앱이 매우 느려짐.

### 원인
- `core/db.py`의 **get_sido_vdb_stats()** 가 **전국** `virtual_population_db` 행을 조회할 때 **data_json 컬럼을 포함**해 가져옴.
- `sido_code=None` 이므로 **모든 시·도** 대상.
- `page_size=VPD_FETCH_PAGE_SIZE_HEAVY`(10) 이라 **10행씩** 수백~수천 번 Supabase 왕복.
- 각 행의 `data_json`은 수백 KB~MB급이라 **전송량·파싱 부하**가 매우 큼.
- `_vpd_group_and_merge()` 에서 청크별로 JSON 병합·파싱까지 수행해 **CPU·메모리**도 많이 사용.

### 호출 위치
- `pages/virtual_population_db.py` 164행: 지도/드롭다운 옆 **시도별 인원 수** 표시를 위해 매번 호출.
- `@st.cache_data(ttl=60)` 이라 **60초마다 캐시 만료** 후 다시 전국 data_json 조회.

### 결론
- 지도에 "N명"만 보여주기 위해 **전국 대용량 data_json**을 매번 가져오는 구조가 주된 병목.

---

## 2. 대응: `row_count` 컬럼 + 경량 집계

### 조치
1. **virtual_population_db** 테이블에 **row_count** 컬럼 추가 (각 청크가 담고 있는 **인원 수**만 저장).
2. **get_sido_vdb_stats()** 는 **data_json 없이** `sido_code`, `sido_name`, `row_count` 만 조회해 시도별로 합산.
3. INSERT/UPDATE 시 **row_count** 를 함께 넣어 주어, 이후 통계는 항상 경량 조회만 하도록 함.

### 효과
- 가상인구 DB 페이지 로딩 시 **전국 data_json 조회 제거** → Supabase 왕복·전송량·파싱이 크게 감소.
- 지도/통계는 **작은 메타만** 조회하므로 체감 속도 개선.

### 마이그레이션
- `docs/SUPABASE_VPD_ROW_COUNT_MIGRATION.sql` 참고하여 **row_count** 컬럼 추가 및 기존 데이터 backfill(선택).

---

## 3. 기타 참고 사항

| 항목 | 설명 |
|------|------|
| **get_virtual_population_db_combined_df** | 설문/심층면접에서 **해당 시도만** data_json 조회. 캐시 60초. 필요 시에만 호출되므로 get_sido_vdb_stats보다 영향 작음. |
| **list_virtual_population_db_metadata** | **data_json 제외** 메타만 조회. 이미 경량. |
| **list_virtual_population_db_data_jsons** | **전체 데이터 불러오기** 버튼 클릭 시에만 호출. 사용자 액션에 따른 부하는 유지. |
| **st.cache_data / st.cache_resource** | 60초 TTL 등으로 캐시 사용 중. get_sido_vdb_stats의 **캐시 미스 시** 비용이 가장 큼. |

---

## 4. 요약

- **주된 원인:** 가상인구 DB 페이지에서 시도별 "N명" 표시를 위해 **get_sido_vdb_stats()** 가 전국 **data_json**을 매번(캐시 만료 시) 조회·병합·집계함.
- **해결:** **row_count** 컬럼 도입 후 통계는 **row_count만** 조회해 합산하도록 변경하여, data_json 조회를 제거함.
