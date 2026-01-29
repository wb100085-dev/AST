# utils/kosis_client.py
from __future__ import annotations

import re
import time
import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
import numpy as np

from utils.gemini_client import GeminiClient


# ========================================
# 컬럼 매핑 정규화 딕셔너리
# ========================================
COLUMN_MAPPING = {
    # 6축 기본 컬럼
    "거주지역": "거주지역",
    "성별": "성별",
    "연령": "연령",
    "경제활동": "경제활동",
    "교육정도": "교육정도",
    "월평균소득": "월평균소득",
    # 추가 통계 컬럼 (일반적인 패턴)
    "주택유형": "주택유형",
    "거처종류": "거처종류",
    "반려동물": "반려동물",
    "주택만족도": "주택만족도",
    "주택형태": "주택형태",
    "주거형태": "주거형태",
}


def apply_weighted_sampling(
    stats_dict: Dict[str, float],
    default_value: str = "Unknown"
) -> str:
    """
    확률 분포에 따른 가중치 샘플링
    
    Args:
        stats_dict: {카테고리명: 가중치(빈도/비율)} 형태의 딕셔너리
                   예: {'아파트': 50, '빌라': 30, '단독주택': 20}
        default_value: 통계 데이터가 없을 때 반환할 기본값
    
    Returns:
        샘플링된 카테고리명 (예: '아파트')
    """
    if not stats_dict:
        return default_value
    
    # 값 정규화 (음수 제거, 0 이하 값 처리)
    cleaned_dict = {k: max(0.0, float(v)) for k, v in stats_dict.items() if v is not None}
    
    if not cleaned_dict:
        return default_value
    
    # 총합 계산
    total = sum(cleaned_dict.values())
    
    if total <= 0:
        # 모든 값이 0이면 균등 분포
        keys = list(cleaned_dict.keys())
        return random.choice(keys) if keys else default_value
    
    # 확률 정규화
    normalized_weights = {k: v / total for k, v in cleaned_dict.items()}
    
    # random.choices를 사용한 가중치 샘플링
    categories = list(normalized_weights.keys())
    weights = list(normalized_weights.values())
    
    try:
        sampled = random.choices(categories, weights=weights, k=1)[0]
        return sampled
    except (ValueError, IndexError):
        # 가중치 오류 시 첫 번째 키 반환
        return categories[0] if categories else default_value


@dataclass
class StatItem:
    id: int
    category: str
    name: str
    url: str
    is_active: int = 1


class KosisClient:
    def __init__(self, cache_ttl_seconds: int = 3600, timeout_seconds: int = 30, use_gemini: bool = True):
        self.cache_ttl = cache_ttl_seconds
        self.timeout = timeout_seconds
        self._cache: Dict[str, Tuple[float, Any]] = {}

        self.use_gemini = bool(use_gemini)
        self._gemini: Optional[GeminiClient] = None
        if self.use_gemini:
            try:
                self._gemini = GeminiClient()
            except Exception:
                self._gemini = None

    def fetch_json(self, url: str) -> List[Dict[str, Any]]:
        key = self._cache_key(url)
        now = time.time()

        if key in self._cache:
            ts, data = self._cache[key]
            if now - ts <= self.cache_ttl:
                return data

        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()

        data = resp.json()
        if not isinstance(data, list):
            data = data.get("data", []) if isinstance(data, dict) else []

        self._cache[key] = (now, data)
        return data

    def extract_sigungu_list_from_population_table(
        self,
        data: List[Dict[str, Any]],
        *,
        sido_prefix: str = "경상북도",
    ) -> List[str]:
        names = set()
        for row in data:
            c1_nm = str(row.get("C1_NM", "")).strip()
            if not c1_nm:
                continue
            if c1_nm == sido_prefix:
                continue
            if re.search(r"(시|군)$", c1_nm):
                names.add(c1_nm)
        return sorted(names)

    def url_info(self, url: str) -> Dict[str, Any]:
        u = urlparse(url)
        q = parse_qs(u.query)
        return {
            "path": u.path,
            "endpoint": "statisticsParameterData.do" if "statisticsParameterData.do" in u.path else "statisticsData.do",
            "method": (q.get("method", [""])[0] if q else ""),
            "userStatsId": (q.get("userStatsId", [""])[0] if q else ""),
            "tblId": (q.get("tblId", [""])[0] if q else ""),
            "orgId": (q.get("orgId", [""])[0] if q else ""),
        }

    def reorder_columns_by_category(self, df: pd.DataFrame, *, base_cols: List[str]) -> pd.DataFrame:
        exist_base = [c for c in base_cols if c in df.columns]
        other = [c for c in df.columns if c not in exist_base]

        def _cat_rank(col: str) -> Tuple[int, str]:
            m = re.match(r"^kosis_(\d{2})__", str(col))
            if not m:
                return (999, str(col))
            return (int(m.group(1)), str(col))

        other_sorted = sorted(other, key=_cat_rank)
        return df[exist_base + other_sorted]

    # -----------------------------
    # Gemini 룰 추출: gemini_client.py의 함수명에 맞춰 고정
    # -----------------------------
    def get_assignment_rule_with_gemini(
        self,
        *,
        category: str,
        stat_name: str,
        url: Optional[str],
        data: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self._gemini:
            return None

        axis_keys_kr = {
            "거주지역": "sigungu",
            "성별": "gender",
            "연령": "age",
            "경제활동": "econ",
            "교육정도": "edu",
            "월평균소득": "income",
        }

        try:
            return self._gemini.extract_kosis_assignment_rule(
                category_code=str(category),
                stat_name=str(stat_name),
                kosis_url=str(url or ""),
                kosis_data_sample=data[:50],
                axis_keys_kr=axis_keys_kr,
            )
        except Exception:
            return None

    # -----------------------------
    # 통계목록 N개면 컬럼 N개 이상 "무조건 생성" 보장
    # -----------------------------
    def assign_stat_columns_to_population(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        category: str,
        stat_name: str,
        url: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        out = pop_df.copy()

        cat_code = self._extract_category_code(category)
        col_name = f"kosis_{cat_code}__{self._slug(stat_name)}"

        # KOSIS 데이터가 비어도 컬럼은 생성
        if not kosis_data:
            out[col_name] = [None] * len(out)
            return out, False

        rule = self.get_assignment_rule_with_gemini(category=category, stat_name=stat_name, url=url, data=kosis_data)
        if not rule:
            # Gemini 실패해도 “컬럼 1개 생성” 보장(대표값으로 채움)
            rep = self._fallback_representative_value(kosis_data)
            out[col_name] = [rep] * len(out)
            return out, False

        # 룰 기반 대입(최소 동작 위주)
        mode = str(rule.get("mode", "categorical_sampling")).lower()

        value_field = "DT"
        if rule.get("value_field"):
            value_field = str(rule.get("value_field"))
        elif isinstance(rule.get("value_field_candidates", None), list) and rule["value_field_candidates"]:
            value_field = str(rule["value_field_candidates"][0])

        item_field = "ITM_NM"
        if isinstance(rule.get("label_fields", None), list) and rule["label_fields"]:
            item_field = str(rule["label_fields"][0])

        dims_raw = rule.get("dimensions", {}) or {}
        # dimensions가 리스트인 경우 딕셔너리로 변환 시도
        if isinstance(dims_raw, list):
            dims = {}
            # 리스트를 딕셔너리로 변환 시도 (예: [{"key": "거주지역", "source": "C1_NM"}, ...])
            for item in dims_raw:
                if isinstance(item, dict):
                    key = item.get("key") or item.get("name")
                    if key:
                        dims[key] = item
        elif isinstance(dims_raw, dict):
            dims = dims_raw
        else:
            dims = {}

        base_keys = ["거주지역", "성별", "연령", "경제활동", "교육정도", "월평균소득"]
        for k in base_keys:
            if k not in out.columns:
                out[k] = None

        def _dim_source(key_kr: str) -> Optional[str]:
            if not isinstance(dims, dict):
                return None
            v = dims.get(key_kr)
            if isinstance(v, dict):
                return v.get("source") or v.get("field")
            if isinstance(v, str):
                return v
            return None

        index_dist: Dict[Tuple[str, ...], Dict[str, float]] = {}
        index_num: Dict[Tuple[str, ...], float] = {}

        # ✅ 카테고리 값을 추출할 필드 후보 목록
        # 통계명이 아닌 실제 카테고리 값(예: '아파트', '빌라')을 찾기 위함
        category_field_candidates = [item_field]
        if isinstance(rule.get("label_fields", None), list):
            category_field_candidates.extend(rule["label_fields"])
        # 일반적인 KOSIS 필드명 추가
        category_field_candidates.extend(["C1_NM", "C2_NM", "C3_NM", "C4_NM", "ITM_NM", "PRD_NM"])

        for r in kosis_data:
            key_parts = []
            for pk in base_keys:
                src = _dim_source(pk)
                if not src:
                    continue
                key_parts.append(str(r.get(src, "")).strip())
            kkey = tuple(key_parts)

            raw = r.get(value_field, None)
            if raw is None:
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue

            if "categorical" in mode:
                # ✅ 실제 카테고리 값 추출 (통계명·메타 컬럼명 제외)
                EXCLUDE_LABELS = ("계", "합계", "소계", "Total", "비율")
                lab = None
                for field in category_field_candidates:
                    candidate = str(r.get(field, "")).strip()
                    # 통계명/메타데이터/비율 컬럼이 아닌 실제 값만 사용
                    if candidate and candidate not in [*EXCLUDE_LABELS, stat_name]:
                        if not re.match(r'^\d+\.?\d*%?$', candidate):
                            lab = candidate
                            break
                if not lab:
                    for field in category_field_candidates:
                        candidate = str(r.get(field, "")).strip()
                        if candidate and candidate not in [*EXCLUDE_LABELS, "계", "합계", "소계", "Total"]:
                            lab = candidate
                            break
                if not lab:
                    continue
                index_dist.setdefault(kkey, {})
                index_dist[kkey][lab] = index_dist[kkey].get(lab, 0.0) + v
            else:
                index_num[kkey] = v

        assigned: List[Any] = []
        for _, row in out.iterrows():
            key_parts = []
            for pk in base_keys:
                src = _dim_source(pk)
                if not src:
                    continue
                key_parts.append(str(row.get(pk, "")).strip())
            kkey = tuple(key_parts)

            if "categorical" in mode:
                # ✅ 확률 분포 기반 샘플링 사용
                dist = index_dist.get(kkey)
                if not dist:
                    # 매칭되는 통계가 없으면 기본값 사용
                    assigned.append("Unknown")
                    continue
                
                # apply_weighted_sampling 함수 사용
                sampled_value = apply_weighted_sampling(dist, default_value="Unknown")
                assigned.append(sampled_value)
            else:
                # 수치형 데이터는 그대로 사용
                numeric_value = index_num.get(kkey, None)
                assigned.append(numeric_value)

        out[col_name] = assigned
        return out, True

    def _fallback_representative_value(self, data: List[Dict[str, Any]]) -> Optional[float]:
        candidates = ["DT", "DATA", "VAL", "VALUE"]
        nums: List[float] = []
        for r in data:
            for f in candidates:
                raw = r.get(f, None)
                if raw is None:
                    continue
                try:
                    nums.append(float(str(raw).replace(",", "").strip()))
                    break
                except Exception:
                    continue
        if not nums:
            return None
        return sum(nums) / len(nums)

    def _cache_key(self, url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _extract_category_code(self, category: str) -> str:
        m = re.match(r"^\s*(\d{2})\.", str(category))
        if m:
            return m.group(1)
        m2 = re.match(r"^\s*(\d{2})\s", str(category))
        if m2:
            return m2.group(1)
        return "99"

    def _slug(self, s: str, max_len: int = 60) -> str:
        s = str(s).strip().lower()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^0-9a-zA-Z가-힣_]+", "", s)
        if len(s) > max_len:
            s = s[:max_len]
        return s or "na"

    # -----------------------------
    # 거주기간·정주의사 형 통계: 확률 분포 + 행 간 제약(시도거주기간 >= 시군구거주기간) + 다열 대입
    # -----------------------------

    # 거주기간 순서(짧음 → 김): 인덱스가 클수록 더 긴 거주기간
    RESIDENCE_DURATION_ORDER = [
        "5년 미만",
        "5년 이상 10년 미만",
        "5년~10년",
        "10년 이상 20년 미만",
        "10년~20년",
        "20년 이상",
    ]
    # 표준 라벨로 정규화하는 매핑 (KOSIS 변형 → 표준)
    RESIDENCE_DURATION_NORMALIZE = {
        "5년 미만": "5년 미만",
        "5년이상 10년미만": "5년 이상 10년 미만",
        "5년 이상 10년 미만": "5년 이상 10년 미만",
        "5년~10년": "5년 이상 10년 미만",
        "10년이상 20년미만": "10년 이상 20년 미만",
        "10년 이상 20년 미만": "10년 이상 20년 미만",
        "10년~20년": "10년 이상 20년 미만",
        "20년이상": "20년 이상",
        "20년 이상": "20년 이상",
    }
    INTENT_ORDER = [
        "전혀 그렇지 않다",
        "그렇지 않다",
        "보통이다",
        "그렇다",
        "매우 그렇다",
    ]
    # 통계에 포함하면 안 되는 계열 라벨 (비율 샘플링 제외)
    EXCLUDE_FROM_DIST = ("계", "합계", "소계", "Total", "평균", "평균값")

    def _is_residence_duration_label(self, label: str) -> bool:
        """라벨이 거주기간 구간(5년 미만, 20년 이상 등)이면 True. 평균·정주의사 제외."""
        if not label or label.strip() in self.EXCLUDE_FROM_DIST:
            return False
        s = label.strip()
        if "년" in s and ("미만" in s or "이상" in s or "~" in s):
            return True
        if s in self.RESIDENCE_DURATION_NORMALIZE:
            return True
        return False

    def _is_intent_label(self, label: str) -> bool:
        """라벨이 향후 10년 거주 희망의사(정주의사) 응답이면 True."""
        if not label or label.strip() in self.EXCLUDE_FROM_DIST:
            return False
        s = label.strip()
        if s in self.INTENT_ORDER:
            return True
        if "그렇지" in s or "그렇다" in s or "보통" in s or "그런 편" in s:
            return True
        return False

    def _normalize_residence_label(self, label: str) -> str:
        if not label or not isinstance(label, str):
            return ""
        key = label.strip()
        return self.RESIDENCE_DURATION_NORMALIZE.get(key, key)

    def _residence_label_to_order(self, label: str) -> int:
        """거주기간 라벨의 순서 인덱스 (0=가장 짧음)."""
        norm = self._normalize_residence_label(label)
        for i, standard in enumerate(["5년 미만", "5년 이상 10년 미만", "10년 이상 20년 미만", "20년 이상"]):
            if standard in norm or norm == standard:
                return i
        if "5년" in norm and "10년" not in norm:
            return 0
        if "10년" in norm and "20년" not in norm:
            return 1
        if "20년" in norm:
            return 3
        return 0

    def parse_residence_duration_kosis(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        KOSIS '총거주기간 및 정주의사' 형 JSON에서
        시도거주기간, 시군구거주기간, 향후 10년 거주 희망의사 비율을 추출.
        - 거주기간 열에는 '5년 미만', '20년 이상' 등만 사용하고 '평균'·정주의사 문구는 제외.
        - 향후 10년 열에는 '전혀 그렇지 않다', '매우 그렇다' 등만 사용.
        반환: (시도_dist, 시군구_dist, 향후10년_dist) 각각 {라벨: 비율(0~1)}.
        """
        value_field = "DT"
        dim_fields = ["C2_NM", "C3_NM", "C4_NM"]
        margins: List[Dict[str, float]] = [{}, {}, {}]
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            raw = r.get(value_field)
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            for i, dim_key in enumerate(dim_fields):
                lab = str(r.get(dim_key, "")).strip()
                if not lab or lab in self.EXCLUDE_FROM_DIST:
                    continue
                # 시도(0)·시군구(1): 거주기간 라벨만 (평균·정주의사 문구 제외)
                if i in (0, 1) and self._is_residence_duration_label(lab):
                    margins[i][lab] = margins[i].get(lab, 0.0) + v
                # 향후 10년(2): 정주의사 라벨만. C2/C3에 정주의사가 섞여 있어도 m2로만 수집
                elif self._is_intent_label(lab):
                    margins[2][lab] = margins[2].get(lab, 0.0) + v
        # 정규화하여 비율로
        def to_probs(d: Dict[str, float]) -> Dict[str, float]:
            total = sum(d.values())
            if total <= 0:
                return d
            return {k: v / total for k, v in d.items()}

        m0, m1, m2 = [to_probs(m) for m in margins]
        # KOSIS가 C3_NM/C4_NM 없이 C2_NM만 있는 경우: 동일 비율을 시군구에도 사용
        if not m1 and m0:
            m1 = dict(m0)
        if not m2:
            m2 = {"보통이다": 1.0}  # 향후 10년은 별도 항목이 없으면 기본값
        return (m0, m1, m2)

    def generate_residence_duration_with_constraint(
        self,
        n: int,
        dist_sido: Dict[str, float],
        dist_sigungu: Dict[str, float],
        dist_intent: Dict[str, float],
        seed: Optional[int] = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        N명에 대해 시도거주기간, 시군구거주기간, 향후 10년 거주 희망의사를 생성.
        제약: 시군구 거주기간 <= 시도 거주기간 (순서 기준).
        """
        if seed is not None:
            np.random.seed(seed)
        labels_sido = list(dist_sido.keys()) if dist_sido else ["5년 미만"]
        probs_sido = list(dist_sido.values()) if dist_sido else [1.0]
        if sum(probs_sido) <= 0:
            probs_sido = [1.0 / len(labels_sido)] * len(labels_sido)
        else:
            probs_sido = [p / sum(probs_sido) for p in probs_sido]

        labels_sigungu = list(dist_sigungu.keys()) if dist_sigungu else ["5년 미만"]
        probs_sigungu = list(dist_sigungu.values()) if dist_sigungu else [1.0]
        if sum(probs_sigungu) <= 0:
            probs_sigungu = [1.0 / len(labels_sigungu)] * len(labels_sigungu)
        else:
            probs_sigungu = [p / sum(probs_sigungu) for p in probs_sigungu]

        labels_intent = list(dist_intent.keys()) if dist_intent else ["보통이다"]
        probs_intent = list(dist_intent.values()) if dist_intent else [1.0]
        if sum(probs_intent) <= 0:
            probs_intent = [1.0 / len(labels_intent)] * len(labels_intent)
        else:
            probs_intent = [p / sum(probs_intent) for p in probs_intent]

        out_sido: List[str] = []
        out_sigungu: List[str] = []
        out_intent: List[str] = []
        for _ in range(n):
            sido = str(np.random.choice(labels_sido, p=probs_sido))
            o_sido = self._residence_label_to_order(sido)
            # 시군구는 시도 이하만 허용
            candidates_sg = [(l, self._residence_label_to_order(l)) for l in labels_sigungu]
            allowed = [l for l, o in candidates_sg if o <= o_sido]
            if not allowed:
                allowed = [l for l, o in candidates_sg if o <= o_sido + 1]
            if not allowed:
                allowed = labels_sigungu
            idx_allowed = [labels_sigungu.index(l) for l in allowed]
            p_allowed = [probs_sigungu[i] for i in idx_allowed]
            s = sum(p_allowed)
            if s <= 0:
                sigungu = allowed[0]
            else:
                sigungu = str(np.random.choice(allowed, p=[x / s for x in p_allowed]))
            intent = str(np.random.choice(labels_intent, p=probs_intent))
            out_sido.append(sido)
            out_sigungu.append(sigungu)
            out_intent.append(intent)
        return (out_sido, out_sigungu, out_intent)

    def assign_residence_duration_columns(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        column_names: Tuple[str, str, str],
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        '총거주기간 및 정주의사' 형 통계: 시도거주기간, 시군구거주기간, 향후 10년 거주 희망의사
        세 컬럼을 확률 분포 + 제약(시도 >= 시군구)으로 생성해 채움.
        column_names: (시도거주기간 컬럼명, 시군구거주기간 컬럼명, 향후 10년 거주 희망의사 컬럼명)
        """
        out = pop_df.copy()
        n = len(out)
        if not kosis_data:
            for col in column_names:
                out[col] = ""
            return out, False
        dist_sido, dist_sigungu, dist_intent = self.parse_residence_duration_kosis(kosis_data)
        if not dist_sido and not dist_sigungu and not dist_intent:
            for col in column_names:
                out[col] = ""
            return out, False
        sid_list, sig_list, intent_list = self.generate_residence_duration_with_constraint(
            n, dist_sido, dist_sigungu, dist_intent, seed=seed
        )
        out[column_names[0]] = sid_list
        out[column_names[1]] = sig_list
        out[column_names[2]] = intent_list
        return out, True

    # -----------------------------
    # 학생 및 미취학자녀수: 2열(자녀 유무, 자녀 수) — F열·G열
    # -----------------------------
    CHILDREN_HAS_LABELS = ("있다", "없다")
    CHILDREN_COUNT_LABELS = ("1명", "2명", "3명이상")

    def parse_children_student_kosis(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        KOSIS '학생 및 미취학자녀수' 형 JSON에서
        자녀 유무(있다/없다) 비율과, 자녀가 있을 때 자녀 수(1명/2명/3명이상) 비율 추출.
        C1_NM='전체' 행만 사용. C2_NM '평균' 제외.
        반환: (dist_has, dist_count) 각각 {라벨: 비율(0~1)}.
        """
        dist_has: Dict[str, float] = {}
        dist_count: Dict[str, float] = {}
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            if str(r.get("C1_NM") or "").strip() != "전체":
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            lab = str(r.get("C2_NM") or "").strip()
            if not lab or lab == "평균":
                continue
            if lab in self.CHILDREN_HAS_LABELS:
                dist_has[lab] = dist_has.get(lab, 0.0) + v
            elif lab in self.CHILDREN_COUNT_LABELS:
                dist_count[lab] = dist_count.get(lab, 0.0) + v
        def to_probs(d: Dict[str, float]) -> Dict[str, float]:
            total = sum(d.values())
            if total <= 0:
                return d
            return {k: v / total for k, v in d.items()}
        return to_probs(dist_has), to_probs(dist_count)

    # F열 표기: 유/무 (KOSIS '있다'→유, '없다'→무)
    CHILDREN_HAS_DISPLAY = {"있다": "유", "없다": "무"}
    # G열: 총학생 수 숫자. 1명→1, 2명→2, 3명이상→3, 자녀 없음→0
    CHILDREN_COUNT_TO_NUM = {"1명": 1, "2명": 2, "3명이상": 3}

    def generate_children_student(
        self,
        n: int,
        dist_has: Dict[str, float],
        dist_count: Dict[str, float],
        seed: Optional[int] = None,
    ) -> Tuple[List[str], List[Any]]:
        """
        N명에 대해 F열=유/무, G열=총학생 수(0,1,2,3) 생성.
        자녀 없음(무)이면 G열은 0.
        """
        if seed is not None:
            np.random.seed(seed)
        labels_has = list(dist_has.keys()) if dist_has else ["없다"]
        probs_has = list(dist_has.values()) if dist_has else [1.0]
        if sum(probs_has) <= 0:
            probs_has = [1.0 / len(labels_has)] * len(labels_has)
        else:
            probs_has = [p / sum(probs_has) for p in probs_has]
        labels_count = list(dist_count.keys()) if dist_count else ["1명"]
        probs_count = list(dist_count.values()) if dist_count else [1.0]
        if sum(probs_count) <= 0:
            probs_count = [1.0 / len(labels_count)] * len(labels_count)
        else:
            probs_count = [p / sum(probs_count) for p in probs_count]
        out_has: List[str] = []
        out_count: List[Any] = []
        for _ in range(n):
            has_val = str(np.random.choice(labels_has, p=probs_has))
            out_has.append(self.CHILDREN_HAS_DISPLAY.get(has_val, "무" if has_val == "없다" else "유"))
            if has_val == "있다":
                count_lab = str(np.random.choice(labels_count, p=probs_count))
                out_count.append(self.CHILDREN_COUNT_TO_NUM.get(count_lab, 1))
            else:
                out_count.append(0)
        return (out_has, out_count)

    def assign_children_student_columns(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        column_names: Tuple[str, str],
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        '학생 및 미취학자녀수' 형 통계: F열(자녀 유무), G열(자녀 수) 두 컬럼을
        확률 분포로 생성해 채움. 자녀 없으면 G열은 빈 문자열.
        column_names: (자녀 유무 컬럼명, 자녀 수 컬럼명)
        """
        out = pop_df.copy()
        n = len(out)
        if not kosis_data:
            for col in column_names:
                out[col] = ""
            return out, False
        dist_has, dist_count = self.parse_children_student_kosis(kosis_data)
        if not dist_has:
            for col in column_names:
                out[col] = ""
            return out, False
        has_list, count_list = self.generate_children_student(
            n, dist_has, dist_count, seed=seed
        )
        out[column_names[0]] = has_list
        out[column_names[1]] = count_list
        return out, True

    # -----------------------------
    # 반려동물 현황: 2열(K=유무 예/아니오, L=종류) — K열·L열
    # -----------------------------
    PET_HAS_LABELS = ("예", "아니오")

    def parse_pet_kosis(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        KOSIS '반려동물 현황' 형 JSON에서
        반려동물 유무(예/아니오) 비율과, 유할 때 종류(개(강아지), 고양이, 새, 물고기, 기타) 비율 추출.
        C1_NM='전체' 행만 사용. DT가 '-'이면 제외.
        반환: (dist_has, dist_type) 각각 {라벨: 비율(0~1)}.
        """
        dist_has: Dict[str, float] = {}
        dist_type: Dict[str, float] = {}
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            if str(r.get("C1_NM") or "").strip() != "전체":
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            lab = str(r.get("C2_NM") or "").strip()
            if not lab:
                continue
            if lab in self.PET_HAS_LABELS:
                dist_has[lab] = dist_has.get(lab, 0.0) + v
            else:
                dist_type[lab] = dist_type.get(lab, 0.0) + v
        def to_probs(d: Dict[str, float]) -> Dict[str, float]:
            total = sum(d.values())
            if total <= 0:
                return d
            return {k: v / total for k, v in d.items()}
        return to_probs(dist_has), to_probs(dist_type)

    def generate_pet(
        self,
        n: int,
        dist_has: Dict[str, float],
        dist_type: Dict[str, float],
        seed: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        N명에 대해 K열=예/아니오, L열=종류(개(강아지), 고양이 등, 해당 없으면 빈 문자열) 생성.
        """
        if seed is not None:
            np.random.seed(seed)
        labels_has = list(dist_has.keys()) if dist_has else ["아니오"]
        probs_has = list(dist_has.values()) if dist_has else [1.0]
        if sum(probs_has) <= 0:
            probs_has = [1.0 / len(labels_has)] * len(labels_has)
        else:
            probs_has = [p / sum(probs_has) for p in probs_has]
        labels_type = list(dist_type.keys()) if dist_type else ["개(강아지)"]
        probs_type = list(dist_type.values()) if dist_type else [1.0]
        if sum(probs_type) <= 0:
            probs_type = [1.0 / len(labels_type)] * len(labels_type)
        else:
            probs_type = [p / sum(probs_type) for p in probs_type]
        out_has: List[str] = []
        out_type: List[str] = []
        for _ in range(n):
            has_val = str(np.random.choice(labels_has, p=probs_has))
            out_has.append(has_val)
            if has_val == "예":
                out_type.append(str(np.random.choice(labels_type, p=probs_type)))
            else:
                out_type.append("")
        return (out_has, out_type)

    def assign_pet_columns(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        column_names: Tuple[str, str],
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        '반려동물 현황' 형 통계: K열(유무 예/아니오), L열(종류) 두 컬럼을 확률 분포로 생성해 채움.
        column_names: (유무 컬럼명, 종류 컬럼명)
        """
        out = pop_df.copy()
        n = len(out)
        if not kosis_data:
            for col in column_names:
                out[col] = ""
            return out, False
        dist_has, dist_type = self.parse_pet_kosis(kosis_data)
        if not dist_has:
            for col in column_names:
                out[col] = ""
            return out, False
        has_list, type_list = self.generate_pet(n, dist_has, dist_type, seed=seed)
        out[column_names[0]] = has_list
        out[column_names[1]] = type_list
        return out, True

    # -----------------------------
    # 거처 종류 및 점유 형태: 2열(거처 종류, 주택 점유 형태) — 독립 분포
    # -----------------------------

    def parse_dwelling_kosis(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        KOSIS '거처 종류 및 점유형태' 형 JSON에서
        C1_NM='전체' 행만 사용. C2 코드 B01xx = 거처 종류, B02xx = 주택 점유 형태.
        반환: (dist_dwelling, dist_occupancy) 각각 {라벨: 비율(0~1)}.
        """
        dist_dwelling: Dict[str, float] = {}
        dist_occupancy: Dict[str, float] = {}
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            if str(r.get("C1_NM") or "").strip() != "전체":
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            lab = str(r.get("C2_NM") or "").strip()
            c2 = str(r.get("C2") or "").strip()
            if not lab:
                continue
            if c2.startswith("B01"):
                dist_dwelling[lab] = dist_dwelling.get(lab, 0.0) + v
            elif c2.startswith("B02"):
                dist_occupancy[lab] = dist_occupancy.get(lab, 0.0) + v

        def to_probs(d: Dict[str, float]) -> Dict[str, float]:
            total = sum(d.values())
            if total <= 0:
                return d
            return {k: v / total for k, v in d.items()}

        return to_probs(dist_dwelling), to_probs(dist_occupancy)

    def generate_dwelling(
        self,
        n: int,
        dist_dwelling: Dict[str, float],
        dist_occupancy: Dict[str, float],
        seed: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        N명에 대해 '거처 종류', '주택 점유 형태' 두 컬럼을 독립 확률로 생성 (문자열).
        """
        if seed is not None:
            np.random.seed(seed)
        labels_dw = list(dist_dwelling.keys()) if dist_dwelling else ["아파트"]
        probs_dw = list(dist_dwelling.values()) if dist_dwelling else [1.0]
        if sum(probs_dw) <= 0:
            probs_dw = [1.0 / len(labels_dw)] * len(labels_dw)
        else:
            probs_dw = [p / sum(probs_dw) for p in probs_dw]
        labels_occ = list(dist_occupancy.keys()) if dist_occupancy else ["자기집"]
        probs_occ = list(dist_occupancy.values()) if dist_occupancy else [1.0]
        if sum(probs_occ) <= 0:
            probs_occ = [1.0 / len(labels_occ)] * len(labels_occ)
        else:
            probs_occ = [p / sum(probs_occ) for p in probs_occ]
        out_dw: List[str] = []
        out_occ: List[str] = []
        for _ in range(n):
            out_dw.append(str(np.random.choice(labels_dw, p=probs_dw)))
            out_occ.append(str(np.random.choice(labels_occ, p=probs_occ)))
        return (out_dw, out_occ)

    def assign_dwelling_columns(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        column_names: Tuple[str, str],
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        '거처 종류 및 점유형태' 통계: '거처 종류', '주택 점유 형태' 두 컬럼을 확률 분포로 생성해 채움.
        column_names: (거처 종류 컬럼명, 주택 점유 형태 컬럼명)
        """
        out = pop_df.copy()
        n = len(out)
        if not kosis_data:
            for col in column_names:
                out[col] = ""
            return out, False
        dist_dwelling, dist_occupancy = self.parse_dwelling_kosis(kosis_data)
        if not dist_dwelling and not dist_occupancy:
            for col in column_names:
                out[col] = ""
            return out, False
        if not dist_dwelling:
            dist_dwelling = {"아파트": 1.0}
        if not dist_occupancy:
            dist_occupancy = {"자기집": 1.0}
        dw_list, occ_list = self.generate_dwelling(n, dist_dwelling, dist_occupancy, seed=seed)
        out[column_names[0]] = dw_list
        out[column_names[1]] = occ_list
        return out, True

    # -----------------------------
    # 부모님 생존여부 및 동거여부: 2열(생존여부, 부모님 동거 여부)
    # KOSIS C2 B01* = 생존여부, B02* = 동거여부. C1_NM별(연령대 등) 분포 사용 + 행 간 논리 일관성
    # - 생존=해당없음 이면 동거=같이 살고 있지 않음 고정
    # - 연령대별 생존 분포 사용 (고령일수록 해당없음 비율 높음)
    # - age_cap(기본 75세) 이상은 생존=해당없음 고정
    #
    # [통계 추가 시 행방향 개연성 가이드]
    # - 열별 분포만 적용하면 행 간 논리 모순이 생길 수 있음(예: 해당없음인데 동거함).
    # - 조건부 적용: 다른 열 값(연령, 성별, 거주지역 등)에 따라 분포 또는 허용값을 제한할 것.
    # - 상호 배타/필수 관계: A이면 B 불가, A이면 B 필수 등 규칙을 코드로 강제할 것.
    # -----------------------------

    # 연령 값 → KOSIS C1_NM 연령대 매핑용
    AGE_BAND_ORDER = [
        "29세 이하",
        "30 ~ 39세",
        "40 ~ 49세",
        "50 ~ 59세",
        "60 ~ 69세",
        "70세 이상",
    ]

    def _age_to_parents_band(self, age_val: Any) -> str:
        """연령 컬럼값(숫자 또는 '70세 이상' 등)을 KOSIS 연령대 문자열로 변환."""
        if age_val is None or (isinstance(age_val, float) and pd.isna(age_val)):
            return "전체"
        s = str(age_val).strip()
        if s in self.AGE_BAND_ORDER:
            return s
        num = None
        for part in re.findall(r"\d+", s):
            num = int(part)
            break
        if num is None:
            return "전체"
        if num <= 29:
            return "29세 이하"
        if num <= 39:
            return "30 ~ 39세"
        if num <= 49:
            return "40 ~ 49세"
        if num <= 59:
            return "50 ~ 59세"
        if num <= 69:
            return "60 ~ 69세"
        return "70세 이상"

    def _parents_survival_is_none(self, survival_label: str) -> bool:
        """생존여부 라벨이 '해당없음'(또는 '해당 없음' 등)이면 True. 행 일관성용."""
        if not survival_label:
            return True
        s = str(survival_label).strip().replace(" ", "")
        return s == "해당없음"

    def _numeric_age_from_row(self, row: "pd.Series", age_col: Optional[str]) -> Optional[int]:
        """행에서 연령 숫자 추출 (없으면 None)."""
        if not age_col:
            return None
        val = row.get(age_col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        s = str(val).strip()
        for part in re.findall(r"\d+", s):
            return int(part)
        return None

    def parse_parents_survival_cohabitation_kosis(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        C1_NM='전체' 기준 종합 분포 반환 (기존 호환용).
        반환: (dist_survival, dist_cohabitation) 각각 {라벨: 비율(0~1)}.
        """
        by_key_s, by_key_c = self.parse_parents_survival_cohabitation_kosis_by_key(kosis_data)
        dist_s = by_key_s.get("전체", {})
        dist_c = by_key_c.get("전체", {})
        if not dist_s and by_key_s:
            dist_s = next(iter(by_key_s.values()), {})
        if not dist_c and by_key_c:
            dist_c = next(iter(by_key_c.values()), {})
        def to_probs(d: Dict[str, float]) -> Dict[str, float]:
            total = sum(d.values())
            if total <= 0:
                return d
            return {k: v / total for k, v in d.items()}
        return to_probs(dist_s), to_probs(dist_c)

    def parse_parents_survival_cohabitation_kosis_by_key(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        KOSIS '부모님 생존여부 및 동거여부' JSON에서 C1_NM별로 생존/동거 분포 추출.
        반환: (dist_survival_by_key, dist_cohabitation_by_key), key = C1_NM(전체, 29세 이하, ...).
        """
        dist_survival_by_key: Dict[str, Dict[str, float]] = {}
        dist_cohabitation_by_key: Dict[str, Dict[str, float]] = {}
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            c1_nm = str(r.get("C1_NM") or "").strip()
            if not c1_nm:
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            lab = str(r.get("C2_NM") or "").strip()
            c2 = str(r.get("C2") or "").strip()
            if not lab:
                continue
            if c1_nm not in dist_survival_by_key:
                dist_survival_by_key[c1_nm] = {}
            if c1_nm not in dist_cohabitation_by_key:
                dist_cohabitation_by_key[c1_nm] = {}
            if c2.startswith("B01"):
                dist_survival_by_key[c1_nm][lab] = dist_survival_by_key[c1_nm].get(lab, 0.0) + v
            elif c2.startswith("B02"):
                dist_cohabitation_by_key[c1_nm][lab] = dist_cohabitation_by_key[c1_nm].get(lab, 0.0) + v

        def to_probs(d: Dict[str, float]) -> Dict[str, float]:
            total = sum(d.values())
            if total <= 0:
                return d
            return {k: v / total for k, v in d.items()}

        for k in dist_survival_by_key:
            dist_survival_by_key[k] = to_probs(dist_survival_by_key[k])
        for k in dist_cohabitation_by_key:
            dist_cohabitation_by_key[k] = to_probs(dist_cohabitation_by_key[k])
        return dist_survival_by_key, dist_cohabitation_by_key

    def generate_parents_survival_cohabitation(
        self,
        n: int,
        dist_survival: Dict[str, float],
        dist_cohabitation: Dict[str, float],
        seed: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        N명에 대해 생존/동거 두 컬럼 생성. (동거는 생존=해당없음이면 '같이 살고 있지 않음'으로 일관성 유지)
        """
        if seed is not None:
            np.random.seed(seed)
        labels_survival = list(dist_survival.keys()) if dist_survival else ["해당없음"]
        probs_survival = list(dist_survival.values()) if dist_survival else [1.0]
        if sum(probs_survival) <= 0:
            probs_survival = [1.0 / len(labels_survival)] * len(labels_survival)
        else:
            probs_survival = [p / sum(probs_survival) for p in probs_survival]
        labels_cohabitation = list(dist_cohabitation.keys()) if dist_cohabitation else ["같이 살고 있지 않음"]
        probs_cohabitation = list(dist_cohabitation.values()) if dist_cohabitation else [1.0]
        if sum(probs_cohabitation) <= 0:
            probs_cohabitation = [1.0 / len(labels_cohabitation)] * len(labels_cohabitation)
        else:
            probs_cohabitation = [p / sum(probs_cohabitation) for p in probs_cohabitation]
        out_survival: List[str] = []
        out_cohabitation: List[str] = []
        no_cohab_label = "같이 살고 있지 않음"
        for _ in range(n):
            survival = str(np.random.choice(labels_survival, p=probs_survival))
            out_survival.append(survival)
            if self._parents_survival_is_none(survival):
                out_cohabitation.append(no_cohab_label)
            else:
                out_cohabitation.append(str(np.random.choice(labels_cohabitation, p=probs_cohabitation)))
        return (out_survival, out_cohabitation)

    def assign_parents_survival_cohabitation_columns(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        column_names: Tuple[str, str] = ("생존여부", "부모님 동거 여부"),
        seed: Optional[int] = None,
        age_cap: Optional[int] = 75,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        행 간 논리 일관성:
        - 연령대별 생존 분포 사용.
        - 생존=해당없음(또는 표기 변형)이면 동거=같이 살고 있지 않음 고정.
        - age_cap 이상(기본 75세)이면 생존=해당없음, 동거=같이 살고 있지 않음으로 고정(고령 논리 일관성).
        """
        out = pop_df.copy()
        n = len(out)
        if not kosis_data:
            for col in column_names:
                out[col] = ""
            return out, False
        dist_survival_by_key, dist_cohabitation_by_key = self.parse_parents_survival_cohabitation_kosis_by_key(
            kosis_data
        )
        if not dist_survival_by_key and not dist_cohabitation_by_key:
            for col in column_names:
                out[col] = ""
            return out, False

        age_col = "연령"
        if age_col not in out.columns:
            age_col = next((c for c in out.columns if "연령" in str(c)), None)
        if seed is not None:
            np.random.seed(seed)

        survival_list: List[str] = []
        cohabitation_list: List[str] = []
        default_survival = {"해당없음": 1.0}
        default_cohabitation = {"같이 살고 있지 않음": 1.0}
        no_cohab_label = "같이 살고 있지 않음"
        none_survival_label = "해당없음"

        for _, row in out.iterrows():
            numeric_age = self._numeric_age_from_row(row, age_col)
            # 고령: age_cap 이상이면 부모 생존 해당없음으로 고정
            if age_cap is not None and numeric_age is not None and numeric_age >= age_cap:
                survival_list.append(none_survival_label)
                cohabitation_list.append(no_cohab_label)
                continue
            age_band = self._age_to_parents_band(row.get(age_col)) if age_col else "전체"
            dist_s = dist_survival_by_key.get(age_band) or dist_survival_by_key.get("전체") or default_survival
            dist_c = dist_cohabitation_by_key.get(age_band) or dist_cohabitation_by_key.get("전체") or default_cohabitation
            labels_s = list(dist_s.keys())
            probs_s = list(dist_s.values())
            if not labels_s or sum(probs_s) <= 0:
                labels_s, probs_s = [none_survival_label], [1.0]
            else:
                probs_s = [p / sum(probs_s) for p in probs_s]
            survival = str(np.random.choice(labels_s, p=probs_s))
            survival_list.append(survival)
            if self._parents_survival_is_none(survival):
                cohabitation_list.append(no_cohab_label)
            else:
                labels_c = list(dist_c.keys())
                probs_c = list(dist_c.values())
                if not labels_c or sum(probs_c) <= 0:
                    cohabitation_list.append(no_cohab_label)
                else:
                    probs_c = [p / sum(probs_c) for p in probs_c]
                    cohabitation_list.append(str(np.random.choice(labels_c, p=probs_c)))

        out[column_names[0]] = survival_list
        out[column_names[1]] = cohabitation_list
        return out, True

    # -----------------------------
    # 부모님 생활비 주 제공자: 단일 컬럼 (C2_NM = 주제공자별 라벨, DT = 비율)
    # 허용 문자: 장남 또는 맏며느리, 아들 또는 며느리, 딸 또는 사위, 모든 자녀, 부모님 스스로 해결,
    #            정부 또는 사회단체, 가족과 정부, 사회단체, 기타 — KOSIS C2_NM 그대로 사용
    # -----------------------------
    PARENTS_EXPENSE_PROVIDER_EXCLUDE = ("계", "합계", "소계", "비율", "Total")

    def parse_parents_expense_provider_kosis(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        KOSIS '부모님 생활비 주 제공자' JSON에서 C1_NM='전체' 기준 C2_NM(주제공자별) 비율 분포 추출.
        반환: { 라벨: 비율(0~1) } (예: 장남 또는 맏며느리, 부모님 스스로 해결, ...).
        """
        dist: Dict[str, float] = {}
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            c1_nm = str(r.get("C1_NM") or "").strip()
            if c1_nm and c1_nm != "전체":
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            lab = str(r.get("C2_NM") or "").strip()
            if not lab or lab in self.PARENTS_EXPENSE_PROVIDER_EXCLUDE:
                continue
            dist[lab] = dist.get(lab, 0.0) + v
        total = sum(dist.values())
        if total <= 0:
            return dist
        return {k: v / total for k, v in dist.items()}

    def assign_parents_expense_provider_column(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        column_name: str = "부모님 생활비 주 제공자",
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        '부모님 생활비 주 제공자' 단일 컬럼을 비율 분포로 샘플링해 채움.
        값: 장남 또는 맏며느리, 아들 또는 며느리, 딸 또는 사위, 모든 자녀, 부모님 스스로 해결,
            정부 또는 사회단체, 가족과 정부, 사회단체, 기타 (KOSIS C2_NM 문자열 그대로).
        """
        out = pop_df.copy()
        n = len(out)
        if not kosis_data:
            out[column_name] = ""
            return out, False
        dist = self.parse_parents_expense_provider_kosis(kosis_data)
        if not dist:
            out[column_name] = ""
            return out, False
        labels = list(dist.keys())
        probs = list(dist.values())
        if sum(probs) <= 0:
            probs = [1.0 / len(labels)] * len(labels)
        else:
            probs = [p / sum(probs) for p in probs]
        if seed is not None:
            np.random.seed(seed)
        out[column_name] = [str(np.random.choice(labels, p=probs)) for _ in range(n)]
        return out, True

    # -----------------------------
    # 현재 거주주택 만족도: 3열 (주택 만족도, 기반시설 만족도, 주차장 만족도)
    # 값: 매우 불만족, 약간 불만족, 보통, 약간 만족, 매우 만족
    # -----------------------------
    SATISFACTION_LABELS = ("매우 불만족", "약간 불만족", "보통", "약간 만족", "매우 만족")
    SATISFACTION_LABEL_MAP = {
        "매우불만족": "매우 불만족", "약간불만족": "약간 불만족",
        "매우만족": "매우 만족", "약간만족": "약간 만족",
        "매우 불만족": "매우 불만족", "약간 불만족": "약간 불만족",
        "보통": "보통", "약간 만족": "약간 만족", "매우 만족": "매우 만족",
    }
    HOUSING_SATISFACTION_COLUMN_KEYS = (
        "주택",           # 0: 현재 거주 주택 만족도
        "기반시설",       # 1: 현재 상하수도, 도시가스 도로 등 기반시설 만족도
        "주차",           # 2: 주거지역내 주차장 이용 만족도
    )

    def _normalize_satisfaction_label(self, lab: str) -> Optional[str]:
        s = str(lab).strip().replace(" ", "")
        for k, v in self.SATISFACTION_LABEL_MAP.items():
            if k.replace(" ", "") == s:
                return v
        if lab.strip() in self.SATISFACTION_LABELS:
            return lab.strip()
        return None

    def parse_housing_satisfaction_kosis(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        KOSIS '현재 거주주택 만족도' 형 JSON에서 3개 항목별 만족도 비율 추출.
        C1_NM(항목)별로 C2_NM(만족도 수준) 비율 수집. 항목은 주택/기반시설/주차 키워드로 구분.
        반환: (dist_col0, dist_col1, dist_col2) 각각 {라벨: 비율}, 라벨은 SATISFACTION_LABELS.
        """
        from collections import defaultdict
        key_to_dist: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            c1 = str(r.get("C1_NM") or "").strip()
            c2 = str(r.get("C2_NM") or "").strip()
            if not c2 or c2 in ("계", "합계", "소계", "비율"):
                continue
            norm = self._normalize_satisfaction_label(c2)
            if not norm:
                continue
            # 항목 구분: C1_NM에 키워드 포함 여부로 매핑
            if "주차" in c1 or "주차장" in c1:
                key = "주차"
            elif "기반시설" in c1 or "상하수도" in c1 or "도시가스" in c1 or "도로" in c1:
                key = "기반시설"
            elif "주택" in c1 or not c1 or c1 == "전체":
                key = "주택"
            else:
                key = "주택"
            key_to_dist[key][norm] = key_to_dist[key].get(norm, 0.0) + v
        def to_probs(d: Dict[str, float]) -> Dict[str, float]:
            total = sum(d.values())
            if total <= 0:
                return d
            return {k: v / total for k, v in d.items()}
        dist0 = to_probs(dict(key_to_dist.get("주택", {})))
        dist1 = to_probs(dict(key_to_dist.get("기반시설", {})))
        dist2 = to_probs(dict(key_to_dist.get("주차", {})))
        if not dist0 and not dist1 and not dist2:
            flat: Dict[str, float] = defaultdict(float)
            for r in kosis_data:
                if not isinstance(r, dict):
                    continue
                raw = r.get("DT")
                if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                    continue
                try:
                    v = float(str(raw).replace(",", "").strip())
                except Exception:
                    continue
                lab = str(r.get("C2_NM") or "").strip()
                norm = self._normalize_satisfaction_label(lab)
                if norm:
                    flat[norm] += v
            single = to_probs(dict(flat))
            if single:
                dist0 = dist1 = dist2 = single
        return (dist0, dist1, dist2)

    def assign_housing_satisfaction_columns(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        column_names: Tuple[str, str, str] = (
            "현재 거주 주택 만족도",
            "현재 상하수도, 도시가스 도로 등 기반시설 만족도",
            "주거지역내 주차장이용 만족도",
        ),
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        """3개 만족도 컬럼을 비율로 샘플링해 채움. 값은 SATISFACTION_LABELS 문자열."""
        out = pop_df.copy()
        n = len(out)
        if not kosis_data:
            for c in column_names:
                out[c] = ""
            return out, False
        d0, d1, d2 = self.parse_housing_satisfaction_kosis(kosis_data)
        if not d0 and not d1 and not d2:
            for c in column_names:
                out[c] = ""
            return out, False
        labels = list(self.SATISFACTION_LABELS)
        if seed is not None:
            np.random.seed(seed)
        def sample_dist(d: Dict[str, float]) -> List[str]:
            if not d:
                return [labels[2]] * n
            lab_list = list(d.keys())
            probs = [d.get(l, 0.0) for l in lab_list]
            s = sum(probs)
            if s <= 0:
                probs = [1.0 / len(lab_list)] * len(lab_list)
            else:
                probs = [p / s for p in probs]
            return [str(np.random.choice(lab_list, p=probs)) for _ in range(n)]
        out[column_names[0]] = sample_dist(d0) if d0 else sample_dist(d1 or d2)
        out[column_names[1]] = sample_dist(d1) if d1 else sample_dist(d0 or d2)
        out[column_names[2]] = sample_dist(d2) if d2 else sample_dist(d0 or d1)
        return out, True

    # -----------------------------
    # 배우자의 경제활동 상태: 1열, 유/무
    # -----------------------------
    SPOUSE_ECONOMIC_LABELS = ("유", "무")

    # KOSIS '배우자의 경제활동 상태' C2_NM(경제활동상태별): "하였다"=유, "하지 않았다"=무
    SPOUSE_ECONOMIC_C2_MAP = {"하였다": "유", "하지 않았다": "무"}

    def parse_spouse_economic_kosis(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        KOSIS '배우자의 경제활동 상태' 형 JSON에서 유/무 비율 추출.
        C1_NM='전체' 행만 사용. C2_NM '하였다'→유, '하지 않았다'→무 (실제 API 라벨 기준).
        """
        dist: Dict[str, float] = {}
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            c1 = str(r.get("C1_NM") or "").strip()
            if c1 and c1 != "전체":
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            lab = str(r.get("C2_NM") or "").strip()
            if not lab or lab in ("계", "합계", "소계", "비율"):
                continue
            # KOSIS 실제 라벨: "하였다"(경제활동 함)=유, "하지 않았다"=무
            key = self.SPOUSE_ECONOMIC_C2_MAP.get(lab)
            if key is None:
                if "유" in lab or "있음" in lab or "하였다" in lab or lab in ("유", "있음", "경제활동"):
                    key = "유"
                else:
                    key = "무"
            dist[key] = dist.get(key, 0.0) + v
        total = sum(dist.values())
        if total <= 0:
            return dist
        return {k: v / total for k, v in dist.items()}

    def assign_spouse_economic_column(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        column_name: str = "배우자의 경제활동 상태",
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        """'배우자의 경제활동 상태' 단일 컬럼, 값: 유/무."""
        out = pop_df.copy()
        n = len(out)
        if not kosis_data:
            out[column_name] = ""
            return out, False
        dist = self.parse_spouse_economic_kosis(kosis_data)
        if not dist:
            out[column_name] = "무"
            return out, False
        labels = list(dist.keys())
        probs = list(dist.values())
        if sum(probs) <= 0:
            probs = [1.0 / len(labels)] * len(labels)
        else:
            probs = [p / sum(probs) for p in probs]
        if seed is not None:
            np.random.seed(seed)
        out[column_name] = [str(np.random.choice(labels, p=probs)) for _ in range(n)]
        return out, True

    # -----------------------------
    # 종사상 지위: 1열 (상용근로자, 임시근로자, 일용근로자, 고용원이 있는 자영업자, 고용원이 없는 자영업자, 무급가족 종사자)
    # -----------------------------
    EMPLOYMENT_STATUS_LABELS = (
        "상용근로자", "임시근로자", "일용근로자",
        "고용원이 있는 자영업자", "고용원이 없는 자영업자", "무급가족 종사자",
    )
    EMPLOYMENT_STATUS_NORMALIZE = {
        "상용근로자": "상용근로자", "임시근로자": "임시근로자", "일용근로자": "일용근로자",
        "고용원이 있는 자영업자": "고용원이 있는 자영업자",
        "고용원이 없는 자영업자": "고용원이 없는 자영업자",
        "무급가족종사자": "무급가족 종사자", "무급가족 종사자": "무급가족 종사자",
    }

    def parse_employment_status_kosis(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """KOSIS '종사상 지위' JSON에서 C2_NM(종사상 지위) 비율 추출. C1_NM='전체' 우선."""
        from collections import defaultdict
        dist: Dict[str, float] = defaultdict(float)
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            lab = str(r.get("C2_NM") or r.get("C1_NM") or "").strip()
            if not lab or lab in ("계", "합계", "소계", "비율", "Total"):
                continue
            norm = lab
            for k, canon in self.EMPLOYMENT_STATUS_NORMALIZE.items():
                if k in lab or lab in k:
                    norm = canon
                    break
            if norm not in self.EMPLOYMENT_STATUS_LABELS:
                if "상용" in lab:
                    norm = "상용근로자"
                elif "임시" in lab:
                    norm = "임시근로자"
                elif "일용" in lab:
                    norm = "일용근로자"
                elif "고용원이 있는" in lab or "고용원 있는" in lab:
                    norm = "고용원이 있는 자영업자"
                elif "고용원이 없는" in lab or "고용원 없는" in lab:
                    norm = "고용원이 없는 자영업자"
                elif "무급" in lab or "가족" in lab:
                    norm = "무급가족 종사자"
                else:
                    norm = lab
            dist[norm] = dist.get(norm, 0.0) + v
        total = sum(dist.values())
        if total <= 0:
            return dict(dist)
        return {k: v / total for k, v in dist.items()}

    def assign_employment_status_column(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        column_name: str = "종사상 지위",
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        """'종사상 지위' 단일 컬럼. 값: 상용근로자, 임시근로자, ... (EMPLOYMENT_STATUS_LABELS)."""
        out = pop_df.copy()
        n = len(out)
        if not kosis_data:
            out[column_name] = ""
            return out, False
        dist = self.parse_employment_status_kosis(kosis_data)
        if not dist:
            out[column_name] = ""
            return out, False
        labels = list(dist.keys())
        probs = list(dist.values())
        if sum(probs) <= 0:
            probs = [1.0 / len(labels)] * len(labels)
        else:
            probs = [p / sum(probs) for p in probs]
        if seed is not None:
            np.random.seed(seed)
        out[column_name] = [str(np.random.choice(labels, p=probs)) for _ in range(n)]
        return out, True

    # -----------------------------
    # 직장명(산업 대분류): 1열
    # -----------------------------
    INDUSTRY_MAJOR_LABELS = (
        "농업, 임업 및 어업",
        "광업 및 제조업",
        "공급업/원료재생업/건설업",
        "도매업, 숙박 및 음식점업",
        "공공행정, 국방 및 사회보장행정",
        "교육서비스업",
        "기타서비스업",
    )

    def _normalize_industry_major(self, lab: str) -> str:
        s = lab.strip()
        for candidate in self.INDUSTRY_MAJOR_LABELS:
            if candidate in s or s in candidate:
                return candidate
        if "농업" in s or "임업" in s or "어업" in s:
            return "농업, 임업 및 어업"
        if "광업" in s or "제조업" in s:
            return "광업 및 제조업"
        if "공급" in s or "원료재생" in s or "건설" in s:
            return "공급업/원료재생업/건설업"
        if "도매" in s or "숙박" in s or "음식점" in s:
            return "도매업, 숙박 및 음식점업"
        if "공공" in s or "국방" in s or "사회보장" in s:
            return "공공행정, 국방 및 사회보장행정"
        if "교육" in s:
            return "교육서비스업"
        if "기타" in s or "서비스" in s:
            return "기타서비스업"
        return s

    def parse_industry_major_kosis(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """KOSIS '직장명(산업 대분류)' JSON에서 C1_NM='전체' 기준 C2_NM 비율 추출."""
        dist: Dict[str, float] = {}
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            c1 = str(r.get("C1_NM") or "").strip()
            if c1 and c1 != "전체":
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            lab = str(r.get("C2_NM") or "").strip()
            if not lab or lab in ("계", "합계", "소계", "비율"):
                continue
            norm = self._normalize_industry_major(lab)
            key = norm if norm in self.INDUSTRY_MAJOR_LABELS else lab
            dist[key] = dist.get(key, 0.0) + v
        total = sum(dist.values())
        if total <= 0:
            return dist
        return {k: v / total for k, v in dist.items()}

    def assign_industry_major_column(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        column_name: str = "직장명(산업 대분류)",
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        out = pop_df.copy()
        n = len(out)
        if not kosis_data:
            out[column_name] = ""
            return out, False
        dist = self.parse_industry_major_kosis(kosis_data)
        if not dist:
            out[column_name] = ""
            return out, False
        labels = list(dist.keys())
        probs = list(dist.values())
        if sum(probs) <= 0:
            probs = [1.0 / len(labels)] * len(labels)
        else:
            probs = [p / sum(probs) for p in probs]
        if seed is not None:
            np.random.seed(seed)
        out[column_name] = [str(np.random.choice(labels, p=probs)) for _ in range(n)]
        return out, True

    # -----------------------------
    # 하는 일의 종류(직업 종분류): 1열
    # -----------------------------
    JOB_CLASS_LABELS = (
        "관리자",
        "전문가 및 관련 종사자",
        "사무 종사자",
        "서비스 종사자/판매종사자",
        "농업,임업 및 어업 숙련 종사자",
        "기능원 및 관련 기능종사자/장치, 기계조작 및 조립종사자",
        "단순노무 종사자",
        "군인",
    )

    def _normalize_job_class(self, lab: str) -> str:
        s = lab.strip()
        for candidate in self.JOB_CLASS_LABELS:
            if candidate in s or s in candidate:
                return candidate
        if "관리자" in s:
            return "관리자"
        if "전문가" in s:
            return "전문가 및 관련 종사자"
        if "사무" in s:
            return "사무 종사자"
        if "서비스" in s or "판매" in s:
            return "서비스 종사자/판매종사자"
        if "농업" in s or "임업" in s or "어업" in s:
            return "농업,임업 및 어업 숙련 종사자"
        if "기능원" in s or "기계조작" in s or "조립" in s or "장치" in s:
            return "기능원 및 관련 기능종사자/장치, 기계조작 및 조립종사자"
        if "단순노무" in s:
            return "단순노무 종사자"
        if "군인" in s:
            return "군인"
        return s

    def parse_job_class_kosis(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """KOSIS '하는 일의 종류(직업 종분류)' JSON에서 C2_NM 비율 추출."""
        dist: Dict[str, float] = {}
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            c1 = str(r.get("C1_NM") or "").strip()
            if c1 and c1 != "전체":
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            lab = str(r.get("C2_NM") or "").strip()
            if not lab or lab in ("계", "합계", "소계", "비율"):
                continue
            norm = self._normalize_job_class(lab)
            dist[norm] = dist.get(norm, 0.0) + v
        total = sum(dist.values())
        if total <= 0:
            return dist
        return {k: v / total for k, v in dist.items()}

    def assign_job_class_column(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        column_name: str = "하는 일의 종류(직업 종분류)",
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        out = pop_df.copy()
        n = len(out)
        if not kosis_data:
            out[column_name] = ""
            return out, False
        dist = self.parse_job_class_kosis(kosis_data)
        if not dist:
            out[column_name] = ""
            return out, False
        labels = list(dist.keys())
        probs = list(dist.values())
        if sum(probs) <= 0:
            probs = [1.0 / len(labels)] * len(labels)
        else:
            probs = [p / sum(probs) for p in probs]
        if seed is not None:
            np.random.seed(seed)
        out[column_name] = [str(np.random.choice(labels, p=probs)) for _ in range(n)]
        return out, True

    # -----------------------------
    # 취업자 근로여건 만족도: 5열 (하는일, 임금/가구소득, 근로시간, 근무환경, 전반적인 만족도)
    # 값: 매우불만족, 약간불만족, 보통, 약간만족, 매우만족
    # -----------------------------
    WORK_SATISFACTION_LABELS = ("매우불만족", "약간불만족", "보통", "약간만족", "매우만족")
    WORK_SATISFACTION_COLUMN_KEYS = ("하는일", "임금", "근로시간", "근무환경", "전반")

    def _normalize_work_satisfaction_label(self, lab: str) -> Optional[str]:
        s = str(lab).strip().replace(" ", "")
        for L in self.WORK_SATISFACTION_LABELS:
            if L.replace(" ", "") == s or s in L.replace(" ", ""):
                return L
        if "매우" in lab and "불만" in lab:
            return "매우불만족"
        if "약간" in lab and "불만" in lab:
            return "약간불만족"
        if "보통" in lab:
            return "보통"
        if "약간" in lab and "만족" in lab:
            return "약간만족"
        if "매우" in lab and "만족" in lab:
            return "매우만족"
        return None

    def parse_work_satisfaction_kosis(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], ...]:
        """
        5개 항목별 근로여건 만족도 분포.
        KOSIS 구조: C2_NM=항목(하는일, 임금/가구소득, 근로시간, 근무환경, 전반), C3_NM=만족도(매우불만족 등), DT=비율.
        """
        from collections import defaultdict
        key_to_dist: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            c2 = str(r.get("C2_NM") or "").strip()
            c3 = str(r.get("C3_NM") or "").strip()
            lab = c3 if c3 else c2
            if not lab or lab in ("계", "합계", "소계", "비율"):
                continue
            norm = self._normalize_work_satisfaction_label(lab)
            if not norm:
                continue
            item_key = c2
            if "하는일" in item_key or "하는 일" in item_key:
                key = "하는일"
            elif "임금" in item_key or "가구소득" in item_key:
                key = "임금"
            elif "근로시간" in item_key or "근로 시간" in item_key:
                key = "근로시간"
            elif "근무환경" in item_key or "근무 환경" in item_key:
                key = "근무환경"
            elif "전반" in item_key:
                key = "전반"
            else:
                key = "하는일"
            key_to_dist[key][norm] = key_to_dist[key].get(norm, 0.0) + v
        def to_probs(d: Dict[str, float]) -> Dict[str, float]:
            total = sum(d.values())
            if total <= 0:
                return d
            return {k: v / total for k, v in d.items()}
        keys_order = ("하는일", "임금", "근로시간", "근무환경", "전반")
        out = tuple(to_probs(dict(key_to_dist.get(k, {}))) for k in keys_order)
        if not any(out):
            flat: Dict[str, float] = defaultdict(float)
            for r in kosis_data:
                if not isinstance(r, dict):
                    continue
                raw = r.get("DT")
                if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                    continue
                try:
                    v = float(str(raw).replace(",", "").strip())
                except Exception:
                    continue
                lab = str(r.get("C3_NM") or r.get("C2_NM") or "").strip()
                norm = self._normalize_work_satisfaction_label(lab)
                if norm:
                    flat[norm] += v
            single = to_probs(dict(flat))
            if single:
                out = (single, single, single, single, single)
        return out

    def assign_work_satisfaction_columns(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        column_names: Tuple[str, str, str, str, str] = (
            "하는일",
            "임금/가구소득",
            "근로시간",
            "근무환경",
            "전반적인 만족도",
        ),
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        out = pop_df.copy()
        n = len(out)
        if not kosis_data:
            for c in column_names:
                out[c] = ""
            return out, False
        d1, d2, d3, d4, d5 = self.parse_work_satisfaction_kosis(kosis_data)
        dists = [d1, d2, d3, d4, d5]
        if not any(dists):
            for c in column_names:
                out[c] = ""
            return out, False
        if seed is not None:
            np.random.seed(seed)
        labels = list(self.WORK_SATISFACTION_LABELS)
        for i, col in enumerate(column_names):
            d = dists[i] if i < len(dists) else (dists[0] or dists[1])
            if not d:
                out[col] = "보통"
                continue
            lab_list = list(d.keys())
            probs = list(d.values())
            if sum(probs) <= 0:
                probs = [1.0 / len(lab_list)] * len(lab_list)
            else:
                probs = [p / sum(probs) for p in probs]
            out[col] = [str(np.random.choice(lab_list, p=probs)) for _ in range(n)]
        return out, True

    # -----------------------------
    # 반려동물 양육비용: 1열, 원 단위 숫자
    # KOSIS 형식 A: C1_NM=지역(전체, 포항시, ...), DT=월평균 원 → 지역별 평균 사용
    # KOSIS 형식 B: C2_NM=구간, DT=비율 → 구간 샘플링 (기존)
    # -----------------------------
    def parse_pet_cost_kosis(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[float], List[Tuple[int, int]]]:
        """구간 형식: 라벨, 비율, (원 최소, 최대). 비어있으면 ([], [], [])."""
        from collections import defaultdict
        dist: Dict[str, float] = defaultdict(float)
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            lab = str(r.get("C2_NM") or "").strip()
            if not lab or lab in ("계", "합계", "소계", "비율"):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            dist[lab] = dist.get(lab, 0.0) + v
        total = sum(dist.values())
        if total <= 0:
            return [], [], []
        probs = [dist[k] / total for k in dist]
        labels = list(dist.keys())
        ranges_won: List[Tuple[int, int]] = [self._parse_pet_cost_range_won(lab) for lab in labels]
        return labels, probs, ranges_won

    def parse_pet_cost_kosis_regional(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """KOSIS '반려동물 양육비용' 월평균 원 형식: C1_NM=지역, DT=원 → { 지역명: 평균원 }."""
        out: Dict[str, float] = {}
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            c1 = str(r.get("C1_NM") or "").strip()
            if not c1 or c1 in ("계", "합계", "소계", "비율"):
                continue
            out[c1] = v
        return out

    def _parse_pet_cost_range_won(self, lab: str) -> Tuple[int, int]:
        """구간 문자열 → (최소원, 최대원)."""
        import re
        s = str(lab).strip()
        nums = [int(x) for x in re.findall(r"\d+", s)]
        if not nums:
            return (0, 0)
        if "만원" in s or "만 원" in s:
            if len(nums) == 1:
                if "미만" in s or "이하" in s:
                    return (0, nums[0] * 10000 - 1)
                if "이상" in s:
                    return (nums[0] * 10000, nums[0] * 10000 + 100000)
                return (nums[0] * 10000, nums[0] * 10000)
            lo, hi = nums[0] * 10000, nums[-1] * 10000
            if "미만" in s and len(nums) == 1:
                return (0, lo - 1)
            return (lo, hi)
        return (nums[0], nums[-1] if len(nums) > 1 else nums[0])

    def assign_pet_cost_column(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        column_name: str = "반려동물 양육비용",
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        1열, 원 단위 정수.
        KOSIS가 지역별 월평균 원(C1_NM, DT)이면 거주지역 매칭 후 평균 주변 분산으로 대입.
        반려동물유무=아니오 이면 0. 구간 비율 형식이면 기존처럼 구간 샘플링.
        """
        out = pop_df.copy()
        n = len(out)
        if not kosis_data:
            out[column_name] = 0
            return out, False
        regional = self.parse_pet_cost_kosis_regional(kosis_data)
        if regional:
            overall = regional.get("전체")
            if overall is None and regional:
                overall = next(iter(regional.values()))
            if seed is not None:
                np.random.seed(seed)
            region_col = "거주지역"
            if region_col not in out.columns:
                region_col = next((c for c in out.columns if "거주" in str(c) or "지역" in str(c)), None)
            pet_col = "반려동물유무"
            has_pet = pet_col in out.columns
            values: List[int] = []
            for _, row in out.iterrows():
                if has_pet and str(row.get(pet_col, "")).strip() == "아니오":
                    values.append(0)
                    continue
                region = str(row.get(region_col, "")).strip() if region_col else ""
                mean_won = regional.get(region) or regional.get("전체") or overall or 0
                if mean_won <= 0:
                    values.append(0)
                    continue
                std = max(1, mean_won * 0.35)
                val = int(round(np.random.normal(mean_won, std)))
                if val < 0:
                    val = 0
                values.append(val)
            out[column_name] = values
            return out, True
        labels, probs, ranges_won = self.parse_pet_cost_kosis(kosis_data)
        if not labels or not probs or not ranges_won:
            out[column_name] = 0
            return out, False
        probs = [p / sum(probs) for p in probs]
        if seed is not None:
            np.random.seed(seed)
        pet_col = "반려동물유무"
        has_pet = pet_col in out.columns
        values = []
        for i in range(n):
            if has_pet and str(out[pet_col].iloc[i]).strip() == "아니오":
                values.append(0)
                continue
            idx = int(np.random.choice(len(labels), p=probs))
            lo, hi = ranges_won[idx]
            val = lo if lo >= hi else int(np.random.randint(lo, hi + 1))
            values.append(val)
        out[column_name] = values
        return out, True

    # -----------------------------
    # 소득 및 소비생활 만족도: 3열 (소득 여부 예/아니오, 소득 만족도, 소비생활만족도)
    # KOSIS: C2 B01*=여부(예/아니오), B02*=소득만족도, B03*=소비생활만족도(5단계)
    # -----------------------------
    INCOME_SATISFACTION_LABELS = ("매우불만족", "약간불만족", "보통", "약간만족", "매우만족")

    def parse_income_consumption_satisfaction_kosis(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """C2 prefix B01=소득여부, B02=소득만족도, B03=소비생활만족도. 각각 {라벨: 비율}."""
        from collections import defaultdict
        d1: Dict[str, float] = defaultdict(float)
        d2: Dict[str, float] = defaultdict(float)
        d3: Dict[str, float] = defaultdict(float)
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            c1 = str(r.get("C1_NM") or "").strip()
            if c1 and c1 != "전체":
                continue
            c2_code = str(r.get("C2") or "").strip().upper()
            lab = str(r.get("C2_NM") or "").strip()
            if not lab or lab in ("계", "합계", "소계", "비율"):
                continue
            if c2_code.startswith("B01"):
                d1[lab] = d1.get(lab, 0.0) + v
            elif c2_code.startswith("B02"):
                norm = self._normalize_work_satisfaction_label(lab)
                if norm:
                    d2[norm] = d2.get(norm, 0.0) + v
            elif c2_code.startswith("B03"):
                norm = self._normalize_work_satisfaction_label(lab)
                if norm:
                    d3[norm] = d3.get(norm, 0.0) + v
        def to_probs(d: Dict[str, float]) -> Dict[str, float]:
            total = sum(d.values())
            if total <= 0:
                return d
            return {k: v / total for k, v in d.items()}
        return to_probs(dict(d1)), to_probs(dict(d2)), to_probs(dict(d3))

    def assign_income_consumption_satisfaction_columns(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        column_names: Tuple[str, str, str] = ("소득 여부", "소득 만족도", "소비생활만족도"),
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        out = pop_df.copy()
        n = len(out)
        if not kosis_data:
            for c in column_names:
                out[c] = ""
            return out, False
        dist_yesno, dist_income_sat, dist_consume_sat = self.parse_income_consumption_satisfaction_kosis(kosis_data)
        if not dist_yesno:
            for c in column_names:
                out[c] = ""
            return out, False
        if seed is not None:
            np.random.seed(seed)
        yesno_labels = list(dist_yesno.keys())
        yesno_probs = list(dist_yesno.values())
        if sum(yesno_probs) <= 0:
            yesno_probs = [1.0 / len(yesno_labels)] * len(yesno_labels)
        else:
            yesno_probs = [p / sum(yesno_probs) for p in yesno_probs]
        out[column_names[0]] = [str(np.random.choice(yesno_labels, p=yesno_probs)) for _ in range(n)]
        for i, col in enumerate(column_names[1:], 1):
            dist = (dist_income_sat, dist_consume_sat)[i - 1]
            if not dist:
                dist = dist_income_sat or dist_consume_sat
            if dist:
                labels = list(dist.keys())
                probs = list(dist.values())
                if sum(probs) <= 0:
                    probs = [1.0 / len(labels)] * len(labels)
                else:
                    probs = [p / sum(probs) for p in probs]
                out[col] = [str(np.random.choice(labels, p=probs)) for _ in range(n)]
            else:
                out[col] = ["보통"] * n
        return out, True

    # -----------------------------
    # 월평균 공교육 및 사교육비: 2열 (공교육비, 사교육비) 만원 단위
    # KOSIS: C2_NM 공교육비/사교육비, DT=만원 (소계 활용)
    # -----------------------------
    def parse_education_cost_kosis(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Tuple[Optional[float], Optional[float]]:
        """C1_NM=전체, C2_NM='공교육비'/'사교육비'인 상위 소계 DT(만원) 반환. (공교육비_만원, 사교육비_만원)."""
        gong, sa = None, None
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            c1 = str(r.get("C1_NM") or "").strip()
            if c1 and c1 != "전체":
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            lab = str(r.get("C2_NM") or "").strip()
            if lab == "공교육비":
                gong = v
            elif lab == "사교육비":
                sa = v
        return (gong, sa)

    def assign_education_cost_columns(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        column_names: Tuple[str, str] = ("공교육비", "사교육비"),
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        """2열 만원 단위. 소계 주변 분산(약 0.5~1.5배)으로 정수 만원 대입."""
        out = pop_df.copy()
        n = len(out)
        if not kosis_data:
            for c in column_names:
                out[c] = 0
            return out, False
        gong_mean, sa_mean = self.parse_education_cost_kosis(kosis_data)
        if gong_mean is None:
            gong_mean = 0.0
        if sa_mean is None:
            sa_mean = 0.0
        if gong_mean <= 0 and sa_mean <= 0:
            for c in column_names:
                out[c] = 0
            return out, False
        if seed is not None:
            np.random.seed(seed)
        gong_vals = [max(0, int(round(gong_mean * (0.5 + np.random.random())))) for _ in range(n)]
        sa_vals = [max(0, int(round(sa_mean * (0.5 + np.random.random())))) for _ in range(n)]
        out[column_names[0]] = gong_vals
        out[column_names[1]] = sa_vals
        return out, True

    # -----------------------------
    # 타지역 소비: 4열 (소비 경험 여부 있다/없다, 경북 외 주요 소비지역, 1순위 상품/서비스, 2순위 상품/서비스)
    # 1·2순위 중복 없음. 지역: 서울, 대구, 강원, 기타. 서비스: 식료품, 의류 및 잡화, ... 기타
    # -----------------------------
    OTHER_REGION_LABELS = ("서울", "대구", "강원", "기타")
    OTHER_SERVICE_LABELS = (
        "식료품", "의류 및 잡화", "기타 생활 용품", "외식 서비스", "문화/여가 서비스",
        "교육 서비스", "의료/보건 서비스", "미용 및 뷰티 서비스", "자동차 서비스", "기타",
    )

    def parse_other_region_consumption_kosis(
        self, kosis_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """B01*=여부(있다/없다), B02*=지역(서울,대구,강원,기타), B03* 또는 나머지=상품/서비스 비율."""
        from collections import defaultdict
        d_has: Dict[str, float] = defaultdict(float)
        d_region: Dict[str, float] = defaultdict(float)
        d_service: Dict[str, float] = defaultdict(float)
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            c1 = str(r.get("C1_NM") or "").strip()
            if c1 and c1 != "전체":
                continue
            c2_code = str(r.get("C2") or "").strip().upper()
            lab = str(r.get("C2_NM") or "").strip()
            if not lab or lab in ("계", "합계", "소계", "비율"):
                continue
            if c2_code.startswith("B01"):
                d_has[lab] = d_has.get(lab, 0.0) + v
            elif c2_code.startswith("B02"):
                key = lab
                if key not in self.OTHER_REGION_LABELS:
                    if "서울" in lab:
                        key = "서울"
                    elif "대구" in lab:
                        key = "대구"
                    elif "강원" in lab:
                        key = "강원"
                    else:
                        key = "기타"
                d_region[key] = d_region.get(key, 0.0) + v
            else:
                key = lab
                for cand in self.OTHER_SERVICE_LABELS:
                    if cand in lab or lab in cand:
                        key = cand
                        break
                if "식료" in lab:
                    key = "식료품"
                elif "의류" in lab or "잡화" in lab:
                    key = "의류 및 잡화"
                elif "생활 용품" in lab:
                    key = "기타 생활 용품"
                elif "외식" in lab:
                    key = "외식 서비스"
                elif "문화" in lab or "여가" in lab:
                    key = "문화/여가 서비스"
                elif "교육" in lab:
                    key = "교육 서비스"
                elif "의료" in lab or "보건" in lab:
                    key = "의료/보건 서비스"
                elif "미용" in lab or "뷰티" in lab:
                    key = "미용 및 뷰티 서비스"
                elif "자동차" in lab:
                    key = "자동차 서비스"
                else:
                    key = key if key in self.OTHER_SERVICE_LABELS else "기타"
                d_service[key] = d_service.get(key, 0.0) + v
        def to_probs(d: Dict[str, float]) -> Dict[str, float]:
            total = sum(d.values())
            if total <= 0:
                return d
            return {k: v / total for k, v in d.items()}
        return to_probs(dict(d_has)), to_probs(dict(d_region)), to_probs(dict(d_service))

    def assign_other_region_consumption_columns(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        column_names: Tuple[str, str, str, str] = (
            "소비 경험 여부",
            "경북 외 주요 소비지역",
            "경북 외 주요 소비 상품 및 서비스(1순위)",
            "경북 외 주요 소비 상품 및 서비스(2순위)",
        ),
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        out = pop_df.copy()
        n = len(out)
        if not kosis_data:
            for c in column_names:
                out[c] = ""
            return out, False
        dist_has, dist_region, dist_service = self.parse_other_region_consumption_kosis(kosis_data)
        if not dist_has:
            for c in column_names:
                out[c] = ""
            return out, False
        if seed is not None:
            np.random.seed(seed)
        has_labels = list(dist_has.keys())
        has_probs = [dist_has[k] for k in has_labels]
        if sum(has_probs) <= 0:
            has_probs = [1.0 / len(has_labels)] * len(has_labels)
        else:
            has_probs = [p / sum(has_probs) for p in has_probs]
        region_labels = list(dist_region.keys()) if dist_region else list(self.OTHER_REGION_LABELS)
        region_probs = list(dist_region.values()) if dist_region else [1.0 / len(region_labels)] * len(region_labels)
        if dist_region and sum(region_probs) > 0:
            region_probs = [p / sum(region_probs) for p in region_probs]
        service_labels = list(dist_service.keys()) if dist_service else list(self.OTHER_SERVICE_LABELS)
        service_probs = list(dist_service.values()) if dist_service else [1.0 / len(service_labels)] * len(service_labels)
        if dist_service and sum(service_probs) > 0:
            service_probs = [p / sum(service_probs) for p in service_probs]
        col_has, col_region, col_1st, col_2nd = column_names
        has_list: List[str] = []
        region_list: List[str] = []
        first_list: List[str] = []
        second_list: List[str] = []
        for _ in range(n):
            has_val = str(np.random.choice(has_labels, p=has_probs))
            has_list.append(has_val)
            if has_val.strip() in ("있다", "예"):
                region_list.append(str(np.random.choice(region_labels, p=region_probs)))
                idx1 = int(np.random.choice(len(service_labels), p=service_probs))
                first_list.append(service_labels[idx1])
                remaining = [j for j in range(len(service_labels)) if j != idx1]
                if not remaining:
                    second_list.append(service_labels[idx1])
                else:
                    rem_probs = [service_probs[j] for j in remaining]
                    rem_probs = [p / sum(rem_probs) for p in rem_probs]
                    idx2 = remaining[int(np.random.choice(len(remaining), p=rem_probs))]
                    second_list.append(service_labels[idx2])
            else:
                region_list.append("")
                first_list.append("")
                second_list.append("")
        out[col_has] = has_list
        out[col_region] = region_list
        out[col_1st] = first_list
        out[col_2nd] = second_list
        return out, True

    # -----------------------------
    # 프리셋 통계: C2(항목)·C3(만족도 등) 구조 공통 파싱 및 다열 대입
    # -----------------------------
    def _parse_c2_c3_distributions(
        self,
        kosis_data: List[Dict[str, Any]],
        *,
        c2_to_col_name: Optional[Dict[str, str]] = None,
        value_normalize: Optional[Dict[str, str]] = None,
        c1_all_only: bool = True,
        c2_fallback_map: Optional[Dict[str, List[str]]] = None,
        c2_code_to_col: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """C2_NM=컬럼 키, C3_NM=값. c2_fallback_map: C2_NM 부분일치 시 컬럼. c2_code_to_col: C2 코드(B01 등)로 컬럼 매핑."""
        from collections import defaultdict
        col_dists: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            if c1_all_only and str(r.get("C1_NM") or "").strip() not in ("", "전체"):
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            c2 = str(r.get("C2_NM") or "").strip()
            c3 = str(r.get("C3_NM") or "").strip()
            c2_code = str(r.get("C2") or "").strip().upper()
            if not c2 or c2 in ("계", "합계", "소계", "평균"):
                continue
            col_name = (c2_to_col_name or {}).get(c2)
            if col_name is None and c2_fallback_map:
                for target_col, substrings in c2_fallback_map.items():
                    if any(s in c2 for s in substrings):
                        col_name = target_col
                        break
            if col_name is None and c2_code_to_col and c2_code:
                col_name = c2_code_to_col.get(c2_code) or c2_code_to_col.get(c2_code[:3])
            if col_name is None:
                col_name = c2
            if value_normalize:
                c3 = value_normalize.get(c3, value_normalize.get(c3.replace(" ", ""), c3))
            if c3 and c3 not in ("계", "합계", "소계", "평균", "평균값"):
                col_dists[col_name][c3] = col_dists[col_name].get(c3, 0.0) + v
        out: Dict[str, Dict[str, float]] = {}
        for col, dist in col_dists.items():
            total = sum(dist.values())
            out[col] = {k: v / total for k, v in dist.items()} if total > 0 else {}
        return out

    def _parse_single_c2_distribution(
        self,
        kosis_data: List[Dict[str, Any]],
        *,
        value_normalize: Optional[Dict[str, str]] = None,
        c1_all_only: bool = True,
    ) -> Dict[str, float]:
        """단일 컬럼: C2_NM=값, DT=비중. 반환 { 값: 비율 }."""
        from collections import defaultdict
        dist: Dict[str, float] = defaultdict(float)
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            if c1_all_only and str(r.get("C1_NM") or "").strip() not in ("", "전체"):
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            lab = str(r.get("C2_NM") or "").strip()
            if not lab or lab in ("계", "합계", "소계", "평균"):
                continue
            if value_normalize:
                lab = value_normalize.get(lab, value_normalize.get(lab.replace(" ", ""), lab))
            dist[lab] = dist.get(lab, 0.0) + v
        total = sum(dist.values())
        return {k: v / total for k, v in dist.items()} if total > 0 else {}

    def _parse_numeric_by_c2(
        self,
        kosis_data: List[Dict[str, Any]],
        *,
        c2_to_col_name: Optional[Dict[str, str]] = None,
        c1_all_only: bool = True,
    ) -> Dict[str, float]:
        """C2_NM=컬럼, DT=수치(평균 등). 반환 { 컬럼명: 평균값 }."""
        from collections import defaultdict
        sums: Dict[str, List[float]] = defaultdict(list)
        for r in kosis_data:
            if not isinstance(r, dict):
                continue
            if c1_all_only and str(r.get("C1_NM") or "").strip() not in ("", "전체"):
                continue
            raw = r.get("DT")
            if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                continue
            try:
                v = float(str(raw).replace(",", "").strip())
            except Exception:
                continue
            c2 = str(r.get("C2_NM") or "").strip()
            if not c2 or c2 in ("계", "합계", "소계"):
                continue
            col = (c2_to_col_name or {}).get(c2, c2)
            sums[col].append(v)
        return {col: (sum(vals) / len(vals) if vals else 0.0) for col, vals in sums.items()}

    def assign_preset_stat_columns(
        self,
        pop_df: pd.DataFrame,
        kosis_data: List[Dict[str, Any]],
        *,
        stat_name: str,
        column_names: Tuple[str, ...],
        seed: Optional[int] = None,
        preset: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        프리셋에 정의된 통계: 다열 카테고리 또는 수치 대입.
        preset이 None이면 stat_name으로 PRESET_STAT_CONFIG에서 조회.
        """
        out = pop_df.copy()
        n = len(out)
        if not kosis_data:
            for c in column_names:
                out[c] = "" if preset and preset.get("numeric") else None
            return out, False

        # 긴 키부터 매칭해 '지역소속감'이 '지역의 사회복지...'보다 먼저 잡히지 않도록 함
        cfg = preset
        if not cfg and stat_name:
            candidates = [(k, v) for k, v in PRESET_STAT_CONFIG.items() if k in stat_name]
            cfg = max(candidates, key=lambda x: len(x[0]))[1] if candidates else None
        if not cfg:
            for c in column_names:
                out[c] = ""
            return out, False

        if seed is not None:
            np.random.seed(seed)
        mode = cfg.get("mode", "c2_c3")
        value_norm = cfg.get("value_normalize") or {}
        c2_to_col = cfg.get("c2_to_col") or {}
        cols_ordered = list(column_names)

        if mode == "single_c2":
            dist = self._parse_single_c2_distribution(kosis_data, value_normalize=value_norm, c1_all_only=cfg.get("c1_all_only", True))
            if not dist:
                out[cols_ordered[0]] = [""] * n
                return out, False
            labels, probs = list(dist.keys()), list(dist.values())
            if sum(probs) <= 0:
                probs = [1.0 / len(labels)] * len(labels)
            else:
                probs = [p / sum(probs) for p in probs]
            out[cols_ordered[0]] = [str(np.random.choice(labels, p=probs)) for _ in range(n)]
            return out, True

        if mode == "c2_c3":
            col_dists = self._parse_c2_c3_distributions(
                kosis_data,
                c2_to_col_name=c2_to_col,
                value_normalize=value_norm,
                c1_all_only=cfg.get("c1_all_only", True),
                c2_fallback_map=cfg.get("c2_fallback_map"),
                c2_code_to_col=cfg.get("c2_code_to_col"),
            )
            fallback = cfg.get("fallback_value")
            for col_name in cols_ordered:
                dist = col_dists.get(col_name, {})
                if not dist:
                    out[col_name] = [fallback if fallback is not None else ""] * n
                    continue
                labels, probs = list(dist.keys()), list(dist.values())
                if sum(probs) <= 0:
                    probs = [1.0 / len(labels)] * len(labels)
                else:
                    probs = [p / sum(probs) for p in probs]
                out[col_name] = [str(np.random.choice(labels, p=probs)) for _ in range(n)]
            return out, True

        if mode == "numeric_c2":
            means = self._parse_numeric_by_c2(kosis_data, c2_to_col_name=c2_to_col, c1_all_only=cfg.get("c1_all_only", True))
            for col_name in cols_ordered:
                mu = means.get(col_name, 0.0)
                if mu <= 0:
                    mu = 5.0
                out[col_name] = [round(float(np.clip(mu + np.random.randn() * 0.5, 0, 10)), 2) for _ in range(n)]
            return out, True

        if mode == "c2_code_two":
            from collections import defaultdict
            p0, p1 = cfg.get("col0_prefix", "B01"), cfg.get("col1_prefix", "B02")
            d0: Dict[str, float] = defaultdict(float)
            d1: Dict[str, float] = defaultdict(float)
            for r in kosis_data:
                if not isinstance(r, dict):
                    continue
                if cfg.get("c1_all_only", True) and str(r.get("C1_NM") or "").strip() not in ("", "전체"):
                    continue
                raw = r.get("DT")
                if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                    continue
                try:
                    v = float(str(raw).replace(",", "").strip())
                except Exception:
                    continue
                c2_code = str(r.get("C2") or "").strip().upper()
                lab = str(r.get("C2_NM") or "").strip()
                if not lab or lab in ("계", "합계", "소계"):
                    continue
                lab = (value_norm or {}).get(lab, lab)
                if c2_code.startswith(p0):
                    d0[lab] = d0.get(lab, 0.0) + v
                elif c2_code.startswith(p1):
                    d1[lab] = d1.get(lab, 0.0) + v
            for idx, col_name in enumerate(cols_ordered[:2]):
                dist = (d0, d1)[idx]
                if not dist:
                    out[col_name] = [""] * n
                    continue
                total = sum(dist.values())
                probs = [v / total for v in dist.values()] if total > 0 else [1.0 / len(dist)] * len(dist)
                labels = list(dist.keys())
                out[col_name] = [str(np.random.choice(labels, p=probs)) for _ in range(n)]
            return out, True

        if mode == "c2_code_three":
            from collections import defaultdict
            p0, p1 = cfg.get("col0_prefix", "B01"), cfg.get("col1_prefix", "B02")
            p2 = cfg.get("col2_prefix")
            col2_num = cfg.get("col2_numeric", False)
            col0_is_satisfaction = cfg.get("col0_is_satisfaction", False)  # col0 데이터를 col1에도 사용 (여가활동 만족도)
            d0 = defaultdict(float)
            d1 = defaultdict(float)
            d2 = defaultdict(float) if not col2_num else []
            for r in kosis_data:
                if not isinstance(r, dict):
                    continue
                if cfg.get("c1_all_only", True) and str(r.get("C1_NM") or "").strip() not in ("", "전체"):
                    continue
                raw = r.get("DT")
                if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                    continue
                try:
                    v = float(str(raw).replace(",", "").strip())
                except Exception:
                    continue
                c2_code = str(r.get("C2") or "").strip().upper()
                lab = str(r.get("C2_NM") or "").strip()
                if c2_code.startswith(p0):
                    if lab and lab not in ("계", "합계", "소계"):
                        lab = (value_norm or {}).get(lab, lab)
                        d0[lab] = d0.get(lab, 0.0) + v
                        if col0_is_satisfaction and p0 == p1:  # col0과 col1이 같은 prefix면 col1에도 추가
                            d1[lab] = d1.get(lab, 0.0) + v
                elif c2_code.startswith(p1) and not (col0_is_satisfaction and p0 == p1):
                    if lab and lab not in ("계", "합계", "소계"):
                        lab = (value_norm or {}).get(lab, lab)
                        d1[lab] = d1.get(lab, 0.0) + v
                elif p2 and c2_code.startswith(p2):
                    if lab and lab not in ("계", "합계", "소계"):
                        lab = (value_norm or {}).get(lab, lab)
                        d2[lab] = d2.get(lab, 0.0) + v
                elif col2_num:
                    # 숫자형 col2: B03으로 시작하거나, C2_OBJ_NM에 "금액"이 있고 B01/B02가 아닌 경우만 처리
                    c2_obj_nm = str(r.get("C2_OBJ_NM") or "").strip()
                    if c2_code.startswith("B03"):
                        d2.append(v)
                    elif ("금액" in c2_obj_nm or "시간" in c2_obj_nm) and not c2_code.startswith(p0) and not c2_code.startswith(p1):
                        d2.append(v)
            # col0, col1
            for idx, col_name in enumerate(cols_ordered[:2]):
                dist = (d0, d1)[idx]
                if not dist:
                    out[col_name] = [""] * n
                    continue
                total = sum(dist.values())
                probs = [v / total for v in dist.values()] if total > 0 else [1.0 / len(dist)] * len(dist)
                labels = list(dist.keys())
                out[col_name] = [str(np.random.choice(labels, p=probs)) for _ in range(n)]
            # col2
            if len(cols_ordered) >= 3:
                if col2_num:
                    mean_val = (sum(d2) / len(d2)) if d2 else 0.0
                    # col0이 "기부 여부" 같은 경우, "없다"/"없음"이면 0으로 설정
                    col0_name = cols_ordered[0] if len(cols_ordered) > 0 else ""
                    col2_values = []
                    for i in range(n):
                        if col0_name and col0_name in out:
                            col0_val = str(out[col0_name][i] if i < len(out[col0_name]) else "").strip()
                            # "없다", "없음" 등이면 0
                            if col0_val in ("없다", "없음", "아니오", "No", "no"):
                                col2_values.append(0.0)
                            elif mean_val > 0:
                                # 금액 생성 (평균값 기준으로 변동)
                                col2_values.append(round(max(0, mean_val * (0.5 + np.random.rand())), 2))
                            else:
                                col2_values.append(0.0)
                        elif mean_val > 0:
                            col2_values.append(round(max(0, mean_val * (0.5 + np.random.rand())), 2))
                        else:
                            col2_values.append(0.0)
                    out[cols_ordered[2]] = col2_values
                elif d2:
                    total = sum(d2.values())
                    probs = [v / total for v in d2.values()] if total > 0 else [1.0 / len(d2)] * len(d2)
                    labels = list(d2.keys())
                    out[cols_ordered[2]] = [str(np.random.choice(labels, p=probs)) for _ in range(n)]
                else:
                    out[cols_ordered[2]] = [""] * n
            return out, True

        if mode == "itm_nm_c2":
            itm_to_col = cfg.get("itm_to_col") or {}
            from collections import defaultdict
            col_dists = defaultdict(lambda: defaultdict(float))
            for r in kosis_data:
                if not isinstance(r, dict):
                    continue
                if cfg.get("c1_all_only", True) and str(r.get("C1_NM") or "").strip() not in ("", "전체"):
                    continue
                raw = r.get("DT")
                if raw is None or raw == "" or str(raw).strip() in ("-", ""):
                    continue
                try:
                    v = float(str(raw).replace(",", "").strip())
                except Exception:
                    continue
                itm = str(r.get("ITM_NM") or "").strip()
                c2 = str(r.get("C2_NM") or "").strip()
                if not itm or not c2 or c2 in ("계", "합계", "소계"):
                    continue
                col_name = itm_to_col.get(itm, itm)
                lab = (value_norm or {}).get(c2, c2)
                col_dists[col_name][lab] = col_dists[col_name].get(lab, 0.0) + v
            for col_name in cols_ordered:
                dist = col_dists.get(col_name, {})
                if not dist:
                    out[col_name] = [""] * n
                    continue
                labels, probs = list(dist.keys()), list(dist.values())
                if sum(probs) <= 0:
                    probs = [1.0 / len(labels)] * len(labels)
                else:
                    probs = [p / sum(probs) for p in probs]
                out[col_name] = [str(np.random.choice(labels, p=probs)) for _ in range(n)]
            return out, True

        for c in cols_ordered:
            out[c] = [""] * n
        return out, False

    def get_preset_target_distributions(
        self,
        stat_name: str,
        kosis_data: List[Dict[str, Any]],
        column_names: List[str],
    ) -> List[Tuple[str, List[Any], List[float], Any]]:
        """
        프리셋 통계에 대한 검증용 목표 분포 반환.
        반환: [(컬럼명, labels, target_p, None), ...] — 컬럼별 만족도(또는 값) 수준 분포.
        """
        out: List[Tuple[str, List[Any], List[float], Any]] = []
        if not stat_name or not kosis_data or not column_names:
            return out
        candidates = [(k, v) for k, v in PRESET_STAT_CONFIG.items() if k in stat_name]
        cfg = max(candidates, key=lambda x: len(x[0]))[1] if candidates else None
        if not cfg:
            return out
        mode = cfg.get("mode", "c2_c3")
        value_norm = cfg.get("value_normalize") or {}
        c2_to_col = cfg.get("c2_to_col") or {}
        cols_ordered = list(column_names)
        if mode == "single_c2":
            dist = self._parse_single_c2_distribution(
                kosis_data, value_normalize=value_norm, c1_all_only=cfg.get("c1_all_only", True)
            )
            if dist:
                labels = list(dist.keys())
                p = list(dist.values())
                out.append((cols_ordered[0], labels, p, None))
            return out
        if mode == "c2_c3":
            col_dists = self._parse_c2_c3_distributions(
                kosis_data,
                c2_to_col_name=c2_to_col,
                value_normalize=value_norm,
                c1_all_only=cfg.get("c1_all_only", True),
                c2_fallback_map=cfg.get("c2_fallback_map"),
                c2_code_to_col=cfg.get("c2_code_to_col"),
            )
            for col_name in cols_ordered:
                dist = col_dists.get(col_name, {})
                if not dist:
                    continue
                labels = list(dist.keys())
                p = list(dist.values())
                if labels and p:
                    out.append((col_name, labels, p, None))
            return out
        if mode == "numeric_c2":
            return out
        if mode in ("c2_code_two", "c2_code_three", "itm_nm_c2"):
            return out
        return out


# stat_name substring -> preset config (mode, value_normalize, c2_to_col, c1_all_only)
PRESET_STAT_CONFIG: Dict[str, Dict[str, Any]] = {
    "거주지역 대중교통 만족도": {
        "mode": "c2_c3",
        "c1_all_only": True,
        "c2_to_col": {
            "시내버스/마을버스": "시내버스/마을버스 만족도",
            "시외/고속버스": "시외/고속버스 만족도",
            "택시": "택시 만족도",
            "기타(기차,선박)": "기타(기차,선박)만족도",
            "기타(기차·선박)": "기타(기차,선박)만족도",
            "기타": "기타(기차,선박)만족도",
        },
        "value_normalize": {
            "매우 불만족": "매우 불만족", "약간 불만족": "약간 불만족", "보통": "보통",
            "약간 만족": "약간 만족", "매우만족": "매우만족", "해당없음": "해당없음",
        },
        "fallback_value": "해당없음",
        "c2_fallback_map": {"기타(기차,선박)만족도": ["기타"]},
        "c2_code_to_col": {"B01": "시내버스/마을버스 만족도", "B02": "시외/고속버스 만족도", "B03": "택시 만족도", "B04": "기타(기차,선박)만족도"},
    },
    "의료기관 주 이용시설": {
        "mode": "single_c2",
        "c1_all_only": True,
        "value_normalize": {
            "종합병원": "종합병원", "의원(외래중심)": "의원(외래중심)", "치과 병·의원": "치과 병·의원",
            "한방병·의원": "한방병·의원", "보건소": "보건소", "약국": "약국", "기타": "기타",
        },
    },
    "의료시설 만족도": {
        "mode": "single_c2",
        "c1_all_only": True,
        "value_normalize": {
            "매우 불만족": "매우 불만족", "불만족": "불만족", "보통": "보통", "만족": "만족", "매우만족": "매우만족",
        },
    },
    "지역의 사회복지 서비스 만족도": {
        "mode": "c2_c3",
        "c1_all_only": True,
        "c2_to_col": {
            "임신·출산·육아에 대한 복지": "임신·출산·육아에 대한 복지",
            "저소득층 등 취약계층에 대한 복지": "저소득층 등 취약계층에 대한 복지",
        },
        "value_normalize": {
            "전혀 그렇지  않다": "전혀 그렇지 않다", "전혀 그렇지 않다": "전혀 그렇지 않다",
            "그렇지 않은 편이다": "그렇지 않은 편이다", "보통이다": "보통이다",
            "그런편이다": "그런 편이다", "그런 편이다": "그런 편이다",
            "매우그렇다": "매우그렇다", "잘모르겠다": "잘모르겠다",
        },
    },
    "도정만족도": {
        "mode": "c2_c3",
        "c1_all_only": True,
        "c2_to_col": {
            "도정정책 만족도": "도정정책 만족도",
            "행정서비스 만족도": "행정서비스 만족도",
        },
        "value_normalize": {
            "매우 불만족": "매우 불만족", "비교적 불만족": "비교적 불만족", "보통": "보통",
            "비교적 만족": "비교적 만족", "매우 만족": "매우 만족",
        },
    },
    "사회적관계별 소통정도": {
        "mode": "single_c2",
        "c1_all_only": True,
        "value_normalize": {"있다": "있다", "없다": "없다"},
    },
    "일반인에 대한 신뢰": {
        "mode": "single_c2",
        "c1_all_only": True,
        "value_normalize": {
            "전혀 신뢰하지 않는다": "전혀 신뢰하지 않는다", "별로 신뢰하지 않는다": "별로 신뢰하지 않는다",
            "약간 신뢰한다": "약간 신뢰한다", "완전히 신뢰한다": "완전히 신뢰한다",
        },
    },
    "주관적 귀속계층": {
        "mode": "single_c2",
        "c1_all_only": True,
        "value_normalize": {
            "상상": "상상", "상하": "상하", "중상": "중상", "중하": "중하", "하상": "하상", "하하": "하하",
        },
    },
    "지역소속감": {
        "mode": "c2_c3",
        "c1_all_only": True,
        "c2_to_col": {
            "동네": "동네", "시군": "시군", "경상북도": "경상북도",
        },
        "value_normalize": {
            "전혀 소속감이 없다": "전혀 소속감이 없다", "별로 소속감이 없다": "별로 소속감이 없다",
            "다소 소속감이 있다": "다소 소속감이 있다", "매우 소속감이 있다": "매우 소속감이 있다",
        },
    },
    "안전환경에 대한 평가": {
        "mode": "c2_c3",
        "c1_all_only": True,
        "c2_to_col": {
            "어둡고 후미진 곳이 많다": "어둡고 후미진 곳이 많다",
            "주변에 쓰레기가 아무렇게 버려져 있고 지저분 하다": "주변에 쓰레기가 아무렇게 버려져 있고 지저분 하다",
            "주변에 방치된 차나 빈 건물이 많다": "주변에 방치된 차나 빈 건물이 많다",
            "무리 지어 다니는 불량 청소년이 많다": "무리 지어 다니는 불량 청소년이 많다",
            "기초질서(무단횡단, 불법 주정차 등)를 지키지 않는 사람이 많다": "기초질서(무단횡단, 불법 주정차 등)를 지키지 않는 사람이 많다",
            "큰소리로 다투거나 싸우는 사람들을 자주 볼 수 있다": "큰소리로 다투거나 싸우는 사람들을 자주 볼 수 있다",
        },
        "c2_fallback_map": {
            "기초질서(무단횡단, 불법 주정차 등)를 지키지 않는 사람이 많다": ["기초질서"],
        },
        "value_normalize": {
            "전혀 그렇지 않다": "전혀 그렇지 않다", "그렇지 않은 편이다": "그렇지 않은 편이다",
            "보통이다": "보통이다", "그런 편이다": "그런 편이다", "매우 그렇다": "매우 그렇다",
        },
    },
    "일상생활 범죄피해 두려움": {
        "mode": "c2_c3",
        "c1_all_only": True,
        "c2_to_col": {
            "나자신": "나자신", "배우자(애인)": "배우자(애인)", "자녀": "자녀", "부모": "부모",
        },
        "value_normalize": {
            "전혀 두렵지 않다": "전혀 두렵지 않다", "두렵지 않은 편이다": "두렵지 않은 편이다",
            "보통이다": "보통이다", "두려운편이다": "두려운 편이다", "매우 두렵다": "매우 두렵다",
            "해당자 없음": "해당자 없음",
        },
    },
    "일상생활에서 두려움": {
        "mode": "c2_c3",
        "c1_all_only": True,
        "c2_to_col": {
            "밤에 혼자 집에 있을 때": "밤에 혼자 집에 있을 때",
            "밤에 혼자 지역(동네)의 골목길을 걸을때": "밤에 혼자 지역(동네)의 골목길을 걸을때",
        },
        "value_normalize": {
            "전혀두렵지 않다": "전혀 두렵지 않다", "전혀 두렵지 않다": "전혀 두렵지 않다",
            "두렵지 않은 편이다": "두렵지 않은 편이다", "보통이다": "보통이다",
            "두려운편이다": "두려운 편이다", "매우 두렵다": "매우 두렵다",
        },
    },
    "환경체감도": {
        "mode": "c2_c3",
        "c1_all_only": True,
        "c2_to_col": {
            "대기(미세먼지 악취 매연 오존경보)": "대기",
            "대기": "대기",
            "수질": "수질",
            "토양": "토양",
            "소음/진동": "소음/진동",
            "소음": "소음/진동",
            "진동": "소음/진동",
            "녹지환경": "녹지환경",
            "녹지": "녹지환경",
        },
        "c2_fallback_map": {
            "대기": ["대기"],
            "수질": ["수질"],
            "토양": ["토양"],
            "소음/진동": ["소음", "진동"],
            "녹지환경": ["녹지"],
        },
        "value_normalize": {
            "매우나쁘다": "매우나쁘다", "약간나쁘다": "약간나쁘다", "보통이다": "보통이다",
            "약간좋다": "약간좋다", "매우좋다": "매우좋다",
        },
    },
    "생활시간 압박": {
        "mode": "itm_nm_c2",
        "c1_all_only": True,
        "itm_to_col": {"평일": "평일 생활시간 압박", "주말": "주말 생활시간 압박"},
        "value_normalize": {
            "전혀 그렇지 않다": "전혀 그렇지 않다", "별로 그렇지 않다": "별로 그렇지 않다",
            "가끔 그렇다": "가끔 그렇다", "항상 그렇다": "항상 그렇다",
        },
    },
    "자신의 평소 준법수준": {
        "mode": "c2_code_two",
        "c1_all_only": True,
        "col0_prefix": "B01", "col1_prefix": "B02",
        "value_normalize": {
            "전혀 지키지 않는다": "전혀 지키지 않는다", "비교적 지키지 않는다": "비교적 지키지 않는다",
            "보통이다": "보통이다", "비교적 잘 지킨다": "비교적 잘 지킨다", "아주 잘 지킨다": "아주 잘 지킨다",
            "법을지키면 손해볼것 같아서": "법을지키면 손해볼것 같아서", "처벌규정이 미약하기 때문": "처벌규정이 미약하기 때문",
            "다른 사람도 지키지 않아서": "다른 사람도 지키지 않아서", "귀찮아서": "귀찮아서",
            "단속이 잘 안되기 때문에": "단속이 잘 안되기 때문에", "준법교육을 잘 받지 않았기 때문에": "준법교육을 잘 받지 않았기 때문에", "기타": "기타",
        },
    },
    "지난1년 동안 자원봉사활동 여부및시간": {
        "mode": "c2_code_three",
        "c1_all_only": True,
        "col0_prefix": "B01", "col1_prefix": "B02", "col2_numeric": True,
        "value_normalize": {"있다": "있다", "없다": "없다"},
    },
    "지난1년 동안 후원금 금액": {
        "mode": "c2_code_three",
        "c1_all_only": True,
        "col0_prefix": "B01", "col1_prefix": "B02", "col2_numeric": True,
        "value_normalize": {"있다": "있다", "없다": "없다"},
    },
    "여가활동 만족도 및 불만족 이유": {
        "mode": "c2_code_three",
        "c1_all_only": True,
        "col0_prefix": "B01", "col1_prefix": "B01", "col2_prefix": "B02",
        "value_normalize": {
            "매우 불만족": "매우 불만족", "약간 불만족": "약간 불만족", "보통": "보통",
            "약간 만족": "약간 만족", "매우 만족": "매우 만족",
        },
        "col0_is_satisfaction": True,  # col0에 만족도 수준이 들어감
    },
    "지난 1년 동안 문화예술행사 관람 경험 여부 및 관람 횟수": {
        "mode": "c2_code_two",
        "c1_all_only": True,
        "col0_prefix": "B01", "col1_prefix": "B02",
        "value_normalize": {"있다": "있다", "없다": "없다"},
    },
    "삶에 대한 만족감과 정서경험": {
        "mode": "numeric_c2",
        "c1_all_only": True,
        "c2_to_col": {
            "삶에 대한 전반적 만족감": "삶에 대한 전반적 만족감(10점 만점)",
            "살고있는 지역의 전반적 만족감": "살고있는 지역의 전반적 만족감(10점 만점)",
            "어제 행복 정도": "어제 행복 정도(10점 만점)",
            "어제 걱정 정도": "어제 걱정 정도(10점 만점)",
        },
    },
    "행복수준": {
        "mode": "numeric_c2",
        "c1_all_only": True,
        "c2_to_col": {
            "생활수준": "생활수준(10점 만점)", "건강상태": "건강상태(10점 만점)", "성취도": "성취도(10점 만점)",
            "대인관계": "대인관계(10점 만점)", "안전정도": "안전정도(10점 만점)",
            "지역사회소속감": "지역사회소속감(10점 만점)", "미래안정성": "미래안정성(10점 만점)",
        },
    },
}
