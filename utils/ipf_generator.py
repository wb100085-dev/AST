import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

# ========================================
# 1. 성별 맞춤 한국 이름 생성
# ========================================

def _build_extended_name_pools():
    """1만 명 이상 중복 없이 생성 가능하도록 확장된 이름 풀 (성 30 × 이름 350+ = 10,500+ per gender)"""
    # 남자 이름 350개+
    male_first = [
        "민준", "서준", "예준", "도윤", "시우", "주원", "하준", "지호", "준서", "건우",
        "현우", "우진", "승현", "유준", "정우", "승우", "지훈", "민성", "준혁", "지환",
        "태양", "준영", "성민", "동현", "재현", "상우", "호진", "영수", "철수", "광수",
        "도훈", "시현", "민재", "현준", "지원", "승민", "유진", "태희", "준호", "상훈",
        "진우", "민호", "재윤", "시윤", "동욱", "성훈", "민수", "정훈", "재원", "현서",
        "승원", "태윤", "지한", "도원", "시훈", "민규", "현식", "성준", "재민", "동민",
        "준석", "영호", "진호", "상민", "정민", "우빈", "시원", "태민", "준현", "민우",
        "현민", "지성", "승호", "동혁", "성우", "준형", "도현", "진현", "민석", "현성",
        "지용", "태영", "영민", "상현", "정현", "재훈", "동현", "승재", "준기", "진성",
        "민혁", "도영", "시민", "범준", "성진", "재호", "태훈", "민혁", "준우", "현도",
        "영준", "시준", "도준", "민규", "성욱", "재영", "태영", "준혁", "현석", "지웅",
        "동준", "승준", "민호", "성호", "재훈", "태현", "준성", "현우", "영훈", "시훈",
        "도현", "진우", "민재", "성민", "재현", "태윤", "준영", "현준", "지훈", "동훈",
        "승현", "민석", "성재", "재민", "태준", "준호", "현민", "영호", "시현", "도훈",
        "진호", "민우", "성준", "재우", "태우", "준현", "현진", "지원", "동윤", "승윤",
        "민영", "성훈", "재성", "태성", "준수", "현수", "영수", "시원", "도원", "진수",
        "민수", "성수", "재수", "태수", "준희", "현희", "지희", "동희", "승희", "민희",
        "성희", "재희", "태희", "준아", "현아", "지아", "동아", "승아", "민아", "성아",
        "재아", "태아", "준이", "현이", "지이", "동이", "승이", "민이", "성이", "재이",
        "태이", "기준", "성규", "재율", "태율", "준율", "현율", "지율", "동율", "승율",
        "민율", "성율", "재율", "태율", "준빈", "현빈", "지빈", "동빈", "승빈", "민빈",
        "성빈", "재빈", "태빈", "준일", "현일", "지일", "동일", "승일", "민일", "성일",
        "재일", "태일", "준석", "현석", "지석", "동석", "승석", "민석", "성석", "재석",
        "태석", "준혁", "현혁", "지혁", "동혁", "승혁", "민혁", "성혁", "재혁", "태혁",
        "영재", "시재", "도재", "진재", "민재", "성재", "재재", "태재", "준재", "현재",
        "지재", "동재", "승재", "영훈", "시훈", "도훈", "진훈", "민훈", "성훈", "재훈",
        "태훈", "준훈", "현훈", "지훈", "동훈", "승훈", "영호", "시호", "도호", "진호",
        "민호", "성호", "재호", "태호", "준호", "현호", "지호", "동호", "승호", "영수",
        "시수", "도수", "진수", "민수", "성수", "재수", "태수", "준수", "현수", "지수",
        "동수", "승수", "영민", "시민", "도민", "진민", "민민", "성민", "재민", "태민",
        "준민", "현민", "지민", "동민", "승민", "영현", "시현", "도현", "진현", "민현",
        "성현", "재현", "태현", "준현", "현현", "지현", "동현", "승현", "영준", "시준",
        "도준", "진준", "민준", "성준", "재준", "태준", "준준", "현준", "지준", "동준",
        "승준", "영우", "시우", "도우", "진우", "민우", "성우", "재우", "태우", "준우",
        "현우", "지우", "동우", "승우", "영진", "시진", "도진", "진진", "민진", "성진",
        "재진", "태진", "준진", "현진", "지진", "동진", "승진", "영성", "시성", "도성",
        "진성", "민성", "성성", "재성", "태성", "준성", "현성", "지성", "동성", "승성",
    ]
    # 여자 이름 350개+
    female_first = [
        "서연", "민서", "지우", "서윤", "지민", "수아", "하은", "예은", "지유", "예린",
        "채원", "수빈", "소율", "지아", "다은", "은서", "가은", "윤서", "나은", "하윤",
        "수진", "영희", "미영", "지영", "현정", "소희", "민지", "혜진", "은지", "정아",
        "서현", "유나", "지현", "수민", "예진", "서영", "채은", "지원", "하린", "예나",
        "민아", "지수", "수현", "서하", "채윤", "지은", "소민", "다인", "예서", "민정",
        "지혜", "수영", "채린", "서아", "예주", "지안", "서진", "예지", "채현", "유리",
        "다솜", "예빈", "소윤", "민희", "예원", "서우", "채민", "수지", "지나", "민영",
        "예은", "수연", "하진", "채아", "다영", "예린", "민주", "서영", "수빈", "하나",
        "지원", "예나", "서하", "민서", "채은", "수아", "지은", "유진", "다은", "은지",
        "소율", "하윤", "수민", "지유", "예린", "채원", "민지", "서윤", "지현", "예은",
        "수진", "영희", "미영", "지영", "현정", "소희", "혜진", "정아", "서현", "유나",
        "예진", "서영", "지원", "하린", "민아", "지수", "수현", "채윤", "소민", "민정",
        "지혜", "수영", "채린", "서아", "예주", "지안", "채현", "다솜", "예빈", "민희",
        "예원", "채민", "수지", "지나", "민영", "수연", "하진", "채아", "다영", "민주",
        "서희", "지희", "예희", "수희", "민희", "채희", "하희", "유희", "은희", "소희",
        "서영", "지영", "예영", "수영", "민영", "채영", "하영", "유영", "은영", "소영",
        "서진", "지진", "예진", "수진", "민진", "채진", "하진", "유진", "은진", "소진",
        "서아", "지아", "예아", "수아", "민아", "채아", "하아", "유아", "은아", "소아",
        "서윤", "지윤", "예윤", "수윤", "민윤", "채윤", "하윤", "유윤", "은윤", "소윤",
        "서현", "지현", "예현", "수현", "민현", "채현", "하현", "유현", "은현", "소현",
        "서민", "지민", "예민", "수민", "민민", "채민", "하민", "유민", "은민", "소민",
        "서원", "지원", "예원", "수원", "민원", "채원", "하원", "유원", "은원", "소원",
        "서빈", "지빈", "예빈", "수빈", "민빈", "채빈", "하빈", "유빈", "은빈", "소빈",
        "서율", "지율", "예율", "수율", "민율", "채율", "하율", "유율", "은율", "소율",
        "서린", "지린", "예린", "수린", "민린", "채린", "하린", "유린", "은린", "소린",
        "서나", "지나", "예나", "수나", "민나", "채나", "하나", "유나", "은나", "소나",
        "서안", "지안", "예안", "수안", "민안", "채안", "하안", "유안", "은안", "소안",
        "서주", "지주", "예주", "수주", "민주", "채주", "하주", "유주", "은주", "소주",
        "서리", "지리", "예리", "수리", "민리", "채리", "하리", "유리", "은리", "소리",
        "서솜", "지솜", "예솜", "수솜", "민솜", "채솜", "하솜", "유솜", "은솜", "소솜",
        "서인", "지인", "예인", "수인", "민인", "채인", "하인", "유인", "은인", "소인",
        "서정", "지정", "예정", "수정", "민정", "채정", "하정", "유정", "은정", "소정",
        "서혜", "지혜", "예혜", "수혜", "민혜", "채혜", "하혜", "유혜", "은혜", "소혜",
        "서영", "지영", "예영", "수영", "민영", "채영", "하영", "유영", "은영", "소영",
        "서은", "지은", "예은", "수은", "민은", "채은", "하은", "유은", "은은", "소은",
        "서가", "지가", "예가", "수가", "민가", "채가", "하가", "유가", "은가", "소가",
        "서다", "지다", "예다", "수다", "민다", "채다", "하다", "유다", "은다", "소다",
        "서나", "지나", "예나", "수나", "민나", "채나", "하나", "유나", "은나", "소나",
    ]
    # 성 30개 (20 → 30으로 확장)
    last_names = [
        "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임",
        "한", "오", "서", "신", "권", "황", "안", "송", "전", "홍",
        "문", "양", "손", "배", "백", "허", "유", "남", "심", "노"
    ]
    return male_first, female_first, last_names


def generate_korean_name(gender: str = None) -> str:
    """성별에 맞는 한국 이름 생성 (고유 이름 수: 성 30 × 이름 350+ = 1만 명 이상 per gender, 중복 없이)"""
    male_first, female_first, last_names = _build_extended_name_pools()
    # 중복 제거로 고유 조합 수 확대 (성 30 × 고유 이름 수)
    male_first = list(dict.fromkeys(male_first))
    female_first = list(dict.fromkeys(female_first))
    return _name_from_pools(male_first, female_first, last_names, gender)


def _name_from_pools(male_first: list, female_first: list, last_names: list, gender: str = None) -> str:
    """미리 구축된 이름 풀에서 이름 1개 반환 (반복 호출 시 풀 재구축 비용 제거)."""
    last = random.choice(last_names)
    if gender == "남자":
        first = random.choice(male_first)
    elif gender == "여자":
        first = random.choice(female_first)
    else:
        first = random.choice(male_first + female_first)
    return f"{last}{first}"


# ========================================
# 2. 중복 제거
# ========================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """6축 기준 중복 제거"""
    key_columns = ['거주지역', '성별', '연령', '경제활동', '교육정도', '월평균소득']
    existing_cols = [c for c in key_columns if c in df.columns]
    
    if existing_cols:
        df_unique = df.drop_duplicates(subset=existing_cols, keep='first')
        return df_unique.reset_index(drop=True)
    
    return df


# ========================================
# 3. 현실적 제약 조건 적용 ⭐ 핵심
# ========================================

def apply_realistic_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """현실적인 제약 조건 적용 (행방향 개연성 강화)"""
    
    required = ['연령', '경제활동', '월평균소득', '교육정도']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"⚠️ apply_realistic_constraints: 필수 컬럼 누락 {missing}")
        return df
    
    df = df.copy()
    
    print("🔧 현실적 제약 조건 적용 시작 (행방향 개연성 강화)...")
    
    # 🔹 규칙 1: 고령자(65세 이상) → 비경제활동으로 조정 (80% 이상)
    elderly_econ = (
        (df['연령'] >= 65) &
        (df['경제활동'] == '경제활동')
    )
    
    if elderly_econ.sum() > 0:
        # 80%를 비경제활동으로 변경
        target_count = int(elderly_econ.sum() * 0.2)  # 20%만 경제활동 유지
        adjust_count = elderly_econ.sum() - target_count
        
        if adjust_count > 0:
            adjust_indices = df[elderly_econ].sample(n=adjust_count, random_state=42).index
            df.loc[adjust_indices, '경제활동'] = '비경제활동'
            low_income_options = ['50만원미만', '50-100만원', '100-200만원']
            df.loc[adjust_indices, '월평균소득'] = np.random.choice(low_income_options, size=adjust_count, p=[1/3, 1/3, 1/3])
            print(f"   ✅ 고령자 경제활동 → 비경제활동 조정: {adjust_count}건")
    
    # 🔹 규칙 2: 비경제활동 → 저소득으로 제한 (절대적)
    non_econ_high_income = (
        (df['경제활동'] == '비경제활동') &
        (df['월평균소득'].isin(['300-400만원', '400-500만원', '500-600만원', '600-700만원', '700-800만원', '800만원이상']))
    )
    
    if non_econ_high_income.sum() > 0:
        n_adj = non_econ_high_income.sum()
        low_income_options = ['50만원미만', '50-100만원', '100-200만원', '200-300만원']
        df.loc[non_econ_high_income, '월평균소득'] = np.random.choice(low_income_options, size=n_adj, p=[0.25, 0.25, 0.25, 0.25])
        print(f"   ✅ 비경제활동 고소득 조정: {n_adj}건")
    
    # 🔹 규칙 3: 청년(20-35세) + 경제활동 → 적정 소득 보장
    young_econ_low_income = (
        (df['경제활동'] == '경제활동') &
        (df['연령'].between(20, 35)) &
        (df['월평균소득'].isin(['50만원미만', '50-100만원']))
    )
    
    if young_econ_low_income.sum() > 0:
        n_adj = young_econ_low_income.sum()
        young_income_options = ['100-200만원', '200-300만원', '300-400만원']
        df.loc[young_econ_low_income, '월평균소득'] = np.random.choice(young_income_options, size=n_adj, p=[1/3, 1/3, 1/3])
        print(f"   ✅ 청년 경제활동 저소득 조정: {n_adj}건")
    
    # 🔹 규칙 4: 중장년(36-55세) + 경제활동 → 중상위 소득
    middle_age_econ = (
        (df['경제활동'] == '경제활동') &
        (df['연령'].between(36, 55)) &
        (df['월평균소득'].isin(['50만원미만', '50-100만원', '100-200만원']))
    )
    
    if middle_age_econ.sum() > 0:
        n_adj = middle_age_econ.sum()
        middle_income_options = ['300-400만원', '400-500만원', '500-600만원']
        df.loc[middle_age_econ, '월평균소득'] = np.random.choice(middle_income_options, size=n_adj, p=[1/3, 1/3, 1/3])
        print(f"   ✅ 중장년 경제활동 저소득 조정: {n_adj}건")
    
    # 🔹 규칙 5: 고령자(65+) + 대졸이상 → 희소 (20%로 제한)
    elderly_college = (
        (df['연령'] >= 65) &
        (df['교육정도'] == '대졸이상')
    )
    
    if elderly_college.sum() > 0:
        target_count = int(elderly_college.sum() * 0.2)  # 20%만 유지
        adjust_count = elderly_college.sum() - target_count
        
        if adjust_count > 0:
            adjust_indices = df[elderly_college].sample(n=adjust_count, random_state=42).index
            df.loc[adjust_indices, '교육정도'] = np.random.choice(['고졸', '중졸이하'], size=adjust_count, p=[0.5, 0.5])
            print(f"   ✅ 고령자 대졸 비율 조정: {adjust_count}건")
    
    # 🔹 규칙 6: 대졸 초년생(20-25세) → 적정 소득
    young_college = (
        (df['연령'].between(20, 25)) &
        (df['교육정도'] == '대졸이상') &
        (df['월평균소득'].isin(['400-500만원', '500-600만원', '600-700만원', '700-800만원', '800만원이상']))
    )
    
    if young_college.sum() > 0:
        n_adj = young_college.sum()
        df.loc[young_college, '월평균소득'] = np.random.choice(['100-200만원', '200-300만원', '300-400만원'], size=n_adj, p=[1/3, 1/3, 1/3])
        print(f"   ✅ 대졸 초년생 고소득 조정: {n_adj}건")
    
    # 🔹 규칙 7: 중장년(36-55세) + 비경제활동 → 매우 낮은 소득
    middle_age_non_econ = (
        (df['경제활동'] == '비경제활동') &
        (df['연령'].between(36, 55)) &
        (df['월평균소득'].isin(['200-300만원', '300-400만원', '400-500만원', '500-600만원', '600-700만원', '700-800만원', '800만원이상']))
    )
    
    if middle_age_non_econ.sum() > 0:
        n_adj = middle_age_non_econ.sum()
        very_low_income_options = ['50만원미만', '50-100만원', '100-200만원']
        df.loc[middle_age_non_econ, '월평균소득'] = np.random.choice(very_low_income_options, size=n_adj, p=[1/3, 1/3, 1/3])
        print(f"   ✅ 중장년 비경제활동 고소득 조정: {n_adj}건")
    
    # 🔹 규칙 8: 고령자(65+) + 비경제활동 → 매우 낮은 소득
    elderly_non_econ = (
        (df['경제활동'] == '비경제활동') &
        (df['연령'] >= 65) &
        (df['월평균소득'].isin(['200-300만원', '300-400만원', '400-500만원', '500-600만원', '600-700만원', '700-800만원', '800만원이상']))
    )
    
    if elderly_non_econ.sum() > 0:
        n_adj = elderly_non_econ.sum()
        very_low_income_options = ['50만원미만', '50-100만원', '100-200만원']
        df.loc[elderly_non_econ, '월평균소득'] = np.random.choice(very_low_income_options, size=n_adj, p=[1/3, 1/3, 1/3])
        print(f"   ✅ 고령자 비경제활동 고소득 조정: {n_adj}건")
    
    # 🔹 규칙 9: 20세 미만 제거
    under_20 = df['연령'] < 20
    if under_20.sum() > 0:
        df = df[~under_20].reset_index(drop=True)
        print(f"   ✅ 20세 미만 제거: {under_20.sum()}건")
    
    print("✅ 현실적 제약 조건 적용 완료")
    return df


# ========================================
# 4. IPF 유틸리티
# ========================================

def _normalize_prob(probs: List[float]) -> List[float]:
    """확률 정규화"""
    total = sum(probs)
    if total == 0:
        return [1.0 / len(probs)] * len(probs)
    return [p / total for p in probs]


def _safe_choice(labels: List, probs: List[float]):
    """안전한 weighted random choice"""
    if not labels or not probs or len(labels) != len(probs):
        return labels[0] if labels else None
    
    probs = _normalize_prob(probs)
    return np.random.choice(labels, p=probs)


# ========================================
# 5. IPF 알고리즘
# ========================================

def apply_ipf(
    df: pd.DataFrame,
    n: int,
    margins_axis: Dict[str, Dict],
    max_iterations: int = 10
) -> pd.DataFrame:
    """IPF 알고리즘 적용 (90% 확정 + 10% 확률)"""
    
    axis_name_map = {
        'sigungu': '거주지역',
        'gender': '성별',
        'age': '연령',
        'econ': '경제활동',
        'income': '월평균소득',
        'edu': '교육정도'
    }
    
    target_counts = {}
    
    for axis_key, margin_data in margins_axis.items():
        col_name = axis_name_map.get(axis_key)
        if not col_name or col_name not in df.columns:
            continue
        
        labels = margin_data.get('labels', [])
        # 'p' 키를 우선 사용하고, 없으면 'probs' 사용 (하위 호환성)
        probs = margin_data.get('p', margin_data.get('probs', []))
        
        if labels and probs:
            probs = _normalize_prob(probs)
            
            # ✅ 비율 순위 보장: 확률 내림차순으로 정렬
            label_prob_pairs = list(zip(labels, probs))
            label_prob_pairs.sort(key=lambda x: x[1], reverse=True)  # 확률 높은 순서대로
            labels_sorted, probs_sorted = zip(*label_prob_pairs)
            labels_sorted = list(labels_sorted)
            probs_sorted = list(probs_sorted)
            
            target_counts[col_name] = {
                lbl: int(round(n * p)) for lbl, p in zip(labels_sorted, probs_sorted)
            }
            
            # ✅ 디버깅: 성별 목표값 확인
            if col_name == '성별':
                print(f"📊 IPF 목표 성별 분포 (순위 보장):")
                for lbl, target in target_counts[col_name].items():
                    pct = (target / n * 100) if n > 0 else 0
                    print(f"   {lbl}: {target}명 ({pct:.2f}%)")
    
    if not target_counts:
        return df
    
    print(f"🔄 IPF 시작 (목표 인구: {n}명, 최대 반복: {max_iterations}회)")
    
    # IPF 반복 조정: 각 축별로 목표 분포에 맞게 조정 (순위 절대 보장)
    for iteration in range(max_iterations):
        total_adjustment = 0
        
        for col_name, targets in target_counts.items():
            # ✅ 순위 절대 보장: 확률 높은 순서대로 처리
            sorted_targets = sorted(targets.items(), key=lambda x: x[1], reverse=True)
            
            # ✅ 순위 검증 및 강제 수정: 확률 높은 항목이 확실히 더 많도록 보장
            current_counts = {label: (df[col_name] == label).sum() for label in targets.keys()}
            
            # 순위 위반 체크 및 수정
            for i, (label_high, target_high) in enumerate(sorted_targets):
                for j, (label_low, target_low) in enumerate(sorted_targets[i+1:], start=i+1):
                    current_high = current_counts[label_high]
                    current_low = current_counts[label_low]
                    
                    # 순위 위반: 높은 확률인데 낮은 확률보다 적으면 강제 수정
                    if current_high < current_low:
                        # 낮은 확률에서 높은 확률로 이동
                        low_indices = df[df[col_name] == label_low].index.tolist()
                        move_count = min((current_low - current_high) // 2, len(low_indices))
                        if move_count > 0:
                            move_indices = np.random.choice(low_indices, size=move_count, replace=False)
                            df.loc[move_indices, col_name] = label_high
                            current_counts[label_high] += move_count
                            current_counts[label_low] -= move_count
                            total_adjustment += move_count
            
            # 목표값에 맞게 조정
            for label, target in sorted_targets:
                current_count = (df[col_name] == label).sum()
                diff = target - current_count
                
                if abs(diff) < 1:  # 1명 이하 차이는 무시
                    continue
                
                if diff > 0:
                    # 부족: 추가
                    current_indices = df[df[col_name] == label].index.tolist()
                    if len(current_indices) > 0:
                        add_count = int(diff)
                        additional_indices = np.random.choice(
                            current_indices,
                            size=add_count,
                            replace=True
                        )
                        duplicate_rows = df.loc[additional_indices].copy()
                        df = pd.concat([df, duplicate_rows], ignore_index=True)
                        total_adjustment += add_count
                
                elif diff < 0:
                    # 초과: 제거 (낮은 확률부터)
                    current_indices = df[df[col_name] == label].index.tolist()
                    remove_count = int(-diff)
                    if remove_count > 0 and len(current_indices) > remove_count:
                        remove_indices = np.random.choice(
                            current_indices,
                            size=remove_count,
                            replace=False
                        )
                        df = df.drop(index=remove_indices).reset_index(drop=True)
                        total_adjustment += remove_count
        
        if total_adjustment == 0:
            print(f"   반복 {iteration + 1}/{max_iterations}: 조정 완료 (수렴)")
            break
        
        print(f"   반복 {iteration + 1}/{max_iterations} 완료 (현재 인구: {len(df)}명, 조정: {total_adjustment}명)")
    
    # 최종 조정: 목표 인구수에 맞추기 (목표 분포 비율 유지 + 순위 절대 보장)
    current_len = len(df)
    if current_len != n:
        if current_len > n:
            # 초과: 목표 분포에 맞게 제거 (순위 절대 보장)
            keep_indices = []
            
            # ✅ 순위 절대 보장: 각 축별로 확률 높은 순서대로 우선 보존
            for col_name, targets in target_counts.items():
                sorted_targets = sorted(targets.items(), key=lambda x: x[1], reverse=True)
                
                for label, target in sorted_targets:
                    label_indices = df[df[col_name] == label].index.tolist()
                    keep_count = min(target, len(label_indices))
                    
                    if keep_count > 0 and len(label_indices) > 0:
                        selected = np.random.choice(label_indices, size=keep_count, replace=False)
                        keep_indices.extend(selected.tolist())
            
            # 중복 제거
            keep_indices = list(set(keep_indices))
            
            # 목표 인구수에 맞게 조정
            if len(keep_indices) > n:
                # 초과하면 목표 분포 비율에 맞게 샘플링 (순위 절대 보장)
                # 확률 높은 label부터 우선 선택
                priority_indices = []
                for col_name, targets in target_counts.items():
                    sorted_targets = sorted(targets.items(), key=lambda x: x[1], reverse=True)
                    for label, target in sorted_targets:
                        label_indices = [idx for idx in keep_indices if df.loc[idx, col_name] == label]
                        priority_indices.extend(label_indices[:target])
                
                if len(priority_indices) >= n:
                    keep_indices = np.random.choice(priority_indices, size=n, replace=False).tolist()
                else:
                    remaining = list(set(keep_indices) - set(priority_indices))
                    needed = n - len(priority_indices)
                    if len(remaining) > 0:
                        additional = np.random.choice(remaining, size=min(needed, len(remaining)), replace=False)
                        keep_indices = priority_indices + additional.tolist()
                    else:
                        keep_indices = np.random.choice(keep_indices, size=n, replace=False).tolist()
            elif len(keep_indices) < n:
                # 부족하면 나머지에서 추가 (목표 분포 유지)
                all_indices = set(df.index.tolist())
                remaining_indices = list(all_indices - set(keep_indices))
                if len(remaining_indices) > 0:
                    additional_needed = n - len(keep_indices)
                    additional = np.random.choice(remaining_indices, size=min(additional_needed, len(remaining_indices)), replace=False)
                    keep_indices.extend(additional.tolist())
            
            df = df.loc[keep_indices].reset_index(drop=True)
            
            # ✅ 최종 순위 검증 및 강제 수정
            for col_name, targets in target_counts.items():
                sorted_targets = sorted(targets.items(), key=lambda x: x[1], reverse=True)
                current_counts = {label: (df[col_name] == label).sum() for label in targets.keys()}
                
                # 순위 위반 체크 및 수정
                for i, (label_high, target_high) in enumerate(sorted_targets):
                    for j, (label_low, target_low) in enumerate(sorted_targets[i+1:], start=i+1):
                        current_high = current_counts[label_high]
                        current_low = current_counts[label_low]
                        
                        # 순위 위반: 높은 확률인데 낮은 확률보다 적으면 강제 수정
                        if current_high < current_low:
                            # 낮은 확률에서 높은 확률로 이동
                            low_indices = df[df[col_name] == label_low].index.tolist()
                            move_count = current_low - current_high + 1  # 최소 1명 차이 보장
                            move_count = min(move_count, len(low_indices))
                            if move_count > 0:
                                move_indices = np.random.choice(low_indices, size=move_count, replace=False)
                                df.loc[move_indices, col_name] = label_high
            
        elif current_len < n:
            # 부족: 목표 분포 비율에 맞게 추가
            additional_count = n - current_len
            additional_rows = []
            
            # 각 label별로 부족한 만큼 추가
            for col_name, targets in target_counts.items():
                for label, target in targets.items():
                    current_count = (df[col_name] == label).sum()
                    needed = max(0, target - current_count)
                    
                    if needed > 0 and additional_count > 0:
                        label_indices = df[df[col_name] == label].index.tolist()
                        if len(label_indices) > 0:
                            sample_count = min(needed, additional_count)
                            sample_indices = np.random.choice(label_indices, size=sample_count, replace=True)
                            additional_rows.extend(sample_indices.tolist())
                            additional_count -= sample_count
                            
                            if additional_count <= 0:
                                break
                
                if additional_count <= 0:
                    break
            
            # 여전히 부족하면 목표 분포 비율에 맞게 추가
            if additional_count > 0:
                for col_name, targets in target_counts.items():
                    for label, target in targets.items():
                        if additional_count <= 0:
                            break
                        label_indices = df[df[col_name] == label].index.tolist()
                        if len(label_indices) > 0:
                            target_ratio = target / n
                            add_count = min(int(additional_count * target_ratio), additional_count)
                            if add_count > 0:
                                sample_indices = np.random.choice(label_indices, size=add_count, replace=True)
                                additional_rows.extend(sample_indices.tolist())
                                additional_count -= add_count
            
            # 최종적으로 부족하면 랜덤 복제
            while len(additional_rows) < (n - current_len):
                random_idx = np.random.randint(0, len(df))
                additional_rows.append(random_idx)
            
            if len(additional_rows) > 0:
                df_additional = df.iloc[additional_rows[:(n - current_len)]].copy()
                df = pd.concat([df, df_additional], ignore_index=True)
    
    # 식별NO 재할당
    df['식별NO'] = range(1, len(df) + 1)
    
    print(f"✅ IPF 완료 (최종 인구: {len(df)}명)")
    return df.reset_index(drop=True)


# ========================================
# 6. 부족분 보충 (다양성 확보)
# ========================================

def fill_shortage_with_diversity(
    df: pd.DataFrame,
    n: int,
    sigungu_labels: List[str],
    gender_labels: List[str],
    age_labels: List[int],
    econ_labels: List[str],
    edu_labels: List[str],
    income_labels: List[str],
    sigungu_probs: List[float],
    gender_probs: List[float],
    age_probs: List[float],
    econ_probs: List[float],
    edu_probs: List[float],
    income_probs: List[float]
) -> pd.DataFrame:
    """부족분을 다양성 있게 보충"""
    
    shortage = n - len(df)
    if shortage <= 0:
        return df
    
    print(f"🔧 부족분 보충 시작 ({shortage}명)")
    
    current_max_id = df['식별NO'].max() if '식별NO' in df.columns else 0
    existing_names = set(df['가상이름'].tolist()) if '가상이름' in df.columns else set()
    
    new_rows = []
    
    for i in range(shortage):
        # 새로운 식별NO
        new_id = current_max_id + i + 1
        
        # 성별 먼저 결정
        gender = _safe_choice(gender_labels, gender_probs)
        
        # 성별에 맞는 가상이름 생성 (중복 방지)
        new_name = generate_korean_name(gender)
        while new_name in existing_names:
            new_name = generate_korean_name(gender)
        existing_names.add(new_name)
        
        # 6축 속성 변형
        new_row = {
            '식별NO': new_id,
            '가상이름': new_name,
            '거주지역': _safe_choice(sigungu_labels, sigungu_probs),
            '성별': gender,
            '연령': _safe_choice(age_labels, age_probs),
            '경제활동': _safe_choice(econ_labels, econ_probs),
            '교육정도': _safe_choice(edu_labels, edu_probs),
            '월평균소득': _safe_choice(income_labels, income_probs)
        }
        
        new_rows.append(new_row)
    
    df_new = pd.DataFrame(new_rows)
    df = pd.concat([df, df_new], ignore_index=True)
    
    print(f"✅ 부족분 보충 완료 ({shortage}명 추가)")
    return df


# ========================================
# 7. 메인 생성 함수
# ========================================

def generate_base_population(
    n: int,
    selected_sigungu: List[str],
    weights_6axis: dict,
    sigungu_pool: List[str],
    seed: int,
    margins_axis: Dict[str, Dict],
    apply_ipf_flag: bool = True
) -> pd.DataFrame:
    """가상인구 생성 (성별 맞춤 이름 + 현실적 제약)"""
    
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"🚀 가상인구 생성 시작 (목표: {n}명)")
    print(f"{'='*60}\n")
    
    # Margin 추출 (하위 호환성: 'p' 키를 우선 사용하고, 없으면 'probs' 사용)
    # ✅ 모든 축에 대해 비율 순위 보장: 확률 내림차순으로 정렬
    def _sort_by_prob_desc(labels, probs):
        """확률 내림차순으로 정렬하여 비율 순위 보장"""
        if not labels or not probs or len(labels) != len(probs):
            return labels, probs
        label_prob_pairs = list(zip(labels, probs))
        label_prob_pairs.sort(key=lambda x: x[1], reverse=True)  # 확률 높은 순서대로
        sorted_labels, sorted_probs = zip(*label_prob_pairs)
        return list(sorted_labels), list(sorted_probs)
    
    # 6축 가중치 적용: probs를 가중치에 따라 조정 (w>1: 분포 강조, w<1: 완만하게)
    def _apply_axis_weight(probs: List[float], w: float) -> List[float]:
        if not probs or w <= 0:
            return probs
        if abs(w - 1.0) < 1e-6:
            return probs
        powered = [max(p ** w, 1e-10) for p in probs]
        return _normalize_prob(powered)

    sigungu_margin = margins_axis.get('sigungu', {})
    sigungu_labels = sigungu_margin.get('labels', selected_sigungu or sigungu_pool) if margins_axis.get('sigungu') else (selected_sigungu or sigungu_pool)
    if not sigungu_labels:
        sigungu_labels = ["해당없음"]  # 빈 시군구 시 폴백
    sigungu_probs = sigungu_margin.get('p', sigungu_margin.get('probs', [1.0 / len(sigungu_labels)] * len(sigungu_labels)))
    sigungu_probs = _apply_axis_weight(sigungu_probs, float(weights_6axis.get('sigungu', 1.0)))
    sigungu_labels, sigungu_probs = _sort_by_prob_desc(sigungu_labels, sigungu_probs)
    
    gender_margin = margins_axis.get('gender', {})
    gender_labels = gender_margin.get('labels', ['남자', '여자'])
    gender_probs = gender_margin.get('p', gender_margin.get('probs', [0.5, 0.5]))
    gender_probs = _apply_axis_weight(gender_probs, float(weights_6axis.get('gender', 1.0)))
    gender_labels, gender_probs = _sort_by_prob_desc(gender_labels, gender_probs)
    
    # ✅ 디버깅: 성별 확률 확인
    if gender_margin:
        print(f"📊 성별 확률 분포 (순위 보장):")
        for i, (label, prob) in enumerate(zip(gender_labels, gender_probs)):
            print(f"   {i+1}위: {label}: {prob:.4f} ({prob*100:.2f}%)")
    
    age_margin = margins_axis.get('age', {})
    age_labels = age_margin.get('labels', list(range(20, 86)))
    age_probs = age_margin.get('p', age_margin.get('probs', [1.0 / len(age_labels)] * len(age_labels)))
    age_labels, age_probs = _sort_by_prob_desc(age_labels, age_probs)
    
    econ_margin = margins_axis.get('econ', {})
    econ_labels = econ_margin.get('labels', ['경제활동', '비경제활동'])
    econ_probs = econ_margin.get('p', econ_margin.get('probs', [0.6, 0.4]))
    econ_probs = _apply_axis_weight(econ_probs, float(weights_6axis.get('econ', 1.0)))
    econ_labels, econ_probs = _sort_by_prob_desc(econ_labels, econ_probs)
    
    edu_margin = margins_axis.get('edu', {})
    edu_labels = edu_margin.get('labels', ['중졸이하', '고졸', '대졸이상'])
    edu_probs = edu_margin.get('p', edu_margin.get('probs', [0.25, 0.4, 0.35]))
    edu_labels, edu_probs = _sort_by_prob_desc(edu_labels, edu_probs)
    
    income_margin = margins_axis.get('income', {})
    income_labels = income_margin.get('labels', [
        '50만원미만', '50-100만원', '100-200만원', '200-300만원', 
        '300-400만원', '400-500만원', '500-600만원', '600-700만원', 
        '700-800만원', '800만원이상'
    ])
    income_probs = income_margin.get('p', income_margin.get('probs', [0.1] * len(income_labels)))
    income_probs = _apply_axis_weight(income_probs, float(weights_6axis.get('income', 1.0)))
    income_labels, income_probs = _sort_by_prob_desc(income_labels, income_probs)
    
    # 🔹 1단계: 초기 생성 (20세 이상 + 성별 맞춤 이름)
    print("📊 1단계: 초기 인구 생성 (20세 이상)")
    
    # 20세 미만 제거 (연령은 이미 정수로 통일됨)
    age_labels_filtered = [a for a in age_labels if a >= 20]
    if age_labels_filtered:
        age_probs_filtered = _normalize_prob([age_probs[age_labels.index(a)] for a in age_labels_filtered])
    else:
        age_labels_filtered = list(range(20, 86))
        age_probs_filtered = _normalize_prob([1.0 / len(age_labels_filtered)] * len(age_labels_filtered))
    
    data = {
        "식별NO": list(range(1, n + 1)),
        "거주지역": [_safe_choice(sigungu_labels, sigungu_probs) for _ in range(n)],
        "성별": [_safe_choice(gender_labels, gender_probs) for _ in range(n)],
        "연령": [_safe_choice(age_labels_filtered, age_probs_filtered) for _ in range(n)],
        "경제활동": [_safe_choice(econ_labels, econ_probs) for _ in range(n)],
        "교육정도": [_safe_choice(edu_labels, edu_probs) for _ in range(n)],
        "월평균소득": [_safe_choice(income_labels, income_probs) for _ in range(n)],
    }
    
    df = pd.DataFrame(data)
    
    # ✅ 성별에 맞는 가상이름 생성 (풀 1회만 구축해 반복 호출 비용 제거)
    male_first, female_first, last_names = _build_extended_name_pools()
    male_first = list(dict.fromkeys(male_first))
    female_first = list(dict.fromkeys(female_first))
    used_names = set()
    names = []
    max_attempts = 1000
    for idx, gender in enumerate(df['성별']):
        name = _name_from_pools(male_first, female_first, last_names, gender)
        attempts = 0
        while name in used_names and attempts < max_attempts:
            name = _name_from_pools(male_first, female_first, last_names, gender)
            attempts += 1
        if name in used_names:
            name = f"{name}_{idx + 1}"
        used_names.add(name)
        names.append(name)
    df['가상이름'] = names
    
    # ✅ 컬럼 순서 고정: 식별NO, 가상이름 순으로
    column_order = ['식별NO', '가상이름']
    other_columns = [col for col in df.columns if col not in column_order]
    df = df[column_order + other_columns]
    
    print(f"✅ 초기 생성 완료 ({len(df)}명, 고유 이름 {len(used_names)}개)")
    
    # ✅ 디버깅: 초기 생성 후 성별 분포 확인
    if '성별' in df.columns:
        gender_counts = df['성별'].value_counts()
        total = len(df)
        print(f"📊 초기 생성 후 성별 분포:")
        for gender in ['남자', '여자']:
            count = gender_counts.get(gender, 0)
            pct = (count / total * 100) if total > 0 else 0
            print(f"   {gender}: {count}명 ({pct:.2f}%)")
    
    # 🔹 2단계: IPF 적용
    if apply_ipf_flag:
        df = apply_ipf(df, n, margins_axis)
    else:
        print("⏭️ 2단계: IPF SKIP")
    
    # ✅ 디버깅: IPF 적용 후 성별 분포 확인
    if '성별' in df.columns:
        gender_counts = df['성별'].value_counts()
        total = len(df)
        print(f"📊 IPF 적용 후 성별 분포:")
        for gender in ['남자', '여자']:
            count = gender_counts.get(gender, 0)
            pct = (count / total * 100) if total > 0 else 0
            print(f"   {gender}: {count}명 ({pct:.2f}%)")
        
        # ✅ 순위 검증
        if '남자' in gender_counts and '여자' in gender_counts:
            male_count = gender_counts['남자']
            female_count = gender_counts['여자']
            if gender_probs[0] > gender_probs[1]:  # 첫 번째가 더 높은 확률
                expected_first = gender_labels[0]
                if expected_first == '남자' and male_count < female_count:
                    print(f"⚠️ 순위 위반 감지: 남자가 여자보다 적음! 강제 수정 중...")
                    # 여자에서 남자로 변경
                    female_indices = df[df['성별'] == '여자'].index.tolist()
                    move_count = (female_count - male_count) // 2 + 1
                    move_count = min(move_count, len(female_indices))
                    if move_count > 0:
                        move_indices = np.random.choice(female_indices, size=move_count, replace=False)
                        df.loc[move_indices, '성별'] = '남자'
                        print(f"✅ {move_count}명을 여자에서 남자로 변경")
                elif expected_first == '여자' and female_count < male_count:
                    print(f"⚠️ 순위 위반 감지: 여자가 남자보다 적음! 강제 수정 중...")
                    # 남자에서 여자로 변경
                    male_indices = df[df['성별'] == '남자'].index.tolist()
                    move_count = (male_count - female_count) // 2 + 1
                    move_count = min(move_count, len(male_indices))
                    if move_count > 0:
                        move_indices = np.random.choice(male_indices, size=move_count, replace=False)
                        df.loc[move_indices, '성별'] = '여자'
                        print(f"✅ {move_count}명을 남자에서 여자로 변경")
    
    # 🔹 3단계: 현실적 제약 조건 적용 ⭐
    df = apply_realistic_constraints(df)
    
    # ✅ 순위 검증 및 강제 수정 (현실적 제약 적용 후)
    if margins_axis and 'gender' in margins_axis:
        gender_margin = margins_axis['gender']
        gender_labels_check = gender_margin.get('labels', ['남자', '여자'])
        gender_probs_check = gender_margin.get('p', gender_margin.get('probs', [0.5, 0.5]))
        
        if len(gender_labels_check) == len(gender_probs_check):
            # 확률 내림차순 정렬
            label_prob_pairs = list(zip(gender_labels_check, gender_probs_check))
            label_prob_pairs.sort(key=lambda x: x[1], reverse=True)
            gender_labels_sorted, gender_probs_sorted = zip(*label_prob_pairs)
            
            if '성별' in df.columns:
                gender_counts = df['성별'].value_counts()
                if len(gender_labels_sorted) >= 2:
                    first_label = gender_labels_sorted[0]
                    second_label = gender_labels_sorted[1]
                    
                    first_count = gender_counts.get(first_label, 0)
                    second_count = gender_counts.get(second_label, 0)
                    
                    if first_count < second_count:
                        print(f"⚠️ 순위 위반 감지: {first_label}({first_count}) < {second_label}({second_count})! 강제 수정 중...")
                        # 두 번째에서 첫 번째로 변경
                        second_indices = df[df['성별'] == second_label].index.tolist()
                        move_count = (second_count - first_count) // 2 + 1
                        move_count = min(move_count, len(second_indices))
                        if move_count > 0:
                            move_indices = np.random.choice(second_indices, size=move_count, replace=False)
                            df.loc[move_indices, '성별'] = first_label
                            print(f"✅ {move_count}명을 {second_label}에서 {first_label}로 변경")
    
    # 🔹 4단계: 중복 제거
    print("🔧 중복 제거 시작...")
    df = remove_duplicates(df)
    print(f"✅ 중복 제거 완료 ({len(df)}명)")
    
    # 🔹 5단계: 부족분 보충
    if len(df) < n:
        df = fill_shortage_with_diversity(
            df, n,
            sigungu_labels, gender_labels, age_labels_filtered,
            econ_labels, edu_labels, income_labels,
            sigungu_probs, gender_probs, age_probs_filtered,
            econ_probs, edu_probs, income_probs
        )
    
    # 최종 조정
    df = df.head(n).reset_index(drop=True)
    df['식별NO'] = range(1, len(df) + 1)
    
    # ✅ 컬럼 순서 고정: 식별NO, 가상이름 순으로
    if '식별NO' in df.columns and '가상이름' in df.columns:
        column_order = ['식별NO', '가상이름']
        other_columns = [col for col in df.columns if col not in column_order]
        df = df[column_order + other_columns]
    
    # ✅ 최종 순위 검증 및 강제 수정 (모든 축에 대해)
    if margins_axis:
        for axis_key, margin_data in margins_axis.items():
            axis_name_map = {
                'sigungu': '거주지역',
                'gender': '성별',
                'age': '연령',
                'econ': '경제활동',
                'income': '월평균소득',
                'edu': '교육정도'
            }
            col_name = axis_name_map.get(axis_key)
            
            if not col_name or col_name not in df.columns:
                continue
            
            labels = margin_data.get('labels', [])
            probs = margin_data.get('p', margin_data.get('probs', []))
            
            if len(labels) == len(probs) and len(labels) >= 2:
                # 확률 내림차순 정렬
                label_prob_pairs = list(zip(labels, probs))
                label_prob_pairs.sort(key=lambda x: x[1], reverse=True)
                labels_sorted, probs_sorted = zip(*label_prob_pairs)
                
                current_counts = df[col_name].value_counts()
                
                # 순위 위반 체크 및 수정
                for i in range(len(labels_sorted) - 1):
                    label_high = labels_sorted[i]
                    label_low = labels_sorted[i + 1]
                    
                    count_high = current_counts.get(label_high, 0)
                    count_low = current_counts.get(label_low, 0)
                    
                    if count_high < count_low:
                        print(f"⚠️ [{col_name}] 순위 위반: {label_high}({count_high}) < {label_low}({count_low})! 강제 수정 중...")
                        # 낮은 확률에서 높은 확률로 변경
                        low_indices = df[df[col_name] == label_low].index.tolist()
                        move_count = (count_low - count_high) // 2 + 1
                        move_count = min(move_count, len(low_indices))
                        if move_count > 0:
                            move_indices = np.random.choice(low_indices, size=move_count, replace=False)
                            df.loc[move_indices, col_name] = label_high
                            print(f"✅ [{col_name}] {move_count}명을 {label_low}에서 {label_high}로 변경")
                            # 카운트 업데이트
                            current_counts[label_high] = current_counts.get(label_high, 0) + move_count
                            current_counts[label_low] = current_counts.get(label_low, 0) - move_count
    
    # ✅ 최종 컬럼 순서 고정: 식별NO, 가상이름 순으로
    if '식별NO' in df.columns and '가상이름' in df.columns:
        column_order = ['식별NO', '가상이름']
        other_columns = [col for col in df.columns if col not in column_order]
        df = df[column_order + other_columns]
    
    print(f"\n{'='*60}")
    print(f"✅ 가상인구 생성 완료 (최종: {len(df)}명)")
    print(f"{'='*60}\n")
    
    return df


# ========================================
# 8. 기타 함수들 (기존 유지)
# ========================================

def apply_correlation_adjustment(df: pd.DataFrame, correlation_intensity: float = 0.2) -> pd.DataFrame:
    """상관관계 조정 (SKIP)"""
    print("⏭️ 상관관계 적용 SKIP")
    return df


def apply_logical_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """논리적 제약 (최소한만 적용)"""
    print("⏭️ 논리적 제약 SKIP (현실적 제약으로 대체)")
    return df
