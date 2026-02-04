"""
AI Social Twin - 공통 상수 및 마스터 데이터
app.py에서 분리
"""
import os
import tempfile

# 프로젝트 루트 기준 경로 (core/constants.py 기준 상위 2단계 = 프로젝트 루트)
_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_APP_ROOT, "data")


def _writable_data_dir():
    """쓰기 가능한 데이터 디렉터리 반환. Streamlit Cloud 등 읽기 전용 환경에서는 /tmp 사용."""
    if os.path.exists(_DATA_DIR):
        if os.access(_DATA_DIR, os.W_OK):
            return _DATA_DIR
    else:
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            if os.access(_DATA_DIR, os.W_OK):
                return _DATA_DIR
        except (OSError, PermissionError):
            pass
    # 읽기 전용(또는 생성 불가)이면 임시 디렉터리 사용
    _fallback = os.path.join(tempfile.gettempdir(), "ast_data")
    os.makedirs(_fallback, exist_ok=True)
    return _fallback


_DATA_DIR_EFFECTIVE = _writable_data_dir()
DB_PATH = os.path.join(_DATA_DIR_EFFECTIVE, "app.db")
AUTOSAVE_PATH = os.path.join(_DATA_DIR_EFFECTIVE, "temp_step1_autosave.pkl")

APP_TITLE = "AI Social Twin"

EXPORT_SHEET_NAME = "통계목록"
EXPORT_COLUMNS = ["카테고리", "통계명", "URL", "활성여부"]

# 2단계 생성 결과 컬럼명 최종 표기 (내부명 → 출력명)
STEP2_COLUMN_RENAME = {
    "부모님 생존 여부": "부모님 생존 여부",
    "하는일": "하는일 만족도",
    "임금/가구소득": "임금/가구소득 만족도",
    "근로시간": "근로시간 만족도",
    "근무환경": "근무환경 만족도",
    "전반적인 만족도": "근무 여건 전반적인 만족도",
    "자녀유무": "학생 및 미취학 자녀 유무",
    "총학생수": "학생 및 미취학 자녀 수",
    "소비 경험 여부": "경북 외 소비 경험 여부",
    "임신·출산·육아에 대한 복지": "임신·출산·육아에 대한 복지 만족도",
    "저소득층 등 취약계층에 대한 복지": "저소득층 등 취약계층에 대한 복지 만족도",
    "동네": "동네 소속감",
    "시군": "시군 소속감",
    "경상북도": "경상북도 소속감",
    "어둡고 후미진 곳이 많다": "(안전환경)어둡고 후미진 곳이 많다",
    "주변에 쓰레기가 아무렇게 버려져 있고 지저분 하다": "(안전환경)주변에 쓰레기가 아무렇게 버려져 있고 지저분 하다",
    "주변에 방치된 차나 빈 건물이 많다": "(안전환경)주변에 방치된 차나 빈 건물이 많다",
    "무리 지어 다니는 불량 청소년이 많다": "(안전환경)무리 지어 다니는 불량 청소년이 많다",
    "기초질서(무단횡단, 불법 주정차 등)를 지키지 않는 사람이 많다": "(안전환경)기초질서를 지키지 않는 사람이 많다",
    "큰소리로 다투거나 싸우는 사람들을 자주 볼 수 있다": "(안전환경)큰소리로 다투거나 싸우는 사람들을 자주 볼 수 있다",
    "나자신": "(일상생활 범죄피해 두려움)나자신",
    "배우자(애인)": "(일상생활 범죄피해 두려움)배우자(애인)",
    "자녀": "(일상생활 범죄피해 두려움)자녀",
    "부모": "(일상생활 범죄피해 두려움)부모",
    "밤에 혼자 집에 있을 때": "(일상생활에서 두려움)밤에 혼자 집에 있을 때",
    "밤에 혼자 지역(동네)의 골목길을 걸을때": "(일상생활에서 두려움)밤에 혼자 지역(동네)의 골목길을 걸을때",
    "대기": "대기환경 체감도",
    "수질": "수질환경 체감도",
    "토양": "토양환경 체감도",
    "소음/진동": "소음/진동환경 체감도",
    "녹지환경": "녹지환경 체감도",
    "불만족 이유": "여가활동 불만족 이유",
    "관람 여부": "문화예술행사 관람 여부",
    "관람 분야": "문화예술행사 관람 분야",
}

DEFAULT_WEIGHTS_SCORE = {
    "income": 0.50,
    "age": 0.20,
    "education": 0.15,
    "gender": 0.10,
    "random": 0.05,
}

# 시도 마스터
SIDO_MASTER = [
    {"sido_name": "전국", "sido_code": "00"},
    {"sido_name": "서울특별시", "sido_code": "11"},
    {"sido_name": "부산광역시", "sido_code": "21"},
    {"sido_name": "대구광역시", "sido_code": "22"},
    {"sido_name": "인천광역시", "sido_code": "23"},
    {"sido_name": "광주광역시", "sido_code": "24"},
    {"sido_name": "대전광역시", "sido_code": "25"},
    {"sido_name": "울산광역시", "sido_code": "26"},
    {"sido_name": "세종특별자치시", "sido_code": "29"},
    {"sido_name": "경기도", "sido_code": "31"},
    {"sido_name": "강원도", "sido_code": "32"},
    {"sido_name": "충청북도", "sido_code": "33"},
    {"sido_name": "충청남도", "sido_code": "34"},
    {"sido_name": "전라북도", "sido_code": "35"},
    {"sido_name": "전라남도", "sido_code": "36"},
    {"sido_name": "경상북도", "sido_code": "37"},
    {"sido_name": "경상남도", "sido_code": "38"},
    {"sido_name": "제주특별자치도", "sido_code": "39"},
]

SIDO_CODE = {item["sido_code"]: item["sido_name"] for item in SIDO_MASTER}
SIDO_NAME = {item["sido_code"]: item["sido_name"] for item in SIDO_MASTER}
SIDO_LABELS = [f"{x['sido_name']} ({x['sido_code']})" for x in SIDO_MASTER]
SIDO_LABEL_TO_CODE = {f"{x['sido_name']} ({x['sido_code']})": x["sido_code"] for x in SIDO_MASTER}
SIDO_CODE_TO_NAME = {x["sido_code"]: x["sido_name"] for x in SIDO_MASTER}

# 시도별 주민등록인구 (총인구) 참고용 - 행정안전부 주민등록 인구통계 기준 근사치
SIDO_TOTAL_POP = {
    "11": 9_330_000,   # 서울
    "21": 3_280_000,   # 부산
    "22": 2_380_000,   # 대구
    "23": 2_990_000,   # 인천
    "24": 1_430_000,   # 광주
    "25": 1_450_000,   # 대전
    "26": 1_100_000,   # 울산
    "29": 380_000,     # 세종
    "31": 13_550_000,  # 경기
    "32": 1_540_000,   # 강원
    "33": 1_600_000,   # 충북
    "34": 2_120_000,   # 충남
    "35": 1_780_000,   # 전북
    "36": 1_800_000,   # 전남
    "37": 2_600_000,   # 경북
    "38": 3_330_000,   # 경남
    "39": 680_000,     # 제주
}

AXIS_KEYS = ["sigungu", "gender", "age", "econ", "income", "edu"]
AXIS_LABELS = {
    "sigungu": "시군(거주지역)",
    "gender": "성별",
    "age": "연령",
    "econ": "경제활동",
    "income": "소득",
    "edu": "교육정도",
}
