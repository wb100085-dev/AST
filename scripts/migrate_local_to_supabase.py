"""
로컬 데이터 → Supabase 마이그레이션 스크립트

사용 조건:
- .env 또는 환경변수에 SUPABASE_URL, SUPABASE_KEY 설정
- (선택) 복구할 로컬 파일이 있어야 함:
  - data/app.db (예전 SQLite DB 백업)
  - data/step2_records/*.xlsx + *.json (2차 대입 결과만 있어도 가상인구 DB 복구 가능)

실행 (프로젝트 루트에서):
  python scripts/migrate_local_to_supabase.py
  python scripts/migrate_local_to_supabase.py --sqlite path/to/backup.db
"""
import argparse
import base64
import os
import sys
from datetime import datetime

# 프로젝트 루트를 path에 추가
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.dirname(_SCRIPT_DIR)
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def get_supabase_client():
    """환경변수로 Supabase 클라이언트 생성 (Streamlit 없이)."""
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        # .streamlit/secrets.toml 은 TOML이라 수동 로드
        secrets_path = os.path.join(_APP_ROOT, ".streamlit", "secrets.toml")
        if os.path.isfile(secrets_path):
            try:
                import toml
                secrets = toml.load(secrets_path)
                url = (secrets.get("SUPABASE_URL") or "").strip()
                key = (secrets.get("SUPABASE_KEY") or "").strip()
            except Exception:
                pass
    if not url or not key:
        print("오류: SUPABASE_URL, SUPABASE_KEY가 없습니다.")
        print("  .env 또는 .streamlit/secrets.toml 또는 환경변수에 설정하세요.")
        sys.exit(1)
    from supabase import create_client
    return create_client(url, key)


def migrate_from_sqlite(sb, sqlite_path: str) -> dict:
    """SQLite app.db 백업에서 Supabase로 이전. 통계·가상인구 DB·템플릿 등."""
    import sqlite3
    counts = {"stats": 0, "sido_axis_margin_stats": 0, "sido_templates": 0, "virtual_population_db": 0}
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1) stats
    try:
        cur.execute("SELECT id, sido_code, category, name, url, is_active FROM stats")
        for row in cur.fetchall():
            row = dict(row)
            existing = sb.table("stats").select("id").eq("sido_code", row["sido_code"]).eq("category", row["category"]).eq("name", row["name"]).limit(1).execute()
            if not (existing.data and len(existing.data) > 0):
                sb.table("stats").insert({
                    "sido_code": row["sido_code"],
                    "category": row["category"],
                    "name": row["name"],
                    "url": row["url"],
                    "is_active": row["is_active"] if row["is_active"] is not None else 1,
                }).execute()
                counts["stats"] += 1
    except sqlite3.OperationalError as e:
        print(f"  [stats] 테이블 없음 또는 오류: {e}")

    # 2) sido_axis_margin_stats
    try:
        cur.execute("SELECT id, sido_code, axis_key, stat_id FROM sido_axis_margin_stats")
        for row in cur.fetchall():
            row = dict(row)
            existing = sb.table("sido_axis_margin_stats").select("id").eq("sido_code", row["sido_code"]).eq("axis_key", row["axis_key"]).limit(1).execute()
            if not (existing.data and len(existing.data) > 0):
                sb.table("sido_axis_margin_stats").insert({
                    "sido_code": row["sido_code"],
                    "axis_key": row["axis_key"],
                    "stat_id": int(row["stat_id"]),
                }).execute()
                counts["sido_axis_margin_stats"] += 1
    except sqlite3.OperationalError as e:
        print(f"  [sido_axis_margin_stats] 테이블 없음 또는 오류: {e}")

    # 3) sido_templates (BLOB → base64 TEXT)
    try:
        cur.execute("SELECT sido_code, filename, file_blob FROM sido_templates")
        for row in cur.fetchall():
            row = dict(row)
            blob = row["file_blob"]
            if blob is None:
                continue
            if isinstance(blob, bytes):
                b64 = base64.b64encode(blob).decode("ascii")
            else:
                b64 = blob
            existing = sb.table("sido_templates").select("sido_code").eq("sido_code", row["sido_code"]).limit(1).execute()
            payload = {"sido_code": row["sido_code"], "filename": row["filename"] or "", "file_blob": b64}
            if existing.data and len(existing.data) > 0:
                sb.table("sido_templates").update({"filename": payload["filename"], "file_blob": payload["file_blob"]}).eq("sido_code", row["sido_code"]).execute()
            else:
                sb.table("sido_templates").insert(payload).execute()
            counts["sido_templates"] += 1
    except sqlite3.OperationalError as e:
        print(f"  [sido_templates] 테이블 없음 또는 오류: {e}")

    # 4) virtual_population_db
    try:
        cur.execute("SELECT sido_code, sido_name, record_timestamp, record_excel_path, data_json FROM virtual_population_db ORDER BY added_at")
        for row in cur.fetchall():
            row = dict(row)
            existing = sb.table("virtual_population_db").select("id").eq("sido_code", row["sido_code"]).eq("record_timestamp", row["record_timestamp"] or "").eq("record_excel_path", row["record_excel_path"] or "").limit(1).execute()
            if not (existing.data and len(existing.data) > 0):
                sb.table("virtual_population_db").insert({
                    "sido_code": row["sido_code"],
                    "sido_name": row["sido_name"],
                    "record_timestamp": row["record_timestamp"] or "",
                    "record_excel_path": row["record_excel_path"] or "",
                    "data_json": row["data_json"] if row["data_json"] else "[]",
                }).execute()
                counts["virtual_population_db"] += 1
    except sqlite3.OperationalError as e:
        print(f"  [virtual_population_db] 테이블 없음 또는 오류: {e}")

    conn.close()
    return counts


def migrate_from_step2_records(sb, step2_dir: str) -> int:
    """data/step2_records/ 의 Excel+JSON만으로 가상인구 DB에 넣기 (SQLite 백업 없을 때)."""
    import json
    import pandas as pd
    count = 0
    if not os.path.isdir(step2_dir):
        return count
    for f in os.listdir(step2_dir):
        if not f.endswith(".json"):
            continue
        meta_path = os.path.join(step2_dir, f)
        excel_path = os.path.join(step2_dir, f.replace(".json", ".xlsx"))
        if not os.path.isfile(excel_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
        except Exception:
            continue
        sido_code = meta.get("sido_code", "")
        sido_name = meta.get("sido_name", "")
        timestamp = meta.get("timestamp", "")
        if not sido_code:
            continue
        existing = sb.table("virtual_population_db").select("id").eq("sido_code", sido_code).eq("record_timestamp", timestamp).eq("record_excel_path", excel_path).limit(1).execute()
        if existing.data and len(existing.data) > 0:
            continue
        try:
            df = pd.read_excel(excel_path, engine="openpyxl")
            data_json = df.to_json(orient="records", force_ascii=False)
        except Exception as e:
            print(f"  [skip] {excel_path}: {e}")
            continue
        sb.table("virtual_population_db").insert({
            "sido_code": sido_code,
            "sido_name": sido_name,
            "record_timestamp": timestamp,
            "record_excel_path": excel_path,
            "data_json": data_json,
        }).execute()
        count += 1
        print(f"  추가: {timestamp} {sido_name} ({len(df)}명)")
    return count


def main():
    parser = argparse.ArgumentParser(description="로컬 SQLite/step2_records → Supabase 마이그레이션")
    parser.add_argument("--sqlite", default=None, help="SQLite DB 경로 (기본: data/app.db)")
    parser.add_argument("--step2", default=None, help="2차 대입 결과 폴더 (기본: data/step2_records)")
    parser.add_argument("--dry-run", action="store_true", help="실제 INSERT 없이 확인만")
    args = parser.parse_args()

    sqlite_path = args.sqlite or os.path.join(_APP_ROOT, "data", "app.db")
    step2_dir = args.step2 or os.path.join(_APP_ROOT, "data", "step2_records")

    print("=== 로컬 → Supabase 마이그레이션 ===\n")
    sb = None if args.dry_run else get_supabase_client()
    if args.dry_run:
        print("[DRY RUN] 실제 전송 없이 경로만 확인합니다.\n")

    total = 0

    # 1) SQLite 백업이 있으면 이전
    if os.path.isfile(sqlite_path):
        print(f"1) SQLite: {sqlite_path}")
        if sb:
            counts = migrate_from_sqlite(sb, sqlite_path)
            for k, v in counts.items():
                if v > 0:
                    print(f"   {k}: {v}건 추가")
                    total += v
        else:
            print("   (dry-run) 건너뜀")
    else:
        print(f"1) SQLite 없음: {sqlite_path}")
        print("   → 백업한 app.db 가 있다면 해당 경로를 --sqlite 로 지정하세요.")

    # 2) step2_records 폴더가 있으면 가상인구 DB에만 넣기
    print(f"\n2) 2차 대입 결과 폴더: {step2_dir}")
    if os.path.isdir(step2_dir) and sb:
        n = migrate_from_step2_records(sb, step2_dir)
        total += n
        if n == 0:
            print("   (이미 Supabase에 있거나 .xlsx/.json 쌍 없음)")
    elif os.path.isdir(step2_dir) and args.dry_run:
        print("   (dry-run) 건너뜀")
    else:
        print("   폴더 없음.")

    print(f"\n완료. 총 {total}건 반영." if total else "\n반영할 데이터가 없었습니다.")
    if total == 0 and not os.path.isfile(sqlite_path) and not os.path.isdir(step2_dir):
        print("\n복구 방법:")
        print("  - 예전에 data/app.db 백업이 있다면 프로젝트의 data/ 에 넣고 다시 실행.")
        print("  - 2차 대입 결과 Excel을 data/step2_records/ 에 넣고, 동일한 이름의 .json 메타파일이 있으면 함께 인식됩니다.")
        print("  - Windows '이전 버전' 또는 백업에서 data 폴더를 복원해 보세요.")


if __name__ == "__main__":
    main()
