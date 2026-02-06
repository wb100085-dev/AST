# -*- coding: utf-8 -*-
"""Cursor assets에 저장된 FGI/FGD/IDI 이미지를 프로젝트 assets/interview_methods/로 복사합니다."""
import os
import shutil

# 프로젝트 루트 (이 스크립트는 scripts/ 에 있음)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST_DIR = os.path.join(ROOT, "assets", "interview_methods")
os.makedirs(DST_DIR, exist_ok=True)

# Cursor 프로젝트 assets 경로 (이미지가 저장된 위치)
CURSOR_ASSETS = os.path.join(
    os.path.expanduser("~"),
    ".cursor", "projects", "c-Users-user-Desktop-virtual-population-generator", "assets"
)

# 사용자가 첨부한 순서: 1.FGI, 2.FGD, 3.IDI (파일명 패턴으로 구분)
MAPPING = [
    ("fgi.png", "12fff090-b481-49ca-a1dc-024182fb82c4"),   # 1번
    ("fgd.png", "__1_-4f91d987-e6cd-4b90-8178-2ae252acc536"),  # 2번
    ("idi.png", "__2_-a0722578-8a4c-4500-beff-3313d4866015"),  # 3번
]

if not os.path.isdir(CURSOR_ASSETS):
    print("Cursor assets 폴더가 없습니다:", CURSOR_ASSETS)
    exit(1)

for dst_name, pattern in MAPPING:
    dst_path = os.path.join(DST_DIR, dst_name)
    found = None
    for f in os.listdir(CURSOR_ASSETS):
        if pattern in f and f.endswith(".png"):
            found = os.path.join(CURSOR_ASSETS, f)
            break
    if found and os.path.isfile(found):
        shutil.copy2(found, dst_path)
        print("OK:", dst_name, "<-", os.path.basename(found))
    else:
        print("NOT FOUND:", dst_name, "(pattern:", pattern, ")")

print("대상 폴더:", DST_DIR)
print("파일 목록:", os.listdir(DST_DIR))
