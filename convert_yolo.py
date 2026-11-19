#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Hub '085.생활 폐기물 이미지' 라벨(JSON)을 YOLO 포맷(.txt)으로 변환.

- 종이류 -> 종이 (class_id 2)
- 플라스틱류 -> 플라스틱 (class_id 1)
- 유리병 -> 병 (class_id 0)
- 가구류/고철류/나무/도기류/비닐/스티로폼/의류/자전거/전자제품/캔류/페트병/형광등
  (각 1개 하위항목만 다운로드됨) -> 일반쓰레기 (class_id 3)

이미지 파일은 복사하지 않고 원본 위치 그대로 두고, 같은 폴더에 같은 이름의
.txt 라벨 파일만 새로 만듭니다. 그리고 train.txt / val.txt 에 이번에 처리된
이미지들의 절대경로 목록을 저장합니다 (Ultralytics YOLO가 바로 읽을 수 있는 형식).

사용법:
    python convert_to_yolo.py

실행 전 확인:
    - pip install pillow 가 되어 있어야 합니다 (이미지 실제 크기 확인용).
      pip install pillow --break-system-packages   (또는 그냥 pip install pillow)
"""

import json
import os
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️  Pillow가 설치되어 있지 않습니다. RESOLUTION 필드값을 대신 사용합니다.")
    print("    정확도를 높이려면: pip install pillow")

# ---- 설정 ----
BASE_DIR = Path(".")  # 이 스크립트를 /d/waste-dataset 에서 실행한다고 가정
ROOT = BASE_DIR / "085.생활_폐기물_이미지" / "01.데이터"

SPLITS = {
    "1.Training": "train",
    "2.Validation": "val",
}

# AI Hub 대분류명 -> 새 클래스 이름/ID
CLASS_MAP = {
    "유리병": (0, "병"),
    "플라스틱류": (1, "플라스틱"),
    "종이류": (2, "종이"),
    # 아래 12개는 각각 1개 하위항목만 받아서 '일반쓰레기'로 통합
    "가구류": (3, "일반쓰레기"),
    "고철류": (3, "일반쓰레기"),
    "나무": (3, "일반쓰레기"),
    "도기류": (3, "일반쓰레기"),
    "비닐": (3, "일반쓰레기"),
    "스티로폼": (3, "일반쓰레기"),
    "의류": (3, "일반쓰레기"),
    "자전거": (3, "일반쓰레기"),
    "전자제품": (3, "일반쓰레기"),
    "캔류": (3, "일반쓰레기"),
    "페트병": (3, "일반쓰레기"),
    "형광등": (3, "일반쓰레기"),
}

# 라벨링데이터 압축을 푼 하위 폴더 이름 (사용자가 unzip -d 로 만든 폴더명)
LABEL_UNZIP_DIRNAME = "풀린데이터"

# JSON 안의 "CLASS" 값은 폴더 이름과 다를 수 있습니다.
# (실제로 '유리병' 폴더의 CLASS 값은 '유리병류' 였고, 그래서 병 라벨이 전부 버려졌었음)
# 그래서 병/플라스틱/종이는 별칭 목록으로 판정하고, 그 외 CLASS 는 전부 일반쓰레기(3)로 처리합니다.
# (AI Hub 15개 분류 = 이 3개 + 일반쓰레기로 묶은 12개이므로 결과는 같음)
TARGET_CLASS_ALIASES = {
    0: {"유리병", "유리병류", "유리류", "유리"},
    1: {"플라스틱", "플라스틱류"},
    2: {"종이", "종이류"},
}
GENERAL_CLASS_ID = 3

# 어떤 CLASS 문자열이 어느 클래스로 분류됐는지 세어서 마지막에 보여줍니다 (매핑 실수를 바로 발견하기 위함)
class_value_counts = Counter()


def class_id_for(raw_class):
    """JSON 의 CLASS 값 -> 클래스 번호. 비어 있으면 None."""
    raw = (raw_class or "").strip()
    if not raw:
        return None
    for class_id, aliases in TARGET_CLASS_ALIASES.items():
        if raw in aliases:
            class_value_counts[(raw, class_id)] += 1
            return class_id
    class_value_counts[(raw, GENERAL_CLASS_ID)] += 1
    return GENERAL_CLASS_ID

# ---- 통계 ----
stats = {
    "processed_json": 0,
    "skipped_no_image": 0,
    "skipped_no_bbox": 0,
    "skipped_bad_class": 0,
    "total_boxes_written": 0,
}


def get_image_size(image_path: Path, fallback_resolution: str):
    """실제 이미지 크기를 얻는다. Pillow 있으면 실제로 열어서 확인, 없으면 RESOLUTION 필드 사용."""
    if HAS_PIL:
        try:
            with Image.open(image_path) as img:
                return img.size  # (width, height)
        except Exception:
            pass
    # fallback: "1921*1081" 같은 문자열 파싱
    try:
        w_str, h_str = fallback_resolution.lower().split("*")
        return int(w_str), int(h_str)
    except Exception:
        return None


def build_image_index(image_root: Path):
    """image_root 아래 모든 이미지 파일을 한 번 훑어서 {파일명: 경로} 인덱스를 만든다."""
    index = {}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
    for p in image_root.rglob("*"):
        if p.is_file() and p.suffix in exts:
            index[p.name] = p
            # 확장자 대소문자/종류가 라벨과 다를 경우 대비, stem으로도 인덱싱
            index.setdefault(p.stem, p)
    return index


def convert_split(training_key: str, split_name: str):
    split_root = ROOT / training_key
    label_root = split_root / "라벨링데이터" / LABEL_UNZIP_DIRNAME
    image_root = split_root / "원천데이터"

    if not label_root.exists():
        print(f"⚠️  라벨 폴더 없음: {label_root} - 건너뜁니다 (unzip 먼저 해주세요)")
        return []

    manifest_lines = []

    for category_name, (class_id, _) in CLASS_MAP.items():
        category_label_dir = label_root / category_name
        category_image_dir = image_root / category_name
        if not category_label_dir.exists():
            print(f"⚠️  카테고리 라벨 폴더 없음: {category_label_dir}")
            continue
        if not category_image_dir.exists():
            print(f"⚠️  카테고리 이미지 폴더 없음: {category_image_dir}")
            continue

        print(f"[{split_name}] {category_name}: 이미지 인덱스 구축 중...")
        image_index = build_image_index(category_image_dir)
        print(f"[{split_name}] {category_name}: 이미지 {len(image_index)}개 인덱싱 완료")

        json_files = list(category_label_dir.rglob("*.[Jj][Ss][Oo][Nn]"))
        print(f"[{split_name}] {category_name}: JSON {len(json_files)}개 발견")

        for json_path in json_files:
            stats["processed_json"] += 1
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  ❌ JSON 읽기 실패: {json_path} ({e})")
                continue

            bounding_list = data.get("Bounding", [])
            if not bounding_list:
                stats["skipped_no_bbox"] += 1
                continue

            file_name = data.get("FILE NAME")
            if not file_name:
                stats["skipped_no_image"] += 1
                continue

            image_path = image_index.get(file_name) or image_index.get(Path(file_name).stem)
            if image_path is None:
                stats["skipped_no_image"] += 1
                continue

            label_txt_path = image_path.with_suffix(".txt")
            if label_txt_path.exists():
                # 이전 실행에서 이미 처리된 이미지 - 건너뛰고 manifest에만 추가
                manifest_lines.append(str(image_path.resolve()))
                stats["already_done_skip"] = stats.get("already_done_skip", 0) + 1
                continue

            size = get_image_size(image_path, data.get("RESOLUTION", ""))
            if size is None:
                print(f"  ❌ 이미지 크기 확인 실패: {image_path}")
                stats["skipped_bad_size"] = stats.get("skipped_bad_size", 0) + 1
                continue
            img_w, img_h = size
            if img_w <= 0 or img_h <= 0:
                stats["skipped_bad_size"] = stats.get("skipped_bad_size", 0) + 1
                continue

            yolo_lines = []
            for box in bounding_list:
                box_class_id = class_id_for(box.get("CLASS", ""))
                if box_class_id is None:
                    # CLASS 값이 비어 있는 박스는 무시
                    stats["skipped_bad_class"] += 1
                    continue
                try:
                    x1 = float(box["x1"])
                    y1 = float(box["y1"])
                    x2 = float(box["x2"])
                    y2 = float(box["y2"])
                except (KeyError, ValueError):
                    continue

                # 좌표 정렬 보정 (혹시 x1>x2 인 경우 대비)
                x_min, x_max = sorted((x1, x2))
                y_min, y_max = sorted((y1, y2))

                x_center = ((x_min + x_max) / 2) / img_w
                y_center = ((y_min + y_max) / 2) / img_h
                w = (x_max - x_min) / img_w
                h = (y_max - y_min) / img_h

                # 0~1 범위로 클리핑
                x_center = min(max(x_center, 0.0), 1.0)
                y_center = min(max(y_center, 0.0), 1.0)
                w = min(max(w, 0.0), 1.0)
                h = min(max(h, 0.0), 1.0)

                yolo_lines.append(f"{box_class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            if not yolo_lines:
                stats["skipped_zero_valid_boxes"] = stats.get("skipped_zero_valid_boxes", 0) + 1
                continue

            # 라벨 .txt 파일을 이미지와 같은 폴더에, 같은 basename으로 저장
            label_txt_path = image_path.with_suffix(".txt")
            with open(label_txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + "\n")

            stats["total_boxes_written"] += len(yolo_lines)
            manifest_lines.append(str(image_path.resolve()))

    return manifest_lines


def main():
    all_manifests = {}
    for training_key, split_name in SPLITS.items():
        print(f"\n===== {split_name} 처리 시작 =====")
        lines = convert_split(training_key, split_name)
        all_manifests[split_name] = lines

        manifest_path = BASE_DIR / f"{split_name}.txt"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"✅ {split_name}.txt 저장 완료 ({len(lines)}개 이미지)")

    print("\n===== 전체 통계 =====")
    for k, v in stats.items():
        print(f"{k}: {v}")

    class_names = {0: "병", 1: "플라스틱", 2: "종이", 3: "일반쓰레기"}
    print("\n===== JSON 의 CLASS 값이 어느 클래스로 분류됐는지 (박스 개수) =====")
    per_class_total = Counter()
    for (raw, class_id), n in sorted(class_value_counts.items(), key=lambda x: (x[0][1], -x[1])):
        print(f"  {raw!r:>14} -> {class_id} ({class_names[class_id]}): {n}")
        per_class_total[class_id] += n
    print("\n----- 클래스별 박스 합계 -----")
    for class_id in sorted(class_names):
        print(f"  {class_id} ({class_names[class_id]}): {per_class_total[class_id]}")
    if per_class_total[0] == 0:
        print("\n⚠️  병(0) 박스가 하나도 없습니다. 위의 CLASS 값 목록에서 유리병에 해당하는 이름을 확인하세요.")


if __name__ == "__main__":
    main()