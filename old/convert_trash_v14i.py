#!/usr/bin/env python3
"""
Trash Detection.v14i.yolov8 데이터셋만 7클래스 YOLO 형식으로 변환합니다.
OBB(회전 박스) 라벨을 일반 bbox로 변환하고 별도 폴더에 저장합니다.
"""

import shutil
from pathlib import Path
import yaml

TRASH_DATASET_DIR = r"C:\Users\aerty\Desktop\전공\Trash Detection.v14i.yolov8"
OUTPUT_DIR = r"C:\Users\aerty\Desktop\전공\Trash Detection v14i Simplified"

NEW_CLASSES = ["Paper", "Plastic", "Film", "Metal", "Glass", "Organic", "Other"]

# Trash Detection 5클래스 → 통합 7클래스
TRASH_CLASS_MAPPING = {
    0: 4,  # Glass  → Glass
    1: 3,  # Metal  → Metal
    2: 0,  # Paper  → Paper
    3: 1,  # Plastic → Plastic
    4: 6,  # Waste  → Other
}


def obb_to_bbox(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    return x_min + w / 2, y_min + h / 2, w, h


def convert_label_file(src_path, dst_path):
    new_lines = []
    with open(src_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            old_class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            new_class_id = TRASH_CLASS_MAPPING.get(old_class_id, 6)

            if len(coords) == 4:
                x_c, y_c, w, h = coords
            else:
                x_c, y_c, w, h = obb_to_bbox(coords)

            new_lines.append(f"{new_class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    with open(dst_path, "w") as f:
        f.writelines(new_lines)


def convert_split(split):
    src_images = Path(TRASH_DATASET_DIR) / split / "images"
    src_labels = Path(TRASH_DATASET_DIR) / split / "labels"
    dst_images = Path(OUTPUT_DIR) / split / "images"
    dst_labels = Path(OUTPUT_DIR) / split / "labels"

    if not src_images.exists():
        print(f"  skip: {split} (폴더 없음)")
        return 0

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_files = list(src_images.glob("*.jpg")) + list(src_images.glob("*.png"))
    for img_file in image_files:
        shutil.copy2(img_file, dst_images / img_file.name)
        label_file = src_labels / f"{img_file.stem}.txt"
        if label_file.exists():
            convert_label_file(label_file, dst_labels / label_file.name)

    print(f"  {split}: {len(image_files)}개")
    return len(image_files)


def create_data_yaml():
    data = {
        "path": str(Path(OUTPUT_DIR).absolute()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(NEW_CLASSES),
        "names": NEW_CLASSES,
    }
    yaml_path = Path(OUTPUT_DIR) / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


def main():
    print("=" * 70)
    print("  Trash Detection v14i -> 7클래스 변환")
    print("=" * 70)

    if not Path(TRASH_DATASET_DIR).exists():
        print(f"오류: 데이터셋을 찾을 수 없습니다: {TRASH_DATASET_DIR}")
        return

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"원본: {TRASH_DATASET_DIR}")
    print(f"출력: {OUTPUT_DIR}\n")

    total = 0
    for split in ["train", "valid", "test"]:
        total += convert_split(split)

    yaml_path = create_data_yaml()
    print(f"\n완료: 총 {total}개 이미지")
    print(f"data.yaml: {yaml_path}")
    print("\n학습: python train.py --dataset v14i")


if __name__ == "__main__":
    main()
