#!/usr/bin/env python3
"""
YOLO 쓰레기 탐지 모델 새 학습 + 현재 모델(v2)과 비교 스크립트

특징
  - 기존 결과를 절대 지우지 않음 (이미 있는 이름이면 학습을 거부함)
  - 시작 전에 현재 모델(v2 best.pt)을 models 폴더에 백업
  - 학습에 쓴 데이터 구성(이미지 수, 클래스별 인스턴스 수)을 기록
  - 학습이 끝나면 v2 와 새 모델을 "같은 val.txt" 로 다시 평가해서
    전체 지표 / 클래스별 지표 / 어떤 클래스로 잘못 나오는지(오분류)를 표로 비교

사용 예 (PowerShell 에서, 이 파일이 있는 폴더에서):
    python train_v3.py                         # v2 와 같은 출발점에서 40 epoch 학습 후 비교
    python train_v3.py --epochs 20             # 짧게 20 epoch
    python train_v3.py --start v2              # 현재 v2 모델에서 이어서 추가 학습
    python train_v3.py --start coco            # 예전 모델 없이 yolov8s 기본 가중치에서 시작
    python train_v3.py --name v3_try2          # 결과 폴더 이름 바꾸기 (다른 실험)
    python train_v3.py --resume                # 중단된 학습 이어서 하기
    python train_v3.py --compare-only --candidate "D:\\...\\best.pt"   # 학습 없이 두 모델 비교만
    python train_v3.py --counts-only           # 학습 없이 클래스별 데이터 개수만 확인
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# ── 경로 설정 ─────────────────────────────────────────
WASTE_DATASET_ROOT = Path(r"D:\waste-dataset")   # 결과·백업 저장 위치 (그대로)
SMALL_DATASET_ROOT = Path(r"C:\waste-small")     # 리사이즈한 학습 데이터
DATA_YAML = SMALL_DATASET_ROOT / "data.yaml"
TRAIN_MANIFEST = SMALL_DATASET_ROOT / "train.txt"
VAL_MANIFEST = SMALL_DATASET_ROOT / "val.txt"

RUNS_DIR = WASTE_DATASET_ROOT / "runs"
COMPARE_DIR = RUNS_DIR / "compare"
SNAPSHOT_DIR = RUNS_DIR / "snapshots"
MODELS_DIR = WASTE_DATASET_ROOT / "models"

# 예전 모델 (v2 의 출발점) - 절대 삭제하면 안 됨
OLD_BEST_MODEL_PATH = Path(r"C:\Users\aerty\Desktop\전공\runs\ultimate_v1i_4class\weights\best.pt")

# 현재 모델 (v2) 과 그 백업본
V2_BEST = RUNS_DIR / "v2_4class" / "weights" / "best.pt"
V2_BACKUP = MODELS_DIR / "v2_4class_best_backup.pt"

IMGSZ = 320  # Jetson 배포 규격과 같게 유지


# ── 공통 도우미 ───────────────────────────────────────
def load_names():
    """data.yaml 에서 {클래스번호: 이름} 읽기"""
    import yaml
    with open(DATA_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def img2label(img_path):
    """ultralytics 와 같은 규칙: /images/ -> /labels/ 로 바꾸고 확장자를 .txt 로"""
    sa, sb = os.sep + "images" + os.sep, os.sep + "labels" + os.sep
    return sb.join(img_path.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"


def ensure_v2_backup():
    """현재 v2 모델을 안전한 곳에 한 번 복사해 둔다 (이미 있으면 건드리지 않음)."""
    if V2_BACKUP.exists():
        print(f"📦 v2 백업 이미 있음: {V2_BACKUP}")
        return
    if not V2_BEST.exists():
        print(f"⚠️  v2 모델을 찾을 수 없어 백업을 건너뜁니다: {V2_BEST}")
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(V2_BEST, V2_BACKUP)
    print(f"📦 v2 모델 백업 완료: {V2_BACKUP}")


def resolve_start(start):
    if start == "old":
        return OLD_BEST_MODEL_PATH
    if start == "v2":
        return V2_BACKUP if V2_BACKUP.exists() else V2_BEST
    if start == "coco":
        return "yolov8s.pt"  # 처음 실행 시 자동 다운로드
    return Path(start)


# ── 학습 데이터 구성 기록 ─────────────────────────────
def count_instances(manifest_path):
    """목록 파일을 읽어서 (줄 수, 고유 이미지 수, 클래스별 인스턴스 수, 라벨 없는 이미지 수) 반환.
    목록에 같은 줄이 여러 번 있으면(오버샘플링) 그만큼 센다 = 실제 학습 때 보는 빈도."""
    with open(manifest_path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    repeats = Counter(lines)
    per_class = Counter()
    missing = 0
    for img, times in repeats.items():
        try:
            with open(img2label(img), encoding="utf-8") as f:
                for ln in f:
                    parts = ln.split()
                    if parts:
                        per_class[int(parts[0])] += times
        except OSError:
            missing += 1
    return len(lines), len(repeats), per_class, missing


def save_dataset_snapshot(name, names, skip_count):
    """이번 학습이 어떤 데이터로 이뤄졌는지 기록 (나중에 '이 모델은 뭘로 학습했지?' 를 알 수 있게)."""
    snap = SNAPSHOT_DIR / name
    snap.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TRAIN_MANIFEST, snap / "train.txt")
    shutil.copy2(VAL_MANIFEST, snap / "val.txt")
    shutil.copy2(DATA_YAML, snap / "data.yaml")

    info = {}
    if skip_count:
        print("ℹ️  --skip-count: 클래스별 인스턴스 수 집계는 건너뜁니다.")
    else:
        print("🔎 데이터 구성 집계 중... (라벨 파일을 읽는 중이라 1~3분 걸릴 수 있어요)")
        for label, path in (("train", TRAIN_MANIFEST), ("val", VAL_MANIFEST)):
            total, unique, per_class, missing = count_instances(path)
            info[label] = {
                "목록_줄수": total,
                "고유_이미지수": unique,
                "라벨_없는_이미지": missing,
                "클래스별_인스턴스": {names.get(c, str(c)): n for c, n in sorted(per_class.items())},
            }
        print("\n   [학습 데이터 구성]")
        for label in ("train", "val"):
            d = info[label]
            print(f"   {label}: 목록 {d['목록_줄수']}줄 / 고유 이미지 {d['고유_이미지수']}장 / 라벨 없음 {d['라벨_없는_이미지']}장")
            for cname, n in d["클래스별_인스턴스"].items():
                print(f"      - {cname}: {n}")
        print()

    with open(snap / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"🗂️  데이터 구성 기록 저장: {snap}")


# ── 학습 ──────────────────────────────────────────────
def train(args, names):
    from ultralytics import YOLO

    run_dir = RUNS_DIR / args.name
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"

    if args.resume:
        if not last_path.exists():
            print(f"❌ 이어서 할 last.pt 가 없습니다: {last_path}")
            return None
        print(f"⏯️  중단된 학습을 이어서 합니다: {last_path}")
        YOLO(str(last_path)).train(resume=True)
        return best_path

    if run_dir.exists():
        print(f"❌ 이미 '{args.name}' 결과 폴더가 있습니다: {run_dir}")
        print("   기존 결과를 지우지 않기 위해 학습을 시작하지 않습니다.")
        print("   --name 으로 다른 이름을 쓰거나, 이어서 하려면 --resume 을 쓰세요.")
        return None

    start = resolve_start(args.start)
    if isinstance(start, Path) and not start.exists():
        print(f"❌ 시작 가중치를 찾을 수 없습니다: {start}")
        return None

    save_dataset_snapshot(args.name, names, args.skip_count)

    print("\n" + "=" * 70)
    print(f"학습 시작: {args.name}")
    print("=" * 70)
    print(f"출발 가중치 : {start}")
    print(f"데이터      : {DATA_YAML}")
    print(f"epoch       : {args.epochs}" + (f"  (시간 제한 {args.time}시간)" if args.time else "  (시간 제한 없음)"))
    print(f"batch/imgsz : {args.batch} / {IMGSZ}")
    if args.time:
        print("※ time 을 쓰면 ultralytics 가 총 epoch 수를 시간에 맞춰 계속 다시 계산합니다 (1/20 → 2/36 …).")
        print("  이 경우 학습률 일정이 매번 늘어나서 다른 실험과 공정하게 비교하기 어려워요.")
    print()

    kwargs = dict(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=IMGSZ,
        batch=args.batch,
        patience=args.patience,
        device=0,
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.001,
        mixup=0.1,
        copy_paste=0.1,
        close_mosaic=5,
        warmup_epochs=3,
        save=True,
        save_period=-1,
        verbose=True,
        workers=4,
        project=str(RUNS_DIR),
        name=args.name,
        resume=False,
        exist_ok=False,
        plots=True,
        val=True,
    )
    if args.time:
        kwargs["time"] = args.time

    try:
        model = YOLO(str(start))
        model.train(**kwargs)
    except Exception as e:
        print(f"❌ 학습 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

    print("\n✅ 학습 완료!")
    print(f"   최고 성능 모델: {best_path}")
    return best_path


# ── 평가/비교 ─────────────────────────────────────────
def confusion_summary(matrix, names):
    """matrix[예측, 정답] (마지막 행/열 = 배경). 정답 클래스별로 어디로 갔는지 비율 계산."""
    nc = len(names)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (nc + 1, nc + 1):
        return None
    out = {}
    for t in range(nc):
        col = matrix[:, t]
        total = col.sum()
        if total <= 0:
            continue
        frac = col / total
        confused = [(names[p], float(frac[p])) for p in range(nc) if p != t and frac[p] > 0]
        confused.sort(key=lambda x: -x[1])
        out[t] = {"correct": float(frac[t]), "missed": float(frac[nc]), "confused": confused}
    return out


def evaluate(model_path, label, names):
    """같은 val.txt 로 모델을 평가해서 전체/클래스별/오분류 결과를 돌려준다."""
    from ultralytics import YOLO

    print(f"\n⏳ [{label}] 평가 중... ({model_path})")
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(DATA_YAML), split="val", imgsz=IMGSZ, batch=64, device=0,
        plots=True, project=str(COMPARE_DIR), name=f"val_{label}", exist_ok=True, verbose=False,
    )

    result = {
        "path": str(model_path),
        "overall": {
            "P": float(metrics.box.mp), "R": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map),
        },
        "per_class": {},
        "confusion": None,
    }
    for i, c in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        result["per_class"][int(c)] = {"P": float(p), "R": float(r), "mAP50": float(ap50), "mAP50-95": float(ap)}

    try:
        cm = metrics.confusion_matrix
        result["confusion"] = confusion_summary(cm.matrix, names)
    except Exception as e:
        print(f"ℹ️  혼동행렬 수치는 읽지 못했습니다 ({e}). 이미지(confusion_matrix.png)는 {COMPARE_DIR / ('val_' + label)} 에 있어요.")
    return result


def _fmt_delta(a, b):
    return f"{b - a:+.3f}"


def format_comparison(base, cand, names, base_label, cand_label):
    L = []
    L.append(f"# 모델 비교: {base_label} (기준) vs {cand_label} (새 모델)")
    L.append("")
    L.append(f"- {base_label}: {base['path']}")
    L.append(f"- {cand_label}: {cand['path']}")
    L.append("- 두 모델 모두 같은 val.txt 로 다시 평가한 결과예요. 차이가 ±0.005 안쪽이면 사실상 같은 수준으로 보세요.")
    L.append("- val 은 학습과 같은 출처의 이미지라서, 실제 Jetson 카메라에서의 성능과는 다를 수 있어요.")
    L.append("")
    L.append("## 전체 지표")
    L.append("")
    L.append(f"| 지표 | {base_label} | {cand_label} | 차이 |")
    L.append("|---|---|---|---|")
    for k in ("P", "R", "mAP50", "mAP50-95"):
        a, b = base["overall"][k], cand["overall"][k]
        L.append(f"| {k} | {a:.3f} | {b:.3f} | {_fmt_delta(a, b)} |")
    L.append("")
    L.append("## 클래스별 지표 (mAP50 / 재현율 R / 정밀도 P)")
    L.append("")
    L.append(f"| 클래스 | mAP50 {base_label} → {cand_label} | R {base_label} → {cand_label} | P {base_label} → {cand_label} |")
    L.append("|---|---|---|---|")
    for c in sorted(names):
        a, b = base["per_class"].get(c), cand["per_class"].get(c)
        if a is None or b is None:
            continue
        L.append(
            f"| {names[c]} | {a['mAP50']:.3f} → {b['mAP50']:.3f} ({_fmt_delta(a['mAP50'], b['mAP50'])}) "
            f"| {a['R']:.3f} → {b['R']:.3f} ({_fmt_delta(a['R'], b['R'])}) "
            f"| {a['P']:.3f} → {b['P']:.3f} ({_fmt_delta(a['P'], b['P'])}) |"
        )

    if base.get("confusion") and cand.get("confusion"):
        L.append("")
        L.append("## 정답 클래스가 어디로 나왔나 (오분류 비교, % 단위)")
        L.append("")
        L.append(f"| 진짜 클래스 | 정답 {base_label} → {cand_label} | 놓침(배경) {base_label} → {cand_label} | 가장 많이 틀린 쪽 {base_label} → {cand_label} |")
        L.append("|---|---|---|---|")
        for c in sorted(names):
            a, b = base["confusion"].get(c), cand["confusion"].get(c)
            if a is None or b is None:
                continue

            def top(x):
                return f"{x['confused'][0][0]} {x['confused'][0][1] * 100:.1f}%" if x["confused"] else "없음"

            L.append(
                f"| {names[c]} | {a['correct'] * 100:.1f} → {b['correct'] * 100:.1f} "
                f"| {a['missed'] * 100:.1f} → {b['missed'] * 100:.1f} | {top(a)} → {top(b)} |"
            )
    return L


def compare(base_path, cand_path, base_label, cand_label, names):
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    base = evaluate(base_path, base_label, names)
    cand = evaluate(cand_path, cand_label, names)

    lines = format_comparison(base, cand, names, base_label, cand_label)
    print("\n" + "=" * 70)
    print("\n".join(lines))
    print("=" * 70)

    out_md = COMPARE_DIR / f"compare_{base_label}_vs_{cand_label}.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n📝 비교표 저장: {out_md}")
    print(f"🖼️  혼동행렬 이미지: {COMPARE_DIR / ('val_' + base_label)}  /  {COMPARE_DIR / ('val_' + cand_label)}")


# ── 실행 ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="새 학습 + v2 와 비교")
    p.add_argument("--name", default="v3_4class", help="결과 폴더 이름 (기본 v3_4class)")
    p.add_argument("--start", default="old",
                   help="출발 가중치: old(예전 ultimate 모델, v2 와 같은 조건) / v2(현재 모델에서 이어서) / coco(yolov8s 기본) / 경로")
    p.add_argument("--epochs", type=int, default=40, help="학습 epoch 수 (기본 40)")
    p.add_argument("--time", type=float, default=None, help="시간 제한(시간 단위). 쓰면 총 epoch 수가 계속 재계산돼요")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--resume", action="store_true", help="중단된 학습 이어서 하기")
    p.add_argument("--skip-count", action="store_true", help="데이터 구성 집계(1~3분)를 건너뜀")
    p.add_argument("--compare-only", action="store_true", help="학습 없이 두 모델 비교만")
    p.add_argument("--counts-only", action="store_true", help="학습 없이 현재 데이터 구성(클래스별 인스턴스 수)만 출력")
    p.add_argument("--baseline", default=None, help="기준 모델 경로 (기본: v2 백업)")
    p.add_argument("--candidate", default=None, help="새 모델 경로 (기본: 이번 학습의 best.pt)")
    p.add_argument("--no-compare", action="store_true", help="학습 후 비교를 하지 않음")
    return p.parse_args()


def main():
    args = parse_args()

    if not DATA_YAML.exists() or not TRAIN_MANIFEST.exists() or not VAL_MANIFEST.exists():
        print(f"❌ data.yaml / train.txt / val.txt 를 {SMALL_DATASET_ROOT} 에서 찾을 수 없습니다.")
        sys.exit(1)

    names = load_names()

    if args.counts_only:
        save_dataset_snapshot("counts_preview", names, skip_count=False)
        return

    ensure_v2_backup()

    baseline = Path(args.baseline) if args.baseline else (V2_BACKUP if V2_BACKUP.exists() else V2_BEST)

    if args.compare_only:
        if not args.candidate:
            print("❌ --compare-only 에는 --candidate <새 모델 경로> 가 필요합니다.")
            sys.exit(1)
        if not baseline.exists() or not Path(args.candidate).exists():
            print("❌ 비교할 모델 파일을 찾을 수 없습니다.")
            sys.exit(1)
        compare(baseline, Path(args.candidate), "v2", Path(args.candidate).parent.parent.name, names)
        return

    best = train(args, names)
    if best is None:
        sys.exit(1)

    if args.no_compare:
        return

    candidate = Path(args.candidate) if args.candidate else best
    if not baseline.exists() or not candidate.exists():
        print("⚠️  비교할 모델 파일이 없어 비교를 건너뜁니다.")
        return
    compare(baseline, candidate, "v2", args.name, names)

    print("\n다음 단계: 새 모델이 더 좋으면 best.pt 를 Jetson 으로 옮겨 engine 으로 다시 변환하세요.")
    print(f"   새 모델 파일: {candidate}")


if __name__ == "__main__":
    main()
