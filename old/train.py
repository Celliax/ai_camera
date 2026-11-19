#!/usr/bin/env python3
"""
YOLO 쓰레기 탐지 모델 학습 스크립트 (Ultimate v1i - 일상 4클래스)

사용법:
    python train.py
"""

import os
import shutil
import sys
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(r"C:\Users\aerty\Desktop\전공")
DATASET_DIR = BASE_DIR / "Ultimate.v1i.yolov8_4class"
DATA_YAML = DATASET_DIR / "data.yaml"
PRETRAINED_MODEL = "yolov8n.pt"
RUN_NAME = "ultimate_v1i_4class"
RUNS_DIR = BASE_DIR / "runs"
RUN_DIR = RUNS_DIR / RUN_NAME
BEST_MODEL_PATH = RUN_DIR / "weights" / "best.pt"


def clean_old_models():
    """이전 학습 결과 삭제"""
    targets = [
        RUNS_DIR,
        BASE_DIR / "best_model.pt",
        BASE_DIR / "yolov8n.pt",
        BASE_DIR / "yolo26n.pt",
    ]
    for path in targets:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"삭제: {path}")


def setup_single_line_progress():
    """학습 진행률을 한 줄로 갱신되도록 터미널/TQDM 설정"""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass

    import shutil
    from ultralytics.utils.tqdm import TQDM

    _orig_display = TQDM._display

    def _display(self, final=False):
        real_file = self.file
        captured = []

        class _Capture:
            def write(_, s):
                captured.append(s)

            def flush(_):
                pass

        self.file = _Capture()
        try:
            _orig_display(self, final=final)
        finally:
            self.file = real_file

        if not captured:
            return

        text = captured[-1].replace("\r\033[K", "").replace("\r", "").strip()
        width = shutil.get_terminal_size((120, 20)).columns
        real_file.write("\r" + text.ljust(width)[:width])
        real_file.flush()

    TQDM._display = _display


def check_files():
    print("\n" + "=" * 70)
    print("필수 파일 확인")
    print("=" * 70)
    print("데이터셋: Ultimate v1i (일상 4클래스)")
    print(f"현재 디렉토리: {os.getcwd()}")
    print(f"베이스 모델: {PRETRAINED_MODEL}")
    print(f"데이터셋 경로: {DATASET_DIR}")

    if not DATASET_DIR.exists():
        print(f"오류: '{DATASET_DIR}' 폴더를 찾을 수 없습니다!")
        print("먼저 remap_labels.py를 실행해서 4클래스 데이터셋을 생성하세요.")
        return False

    if not DATA_YAML.exists():
        print(f"오류: '{DATA_YAML}' 파일을 찾을 수 없습니다!")
        return False

    train_count = len(list((DATASET_DIR / "train" / "images").glob("*")))
    val_count = len(list((DATASET_DIR / "valid" / "images").glob("*")))
    print("사전학습 모델(yolov8n.pt)을 자동 다운로드합니다.")
    print("data.yaml 찾음")
    print(f"train 이미지: {train_count}개 / valid: {val_count}개")
    return True


def train_model():
    print("\n" + "=" * 70)
    print("모델 학습 시작 - Ultimate v1i (일상 4클래스)")
    print("=" * 70)
    print(f"베이스 모델: {PRETRAINED_MODEL}")
    print(f"데이터셋: {DATA_YAML}")
    print("\n학습 시간 목표: 약 5시간")
    print("과적합 방지 설정:")
    print("  - 최대 30 epoch / 5시간 중 먼저 도달하면 종료")
    print("  - 조기 종료(patience=8): val 성능 정체 시 자동 중단")
    print("  - 낮은 학습률 + cosine decay")
    print("  - weight decay / mixup 증강")
    print("  - 마지막 5 epoch mosaic 비활성화")

    try:
        print("\n📥 모델 로드 중...")
        model = YOLO(PRETRAINED_MODEL)
        print("✅ 모델 로드 완료!")

        setup_single_line_progress()

        results = model.train(
            data=str(DATA_YAML),
            epochs=30,
            time=5.0,
            imgsz=640,
            batch=8,
            patience=8,
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
            workers=0,
            project=str(RUNS_DIR),
            name=RUN_NAME,
            resume=False,
            exist_ok=False,
            plots=True,
            val=True,
        )

        print("\n" + "=" * 70)
        print("학습 완료!")
        print("=" * 70)
        print(f"최고 성능 모델: {BEST_MODEL_PATH}")
        print(f"마지막 모델: {RUN_DIR / 'weights' / 'last.pt'}")

        return results

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_model():
    print("\n" + "=" * 70)
    print("모델 평가 (valid 세트)")
    print("=" * 70)

    if not BEST_MODEL_PATH.exists():
        print(f"❌ best.pt 없음: {BEST_MODEL_PATH}")
        return None

    try:
        model = YOLO(str(BEST_MODEL_PATH))
        print("⏳ 검증 데이터셋으로 평가 중...")
        metrics = model.val(data=str(DATA_YAML), split="val")
        print("\n✅ 평가 완료!")
        print(f"   mAP50: {metrics.box.map50:.4f}")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        return metrics
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return None


def main():
    print("\n" + "=" * 70)
    print("  YOLO 쓰레기 탐지 모델 학습 (Ultimate v1i - 일상 4클래스)")
    print("=" * 70 + "\n")

    print("🗑️  이전 모델/학습 결과 삭제 중...")
    clean_old_models()

    if not check_files():
        sys.exit(1)

    if train_model() is None:
        sys.exit(1)

    validate_model()

    print("\n" + "=" * 70)
    print("모든 과정이 완료되었습니다!")
    print("=" * 70)
    print(f"""
학습 결과:
    {BEST_MODEL_PATH}

다시 학습:
    python train.py
""")


if __name__ == "__main__":
    main()