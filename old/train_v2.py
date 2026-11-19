#!/usr/bin/env python3
"""
YOLO 쓰레기 탐지 모델 재학습 스크립트 (v2 - 병/플라스틱/종이/일반쓰레기 4클래스)

예전 ultimate_v1i_4class(캔/플라스틱/종이/일반쓰레기) 모델의 best.pt를
전이학습 출발점으로 사용해서, 캔 대신 병 클래스로 새로 학습한다.
데이터셋은 AI Hub 변환분 + 예전 데이터셋(캔 제외) + 일반쓰레기 오버샘플링이
전부 합쳐진 D:\\waste-dataset 의 train.txt / val.txt 를 사용한다.

사용법:
    python train_v2.py
"""

import os
import shutil
import sys
from pathlib import Path
from ultralytics import YOLO

# ---- 새 데이터셋 (AI Hub + 예전데이터 + 일반쓰레기, 전부 합쳐진 곳) ----
WASTE_DATASET_ROOT = Path(r"D:\waste-dataset")
DATA_YAML = WASTE_DATASET_ROOT / "data.yaml"
TRAIN_MANIFEST = WASTE_DATASET_ROOT / "train.txt"
VAL_MANIFEST = WASTE_DATASET_ROOT / "val.txt"

# ---- 예전 프로젝트 (전이학습 출발점 best.pt가 있는 곳 - 절대 삭제하면 안 됨!) ----
OLD_BASE_DIR = Path(r"C:\Users\aerty\Desktop\전공")
OLD_BEST_MODEL_PATH = OLD_BASE_DIR / "runs" / "ultimate_v1i_4class" / "weights" / "best.pt"

# ---- 이번 학습 결과가 저장될 새 위치 (여기만 정리 대상) ----
RUN_NAME = "v2_4class"
RUNS_DIR = WASTE_DATASET_ROOT / "runs"
RUN_DIR = RUNS_DIR / RUN_NAME
BEST_MODEL_PATH = RUN_DIR / "weights" / "best.pt"


def clean_old_models():
    """이번 재학습(v2_4class) 결과만 삭제. 예전 모델(OLD_BASE_DIR)은 절대 건드리지 않음."""
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
        print(f"삭제: {RUN_DIR}")


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

    import shutil as _shutil
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
        # 터미널 너비 계산에 의존하지 않고, ANSI 코드로 줄 전체를 확실히 지운 뒤 새로 씀
        # (일부 터미널에서 너비 계산이 부정확해 이전 줄이 안 지워지고 쌓이는 문제 대응)
        real_file.write("\r\x1b[2K" + text)
        real_file.flush()

    TQDM._display = _display


def check_files():
    print("\n" + "=" * 70)
    print("필수 파일 확인")
    print("=" * 70)
    print("데이터셋: v2 (병/플라스틱/종이/일반쓰레기 4클래스)")
    print(f"현재 디렉토리: {os.getcwd()}")
    print(f"베이스 모델(전이학습 출발점): {OLD_BEST_MODEL_PATH}")
    print(f"데이터셋 경로: {WASTE_DATASET_ROOT}")

    if not OLD_BEST_MODEL_PATH.exists():
        print(f"오류: 전이학습용 best.pt를 찾을 수 없습니다: {OLD_BEST_MODEL_PATH}")
        return False

    if not DATA_YAML.exists():
        print(f"오류: '{DATA_YAML}' 파일을 찾을 수 없습니다!")
        return False

    if not TRAIN_MANIFEST.exists() or not VAL_MANIFEST.exists():
        print("오류: train.txt / val.txt 를 찾을 수 없습니다!")
        return False

    with open(TRAIN_MANIFEST, "r", encoding="utf-8") as f:
        train_count = sum(1 for line in f if line.strip())
    with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
        val_count = sum(1 for line in f if line.strip())

    print("data.yaml 찾음")
    print(f"train 목록: {train_count}개 / val 목록: {val_count}개")
    return True


def train_model():
    print("\n" + "=" * 70)
    print("모델 학습 시작 - v2 (병/플라스틱/종이/일반쓰레기 4클래스)")
    print("=" * 70)
    print(f"베이스 모델: {OLD_BEST_MODEL_PATH}")
    print(f"데이터셋: {DATA_YAML}")
    print("\n학습 시간 목표: 약 5시간")
    print("과적합 방지 설정:")
    print("  - 최대 100 epoch / 5시간 중 먼저 도달하면 종료")
    print("  - 조기 종료(patience=20): val 성능 정체 시 자동 중단")
    print("  - 낮은 학습률 + cosine decay")
    print("  - weight decay / mixup 증강")
    print("  - 마지막 5 epoch mosaic 비활성화")

    try:
        print("\n📥 모델 로드 중 (예전 best.pt에서 전이학습)...")
        model = YOLO(str(OLD_BEST_MODEL_PATH))
        print("✅ 모델 로드 완료!")

        # 참고: 커스텀 한 줄 진행률 패치는 터미널 환경에 따라 오작동할 수 있어
        # 제거함 (setup_single_line_progress). ultralytics 기본 표시 방식을 사용.

        results = model.train(
            data=str(DATA_YAML),
            epochs=100,
            time=7.0,
            imgsz=320,
            batch=64,
            patience=20,
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
            workers=8,
            cache="disk",  # RAM 캐시는 전체 데이터셋 기준 수십GB가 필요할 수 있어 disk 캐시로 안전하게
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
    print("모델 평가 (val 세트)")
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
    print("  YOLO 쓰레기 탐지 모델 재학습 (v2 - 병/플라스틱/종이/일반쓰레기)")
    print("=" * 70 + "\n")

    print("🗑️  이번 학습(v2_4class) 이전 결과만 정리 중 (예전 모델은 보존)...")
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
    python train_v2.py
""")


if __name__ == "__main__":
    main()