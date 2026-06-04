"""
Prepare real car engine recordings for training.

Input layout:
  data_preparation/car_engine_raw/{normal,warning,fault,critical}/*.wav (or .mp3 .m4a .flac .ogg)

Output:
  data/{normal,warning,fault,critical}/*_seg<N>.wav  — 3 s clips at 44100 Hz mono

Usage:
  python data_preparation/car_engine_prepare.py
  python data_preparation/car_engine_prepare.py --raw_dir <path> --output_dir <path>
"""

import argparse
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

GRADE_NAMES = {0: "normal", 1: "warning", 2: "fault", 3: "critical"}
GRADE_MAP   = {v: k for k, v in GRADE_NAMES.items()}
TARGET_SR   = 44100
SEGMENT_S   = 3.0
HOP_S       = 1.0
TARGET_RMS  = 0.1
AUDIO_EXTS  = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}


def rms_normalize(signal: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(signal ** 2)) + 1e-9
    return signal * (TARGET_RMS / rms)


def segment_signal(signal: np.ndarray) -> list:
    seg_len = int(SEGMENT_S * TARGET_SR)
    hop_len = int(HOP_S * TARGET_SR)
    if len(signal) < seg_len:
        repeats = int(np.ceil(seg_len / len(signal)))
        signal = np.tile(signal, repeats)[:seg_len]
    segments, start = [], 0
    while start + seg_len <= len(signal):
        segments.append(signal[start : start + seg_len].astype(np.float32))
        start += hop_len
    return segments


def prepare_car_engine(raw_dir: str, output_dir: str) -> None:
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    if not raw_path.exists():
        print(f"ERROR: '{raw_path}' does not exist.")
        print("Create subfolders: normal/ warning/ fault/ critical/ inside it.")
        return

    for name in GRADE_NAMES.values():
        (out_path / name).mkdir(parents=True, exist_ok=True)

    stats, errored = {g: 0 for g in GRADE_NAMES}, 0

    for grade_name, grade in GRADE_MAP.items():
        grade_dir = raw_path / grade_name
        if not grade_dir.exists():
            print(f"[skip] '{grade_dir}' not found")
            continue

        audio_files = sorted(p for p in grade_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
        if not audio_files:
            print(f"[skip] no audio files in '{grade_dir}'")
            continue

        print(f"\nGrade {grade} ({grade_name}): {len(audio_files)} file(s)")
        for audio_path in audio_files:
            print(f"  {audio_path.name}", end=" ")
            try:
                signal, _ = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
                signal     = rms_normalize(signal.astype(np.float32))
                segments   = segment_signal(signal)
                for i, seg in enumerate(segments):
                    sf.write(str(out_path / grade_name / f"{audio_path.stem}_seg{i:04d}.wav"), seg, TARGET_SR)
                print(f"-> {len(segments)} segments ({len(signal)/TARGET_SR:.1f} s)")
                stats[grade] += len(segments)
            except Exception as exc:
                print(f"-> ERROR: {exc}")
                errored += 1

    print("\n" + "=" * 50)
    total = sum(stats.values())
    for g, name in GRADE_NAMES.items():
        flag = "  *** too few — record more" if stats[g] < 200 else ""
        print(f"  Grade {g} ({name:8s}): {stats[g]:4d} segments{flag}")
    print(f"  Total : {total} -> {out_path}/")
    if errored:
        print(f"  Errors: {errored} file(s) skipped")
    if total >= 800:
        print("\n  Ready. Run: python data_preparation/dataset_stats.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",    default="data_preparation/car_engine_raw")
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()
    print(f"Input: {args.raw_dir}  |  Output: {args.output_dir}  |  SR: {TARGET_SR} Hz  |  Seg: {SEGMENT_S}s / Hop: {HOP_S}s")
    prepare_car_engine(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
