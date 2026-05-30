"""
Multimodal Engine Health Grader — CLI entry point.

Usage examples
--------------
# Both modalities (full multimodal grade):
  python grade_engine.py --audio engine.wav --obd2 readings.csv

# Single OBD2 snapshot (dict-style, replicated to 30-step window):
  python grade_engine.py --audio engine.wav \\
      --rpm 2500 --coolant 90 --load 65 --map 101 \\
      --iat 35 --throttle 40 --speed 80 --catalyst 450

# Audio only (no OBD2 available):
  python grade_engine.py --audio engine.wav --audio_only

# OBD2 only (no audio available):
  python grade_engine.py --obd2 readings.csv --obd2_only

# Save result to JSON:
  python grade_engine.py --audio engine.wav --obd2 readings.csv --output result.json

# Custom config:
  python grade_engine.py --audio engine.wav --obd2 readings.csv --config configs/config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

# Make sure the project root is on sys.path for src.* imports
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.grader import EngineHealthGrader
from src.fusion import GRADE_INFO

# ── Grade display symbols ─────────────────────────────────────
SYMBOLS = {0: "✅", 1: "⚠️ ", 2: "🔴", 3: "🚨"}
BAR_WIDTH = 22


def render_bar(p: float) -> str:
    filled = int(round(p * BAR_WIDTH))
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def print_result(result: dict) -> None:
    grade = result["grade"]
    info  = GRADE_INFO[grade]
    sym   = SYMBOLS[grade]

    print()
    print("━" * 62)
    print(f"  ENGINE HEALTH GRADE")
    print("━" * 62)
    print(f"  {sym}  {info['label'].upper()} (Grade {grade})")
    print(f"  Confidence : {result['confidence']:.1%}")
    print(f"  Action     : {info['action']}")
    print()

    # ── Fused probabilities ─────────────────────────────────
    print("  Fused Probabilities:")
    probs_dict = {GRADE_INFO[i]["label"]: float(result["probs"][i]) for i in range(4)}
    for label, p in probs_dict.items():
        marker = " ←" if label == info["label"] else ""
        print(f"    {label:<10} {render_bar(p)} {p:5.1%}{marker}")

    # ── Per-modality breakdown ───────────────────────────────
    if "obd2_grade" in result and "audio_grade" in result:
        print()
        print("  Modality Breakdown:")
        obd2_label  = GRADE_INFO[result["obd2_grade"]]["label"]
        audio_label = GRADE_INFO[result["audio_grade"]]["label"]
        w = result.get("fusion_weights", {})
        print(f"    OBD2  ({w.get('obd2',0.4):.0%} weight) → {obd2_label:<10}  "
              f"[conf {result['obd2_probs'][result['obd2_grade']]:.1%}]")
        print(f"    Audio ({w.get('audio',0.6):.0%} weight) → {audio_label:<10}  "
              f"[conf {result['audio_probs'][result['audio_grade']]:.1%}]  "
              f"({result.get('audio_n_segments','-')} segments)")

        if not result["agreement"]:
            sev = result["disagreement_severity"]
            tag = "MINOR" if sev == 1 else "SIGNIFICANT"
            print(f"    ⚡ {tag} DISAGREEMENT between modalities (Δgrade={sev})")
            print(f"       Fused result leans {'more' if sev == 1 else 'strongly'} "
                  f"toward the higher grade for safety.")

    elif result.get("modality") == "audio_only":
        print(f"\n  Mode: Audio-only  ({result.get('n_segments','-')} segments)")
    elif result.get("modality") == "obd2_only":
        print(f"\n  Mode: OBD2-only")

    print("━" * 62)
    print()


# ── Argument parser ───────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multimodal Engine Health Grader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--audio",      type=str, help="Path to engine audio file (WAV/MP3/FLAC)")
    p.add_argument("--obd2",       type=str, help="Path to OBD2 CSV file")
    p.add_argument("--config",     type=str, default="configs/config.yaml")
    p.add_argument("--output",     type=str, default=None, help="Save result to JSON")
    p.add_argument("--audio_only", action="store_true", help="Use audio branch only")
    p.add_argument("--obd2_only",  action="store_true", help="Use OBD2 branch only")

    # Single-timestep OBD2 values (shorthand for quick testing)
    g = p.add_argument_group("OBD2 single-reading (alternative to --obd2 CSV)")
    g.add_argument("--rpm",      type=float, help="Engine RPM")
    g.add_argument("--coolant",  type=float, help="Coolant temperature (°C)")
    g.add_argument("--load",     type=float, help="Engine load (%)")
    g.add_argument("--map",      type=float, help="Manifold absolute pressure (kPa)")
    g.add_argument("--iat",      type=float, help="Intake air temperature (°C)")
    g.add_argument("--throttle", type=float, help="Throttle position (%)")
    g.add_argument("--speed",    type=float, help="Vehicle speed (km/h)")
    g.add_argument("--catalyst", type=float, help="Catalyst temperature (°C)")
    return p


def args_to_obd2_dict(args) -> dict | None:
    """Build an OBD2 dict from CLI args if the single-reading flags were given."""
    mapping = {
        "engine_rpm":        args.rpm,
        "coolant_temp":      args.coolant,
        "engine_load":       args.load,
        "map_pressure":      args.map,
        "intake_air_temp":   args.iat,
        "throttle_position": args.throttle,
        "vehicle_speed":     args.speed,
        "catalyst_temp":     args.catalyst,
    }
    if all(v is None for v in mapping.values()):
        return None
    # Replace missing values with a safe default (0 → will be scaled)
    return {k: (v if v is not None else 0.0) for k, v in mapping.items()}


def serialize_result(result: dict) -> dict:
    """Convert numpy arrays to plain lists for JSON serialization."""
    out = {}
    for k, v in result.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = {sk: (sv.tolist() if hasattr(sv, "tolist") else sv)
                      for sk, sv in v.items()}
        else:
            out[k] = v
    return out


# ── Main ─────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    grader = EngineHealthGrader(config_path=args.config)

    # ── Determine mode ───────────────────────────────────────
    obd2_inline = args_to_obd2_dict(args)

    if args.audio_only:
        if not args.audio:
            parser.error("--audio_only requires --audio")
        result = grader.grade_audio_only(args.audio)

    elif args.obd2_only:
        obd2_src = obd2_inline or args.obd2
        if not obd2_src:
            parser.error("--obd2_only requires --obd2 or the single-reading flags")
        result = grader.grade_obd2_only(obd2_src)

    else:
        # Full multimodal grade
        if not args.audio:
            parser.error("--audio is required for multimodal grading (or use --audio_only)")
        obd2_src = obd2_inline or args.obd2
        if not obd2_src:
            parser.error("Provide OBD2 data via --obd2 CSV or the single-reading flags, "
                         "or use --audio_only")
        result = grader.grade(obd2=obd2_src, audio=args.audio)

    print_result(result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(serialize_result(result), f, indent=2)
        print(f"Result saved → {args.output}\n")


if __name__ == "__main__":
    main()
