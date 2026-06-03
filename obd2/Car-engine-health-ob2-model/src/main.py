"""
Entry point — runs the full pipeline end to end:
  1. Preprocess
  2. Train
  3. Evaluate
  4. Explain (SHAP)

Usage:
  python main.py                    # uses config.yaml in parent directory
  python main.py ../config.yaml     # explicit path
"""

import sys
import os

# Always run with the project root as cwd, regardless of where the script is called from
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "config.yaml")

import preprocess
import augment
import train
import evaluate
import explain
import baseline
import cross_validate

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ENGINE HEALTH GRADING — Full Pipeline")
    print("=" * 60 + "\n")

    # Step 1
    preprocess.run(config_path)

    # Step 2 — augment training data only (val/test stay clean)
    augment.run(config_path)

    # Step 3
    history = train.run(config_path)

    # Step 3
    evaluate.run(config_path, history=history)

    # Step 4
    explain.run(config_path)

    # Step 5
    baseline.run(config_path)

    # Step 6
    cross_validate.run(config_path)

    print("\n Pipeline complete. Check data/processed/ for outputs.")
