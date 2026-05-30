"""
Fusion strategies for combining OBD2 and audio branch predictions.

Three strategies:
  weighted_average  — fixed weights from config (default: 0.4 obd2 / 0.6 audio)
  confidence        — dynamic weights proportional to each branch's max probability
  equal             — 0.5 / 0.5

All strategies operate on class probability vectors (4,) from each branch
and return a fused probability vector + decoded grade.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy as scipy_entropy


# Grade labels (shared across the system)
GRADE_INFO = {
    0: {"label": "Normal",   "action": "No action required."},
    1: {"label": "Warning",  "action": "Schedule preventive maintenance soon."},
    2: {"label": "Fault",    "action": "Immediate maintenance required."},
    3: {"label": "Critical", "action": "Stop engine immediately. Serious damage risk."},
}


def fuse(obd2_probs: np.ndarray, audio_probs: np.ndarray,
         strategy: str = "weighted_average",
         obd2_weight: float = 0.4,
         audio_weight: float = 0.6) -> dict:
    """
    Fuse two (4,) probability vectors into one final prediction.

    Returns a dict with:
        grade          int   0-3
        grade_label    str
        confidence     float probability of the predicted grade
        probs          np.ndarray (4,) fused probabilities
        action         str   recommended action
        obd2_grade     int
        audio_grade    int
        agreement      bool  True if both branches predict the same grade
        disagreement_severity  int  |obd2_grade - audio_grade| (0 = perfect agreement)
        obd2_probs     np.ndarray (4,)
        audio_probs    np.ndarray (4,)
    """
    if strategy == "weighted_average":
        w1, w2 = obd2_weight, audio_weight
    elif strategy == "confidence":
        # Weight proportional to each branch's certainty (max probability)
        c1 = float(np.max(obd2_probs))
        c2 = float(np.max(audio_probs))
        total = c1 + c2
        w1, w2 = c1 / total, c2 / total
    else:  # equal
        w1, w2 = 0.5, 0.5

    fused = w1 * obd2_probs + w2 * audio_probs
    fused = fused / fused.sum()   # renormalize for numerical safety

    grade      = int(np.argmax(fused))
    confidence = float(fused[grade])
    obd2_grade = int(np.argmax(obd2_probs))
    audio_grade = int(np.argmax(audio_probs))

    return {
        "grade":                  grade,
        "grade_label":            GRADE_INFO[grade]["label"],
        "confidence":             confidence,
        "probs":                  fused,
        "action":                 GRADE_INFO[grade]["action"],
        "obd2_grade":             obd2_grade,
        "audio_grade":            audio_grade,
        "agreement":              obd2_grade == audio_grade,
        "disagreement_severity":  abs(obd2_grade - audio_grade),
        "obd2_probs":             obd2_probs,
        "audio_probs":            audio_probs,
        "fusion_weights":         {"obd2": round(w1, 3), "audio": round(w2, 3)},
    }
