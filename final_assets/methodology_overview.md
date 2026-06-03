# Methodology Overview — A Story of Listening and Measuring

This project began with a simple question: can the combined voice of an engine and the story told by its internal sensor rhythms be read together to reveal not just whether something is wrong, but how wrong it is?

We framed the problem as an ordinal grading task — four ordered states from Normal → Warning → Fault → Critical — because machine health is not a flat set of labels but a progression. Our methodology blends two proven single-modality systems (an OBD2 signal model and an audio spectrogram model) into a multimodal grader designed to respect ordinal structure, robustness across drivers, and explainability.

## The characters: two expert witnesses

- The OBD2 branch is the pragmatic witness: a 1‑D temporal expert that sees thirty timesteps of eight signals (RPM, coolant, load, MAP, IAT, throttle, speed, catalyst). Its architecture is Conv1D → BiGRU → Additive Attention, producing a compact 256‑dim embedding that encodes rhythm, drift, and abrupt changes.

- The Audio branch is the acoustic witness: a frequency‑aware specialist built around MSDA‑Net that ingests a 3‑second log‑mel spectrogram and produces a 512‑dim embedding after multi‑scale encoders and frequency attention.

Each unimodal model already demonstrates strong competence on its own. The audio branch, trained with ordinal heads, achieves near‑ceiling accuracy; the OBD2 branch offers complementary temporal perspective and is more sensitive to mechanical patterns the microphone may miss.

## The plot: how we fuse their testimony

We progress through a measured sequence of experiments, each increasing the intimacy of cross‑modal interaction.

1. Late Fusion Baseline — first meeting

   As a pragmatic baseline, we combine frozen unimodal logits using a learned temperature‑scaled or weighted average combiner (audio heavier by default). This fast ensemble reveals the "free lunch": simply pooling two strong opinions improves robustness and sets a realistic baseline for more complex fusion.

2. Low‑Rank Multiplicative Fusion (OLRMF) — statistical handshake

   Here the modalities interact multiplicatively via a low‑rank decomposition. The LMF block models pairwise interactions between features while keeping parameter growth manageable. This step tests whether multiplicative interactions extract joint signals the late fusion misses.

3. MMTM / SE‑style recalibration — a polite conversation

   At intermediate feature levels we add Multimodal Transfer Modules (MMTM) that compute compact cross‑modal gates. These gates recalibrate channel responses in each branch using the other's summary statistics, enabling audio to spotlight informative OBD2 channels and vice‑versa.

4. Cross‑Modal Ordinal Transformer (CMOFT / DAMOG) — a full dialogue

   The climax is a bidirectional cross‑attention transformer where OBD2 timesteps and audio tokens attend to each other. This allows the network to align events across time–frequency and sensor dimensions, discover long‑range correspondences, and surface interpretable cross‑attention maps for thesis figures.

## Training with ordinal consciousness

Grading is inherently ordered. We therefore preserve ordinal structure at the loss level: the fused head uses CORN (conditional ordinal heads), augmented with a Weighted Cohen's Kappa term to directly optimize our evaluation metric and (optionally) pairwise ranking constraints (THOR). Auxiliary ordinal heads on each unimodal embedding are included during training to keep the encoders grounded and to stabilize learning.

A representative loss looks like:

L = L_CORN(fused) + 0.5·L_WeightedKappa(fused) + 0.3·(L_CE(obd2_aux)+L_CE(audio_aux))

Training follows a phased protocol for the transformer fusion: start by training heads and encoders (cross‑attn frozen), then unfreeze and train jointly, and finally fine‑tune attention + classifier with lower LR.

## Data and pairing

We build a paired dataset that yields (obd2_window, audio_segment, label). Temporal scale mismatches are handled by hierarchical pairing: short fixed OBD2 windows (30 timesteps) mapped to corresponding audio frames (3‑sec segments). When exact timestamps are unavailable, stratified pairing by label and driver is used to preserve cross‑driver generalization checks.

Preprocessing reuses existing unimodal scalers and augmentations; we import rather than rewrite those pipelines to maintain fidelity with the pretrained encoders.

## Evaluation and experiments

Experiments are staged to provide clear ablation and narrative:

- Unimodal references (already done): validate both branches in isolation and measure cross‑driver generalization.
- Late fusion baseline: the practical ensemble comparator.
- CNN‑BiGRU concatenation (paper‑replication): a strong literature‑backed fusion model (fast to implement). 
- DAMOG (dual‑attention): primary thesis architecture — expected to give the best ordinal performance and interpretability.
- Ordinal loss ablation: quantify the importance of CORN, Weighted Kappa and THOR.

Primary metrics are Quadratic Cohen's Kappa (κ_w), Macro F1, MCC, and per‑class F1 (Critical class prioritized). Cross‑driver k‑fold tests demonstrate robustness.

## Explainability and thesis deliverables

We craft interpretable artifacts that tell the same story the model learned:

- Cross‑attention heatmaps (OBD2 timestep × audio frame) to show where modalities align.
- MMTM gate visualizations to show channel recalibration effects.
- SHAP/feature‑contribution analysis on fused embeddings to quantify modality influence per grade.
- Modal disagreement case studies highlighting transitional grades.

These figures become the backbone of the results section: not only how well, but why.

## Practical notes and invariants

- We treat the two unimodal projects as read‑only dependencies and import their encoders and scalers rather than copying them into the repo.
- Configurable fusion strategies (weighted, confidence, equal) allow rapid triage and deployment.
- The master `configs/config.yaml` centralizes weights, training hyperparameters, and grade semantics.

## Final thought — the method as a story

At the heart of the methodology is a human‑like inquiry: listen, measure, compare, and reconcile. Audio listens for timbre and harmonic signatures; OBD2 measures the machine's internal rhythms. Fusion is the conversation where the two witnesses reconcile their views into a single, ordered verdict. Our experiments move from quiet agreement (late fusion) through statistical conversation (LMF), to intimate recalibration (MMTM), to full dialogue (cross‑modal attention). Each step is designed to reveal more of the engine's story while keeping the grading ordinal, reliable, and explainable.

---

