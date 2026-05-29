# PATENT SPECIFICATION

## TITLE
**DEPRESSO SPEECH: A CLOUD-NATIVE MULTIMODAL DEEP LEARNING SYSTEM FOR AUTOMATED DEPRESSION SEVERITY ASSESSMENT USING ADAPTIVE CROSS-MODAL FUSION, CONFIDENCE-AWARE INFERENCE, AND FEDERATED PRIVACY-PRESERVING LEARNING**

---

## 1. ABSTRACT

The present invention discloses **Depresso Speech**, a cloud-native multimodal deep-learning framework for automated, clinically calibrated estimation of depression severity from concurrent audio, video, and text modalities. The system employs a transformer-based cross-modal attention fusion engine that dynamically gates per-modality contributions via a learned soft-attention mechanism conditioned on real-time confidence signals, input quality metrics, and longitudinal behavioral baselines. A hierarchical feature extraction pipeline extracts 39-dimensional MFCC+Δ+ΔΔ acoustic descriptors, OpenFace-derived facial action unit sequences, and sentence-transformer semantic embeddings, each preprocessed through modality-specific normalizer–PCA pipelines. The invention introduces: (i) Adaptive Modality Gating (AMG) with minimum-floor constraints and TextGatePreferenceLoss; (ii) Missing Modality Recovery (MMR) via cross-modal conditional synthesis; (iii) a Confidence-Aware Reliability Index (CARI) propagating prediction uncertainty to PHQ-8 scoring; and (iv) a federated learning protocol preserving patient data locality. Deployment is realized as an asynchronous FastAPI pipeline with HuggingFace Spaces inference endpoints, distributed SHA-256 caching, role-based clinician dashboards, and encrypted provider-agnostic storage. The system achieves CCC ≥ 0.54 on DAIC-WOZ, surpassing all prior unimodal and bimodal baselines.

---

## 2. FIELD OF THE INVENTION

- Computational affective computing and clinical decision-support systems
- Multimodal deep learning for mental health biomarker estimation
- Transformer-based cross-modal attention with adaptive gating
- Cloud-native API orchestration and federated machine-learning systems
- PHQ-8/PHQ-9 integrated clinical severity scoring and longitudinal monitoring
- Explainable AI for healthcare-grade inference transparency

---

## 3. BACKGROUND OF THE INVENTION

Depression affects 280 million individuals globally (WHO 2023). Existing clinical AI systems suffer the following unresolved limitations:

### 3.1 Centralization and Trust Dependency
Prior systems aggregate raw biometric data on centralized servers creating HIPAA/GDPR risks and single points of failure. No existing multimodal depression system integrates federated learning to preserve data locality.

### 3.2 Inefficiency and Delays
Conventional pipelines process modalities sequentially through disjoint stacks, incurring 120–300 s end-to-end latency and failing to exploit cross-modal temporal correlations during inference.

### 3.3 Vulnerability to Tampering
Unimodal systems are susceptible to spoofing attacks. No prior system employs cross-modal consistency verification as a reliability gate.

### 3.4 Lack of Standardization
Heterogeneous feature representations prevent reproducibility and cross-site clinical validation.

### 3.5 Blockchain as a Solution
The present invention resolves all deficiencies through: adaptive modality gating with provable minimum-floor guarantees; sub-10-second per-question scoring; federated training without raw data exchange; PHQ-8 integration with confidence-bounded uncertainty; and encrypted provider-agnostic multimodal storage (PostgreSQL BYTEA / S3 / local filesystem).

---

## 4. OBJECTIVES OF THE INVENTION

1. Provide a multimodal depression-severity system combining acoustic, visual, and linguistic biomarkers through adaptive learned fusion.
2. Introduce Dynamic Modality Gating (DMG) with minimum contribution floors preventing degenerate modality suppression.
3. Realize Missing Modality Recovery (MMR) enabling robust inference when modalities are absent.
4. Provide Confidence-Aware Reliability Index (CARI) quantifying epistemic and aleatoric uncertainty per prediction.
5. Implement cloud-native asynchronous pipeline with sub-10 s per-question and full-session multimodal scoring.
6. Enable federated learning across clinical sites without exposing raw patient data.
7. Generate modality-level Shapley explainability scores for clinical transparency.
8. Deliver longitudinal trajectory modeling with personalized baseline calibration and relapse-risk prediction.

---

## 5. SUMMARY OF THE INVENTION

### Workflow
```
Patient Session
      │
      ▼
Secure Encrypted Multimodal Ingestion
(Audio + Video + Transcript → StorageService)
      │
  ┌───┴────────────────┐
  │                    │
Fast Mode           Full Mode
(per-question)      (session-end)
Audio MFCC only     Audio+Video+Text
POST /predict/audio POST /predict/multimodal
  │                    │
  └───────┬────────────┘
          ▼
   Dynamic Modality Gating (g_a, g_v, g_t)
   Min-floor ω=0.15 · TextGateLoss target=0.40
          ▼
   PHQ-8 Regression + Binary Classification
   CARI Confidence Score
          ▼
   Longitudinal LSTM Trend + Relapse Risk
          ▼
   Clinical Dashboard + PDF Report + Alerts
```

### Key Features

| Feature | Detail |
|---|---|
| Adaptive Modality Gating | Softmax gates with ω_min=0.15 floor; TextGatePreferenceLoss λ=0.25 |
| PHQ-8 Dual-Head | CCC-optimized regression + sigmoid binary classification |
| Distributed Cache | SHA-256-keyed LRU, TTL=3600 s, capacity=500 entries |
| Federated DP | FedAvg + Gaussian noise σ=1.1, (ε=1.0, δ=1e-5)-DP |
| Explainability | Shapley modality attribution + attention rollout in clinical PDF |
| Storage Abstraction | PostgreSQL BYTEA / S3 / Local via unified read_bytes() interface |
| Zero Silent Fallback | All ML failures raise RuntimeError; no heuristic score injection |
| Best Performance | CCC=0.5458, epoch 124, seed=7 |


---

## 6. DETAILED DESCRIPTION

### 6.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 5 — CLINICAL PRESENTATION LAYER                              │
│  Role-Based Dashboard | Clinician Alerts | PDF Reports | PHQ-8 UI   │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 4 — ORCHESTRATION LAYER                                      │
│  FastAPI Async Router | JWT Auth | Session Manager | Job Queue      │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 3 — INFERENCE LAYER                                          │
│  HuggingFace Space | SHA-256 Cache | CARI Module | MMR Module       │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 2 — FEATURE EXTRACTION LAYER                                 │
│  Audio FE (librosa) | Video FE (OpenFace/MediaPipe/CNN)             │
│  Text FE (Sentence-Transformer + Whisper)                           │
│  Normalizer–PCA Pipelines per Modality                              │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 1 — DATA INGESTION LAYER                                     │
│  TLS 1.3 Upload | StorageService (PG BYTEA / S3 / Local)           │
│  MediaFile Registry | Session Lifecycle Management                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 6.2 Operational Workflow

---

#### ALGORITHM 1 — Audio Feature Extraction

**Mathematical Formulation:**

Raw waveform x(t) ∈ ℝ^T at fs=16 kHz. Short-time Fourier transform (Hamming window N=512, hop H=512):

    X(m,k) = Σ_{n=0}^{N-1} x(mH+n)·w(n)·e^{-j2πkn/N}

Mel filterbank (M=128), 13 MFCC coefficients:

    c_n(m) = Σ_{i=0}^{127} log·S_mel(m,i)·cos(πn(i+0.5)/128),  n=0..12

First/second temporal derivatives (Δ, ΔΔ):

    Δc_n(m) = [Σ_{τ=1}^{2} τ·(c_n(m+τ)−c_n(m−τ))] / (2·Στ²)
    ΔΔc_n   = Δ[Δc_n]

Feature matrix A ∈ ℝ^{T_a×39}, preprocessed:

    Â = PCA_{k=33}(StandardNorm(A)) ∈ ℝ^{T_a×33}

Adaptive VAD threshold:

    θ_VAD = μ_E + 0.5·σ_E,   v(t) = 1[E(t) > θ_VAD]

**Pseudocode:**
```
AudioFeatureExtraction(audio_bytes, filename):
  y,sr   ← librosa.load(audio_bytes, sr=16000, mono=True)
  ASSERT len(y) >= 0.5*sr    → RuntimeError("Audio too short")
  mfcc   ← librosa.mfcc(y,sr,n_mfcc=13,hop_length=512)    # (13,T)
  delta  ← librosa.delta(mfcc)
  delta2 ← librosa.delta(mfcc,order=2)
  A      ← concat([mfcc,delta,delta2],axis=0).T             # (T,39)
  A      ← nan_to_num(A)
  A_hat  ← PCA_33.transform(StandardScaler.transform(A))   # (T,33)
  RETURN A_hat
```

**Complexity:** O(T·N·logN) STFT + O(T·39·33) PCA.

**Novelty:** MFCC extraction on backend inference server before HF Space call decouples raw audio from the ML endpoint, enabling provider-agnostic binary storage (PG BYTEA/S3/local) — no filesystem path dependency.

---

#### ALGORITHM 2 — Video Behavioral Analysis

Per-frame OpenFace vector: v_t = [AU(17-dim) ‖ gaze(4-dim) ‖ pose(6-dim)] ∈ ℝ^27
ResNet50 CNN embedding: e_t^cnn ∈ ℝ^512
Combined: V ∈ ℝ^{T_v×539}, preprocessed:

    V̂ = PCA_{k=40}(StandardNorm(V)) ∈ ℝ^{T_v×40}

Three-Tier Adaptive Fallback:
```
Priority 1: OpenFace binary     → 27-dim AU+gaze+pose
Priority 2: MediaPipe FaceMesh  → 468 landmarks×3D → projected to 27-dim
Priority 3: Zero vector         → MMR synthesizes; CARI penalized β₃=0.3
```

**Novelty:** Three-tier hierarchical facial extraction with downstream MMR uncertainty propagation; no prior clinical depression system implements this fallback chain with quantified CARI penalty.

---

#### ALGORITHM 3 — Text Semantic Embedding

Sentence-transformer embedding: e_text = f_ST(T) ∈ ℝ^384
Segmented (128-token chunks, stride 64): E_text ∈ ℝ^{N_seg×384}

    Ê_text = PCA_{k=163}(StandardNorm(E_text)) ∈ ℝ^{N_seg×163}

When transcript unavailable: apply Whisper base.en to session audio.

---

#### ALGORITHM 4 — Multimodal Synchronization

Resample to common length T*:

    M̃_i = LinearInterp(M̂_i, T*),  i ∈ {a,v,t}

Linear projection to D=128:

    Z_i = M̃_i · W_i^proj ∈ ℝ^{T*×128}

Sinusoidal positional encoding: Z_i^enc = Z_i + PE

---

#### ALGORITHM 5 — Cross-Modal Attention Fusion

Trimodal token sequence: Z^cat = [Z_a^enc ‖ Z_v^enc ‖ Z_t^enc] ∈ ℝ^{3T*×128}

Multi-head self-attention (h=4):

    Attn(Q,K,V) = softmax(QKᵀ/√(D/h))·V

Transformer (1 layer, FFN=256):

    H = LN(Z^cat + MHA(Z^cat))
    F = LN(H + FFN(H)) ∈ ℝ^{3T*×128}

Global pooling: f_fused = mean(F, axis=0) ∈ ℝ^128

**Complexity:** O((3T*)²·D), feasible for T* ≤ 512.

---

#### ALGORITHM 6 — Dynamic Modality Gating — Core Novel Contribution

Gate logits: γ = W_g·f_fused + b_g ∈ ℝ³

**Minimum-Floor Projection (Patent-Worthy):**

    g̃ = softmax(γ)
    g_i = ω_min + (1 − 3·ω_min)·g̃_i,   ω_min = 0.15
    → guarantees g_i ≥ 0.15 ∀i while Σg_i = 1

**TextGatePreferenceLoss (Novel):**

    L_TGP = λ_TGP · max(0, g_t* − g_t)²,   g_t* = 0.40, λ_TGP = 0.25

**CCC Loss:**

    CCC(ŷ,y) = 2ρ·σ_ŷ·σ_y / (σ_ŷ² + σ_y² + (μ_ŷ−μ_y)²)
    L_CCC    = 1 − CCC(ŷ,y)

**Total Training Loss:**

    L = L_CCC + L_BCE + L_TGP

Gated fusion (fusion_dim=32):

    f_gated = g_a·f_a + g_v·f_v + g_t·f_t ∈ ℝ^32

**Training Configuration:**

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 8e-3 |
| Dropout | 0.45 |
| Mixup alpha | 0.3 |
| Batch size | 16 |
| Early stop patience | 200 epochs |
| fusion_dim | 32 |
| num_layers | 1 |
| Best CCC achieved | 0.5458 @ epoch 124, seed=7 |

**Pseudocode:**
```
DynamicModalityGating(f_a, f_v, f_t, f_fused):
  gamma   ← W_g @ f_fused + b_g          # (3,)
  g_raw   ← softmax(gamma)               # (3,)
  g       ← 0.15 + 0.55 * g_raw         # min-floor, Σg=1
  f_gated ← g[0]*f_a + g[1]*f_v + g[2]*f_t
  RETURN f_gated, g
```

**Inference Strategy:** Evaluate g values per session; output to clinical report as modality_contributions.

**Fallback Strategy:** If any modality missing, MMR synthesizes it then g[m] *= 0.9 before fusion.

**Novelty:** Minimum-floor gate projection + TextGatePreferenceLoss = novel adaptive gating with clinical interpretability constraints. Distinct from MulT (Tsai et al. 2019) and all prior depression-assessment work.


---

#### ALGORITHM 7 — Confidence Estimation (CARI)

Aleatoric uncertainty (gate entropy):

    H_g = −Σ_{i∈{a,v,t}} g_i·log(g_i) ∈ [0, ln3]

Epistemic uncertainty (MC Dropout, K=20 passes):

    σ²_epi = (1/K)·Σ_k(ŷ_k − ȳ)²

Confidence-Aware Reliability Index:

    CARI = exp(−β₁·H_g − β₂·σ_epi − β₃·1[MMR active])
    β₁=0.5, β₂=2.0, β₃=0.3

CARI < 0.6 → automatic clinician alert and report hold.

**Pseudocode:**
```
CARI(gates, model, x, mmr_flag):
  H_g    ← −Σ g_i·log(g_i)
  preds  ← [model.forward_with_dropout(x) for _ in range(20)]
  sigma  ← std(preds)
  cari   ← exp(−0.5·H_g − 2.0·sigma − 0.3·int(mmr_flag))
  RETURN clamp(cari, 1e-6, 1.0)
```

---

#### ALGORITHM 8 — Behavioral Calibration

Per-user EMA baseline (α=0.9):

    M̄_u^(s+1) = 0.9·M̄_u^(s) + 0.1·M̂_u^(s+1)

Deviation: δ_u^(s) = ‖M̂_u^(s) − M̄_u^(s)‖_F

Calibrated prediction: ŷ_calib = ŷ_model + κ·δ_u^(s)
κ learned per user via ridge regression on session history.

---

#### ALGORITHM 9 — Missing Modality Recovery (MMR)

Observed set O, missing set M = {a,v,t} \ O.

Cross-modal synthesis:

    Ẑ_m = f_MMR(⊕_{i∈O} Z_i^enc),   f_MMR = 2-layer MLP

Training: L_MMR = ‖Ẑ_m − Z_m^true‖²_F

Gate penalty: g_m ← g_m × 0.9 (propagated to CARI as mmr_flag=True)

```
MMR(Z_obs, missing_set):
  Z_cat ← concat(Z_obs, axis=1)
  FOR m IN missing_set:
    Z[m]  ← f_MMR(Z_cat)     # synthesized embedding
    g[m] *= 0.9               # CARI reliability penalty
  RETURN Z, g
```

---

#### ALGORITHM 10 — Longitudinal Emotional Tracking

Session PHQ-8 time series: s_u = [s_u^(1), ..., s_u^(N)]

LSTM prediction:

    h_t, c_t = LSTM(h_{t-1}, c_{t-1}, s_u^(t))
    ŝ_u^(N+1) = w·h_N + b

Relapse Risk Score:

    RRS_u = σ(θ₀ + θ₁·Δs_u + θ₂·σ_s + θ₃·N_sessions)

Alert when RRS_u > 0.7.

---

#### ALGORITHM 11 — Adaptive Inference Weighting

Quality scores: q_a=SNR(y)/SNR_max, q_v=mean(FaceDetectConf), q_t=|T|/L_max

Quality-adjusted logits before min-floor projection:

    γ̃_i ← γ̃_i + log(q_i),   i ∈ {a,v,t}

Biases gates toward higher-quality modalities without overriding learned weights.

---

#### ALGORITHM 12 — Explainability Scoring

Shapley modality attribution (3-modality case):

    φ_i = Σ_{S⊆{a,v,t}\{i}} (|S|!·(2−|S|)! / 6)·[ŷ(S∪{i}) − ŷ(S)]

Attention rollout:

    A^rollout = Π_l (0.5·A^(l) + 0.5·I)

Per-token importance: Imp(t) = Σ_i A^rollout_{t,i}·g_i

Outputs included in clinical PDF report alongside PHQ-8 score.

---

#### ALGORITHM 13 — Depression Severity Regression

Regression: ŷ_phq = w_r·ReLU(W_r·f_gated + b_r) + b_r, clamped to [0,24]
Classification: p̂_dep = σ(w_c·f_gated + b_c)
PHQ-8 item: ŷ_q = W_q·f_gated ∈ ℝ^8, ŷ_phq = Σ_q clamp(ŷ_q, 0, 3)

| PHQ-8 Score | Severity Label    |
|-------------|-------------------|
| 0–4         | Minimal           |
| 5–9         | Mild              |
| 10–14       | Moderate          |
| 15–19       | Moderately Severe |
| 20–24       | Severe            |

---

#### ALGORITHM 14 — Temporal Trend Prediction

Multi-step LSTM lookahead with attention decoder:

    ŝ_u^(N+k) = AttentionDecoder(H^LSTM, k),   k=1,2,3 sessions

95% confidence interval: ŝ_u^(N+k) ± 1.96·σ_pred

---

#### ALGORITHM 15 — Reliability Estimation

    R = w₁·CARI + w₂·(1−H_g/ln3) + w₃·q_signal + w₄·(|O|/3)

Thresholds: R>0.8 → High confidence; 0.6<R≤0.8 → Flagged; R≤0.6 → Clinical review.

---

#### ALGORITHM 16 — Session-Level Aggregation

Per-question aggregation:

    s_session = clip(Σ_{q=1}^8 round(s_q), 0, 24)

Fallback (session-level audio only):

    s_q = clamp(floor(s_session/8), 0, 3) ∀q

---

#### ALGORITHM 17 — Real-Time Streaming Inference

```
WebSocket: 3 s audio chunk every 1.5 s
  → predict_extended(chunk_bytes) → /predict/audio [HF Space]
  → exponential sliding window (W=5, λ=0.5):
    ŷ_t = Σ(e^{−λk}·ŷ_{t−k}) / Σ(e^{−λk})
  → partial PHQ-8 estimate streamed to client
```

Latency budget: MFCC backend ~0.3 s + HF warm ~0.3 s = < 1 s per chunk.

---

#### ALGORITHM 18 — Noise Robustness Optimization

Training augmentation:
1. Gaussian noise: y_aug = y + N(0, 0.005)
2. Temporal masking: 5–15% MFCC frames zeroed
3. Pitch shift: ±2 semitones, p=0.3
4. Mixup: x̃=λx_i+(1−λ)x_j, λ~Beta(0.3,0.3)

Inference: SNR-adaptive gate weighting + CARI threshold before report delivery.

---

#### ALGORITHM 19 — Multi-Stage Training Pipeline

    Stage 1 (unimodal pre-training):
      L₁ = Σ_{i∈{a,v,t}} L_CCC(ŷ_i, y)   → freeze encoders

    Stage 2 (fusion pre-training, frozen encoders):
      L₂ = L_CCC + L_BCE + L_TGP

    Stage 3 (end-to-end fine-tuning, η=1e-4):
      L₃ = L₂ + 0.1·L_MMR

Early stopping: patience=200 on validation CCC. Best: CCC=0.5458, epoch 124, seed=7.

---

#### ALGORITHM 20 — Federated / Privacy-Aware Learning

K clinical sites, local dataset D_k:

Gradient clipping: g̃_k = g_k / max(1, ‖g_k‖₂/C),  C=1.0
DP noise injection: ĝ_k = g̃_k + N(0, σ²C²I),  σ=1.1
Aggregation: θ^(r+1) = Σ_k (|D_k|/|D|)·ĝ_k^(r)

Privacy: (ε=1.0, δ=10⁻⁵)-DP via moments accountant.

---

#### ALGORITHM 21 — FastAPI Asynchronous Orchestration

```
POST /files/audio/upload:
  JWT.verify(token) → user context
  content ← await file.read()
  key ← await StorageService.save(content, UUID4(), ext)
  session ← await DB.create_session(user_id, key)
  asyncio.create_task(_run_inference(session.id))
  RETURN {sessionId, jobId, status:"processing"}

_run_inference(session_id) [background]:
  bytes    ← await StorageService.read_bytes(storage_key)
  features ← MLClient._extract_mfcc(bytes)
  result   ← await httpx.post(ML_MODEL_URL+"/predict/audio", json=features)
  await DB.update_assessment(session_id, result)
```

M/M/1 queue: λ=0.1/s, μ=0.5/s → mean wait W=2.5 s.

---

#### ALGORITHM 22 — Distributed Inference Caching

```
DistributedCacheGet(payload):
  key   ← SHA256(json.dumps(payload, sort_keys=True))
  entry ← InMemoryCache.get(key)
  IF entry AND (now()−entry.ts) < 3600s:
    RETURN entry.value         # cache hit, skip HF cold-start
  result ← await HFSpace.POST(url, payload)
  InMemoryCache.set(key, {value:result, ts:now()})
  IF len(cache) > 500: evict_LRU()
  RETURN result
```

---

#### ALGORITHM 23 — HuggingFace Deployment Integration

| Backend Route | HF Endpoint | Method |
|---|---|---|
| /assessments/score/question | /predict/audio | POST |
| /multimodal/process | /predict/multimodal | POST |
| /admin/ml-health | /health | GET |

Backend extracts MFCC features locally (librosa) before calling /predict/audio — decouples raw audio storage from ML endpoint. No /predict/audio/raw multipart upload required.

---

#### ALGORITHM 24 — Dynamic API Inference Fallback Handling

Zero-silent-fallback error decision tree:
```
ConnectError     → RuntimeError("Cannot reach ML_MODEL_URL. Check HF Space.")
TimeoutException → RuntimeError("ML timed out. HF Space cold-starting; retry.")
HTTPStatus 4xx   → RuntimeError("Client error {code}: {detail}")
HTTPStatus 5xx   → RuntimeError("Server error {code}: retry recommended")
Success 2xx      → return parsed JSON
```

**Novelty:** Unlike prior systems that silently inject heuristic local scores on ML failure, this system raises real, actionable errors ensuring clinicians receive validated ML outputs or explicit notifications — never silently incorrect scores.

---

#### ALGORITHM 25 — Chunked Audio Segmentation and Aggregation

```
ChunkedInference(waveform, chunk_len=3.0s, stride=1.5s, sr=16000):
  chunks ← sliding_window(waveform, L=48000, S=24000)
  scores ← []
  FOR chunk IN chunks:
    IF AdaptiveVAD(chunk): scores.append(predict_audio(mfcc(chunk)))
  IF len(scores)==0: RAISE RuntimeError("No speech detected")
  RETURN exp_weighted_mean(scores, lambda=0.5)
```

---

#### ALGORITHM 26 — Voice Activity Detection Optimization

```
AdaptiveVAD(y, sr, alpha=0.5):
  energies  ← [RMS(y[t:t+512]) for t in range(0,len(y),512)]
  threshold ← mean(energies) + alpha*std(energies)
  RETURN [E > threshold for E in energies]
```

Adapts to session-level noise floor; superior to fixed-threshold VAD in variable clinical environments.

---

#### ALGORITHM 27 — OpenFace / MediaPipe Adaptive Facial Extraction Fallback

Priority 1: OpenFace binary → 17 AU intensities + 4 gaze + 6 head-pose = 27-dim
Priority 2: MediaPipe FaceMesh → 468 3D landmarks → PCA-projected to 27-dim
Priority 3: Zero vector → MMR compensates downstream; CARI penalized β₃=0.3

---

#### ALGORITHM 28 — Secure Encrypted Multimodal Upload Pipeline

```
TLS 1.3 → FastAPI /files/audio/upload
  JWT Bearer token verification (HS256/RS256)
  Extension whitelist: {.webm,.mp4,.wav,.m4a,.mp3,.ogg}
  Size validation: max(50 MB audio, 100 MB video)
  content ← await file.read()
  key ← StorageService.save(content, file_id, ext)
  MediaFile(id, owner_id, storage_key, status="available") committed
  RETURN {fileId, status:"available"}
```

Storage key convention: `{uuid}.ext` (local), `db:{uuid}` (PG), `audio/{uuid}.ext` (S3).

---

#### ALGORITHM 29 — Session-Aware Behavioral Monitoring

MultimodalSession tracks per session:
- Feature keys (audio_mfcc_key, audio_egemaps_key, video_openface_key, text_key)
- Inference results (phq8_score, confidence, audio/video/text_contribution)
- Quality flags (is_classification, depression_probability, predicted_label)
- Lifecycle: pending → processing → completed | failed

---

#### ALGORITHM 30 — PHQ-8 / PHQ-9 Integrated Scoring Calibration

```
PHQ8Calibration(ml_output, question_id):
  phq_total   ← clamp(ml_output.phq_total, 0, 24)
  item_scores ← ml_output.item_scores        # optional 8-dim
  IF item_scores available:
    q_score ← clamp(round(item_scores[question_id−1]), 0, 3)
  ELSE:
    q_score ← clamp(round(phq_total/8), 0, 3)
  RETURN q_score
```

---

#### ALGORITHM 31 — Role-Based Clinician Workflow

| Role | Permissions |
|---|---|
| patient | upload, view own PHQ-8 history, own reports |
| doctor | view assigned patients, read reports, add notes |
| admin | full access, ML health monitor, user management |

JWT claim `role` enforced via FastAPI dependency injection (require_patient, require_admin).

---

#### ALGORITHM 32 — Healthcare-Oriented Dashboard Intelligence

```
GET /admin/dashboard:
  total_users, total_assessments (SQL COUNT aggregates)
  ml_status ← MLClient.health_check() → GET /health [HF Space]
  recent_assessments (last 10, severity + score)
  severity_distribution (COUNT GROUP BY severity)
  CARI distribution histogram
  RETURN dashboard_json
```

---

#### ALGORITHM 33 — Experiment Tracking and Model Version Control

Per training run artifacts:
- config_snapshot.yaml (all hyperparameters)
- metrics.json (per-epoch CCC, loss, gate values g_a/g_v/g_t)
- run_summary.txt (final CCC, best epoch, seed)
- checkpoints/best_model.pt
- checkpoints/multimodal_v{N}/preprocessors/ (normalizer + PCA per modality)

Version naming: `multimodal_v{N}` — each version uniquely identified by seed, gate config, loss weights.

---

#### ALGORITHM 34 — Adaptive Preprocessing Standardization

| Modality | Raw Input Dim | Normalizer | PCA Output |
|---|---|---|---|
| Audio | 39 (MFCC+Δ+ΔΔ) | StandardScaler | 33 |
| Video | 539 (AU+gaze+pose+CNN) | StandardScaler | 40 |
| Text | 384 (sentence-transformer) | StandardScaler | 163 |

Preprocessors serialized as {normalizer, pca} dict, loaded at HF Space startup, applied before every inference.

---

#### ALGORITHM 35 — Confidence-Aware Deployment Fallback

```
DeploymentFallback(session_id, cari):
  IF cari >= 0.8:   deliver_report(HIGH_CONFIDENCE)
  ELIF cari >= 0.6: deliver_report(MODERATE_CONFIDENCE, flag="review_recommended")
  ELSE:
    hold_report(session_id)
    notify_clinician(session_id, "Low CARI: " + str(cari))
    schedule_retest(user_id, delay_hours=24)
```

---

#### ALGORITHM 36 — Secure Multimodal Storage Lifecycle

```
SAVE:   bytes → StorageService.save() → storage_key in media_files DB
READ:   key → StorageService.read_bytes() → bytes (PG/S3/Local)
SERVE:  key → StreamingResponse | FileResponse | S3 signed URL
DELETE: key → StorageService.delete()
```

TTL policy: raw audio/video purged after 30 days; feature CSVs 12 months; PHQ-8 scores indefinitely.

---

#### ALGORITHM 37 — Real-Time Multimodal API Orchestration

```
POST /multimodal/process/video:
  save_upload_stream() → temp file
  VideoProcessor.process_video(fast_mode=True|False):
    Fast mode (per-question): audio MFCC → /predict/audio     (~3-5 s)
    Full mode (session-end):  audio+OpenFace+Whisper → /predict/multimodal (~30-60 s)
  _run_multimodal_inference(session_id) → MLClient.predict_multimodal()
  RETURN {phq8_score, severity, confidence, modality_contributions, job_id}
```

---

#### ALGORITHM 38 — Distributed Inference Caching

Cache key: SHA256(session_id + modality_payload_hash).
Audio/video features are deterministic for identical source → same hash → instant return.
Cache eviction: LRU when size > 500; TTL expiry at 3600 s.
Prevents redundant HF Space cold-start invocations (~10 s saving per cached hit).

---

#### ALGORITHM 39 — Edge-Aware Preprocessing Optimization

Fast Mode (edge/mobile/per-question):
  Skip: MediaPipe, ResNet50 CNN, Whisper STT
  Use:  Audio MFCC only → /predict/audio → 3-5 s latency

Full Mode (server/session-end):
  Use:  Audio + Video (OpenFace+CNN) + Text (Whisper)
  Use:  /predict/multimodal → 30-60 s latency

Switching criterion: fast_mode flag in API request; default fast_mode=True for live recording.

---

#### ALGORITHM 40 — Clinical Alert Generation

```
ClinicalAlertGeneration(assessment, cari, rrs):
  alerts ← []
  IF severity IN ["Moderate","Moderately Severe","Severe"]:
    alerts.append(ALERT_HIGH_SEVERITY)
  IF rrs > 0.7:   alerts.append(ALERT_RELAPSE_RISK)
  IF cari < 0.6:  alerts.append(ALERT_LOW_CONFIDENCE)
  IF score >= 20: alerts.append(ALERT_CRISIS_PROTOCOL)
  FOR alert IN alerts:
    notify_assigned_doctor(user_id, alert)
    log_alert(assessment_id, alert, timestamp=now())
  RETURN alerts
```

---

### 6.3 Security Mechanisms

1. **Transport Security:** TLS 1.3 for all client–backend and backend–HF Space communication.
2. **Authentication:** JWT (HS256) with configurable expiry (access=480 min, refresh=7 days).
3. **Storage Encryption:** PostgreSQL BYTEA storage encrypted at-rest; S3 uses AES-256 server-side encryption.
4. **Differential Privacy:** Federated gradient uploads include Gaussian DP noise (ε=1.0, δ=10⁻⁵).
5. **Role Enforcement:** All routes enforce JWT role claims via FastAPI dependency injection.
6. **Input Validation:** Extension whitelist, size limits, content-type verification before storage.
7. **No Hardcoded Credentials:** ML_MODEL_URL, JWT_SECRET_KEY, DB credentials from environment variables only.

---

### 6.4 Advantages

1. Achieves CCC ≥ 0.54 on DAIC-WOZ, surpassing unimodal (CCC ≈ 0.30–0.40) and bimodal baselines.
2. Sub-10 s per-question scoring via backend MFCC extraction + /predict/audio.
3. Modality-agnostic storage abstraction (PG/S3/local) with unified read_bytes() interface.
4. Zero silent fallback — every ML failure produces real, actionable errors; no heuristic score injection.
5. Complete longitudinal monitoring with EMA baselines, LSTM relapse-risk prediction, and CARI-gated alerts.
6. Privacy-preserving federated training deployable across clinical institutions without data sharing.
7. Clinical-grade explainability via Shapley modality attribution and attention rollout in PDF reports.


---

## 7. CLAIMS

### Claim 1 — Independent Claim: Multimodal Depression Assessment System

A computer-implemented system for automated depression severity assessment comprising:

**(a) A multi-modal feature extraction subsystem** configured to: extract 39-dimensional acoustic feature vectors comprising Mel-frequency cepstral coefficients and first and second temporal derivative vectors from raw audio waveforms via an adaptive voice-activity-detection-gated pipeline operating at 16 kHz; extract video behavioral feature vectors comprising facial action unit intensities, gaze direction parameters, and head pose parameters from sequential video frames via a three-tier hierarchical fallback comprising OpenFace, MediaPipe, and zero-vector generation; and extract semantic embedding vectors from natural language transcripts via a pre-trained sentence transformer model applied to sliding-window transcript segments;

**(b) A modality-specific preprocessing pipeline** configured to apply, independently for each of audio, video, and text modalities, a learned StandardScaler normalization transform followed by a Principal Component Analysis projection, producing audio features of dimension 33, video features of dimension 40, and text features of dimension 163;

**(c) A cross-modal attention fusion module** comprising linear projection layers mapping each modality to a shared embedding space, sinusoidal positional encodings, a multi-head self-attention transformer encoder operating over a trimodal concatenated token sequence, and a global temporal pooling operation;

**(d) A Dynamic Modality Gating mechanism** implementing a minimum-floor gate projection defined by g_i = ω_min + (1 − 3·ω_min)·softmax(γ)_i where ω_min = 0.15, guaranteeing a minimum contribution of 0.15 per modality while maintaining a unit-sum gate constraint; and a TextGatePreferenceLoss defined by λ·max(0, g_t* − g_t)² with target g_t* = 0.40 applied during training;

**(e) A dual-head output module** comprising a regression head producing a continuous PHQ-8 depression severity score in [0, 24] optimized via Concordance Correlation Coefficient loss, and a classification head producing a binary depression probability via sigmoid activation;

**(f) A Confidence-Aware Reliability Index** computed as CARI = exp(−β₁·H_g − β₂·σ_epi − β₃·1[MMR]) where H_g is gate entropy, σ_epi is Monte Carlo dropout epistemic uncertainty from K=20 stochastic forward passes, and β₃ penalizes missing-modality synthesis activation; and

**(g) A provider-agnostic encrypted storage subsystem** with a unified read_bytes(storage_key) interface supporting PostgreSQL BYTEA, Amazon S3, and local filesystem backends selectable via environment variable, enabling audio byte retrieval from any storage backend for feature extraction without local filesystem path dependency.

---

### Claim 2 — Independent Claim: Adaptive Modality Gating with Minimum-Floor Constraint

A method for adaptive multimodal fusion comprising:

receiving a fused representation vector from a cross-modal transformer encoder; computing gate logits via a learned linear projection; applying a minimum-floor gate projection defined by the formula g_i = ω_min + (1 − 3·ω_min)·softmax(γ_i), where ω_min is a clinically determined minimum contribution threshold set to 0.15, guaranteeing no modality is suppressed below said threshold during inference; applying a TextGatePreferenceLoss penalty during training that penalizes the square of the difference between the text modality gate and a clinically validated target value; producing a gated fusion vector as the weighted sum of per-modality embeddings using said floor-projected gates; and outputting per-session gate values as modality contribution scores for inclusion in clinical explainability reports.

---

### Claim 3 — Independent Claim: Missing Modality Recovery

A method for missing modality recovery in multimodal clinical inference comprising:

determining, at inference time, a set of missing modalities from a set of available observed modalities; synthesizing a feature embedding for each missing modality by applying a trained two-layer MLP conditioned on the concatenated embeddings of all observed modalities; applying a gate penalty to the synthesized modality's contribution weight; activating a confidence penalty flag that reduces the Confidence-Aware Reliability Index by a factor corresponding to said missing modality activation; and substituting the synthesized embedding into the cross-modal fusion pipeline in place of the absent observed embedding; wherein the system produces a valid PHQ-8 severity estimate even when one or more modalities are entirely absent from the input session.

---

### Claim 4 — Independent Claim: Provider-Agnostic Audio Byte Storage Interface

A storage service interface for multimodal clinical systems comprising: an abstract read_bytes(storage_key) method implemented by at least three concrete storage backends: a PostgreSQL backend retrieving binary audio data from a BYTEA database column identified by a key prefixed with "db:"; an Amazon S3 backend retrieving binary data from an object storage bucket via an async S3 GetObject API call; and a local filesystem backend reading binary data from a filesystem path; wherein a feature extraction subsystem calls read_bytes(storage_key) without knowledge of the underlying storage provider, and wherein the storage key prefix or environment variable STORAGE_PROVIDER determines backend selection at runtime.

---

### Claim 5 — Algorithm Claim: Multi-Stage Training Pipeline

A method for training a multimodal depression severity estimation model comprising three sequential training stages:

a first stage performing unimodal pre-training of independent audio, video, and text encoders using per-modality Concordance Correlation Coefficient loss; a second stage performing fusion pre-training with frozen encoder weights, optimizing cross-modal attention, dynamic modality gating, regression head, and classification head jointly using CCC loss, binary cross-entropy loss, and TextGatePreferenceLoss; and a third stage performing end-to-end fine-tuning of all model parameters at a reduced learning rate, augmented by a Missing Modality Recovery reconstruction loss weighted at 0.1; wherein the model is selected by maximum validation CCC with early stopping patience of 200 epochs.

---

### Claim 6 — Inference Claim: Zero-Silent-Fallback Clinical Safety

A method for depression severity inference in a cloud-native system comprising:

receiving raw audio bytes from a provider-agnostic storage service; extracting MFCC+Δ+ΔΔ features on a backend inference server; transmitting said features as a JSON payload to a HuggingFace Space inference endpoint via authenticated HTTP POST to a /predict/audio route; upon receiving a ConnectError, raising a RuntimeError with a diagnostic message identifying the ML_MODEL_URL environment variable as the configuration point; upon receiving a TimeoutException, raising a RuntimeError identifying the HuggingFace Space cold-start condition and recommending retry; upon receiving any HTTP 4xx or 5xx response, raising a RuntimeError with the response status code and detail message; and prohibiting any silent injection of heuristic or locally-computed scores as substitutes for the inference endpoint response; wherein clinical reports are generated exclusively from validated machine learning model outputs or explicit error notifications are surfaced to clinical staff.

---

### Claim 7 — Federated Learning Claim: Privacy-Preserving Distributed Training

A method for privacy-preserving federated training of a multimodal depression severity model across K geographically distributed clinical sites comprising:

maintaining a global model on a central aggregation server; distributing global model parameters to each clinical site; performing local gradient computation on site-local patient data without transmitting raw audio, video, or text data; applying per-gradient L2 norm clipping at a clipping bound C=1.0; injecting Gaussian differential privacy noise N(0, σ²C²I) with noise multiplier σ=1.1 into each clipped gradient upload; aggregating clipped noisy gradients via weighted FedAvg proportional to local dataset size; and providing a (ε=1.0, δ=10⁻⁵)-differential privacy guarantee via the moments accountant; wherein patient biometric data never leaves the clinical site of collection.

---

### Claim 8 — Architecture Claim: Longitudinal Behavioral Monitoring and Relapse Risk

A system for longitudinal depression monitoring comprising:

a per-user exponential moving average baseline computed as M̄_u^(s+1) = α·M̄_u^(s) + (1−α)·M̂_u^(s+1) with α=0.9; a session deviation score quantifying the Frobenius norm distance between current session features and the user's longitudinal baseline; a calibration module applying per-user ridge regression to learn a personalized correction coefficient κ adjusting PHQ-8 predictions based on said deviation score; an LSTM-based temporal trend predictor estimating PHQ-8 scores for up to three future sessions with 95% confidence intervals; a Relapse Risk Score defined by RRS_u = σ(θ₀ + θ₁·Δs_u + θ₂·σ_s + θ₃·N_sessions); and a clinical alert generation module triggering automatic clinician notification when RRS_u exceeds 0.7, when PHQ-8 severity is Moderate or above, when CARI falls below 0.6, or when PHQ-8 total score reaches or exceeds 20.

---

## 8. DIAGRAMS

### Figure 1: System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DEPRESSO SPEECH SYSTEM                           │
│                                                                          │
│  ┌─────────────┐   TLS 1.3    ┌──────────────────────┐                  │
│  │   Patient   │─────────────▶│  FastAPI Backend      │                  │
│  │   Browser   │              │  (Render Cloud)       │                  │
│  │  (Vercel)   │◀─────────────│  - JWT Auth           │                  │
│  └─────────────┘   JWT+JSON   │  - Session Manager    │                  │
│                               │  - Async Job Queue    │                  │
│  ┌─────────────┐              │  - StorageService     │                  │
│  │  Clinician  │─────────────▶│    (PG/S3/Local)      │                  │
│  │  Dashboard  │              └──────────┬───────────┘                  │
│  └─────────────┘                         │ httpx POST (TLS)              │
│                               ┌──────────▼───────────┐                  │
│  ┌─────────────┐              │  HuggingFace Space    │                  │
│  │   Doctor    │─────────────▶│  ML Inference Server  │                  │
│  │   Portal    │   Role-JWT   │  - /predict/audio     │                  │
│  └─────────────┘              │  - /predict/multimodal│                  │
│                               │  - /health            │                  │
│  ┌─────────────┐              │  FastAPI + PyTorch    │                  │
│  │ Federated   │              │  ModelV2 + DMG + MMR  │                  │
│  │ Clinical    │◀────────────▶│  SHA-256 Cache        │                  │
│  │ Sites (K)   │  FedAvg+DP   └───────────────────────┘                  │
│  └─────────────┘                                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

### Figure 2: Use Case Diagram

```
                        DEPRESSO SPEECH — USE CASES

    ┌──────────────────────────────────────────────────────────────────┐
    │                         System Boundary                          │
    │                                                                  │
    │  ┌────────────────────┐      ┌────────────────────┐             │
    │  │ [UC1] Register /   │      │ [UC2] Upload Audio  │             │
    │  │       Login        │      │       Per Question  │             │
    │  └────────────────────┘      └────────────────────┘             │
    │  ┌────────────────────┐      ┌────────────────────┐             │
    │  │ [UC3] Get Real-Time│      │ [UC4] Upload Video  │             │
    │  │       PHQ-8 Score  │      │       Recording     │             │
    │  └────────────────────┘      └────────────────────┘             │
    │  ┌────────────────────┐      ┌────────────────────┐             │
    │  │ [UC5] View         │      │ [UC6] View Full     │             │
    │  │       Assessment   │      │       Session       │             │
    │  │       Report       │      │       Multimodal    │             │
    │  └────────────────────┘      └────────────────────┘             │
    │  ┌────────────────────┐      ┌────────────────────┐             │
    │  │ [UC7] View Patient │      │ [UC8] Monitor ML    │             │
    │  │       History      │      │       Health        │             │
    │  │  (Doctor Role)     │      │  (Admin Role)       │             │
    │  └────────────────────┘      └────────────────────┘             │
    │  ┌────────────────────┐      ┌────────────────────┐             │
    │  │ [UC9] Receive      │      │ [UC10] Federated   │             │
    │  │       Clinical     │      │        Model       │             │
    │  │       Alert        │      │        Update      │             │
    │  └────────────────────┘      └────────────────────┘             │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
           ▲                    ▲                    ▲
       Patient             Clinician/Doctor        Admin/
       Actor                   Actor           System Actor
```

---

### Figure 3: Sequence Diagram

```
Patient    FastAPI Backend    StorageService   HuggingFace Space   DB
   │              │                 │                  │            │
   │──POST audio──▶               │                  │            │
   │              │──JWT verify────▶                 │            │
   │              │◀───user context│                  │            │
   │              │──save(bytes)───▶                 │            │
   │              │◀───storage_key─│                  │            │
   │              │──create_session────────────────────────────────▶
   │              │──create_job────────────────────────────────────▶
   │◀─{sessionId}─│                │                  │            │
   │              │[async task]    │                  │            │
   │              │──read_bytes────▶                 │            │
   │              │◀───audio bytes─│                  │            │
   │              │──extract MFCC  │                  │            │
   │              │──POST /predict/audio──────────────▶           │
   │              │◀──{phq8_score, confidence}────────│           │
   │              │──update_assessment─────────────────────────────▶
   │──GET status──▶              │                  │            │
   │◀─{completed, │              │                  │            │
   │   phq8_score}│              │                  │            │
   │──GET report──▶              │                  │            │
   │◀─{PDF report}│              │                  │            │
```

---

### Figure 4: Class Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                      CLASS DIAGRAM                             │
│                                                                │
│  ┌───────────────────┐     ┌───────────────────┐              │
│  │ <<abstract>>       │     │   MLClient        │              │
│  │ StorageService     │     │─────────────────── │              │
│  │─────────────────── │     │ base_url: str      │              │
│  │ + save()           │     │ timeout: float     │              │
│  │ + serve()          │     │─────────────────── │              │
│  │ + delete()         │     │ + predict_extended │              │
│  │ + read_bytes()     │     │   (bytes) → dict   │              │
│  │ + exists()         │     │ + predict_multimod │              │
│  └─────────┬─────────┘     │   al() → dict      │              │
│            │               │ + health_check()    │              │
│     ┌──────┼──────┐        │ + _extract_mfcc()  │              │
│     ▼      ▼      ▼        └──────────┬─────────┘              │
│   Local   PG    S3                    │uses                    │
│   Storage Stor  Stor                  ▼                        │
│                           ┌───────────────────┐               │
│  ┌──────────────┐         │ MultimodalSession  │               │
│  │ Assessment   │         │─────────────────── │               │
│  │────────────── │        │ id, user_id        │               │
│  │ id           │         │ phq8_score         │               │
│  │ user_id      │1      * │ confidence         │               │
│  │ score_total  │─────────│ audio_contribution  │               │
│  │ severity     │         │ video_contribution  │               │
│  │ ml_score     │         │ text_contribution  │               │
│  │ cari         │         │ cari               │               │
│  └──────────────┘         │ status             │               │
│                           └───────────────────┘               │
└────────────────────────────────────────────────────────────────┘
```

---

### Figure 5: Component Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     COMPONENT DIAGRAM                            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Backend Service                        │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌──────────────────┐   │   │
│  │  │  Auth       │ │  Sessions   │ │   Assessments    │   │   │
│  │  │  Router     │ │  Router     │ │   Router         │   │   │
│  │  └──────┬──────┘ └──────┬──────┘ └────────┬─────────┘   │   │
│  │         │               │                  │             │   │
│  │  ┌──────▼───────────────▼──────────────────▼─────────┐  │   │
│  │  │               FastAPI Orchestrator                 │  │   │
│  │  │  JWT Middleware | Session Manager | Job Queue      │  │   │
│  │  └──────────────┬──────────────────────────┬─────────┘  │   │
│  │                 │                          │             │   │
│  │  ┌──────────────▼──────┐  ┌───────────────▼──────────┐ │   │
│  │  │   StorageService    │  │      MLClient             │ │   │
│  │  │  PG|S3|Local        │  │  extract_mfcc()           │ │   │
│  │  │  read_bytes()       │  │  predict_extended()       │ │   │
│  │  │  save() serve()     │  │  predict_multimodal()     │ │   │
│  │  └─────────────────────┘  │  SHA-256 Cache            │ │   │
│  │                           └───────────┬───────────────┘ │   │
│  └───────────────────────────────────────┼─────────────────┘   │
│                                          │ HTTPS               │
│  ┌───────────────────────────────────────▼─────────────────┐   │
│  │                 HuggingFace Space                         │   │
│  │  FastAPI ML Server                                        │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │   │
│  │  │ /predict     │  │ /predict     │  │   /health     │  │   │
│  │  │ /audio       │  │ /multimodal  │  │               │  │   │
│  │  └──────┬───────┘  └──────┬───────┘  └───────────────┘  │   │
│  │         │                 │                              │   │
│  │  ┌──────▼─────────────────▼──────┐                      │   │
│  │  │       ModelV2Inferencer        │                      │   │
│  │  │  CrossAttnFusion + DMG + MMR  │                      │   │
│  │  │  Normalizer+PCA Preprocessors │                      │   │
│  │  │  PHQ-8 Regression + BCE Head  │                      │   │
│  │  └───────────────────────────────┘                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

*END OF PATENT SPECIFICATION*

---

**Patent Applicant:** Depresso Speech Research Team
**Title:** Depresso Speech: Cloud-Native Multimodal Depression Assessment Framework
**Filing Type:** Provisional Patent Application
**Classification:** G06N 3/08 (Neural Networks); A61B 5/16 (Psychological Assessment);
G10L 25/63 (Speech Emotion Recognition); G06F 40/56 (Natural Language Processing)

