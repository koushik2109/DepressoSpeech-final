"""System Architecture Documentation - Multimodal Behavioral AI Framework.

================================================================================
OVERVIEW
================================================================================

This system implements a production-grade multimodal behavioral AI framework
for depression detection from video, audio, and text inputs.

Key Design Principles:
1. TRAIN/INFERENCE CONSISTENCY: Identical preprocessing pipelines
2. SMALL DATASET OPTIMIZATION: Advanced regularization and augmentation
3. ROBUST MULTIMODAL FUSION: Learnable weighting with entropy control
4. MODULAR ARCHITECTURE: Reusable components with clear interfaces
5. EXPLICIT VALIDATION: Dimension checking and consistency verification

================================================================================
FEATURE PIPELINE
================================================================================

AUDIO FEATURES (39D):
- 13x MFCC coefficients
- 13x MFCC delta (first derivative)
- 13x MFCC delta-delta (second derivative)
- Extracted via librosa from raw waveform (16 kHz, mono)
- Normalization: StandardScaler (zero mean, unit variance)
- No PCA reduction (preserves all variance)

VIDEO FEATURES (38D):
- 17x OpenFace Action Units (AU intensity)
- 3x Gaze angles (x, y, z)
- 3x Head pose angles (pitch, yaw, roll)
- 15x Additional behavioral markers
- Source: OpenFace CSV pre-computation
- Normalization: StandardScaler
- No PCA reduction (all features are important)

TEXT FEATURES (384D):
- SBERT embeddings (sentence-transformers/all-mpnet-base-v2)
- Extracted from transcript chunks (~256 tokens per chunk)
- Temporal dimension: multiple chunks per session
- Normalization: StandardScaler
- No PCA reduction (standard embedding dimension)

TEMPORAL AGGREGATION:
- Audio: Variable temporal frames (typically 20-100 frames per session)
- Video: Variable temporal frames from OpenFace
- Text: Multiple chunks (typically 2-5 chunks per session)
- All padded to max length during collation with attention masking

================================================================================
TRAINING PIPELINE
================================================================================

STEP 1: Data Loading (src/dataset/builder.py)
- Load participant IDs and labels from CSV splits
- Locate pre-computed feature NPZ files
- Validate file existence

STEP 2: Preprocessing Fitting (scripts/train_v2.py)
- Fit StandardScaler on training features for each modality
- Save preprocessors to checkpoints for later inference

STEP 3: Feature Preprocessing (src/features/preprocessing.py)
- Apply fitted normalization to all features
- Validate dimensions and NaN/Inf values
- Convert to torch tensors

STEP 4: Batching (src/dataset/collate.py)
- Pad sequences to max length in batch
- Create attention masks for valid timesteps
- Stack batch samples

STEP 5: Model Encoding (src/models/multimodal_model.py)
- Temporal encoding per modality (CNN, GRU, or Transformer)
- Attention pooling to aggregate temporal information
- Per-modality projection to unified dimension

STEP 6: Multimodal Fusion (src/models/fusion.py)
- Learnable gating weights for each modality
- Cross-modal attention for inter-modality interaction
- Entropy regularization to prevent modality collapse

STEP 7: Task Heads
- Regression head: PHQ-9 total score (0-27)
- Question head: Per-question scores (0-3 each)
- Classification head: Binary depression (PHQ >= 10)
- Confidence head: Prediction confidence

STEP 8: Loss Computation (src/training/losses.py)
- Regression loss: MSE + CCC (correlation)
- Question loss: MSE on per-question scores
- Classification loss: BCE with logits
- Entropy loss: encourage multimodal diversity
- Confidence loss: penalize confident wrong predictions
- Total: weighted combination of all losses

================================================================================
INFERENCE PIPELINE
================================================================================

LIVE INFERENCE PROCESS (src/inference/live_pipeline.py):

INPUT SOURCES:
- Audio file (.wav, .mp3, etc.)
- OpenFace CSV (pre-computed video features)
- Transcript (text string)

STEP 1: Feature Extraction
- Audio: Extract MFCC features from waveform
- Video: Load OpenFace CSV features
- Text: Chunk transcript and encode with SBERT

STEP 2: Feature Validation
- Check dimensions match specifications
- Validate dtype (float32)
- Check for NaN/Inf values
- Verify temporal consistency

STEP 3: Preprocessing Application
- Apply saved normalization to features
- Apply PCA if configured

STEP 4: Tensor Preparation
- Convert to torch tensors
- Create attention masks
- Move to GPU if available

STEP 5: Model Inference
- Forward pass through model
- Get predictions for all tasks
- Extract modality confidence scores

STEP 6: Post-Processing
- Convert logits to probabilities
- Clip values to valid ranges
- Aggregate modality contributions

STEP 7: Output Generation
- PHQ-9 total score (continuous)
- Per-question scores
- Binary depression classification
- Confidence estimate
- Modality weights and confidence scores

================================================================================
MULTIMODAL FUSION MECHANISM
================================================================================

FUSION STRATEGY (Hybrid Mode - Recommended):

1. MODALITY ENCODING
   - Each modality processed independently
   - Temporal encoding (CNN, GRU, or Transformer)
   - Attention pooling to fixed-size representation
   - Per-modality projection to fusion_dim

2. CROSS-MODAL ATTENTION
   - Stack modality representations [batch, 3, embed_dim]
   - Apply multihead self-attention
   - Each modality attends to all modalities
   - Respect presence masks (attend only to present modalities)

3. LEARNABLE GATING
   - Per-modality scoring networks
   - Input: modality representation [batch, embed_dim]
   - Output: unnormalized weight [batch, 1]
   - Softmax normalization (respecting presence)
   - Result: [batch, 3] normalized weights

4. CONFIDENCE SCORING
   - Separate confidence networks per modality
   - Input: modality representation
   - Output: confidence in [0, 1]
   - Result: [batch, 3] confidence scores

5. GATING × ATTENTION
   - Multiply attended representations by gate weights
   - Weighted sum across modalities
   - Result: fused representation [batch, embed_dim]

6. ENTROPY REGULARIZATION
   - Compute entropy of gating weights
   - Loss component: encourages balanced weights
   - Prevents single-modality dominance
   - Target entropy: ~0.9 (between 0=collapse, 1=uniform)

BENEFITS:
- No hard modality dropping (learnable soft weighting)
- Cross-modal information sharing
- Per-sample dynamic weights (adapt to content)
- Explicit confidence scoring
- Prevents mode collapse through entropy regularization

================================================================================
SMALL DATASET OPTIMIZATION
================================================================================

REGULARIZATION TECHNIQUES:

1. MODALITY DROPOUT (training only)
   - Randomly drop entire modalities during training
   - Encourages robustness to missing modalities
   - Rate: 10% per modality

2. TEMPORAL DROPOUT (training only)
   - Randomly remove temporal frames
   - Encourages robustness to incomplete sequences
   - Rate: 10% of frames

3. FEATURE NOISE AUGMENTATION
   - Add small Gaussian noise to features
   - Std: 0.01 (1% of typical feature range)
   - Regularizes embedding space

4. CONFIDENCE-AWARE LOSS
   - Weight samples by prediction confidence
   - Penalize confident but wrong predictions
   - Encourages calibrated uncertainty

5. ENTROPY REGULARIZATION
   - Encourage diverse modality usage
   - Prevents collapse to single modality
   - Target entropy: 0.9

6. EARLY STOPPING
   - Monitor validation loss
   - Patience: 15 epochs
   - Min delta: 0.001

7. CURRICULUM LEARNING
   - Start with easy task (all modalities)
   - Gradually increase difficulty (modality dropout increases)
   - Helps learning progression

8. EXPONENTIAL MOVING AVERAGE (EMA)
   - Maintain running average of weights
   - Use for inference (smoother predictions)
   - Decay: 0.999

9. CHECKPOINT AVERAGING
   - Average weights from last N checkpoints
   - Ensemble-like effect without retraining
   - N: 5 checkpoints

10. GRADIENT CLIPPING
    - Clip gradients to ±1.0
    - Prevents exploding gradients
    - Stabilizes training

WARMUP SCHEDULING:
- Linear warmup: 5 epochs from 0 to base_lr
- Cosine annealing: decay from base_lr to min_lr
- Total epochs: 100 (configurable)

================================================================================
ARCHITECTURE CHOICES FOR SMALL DATASETS
================================================================================

ENCODER TYPE (for temporal modeling):
- Transformer (recommended for small data):
  * 2 layers, 4 heads
  * Feed-forward dim: 2 × input dim
  * Dropout: 0.2
  * Advantages: Good attention patterns, stable with regularization
  * Disadvantages: More parameters than RNN

- BiGRU (lightweight alternative):
  * 1 layer bidirectional
  * No inter-layer dropout
  * Output projection to unified dim
  * Advantages: Fewer parameters, good for very small data
  * Disadvantages: Limited modeling capacity

- Conv1D + BiGRU (hybrid):
  * Light CNN (1 layer) for local patterns
  * BiGRU for temporal context
  * Parameter-efficient balance
  * Recommended for memory-constrained environments

POOLING STRATEGY:
- Attention pooling (recommended):
  * Learnable query vector
  * Compute attention scores
  * Weighted sum of representations
  * Advantages: Interpretable, learns important frames

- Statistics pooling:
  * Concatenate mean and std
  * Preserves variance information
  * Advantages: Efficient, low-parameter

FUSION MODE:
- Hybrid (recommended for small data):
  * Cross-modal attention + gating
  * Captures inter-modality relationships
  * Balanced parameter efficiency

================================================================================
VALIDATION & CONSISTENCY
================================================================================

FEATURE VALIDATION (src/inference/validation.py):

Checked at both training and inference:

1. DTYPE CHECK
   - Must be float32
   - Consistent across all features

2. DIMENSIONALITY CHECK
   - Audio: exactly 39D
   - Video: exactly 38D
   - Text: exactly 384D
   - Temporal dimension: > 0 frames

3. NaN/INF CHECK
   - No NaN values allowed
   - No ±Inf values allowed
   - Detects numerical instability

4. VALUE RANGE CHECK
   - Features should be ~normalized (std=1)
   - Warning if max abs value > 100
   - Warning if non-zero min < 1e-6

TRAIN/INFERENCE CONSISTENCY:

1. SPECIFICATION CHECK
   - Model config matches feature specs
   - Audio dim, video dim, text dim all match

2. PREPROCESSOR CHECK
   - Same normalizer types (training vs. inference)
   - Same PCA configuration (if enabled)
   - Same PCA components (if enabled)

3. DIMENSION PROPAGATION
   - Training features → model.audio_dim
   - Inference features → model.audio_dim
   - Both must match exactly

================================================================================
FILE STRUCTURE
================================================================================

Model/
├── src/
│   ├── features/
│   │   ├── specs.py ..................... Feature specifications
│   │   ├── preprocessing.py ............ Normalization & PCA
│   │   ├── audio_features.py ........... Audio extraction
│   │   ├── video_features.py ........... Video extraction
│   │   ├── text_features.py ............ Text extraction
│   │   ├── feature_store.py ............ NPZ storage
│   │   └── sanitization.py ............ NaN/Inf handling
│   ├── models/
│   │   ├── multimodal_model.py ........ Main model
│   │   ├── encoders.py ................ Temporal encoders
│   │   ├── fusion.py .................. Multimodal fusion
│   │   ├── heads.py ................... Output heads
│   │   └── temporal_modules.py ........ Advanced temporal
│   ├── training/
│   │   ├── trainer.py ................. Training loop
│   │   ├── losses.py .................. Loss functions
│   │   ├── metrics.py ................. Evaluation metrics
│   │   └── optimization.py ............ EMA, warmup, etc.
│   ├── inference/
│   │   ├── runtime_extractors.py ...... Live feature extraction
│   │   ├── live_pipeline.py ........... End-to-end inference
│   │   ├── validation.py .............. Feature validation
│   │   └── model_io.py ................ Checkpoint loading
│   ├── dataset/
│   │   ├── builder.py ................. Dataset construction
│   │   ├── multimodal_dataset.py ...... Dataset class
│   │   └── collate.py ................. Batch collation
│   └── preprocessing/
│       ├── audio_preprocessor.py ...... Audio preprocessing
│       ├── video_preprocessor.py ...... Video preprocessing
│       ├── text_preprocessor.py ....... Text preprocessing
│       └── chunker.py ................. Temporal windowing
├── scripts/
│   ├── train_v2.py .................... Improved training script
│   ├── infer_live_v2.py ............... Live inference example
│   └── extract_features.py ............ Feature extraction
└── configs/
    ├── training.yaml .................. Training configuration
    ├── inference.yaml ................. Inference configuration
    └── feature_extraction.yaml ........ Feature extraction config

================================================================================
USAGE EXAMPLES
================================================================================

1. TRAINING:
   ```bash
   cd Model
   python scripts/train_v2.py \
     --config configs/training.yaml \
     --experiment-name depression_v2
   ```

2. LIVE INFERENCE:
   ```bash
   python scripts/infer_live_v2.py \
     --checkpoint checkpoints/best_model.pt \
     --experiment depression_v2 \
     --audio session_audio.wav \
     --openface-csv session_openface.csv \
     --transcript "Transcript text here"
   ```

3. FEATURE EXTRACTION:
   ```bash
   python scripts/extract_features.py \
     --config configs/feature_extraction.yaml \
     --input-dir data/raw \
     --output-dir data/npz
   ```

================================================================================
TROUBLESHOOTING
================================================================================

DIMENSION MISMATCH ERRORS:
- Check configs/training.yaml model section matches expected dimensions
- Audio: 39D (MFCC only, no HuBERT)
- Video: 38D (OpenFace behavioral)
- Text: 384D (SBERT)

NaN/INF VALUES IN PREDICTIONS:
- Usually indicates gradient explosion
- Check grad_clip_norm setting (should be 1.0)
- Consider reducing learning rate
- Check input feature normalization

POOR CONVERGENCE:
- Use warmup scheduler (linear 5 epochs, then cosine annealing)
- Check learning rate (default 1e-4)
- Enable EMA (exponential moving average)
- Check feature normalization is applied

MODALITY DOMINANCE:
- Increase entropy_weight in loss (default 0.02)
- Check modality dropout is enabled (default 0.1)
- Verify gating mechanism is learning (not stuck at uniform)

================================================================================
"""

if __name__ == "__main__":
    print(__doc__)
