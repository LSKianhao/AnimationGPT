# How AnimationGPT Learns Text-to-Animation

This document explains how AnimationGPT learns to convert text descriptions into animations.

---

## 🎯 Overview

AnimationGPT is built on **MotionGPT**, a transformer-based model that learns the relationship between **text descriptions** and **motion sequences** through supervised training on paired data.

```
Training Data:
┌──────────────────────────────────────────────────────────────┐
│ Text: "A man holding a katana, executing a heavy attack"    │
│ Animation: [Frame 1: joints (22,3), Frame 2: ..., Frame N]  │
└──────────────────────────────────────────────────────────────┘
         ↓ Train MotionGPT Model ↓
┌──────────────────────────────────────────────────────────────┐
│ Learned Model: Text → Animation                             │
└──────────────────────────────────────────────────────────────┘
         ↓ At Inference ↓
Input: "A person does a sword slash attack"
Output: Generated animation sequence
```

---

## 📊 Dataset Structure: The Foundation

### CMP Dataset (CombatMotionProcessed)
- **8,700 animations** from game assets
- **26,100 text descriptions** (3 per animation)
- **Format:** HumanML3D compatible

### Example Data Pair

**Animation:** `CMP008388.npy`
- Shape: `(T, 22, 3)` - T frames, 22 joints, 3D positions
- Frame rate: 30 fps
- Duration: ~2-5 seconds

**Text Annotations (3 variants):**

1. **Concise:**
```
weapon attack a man holding a Katana, executing a Charged Heavy Attack,
Dual Wielding, root motion get Forward, Steady, Powerful and Relative Slow,
First slow then fast, Cleanly.
```

2. **With Sensory Details:**
```
weapon attack a man holding a Katana, executing a Charged Heavy Attack,
Dual Wielding, root motion get Forward, Steady, Powerful and Relative Slow,
First slow then fast, Cleanly, which make a sense of Piercing, Wide Open,
Charged, Accumulating strength.
```

3. **Detailed Narrative:**
```
The character grips the wedge with both hands and charges for a powerful
strike. They firmly lower their body, twist to the left, lunge forward
with a bow step, and stab with the sword held in both hands.
```

### Dataset Directory Structure
```
datasets/humanml3d/
├── new_joints/          # Raw joint positions (T, 22, 3)
│   ├── CMP000001.npy
│   ├── CMP000002.npy
│   └── ...
├── new_joint_vecs/      # Processed motion features (T, 263)
│   ├── CMP000001.npy    # Preprocessed for training
│   └── ...
└── texts/               # Text annotations
    ├── CMP000001.txt    # Line 1: text1\nLine 2: text2\nLine 3: text3
    └── ...
```

---

## 🏗️ MotionGPT Architecture

AnimationGPT uses **MotionGPT**, which consists of three main components:

### 1. Motion Tokenizer (VQ-VAE)

**Purpose:** Convert continuous motion sequences into discrete tokens

```python
Motion Sequence (T, 22, 3)
         ↓
   Motion Encoder
         ↓
   Latent Code (T/4, 512)  # Temporal compression
         ↓
   Vector Quantization (VQ)
         ↓
   Motion Tokens (T/4,)     # Discrete indices [0-512]
         ↓
   Motion Decoder
         ↓
Reconstructed Motion (T, 22, 3)
```

**Why VQ-VAE?**
- Converts motion to discrete tokens (like words in language)
- Enables using GPT-style transformers
- Compression: 22×3=66 dimensions → 512-dim codebook

**Training VQ-VAE:**
```python
# Stage 1: Train motion autoencoder
Loss = Reconstruction_Loss + VQ_Loss + Commitment_Loss

# Reconstruction Loss: L1/L2 between original and reconstructed motion
recon_loss = ||motion_original - motion_reconstructed||

# VQ Loss: Make encoder output close to codebook vectors
vq_loss = ||encoder_output - quantized_vector||

# Commitment Loss: Prevent encoder from drifting
commit_loss = ||encoder_output.detach() - quantized_vector||
```

### 2. Text Encoder (T5)

**Purpose:** Convert text descriptions into embeddings

```python
Text: "A man holding a katana does a heavy attack"
         ↓
   T5 Tokenizer
         ↓
   Token IDs: [1, 123, 456, 789, ...]
         ↓
   T5 Encoder (pre-trained)
         ↓
Text Embeddings: (seq_len, 768)
```

**Why T5?**
- Pre-trained on massive text corpus
- Understands language semantics
- Transfer learning: already knows "attack", "sword", "heavy"

### 3. Motion GPT (Transformer)

**Purpose:** Learn to generate motion tokens conditioned on text

```python
Text Embeddings (seq_len, 768)
         ↓
   Cross-Attention
         ↓
Motion Tokens (auto-regressive generation)
  [START] → token_1 → token_2 → ... → token_N → [END]
         ↓
   VQ-VAE Decoder
         ↓
Final Animation (T, 22, 3)
```

**Architecture:**
```python
class MotionGPT(nn.Module):
    def __init__(self):
        self.text_encoder = T5Encoder()
        self.motion_vqvae = VQ_VAE()
        self.transformer = GPTDecoder(
            layers=12,
            heads=8,
            dim=512
        )

    def forward(self, text, motion_tokens):
        # Encode text
        text_emb = self.text_encoder(text)  # (B, L, 768)

        # Transformer with cross-attention
        # Auto-regressive: predict next motion token
        logits = self.transformer(
            motion_tokens,      # What we've generated so far
            context=text_emb    # Condition on text
        )

        return logits  # (B, T, vocab_size=512)
```

---

## 🎓 Training Process

### Overview
```
Step 1: Pre-train VQ-VAE (Motion Autoencoder)
Step 2: Train MotionGPT (Text → Motion Tokens)
Step 3: Fine-tune on Combat Dataset (CMP)
```

### Stage 1: VQ-VAE Pre-training

**Dataset:** All motion sequences (no text needed yet)

**Training Loop:**
```python
for epoch in range(50):
    for motion_batch in dataloader:
        # Forward pass
        encoded = vqvae.encoder(motion_batch)
        quantized, vq_loss = vqvae.quantize(encoded)
        reconstructed = vqvae.decoder(quantized)

        # Compute losses
        recon_loss = F.l1_loss(reconstructed, motion_batch)
        commit_loss = F.mse_loss(encoded, quantized.detach())

        total_loss = recon_loss + 0.25 * vq_loss + 0.02 * commit_loss

        # Backprop
        total_loss.backward()
        optimizer.step()
```

**Result:**
- Codebook of 512 motion "words"
- Each motion sequence → sequence of discrete tokens
- Reconstruction quality: high (small error)

### Stage 2: MotionGPT Training

**Dataset:** Paired (text, motion_tokens)

**Training Loop:**
```python
for epoch in range(100):
    for text_batch, motion_batch in dataloader:
        # Encode motion to tokens (using pre-trained VQ-VAE)
        with torch.no_grad():
            motion_tokens = vqvae.encode_to_tokens(motion_batch)

        # Encode text
        text_emb = t5_encoder(text_batch)

        # Auto-regressive training
        # Predict token_i given text + tokens[0:i-1]
        logits = motiongpt(
            motion_tokens[:-1],  # Input: all but last token
            context=text_emb
        )

        # Cross-entropy loss: predict next token
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            motion_tokens[1:].view(-1)  # Target: shifted by 1
        )

        # Backprop
        loss.backward()
        optimizer.step()
```

**What the model learns:**
- Given text "heavy attack", predict motion tokens that represent heavy attacks
- Given text "sword slash", predict different tokens
- Learns associations: text features ↔ motion patterns

### Stage 3: Fine-tuning on CMP

**Dataset:** 8,700 combat animations + text

**Config:** `config_AGPT.yaml`
```yaml
TRAIN:
  BATCH_SIZE: 16
  END_EPOCH: 50
  OPTIM:
    lr: 1e-4

LOSS:
  LAMBDA_FEATURE: 1.0      # Motion reconstruction
  LAMBDA_VELOCITY: 1.5     # Smooth motion (we increased this!)
  LAMBDA_COMMIT: 0.02      # VQ codebook
  LAMBDA_CLS: 1.0          # Text-motion alignment
```

**Training Command:**
```bash
cd MotionGPT
python train.py \
  --cfg ../AnimationGPT/config_AGPT.yaml \
  --dataset path/to/CMP/dataset \
  --gpu 0
```

**Training takes:** ~1-2 days on RTX 4090

---

## 🔮 Inference: Text → Animation

Once trained, generating animations is straightforward:

### Step-by-Step Process

**Input:** Text prompt
```python
prompt = "A person holding a katana does a heavy overhead attack"
```

**1. Encode Text:**
```python
text_tokens = t5_tokenizer(prompt)
text_emb = t5_encoder(text_tokens)  # (1, L, 768)
```

**2. Auto-regressive Generation:**
```python
# Start with [START] token
motion_tokens = [START_TOKEN]

for i in range(max_length):  # Generate up to max_length tokens
    # Predict next token
    logits = motiongpt(
        torch.tensor(motion_tokens),
        context=text_emb
    )

    # Sample next token (with temperature for diversity)
    next_token = sample(logits[-1], temperature=1.0)

    # Stop if [END] token
    if next_token == END_TOKEN:
        break

    motion_tokens.append(next_token)
```

**3. Decode to Motion:**
```python
# Convert tokens to motion sequence
motion_sequence = vqvae.decode(motion_tokens)  # (T, 22, 3)

# Save to file
np.save('generated_animation.npy', motion_sequence)
```

**Complete Inference Code:**
```python
# From AnimationGPT workflow
cd MotionGPT

# 1. Save prompt to file
echo "A person does a sword slash attack" > input.txt

# 2. Run inference
python demo.py \
  --cfg ../AnimationGPT/config_AGPT.yaml \
  --example ./input.txt

# 3. Output saved to:
# results/mgpt/debug--AGPT/id_out.npy
```

---

## 🧠 What the Model Learns

### Implicit Knowledge

Through training on 8,700 paired examples, the model learns:

1. **Motion Primitives:**
   - "attack" → forward lunging motion
   - "slash" → sweeping arm movement
   - "heavy" → slow, powerful motion
   - "quick" → fast, light motion

2. **Weapon-Specific Patterns:**
   - "katana" → two-handed grip, Japanese sword stance
   - "greatsword" → slower, heavier movements
   - "dual wielding" → coordinated two-arm motion

3. **Temporal Dynamics:**
   - "first slow then fast" → acceleration pattern
   - "charged attack" → windup → strike sequence
   - "dodge roll" → quick evasive motion

4. **Spatial Reasoning:**
   - "forward" → root motion in +Z direction
   - "left" → lateral movement
   - "overhead" → arms raised high

5. **Physical Constraints:**
   - Maintain balance (center of mass)
   - Foot contacts with ground
   - Realistic joint angles
   - Momentum conservation

### Attention Visualization

The model learns which text words map to which motion patterns:

```
Text:  "heavy" "overhead" "attack" "with" "katana"
         ↓        ↓          ↓       ↓       ↓
Attention to Motion Frames:
Frame 1-10:  [0.1,  0.3,     0.2,    0.1,    0.3]  # "katana" grip
Frame 11-20: [0.4,  0.5,     0.1,    0.0,    0.0]  # "heavy" windup
Frame 21-30: [0.2,  0.8,     0.9,    0.0,    0.1]  # "overhead attack"
```

---

## 📈 Training Metrics

### Loss Functions

**1. Feature Loss (LAMBDA_FEATURE = 1.0)**
```python
# Reconstruction accuracy of joint positions
feature_loss = F.l1_smooth_loss(
    predicted_motion,
    ground_truth_motion
)
```

**2. Velocity Loss (LAMBDA_VELOCITY = 1.5)**
```python
# Temporal smoothness
velocity_pred = predicted_motion[1:] - predicted_motion[:-1]
velocity_gt = ground_truth[1:] - ground_truth[:-1]
velocity_loss = F.mse_loss(velocity_pred, velocity_gt)
```

**3. VQ Commitment Loss (LAMBDA_COMMIT = 0.02)**
```python
# Keep encoder aligned with codebook
commit_loss = F.mse_loss(
    encoder_output.detach(),
    quantized_vector
)
```

**4. Classification Loss (LAMBDA_CLS = 1.0)**
```python
# Text-motion alignment
# Contrastive learning: matching text should score high
cls_loss = contrastive_loss(text_emb, motion_emb)
```

### Evaluation Metrics

From your README:

| Metric | MotionGPT Score |
|--------|----------------|
| **Matching Score** ↓ | 5.426 ± 0.017 |
| **R-precision (top 1)** ↑ | 0.044 ± 0.002 |
| **FID** ↓ | 0.531 ± 0.018 |
| **Diversity** → | 5.143 ± 0.052 |

**What these mean:**
- **Matching Score:** How well generated motion matches text (lower is better)
- **R-precision:** Can retrieve correct text for motion? (higher is better)
- **FID:** Quality of generated motion (lower is better)
- **Diversity:** Variety in generations (higher = more diverse)

---

## 🎮 Limitations of Text-to-Motion

### Why We Proposed Motion Matching Instead

| Issue | Text-to-Motion | Motion Matching |
|-------|---------------|-----------------|
| **Control Precision** | ❌ Vague ("heavy attack") | ✅ Exact (Pose A → Pose B) |
| **Speed** | ❌ Slow (seconds) | ✅ Real-time (<1ms) |
| **Quality** | ❌ Variable (artifacts) | ✅ Perfect (real data) |
| **Runtime** | ❌ Needs GPU | ✅ CPU is fine |
| **New Animations** | ❌ Retrain model | ✅ Add to database |

### When to Use Each

**Text-to-Motion (Current System):**
- ✅ Offline content creation
- ✅ Generating many variations
- ✅ Exploratory animation design
- ❌ Real-time gameplay
- ❌ Precise control needed

**Motion Matching (Proposed):**
- ✅ Real-time gameplay
- ✅ Precise pose control
- ✅ Interactive applications
- ✅ Game engine integration
- ❌ Creating totally new motions

---

## 🔧 How to Train Your Own Model

### Prerequisites
```bash
# 1. Download CMP dataset
wget https://drive.google.com/file/d/17tldNzQ2aFqwxwoqBAs4YqyDUnnPy8We

# 2. Setup MotionGPT
git clone https://github.com/OpenMotionLab/MotionGPT.git
cd MotionGPT
conda create -n mgpt python=3.10
conda activate mgpt
pip install -r requirements.txt

# 3. Download pre-trained models
bash prepare/prepare_t5.sh  # T5 text encoder
bash prepare/download_t2m_evaluators.sh  # Evaluation models
```

### Training Script
```bash
# Stage 1: Train VQ-VAE (if starting from scratch)
python train_vqvae.py \
  --dataset datasets/humanml3d \
  --batch-size 256 \
  --epochs 500

# Stage 2: Train MotionGPT
python train.py \
  --cfg ../AnimationGPT/config_AGPT.yaml \
  --dataset datasets/humanml3d \
  --batch-size 16 \
  --epochs 50 \
  --gpu 0
```

### Monitor Training
```bash
# TensorBoard
tensorboard --logdir runs/

# Weights & Biases (if configured)
wandb login
# Training metrics will be logged automatically
```

---

## 📊 Data Flow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING TIME                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Game FBX Assets                                           │
│         ↓                                                   │
│  Fbx2SMPL Conversion                                       │
│         ↓                                                   │
│  SMPL Joint Positions (T, 22, 3)                          │
│         ↓                                                   │
│  Manual Annotation (7 aspects)                            │
│         ↓                                                   │
│  GPT-4 Text Generation                                     │
│         ↓                                                   │
│  CMP Dataset (8,700 pairs)                                │
│         ↓                                                   │
│  ┌──────────────────────────────────┐                      │
│  │  Text: "heavy katana attack"     │                      │
│  │  Motion: [frames of joint data]  │                      │
│  └──────────────────────────────────┘                      │
│         ↓                                                   │
│  MotionGPT Training                                        │
│  - Text Encoder (T5)                                       │
│  - Motion VQ-VAE                                           │
│  - Transformer (GPT)                                       │
│         ↓                                                   │
│  Trained Model (mGPT.ckpt)                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     INFERENCE TIME                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Input: "sword slash attack"                          │
│         ↓                                                   │
│  T5 Encoder → Text Embeddings                             │
│         ↓                                                   │
│  MotionGPT Transformer                                     │
│  (Auto-regressive generation)                              │
│         ↓                                                   │
│  Motion Tokens [12, 45, 78, 23, ...]                      │
│         ↓                                                   │
│  VQ-VAE Decoder                                            │
│         ↓                                                   │
│  Joint Positions (T, 22, 3)                               │
│         ↓                                                   │
│  Post-processing (our improvements!)                       │
│  - Temporal smoothing                                      │
│  - Foot contact fixing                                     │
│  - Velocity clipping                                       │
│         ↓                                                   │
│  Final Animation (id_out.npy)                             │
│         ↓                                                   │
│  Convert to MP4/BVH                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Takeaways

1. **Learning happens through supervised training:**
   - 8,700 paired examples (text + motion)
   - Model learns associations through gradient descent
   - No explicit rules, just pattern matching

2. **Architecture is modular:**
   - VQ-VAE: motion compression
   - T5: text understanding
   - GPT: sequence generation

3. **Training is expensive:**
   - 1-2 days on high-end GPU
   - Requires large paired dataset
   - Pre-trained components help (T5)

4. **Quality depends on data:**
   - Better annotations → better results
   - More diverse data → more diverse outputs
   - Combat-focused data → good at combat

5. **For games, consider motion matching instead:**
   - Faster (< 1ms vs seconds)
   - More precise control
   - No GPU needed
   - Perfect quality

---

## 📚 References

**MotionGPT Paper:**
- "MotionGPT: Human Motion as a Foreign Language"
- Jiang et al., NeurIPS 2024
- https://github.com/OpenMotionLab/MotionGPT

**Related Work:**
- VQ-VAE: "Neural Discrete Representation Learning" (van den Oord et al.)
- T5: "Exploring the Limits of Transfer Learning" (Raffel et al.)
- HumanML3D: "Generating Diverse and Natural 3D Human Motions from Text" (Guo et al.)

---

**Summary:** AnimationGPT learns by training a transformer model on 8,700 paired text-motion examples, learning to predict motion token sequences that match text descriptions. At inference, it generates new animations by auto-regressively predicting motion tokens conditioned on input text.
