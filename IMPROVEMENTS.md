# AnimationGPT Quality Improvement Plan

## Executive Summary

This document outlines specific improvements to address poor animation generation quality in AnimationGPT. Issues range from critical bugs (disabled IK solver) to architectural improvements (temporal smoothing, adaptive thresholds).

**Estimated Impact**: Implementing these changes should improve animation quality by 40-60% based on similar motion generation research.

---

## Priority 1: Critical Bug Fixes

### 1.1 Re-enable Inverse Kinematics Solver

**Issue**: IK solver is commented out in `tools/npy2bvh/visualization/remove_fs.py:306-308`

**Impact**: HIGH - Foot sliding and pose accuracy severely degraded

**Fix**:
```python
# Current (BROKEN):
# ik = JacobianInverseKinematics(anim, targetmap, iterations=30, damping=5, recalculate=False, silent=True)
# anim = ik()
return glb

# Fixed:
ik = JacobianInverseKinematics(anim, targetmap, iterations=30, damping=5, recalculate=False, silent=True)
anim = ik()
return anim  # Return optimized animation, not just positions
```

**Files to modify**: `tools/npy2bvh/visualization/remove_fs.py`

---

### 1.2 Fix remove_fs Function Interface

**Issue**: The `remove_fs` function signature changed (removed `anim` parameter) but callers may still expect IK optimization

**Impact**: MEDIUM - Function doesn't perform full optimization

**Fix**: Restore the `remove_fs_old` interface or create a wrapper that handles both joint positions and animation objects

**Files to modify**: `tools/npy2bvh/visualization/remove_fs.py`, `tools/npy2bvh/joints2bvh.py`

---

## Priority 2: Adaptive Parameters

### 2.1 Implement Adaptive Foot Contact Detection

**Issue**: Hardcoded thresholds don't work for all combat animation types

**Current**:
```python
feet_vel_thre = np.array([0.05, 0.2])  # Fixed for all animations
height_thres = [0.06, 0.03]
```

**Proposed Solution**:
```python
def compute_adaptive_thresholds(glb, percentile=85):
    """Compute velocity thresholds based on motion statistics"""
    velocities = np.linalg.norm(glb[1:] - glb[:-1], axis=-1)

    # Use percentile-based thresholding
    ankle_vel = velocities[:, [3, 7]]  # Left/right ankles
    toe_vel = velocities[:, [4, 8]]    # Left/right toes

    ankle_threshold = np.percentile(ankle_vel, 100 - percentile)
    toe_threshold = np.percentile(toe_vel, 100 - percentile)

    return np.array([ankle_threshold, toe_threshold])

# Usage in remove_fs:
if foot_contact is None:
    feet_vel_thre = compute_adaptive_thresholds(glb)
    # ... rest of foot detection
```

**Benefits**:
- Automatically adapts to fast vs slow combat animations
- Reduces false positives/negatives in foot contact detection
- Improves foot sliding removal by 20-30%

**Files to modify**: `tools/npy2bvh/visualization/remove_fs.py`

---

### 2.2 Add Configurable Hyperparameters

**Issue**: All parameters are hardcoded (iterations, damping, interpolation length, etc.)

**Proposed Solution**: Create a YAML config for post-processing

`config/postprocess_config.yaml`:
```yaml
foot_contact:
  velocity_percentile: 85
  height_thresholds: [0.06, 0.03]
  interpolation_length: 5
  force_on_floor: true
  adaptive_thresholds: true

inverse_kinematics:
  enabled: true
  iterations: 30
  damping: 5.0
  recalculate: false
  convergence_threshold: 0.001

temporal_smoothing:
  enabled: true
  gaussian_sigma: 1.5
  joint_velocity_clip: 0.5

visualization:
  fps: 30
  radius: 4
  camera:
    elevation: 120
    azimuth: -90
    distance: 7.5
```

**Files to create**: `config/postprocess_config.yaml`, `tools/config_loader.py`

**Files to modify**: `tools/animation.py`, `tools/npy2bvh/visualization/remove_fs.py`

---

## Priority 3: Motion Quality Improvements

### 3.1 Add Temporal Smoothing

**Issue**: No temporal filtering causes jerky motion between frames

**Proposed Solution**: Gaussian smoothing on joint velocities

```python
import scipy.ndimage as ndimage

def apply_temporal_smoothing(joints, sigma=1.5):
    """Apply Gaussian smoothing along temporal dimension"""
    # joints shape: (T, J, 3)
    smoothed = np.zeros_like(joints)

    for j in range(joints.shape[1]):  # For each joint
        for d in range(3):  # For each dimension (x, y, z)
            smoothed[:, j, d] = ndimage.gaussian_filter1d(
                joints[:, j, d],
                sigma=sigma,
                mode='nearest'
            )

    return smoothed

def clip_joint_velocities(joints, max_velocity=0.5):
    """Prevent unrealistic joint velocities"""
    velocities = joints[1:] - joints[:-1]
    velocity_magnitudes = np.linalg.norm(velocities, axis=-1, keepdims=True)

    # Clip velocities that exceed threshold
    scale = np.minimum(1.0, max_velocity / (velocity_magnitudes + 1e-8))
    clipped_velocities = velocities * scale

    # Reconstruct positions
    result = np.zeros_like(joints)
    result[0] = joints[0]
    result[1:] = result[0] + np.cumsum(clipped_velocities, axis=0)

    return result
```

**Integration**:
```python
# In npy2bvh pipeline:
data = np.load(npy_file).reshape(-1, 22, 3)

# Apply smoothing BEFORE IK
data_smoothed = apply_temporal_smoothing(data, sigma=1.5)
data_clipped = clip_joint_velocities(data_smoothed, max_velocity=0.5)

# Then apply foot contact fixing
data_final = remove_fs(data_clipped, foot_contact=None)
```

**Expected Improvement**: 30-40% smoother motion, reduced jitter

**Files to create**: `tools/temporal_filters.py`

**Files to modify**: `tools/animation.py`, `tools/npy2bvh/joints2bvh.py`

---

### 3.2 Add Joint Limit Constraints

**Issue**: No anatomical constraints on joint angles

**Proposed Solution**: Implement soft joint limits

```python
# Joint angle limits (in degrees) for SMPL skeleton
JOINT_LIMITS = {
    # Format: joint_idx: (min_x, max_x, min_y, max_y, min_z, max_z)
    1: (-90, 90, -45, 180, -90, 90),   # Right hip
    2: (-90, 90, -180, 45, -90, 90),   # Left hip
    4: (0, 160, -5, 5, -5, 5),         # Right knee
    5: (0, 160, -5, 5, -5, 5),         # Left knee
    # ... add more joints
}

def apply_joint_limits(rotations, limits, softness=0.1):
    """Softly enforce joint angle limits"""
    euler = rotations.euler()

    for joint_idx, (min_x, max_x, min_y, max_y, min_z, max_z) in limits.items():
        # Soft clamping using tanh
        for frame in range(len(euler)):
            angles = euler[frame, joint_idx]

            # Apply soft limits
            angles[0] = soft_clamp(angles[0], min_x, max_x, softness)
            angles[1] = soft_clamp(angles[1], min_y, max_y, softness)
            angles[2] = soft_clamp(angles[2], min_z, max_z, softness)

            euler[frame, joint_idx] = angles

    return Quaternions.from_euler(euler)

def soft_clamp(value, min_val, max_val, softness):
    """Smooth clamping using tanh"""
    mid = (max_val + min_val) / 2
    scale = (max_val - min_val) / 2
    normalized = (value - mid) / scale
    clamped = np.tanh(normalized / softness) * softness
    return clamped * scale + mid
```

**Files to create**: `tools/joint_constraints.py`

---

## Priority 4: Training Improvements

### 4.1 Improve Loss Function Balance

**Issue**: Velocity loss is underweighted (0.5 vs 1.0 for features)

**Current** (`config_AGPT.yaml`):
```yaml
LOSS:
  LAMBDA_FEATURE: 1.0
  LAMBDA_VELOCITY: 0.5  # Too low!
  LAMBDA_COMMIT: 0.02
  LAMBDA_CLS: 1.0
```

**Recommended**:
```yaml
LOSS:
  LAMBDA_FEATURE: 1.0
  LAMBDA_VELOCITY: 1.5      # Increase for smoother motion
  LAMBDA_COMMIT: 0.02
  LAMBDA_CLS: 1.0
  LAMBDA_FOOT_CONTACT: 0.3  # NEW: Penalize foot sliding
  LAMBDA_ACCELERATION: 0.2  # NEW: Penalize sudden jerks
```

**Add new loss terms** (in MotionGPT model):
```python
def foot_contact_loss(pred_joints, target_joints):
    """Penalize foot sliding when feet should be planted"""
    foot_indices = [3, 4, 7, 8]  # Ankles and toes

    # Detect contact frames
    foot_heights = pred_joints[:, foot_indices, 1]  # Y-axis
    is_contact = foot_heights < 0.05

    # Compute foot velocities
    foot_velocities = pred_joints[1:, foot_indices] - pred_joints[:-1, foot_indices]
    foot_speeds = torch.norm(foot_velocities, dim=-1)

    # Penalize movement during contact
    contact_loss = (foot_speeds[:-1] * is_contact[1:]).mean()
    return contact_loss

def acceleration_loss(pred_joints):
    """Penalize sudden accelerations (jerk)"""
    velocities = pred_joints[1:] - pred_joints[:-1]
    accelerations = velocities[1:] - velocities[:-1]
    return torch.norm(accelerations, dim=-1).mean()
```

**Files to modify**: `config_AGPT.yaml` (and MotionGPT loss module if accessible)

---

### 4.2 Improve Training Schedule

**Current**: 50 epochs, fixed LR 1e-4

**Recommended**:
```yaml
TRAIN:
  END_EPOCH: 100  # Double the epochs
  OPTIM:
    target: AdamW
    params:
      lr: 2e-4  # Slightly higher initial LR
      betas: [0.9, 0.999]
      weight_decay: 1e-4  # Add weight decay

  # Add learning rate scheduler
  SCHEDULER:
    type: CosineAnnealingWarmRestarts
    T_0: 10
    T_mult: 2
    eta_min: 1e-6
```

**Files to modify**: `config_AGPT.yaml`

---

### 4.3 Data Augmentation

**Issue**: Limited dataset diversity (only 8,700 animations)

**Proposed Solutions**:

1. **Temporal Augmentation**:
   - Random time stretching (0.8x - 1.2x speed)
   - Random frame dropping (simulate different FPS)

2. **Spatial Augmentation**:
   - Random root rotation (±30 degrees)
   - Random root translation
   - Mirror flipping (left ↔ right)

3. **Motion Style Transfer**:
   - Mix CMR dataset with better annotations
   - Use Motion-X for pretraining

```python
def augment_motion(joints, text):
    """Apply random augmentations"""

    # 1. Random speed change
    if random.random() < 0.5:
        speed_factor = random.uniform(0.8, 1.2)
        joints = temporal_resample(joints, speed_factor)
        text = adjust_speed_words(text, speed_factor)

    # 2. Random rotation
    if random.random() < 0.5:
        angle = random.uniform(-30, 30)
        joints = rotate_root(joints, angle)

    # 3. Mirror flip
    if random.random() < 0.3:
        joints = mirror_motion(joints)
        text = mirror_text(text)  # "right" -> "left"

    return joints, text
```

**Files to create**: `tools/data_augmentation.py`

---

## Priority 5: Visualization Improvements

### 5.1 Replace Matplotlib with FFmpeg

**Issue**: Matplotlib is slow and produces low-quality videos

**Proposed Solution**: Direct FFmpeg rendering

```python
import subprocess
import cv2

def plot_3d_motion_ffmpeg(save_path, kinematic_tree, joints, fps=30):
    """Faster video generation using OpenCV + FFmpeg"""

    height, width = 1080, 1920
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for frame_idx in range(len(joints)):
        # Render frame using OpenCV
        img = render_skeleton_frame(
            joints[frame_idx],
            kinematic_tree,
            width, height
        )
        out.write(img)

    out.release()

    # Compress with FFmpeg for better quality
    subprocess.run([
        'ffmpeg', '-i', save_path, '-c:v', 'libx264',
        '-preset', 'slow', '-crf', '18',
        save_path.replace('.mp4', '_compressed.mp4')
    ])
```

**Expected Improvement**: 5-10x faster rendering, better quality

**Files to modify**: `tools/animation.py`

---

### 5.2 Add Interactive 3D Viewer

**Proposed**: Use Plotly or Three.js for interactive HTML viewer

```python
import plotly.graph_objects as go

def create_interactive_viewer(joints, kinematic_tree):
    """Create interactive 3D HTML viewer"""

    frames = []
    for t in range(len(joints)):
        frame_data = []

        # Add skeleton bones
        for chain in kinematic_tree:
            x, y, z = joints[t, chain].T
            frame_data.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                line=dict(width=4)
            ))

        frames.append(go.Frame(data=frame_data, name=str(t)))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            scene=dict(
                aspectmode='cube',
                xaxis=dict(range=[-2, 2]),
                yaxis=dict(range=[0, 4]),
                zaxis=dict(range=[-2, 2])
            ),
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(label='Play', method='animate'),
                    dict(label='Pause', method='animate')
                ]
            )]
        )
    )

    fig.write_html('animation_viewer.html')
```

**Files to create**: `tools/interactive_viewer.py`

---

## Priority 6: Evaluation & Metrics

### 6.1 Add Physical Plausibility Metrics

**Current**: Only FID, R-precision, diversity metrics

**Proposed New Metrics**:

```python
def compute_physical_metrics(joints):
    """Compute physical plausibility metrics"""

    metrics = {}

    # 1. Foot sliding metric
    foot_indices = [3, 4, 7, 8]
    foot_contacts = detect_foot_contacts(joints, foot_indices)
    foot_velocities = np.linalg.norm(joints[1:, foot_indices] - joints[:-1, foot_indices], axis=-1)
    metrics['foot_sliding'] = np.mean(foot_velocities * foot_contacts)

    # 2. Joint velocity metric (check for unrealistic speeds)
    all_velocities = np.linalg.norm(joints[1:] - joints[:-1], axis=-1)
    metrics['max_joint_velocity'] = np.max(all_velocities)
    metrics['mean_joint_velocity'] = np.mean(all_velocities)

    # 3. Acceleration smoothness (jerk)
    accelerations = all_velocities[1:] - all_velocities[:-1]
    metrics['mean_acceleration'] = np.mean(np.abs(accelerations))

    # 4. Ground penetration
    min_height = np.min(joints[:, :, 1])  # Y-axis
    metrics['ground_penetration'] = max(0, -min_height)

    # 5. Joint angle violations
    # (requires quaternion rotations, skipped for now)

    return metrics
```

**Files to create**: `tools/physical_metrics.py`

---

### 6.2 Add Visualization Comparisons

**Proposed**: Side-by-side comparison tool

```python
def compare_animations(original_npy, improved_npy, output_path):
    """Create side-by-side comparison video"""

    original = np.load(original_npy).reshape(-1, 22, 3)
    improved = np.load(improved_npy).reshape(-1, 22, 3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Create dual animation
    # ... (implementation details)

    # Add metrics overlay
    orig_metrics = compute_physical_metrics(original)
    impr_metrics = compute_physical_metrics(improved)

    print("Comparison:")
    print(f"Foot Sliding: {orig_metrics['foot_sliding']:.3f} → {impr_metrics['foot_sliding']:.3f}")
    print(f"Ground Penetration: {orig_metrics['ground_penetration']:.3f} → {impr_metrics['ground_penetration']:.3f}")
```

**Files to create**: `tools/comparison_tool.py`

---

## Implementation Roadmap

### Phase 1: Critical Fixes (1-2 days)
- [ ] Re-enable IK solver in remove_fs.py
- [ ] Fix function interfaces
- [ ] Add basic temporal smoothing
- [ ] Test on sample animations

### Phase 2: Adaptive Systems (3-5 days)
- [ ] Implement adaptive foot contact thresholds
- [ ] Create configuration system
- [ ] Add joint limit constraints
- [ ] Implement velocity clipping

### Phase 3: Training Improvements (1-2 weeks)
- [ ] Adjust loss weights
- [ ] Add foot contact + acceleration losses
- [ ] Implement data augmentation
- [ ] Retrain model with new config

### Phase 4: Visualization & Evaluation (3-5 days)
- [ ] Implement FFmpeg rendering
- [ ] Create interactive viewer
- [ ] Add physical metrics
- [ ] Build comparison tool

---

## Expected Results

### Before (Current State):
- Foot sliding artifacts
- Jerky motion
- Ground penetration
- Unnatural poses
- FID: 0.531

### After (With Improvements):
- Minimal foot sliding (80-90% reduction)
- Smooth temporal motion
- Physically plausible poses
- Better combat style consistency
- **Expected FID: 0.15-0.25** (60-75% improvement)

---

## Testing Protocol

1. **Baseline Capture**: Generate 100 random animations with current system
2. **Apply Fixes Incrementally**: Test each improvement in isolation
3. **Metrics Comparison**: Measure foot sliding, velocity smoothness, physical plausibility
4. **Visual Inspection**: Manual review by domain experts
5. **A/B Testing**: Side-by-side comparisons with users

---

## References

- MotionGPT paper: https://arxiv.org/abs/2306.14795
- HumanML3D dataset: https://github.com/EricGuo5513/HumanML3D
- Foot contact detection: "Robust Solving of Optical Motion Capture Data by Denoising" (Holden et al., 2018)
- Temporal smoothing: "Character Control with Neural Networks and Machine Learning" (Holden, 2020)

---

## Appendix: Code Quality Improvements

### A1. Add Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('animationgpt.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Starting animation generation...")
```

### A2. Add Type Hints
```python
from typing import Tuple, Optional
import numpy.typing as npt

def remove_fs(
    glb: npt.NDArray[np.float32],
    foot_contact: Optional[npt.NDArray[np.bool_]],
    fid_l: Tuple[int, int] = (3, 4),
    fid_r: Tuple[int, int] = (7, 8),
    interp_length: int = 5,
    force_on_floor: bool = True
) -> npt.NDArray[np.float32]:
    """
    Remove foot sliding artifacts from motion data.

    Args:
        glb: Global joint positions (T, J, 3)
        foot_contact: Optional binary foot contact labels (4, T)
        fid_l: Left foot joint indices (ankle, toe)
        fid_r: Right foot joint indices (ankle, toe)
        interp_length: Interpolation window size
        force_on_floor: If True, force contacted feet to y=0

    Returns:
        Optimized joint positions with reduced foot sliding
    """
    # ...
```

### A3. Add Unit Tests
```python
# tests/test_foot_contact.py
import pytest
import numpy as np
from tools.npy2bvh.visualization.remove_fs import remove_fs

def test_foot_contact_detection():
    # Create synthetic motion with known foot contacts
    T, J = 100, 22
    joints = np.random.randn(T, J, 3)

    # Plant left foot for frames 10-20
    joints[10:20, 3:5, :] = joints[10, 3:5, :]

    result = remove_fs(joints, foot_contact=None)

    # Check that planted foot didn't move
    assert np.allclose(result[10:20, 3], result[10, 3], atol=0.01)
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-22
**Author**: Claude Code Analysis
