"""
Physical plausibility metrics for motion quality evaluation.

This module provides metrics to assess the physical realism of generated animations:
- Foot sliding detection and quantification
- Joint velocity analysis
- Acceleration/jerk measurement
- Ground penetration detection
- Balance and stability metrics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def detect_foot_contacts(joints: np.ndarray,
                         fid_l: Tuple[int, int] = (3, 4),
                         fid_r: Tuple[int, int] = (7, 8),
                         height_threshold: float = 0.05,
                         velocity_threshold: float = 0.05) -> np.ndarray:
    """
    Detect frames where feet are in contact with ground.

    Args:
        joints: Joint positions (T, J, 3)
        fid_l: Left foot joint indices (ankle, toe)
        fid_r: Right foot joint indices (ankle, toe)
        height_threshold: Maximum height for contact (meters)
        velocity_threshold: Maximum velocity for contact (units/frame)

    Returns:
        Binary contact array (4, T) for [left_ankle, left_toe, right_ankle, right_toe]
    """
    T = len(joints)
    foot_indices = list(fid_l) + list(fid_r)

    contacts = np.zeros((4, T))

    for i, fidx in enumerate(foot_indices):
        # Height criterion
        heights = joints[:, fidx, 1]  # Y-axis
        is_low = heights < height_threshold

        # Velocity criterion
        velocities = np.sqrt(np.sum((joints[1:, fidx] - joints[:-1, fidx]) ** 2, axis=-1))
        velocities = np.concatenate([[0], velocities])  # Pad first frame
        is_slow = velocities < velocity_threshold

        # Contact = low AND slow
        contacts[i] = (is_low & is_slow).astype(float)

    return contacts


def compute_foot_sliding(joints: np.ndarray,
                         fid_l: Tuple[int, int] = (3, 4),
                         fid_r: Tuple[int, int] = (7, 8),
                         height_threshold: float = 0.05,
                         velocity_threshold: float = 0.05) -> Dict[str, float]:
    """
    Compute foot sliding metrics.

    Foot sliding occurs when a foot that should be planted on the ground
    moves horizontally. This is a common artifact in generated motion.

    Args:
        joints: Joint positions (T, J, 3)
        fid_l: Left foot joint indices
        fid_r: Right foot joint indices
        height_threshold: Height threshold for contact detection
        velocity_threshold: Velocity threshold for contact detection

    Returns:
        Dictionary with metrics:
        - foot_sliding_total: Total sliding distance
        - foot_sliding_per_frame: Average sliding per frame
        - foot_sliding_per_contact: Average sliding per contact frame
        - contact_frames: Number of frames with foot contact
        - contact_ratio: Fraction of frames with any foot contact
    """
    contacts = detect_foot_contacts(joints, fid_l, fid_r, height_threshold, velocity_threshold)
    foot_indices = list(fid_l) + list(fid_r)

    total_sliding = 0.0
    sliding_frames = 0

    for i, fidx in enumerate(foot_indices):
        # Get frames where this foot is in contact
        in_contact = contacts[i] > 0.5

        # Compute horizontal displacement during contact
        for t in range(1, len(joints)):
            if in_contact[t]:
                # Compute XZ displacement (horizontal)
                displacement = joints[t, fidx, [0, 2]] - joints[t-1, fidx, [0, 2]]
                sliding = np.linalg.norm(displacement)
                total_sliding += sliding
                sliding_frames += 1

    contact_frames = np.sum(np.any(contacts > 0.5, axis=0))
    T = len(joints)

    return {
        'foot_sliding_total': total_sliding,
        'foot_sliding_per_frame': total_sliding / T if T > 0 else 0,
        'foot_sliding_per_contact': total_sliding / sliding_frames if sliding_frames > 0 else 0,
        'contact_frames': int(contact_frames),
        'contact_ratio': contact_frames / T if T > 0 else 0,
    }


def compute_velocity_metrics(joints: np.ndarray,
                             fps: int = 30,
                             exclude_root: bool = True) -> Dict[str, float]:
    """
    Compute joint velocity statistics.

    Args:
        joints: Joint positions (T, J, 3)
        fps: Frame rate for converting to real units
        exclude_root: If True, exclude root joint (index 0) from stats

    Returns:
        Dictionary with velocity metrics:
        - mean_velocity: Average joint velocity
        - max_velocity: Maximum joint velocity
        - std_velocity: Standard deviation of velocities
        - p95_velocity: 95th percentile velocity
        - unrealistic_frames: Frames with unusually high velocity
    """
    # Compute velocities
    velocities = (joints[1:] - joints[:-1]) * fps
    velocity_magnitudes = np.linalg.norm(velocities, axis=-1)

    if exclude_root:
        velocity_magnitudes = velocity_magnitudes[:, 1:]

    # Flatten for statistics
    all_vels = velocity_magnitudes.flatten()

    # Count unrealistic frames (velocity > 10 m/s for any joint)
    unrealistic_threshold = 10.0  # m/s
    unrealistic_frames = np.sum(np.any(velocity_magnitudes > unrealistic_threshold, axis=1))

    return {
        'mean_velocity': float(np.mean(all_vels)),
        'max_velocity': float(np.max(all_vels)),
        'std_velocity': float(np.std(all_vels)),
        'median_velocity': float(np.median(all_vels)),
        'p95_velocity': float(np.percentile(all_vels, 95)),
        'unrealistic_frames': int(unrealistic_frames),
        'unrealistic_ratio': unrealistic_frames / (len(joints) - 1),
    }


def compute_acceleration_metrics(joints: np.ndarray,
                                 fps: int = 30,
                                 exclude_root: bool = True) -> Dict[str, float]:
    """
    Compute acceleration (jerk) metrics.

    High acceleration indicates jerky, unnatural motion.

    Args:
        joints: Joint positions (T, J, 3)
        fps: Frame rate
        exclude_root: If True, exclude root joint

    Returns:
        Dictionary with acceleration metrics:
        - mean_acceleration: Average acceleration magnitude
        - max_acceleration: Maximum acceleration
        - std_acceleration: Standard deviation
        - jerk_score: Average rate of acceleration change
    """
    # Compute velocities and accelerations
    velocities = (joints[1:] - joints[:-1]) * fps
    accelerations = (velocities[1:] - velocities[:-1]) * fps
    acceleration_magnitudes = np.linalg.norm(accelerations, axis=-1)

    if exclude_root:
        acceleration_magnitudes = acceleration_magnitudes[:, 1:]

    # Compute jerk (rate of acceleration change)
    jerk = (accelerations[1:] - accelerations[:-1]) * fps
    jerk_magnitudes = np.linalg.norm(jerk, axis=-1)
    if exclude_root:
        jerk_magnitudes = jerk_magnitudes[:, 1:]

    all_accel = acceleration_magnitudes.flatten()
    all_jerk = jerk_magnitudes.flatten()

    return {
        'mean_acceleration': float(np.mean(all_accel)),
        'max_acceleration': float(np.max(all_accel)),
        'std_acceleration': float(np.std(all_accel)),
        'p95_acceleration': float(np.percentile(all_accel, 95)),
        'jerk_score': float(np.mean(all_jerk)),
        'max_jerk': float(np.max(all_jerk)),
    }


def compute_ground_penetration(joints: np.ndarray,
                               ground_height: float = 0.0) -> Dict[str, float]:
    """
    Detect and quantify ground penetration artifacts.

    Args:
        joints: Joint positions (T, J, 3)
        ground_height: Y-coordinate of ground plane (default 0)

    Returns:
        Dictionary with penetration metrics:
        - min_height: Lowest joint position
        - max_penetration: Maximum penetration depth
        - mean_penetration: Average penetration (over penetrating frames)
        - penetration_frames: Number of frames with any penetration
        - penetration_ratio: Fraction of frames with penetration
    """
    # Get minimum height across all joints and frames
    all_heights = joints[:, :, 1]  # Y-axis
    min_heights_per_frame = np.min(all_heights, axis=1)

    # Find penetrations
    penetrations = ground_height - min_heights_per_frame
    penetrations = np.maximum(penetrations, 0)  # Only negative heights

    penetration_frames = np.sum(penetrations > 0)
    T = len(joints)

    return {
        'min_height': float(np.min(all_heights)),
        'max_penetration': float(np.max(penetrations)),
        'mean_penetration': float(np.mean(penetrations[penetrations > 0])) if penetration_frames > 0 else 0.0,
        'penetration_frames': int(penetration_frames),
        'penetration_ratio': penetration_frames / T if T > 0 else 0,
    }


def compute_balance_metrics(joints: np.ndarray,
                            root_idx: int = 0,
                            foot_indices: List[int] = [3, 4, 7, 8]) -> Dict[str, float]:
    """
    Compute balance and stability metrics.

    Args:
        joints: Joint positions (T, J, 3)
        root_idx: Index of root/pelvis joint
        foot_indices: Indices of foot joints

    Returns:
        Dictionary with balance metrics:
        - mean_com_height: Average height of center of mass
        - com_sway: Average CoM lateral movement
        - foot_distance: Average distance between feet
    """
    # Approximate center of mass as root joint
    com = joints[:, root_idx, :]
    com_height = com[:, 1]  # Y-axis

    # Compute lateral sway (XZ plane movement)
    com_xz = com[:, [0, 2]]
    com_velocity_xz = np.linalg.norm(com_xz[1:] - com_xz[:-1], axis=-1)

    # Compute average foot distance
    left_foot = (joints[:, foot_indices[0]] + joints[:, foot_indices[1]]) / 2
    right_foot = (joints[:, foot_indices[2]] + joints[:, foot_indices[3]]) / 2
    foot_distance = np.linalg.norm(left_foot - right_foot, axis=-1)

    return {
        'mean_com_height': float(np.mean(com_height)),
        'std_com_height': float(np.std(com_height)),
        'com_sway': float(np.mean(com_velocity_xz)),
        'mean_foot_distance': float(np.mean(foot_distance)),
        'std_foot_distance': float(np.std(foot_distance)),
    }


def compute_all_metrics(joints: np.ndarray,
                       fps: int = 30,
                       fid_l: Tuple[int, int] = (3, 4),
                       fid_r: Tuple[int, int] = (7, 8)) -> Dict[str, Dict[str, float]]:
    """
    Compute all physical plausibility metrics.

    Args:
        joints: Joint positions (T, J, 3)
        fps: Frame rate
        fid_l: Left foot joint indices
        fid_r: Right foot joint indices

    Returns:
        Dictionary with all metrics organized by category:
        - foot_sliding: Foot sliding metrics
        - velocity: Velocity metrics
        - acceleration: Acceleration/jerk metrics
        - ground: Ground penetration metrics
        - balance: Balance metrics
    """
    return {
        'foot_sliding': compute_foot_sliding(joints, fid_l, fid_r),
        'velocity': compute_velocity_metrics(joints, fps),
        'acceleration': compute_acceleration_metrics(joints, fps),
        'ground': compute_ground_penetration(joints),
        'balance': compute_balance_metrics(joints),
    }


def print_metrics_report(metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted report of physical metrics.

    Args:
        metrics: Metrics dictionary from compute_all_metrics()
    """
    print("\n" + "=" * 60)
    print(" PHYSICAL PLAUSIBILITY METRICS REPORT")
    print("=" * 60)

    print("\n📏 FOOT SLIDING:")
    fs = metrics['foot_sliding']
    print(f"  Total sliding:        {fs['foot_sliding_total']:.4f} units")
    print(f"  Per frame:            {fs['foot_sliding_per_frame']:.4f} units")
    print(f"  Per contact frame:    {fs['foot_sliding_per_contact']:.4f} units")
    print(f"  Contact frames:       {fs['contact_frames']} ({fs['contact_ratio']*100:.1f}%)")

    print("\n🚀 VELOCITY:")
    vel = metrics['velocity']
    print(f"  Mean velocity:        {vel['mean_velocity']:.3f} m/s")
    print(f"  Max velocity:         {vel['max_velocity']:.3f} m/s")
    print(f"  95th percentile:      {vel['p95_velocity']:.3f} m/s")
    print(f"  Unrealistic frames:   {vel['unrealistic_frames']} ({vel['unrealistic_ratio']*100:.1f}%)")

    print("\n⚡ ACCELERATION:")
    acc = metrics['acceleration']
    print(f"  Mean acceleration:    {acc['mean_acceleration']:.3f} m/s²")
    print(f"  Max acceleration:     {acc['max_acceleration']:.3f} m/s²")
    print(f"  Jerk score:           {acc['jerk_score']:.3f} m/s³")

    print("\n🌍 GROUND INTERACTION:")
    gnd = metrics['ground']
    print(f"  Min height:           {gnd['min_height']:.4f} m")
    print(f"  Max penetration:      {gnd['max_penetration']:.4f} m")
    if gnd['penetration_frames'] > 0:
        print(f"  Penetration frames:   {gnd['penetration_frames']} ({gnd['penetration_ratio']*100:.1f}%)")
    else:
        print(f"  Penetration frames:   None ✓")

    print("\n⚖️  BALANCE:")
    bal = metrics['balance']
    print(f"  Mean CoM height:      {bal['mean_com_height']:.3f} m")
    print(f"  CoM sway:             {bal['com_sway']:.4f} m/frame")
    print(f"  Foot distance:        {bal['mean_foot_distance']:.3f} m")

    print("\n" + "=" * 60 + "\n")


def compare_metrics(before_metrics: Dict[str, Dict[str, float]],
                   after_metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Print a comparison of metrics before and after processing.

    Args:
        before_metrics: Metrics from original animation
        after_metrics: Metrics from processed animation
    """
    print("\n" + "=" * 70)
    print(" BEFORE vs AFTER COMPARISON")
    print("=" * 70)

    def print_comparison(category: str, key: str, label: str, lower_is_better: bool = True):
        before = before_metrics[category][key]
        after = after_metrics[category][key]
        diff = after - before
        pct = (diff / before * 100) if before != 0 else 0

        arrow = "↓" if (diff < 0) == lower_is_better else "↑"
        symbol = "✓" if (diff < 0) == lower_is_better else "✗"

        print(f"  {label:30s} {before:8.4f} → {after:8.4f} ({diff:+.4f}, {pct:+.1f}%) {arrow} {symbol}")

    print("\n📏 FOOT SLIDING:")
    print_comparison('foot_sliding', 'foot_sliding_per_frame', 'Per frame sliding', lower_is_better=True)
    print_comparison('foot_sliding', 'foot_sliding_per_contact', 'Per contact sliding', lower_is_better=True)

    print("\n🚀 VELOCITY:")
    print_comparison('velocity', 'mean_velocity', 'Mean velocity', lower_is_better=False)
    print_comparison('velocity', 'max_velocity', 'Max velocity', lower_is_better=True)
    print_comparison('velocity', 'std_velocity', 'Velocity variance', lower_is_better=True)

    print("\n⚡ ACCELERATION:")
    print_comparison('acceleration', 'jerk_score', 'Jerk score', lower_is_better=True)
    print_comparison('acceleration', 'max_acceleration', 'Max acceleration', lower_is_better=True)

    print("\n🌍 GROUND:")
    print_comparison('ground', 'max_penetration', 'Max penetration', lower_is_better=True)

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Physical Metrics Module")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    import numpy as np
    from physical_metrics import compute_all_metrics, print_metrics_report

    # Load animation
    data = np.load('animation.npy').reshape(-1, 22, 3)

    # Compute metrics
    metrics = compute_all_metrics(data, fps=30)

    # Print report
    print_metrics_report(metrics)

    # Or access specific metrics
    foot_sliding = metrics['foot_sliding']['foot_sliding_per_frame']
    print(f"Foot sliding: {foot_sliding:.4f}")
    """)
