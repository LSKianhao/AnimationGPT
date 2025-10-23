"""
Temporal filtering and smoothing operations for motion data.

This module provides functions to improve motion quality by:
1. Applying Gaussian smoothing along the temporal dimension
2. Clipping unrealistic joint velocities
3. Filtering high-frequency jitter
"""

import numpy as np
import scipy.ndimage as ndimage
from typing import Optional


def apply_temporal_smoothing(joints: np.ndarray, sigma: float = 1.5,
                             preserve_root: bool = True) -> np.ndarray:
    """
    Apply Gaussian smoothing along temporal dimension to reduce jitter.

    Args:
        joints: Joint positions (T, J, 3) where T=frames, J=joints
        sigma: Standard deviation for Gaussian kernel. Higher = more smoothing.
               Typical values: 1.0-2.0
        preserve_root: If True, don't smooth root joint (index 0) to maintain
                      trajectory

    Returns:
        Smoothed joint positions with same shape as input

    Example:
        >>> data = np.load('animation.npy').reshape(-1, 22, 3)
        >>> smoothed = apply_temporal_smoothing(data, sigma=1.5)
    """
    smoothed = np.zeros_like(joints)

    # Determine which joints to smooth
    start_joint = 1 if preserve_root else 0

    for j in range(start_joint, joints.shape[1]):
        for d in range(3):  # x, y, z dimensions
            smoothed[:, j, d] = ndimage.gaussian_filter1d(
                joints[:, j, d],
                sigma=sigma,
                mode='nearest'  # Handle boundaries by extending edge values
            )

    # Copy root if preserved
    if preserve_root:
        smoothed[:, 0, :] = joints[:, 0, :]

    return smoothed


def clip_joint_velocities(joints: np.ndarray, max_velocity: float = 0.5,
                          preserve_root: bool = True) -> np.ndarray:
    """
    Clip unrealistic joint velocities to prevent sudden jumps.

    This helps remove artifacts where joints move too fast between frames,
    which is physically implausible and visually jarring.

    Args:
        joints: Joint positions (T, J, 3)
        max_velocity: Maximum allowed velocity (in units per frame)
                     Typical values: 0.3-0.8
        preserve_root: If True, don't clip root joint velocity

    Returns:
        Joint positions with velocities clipped

    Example:
        >>> data = np.load('animation.npy').reshape(-1, 22, 3)
        >>> clipped = clip_joint_velocities(data, max_velocity=0.5)
    """
    result = np.zeros_like(joints)
    result[0] = joints[0].copy()

    start_joint = 1 if preserve_root else 0

    for t in range(1, len(joints)):
        for j in range(start_joint, joints.shape[1]):
            # Compute velocity
            velocity = joints[t, j] - result[t-1, j]
            velocity_mag = np.linalg.norm(velocity)

            # Clip if exceeds threshold
            if velocity_mag > max_velocity:
                velocity = velocity * (max_velocity / velocity_mag)

            result[t, j] = result[t-1, j] + velocity

        # Copy root if preserved
        if preserve_root:
            result[t, 0] = joints[t, 0]

    return result


def apply_savgol_filter(joints: np.ndarray, window_length: int = 5,
                        polyorder: int = 2, preserve_root: bool = True) -> np.ndarray:
    """
    Apply Savitzky-Golay filter for smoothing while preserving peaks.

    Savitzky-Golay filtering is better at preserving sharp features
    compared to Gaussian smoothing, making it useful for combat animations
    where quick movements are important.

    Args:
        joints: Joint positions (T, J, 3)
        window_length: Length of filter window (must be odd). Typical: 5-11
        polyorder: Order of polynomial fit. Typical: 2-3
        preserve_root: If True, don't filter root joint

    Returns:
        Filtered joint positions

    Example:
        >>> data = np.load('animation.npy').reshape(-1, 22, 3)
        >>> filtered = apply_savgol_filter(data, window_length=7, polyorder=2)
    """
    from scipy.signal import savgol_filter

    if window_length % 2 == 0:
        window_length += 1  # Must be odd

    if window_length > len(joints):
        window_length = len(joints) if len(joints) % 2 == 1 else len(joints) - 1

    smoothed = np.zeros_like(joints)
    start_joint = 1 if preserve_root else 0

    for j in range(start_joint, joints.shape[1]):
        for d in range(3):
            smoothed[:, j, d] = savgol_filter(
                joints[:, j, d],
                window_length=window_length,
                polyorder=polyorder,
                mode='nearest'
            )

    if preserve_root:
        smoothed[:, 0, :] = joints[:, 0, :]

    return smoothed


def compute_acceleration_magnitude(joints: np.ndarray) -> np.ndarray:
    """
    Compute acceleration magnitude for each joint (for analysis/metrics).

    High acceleration values indicate sudden jerky motion.

    Args:
        joints: Joint positions (T, J, 3)

    Returns:
        Acceleration magnitudes (T-2, J)
    """
    velocities = joints[1:] - joints[:-1]
    accelerations = velocities[1:] - velocities[:-1]
    return np.linalg.norm(accelerations, axis=-1)


def filter_high_frequency_noise(joints: np.ndarray, cutoff_freq: float = 0.3,
                                fps: int = 30, preserve_root: bool = True) -> np.ndarray:
    """
    Apply low-pass filter to remove high-frequency noise.

    Uses a Butterworth filter to remove high-frequency jitter while
    preserving the natural motion.

    Args:
        joints: Joint positions (T, J, 3)
        cutoff_freq: Cutoff frequency as fraction of Nyquist frequency (0-1)
                    Lower = more aggressive filtering. Typical: 0.2-0.4
        fps: Frame rate of animation (for proper frequency calculation)
        preserve_root: If True, don't filter root joint

    Returns:
        Filtered joint positions

    Example:
        >>> data = np.load('animation.npy').reshape(-1, 22, 3)
        >>> filtered = filter_high_frequency_noise(data, cutoff_freq=0.3, fps=30)
    """
    from scipy.signal import butter, filtfilt

    # Design low-pass filter
    b, a = butter(N=2, Wn=cutoff_freq, btype='low')

    smoothed = np.zeros_like(joints)
    start_joint = 1 if preserve_root else 0

    for j in range(start_joint, joints.shape[1]):
        for d in range(3):
            # Apply zero-phase filter (forward and backward)
            smoothed[:, j, d] = filtfilt(b, a, joints[:, j, d])

    if preserve_root:
        smoothed[:, 0, :] = joints[:, 0, :]

    return smoothed


def apply_motion_smoothing_pipeline(joints: np.ndarray,
                                    gaussian_sigma: float = 1.5,
                                    max_velocity: float = 0.5,
                                    use_savgol: bool = False,
                                    savgol_window: int = 7,
                                    preserve_root: bool = True) -> np.ndarray:
    """
    Apply a complete smoothing pipeline combining multiple techniques.

    This is the recommended high-level function for improving motion quality.

    Args:
        joints: Joint positions (T, J, 3)
        gaussian_sigma: Sigma for Gaussian smoothing (0 to disable)
        max_velocity: Maximum velocity for clipping (0 to disable)
        use_savgol: If True, use Savitzky-Golay instead of Gaussian
        savgol_window: Window size for Savitzky-Golay filter
        preserve_root: If True, don't modify root joint

    Returns:
        Smoothed joint positions

    Example:
        >>> data = np.load('animation.npy').reshape(-1, 22, 3)
        >>> # Conservative smoothing (preserve details)
        >>> smoothed = apply_motion_smoothing_pipeline(
        ...     data, gaussian_sigma=1.0, max_velocity=0.6
        ... )
        >>> # Aggressive smoothing (remove more jitter)
        >>> smoothed = apply_motion_smoothing_pipeline(
        ...     data, gaussian_sigma=2.0, max_velocity=0.4
        ... )
    """
    result = joints.copy()

    # Step 1: Temporal smoothing
    if use_savgol:
        result = apply_savgol_filter(result, window_length=savgol_window,
                                    polyorder=2, preserve_root=preserve_root)
    elif gaussian_sigma > 0:
        result = apply_temporal_smoothing(result, sigma=gaussian_sigma,
                                         preserve_root=preserve_root)

    # Step 2: Velocity clipping
    if max_velocity > 0:
        result = clip_joint_velocities(result, max_velocity=max_velocity,
                                      preserve_root=preserve_root)

    return result


def analyze_motion_smoothness(joints: np.ndarray, fps: int = 30) -> dict:
    """
    Analyze motion smoothness metrics for quality assessment.

    Args:
        joints: Joint positions (T, J, 3)
        fps: Frame rate for velocity/acceleration calculation

    Returns:
        Dictionary with smoothness metrics:
        - mean_velocity: Average joint velocity
        - max_velocity: Maximum joint velocity
        - mean_acceleration: Average acceleration magnitude
        - max_acceleration: Maximum acceleration magnitude
        - jerk_score: Average jerk (rate of acceleration change)

    Example:
        >>> data = np.load('animation.npy').reshape(-1, 22, 3)
        >>> metrics = analyze_motion_smoothness(data)
        >>> print(f"Mean velocity: {metrics['mean_velocity']:.3f}")
        >>> print(f"Jerk score: {metrics['jerk_score']:.3f}")
    """
    # Compute velocities
    velocities = (joints[1:] - joints[:-1]) * fps
    velocity_magnitudes = np.linalg.norm(velocities, axis=-1)

    # Compute accelerations
    accelerations = (velocities[1:] - velocities[:-1]) * fps
    acceleration_magnitudes = np.linalg.norm(accelerations, axis=-1)

    # Compute jerk (rate of acceleration change)
    jerk = (accelerations[1:] - accelerations[:-1]) * fps
    jerk_magnitudes = np.linalg.norm(jerk, axis=-1)

    return {
        'mean_velocity': np.mean(velocity_magnitudes),
        'max_velocity': np.max(velocity_magnitudes),
        'std_velocity': np.std(velocity_magnitudes),
        'mean_acceleration': np.mean(acceleration_magnitudes),
        'max_acceleration': np.max(acceleration_magnitudes),
        'std_acceleration': np.std(acceleration_magnitudes),
        'jerk_score': np.mean(jerk_magnitudes),
        'max_jerk': np.max(jerk_magnitudes)
    }


if __name__ == "__main__":
    # Example usage
    print("Temporal Filters Module")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    import numpy as np
    from temporal_filters import apply_motion_smoothing_pipeline

    # Load animation data
    data = np.load('animation.npy').reshape(-1, 22, 3)

    # Apply smoothing pipeline
    smoothed = apply_motion_smoothing_pipeline(
        data,
        gaussian_sigma=1.5,
        max_velocity=0.5,
        preserve_root=True
    )

    # Save result
    np.save('animation_smoothed.npy', smoothed)
    """)
