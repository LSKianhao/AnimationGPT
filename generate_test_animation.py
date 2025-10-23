"""
Generate a synthetic test animation with common quality issues.

This creates a simple walking animation with intentional artifacts:
- Foot sliding
- Temporal jitter
- Unrealistic velocities

Used to demonstrate the quality improvement pipeline.
"""

import numpy as np

def generate_test_animation(frames=60, num_joints=22):
    """
    Generate a synthetic walking animation with quality issues.

    Args:
        frames: Number of frames (default 60 = 2 seconds at 30fps)
        num_joints: Number of joints (default 22 for SMPL)

    Returns:
        Joint positions (frames, num_joints, 3) with intentional artifacts
    """
    joints = np.zeros((frames, num_joints, 3))

    # Root trajectory (joint 0) - forward walk
    for t in range(frames):
        progress = t / frames
        joints[t, 0, 0] = progress * 2.0  # Move forward 2 units
        joints[t, 0, 1] = 1.0  # Hip height
        joints[t, 0, 2] = 0.0

    # Add random jitter to root (simulating poor generation)
    joints[:, 0, :] += np.random.normal(0, 0.02, (frames, 3))

    # Spine/torso (joints 3, 6, 9, 12, 15)
    spine_joints = [3, 6, 9, 12, 15]
    for i, jidx in enumerate(spine_joints):
        height_offset = 0.2 * (i + 1)
        for t in range(frames):
            joints[t, jidx] = joints[t, 0] + np.array([0, height_offset, 0])

    # Head (joint 12)
    joints[:, 12, :] = joints[:, 0, :] + np.array([0, 1.5, 0])

    # Arms (simplified)
    # Left arm: joints 2, 5, 8, 11
    # Right arm: joints 1, 4, 7, 10
    for t in range(frames):
        swing = np.sin(2 * np.pi * t / 30) * 0.3

        # Left arm
        joints[t, 2] = joints[t, 9] + np.array([-0.3, 0, swing])
        joints[t, 5] = joints[t, 2] + np.array([-0.3, -0.2, 0])
        joints[t, 8] = joints[t, 5] + np.array([-0.2, -0.2, 0])
        joints[t, 11] = joints[t, 8] + np.array([0, -0.1, 0])

        # Right arm
        joints[t, 1] = joints[t, 9] + np.array([0.3, 0, -swing])
        joints[t, 4] = joints[t, 1] + np.array([0.3, -0.2, 0])
        joints[t, 7] = joints[t, 4] + np.array([0.2, -0.2, 0])
        joints[t, 10] = joints[t, 7] + np.array([0, -0.1, 0])

    # Legs with foot sliding artifact
    # Left leg: joints 13, 16, 18, 20 (hip, knee, ankle, toe)
    # Right leg: joints 14, 17, 19, 21

    for t in range(frames):
        phase = (t / frames) * 2 * np.pi * 2  # 2 steps

        # Left leg
        left_swing = np.sin(phase)
        left_lift = max(0, np.sin(phase)) * 0.3

        # Hip
        joints[t, 13] = joints[t, 0] + np.array([-0.15, -0.1, 0])
        # Knee
        joints[t, 16] = joints[t, 13] + np.array([0, -0.4, left_swing * 0.2])
        # Ankle - ADD SLIDING ARTIFACT
        ankle_z = left_swing * 0.4
        # Intentionally slide foot when it should be planted
        if left_lift < 0.05 and abs(left_swing) > 0.2:  # Should be planted but moving
            ankle_z += np.random.normal(0, 0.05)  # Add sliding
        joints[t, 18] = joints[t, 16] + np.array([0, -0.4 + left_lift, ankle_z])
        # Toe
        joints[t, 20] = joints[t, 18] + np.array([0, -0.05, 0.1])

        # Right leg (opposite phase)
        right_swing = np.sin(phase + np.pi)
        right_lift = max(0, np.sin(phase + np.pi)) * 0.3

        # Hip
        joints[t, 14] = joints[t, 0] + np.array([0.15, -0.1, 0])
        # Knee
        joints[t, 17] = joints[t, 14] + np.array([0, -0.4, right_swing * 0.2])
        # Ankle - ADD SLIDING ARTIFACT
        ankle_z = right_swing * 0.4
        if right_lift < 0.05 and abs(right_swing) > 0.2:
            ankle_z += np.random.normal(0, 0.05)  # Add sliding
        joints[t, 19] = joints[t, 17] + np.array([0, -0.4 + right_lift, ankle_z])
        # Toe
        joints[t, 21] = joints[t, 19] + np.array([0, -0.05, 0.1])

    # Add temporal jitter to all joints (simulating poor generation)
    jitter = np.random.normal(0, 0.01, joints.shape)
    joints += jitter

    # Add some unrealistic velocity spikes (every 10 frames)
    for t in range(10, frames, 10):
        spike = np.random.normal(0, 0.1, (num_joints, 3))
        joints[t] += spike

    # Ensure no ground penetration initially (we'll test if it gets introduced)
    min_height = np.min(joints[:, :, 1])
    if min_height < 0:
        joints[:, :, 1] -= min_height

    return joints


if __name__ == "__main__":
    print("Generating synthetic test animation...")

    # Generate animation with artifacts
    animation = generate_test_animation(frames=60, num_joints=22)

    # Save
    output_path = "test_animation_with_artifacts.npy"
    np.save(output_path, animation)

    print(f"✓ Saved to: {output_path}")
    print(f"  Shape: {animation.shape}")
    print(f"  Frames: {animation.shape[0]}")
    print(f"  Joints: {animation.shape[1]}")
    print(f"\nThis animation contains intentional quality issues:")
    print("  - Foot sliding artifacts")
    print("  - Temporal jitter")
    print("  - Random velocity spikes")
    print("\nUse improve_animation.py to fix these issues!")
