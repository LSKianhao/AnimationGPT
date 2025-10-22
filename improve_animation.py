#!/usr/bin/env python3
"""
AnimationGPT Quality Improvement Script

This script applies post-processing improvements to generated animations:
- Adaptive foot contact detection
- Temporal smoothing to reduce jitter
- Velocity clipping for realistic motion
- Physical plausibility metrics

Usage:
    python improve_animation.py input.npy output.npy
    python improve_animation.py input.npy output.npy --config config/postprocess_config.yaml
    python improve_animation.py input.npy output.npy --preset quality
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent / 'tools'))

from temporal_filters import apply_motion_smoothing_pipeline, analyze_motion_smoothness
from config_loader import load_config
from physical_metrics import compute_all_metrics, print_metrics_report, compare_metrics

# Import foot contact fixing
sys.path.insert(0, str(Path(__file__).parent / 'tools' / 'npy2bvh' / 'visualization'))
from remove_fs import remove_fs


def load_animation(input_path: str) -> np.ndarray:
    """Load animation from .npy file."""
    data = np.load(input_path, allow_pickle=True)

    # Reshape to (T, 22, 3) if needed
    if data.shape[-1] == 3 and data.shape[-2] == 22:
        return data
    elif len(data.shape) == 2:
        # Assume it's (T, 66) and reshape
        return data.reshape(-1, 22, 3)
    else:
        # Try to reshape
        return data.reshape(-1, 22, 3)


def improve_animation(joints: np.ndarray, config=None) -> np.ndarray:
    """
    Apply post-processing improvements to animation.

    Args:
        joints: Joint positions (T, J, 3)
        config: PostProcessConfig instance (optional)

    Returns:
        Improved joint positions
    """
    if config is None:
        # Use default config
        config = load_config()

    result = joints.copy()
    pipeline_order = config.get_pipeline_order()

    print("\n🔧 Applying post-processing pipeline:")
    print(f"   Pipeline: {' → '.join(pipeline_order)}")

    for stage in pipeline_order:
        if config.should_skip_stage(stage):
            print(f"   ⏭️  Skipping: {stage}")
            continue

        if stage == 'foot_contact_fixing':
            print("   ✓ Fixing foot contacts...", end='', flush=True)
            fc_params = config.get_foot_contact_params()
            result = remove_fs(
                result,
                foot_contact=None,
                fid_l=fc_params['fid_l'],
                fid_r=fc_params['fid_r'],
                interp_length=fc_params['interpolation_length'],
                force_on_floor=fc_params['force_on_floor'],
                adaptive_thresholds=fc_params['adaptive_thresholds'],
                velocity_percentile=fc_params['velocity_percentile']
            )
            print(" Done!")

        elif stage == 'temporal_smoothing':
            ts_params = config.get_temporal_smoothing_params()
            if ts_params['enabled']:
                print("   ✓ Applying temporal smoothing...", end='', flush=True)
                result = apply_motion_smoothing_pipeline(
                    result,
                    gaussian_sigma=ts_params.get('gaussian_sigma', 1.5),
                    max_velocity=0,  # Will be applied in next stage
                    preserve_root=ts_params['preserve_root']
                )
                print(" Done!")
            else:
                print("   ⏭️  Temporal smoothing disabled")

        elif stage == 'velocity_clipping':
            vc_params = config.get_velocity_clipping_params()
            if vc_params['enabled']:
                print("   ✓ Clipping velocities...", end='', flush=True)
                from temporal_filters import clip_joint_velocities
                result = clip_joint_velocities(
                    result,
                    max_velocity=vc_params['max_velocity'],
                    preserve_root=vc_params['preserve_root']
                )
                print(" Done!")
            else:
                print("   ⏭️  Velocity clipping disabled")

        elif stage == 'inverse_kinematics':
            ik_params = config.get_ik_params()
            if ik_params['enabled']:
                print("   ⏭️  IK optimization (requires BVH pipeline)")
            else:
                print("   ⏭️  IK disabled")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Improve AnimationGPT generated animations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python improve_animation.py input.npy output.npy

  # Use a specific config file
  python improve_animation.py input.npy output.npy --config my_config.yaml

  # Use a quality preset
  python improve_animation.py input.npy output.npy --preset quality

  # Compare before/after metrics
  python improve_animation.py input.npy output.npy --metrics

  # Disable specific stages
  python improve_animation.py input.npy output.npy --no-smoothing

Presets:
  - fast: Minimal processing, fastest
  - balanced: Good quality/speed tradeoff (default)
  - quality: Maximum quality, slower processing
        """
    )

    parser.add_argument('input', type=str, help='Input .npy file')
    parser.add_argument('output', type=str, help='Output .npy file')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--preset', type=str, choices=['fast', 'balanced', 'quality'],
                       help='Quality preset (overrides config)')
    parser.add_argument('--metrics', action='store_true',
                       help='Compute and display quality metrics')
    parser.add_argument('--no-smoothing', action='store_true',
                       help='Disable temporal smoothing')
    parser.add_argument('--no-foot-fix', action='store_true',
                       help='Disable foot contact fixing')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')

    args = parser.parse_args()

    # Header
    print("\n" + "=" * 70)
    print(" AnimationGPT Quality Improvement Tool")
    print("=" * 70)

    # Load config
    print(f"\n📁 Loading configuration...")
    config = load_config(args.config)

    # Apply preset if specified
    if args.preset:
        print(f"   Using preset: {args.preset}")
        config.set('preset', args.preset)
        config._apply_preset(args.preset)
    else:
        preset = config.get('preset', 'custom')
        print(f"   Active preset: {preset}")

    # Override config with command-line args
    if args.no_smoothing:
        config.set('temporal_smoothing.enabled', False)
    if args.no_foot_fix:
        config.set('pipeline.skip_stages', ['foot_contact_fixing'])

    # Load input
    print(f"\n📥 Loading input: {args.input}")
    try:
        joints = load_animation(args.input)
        print(f"   Shape: {joints.shape}")
        print(f"   Frames: {joints.shape[0]}")
        print(f"   Joints: {joints.shape[1]}")
    except Exception as e:
        print(f"❌ Error loading input: {e}")
        return 1

    # Compute before metrics
    if args.metrics or config.should_compute_metrics():
        print(f"\n📊 Computing metrics (before)...")
        before_metrics = compute_all_metrics(joints, fps=30)
        if args.verbose or args.metrics:
            print_metrics_report(before_metrics)

    # Apply improvements
    print(f"\n🚀 Processing animation...")
    improved = improve_animation(joints, config)

    # Compute after metrics
    if args.metrics or config.should_compute_metrics():
        print(f"\n📊 Computing metrics (after)...")
        after_metrics = compute_all_metrics(improved, fps=30)

        if args.metrics:
            print_metrics_report(after_metrics)
            compare_metrics(before_metrics, after_metrics)
        else:
            # Just print key improvements
            fs_before = before_metrics['foot_sliding']['foot_sliding_per_frame']
            fs_after = after_metrics['foot_sliding']['foot_sliding_per_frame']
            jerk_before = before_metrics['acceleration']['jerk_score']
            jerk_after = after_metrics['acceleration']['jerk_score']

            print(f"\n✨ Quality Improvements:")
            print(f"   Foot sliding:  {fs_before:.4f} → {fs_after:.4f} ({(fs_after-fs_before)/fs_before*100:+.1f}%)")
            print(f"   Jerk score:    {jerk_before:.3f} → {jerk_after:.3f} ({(jerk_after-jerk_before)/jerk_before*100:+.1f}%)")

    # Save output
    print(f"\n💾 Saving output: {args.output}")
    np.save(args.output, improved)
    print(f"   ✓ Saved successfully!")

    print("\n" + "=" * 70)
    print(" ✅ Processing complete!")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
