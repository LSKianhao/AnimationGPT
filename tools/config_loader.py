"""
Configuration loader for AnimationGPT post-processing.

Handles loading and merging of YAML configuration files with preset support.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class PostProcessConfig:
    """
    Configuration manager for post-processing parameters.

    Usage:
        config = PostProcessConfig()
        config.load('config/postprocess_config.yaml')

        # Access values
        sigma = config.get('temporal_smoothing.gaussian.sigma')
        enabled = config.get('temporal_smoothing.enabled')

        # Or use nested dict access
        foot_cfg = config['foot_contact']
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Optional path to YAML config file. If not provided,
                        uses default config/postprocess_config.yaml
        """
        self.config: Dict[str, Any] = {}
        self.config_path: Optional[str] = None

        if config_path is not None:
            self.load(config_path)
        else:
            # Try to load default config
            default_path = Path(__file__).parent.parent / 'config' / 'postprocess_config.yaml'
            if default_path.exists():
                self.load(str(default_path))

    def load(self, config_path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Apply preset if specified
        preset = self.config.get('preset', 'custom')
        if preset != 'custom':
            self._apply_preset(preset)

    def _apply_preset(self, preset_name: str) -> None:
        """
        Apply a preset configuration.

        Args:
            preset_name: Name of preset to apply ('fast', 'balanced', 'quality')
        """
        if 'presets' not in self.config or preset_name not in self.config['presets']:
            print(f"Warning: Preset '{preset_name}' not found. Using default values.")
            return

        preset = self.config['presets'][preset_name]

        for key, value in preset.items():
            self.set(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key (supports dot notation, e.g. 'foot_contact.enabled')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config = PostProcessConfig()
            >>> sigma = config.get('temporal_smoothing.gaussian.sigma', 1.5)
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set

        Example:
            >>> config = PostProcessConfig()
            >>> config.set('temporal_smoothing.gaussian.sigma', 2.0)
        """
        keys = key.split('.')
        target = self.config

        # Navigate to the parent dict
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        # Set the final value
        target[keys[-1]] = value

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access: config['key']"""
        if '.' in key:
            return self.get(key)
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-style setting: config['key'] = value"""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists: 'key' in config"""
        return self.get(key) is not None

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()

    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to YAML file.

        Args:
            output_path: Path to save config. If None, overwrites original file.
        """
        if output_path is None:
            output_path = self.config_path

        if output_path is None:
            raise ValueError("No output path specified and no config was loaded")

        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def get_foot_contact_params(self) -> Dict[str, Any]:
        """
        Get foot contact detection parameters as a dict.

        Returns:
            Dict with keys: adaptive_thresholds, velocity_percentile,
                          fixed_velocity_thresholds, height_thresholds,
                          interpolation_length, force_on_floor, fid_l, fid_r
        """
        fc = self.config.get('foot_contact', {})
        return {
            'adaptive_thresholds': fc.get('adaptive_thresholds', True),
            'velocity_percentile': fc.get('velocity_percentile', 15),
            'fixed_velocity_thresholds': fc.get('fixed_velocity_thresholds', [0.05, 0.2]),
            'height_thresholds': fc.get('height_thresholds', [0.06, 0.03]),
            'interpolation_length': fc.get('interpolation_length', 5),
            'force_on_floor': fc.get('force_on_floor', True),
            'fid_l': tuple(fc.get('foot_indices', {}).get('left', [3, 4])),
            'fid_r': tuple(fc.get('foot_indices', {}).get('right', [7, 8])),
        }

    def get_temporal_smoothing_params(self) -> Dict[str, Any]:
        """
        Get temporal smoothing parameters as a dict.

        Returns:
            Dict with keys: enabled, method, gaussian_sigma, savgol_window,
                          savgol_polyorder, butterworth_cutoff, butterworth_fps,
                          preserve_root
        """
        ts = self.config.get('temporal_smoothing', {})
        method = ts.get('method', 'gaussian')

        params = {
            'enabled': ts.get('enabled', True),
            'method': method,
            'preserve_root': ts.get('preserve_root', True),
        }

        # Add method-specific params
        if method == 'gaussian':
            params['gaussian_sigma'] = ts.get('gaussian', {}).get('sigma', 1.5)
        elif method == 'savgol':
            params['savgol_window'] = ts.get('savgol', {}).get('window_length', 7)
            params['savgol_polyorder'] = ts.get('savgol', {}).get('polyorder', 2)
        elif method == 'butterworth':
            params['butterworth_cutoff'] = ts.get('butterworth', {}).get('cutoff_freq', 0.3)
            params['butterworth_fps'] = ts.get('butterworth', {}).get('fps', 30)

        return params

    def get_velocity_clipping_params(self) -> Dict[str, Any]:
        """
        Get velocity clipping parameters as a dict.

        Returns:
            Dict with keys: enabled, max_velocity, preserve_root
        """
        vc = self.config.get('velocity_clipping', {})
        return {
            'enabled': vc.get('enabled', True),
            'max_velocity': vc.get('max_velocity', 0.5),
            'preserve_root': vc.get('preserve_root', True),
        }

    def get_ik_params(self) -> Dict[str, Any]:
        """
        Get inverse kinematics parameters as a dict.

        Returns:
            Dict with keys: enabled, solver, iterations, damping,
                          recalculate, silent
        """
        ik = self.config.get('inverse_kinematics', {})
        return {
            'enabled': ik.get('enabled', True),
            'solver': ik.get('solver', 'basic'),
            'iterations': ik.get('iterations', 10),
            'damping': ik.get('damping', 5.0),
            'recalculate': ik.get('recalculate', False),
            'silent': ik.get('silent', True),
        }

    def get_pipeline_order(self) -> list:
        """
        Get processing pipeline stage order.

        Returns:
            List of stage names in execution order
        """
        return self.config.get('pipeline', {}).get('order', [
            'foot_contact_fixing',
            'temporal_smoothing',
            'velocity_clipping',
            'inverse_kinematics'
        ])

    def should_skip_stage(self, stage_name: str) -> bool:
        """
        Check if a pipeline stage should be skipped.

        Args:
            stage_name: Name of the stage

        Returns:
            True if stage should be skipped
        """
        skip_list = self.config.get('pipeline', {}).get('skip_stages', [])
        return stage_name in skip_list

    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return self.config.get('debug', {}).get('verbose', False)

    def should_compute_metrics(self) -> bool:
        """Check if quality metrics should be computed."""
        return self.config.get('debug', {}).get('compute_metrics', True)


def load_config(config_path: Optional[str] = None) -> PostProcessConfig:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to config file (optional)

    Returns:
        PostProcessConfig instance

    Example:
        >>> from config_loader import load_config
        >>> config = load_config('config/postprocess_config.yaml')
        >>> sigma = config.get('temporal_smoothing.gaussian.sigma')
    """
    return PostProcessConfig(config_path)


if __name__ == "__main__":
    # Example usage and testing
    print("Config Loader Module")
    print("=" * 50)

    # Load config
    config = PostProcessConfig()
    print(f"\n✓ Loaded config from: {config.config_path}")

    # Show current preset
    preset = config.get('preset', 'custom')
    print(f"✓ Active preset: {preset}")

    # Show key parameters
    print("\nKey Parameters:")
    print(f"  - Adaptive thresholds: {config.get('foot_contact.adaptive_thresholds')}")
    print(f"  - Gaussian sigma: {config.get('temporal_smoothing.gaussian.sigma')}")
    print(f"  - Max velocity: {config.get('velocity_clipping.max_velocity')}")
    print(f"  - IK iterations: {config.get('inverse_kinematics.iterations')}")

    # Test parameter getters
    print("\nFoot Contact Params:")
    fc_params = config.get_foot_contact_params()
    for k, v in fc_params.items():
        print(f"  - {k}: {v}")

    print("\nTemporal Smoothing Params:")
    ts_params = config.get_temporal_smoothing_params()
    for k, v in ts_params.items():
        print(f"  - {k}: {v}")

    print("\n✓ Config loader test complete!")
