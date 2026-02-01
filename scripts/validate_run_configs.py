#!/usr/bin/env python3
"""Validate that all training run configs have the required common keys.

This script checks that all run configuration files for training tasks
contain the standardized schema defined in docs/run_config_schema.md.
"""

from pathlib import Path
import yaml

# Required keys for all training run configs
REQUIRED_KEYS = {
    "output_dir",
    "seed",
    "gpus",
    "resume",
    "fast_dev_run",
    "dry_run",
}

# Training run config files that must have the standard schema
TRAINING_RUN_CONFIGS = [
    "src/blcs/configs/run/train.yaml",
    "src/blcs/configs/run/train_multiview.yaml",
    "src/plcs/configs/run/train.yaml",
    "src/plcs/configs/run/train_multiview.yaml",
    "src/plcs/configs/run/train_sequence.yaml",
    "src/wasb/configs/run/ball_detection.yaml",
    "src/court_detection/configs/run/default.yaml",
    "src/evnet_detection/configs/run/train.yaml",
    "src/trajectory_completion/configs/run/train.yaml",
]

# Expected standard order of keys
EXPECTED_ORDER = [
    "output_dir",
    "seed",
    "gpus",
    "resume",
    "fast_dev_run",
    "dry_run",
]


def validate_config_file(config_path: Path) -> tuple[bool, list[str]]:
    """Validate a single config file.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if not config_path.exists():
        return False, [f"File does not exist: {config_path}"]
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return False, [f"Failed to parse YAML: {e}"]
    
    if config is None:
        return False, ["Config file is empty"]
    
    # Check for missing keys
    config_keys = set(config.keys())
    missing_keys = REQUIRED_KEYS - config_keys
    if missing_keys:
        issues.append(f"Missing required keys: {sorted(missing_keys)}")
    
    # Check key order (warning only, not an error)
    actual_order = [k for k in config.keys() if k in REQUIRED_KEYS]
    expected_subset = [k for k in EXPECTED_ORDER if k in config.keys()]
    if actual_order != expected_subset:
        issues.append(f"Key order mismatch (expected: {expected_subset}, got: {actual_order})")
    
    return len(issues) == 0, issues


def main() -> int:
    """Validate all training run configs."""
    repo_root = Path(__file__).parent.parent
    
    print("Validating run configuration schema...")
    print(f"Required keys: {sorted(REQUIRED_KEYS)}")
    print(f"Expected order: {EXPECTED_ORDER}\n")
    
    all_valid = True
    
    for config_rel_path in TRAINING_RUN_CONFIGS:
        config_path = repo_root / config_rel_path
        is_valid, issues = validate_config_file(config_path)
        
        status = "✓" if is_valid else "✗"
        print(f"{status} {config_rel_path}")
        
        if not is_valid:
            for issue in issues:
                print(f"  - {issue}")
            all_valid = False
        else:
            # Show the config values for reference
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            for key in EXPECTED_ORDER:
                if key in config:
                    print(f"    {key}: {config[key]}")
    
    print()
    if all_valid:
        print("✓ All training run configs are valid!")
        return 0
    else:
        print("✗ Some training run configs have issues")
        return 1


if __name__ == "__main__":
    exit(main())
