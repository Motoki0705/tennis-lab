#!/usr/bin/env python3
"""Test that BaseTrainingRunner can access all common run config keys.

This test verifies that the BaseTrainingRunner correctly reads and uses
all the standardized run configuration keys without errors.
"""

from pathlib import Path
from types import SimpleNamespace
import yaml


def create_mock_config(run_config_dict):
    """Create a mock config object from a run config dictionary."""
    # Create a nested structure that mimics OmegaConf structure
    config = SimpleNamespace()
    config.run = SimpleNamespace(**run_config_dict)
    config.training = SimpleNamespace(
        max_epochs=1,
        gradient_clip_val=None,
    )
    return config


def test_runner_with_config(config_path: Path):
    """Test that a config can be loaded and accessed by runner methods."""
    from src.base.training.runner import BaseTrainingRunner
    
    # Load the config file
    with open(config_path, 'r') as f:
        run_config_dict = yaml.safe_load(f)
    
    # Create mock config
    config = create_mock_config(run_config_dict)
    
    # Create runner instance
    runner = BaseTrainingRunner()
    
    # Test all the methods that access run config
    try:
        # Test output_dir
        output_dir = runner.prepare_output_dir(config)
        assert output_dir is not None, "output_dir should not be None"
        
        # Test seed
        runner.seed_everything(config)  # Should not raise
        
        # Test resume
        resume = runner.resolve_resume(config, output_dir)
        assert resume is None or isinstance(resume, str), "resume should be None or string"
        
        # Test dry_run
        is_dry = runner.is_dry_run(config)
        assert isinstance(is_dry, bool), "dry_run should be boolean"
        
        # Test fast_dev_run
        skip_test = runner.skip_test(config)
        assert isinstance(skip_test, bool), "fast_dev_run should be boolean"
        
        # Test gpus
        accelerator, devices = runner.select_devices(config)
        assert isinstance(accelerator, str), "accelerator should be string"
        assert isinstance(devices, int), "devices should be int"
        
        return True, None
    except Exception as e:
        return False, str(e)


def main():
    """Test all training run configs."""
    repo_root = Path(__file__).parent.parent
    
    # Add src to path so we can import BaseTrainingRunner
    import sys
    sys.path.insert(0, str(repo_root))
    
    training_run_configs = [
        "src/blcs/configs/run/train.yaml",
        "src/plcs/configs/run/train.yaml",
        "src/wasb/configs/run/ball_detection.yaml",
        "src/court_detection/configs/run/default.yaml",
        "src/evnet_detection/configs/run/train.yaml",
        "src/trajectory_completion/configs/run/train.yaml",
    ]
    
    print("Testing BaseTrainingRunner with run configs...\n")
    
    all_passed = True
    for config_rel_path in training_run_configs:
        config_path = repo_root / config_rel_path
        success, error = test_runner_with_config(config_path)
        
        status = "✓" if success else "✗"
        print(f"{status} {config_rel_path}")
        if error:
            print(f"  Error: {error}")
            all_passed = False
    
    print()
    if all_passed:
        print("✓ All configs work with BaseTrainingRunner!")
        return 0
    else:
        print("✗ Some configs failed")
        return 1


if __name__ == "__main__":
    exit(main())
