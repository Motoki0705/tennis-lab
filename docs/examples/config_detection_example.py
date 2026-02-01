"""Example: Detecting and adapting to different config structures.

This example demonstrates how third-party code can detect whether a config
uses the standard structure (config.run.*) or the MAE exception structure.
"""

from pathlib import Path
from typing import Any


def get_output_directory(config: Any) -> Path:
    """Get output directory from either standard or MAE-style config.
    
    Args:
        config: OmegaConf DictConfig or similar
    
    Returns:
        Path to output directory
    
    Examples:
        >>> # Standard structure
        >>> config = OmegaConf.create({
        ...     "run": {"output_dir": "outputs/wasb"},
        ...     "model": {...}
        ... })
        >>> get_output_directory(config)
        Path('outputs/wasb')
        
        >>> # MAE structure (Hydra-managed)
        >>> config = OmegaConf.create({
        ...     "seed": 42,
        ...     "trainer": {"devices": "auto"}
        ... })
        >>> get_output_directory(config)
        Path.cwd()  # Hydra working directory
    """
    if hasattr(config, "run") and hasattr(config.run, "output_dir"):
        # Standard structure (WASB, PLCS, BLCS, Court Detection)
        return Path(config.run.output_dir)
    else:
        # MAE-style structure (Hydra-managed)
        return Path.cwd()


def get_seed(config: Any) -> int | None:
    """Get random seed from either standard or MAE-style config.
    
    Args:
        config: OmegaConf DictConfig or similar
    
    Returns:
        Random seed value or None
    """
    if hasattr(config, "run") and hasattr(config.run, "seed"):
        # Standard structure
        return int(config.run.seed) if config.run.seed is not None else None
    elif hasattr(config, "seed"):
        # MAE structure
        return int(config.seed) if config.seed is not None else None
    return None


def get_device_config(config: Any) -> tuple[str, int]:
    """Get accelerator and device count from either config structure.
    
    Args:
        config: OmegaConf DictConfig or similar
    
    Returns:
        Tuple of (accelerator, num_devices)
        - accelerator: "cpu", "gpu", "auto"
        - num_devices: number of devices to use
    """
    if hasattr(config, "run") and hasattr(config.run, "gpus"):
        # Standard structure
        gpus = int(config.run.gpus)
        if gpus > 0:
            return "gpu", gpus
        return "cpu", 1
    elif hasattr(config, "trainer"):
        # MAE structure
        trainer_cfg = config.trainer
        accelerator = str(trainer_cfg.get("accelerator", "auto"))
        devices = trainer_cfg.get("devices", "auto")
        
        if devices == "auto" or devices is None:
            devices = 1
        elif isinstance(devices, str) and devices.isdigit():
            devices = int(devices)
        elif not isinstance(devices, int):
            devices = 1
            
        return accelerator, devices
    
    # Fallback
    return "cpu", 1


def is_dry_run(config: Any) -> bool:
    """Check if running in dry-run mode.
    
    Args:
        config: OmegaConf DictConfig or similar
    
    Returns:
        True if dry-run mode is enabled
    """
    if hasattr(config, "run") and hasattr(config.run, "dry_run"):
        # Standard structure
        return bool(config.run.dry_run)
    elif hasattr(config, "trainer") and hasattr(config.trainer, "fast_dev_run"):
        # MAE structure uses trainer.fast_dev_run
        return bool(config.trainer.fast_dev_run)
    return False


def print_config_info(config: Any) -> None:
    """Print diagnostic information about config structure.
    
    Args:
        config: OmegaConf DictConfig or similar
    """
    has_run = hasattr(config, "run")
    config_type = "Standard (config.run.*)" if has_run else "MAE-style (flattened)"
    
    print(f"Config Type: {config_type}")
    print(f"Output Directory: {get_output_directory(config)}")
    print(f"Seed: {get_seed(config)}")
    
    accelerator, devices = get_device_config(config)
    print(f"Accelerator: {accelerator}")
    print(f"Devices: {devices}")
    print(f"Dry Run: {is_dry_run(config)}")


# Example usage
if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    # Standard config example
    print("=" * 50)
    print("Standard Config (WASB, PLCS, etc.)")
    print("=" * 50)
    standard_config = OmegaConf.create({
        "run": {
            "seed": 42,
            "gpus": 1,
            "output_dir": "outputs/wasb",
            "dry_run": False,
        },
        "model": {"name": "hrcnet"},
        "training": {"max_epochs": 100},
    })
    print_config_info(standard_config)
    
    print("\n" + "=" * 50)
    print("MAE Config (Exception)")
    print("=" * 50)
    mae_config = OmegaConf.create({
        "seed": 42,
        "trainer": {
            "accelerator": "auto",
            "devices": "auto",
            "fast_dev_run": False,
        },
        "model": {"hidden_dim": 768},
        "training": {"max_epochs": 400},
    })
    print_config_info(mae_config)
