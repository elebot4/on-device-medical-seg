import sys
import yaml
from dataclasses import dataclass, fields, asdict
from typing import Any, List, Tuple, Optional, Dict, Union

@dataclass
class Config:
    # --- Model Settings (explicit control) ---
    in_channels: int = 1
    out_channels: int = 1
    num_stages: int = 4
    base_chs: int = 32  
    dropout: float = 0.1
    norm_groups: int = 8
    deep_supervision: bool = True
    input_shape: Tuple[int, ...] = (64,64,64)
    
    # --- Training Settings --- 
    nb_epochs: int = 100
    lr: float = 3e-4
    batch_size: int = 2
    device: str = "cuda"
    
    # Optimizer settings
    optimizer: str = "ADAMw"  # ADAMw, ADAM, SGD
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9  # For SGD
    
    # Scheduler settings
    scheduler: str = "PolyLR"  # PolyLR, OneCycleLR, MultiStepLR
    gamma: float = 0.9  # For PolyLR decay
    milestones: Optional[List[float]] = None  # For MultiStepLR (as fractions)

    act_type: str = "relu" # relu, gelu, leaky
    norm_type: str = "group" # group, batch, instance, none
    
    def __post_init__(self) -> None:
        """Process config after initialization - clean karpathy style"""
        # Handle None default for milestones (dataclass limitation)
        if self.milestones is None:
            self.milestones = [0.5, 0.75]  # Default milestone fractions

# -----------------------------------------------------------------------------

def get_config() -> Config:
    # 1. Start with hardcoded defaults
    config = Config()
    valid_keys = {f.name for f in fields(Config)}
    
    # 2. Extract YAML path and CLI overrides from sys.argv
    # We do this manually to avoid the boilerplate of argparse
    overrides = {}
    yaml_path = None
    
    for arg in sys.argv[1:]:
        if arg.startswith('--config='):
            yaml_path = arg.split('=')[1]
        elif '=' in arg:
            # Handle key=value overrides
            k, v = arg.split('=', 1)
            k = k.lstrip('-') # allow --key=val or key=val
            overrides[k] = v
        else:
            # This handles the case where someone just passes a path
            if arg.endswith('.yaml') or arg.endswith('.yml'):
                yaml_path = arg

    # 3. Load from YAML first (if exists)
    if yaml_path:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            if data:
                for k, v in data.items():
                    if k not in valid_keys:
                        raise AttributeError(f"Error: '{k}' is not a valid Config parameter (Found in YAML).")
                    _set_config_attr(config, k, v)

    # 4. Layer CLI overrides on top
    for k, v in overrides.items():
        if k not in valid_keys:
            raise AttributeError(f"Error: '{k}' is not a valid Config parameter (Found in CLI).")
        _set_config_attr(config, k, v)

    return config

def _set_config_attr(config: Config, key: str, value: Any) -> None:
    """ Helper to cast values to the correct type defined in the dataclass. """
    # Find the target type from the dataclass fields
    target_field = next(f for f in fields(Config) if f.name == key)
    target_type = target_field.type

    # Handle Boolean type casting from CLI strings
    if target_type == bool and isinstance(value, str):
        casted_value = value.lower() in ("true", "1", "yes")
    # Handle type casting for non booleans types
    else:
        try:
            # Cast to int, float, etc.
            casted_value = target_type(value)
        except (TypeError, ValueError):
            casted_value = value
            
    setattr(config, key, casted_value)

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Usage Examples:
    # python train.py --config=my_settings.yaml
    # python train.py --config=my_settings.yaml lr=0.01 num_stages=3
    # python train.py base_chs=64 nb_epochs=50
    
    try:
        config = get_config()
        print("Successfully loaded config:")
        print(config)
    except AttributeError as e:
        print(e)
        sys.exit(1)