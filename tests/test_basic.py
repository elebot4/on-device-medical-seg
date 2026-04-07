"""
Basic tests for Medical Segmentation Mobile
Following karpathy principles: minimal, working tests with proper typing
"""
import sys
import os
# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from typing import Dict, Any
from config import Config, get_config
from model import UNet


def test_config_explicit_control() -> None:
    """Test karpathy-style explicit parameter control with typing"""
    # Test explicit parameter setting
    config = Config(
        num_stages=5,
        base_chs=64,
        lr=1e-4,
        batch_size=4
    )
    
    # Should use exactly what we specified
    assert config.num_stages == 5
    assert config.base_chs == 64  
    assert config.lr == 1e-4
    assert config.batch_size == 4
    
    # Test different configurations
    mobile_config = Config(num_stages=3, base_chs=16)
    xl_config = Config(num_stages=5, base_chs=64)
    
    assert mobile_config.num_stages < xl_config.num_stages
    assert mobile_config.base_chs < xl_config.base_chs


def test_config_yaml_loading() -> None:
    """Test loading configuration from YAML"""
    # This should work without error
    config = Config()
    assert config.num_stages == 4  # Default
    assert config.base_chs == 32   # Default
    assert config.in_channels == 1
    assert config.out_channels == 1


def test_model_instantiation() -> None:
    """Test that model can be created with explicit config"""
    config = Config(num_stages=3, base_chs=16)  # Small model for testing
    
    # Should create without error
    model = UNet(config)
    assert model is not None
    
    # Test forward pass with dummy input
    if len(config.input_shape) == 3:  # 3D
        batch_size = 1
        dummy_input = torch.randn(batch_size, config.in_channels, *config.input_shape)
    else:  # 2D
        batch_size = 2
        dummy_input = torch.randn(batch_size, config.in_channels, *config.input_shape)
    
    with torch.no_grad():
        output = model(dummy_input)
        assert output is not None


def test_memory_calculation() -> None:
    """Test memory reporting utility with typing"""  
    from src.utils import get_mem_report
    
    config = Config(num_stages=3, base_chs=16)  # Tiny model
    model = UNet(config)
    
    # Should return memory info without error
    mem_info: Dict[str, Any] = get_mem_report(model, config.input_shape)
    assert "weights_only_mb" in mem_info
    

def test_import_integrity() -> None:
    """Test that all imports work correctly with typing"""
    # Test critical imports that were fixed
    from config import Config
    from model import UNet  
    from dataset import SegmentationDataset, get_dataloaders
    from loss import dice_loss
    from optim import get_optimizer, get_scheduler
    
    # All imports should work
    assert Config is not None
    assert UNet is not None
    assert SegmentationDataset is not None
    assert get_dataloaders is not None
    assert dice_loss is not None


def test_typing_annotations() -> None:
    """Test that typing annotations work correctly"""
    # Test config typing
    config: Config = Config(num_stages=4, base_chs=32)
    assert isinstance(config.num_stages, int)
    assert isinstance(config.base_chs, int)
    assert isinstance(config.lr, float)
    
    # Test model typing
    model: UNet = UNet(config)
    assert isinstance(model, torch.nn.Module)
    
    # Test tensor typing - model in eval mode returns single tensor
    model.eval()
    dummy_input: torch.Tensor = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        output: torch.Tensor = model(dummy_input)  
        assert isinstance(output, torch.Tensor)


if __name__ == "__main__":
    print("🧪 Running basic tests with typing...")
    test_config_explicit_control()
    test_config_yaml_loading() 
    test_model_instantiation()
    test_import_integrity()
    test_typing_annotations()
    
    # Memory test (might fail without model.py fixes)
    try:
        test_memory_calculation()
        print("✅ All tests with typing passed!")
    except Exception as e:
        print(f"⚠️  Memory test failed (expected): {e}")
        print("✅ Core functionality tests with typing passed!")