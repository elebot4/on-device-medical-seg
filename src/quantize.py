import copy

import torch
import torch.nn as nn


def prepare_ptq(model: nn.Module, backend: str = "qnnpack") -> nn.Module:
    """
    Prepare model for post-training quantization (PTQ).
    
    Args:
        model: PyTorch model to quantize
        backend: Quantization backend ('qnnpack' for mobile, 'x86' for server)
    
    Returns:
        Prepared model ready for calibration
    """
    if backend not in {"qnnpack", "x86"}:
        raise ValueError(f"Unsupported backend: {backend}. Use 'qnnpack' or 'x86'")

    model = copy.deepcopy(model).cpu().eval()
    
    # Set quantization config - no need to set engine in modern PyTorch
    model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    prepared = torch.ao.quantization.prepare(model, inplace=False)
    return prepared


def calibrate_ptq(model: nn.Module, calibration_loader, num_batches: int = 32) -> None:
    """
    Calibrate the prepared model with sample data.
    
    Args:
        model: Prepared model from prepare_ptq()
        calibration_loader: DataLoader with calibration data
        num_batches: Number of batches to use for calibration
    """
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if i >= num_batches:
                break

            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            _ = model(x.cpu())


def finalize_ptq(prepared_model: nn.Module) -> nn.Module:
    """
    Convert the calibrated model to quantized version.
    
    Args:
        prepared_model: Calibrated model from calibrate_ptq()
    
    Returns:
        Quantized model ready for inference
    """
    return torch.ao.quantization.convert(prepared_model, inplace=False)



    