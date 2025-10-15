"""
Model Optimization Utilities for Edge Computing
Implements quantization, pruning, and other optimization techniques
"""

from typing import Optional, Any

# Optional torch imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


def quantize_model(model: Any, backend: str = 'qnnpack') -> Any:
    """
    Quantize a PyTorch model for edge deployment.
    
    Args:
        model: The PyTorch model to quantize
        backend: Quantization backend ('qnnpack' for mobile, 'fbgemm' for server)
    
    Returns:
        Quantized model
    """
    try:
        # Set quantization backend
        torch.backends.quantized.engine = backend
        
        # Configure quantization
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        
        # Prepare the model for quantization
        model_prepared = torch.quantization.prepare(model, inplace=False)
        
        # Convert to quantized model
        model_quantized = torch.quantization.convert(model_prepared, inplace=False)
        
        return model_quantized
    except Exception as e:
        print(f"Quantization failed: {e}")
        return model


def prune_model(model: Any, amount: float = 0.2, method: str = 'l1') -> Any:
    """
    Prune a PyTorch model to reduce size and improve inference speed.
    
    Args:
        model: The PyTorch model to prune
        amount: Fraction of weights to prune (0.0 to 1.0)
        method: Pruning method ('l1', 'random', 'magnitude')
    
    Returns:
        Pruned model
    """
    try:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                
                if method == 'l1':
                    # L1-based pruning: remove smallest weights
                    threshold = torch.quantile(torch.abs(weight), amount)
                    mask = torch.abs(weight) > threshold
                elif method == 'random':
                    # Random pruning
                    mask = torch.rand_like(weight) > amount
                elif method == 'magnitude':
                    # Magnitude-based pruning
                    threshold = torch.quantile(torch.abs(weight), amount)
                    mask = torch.abs(weight) >= threshold
                else:
                    print(f"Unknown pruning method: {method}")
                    continue
                
                module.weight.data = weight * mask.float()
        
        return model
    except Exception as e:
        print(f"Pruning failed: {e}")
        return model


def optimize_for_inference(model: Any) -> Any:
    """
    Optimize model for inference by applying various techniques.
    
    Args:
        model: The PyTorch model to optimize
    
    Returns:
        Optimized model
    """
    # Set model to eval mode
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def calculate_model_size(model: Any) -> dict:
    """
    Calculate model size and parameter count.
    
    Args:
        model: The PyTorch model
    
    Returns:
        Dictionary with size metrics
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "total_params": total_params,
        "size_mb": size_mb,
        "param_size_mb": param_size / 1024 / 1024,
        "buffer_size_mb": buffer_size / 1024 / 1024
    }


def compress_model(model: Any, compression_level: str = 'medium') -> Any:
    """
    Compress model using multiple techniques based on compression level.
    
    Args:
        model: The PyTorch model to compress
        compression_level: 'low', 'medium', or 'high'
    
    Returns:
        Compressed model
    """
    if compression_level == 'low':
        # Light compression
        model = optimize_for_inference(model)
    elif compression_level == 'medium':
        # Medium compression: pruning + optimization
        model = prune_model(model, amount=0.2)
        model = optimize_for_inference(model)
    elif compression_level == 'high':
        # High compression: pruning + quantization + optimization
        model = prune_model(model, amount=0.4)
        model = quantize_model(model)
        model = optimize_for_inference(model)
    
    return model
