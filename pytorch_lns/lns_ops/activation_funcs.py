import torch
import math
from ..lns_tensor.lns_tensor import LNSTensor

def lns_relu(x):
    """
    ReLU activation function for LNSTensor.
    
    Args:
        x: Input LNSTensor
    
    Returns:
        LNSTensor with ReLU applied
    """
    if not isinstance(x, LNSTensor):
        x = LNSTensor(x)
        
    result = LNSTensor(0)
    result.log_data = x.log_data.clone()
    result.base = x.base
    
    # ReLU sets negative values to zero
    if hasattr(x, 'sign'):
        mask_negative = (x.sign < 0)
        result.log_data[mask_negative] = x.ZERO_PLACEHOLDER
        result.sign = torch.ones_like(x.sign)
    
    return result

def lns_sigmoid(x):
    """
    Sigmoid activation function for LNSTensor.
    
    Args:
        x: Input LNSTensor
    
    Returns:
        LNSTensor with sigmoid applied
    """
    if not isinstance(x, LNSTensor):
        x = LNSTensor(x)
        
    # Convert to float for calculation
    x_float = x.to_float()
    
    # Apply sigmoid
    sigmoid_x = torch.sigmoid(x_float)
    
    # Convert back to LNS
    result = LNSTensor(sigmoid_x, base=x.base)
    return result

def lns_tanh(x):
    """
    Tanh activation function for LNSTensor.
    
    Args:
        x: Input LNSTensor
    
    Returns:
        LNSTensor with tanh applied
    """
    if not isinstance(x, LNSTensor):
        x = LNSTensor(x)
        
    # Convert to float for calculation
    x_float = x.to_float()
    
    # Apply tanh
    tanh_x = torch.tanh(x_float)
    
    # Convert back to LNS
    result = LNSTensor(tanh_x, base=x.base)
    return result

def lns_softmax(x, dim=-1):
    """
    Softmax activation function for LNSTensor.
    
    Args:
        x: Input LNSTensor
        dim: Dimension along which to apply softmax
    
    Returns:
        LNSTensor with softmax applied
    """
    if not isinstance(x, LNSTensor):
        x = LNSTensor(x)
        
    # Softmax: exp(x_i) / sum(exp(x_j))
    # In log domain, this is x_i - log(sum(exp(x_j)))
    
    # First, we need to handle numerical stability by subtracting the max value
    max_val, _ = torch.max(x.log_data, dim=dim, keepdim=True)
    shifted_log = x.log_data - max_val
    
    # Convert to linear domain for sum
    log_e_base = math.log(x.base)  # log_base(e)
    exp_shifted = torch.exp(shifted_log * log_e_base)
    
    # Sum in linear domain
    sum_exp = torch.sum(exp_shifted, dim=dim, keepdim=True)
    
    # Convert back to log domain and adjust
    log_sum_exp = torch.log(sum_exp) / log_e_base
    softmax_log = shifted_log - log_sum_exp
    
    result = LNSTensor(0)
    result.log_data = softmax_log
    result.base = x.base
    result.sign = torch.ones_like(softmax_log)  # Softmax always produces positive values
    
    return result