import torch
import math
import sys
import os
from ..lns_tensor.lns_tensor import LNSTensor

def lns_add(a, b):
    """
    Add two LNSTensor objects.
    This function can be more sophisticated than the __add__ method
    to handle various edge cases.
    """
    return a + b

def lns_subtract(a, b):
    """
    Subtract LNSTensor b from LNSTensor a.
    """
    return a - b

def lns_multiply(a, b):
    """
    Multiply two LNSTensor objects.
    """
    return a * b

def lns_divide(a, b):
    """
    Divide LNSTensor a by LNSTensor b.
    """
    return a / b

def lns_exp(x):
    """
    Compute exponential function in LNS domain.
    exp(x) in LNS is simply adding x to log(1).
    """
    if not isinstance(x, LNSTensor):
        x = LNSTensor(x)
    
    # log(e) in the given base
    log_e = 1.0 / math.log(x.base)
    
    # exp(x) = base^(x * log_e)
    result = x.log_data * log_e
    
    output = LNSTensor(0)
    output.log_data = result
    output.base = x.base
    
    return output

def lns_log(x):
    """
    Compute natural logarithm in LNS domain.
    log(x) in LNS is simply dividing the log value by log(e).
    """
    if not isinstance(x, LNSTensor):
        x = LNSTensor(x)
    
    # log(e) in the given base
    log_e = 1.0 / math.log(x.base)
    
    # log(x) = log_base(x) / log_e
    result = x.log_data / log_e
    
    # Handle logarithm of negative numbers or zero
    mask_zero_or_neg = (x.log_data == x.ZERO_PLACEHOLDER)
    if hasattr(x, 'sign'):
        mask_zero_or_neg = mask_zero_or_neg | (x.sign < 0)
        
    result[mask_zero_or_neg] = float('nan')
    
    # Return as a regular torch tensor, not an LNSTensor
    return result