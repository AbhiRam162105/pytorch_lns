import torch
from torch.autograd import Function
from ..lns_tensor.lns_tensor import LNSTensor
from ..lns_autograd.functions import lns_matmul_autograd

def lns_matmul(a, b):
    """
    Matrix multiplication for LNSTensor objects.
    
    Args:
        a, b: Input LNSTensor objects
    
    Returns:
        LNSTensor result of a @ b
    """
    return lns_matmul_autograd(a, b)

def lns_dot(a, b):
    """
    Dot product for LNSTensor objects.
    
    Args:
        a, b: Input LNSTensor objects (1D tensors)
    
    Returns:
        LNSTensor result of dot(a, b)
    """
    # Check dimensions
    if a.log_data.dim() != 1 or b.log_data.dim() != 1:
        raise ValueError("lns_dot expects 1D tensors")
        
    # Dot product is a sum of element-wise multiplications
    result = None
    
    for i in range(len(a.log_data)):
        ai = LNSTensor(0)
        ai.log_data = a.log_data[i:i+1]
        ai.base = a.base
        if hasattr(a, 'sign'):
            ai.sign = a.sign[i:i+1]
            
        bi = LNSTensor(0)
        bi.log_data = b.log_data[i:i+1]
        bi.base = b.base
        if hasattr(b, 'sign'):
            bi.sign = b.sign[i:i+1]
            
        prod = ai * bi
        
        if result is None:
            result = prod
        else:
            result = result + prod
            
    return result

def lns_transpose(x, dim0=0, dim1=1):
    """
    Transpose dimensions of LNSTensor.
    
    Args:
        x: Input LNSTensor
        dim0, dim1: Dimensions to swap
    
    Returns:
        Transposed LNSTensor
    """
    if not isinstance(x, LNSTensor):
        x = LNSTensor(x)
        
    result = LNSTensor(0)
    result.log_data = x.log_data.transpose(dim0, dim1)
    result.base = x.base
    
    if hasattr(x, 'sign'):
        result.sign = x.sign.transpose(dim0, dim1)
        
    return result

def lns_bmm(a, b):
    """
    Batch matrix multiplication for LNSTensor objects.
    
    Args:
        a, b: Input LNSTensor objects (3D tensors)
    
    Returns:
        LNSTensor result of bmm(a, b)
    """
    if not isinstance(a, LNSTensor):
        a = LNSTensor(a)
    if not isinstance(b, LNSTensor):
        b = LNSTensor(b, base=a.base)
        
    # Convert to float for standard batch matrix multiplication
    a_float = a.to_float()
    b_float = b.to_float()
    
    # Perform batch matrix multiplication in regular domain
    result_float = torch.bmm(a_float, b_float)
    
    # Convert back to LNS
    result = LNSTensor(result_float, base=a.base)
    return result