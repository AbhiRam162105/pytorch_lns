import torch
from torch.autograd import Function
import math
from ..lns_tensor.lns_tensor import LNSTensor

class LNSAdd(Function):
    """
    Custom autograd function for LNS addition.
    """
    @staticmethod
    def forward(ctx, a, b):
        """
        Forward pass for LNS addition.
        
        Args:
            ctx: Context object to save information for backward pass
            a, b: Input LNSTensor objects
        
        Returns:
            LNSTensor result of a + b
        """
        # Convert inputs to LNSTensor if they aren't already
        if not isinstance(a, LNSTensor):
            a = LNSTensor(a)
        if not isinstance(b, LNSTensor):
            b = LNSTensor(b, base=a.base)
            
        # Save inputs for backward pass
        ctx.save_for_backward(a.log_data, b.log_data)
        if hasattr(a, 'sign'):
            ctx.a_sign = a.sign
        if hasattr(b, 'sign'):
            ctx.b_sign = b.sign
        ctx.base = a.base
        
        # Compute forward pass
        result = a + b
        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for LNS addition.
        
        Args:
            ctx: Context object with saved information
            grad_output: Gradient with respect to output
            
        Returns:
            Gradients with respect to inputs a and b
        """
        a_log, b_log = ctx.saved_tensors
        base = ctx.base
        
        # Convert gradient to LNSTensor if needed
        if not isinstance(grad_output, LNSTensor):
            grad_output = LNSTensor(grad_output, base=base)
        
        # For log(a + b), the derivatives are:
        # ∂(a + b)/∂a = 1
        # ∂(a + b)/∂b = 1
        # But in LNS, we need to compute how much each input contributes to the sum
        
        # Create masks for special cases
        mask_a_zero = (a_log == LNSTensor.ZERO_PLACEHOLDER)
        mask_b_zero = (b_log == LNSTensor.ZERO_PLACEHOLDER)
        
        # Compute the relative contribution of each input to the output
        # When a is much larger than b, a contributes almost 100%
        # When b is much larger than a, b contributes almost 100%
        diff = a_log - b_log
        
        # Compute weights using softmax-like formula
        exp_diff = torch.exp(diff * math.log(base))
        a_weight = exp_diff / (1 + exp_diff)
        b_weight = 1 / (1 + exp_diff)
        
        # Adjust for zeros
        a_weight[mask_a_zero] = 0
        b_weight[mask_a_zero] = 1
        a_weight[mask_b_zero] = 1
        b_weight[mask_b_zero] = 0
        
        # Apply the weights to the gradient
        grad_a = grad_output.to_float() * a_weight
        grad_b = grad_output.to_float() * b_weight
        
        return LNSTensor(grad_a, base=base), LNSTensor(grad_b, base=base)


class LNSMultiply(Function):
    """
    Custom autograd function for LNS multiplication.
    """
    @staticmethod
    def forward(ctx, a, b):
        """
        Forward pass for LNS multiplication.
        
        Args:
            ctx: Context object to save information for backward pass
            a, b: Input LNSTensor objects
        
        Returns:
            LNSTensor result of a * b
        """
        # Convert inputs to LNSTensor if they aren't already
        if not isinstance(a, LNSTensor):
            a = LNSTensor(a)
        if not isinstance(b, LNSTensor):
            b = LNSTensor(b, base=a.base)
            
        # Save inputs for backward pass
        ctx.save_for_backward(a.log_data, b.log_data)
        if hasattr(a, 'sign'):
            ctx.a_sign = a.sign
        if hasattr(b, 'sign'):
            ctx.b_sign = b.sign
        ctx.base = a.base
        
        # Compute forward pass (multiplication is addition in log domain)
        result = a * b
        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for LNS multiplication.
        
        Args:
            ctx: Context object with saved information
            grad_output: Gradient with respect to output
            
        Returns:
            Gradients with respect to inputs a and b
        """
        a_log, b_log = ctx.saved_tensors
        base = ctx.base
        
        # Convert gradient to LNSTensor if needed
        if not isinstance(grad_output, LNSTensor):
            grad_output = LNSTensor(grad_output, base=base)
            
        # Reconstruct inputs
        a = LNSTensor(0)
        a.log_data = a_log
        a.base = base
        
        b = LNSTensor(0)
        b.log_data = b_log
        b.base = base
        
        # Restore signs if available
        if hasattr(ctx, 'a_sign'):
            a.sign = ctx.a_sign
        if hasattr(ctx, 'b_sign'):
            b.sign = ctx.b_sign
            
        # For multiplication, the derivatives are:
        # ∂(a * b)/∂a = b
        # ∂(a * b)/∂b = a
        
        # Create masks for special cases
        mask_a_zero = (a_log == LNSTensor.ZERO_PLACEHOLDER)
        mask_b_zero = (b_log == LNSTensor.ZERO_PLACEHOLDER)
        
        # Compute gradients
        grad_a = grad_output * b
        grad_b = grad_output * a
        
        # Handle zero cases properly
        grad_a.log_data[mask_b_zero] = LNSTensor.ZERO_PLACEHOLDER
        grad_b.log_data[mask_a_zero] = LNSTensor.ZERO_PLACEHOLDER
        
        return grad_a, grad_b


class LNSMatMul(Function):
    """
    Custom autograd function for LNS matrix multiplication.
    """
    @staticmethod
    def forward(ctx, a, b):
        """
        Forward pass for LNS matrix multiplication.
        
        Args:
            ctx: Context object to save information for backward pass
            a, b: Input LNSTensor objects
        
        Returns:
            LNSTensor result of a @ b
        """
        # Convert inputs to LNSTensor if they aren't already
        if not isinstance(a, LNSTensor):
            a = LNSTensor(a)
        if not isinstance(b, LNSTensor):
            b = LNSTensor(b, base=a.base)
            
        # Save inputs for backward pass
        ctx.save_for_backward(a.log_data, b.log_data)
        if hasattr(a, 'sign'):
            ctx.a_sign = a.sign
        if hasattr(b, 'sign'):
            ctx.b_sign = b.sign
        ctx.base = a.base
        
        # Convert to float for standard matrix multiplication
        a_float = a.to_float()
        b_float = b.to_float()
        
        # Perform matrix multiplication in regular domain
        result_float = torch.matmul(a_float, b_float)
        
        # Convert back to LNS
        result = LNSTensor(result_float, base=a.base)
        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for LNS matrix multiplication.
        
        Args:
            ctx: Context object with saved information
            grad_output: Gradient with respect to output
            
        Returns:
            Gradients with respect to inputs a and b
        """
        a_log, b_log = ctx.saved_tensors
        base = ctx.base
        
        # Reconstruct inputs
        a = LNSTensor(0)
        a.log_data = a_log
        a.base = base
        
        b = LNSTensor(0)
        b.log_data = b_log
        b.base = base
        
        # Restore signs if available
        if hasattr(ctx, 'a_sign'):
            a.sign = ctx.a_sign
        if hasattr(ctx, 'b_sign'):
            b.sign = ctx.b_sign
            
        # Convert to float domain for gradient computation
        a_float = a.to_float()
        b_float = b.to_float()
        
        # Convert gradient to float if needed
        if isinstance(grad_output, LNSTensor):
            grad_output_float = grad_output.to_float()
        else:
            grad_output_float = grad_output
            
        # Compute gradients using standard rules for matrix multiplication
        grad_a_float = torch.matmul(grad_output_float, b_float.transpose(-1, -2))
        grad_b_float = torch.matmul(a_float.transpose(-1, -2), grad_output_float)
        
        # Convert gradients back to LNS
        grad_a = LNSTensor(grad_a_float, base=base)
        grad_b = LNSTensor(grad_b_float, base=base)
        
        return grad_a, grad_b


class LNSExp(Function):
    """
    Custom autograd function for LNS exponential.
    """
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass for LNS exponential.
        
        Args:
            ctx: Context object to save information for backward pass
            x: Input LNSTensor
        
        Returns:
            LNSTensor result of exp(x)
        """
        # Convert input to LNSTensor if it isn't already
        if not isinstance(x, LNSTensor):
            x = LNSTensor(x)
            
        # Save input for backward pass
        ctx.save_for_backward(x.log_data)
        if hasattr(x, 'sign'):
            ctx.x_sign = x.sign
        ctx.base = x.base
        
        # Compute forward pass
        # log_e in base
        log_e = 1.0 / math.log(x.base)
        
        # Create output LNSTensor
        result = LNSTensor(0)
        result.log_data = x.log_data * log_e
        result.base = x.base
        
        # Exp always produces positive values
        result.sign = torch.ones_like(x.log_data)
        
        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for LNS exponential.
        
        Args:
            ctx: Context object with saved information
            grad_output: Gradient with respect to output
            
        Returns:
            Gradient with respect to input x
        """
        x_log, = ctx.saved_tensors
        base = ctx.base
        
        # Reconstruct input
        x = LNSTensor(0)
        x.log_data = x_log
        x.base = base
        
        # Restore sign if available
        if hasattr(ctx, 'x_sign'):
            x.sign = ctx.x_sign
            
        # The derivative of exp(x) is exp(x) itself
        # So we can reuse the forward pass
        exp_x = LNSExp.forward(ctx, x)
        
        # Multiply by incoming gradient
        if isinstance(grad_output, LNSTensor):
            grad_x = grad_output * exp_x
        else:
            grad_x = LNSTensor(grad_output, base=base) * exp_x
            
        return grad_x


class LNSLog(Function):
    """
    Custom autograd function for LNS natural logarithm.
    """
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass for LNS natural logarithm.
        
        Args:
            ctx: Context object to save information for backward pass
            x: Input LNSTensor
        
        Returns:
            Tensor result of log(x)
        """
        # Convert input to LNSTensor if it isn't already
        if not isinstance(x, LNSTensor):
            x = LNSTensor(x)
            
        # Save input for backward pass
        ctx.save_for_backward(x.log_data)
        if hasattr(x, 'sign'):
            ctx.x_sign = x.sign
        ctx.base = x.base
        
        # log_e in base
        log_e = 1.0 / math.log(x.base)
        
        # log(x) = log_base(x) / log_e
        result = x.log_data / log_e
        
        # Handle logarithm of negative numbers or zero
        mask_zero_or_neg = (x.log_data == LNSTensor.ZERO_PLACEHOLDER)
        if hasattr(x, 'sign'):
            mask_zero_or_neg = mask_zero_or_neg | (x.sign < 0)
            
        result[mask_zero_or_neg] = float('nan')
        
        # Return as regular tensor, not LNSTensor
        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for LNS natural logarithm.
        
        Args:
            ctx: Context object with saved information
            grad_output: Gradient with respect to output
            
        Returns:
            Gradient with respect to input x
        """
        x_log, = ctx.saved_tensors
        base = ctx.base
        
        # Reconstruct input
        x = LNSTensor(0)
        x.log_data = x_log
        x.base = base
        
        # Restore sign if available
        if hasattr(ctx, 'x_sign'):
            x.sign = ctx.x_sign
            
        # The derivative of log(x) is 1/x
        x_float = x.to_float()
        
        # Create mask for valid values (positive non-zero)
        mask_valid = (x_float > 0)
        
        # Initialize gradient with zeros
        grad_x_float = torch.zeros_like(x_float)
        
        # Compute 1/x for valid values and multiply by incoming gradient
        grad_x_float[mask_valid] = grad_output[mask_valid] / x_float[mask_valid]
        
        # Convert back to LNSTensor
        grad_x = LNSTensor(grad_x_float, base=base)
            
        return grad_x


# Helper functions to use the custom autograd functions
def lns_add_autograd(a, b):
    return LNSAdd.apply(a, b)

def lns_multiply_autograd(a, b):
    return LNSMultiply.apply(a, b)

def lns_matmul_autograd(a, b):
    return LNSMatMul.apply(a, b)

def lns_exp_autograd(x):
    return LNSExp.apply(x)

def lns_log_autograd(x):
    return LNSLog.apply(x)