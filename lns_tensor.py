import torch
import numpy as np
import math


class LNSTensor:
    """
    A tensor class for the Logarithmic Number System (LNS).
    
    In LNS, numbers are represented as logarithms, which simplifies
    multiplication and division but complicates addition and subtraction.
    """
    
    # Constants for handling special values
    NEGATIVE_INFINITY = float('-inf')
    ZERO_PLACEHOLDER = float('-inf')  # log(0) is undefined, use -inf as placeholder
    
    def __init__(self, data, base=2.0, already_log=False):
        """
        Initialize an LNS tensor.
        
        Args:
            data: PyTorch tensor or NumPy array or scalar
            base: The base of the logarithm (default: 2.0)
            already_log: If True, assumes data is already in log form
        """
        if isinstance(data, LNSTensor):
            self.log_data = data.log_data
            self.base = data.base
            return
            
        # Convert input to PyTorch tensor if it's not already
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        self.base = base
        
        if already_log:
            self.log_data = data
        else:
            # Handle zeros and negative values
            mask_zero = (data == 0)
            mask_negative = (data < 0)
            
            # Make a copy to avoid modifying the original data
            log_data = torch.clone(data).to(torch.float32)
            
            # Set negative values to their absolute values temporarily
            log_data[mask_negative] = -log_data[mask_negative]
            
            # Convert to logarithmic representation
            log_data = torch.log(log_data) / math.log(self.base)
            
            # Set zeros to ZERO_PLACEHOLDER
            log_data[mask_zero] = self.ZERO_PLACEHOLDER
            
            # Mark negative values with a separate sign tensor or structure
            # This is a simple approach; a more sophisticated implementation might use
            # a dedicated sign bit or other representation
            self.sign = torch.ones_like(data)
            self.sign[mask_negative] = -1
            
            self.log_data = log_data
    
    def to_float(self):
        """Convert from LNS back to floating point representation."""
        # Create a mask for non-zero values
        mask_non_zero = (self.log_data != self.ZERO_PLACEHOLDER)
        
        # Initialize result tensor with zeros
        result = torch.zeros_like(self.log_data)
        
        # Convert non-zero values back to linear domain
        result[mask_non_zero] = torch.pow(self.base, self.log_data[mask_non_zero])
        
        # Apply sign
        if hasattr(self, 'sign'):
            result = result * self.sign
            
        return result
    
    def __add__(self, other):
        """Addition in LNS domain (requires special handling)."""
        if not isinstance(other, LNSTensor):
            other = LNSTensor(other, base=self.base)
        
        # Ensure same base
        if self.base != other.base:
            raise ValueError("LNSTensor bases must match for addition")
        
        # Handle signs (this is a simplified approach)
        # A complete implementation would handle different sign combinations properly
        
        # For positive numbers, use the log addition formula:
        # log(a + b) = log(a) + log(1 + b/a) = log(a) + log(1 + b^(log(b)-log(a)))
        # Assuming a >= b (if not, swap them)
        
        # Make copies to avoid modifying originals
        x = self.log_data.clone()
        y = other.log_data.clone()
        
        # Find which values are larger
        max_log = torch.maximum(x, y)
        min_log = torch.minimum(x, y)
        
        # Use log addition formula
        result = max_log + torch.log(1 + torch.exp(min_log - max_log)) / math.log(self.base)
        
        # Handle cases where one operand is zero
        mask_x_zero = (x == self.ZERO_PLACEHOLDER)
        mask_y_zero = (y == self.ZERO_PLACEHOLDER)
        
        result[mask_x_zero] = y[mask_x_zero]
        result[mask_y_zero] = x[mask_y_zero]
        result[mask_x_zero & mask_y_zero] = self.ZERO_PLACEHOLDER
        
        
        output = LNSTensor(0)
        output.log_data = result
        output.base = self.base
        return output
    
    def __mul__(self, other):
        """Multiplication in LNS domain (just add the logs)."""
        if not isinstance(other, LNSTensor):
            other = LNSTensor(other, base=self.base)
            
        # Ensure same base
        if self.base != other.base:
            raise ValueError("LNSTensor bases must match for multiplication")
        
        # Multiplication is addition in log domain
        result = self.log_data + other.log_data
        
        # Handle multiplication involving zeros
        mask_self_zero = (self.log_data == self.ZERO_PLACEHOLDER)
        mask_other_zero = (other.log_data == self.ZERO_PLACEHOLDER)
        result[mask_self_zero | mask_other_zero] = self.ZERO_PLACEHOLDER
        
        # Handle sign for multiplication
        if hasattr(self, 'sign') and hasattr(other, 'sign'):
            sign = self.sign * other.sign
        elif hasattr(self, 'sign'):
            sign = self.sign
        elif hasattr(other, 'sign'):
            sign = other.sign
        else:
            sign = torch.ones_like(result)
            
        output = LNSTensor(0)
        output.log_data = result
        output.sign = sign
        output.base = self.base
        return output
    
    def __sub__(self, other):
        """Subtraction in LNS domain."""
        if not isinstance(other, LNSTensor):
            other = LNSTensor(other, base=self.base)
            
        # For subtraction, we negate the second operand and add
        # This is a placeholder for a proper implementation
        negated_other = LNSTensor(0)
        negated_other.log_data = other.log_data
        negated_other.base = other.base
        if hasattr(other, 'sign'):
            negated_other.sign = -other.sign
        else:
            negated_other.sign = -torch.ones_like(other.log_data)
            
        return self.__add__(negated_other)
    
    def __truediv__(self, other):
        """Division in LNS domain (subtract the logs)."""
        if not isinstance(other, LNSTensor):
            other = LNSTensor(other, base=self.base)
            
        # Ensure same base
        if self.base != other.base:
            raise ValueError("LNSTensor bases must match for division")
        
        # Division is subtraction in log domain
        result = self.log_data - other.log_data
        
        # Handle division involving zero
        mask_self_zero = (self.log_data == self.ZERO_PLACEHOLDER)
        mask_other_zero = (other.log_data == self.ZERO_PLACEHOLDER)
        
        # x/0 is undefined (or infinity)
        result[mask_other_zero] = float('inf')
        
        # 0/x = 0 (except when x is also 0)
        result[mask_self_zero & ~mask_other_zero] = self.ZERO_PLACEHOLDER
        
        # 0/0 is undefined (NaN)
        result[mask_self_zero & mask_other_zero] = float('nan')
        
        # Handle sign for division
        if hasattr(self, 'sign') and hasattr(other, 'sign'):
            sign = self.sign / other.sign
        elif hasattr(self, 'sign'):
            sign = self.sign
        elif hasattr(other, 'sign'):
            sign = 1.0 / other.sign
        else:
            sign = torch.ones_like(result)
            
        output = LNSTensor(0)
        output.log_data = result
        output.sign = sign
        output.base = self.base
        return output
    
    def __repr__(self):
        """String representation of the LNSTensor."""
        return f"LNSTensor(log_data={self.log_data}, base={self.base})"
    
    def __pow__(self, exponent):
        """Power operation in LNS (multiply the log by the exponent)."""
        if isinstance(exponent, LNSTensor):
            # If exponent is in LNS, need to convert to float
            exponent = exponent.to_float()
        
        # Power operation is multiplication in log domain
        result = self.log_data * exponent
        
        # Handle zeros
        mask_zero = (self.log_data == self.ZERO_PLACEHOLDER)
        result[mask_zero] = self.ZERO_PLACEHOLDER
        
        output = LNSTensor(0)
        output.log_data = result
        output.base = self.base
        
        # Handle sign for power operation (simplified)
        if hasattr(self, 'sign'):
            # For even exponents, result is always positive
            # For odd exponents, sign remains the same
            # This is a simplification; a full implementation would need more cases
            is_even = torch.fmod(exponent, 2) == 0
            sign = torch.ones_like(self.sign)
            sign[~is_even] = self.sign[~is_even]
            output.sign = sign
        
        return output