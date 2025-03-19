import torch
import torch.nn as nn
import sys
import os
from ..lns_tensor.lns_tensor import LNSTensor
from ..lns_autograd.functions import lns_matmul_autograd

class LNSLinear(nn.Module):
    """
    Linear layer implementation using LNS arithmetic.
    
    This layer performs y = xW^T + b using LNS operations.
    """
    def __init__(self, in_features, out_features, bias=True, base=2.0):
        """
        Initialize the LNS linear layer.
        
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If set to False, the layer will not learn an additive bias
            base: The base of the logarithm for LNS representation
        """
        super(LNSLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base = base
        
        # Initialize weights and biases as regular PyTorch parameters
        # The actual conversion to LNS will happen in forward pass
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Reset the parameters using Xavier uniform initialization.
        """
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        """
        Forward pass of the LNS linear layer.
        
        Args:
            x: Input tensor or LNSTensor
            
        Returns:
            LNSTensor result of the linear transformation
        """
        # Convert input to LNSTensor if it isn't already
        if not isinstance(x, LNSTensor):
            x = LNSTensor(x, base=self.base)
            
        # Convert weight to LNSTensor
        weight_lns = LNSTensor(self.weight.data, base=self.base)
        
        # Matrix multiplication: x @ weight.t
        output = lns_matmul_autograd(x, weight_lns.transpose(0, 1))
        
        # Add bias if it exists
        if self.bias is not None:
            bias_lns = LNSTensor(self.bias.data, base=self.base)
            output = output + bias_lns
            
        return output
    
    def extra_repr(self):
        """
        Return a string representation of the layer.
        """
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, base={self.base}'