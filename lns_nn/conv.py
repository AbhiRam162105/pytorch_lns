import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..lns_tensor import LNSTensor

class LNSConv2d(nn.Module):
    """
    2D convolution layer implementation using LNS arithmetic.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, base=2.0):
        """
        Initialize the LNS 2D convolution layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides of the input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections from input to output channels
            bias: If True, adds a learnable bias to the output
            base: The base of the logarithm for LNS representation
        """
        super(LNSConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base = base
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
            
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation
            
        self.groups = groups
        
        # Initialize weight and bias as regular PyTorch parameters
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Reset the parameters using Kaiming uniform initialization.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        """
        Forward pass of the LNS convolution layer.
        
        Args:
            x: Input tensor or LNSTensor
            
        Returns:
            LNSTensor result of the convolution operation
        """
        # Convert input to LNSTensor if it isn't already
        if not isinstance(x, LNSTensor):
            x = LNSTensor(x, base=self.base)
            
        # For convolution, we'll convert back to float domain, perform the operation,
        # and then convert back to LNS. This is a practical approach for complex operations
        # that aren't easily implementable in the LNS domain.
        
        x_float = x.to_float()
        weight_float = self.weight.data
        
        # Perform standard convolution
        if self.bias is not None:
            bias_float = self.bias.data
            output_float = F.conv2d(
                x_float, weight_float, bias_float, 
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
            )
        else:
            output_float = F.conv2d(
                x_float, weight_float, None, 
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
            )
        
        # Convert back to LNS
        output_lns = LNSTensor(output_float, base=self.base)
        
        return output_lns
    
    def extra_repr(self):
        """
        Return a string representation of the layer.
        """
        s = (f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, '
             f'stride={self.stride}, padding={self.padding}')
        if self.dilation != (1, 1):
            s += f', dilation={self.dilation}'
        if self.groups != 1:
            s += f', groups={self.groups}'
        if self.bias is None:
            s += ', bias=False'
        s += f', base={self.base}'
        return s


class LNSConv1d(nn.Module):
    """
    1D convolution layer implementation using LNS arithmetic.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, base=2.0):
        """
        Initialize the LNS 1D convolution layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides of the input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections from input to output channels
            bias: If True, adds a learnable bias to the output
            base: The base of the logarithm for LNS representation
        """
        super(LNSConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.base = base
        
        # Initialize weight and bias as regular PyTorch parameters
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, kernel_size
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Reset the parameters using Kaiming uniform initialization.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        """
        Forward pass of the LNS 1D convolution layer.
        
        Args:
            x: Input tensor or LNSTensor
            
        Returns:
            LNSTensor result of the convolution operation
        """
        # Convert input to LNSTensor if it isn't already
        if not isinstance(x, LNSTensor):
            x = LNSTensor(x, base=self.base)
        
        # Similar to Conv2d, convert to float, perform operation, convert back to LNS
        x_float = x.to_float()
        weight_float = self.weight.data
        
        # Perform standard convolution
        if self.bias is not None:
            bias_float = self.bias.data
            output_float = F.conv1d(
                x_float, weight_float, bias_float, 
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
            )
        else:
            output_float = F.conv1d(
                x_float, weight_float, None, 
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
            )
        
        # Convert back to LNS
        output_lns = LNSTensor(output_float, base=self.base)
        
        return output_lns
    
    def extra_repr(self):
        """
        Return a string representation of the layer.
        """
        s = (f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, '
             f'stride={self.stride}, padding={self.padding}')
        if self.dilation != 1:
            s += f', dilation={self.dilation}'
        if self.groups != 1:
            s += f', groups={self.groups}'
        if self.bias is None:
            s += ', bias=False'
        s += f', base={self.base}'
        return s


class LNSMaxPool2d(nn.Module):
    """
    2D max pooling implementation using LNS arithmetic.
    """
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, base=2.0):
        """
        Initialize the LNS 2D max pooling layer.
        
        Args:
            kernel_size: Size of the window to take a max over
            stride: Stride of the window
            padding: Zero-padding added to both sides of the input
            dilation: A parameter that controls the stride of elements in the window
            base: The base of the logarithm for LNS representation
        """
        super(LNSMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.base = base
    
    def forward(self, x):
        """
        Forward pass of the LNS max pooling layer.
        
        Args:
            x: Input LNSTensor
            
        Returns:
            LNSTensor result of the max pooling operation
        """
        # Convert input to LNSTensor if it isn't already
        if not isinstance(x, LNSTensor):
            x = LNSTensor(x, base=self.base)
            
        # In LNS, the max value corresponds to the maximum log value for positive numbers
        # We need to consider the sign for a complete implementation
        
        # For simplicity, convert to float, perform max pooling, convert back
        x_float = x.to_float()
        
        # Perform max pooling
        output_float = F.max_pool2d(
            x_float, kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, dilation=self.dilation
        )
        
        # Convert back to LNS
        output_lns = LNSTensor(output_float, base=self.base)
        
        return output_lns
    
    def extra_repr(self):
        """
        Return a string representation of the layer.
        """
        return (f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}, dilation={self.dilation}, base={self.base}')


class LNSAvgPool2d(nn.Module):
    """
    2D average pooling implementation using LNS arithmetic.
    """
    def __init__(self, kernel_size, stride=None, padding=0, base=2.0):
        """
        Initialize the LNS 2D average pooling layer.
        
        Args:
            kernel_size: Size of the window to take an average over
            stride: Stride of the window
            padding: Zero-padding added to both sides of the input
            base: The base of the logarithm for LNS representation
        """
        super(LNSAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.base = base
    
    def forward(self, x):
        """
        Forward pass of the LNS average pooling layer.
        
        Args:
            x: Input LNSTensor
            
        Returns:
            LNSTensor result of the average pooling operation
        """
        # Convert input to LNSTensor if it isn't already
        if not isinstance(x, LNSTensor):
            x = LNSTensor(x, base=self.base)
            
        # Average pooling is more complex in LNS, so we'll use the float domain
        x_float = x.to_float()
        
        # Perform average pooling
        output_float = F.avg_pool2d(
            x_float, kernel_size=self.kernel_size, 
            stride=self.stride, padding=self.padding
        )
        
        # Convert back to LNS
        output_lns = LNSTensor(output_float, base=self.base)
        
        return output_lns
    
    def extra_repr(self):
        """
        Return a string representation of the layer.
        """
        return (f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}, base={self.base}')