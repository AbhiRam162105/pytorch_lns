import unittest
import torch
import numpy as np
import sys
import os
import math

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch_lns.lns_tensor.lns_tensor import LNSTensor

class TestLNSTensor(unittest.TestCase):
    """
    Unit tests for the LNSTensor class.
    """
    def test_creation(self):
        """Test LNSTensor creation from different sources."""
        # From scalar
        scalar = 5.0
        tensor_scalar = LNSTensor(scalar)
        self.assertIsInstance(tensor_scalar, LNSTensor)
        self.assertAlmostEqual(tensor_scalar.to_float().item(), scalar, places=5)
        
        # From list
        lst = [1.0, 2.0, 3.0]
        tensor_list = LNSTensor(lst)
        self.assertIsInstance(tensor_list, LNSTensor)
        self.assertTrue(torch.allclose(tensor_list.to_float(), torch.tensor(lst), rtol=1e-5))
        
        # From numpy array
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor_array = LNSTensor(arr)
        self.assertIsInstance(tensor_array, LNSTensor)
        self.assertTrue(torch.allclose(tensor_array.to_float(), torch.tensor(arr), rtol=1e-5))
        
        # From PyTorch tensor
        pt_tensor = torch.tensor([5.0, 6.0, 7.0])
        tensor_pt = LNSTensor(pt_tensor)
        self.assertIsInstance(tensor_pt, LNSTensor)
        self.assertTrue(torch.allclose(tensor_pt.to_float(), pt_tensor, rtol=1e-5))
        
    def test_zero_handling(self):
        """Test handling of zero values."""
        data = torch.tensor([0.0, 1.0, 2.0, 3.0])
        lns = LNSTensor(data)
        
        # Check conversion back to float
        float_data = lns.to_float()
        self.assertTrue(torch.allclose(float_data, data, rtol=1e-5))
        
    def test_negative_handling(self):
        """Test handling of negative values."""
        data = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        lns = LNSTensor(data)
        
        # Check conversion back to float
        float_data = lns.to_float()
        self.assertTrue(torch.allclose(float_data, data, rtol=1e-5))
        
    def test_addition(self):
        """Test addition operation."""
        a = LNSTensor(torch.tensor([1.0, 2.0, 3.0]))
        b = LNSTensor(torch.tensor([4.0, 5.0, 6.0]))
        result = a + b
        
        expected = torch.tensor([5.0, 7.0, 9.0])
        self.assertTrue(torch.allclose(result.to_float(), expected, rtol=1e-3))
        
        # Test addition with zero
        c = LNSTensor(torch.tensor([0.0, 1.0, 2.0]))
        d = LNSTensor(torch.tensor([3.0, 0.0, 5.0]))
        result = c + d
        
        expected = torch.tensor([3.0, 1.0, 7.0])
        self.assertTrue(torch.allclose(result.to_float(), expected, rtol=1e-3))
        
    def test_multiplication(self):
        """Test multiplication operation."""
        a = LNSTensor(torch.tensor([1.0, 2.0, 3.0]))
        b = LNSTensor(torch.tensor([4.0, 5.0, 6.0]))
        result = a * b
        
        expected = torch.tensor([4.0, 10.0, 18.0])
        self.assertTrue(torch.allclose(result.to_float(), expected, rtol=1e-3))
        
        # Test multiplication with zero
        c = LNSTensor(torch.tensor([0.0, 1.0, 2.0]))
        d = LNSTensor(torch.tensor([3.0, 0.0, 5.0]))
        result = c * d
        
        expected = torch.tensor([0.0, 0.0, 10.0])
        self.assertTrue(torch.allclose(result.to_float(), expected, rtol=1e-3))
        
    def test_division(self):
        """Test division operation."""
        a = LNSTensor(torch.tensor([4.0, 10.0, 18.0]))
        b = LNSTensor(torch.tensor([2.0, 5.0, 6.0]))
        result = a / b
        
        expected = torch.tensor([2.0, 2.0, 3.0])
        self.assertTrue(torch.allclose(result.to_float(), expected, rtol=1e-3))
        
    def test_subtraction(self):
        """Test subtraction operation."""
        a = LNSTensor(torch.tensor([5.0, 7.0, 9.0]))
        b = LNSTensor(torch.tensor([1.0, 2.0, 3.0]))
        result = a - b
        
        expected = torch.tensor([4.0, 5.0, 6.0])
        self.assertTrue(torch.allclose(result.to_float(), expected, rtol=1e-3))
        
    def test_different_bases(self):
        """Test using different logarithm bases."""
        data = torch.tensor([1.0, 2.0, 4.0, 8.0])
        
        # Base 2
        lns_base2 = LNSTensor(data, base=2.0)
        self.assertTrue(torch.allclose(lns_base2.to_float(), data, rtol=1e-5))
        
        # Base e
        lns_base_e = LNSTensor(data, base=math.e)
        self.assertTrue(torch.allclose(lns_base_e.to_float(), data, rtol=1e-5))
        
        # Base 10
        lns_base10 = LNSTensor(data, base=10.0)
        self.assertTrue(torch.allclose(lns_base10.to_float(), data, rtol=1e-5))


if __name__ == '__main__':
    unittest.main()