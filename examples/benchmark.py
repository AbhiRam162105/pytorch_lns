import torch
import time
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lns_tensor import LNSTensor
from lns_ops.basic_ops import lns_add, lns_multiply, lns_divide, lns_exp
from lns_ops.matrix_ops import lns_matmul
from lns_nn import LNSLinear, LNSConv2d

class OperationBenchmark:
    """
    Benchmark class to compare LNS operations with standard floating point operations.
    """
    def __init__(self, device='cuda'):
        """
        Initialize the benchmark class.
        
        Args:
            device: Device to run benchmarks on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.results = {}
        self.sizes = [64, 128, 256, 512, 1024, 2048]
        
    def run_benchmark(self, operation_name, lns_op, std_op, input_generator, num_runs=10):
        """
        Run a benchmark for a specific operation.
        
        Args:
            operation_name: Name of the operation to benchmark
            lns_op: LNS operation function
            std_op: Standard operation function
            input_generator: Function that generates inputs based on size
            num_runs: Number of times to run each operation for averaging
        """
        print(f"Benchmarking {operation_name}...")
        lns_times = []
        std_times = []
        
        for size in self.sizes:
            print(f"  Size: {size}")
            # Generate inputs
            inputs = input_generator(size, self.device)
            
            # Warmup
            for _ in range(5):
                lns_output = lns_op(*inputs['lns'])
                std_output = std_op(*inputs['std'])
                
            # Benchmark LNS operation
            lns_start = time.time()
            for _ in range(num_runs):
                lns_output = lns_op(*inputs['lns'])
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
            lns_end = time.time()
            lns_time = (lns_end - lns_start) / num_runs
            
            # Benchmark standard operation
            std_start = time.time()
            for _ in range(num_runs):
                std_output = std_op(*inputs['std'])
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
            std_end = time.time()
            std_time = (std_end - std_start) / num_runs
            
            lns_times.append(lns_time)
            std_times.append(std_time)
            
            # Verify results (convert LNS to float for comparison)
            if isinstance(lns_output, LNSTensor):
                lns_float = lns_output.to_float()
                # Check if results are close
                if torch.allclose(lns_float, std_output, rtol=1e-3, atol=1e-3):
                    print(f"    Results match within tolerance")
                else:
                    print(f"    WARNING: Results differ! Max diff: {torch.max(torch.abs(lns_float - std_output))}")
        
        # Store results
        self.results[operation_name] = {
            'sizes': self.sizes,
            'lns_times': lns_times,
            'std_times': std_times,
            'speedup': [std / lns for lns, std in zip(lns_times, std_times)]
        }
    
    def benchmark_addition(self):
        """Benchmark addition operations."""
        def input_generator(size, device):
            a_std = torch.rand(size, size, device=device)
            b_std = torch.rand(size, size, device=device)
            a_lns = LNSTensor(a_std)
            b_lns = LNSTensor(b_std)
            return {'std': (a_std, b_std), 'lns': (a_lns, b_lns)}
            
        self.run_benchmark(
            operation_name="Addition",
            lns_op=lambda a, b: lns_add(a, b),
            std_op=lambda a, b: a + b,
            input_generator=input_generator
        )
    
    def benchmark_multiplication(self):
        """Benchmark multiplication operations."""
        def input_generator(size, device):
            a_std = torch.rand(size, size, device=device)
            b_std = torch.rand(size, size, device=device)
            a_lns = LNSTensor(a_std)
            b_lns = LNSTensor(b_std)
            return {'std': (a_std, b_std), 'lns': (a_lns, b_lns)}
            
        self.run_benchmark(
            operation_name="Multiplication",
            lns_op=lambda a, b: lns_multiply(a, b),
            std_op=lambda a, b: a * b,
            input_generator=input_generator
        )
    
    def benchmark_division(self):
        """Benchmark division operations."""
        def input_generator(size, device):
            a_std = torch.rand(size, size, device=device)
            # Avoid division by zero
            b_std = torch.rand(size, size, device=device) + 0.1
            a_lns = LNSTensor(a_std)
            b_lns = LNSTensor(b_std)
            return {'std': (a_std, b_std), 'lns': (a_lns, b_lns)}
            
        self.run_benchmark(
            operation_name="Division",
            lns_op=lambda a, b: lns_divide(a, b),
            std_op=lambda a, b: a / b,
            input_generator=input_generator
        )
    
    def benchmark_matrix_multiply(self):
        """Benchmark matrix multiplication operations."""
        def input_generator(size, device):
            a_std = torch.rand(size, size, device=device)
            b_std = torch.rand(size, size, device=device)
            a_lns = LNSTensor(a_std)
            b_lns = LNSTensor(b_std)
            return {'std': (a_std, b_std), 'lns': (a_lns, b_lns)}
            
        self.run_benchmark(
            operation_name="Matrix Multiplication",
            lns_op=lambda a, b: lns_matmul(a, b),
            std_op=lambda a, b: torch.matmul(a, b),
            input_generator=input_generator
        )
    
    def benchmark_exp(self):
        """Benchmark exponential operations."""
        def input_generator(size, device):
            # Use smaller values to avoid overflow
            a_std = torch.rand(size, size, device=device) * 2 - 1  # Range [-1, 1]
            a_lns = LNSTensor(a_std)
            return {'std': (a_std,), 'lns': (a_lns,)}
            
        self.run_benchmark(
            operation_name="Exponential",
            lns_op=lambda a: lns_exp(a),
            std_op=lambda a: torch.exp(a),
            input_generator=input_generator
        )
    
    def benchmark_linear_layer(self):
        """Benchmark linear layer operations."""
        def input_generator(size, device):
            # Use smaller dimension for large sizes to avoid out of memory
            if size > 512:
                in_features = 128
                out_features = 64
                batch_size = size
            else:
                in_features = size // 2
                out_features = size // 4
                batch_size = 32
                
            x_std = torch.rand(batch_size, in_features, device=device)
            x_lns = LNSTensor(x_std)
            
            # Create standard linear layer
            std_linear = torch.nn.Linear(in_features, out_features).to(device)
            
            # Create LNS linear layer with same weights
            lns_linear = LNSLinear(in_features, out_features, base=2.0)
            lns_linear.weight.data = std_linear.weight.data
            lns_linear.bias.data = std_linear.bias.data
            lns_linear = lns_linear.to(device)
            
            return {
                'std': (x_std, std_linear), 
                'lns': (x_lns, lns_linear)
            }
            
        self.run_benchmark(
            operation_name="Linear Layer",
            lns_op=lambda x, layer: layer(x),
            std_op=lambda x, layer: layer(x),
            input_generator=input_generator
        )
    
    def benchmark_conv_layer(self):
        """Benchmark convolutional layer operations."""
        def input_generator(size, device):
            # Adjust dimensions based on size to avoid out of memory
            if size > 512:
                in_channels = 3
                out_channels = 16
                img_size = min(64, size // 8)
                batch_size = 8
            else:
                in_channels = 3
                out_channels = 16
                img_size = min(128, size)
                batch_size = 16
                
            x_std = torch.rand(batch_size, in_channels, img_size, img_size, device=device)
            x_lns = LNSTensor(x_std)
            
            # Create standard conv layer
            std_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1).to(device)
            
            # Create LNS conv layer with same weights
            lns_conv = LNSConv2d(in_channels, out_channels, kernel_size=3, padding=1, base=2.0)
            lns_conv.weight.data = std_conv.weight.data
            lns_conv.bias.data = std_conv.bias.data
            lns_conv = lns_conv.to(device)
            
            return {
                'std': (x_std, std_conv), 
                'lns': (x_lns, lns_conv)
            }
            
        self.run_benchmark(
            operation_name="Convolutional Layer",
            lns_op=lambda x, layer: layer(x),
            std_op=lambda x, layer: layer(x),
            input_generator=input_generator
        )
    
    def run_all_benchmarks(self):
        """Run all benchmark tests."""
        self.benchmark_addition()
        self.benchmark_multiplication()
        self.benchmark_division()
        self.benchmark_matrix_multiply()
        self.benchmark_exp()
        self.benchmark_linear_layer()
        self.benchmark_conv_layer()
    
    def plot_results(self, save_dir='.'):
        """
        Plot the benchmark results.
        
        Args:
            save_dir: Directory to save the plots
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for op_name, data in self.results.items():
            plt.figure(figsize=(12, 6))
            
            # Plot runtimes
            plt.subplot(1, 2, 1)
            plt.plot(data['sizes'], data['lns_times'], 'b-o', label='LNS')
            plt.plot(data['sizes'], data['std_times'], 'r-o', label='FP32')
            plt.xlabel('Matrix Size')
            plt.ylabel('Time (seconds)')
            plt.title(f'{op_name} Runtime Comparison')
            plt.grid(True)
            plt.legend()
            plt.xscale('log', base=2)
            plt.yscale('log')
            
            # Plot speedup
            plt.subplot(1, 2, 2)
            plt.plot(data['sizes'], data['speedup'], 'g-o')
            plt.axhline(y=1.0, color='k', linestyle='--')
            plt.xlabel('Matrix Size')
            plt.ylabel('Speedup (FP32 / LNS)')
            plt.title(f'{op_name} Speedup')
            plt.grid(True)
            plt.xscale('log', base=2)
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{op_name.replace(" ", "_").lower()}_benchmark.png')
            plt.close()
        
        # Create a summary table
        rows = []
        for op_name, data in self.results.items():
            avg_speedup = sum(data['speedup']) / len(data['speedup'])
            max_speedup = max(data['speedup'])
            rows.append([op_name, avg_speedup, max_speedup])
            
        # Sort by average speedup
        rows.sort(key=lambda x: x[1], reverse=True)
        
        # Plot summary
        plt.figure(figsize=(10, 6))
        op_names = [row[0] for row in rows]
        avg_speedups = [row[1] for row in rows]
        max_speedups = [row[2] for row in rows]
        
        x = range(len(op_names))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], avg_speedups, width, label='Average Speedup')
        plt.bar([i + width/2 for i in x], max_speedups, width, label='Maximum Speedup')
        
        plt.axhline(y=1.0, color='k', linestyle='--')
        plt.xlabel('Operation')
        plt.ylabel('Speedup (FP32 / LNS)')
        plt.title('LNS Performance Summary')
        plt.xticks(x, op_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/overall_speedup_summary.png')
        
        # Print summary table
        print("\nPerformance Summary:")
        print(tabulate(rows, headers=['Operation', 'Avg Speedup', 'Max Speedup'], tablefmt='grid'))
        
        # Save summary table
        with open(f'{save_dir}/benchmark_summary.txt', 'w') as f:
            f.write(tabulate(rows, headers=['Operation', 'Avg Speedup', 'Max Speedup'], tablefmt='grid'))
    
    def save_raw_data(self, save_dir='.'):
        """
        Save the raw benchmark data.
        
        Args:
            save_dir: Directory to save the data
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        import json
        
        # Convert data to JSON serializable format
        json_data = {}
        for op_name, data in self.results.items():
            json_data[op_name] = {
                'sizes': data['sizes'],
                'lns_times': data['lns_times'],
                'std_times': data['std_times'],
                'speedup': data['speedup']
            }
            
        with open(f'{save_dir}/benchmark_results.json', 'w') as f:
            json.dump(json_data, f, indent=2)


def main():
    print("PyTorch LNS Benchmarking Tool")
    print("=============================")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("Running on CPU")
        device = 'cpu'
    
    # Create the benchmark instance
    benchmark = OperationBenchmark(device=device)
    
    # Create output directory for results
    output_dir = 'benchmark_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Run all benchmarks
    benchmark.run_all_benchmarks()
    
    # Plot and save results
    benchmark.plot_results(save_dir=output_dir)
    benchmark.save_raw_data(save_dir=output_dir)
    
    print(f"\nBenchmark complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()