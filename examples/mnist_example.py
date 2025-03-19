import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LNS modules
from pytorch_lns.lns_tensor.lns_tensor import LNSTensor
from lns_nn import LNSLinear, LNSConv2d, LNSMaxPool2d
from lns_ops import lns_relu, lns_log_softmax, lns_nll_loss


class LNSNet(nn.Module):
    """
    Simple CNN for MNIST using LNS operations.
    """
    def __init__(self, base=2.0):
        super(LNSNet, self).__init__()
        self.base = base
        
        # Define the convolutional layers
        self.conv1 = LNSConv2d(1, 32, kernel_size=3, stride=1, base=base)
        self.conv2 = LNSConv2d(32, 64, kernel_size=3, stride=1, base=base)
        
        # Pooling layer
        self.pool = LNSMaxPool2d(kernel_size=2, base=base)
        
        # Define the fully connected layers
        self.fc1 = LNSLinear(64 * 12 * 12, 128, base=base)
        self.fc2 = LNSLinear(128, 10, base=base)
    
    def forward(self, x):
        # Convert input to LNSTensor if it's not already
        if not isinstance(x, LNSTensor):
            x = LNSTensor(x, base=self.base)
            
        # First convolutional layer followed by ReLU and pooling
        x = lns_relu(self.conv1(x))
        x = self.pool(x)
        
        # Second convolutional layer followed by ReLU and pooling
        x = lns_relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten the tensor for the fully connected layers
        x_shape = x.to_float().shape
        x = LNSTensor(x.to_float().view(x_shape[0], -1), base=self.base)
        
        # Fully connected layers with ReLU
        x = lns_relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class StandardNet(nn.Module):
    """
    Simple CNN for MNIST using standard floating point operations.
    This is used for comparison with the LNS network.
    """
    def __init__(self):
        super(StandardNet, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # First convolutional layer followed by ReLU and pooling
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        # Second convolutional layer followed by ReLU and pooling
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


def train_lns_model(model, device, train_loader, optimizer, epoch):
    """Training function for the LNS model."""
    model.train()
    correct = 0
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Convert output back to standard representation for loss calculation
        output_float = output.to_float()
        
        # Calculate loss
        loss = nn.CrossEntropyLoss()(output_float, target)
        total_loss += loss.item() * len(data)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = output_float.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # Print training statistics
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Training: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)')
    
    return avg_loss, accuracy


def train_standard_model(model, device, train_loader, optimizer, epoch):
    """Training function for the standard floating point model."""
    model.train()
    correct = 0
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate loss
        loss = nn.CrossEntropyLoss()(output, target)
        total_loss += loss.item() * len(data)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # Print training statistics
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Training: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)')
    
    return avg_loss, accuracy


def test(model, device, test_loader, is_lns=True):
    """Test function for either model type."""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Convert output to float if it's an LNS model
            if is_lns:
                output = output.to_float()
                
            # Calculate loss
            test_loss += nn.CrossEntropyLoss()(output, target).item() * len(data)
            
            # Get the prediction and calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    # Print test statistics
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    
    return test_loss, accuracy


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training parameters
    batch_size = 64
    epochs = 5
    lr = 0.01
    
    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize models
    lns_model = LNSNet(base=2.0).to(device)
    standard_model = StandardNet().to(device)
    
    # Optimizers
    lns_optimizer = optim.SGD(lns_model.parameters(), lr=lr, momentum=0.9)
    standard_optimizer = optim.SGD(standard_model.parameters(), lr=lr, momentum=0.9)
    
    # Lists to store metrics
    lns_train_losses = []
    lns_train_accuracies = []
    lns_test_losses = []
    lns_test_accuracies = []
    
    standard_train_losses = []
    standard_train_accuracies = []
    standard_test_losses = []
    standard_test_accuracies = []
    
    lns_times = []
    standard_times = []
    
    # Training and evaluation loop
    for epoch in range(1, epochs + 1):
        # Train and evaluate LNS model
        print(f"\nEpoch {epoch} - LNS Model Training:")
        start_time = time.time()
        lns_train_loss, lns_train_acc = train_lns_model(lns_model, device, train_loader, lns_optimizer, epoch)
        lns_test_loss, lns_test_acc = test(lns_model, device, test_loader, is_lns=True)
        lns_time = time.time() - start_time
        
        # Store LNS metrics
        lns_train_losses.append(lns_train_loss)
        lns_train_accuracies.append(lns_train_acc)
        lns_test_losses.append(lns_test_loss)
        lns_test_accuracies.append(lns_test_acc)
        lns_times.append(lns_time)
        
        print(f"LNS Epoch time: {lns_time:.2f} seconds")
        
        # Train and evaluate standard model
        print(f"\nEpoch {epoch} - Standard Model Training:")
        start_time = time.time()
        std_train_loss, std_train_acc = train_standard_model(standard_model, device, train_loader, standard_optimizer, epoch)
        std_test_loss, std_test_acc = test(standard_model, device, test_loader, is_lns=False)
        std_time = time.time() - start_time
        
        # Store standard model metrics
        standard_train_losses.append(std_train_loss)
        standard_train_accuracies.append(std_train_acc)
        standard_test_losses.append(std_test_loss)
        standard_test_accuracies.append(std_test_acc)
        standard_times.append(std_time)
        
        print(f"Standard Epoch time: {std_time:.2f} seconds")
    
    # Plot training and test losses
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), lns_train_losses, 'b-', label='LNS Train Loss')
    plt.plot(range(1, epochs + 1), standard_train_losses, 'r-', label='Standard Train Loss')
    plt.plot(range(1, epochs + 1), lns_test_losses, 'b--', label='LNS Test Loss')
    plt.plot(range(1, epochs + 1), standard_test_losses, 'r--', label='Standard Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), lns_train_accuracies, 'b-', label='LNS Train Accuracy')
    plt.plot(range(1, epochs + 1), standard_train_accuracies, 'r-', label='Standard Train Accuracy')
    plt.plot(range(1, epochs + 1), lns_test_accuracies, 'b--', label='LNS Test Accuracy')
    plt.plot(range(1, epochs + 1), standard_test_accuracies, 'r--', label='Standard Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('lns_vs_standard_performance.png')
    
    # Plot training times
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, epochs + 1), lns_times, alpha=0.7, label='LNS')
    plt.bar(range(1, epochs + 1), standard_times, alpha=0.7, label='Standard')
    plt.xlabel('Epochs')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lns_vs_standard_time.png')
    
    # Print a summary of the results
    print("\n=== Summary of Results ===")
    print(f"Final LNS Test Accuracy: {lns_test_accuracies[-1]:.2f}%")
    print(f"Final Standard Test Accuracy: {standard_test_accuracies[-1]:.2f}%")
    print(f"Average LNS Epoch Time: {sum(lns_times) / len(lns_times):.2f} seconds")
    print(f"Average Standard Epoch Time: {sum(standard_times) / len(standard_times):.2f} seconds")
    

if __name__ == "__main__":
    main()