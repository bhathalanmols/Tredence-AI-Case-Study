import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
import numpy as np

# 1. The Prunable Linear Layer
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # weights and biases
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # gate scores
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        nn.init.constant_(self.gate_scores, 1.0)

    def forward(self, x):
        # raw scores to 0 and 1
        gates = torch.sigmoid(self.gate_scores)
        
        # prune weights
        pruned_weights = self.weight * gates
        
        return F.linear(x, pruned_weights, self.bias)

# 2. The Neural Network & Loss
class SelfPruningNetwork(nn.Module):
    def __init__(self, input_size=32*32*3, num_classes=10):
        super(SelfPruningNetwork, self).__init__()
        self.flatten = nn.Flatten()
        
        # Feed-forward
        self.fc1 = PrunableLinear(input_size, 512)
        self.fc2 = PrunableLinear(512, 128)
        self.fc3 = PrunableLinear(128, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x

def calculate_sparsity_loss(model):
    """Calculates the L1 penalty for the gate scores to encourage sparsity."""
    sparsity_loss = 0.0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            # L1 norm (sum of absolute values, but gates are already > 0)
            sparsity_loss += torch.sum(gates)
    return sparsity_loss

# 3. Training & Evaluation Loop
def train_and_evaluate(lmbda, trainloader, testloader, device, epochs=100):
    print(f"\n[{lmbda}] Starting training with lambda (λ) = {lmbda}")
    
    model = SelfPruningNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Model Training
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            cls_loss = criterion(outputs, labels)
            sparse_loss = calculate_sparsity_loss(model)
            
            # Main functionality: classification accuracy with sparsity
            total_loss = cls_loss + (lmbda * sparse_loss)
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        # update every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[{lmbda}] Epoch {epoch+1}/{epochs} | Avg Loss: {running_loss/len(trainloader):.4f}")

    # Evaluation
    model.eval()
    correct, total, total_gates, pruned_gates = 0, 0, 0, 0
    all_gate_values = []

    with torch.no_grad():
        # Accuracy
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Sparsity Level
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                all_gate_values.extend(gates.cpu().numpy().flatten())
                total_gates += gates.numel()
                # A gate is considered "pruned" if its value falls below 0.01
                pruned_gates += torch.sum(gates < 0.01).item()

    accuracy = 100 * correct / total
    sparsity_level = 100 * pruned_gates / total_gates

    print(f"[{lmbda}] Final Test Accuracy: {accuracy:.2f}% | Sparsity Level: {sparsity_level:.2f}%")
    return accuracy, sparsity_level, np.array(all_gate_values)

# 4. Execution & Experimentation
if __name__ == "__main__":
    # Train on local
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Transform and Load CIFAR-10 Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print("Loading datasets...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    # Test against different lambdas
    lambda_values = [0.00001, 0.0001, 0.001]
    results = []
    
    best_gates = None
    best_sparsity = 0

    # Iterations
    for lmbda in lambda_values:
        acc, spars, gates = train_and_evaluate(lmbda, trainloader, testloader, device, epochs=100)
        results.append((lmbda, acc, spars))
        
        # plot
        plt.figure(figsize=(10, 6))
        plt.hist(gates, bins=50, color='#2ca02c', alpha=0.75, edgecolor='black')
        plt.title(f'Distribution of Final Gate Values (λ = {lmbda})')
        plt.xlabel('Gate Value (0 = Pruned, 1 = Active)')
        plt.ylabel('Frequency (Number of Weights)')
        plt.grid(axis='y', alpha=0.75)
        
        if lmbda == 0.00001:
            filename = 'dist_low.png'
        elif lmbda == 0.0001:
            filename = 'dist_mid.png'
        elif lmbda == 0.001:
            filename = 'dist_high.png'
        else:
            filename = f'gate_distribution_{lmbda}.png'
            
        plt.savefig(filename)
        plt.close()
        print(f"Saved distribution plot as '{filename}'.")

    # Markdown Table for the Case Study
    print("\n" + "="*40)
    print("=== Final Results Table (Markdown) ===")
    print("="*40)
    print("| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |")
    print("|---|---|---|")
    for res in results:
        print(f"| {res[0]} | {res[1]:.2f} | {res[2]:.2f} |")
    print("="*40 + "\n")

    print("Experiment complete. Ready for Git push!")