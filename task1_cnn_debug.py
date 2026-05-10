import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
import time

# ── Reproducibility ──────────────────────────────────────────
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Data Loading ──────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False)

classes = train_dataset.classes
print(f"Classes: {classes}")


# ════════════════════════════════════════════════════════════
# ORIGINAL (BUGGY) CNN
# BUG: sigmoid activations cause vanishing gradients in deep
#      networks. Gradients shrink to near-zero, so early layers
#      learn almost nothing.
# ════════════════════════════════════════════════════════════
class InitialCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1   = nn.Linear(256 * 2 * 2, 256)
        self.fc2   = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x)); x = F.max_pool2d(x, 2)  # BUG: sigmoid
        x = torch.sigmoid(self.conv2(x)); x = F.max_pool2d(x, 2)  # BUG: sigmoid
        x = torch.sigmoid(self.conv3(x)); x = F.max_pool2d(x, 2)  # BUG: sigmoid
        x = torch.sigmoid(self.conv4(x)); x = F.max_pool2d(x, 2)  # BUG: sigmoid
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))                             # BUG: sigmoid
        x = self.fc2(x)
        return x


# ════════════════════════════════════════════════════════════
# IMPROVED CNN
# FIX 1: Replace sigmoid → ReLU  (stops vanishing gradients)
# FIX 2: Add BatchNorm after conv1 (stabilises training)
#         This counts as the ONE extra layer required.
# ════════════════════════════════════════════════════════════
class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)   # <-- THE ONE NEW LAYER
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1   = nn.Linear(256 * 2 * 2, 256)
        self.fc2   = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.max_pool2d(x, 2)  # FIXED
        x = F.relu(self.conv2(x));           x = F.max_pool2d(x, 2)  # FIXED
        x = F.relu(self.conv3(x));           x = F.max_pool2d(x, 2)  # FIXED
        x = F.relu(self.conv4(x));           x = F.max_pool2d(x, 2)  # FIXED
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))                                        # FIXED
        x = self.fc2(x)
        return x


# ── Training & Testing Functions ──────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0, 0, 0
    gradient_norms = {}

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Collect gradient norms per layer
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if name not in gradient_norms:
                    gradient_norms[name] = []
                gradient_norms[name].append(grad_norm)

        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100 * correct / total, gradient_norms


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100 * correct / total


def run_training(model, num_epochs=15, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss':  [], 'test_acc': [],
        'grad_norms': []
    }

    for epoch in range(num_epochs):
        t_loss, t_acc, grad_norms = train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = evaluate(model, test_loader, criterion)

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['test_loss'].append(v_loss)
        history['test_acc'].append(v_acc)
        history['grad_norms'].append(grad_norms)

        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train Loss: {t_loss:.4f}  Train Acc: {t_acc:.1f}%  "
              f"Test Acc: {v_acc:.1f}%")

    return history


# ── Run Both Models ────────────────────────────────────────────
NUM_EPOCHS = 15

print("\n" + "="*55)
print("TRAINING: InitialCNN (buggy - sigmoid activations)")
print("="*55)
initial_model = InitialCNN().to(device)
initial_history = run_training(initial_model, num_epochs=NUM_EPOCHS)

print("\n" + "="*55)
print("TRAINING: ImprovedCNN (fixed - ReLU + BatchNorm)")
print("="*55)
improved_model = ImprovedCNN().to(device)
improved_history = run_training(improved_model, num_epochs=NUM_EPOCHS)


# ── Plot 1: Learning Curves Comparison ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs = range(1, NUM_EPOCHS + 1)

# Loss
axes[0].plot(epochs, initial_history['train_loss'], 'r--', label='Initial CNN Train Loss')
axes[0].plot(epochs, improved_history['train_loss'], 'b-',  label='Improved CNN Train Loss')
axes[0].set_title('Training Loss Comparison', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy
axes[1].plot(epochs, initial_history['train_acc'], 'r--', label='Initial CNN Train Acc')
axes[1].plot(epochs, initial_history['test_acc'],  'r:',  label='Initial CNN Test Acc')
axes[1].plot(epochs, improved_history['train_acc'], 'b-',  label='Improved CNN Train Acc')
axes[1].plot(epochs, improved_history['test_acc'],  'b:',  label='Improved CNN Test Acc')
axes[1].set_title('Accuracy Comparison', fontsize=14)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True)

plt.suptitle('Task 1: Initial vs Improved CNN', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task1_learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: task1_learning_curves.png")


# ── Plot 2: Gradient Magnitudes per Layer ────────────────────
def plot_gradient_norms(history, model_name, filename):
    # Average gradient norm per layer per epoch
    all_layer_names = list(history['grad_norms'][0].keys())
    # Only keep weight layers (skip bias) for clarity
    layer_names = [n for n in all_layer_names if 'weight' in n]

    avg_norms = {name: [] for name in layer_names}
    for epoch_grads in history['grad_norms']:
        for name in layer_names:
            if name in epoch_grads:
                avg_norms[name].append(np.mean(epoch_grads[name]))
            else:
                avg_norms[name].append(0)

    n_layers = len(layer_names)
    cols = 3
    rows = (n_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()

    for i, name in enumerate(layer_names):
        axes[i].plot(range(1, NUM_EPOCHS + 1), avg_norms[name], marker='o', ms=4)
        axes[i].set_title(name.replace('.', '\n'), fontsize=9)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Gradient Norm')
        axes[i].grid(True)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'Gradient Magnitudes per Layer: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

plot_gradient_norms(initial_history, 'InitialCNN (sigmoid)', 'task1_gradients_initial.png')
plot_gradient_norms(improved_history, 'ImprovedCNN (ReLU + BN)', 'task1_gradients_improved.png')


# ── Final Summary ─────────────────────────────────────────────
print("\n" + "="*55)
print("FINAL RESULTS SUMMARY")
print("="*55)
print(f"{'Model':<20} {'Train Loss':>12} {'Train Acc':>10} {'Test Acc':>10}")
print("-"*55)
print(f"{'InitialCNN':<20} "
      f"{initial_history['train_loss'][-1]:>12.4f} "
      f"{initial_history['train_acc'][-1]:>10.1f}% "
      f"{initial_history['test_acc'][-1]:>10.1f}%")
print(f"{'ImprovedCNN':<20} "
      f"{improved_history['train_loss'][-1]:>12.4f} "
      f"{improved_history['train_acc'][-1]:>10.1f}% "
      f"{improved_history['test_acc'][-1]:>10.1f}%")
print("="*55)
print("\nDone! Check the saved .png files for your report figures.")
