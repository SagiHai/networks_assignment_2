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

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Data ───────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=128, shuffle=False)


# ── Improved CNN (same as Task 1) ─────────────────────────────
class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1   = nn.Linear(256 * 2 * 2, 256)
        self.fc2   = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x));           x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x));           x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x));           x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ── MLP with 2 hidden layers (as required) ────────────────────
# Images are flattened: 32 x 32 x 3 = 3072 features
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 1024)   # hidden layer 1
        self.fc2 = nn.Linear(1024, 512)             # hidden layer 2
        self.fc3 = nn.Linear(512, 10)               # output

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten image to vector
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ── Training helpers ──────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100 * correct / total


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


def run_training(model, num_epochs=15, lr=0.001, label=""):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    start = time.time()
    for epoch in range(num_epochs):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = evaluate(model, test_loader, criterion)
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['test_loss'].append(v_loss)
        history['test_acc'].append(v_acc)
        print(f"[{label}] Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train Acc: {t_acc:.1f}%  Test Acc: {v_acc:.1f}%")
    total_time = time.time() - start
    print(f"[{label}] Total training time: {total_time:.1f}s")
    return history, total_time


# ── Count parameters ──────────────────────────────────────────
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Train both models ──────────────────────────────────────────
NUM_EPOCHS = 15

print("\n" + "="*55)
print("TRAINING: Improved CNN")
print("="*55)
cnn_model = ImprovedCNN().to(device)
cnn_history, cnn_time = run_training(cnn_model, NUM_EPOCHS, label="CNN")

print("\n" + "="*55)
print("TRAINING: MLP")
print("="*55)
mlp_model = MLP().to(device)
mlp_history, mlp_time = run_training(mlp_model, NUM_EPOCHS, label="MLP")


# ── Stats comparison ──────────────────────────────────────────
cnn_params = count_params(cnn_model)
mlp_params = count_params(mlp_model)
cnn_test_acc = cnn_history['test_acc'][-1]
mlp_test_acc = mlp_history['test_acc'][-1]

print("\n" + "="*55)
print("COMPARISON SUMMARY")
print("="*55)
print(f"{'Metric':<30} {'CNN':>10} {'MLP':>10}")
print("-"*55)
print(f"{'Parameters':<30} {cnn_params:>10,} {mlp_params:>10,}")
print(f"{'Test Accuracy (%)':<30} {cnn_test_acc:>10.1f} {mlp_test_acc:>10.1f}")
print(f"{'Training Time (s)':<30} {cnn_time:>10.1f} {mlp_time:>10.1f}")
print("="*55)


# ── Plot 1: Accuracy Comparison ────────────────────────────────
epochs = range(1, NUM_EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs, cnn_history['train_acc'], 'b-',  label='CNN Train')
axes[0].plot(epochs, cnn_history['test_acc'],  'b--', label='CNN Test')
axes[0].plot(epochs, mlp_history['train_acc'], 'r-',  label='MLP Train')
axes[0].plot(epochs, mlp_history['test_acc'],  'r--', label='MLP Test')
axes[0].set_title('Accuracy: CNN vs MLP', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy (%)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs, cnn_history['train_loss'], 'b-',  label='CNN Train Loss')
axes[1].plot(epochs, mlp_history['train_loss'], 'r-',  label='MLP Train Loss')
axes[1].set_title('Loss: CNN vs MLP', fontsize=14)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.suptitle('Task 2: CNN vs MLP Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task2_cnn_vs_mlp.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: task2_cnn_vs_mlp.png")


# ── Plot 2: Weight Visualization ───────────────────────────────
# CNN: visualise conv1 filters (what the CNN is looking for)
fig, axes = plt.subplots(4, 8, figsize=(14, 7))
conv1_weights = cnn_model.conv1.weight.data.cpu()  # shape: [32, 3, 3, 3]

for i, ax in enumerate(axes.flatten()):
    if i < 32:
        # Normalise filter to [0,1] for display
        w = conv1_weights[i]
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        w = w.permute(1, 2, 0).numpy()
        ax.imshow(w)
        ax.axis('off')
        ax.set_title(f'F{i}', fontsize=6)

plt.suptitle('CNN conv1 Filters (32 learned spatial detectors)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('task2_cnn_filters.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: task2_cnn_filters.png")

# MLP: visualise fc1 weights as a heatmap
fig, ax = plt.subplots(figsize=(12, 5))
fc1_weights = mlp_model.fc1.weight.data.cpu().numpy()  # shape: [1024, 3072]
im = ax.imshow(fc1_weights[:50, :200], aspect='auto', cmap='RdBu_r')
plt.colorbar(im, ax=ax)
ax.set_title('MLP fc1 Weights (first 50 neurons × first 200 inputs)\n'
             'No spatial structure visible — MLP treats all pixels equally', fontsize=12)
ax.set_xlabel('Input pixel index')
ax.set_ylabel('Neuron index')
plt.tight_layout()
plt.savefig('task2_mlp_weights.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: task2_mlp_weights.png")

print("\nDone! All Task 2 figures saved.")
