import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random

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
train_ds = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
test_ds  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)
classes = train_ds.classes


# ── Baseline CNN (from Task 1) ────────────────────────────────
class CNN(nn.Module):
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


# ── Baseline MLP (from Task 2) ────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ════════════════════════════════════════════════════════════
# HYBRID: CNN Extractor + MLP Classifier
#
# Architecture:
#   Input (3×32×32)
#     ↓ Conv1 + BN + ReLU + MaxPool → 32×16×16
#     ↓ Conv2 + ReLU + MaxPool      → 64×8×8
#     ↓ Conv3 + ReLU + MaxPool      → 128×4×4
#     ↓ Conv4 + ReLU + MaxPool      → 256×2×2
#     ↓ Flatten                     → 1024-d vector
#     ↓ [MLP HEAD]
#     ↓ FC(1024→512) + ReLU + Dropout(0.3)
#     ↓ FC(512→256)  + ReLU + Dropout(0.3)
#     ↓ FC(256→10)   → class scores
# ════════════════════════════════════════════════════════════
class HybridCNNMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # ── CNN Feature Extractor ──────────────────────────
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # After 4 max-pools: 32 → 2, so output is 256×2×2 = 1024

        # ── MLP Classifier Head ────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)       # CNN feature extraction
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)     # MLP classification
        return x


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


def run_training(model, num_epochs, label=""):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    for epoch in range(num_epochs):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        _, v_acc = evaluate(model, test_loader, criterion)
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['test_acc'].append(v_acc)
        print(f"[{label}] Ep {epoch+1:02d} | Train {t_acc:.1f}% | Test {v_acc:.1f}%")
    return history


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Train all three ───────────────────────────────────────────
NUM_EPOCHS = 15

print("\n" + "="*55)
print("TRAINING: CNN baseline")
print("="*55)
cnn_model = CNN().to(device)
cnn_history = run_training(cnn_model, NUM_EPOCHS, "CNN")

print("\n" + "="*55)
print("TRAINING: MLP baseline")
print("="*55)
mlp_model = MLP().to(device)
mlp_history = run_training(mlp_model, NUM_EPOCHS, "MLP")

print("\n" + "="*55)
print("TRAINING: Hybrid CNN+MLP")
print("="*55)
hybrid_model = HybridCNNMLP().to(device)
hybrid_history = run_training(hybrid_model, NUM_EPOCHS, "Hybrid")


# ── Parameter counts ──────────────────────────────────────────
print("\n" + "="*55)
print("PARAMETER COUNT COMPARISON")
print("="*55)
print(f"CNN:    {count_params(cnn_model):>10,}")
print(f"MLP:    {count_params(mlp_model):>10,}")
print(f"Hybrid: {count_params(hybrid_model):>10,}")
print("="*55)


# ── Plot 1: Architecture diagram ──────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

def draw_block(ax, x, y, w, h, label, color, fontsize=9):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# CNN blocks (left side)
cnn_blocks = [
    (2, 8.5, 'Input\n3×32×32', '#AED6F1'),
    (2, 7.0, 'Conv1+BN+ReLU\n+MaxPool→32×16×16', '#2E86C1'),
    (2, 5.5, 'Conv2+ReLU\n+MaxPool→64×8×8', '#2E86C1'),
    (2, 4.0, 'Conv3+ReLU\n+MaxPool→128×4×4', '#2E86C1'),
    (2, 2.5, 'Conv4+ReLU\n+MaxPool→256×2×2', '#2E86C1'),
    (2, 1.2, 'Flatten → 1024-d', '#85C1E9'),
]
for x, y, label, color in cnn_blocks:
    draw_block(ax, x, y, 2.8, 0.8, label, color, fontsize=8)

for i in range(len(cnn_blocks) - 1):
    x, y1, *_ = cnn_blocks[i]
    x, y2, *_ = cnn_blocks[i+1]
    draw_arrow(ax, x, y1 - 0.4, x, y2 + 0.4)

# MLP head (right side)
mlp_blocks = [
    (7, 8.5, 'FC(1024→512)\n+ReLU+Dropout(0.3)', '#A9DFBF'),
    (7, 6.8, 'FC(512→256)\n+ReLU+Dropout(0.3)', '#27AE60'),
    (7, 5.1, 'FC(256→10)\nOutput logits', '#1E8449'),
    (7, 3.5, 'Softmax →\nClass Prediction', '#D5F5E3'),
]
for x, y, label, color in mlp_blocks:
    draw_block(ax, x, y, 2.8, 0.9, label, color, fontsize=8)

for i in range(len(mlp_blocks) - 1):
    x, y1, *_ = mlp_blocks[i]
    x, y2, *_ = mlp_blocks[i+1]
    draw_arrow(ax, x, y1 - 0.45, x, y2 + 0.45)

# Arrow connecting CNN flatten to MLP
draw_arrow(ax, 3.4, 1.2, 5.6, 8.5)
ax.text(4.7, 5.1, 'Feature\nVector', ha='center', fontsize=9, color='gray')

# Labels
ax.text(2, 9.5, 'CNN Feature Extractor', ha='center', fontsize=12,
        fontweight='bold', color='#1A5276')
ax.text(7, 9.5, 'MLP Classifier Head', ha='center', fontsize=12,
        fontweight='bold', color='#1E8449')

ax.set_title('Task 5: Hybrid Architecture — CNN Extractor + MLP Classifier',
             fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('task5_architecture.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: task5_architecture.png")


# ── Plot 2: Three-way accuracy comparison ────────────────────
epochs = range(1, NUM_EPOCHS + 1)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, cnn_history['test_acc'],    'b-',  label='CNN')
ax.plot(epochs, mlp_history['test_acc'],    'r-',  label='MLP')
ax.plot(epochs, hybrid_history['test_acc'], 'g-',  linewidth=2.5, label='Hybrid (CNN+MLP)')
ax.set_title('Task 5: Test Accuracy Comparison', fontsize=14)
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy (%)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('task5_accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: task5_accuracy_comparison.png")

print(f"\nFinal CNN Test Acc:    {cnn_history['test_acc'][-1]:.1f}%")
print(f"Final MLP Test Acc:    {mlp_history['test_acc'][-1]:.1f}%")
print(f"Final Hybrid Test Acc: {hybrid_history['test_acc'][-1]:.1f}%")
print("\nDone! All Task 5 figures saved.")
