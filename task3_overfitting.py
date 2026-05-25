import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Data: split train → train + validation ────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
full_train = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
test_ds    = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 80% train, 20% validation
n_train = int(0.8 * len(full_train))
n_val   = len(full_train) - n_train
train_ds, val_ds = random_split(full_train, [n_train, n_val],
                                 generator=torch.Generator().manual_seed(seed))

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)
print(f"Train: {n_train} | Val: {n_val} | Test: {len(test_ds)}")


# ════════════════════════════════════════════════════════════
# BIG CNN (more filters = more parameters = easy to overfit)
# ════════════════════════════════════════════════════════════
class BigCNN(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()
        self.use_dropout = use_dropout
        self.conv1 = nn.Conv2d(3,   64,  3, padding=1)
        self.conv2 = nn.Conv2d(64,  128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.fc1   = nn.Linear(512 * 2 * 2, 1024)
        self.fc2   = nn.Linear(1024, 512)
        self.fc3   = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)   # drops 50% of neurons randomly

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x)); x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x)); x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)   # applied only when use_dropout=True
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
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


def run_training(model, num_epochs, lr=0.001, weight_decay=0.0,
                 early_stopping_patience=None, label=""):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    for epoch in range(num_epochs):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = evaluate(model, val_loader, criterion)

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        print(f"[{label}] Ep {epoch+1:02d} | Train {t_acc:.1f}% | Val {v_acc:.1f}%")

        # Early stopping logic
        if early_stopping_patience is not None:
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"[{label}] Early stop at epoch {epoch+1}. "
                          f"Best was epoch {best_epoch}.")
                    model.load_state_dict(best_model_state)
                    break

    return history, best_epoch if early_stopping_patience else num_epochs


# ════════════════════════════════════════════════════════════
# EXPERIMENT 1: Big model, no regularisation (causes overfitting)
# ════════════════════════════════════════════════════════════
NUM_EPOCHS = 30
print("\n" + "="*55)
print("EXPERIMENT 1: Big model, no regularisation")
print("="*55)
base_model = BigCNN(use_dropout=False).to(device)
base_history, _ = run_training(base_model, NUM_EPOCHS, label="No Reg")

# Find overfitting start: where val_loss starts consistently increasing
val_losses = base_history['val_loss']
overfit_epoch = None
for i in range(2, len(val_losses)):
    if val_losses[i] > val_losses[i-1] > val_losses[i-2]:
        overfit_epoch = i - 1
        break
if overfit_epoch:
    print(f"\nOverfitting begins around epoch {overfit_epoch}")


# ════════════════════════════════════════════════════════════
# EXPERIMENT 2: Dropout
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("EXPERIMENT 2: Dropout (p=0.5)")
print("="*55)
dropout_model = BigCNN(use_dropout=True).to(device)
dropout_history, _ = run_training(dropout_model, NUM_EPOCHS, label="Dropout")


# ════════════════════════════════════════════════════════════
# EXPERIMENT 3: Early Stopping (patience=5)
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("EXPERIMENT 3: Early Stopping (patience=5)")
print("="*55)
es_model = BigCNN(use_dropout=False).to(device)
es_history, stopped_epoch = run_training(es_model, NUM_EPOCHS,
                                          early_stopping_patience=5, label="EarlyStop")


# ── Plotting ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
epochs_base    = range(1, len(base_history['train_loss'])    + 1)
epochs_dropout = range(1, len(dropout_history['train_loss']) + 1)
epochs_es      = range(1, len(es_history['train_loss'])      + 1)

# No regularisation
ax = axes[0, 0]
ax.plot(epochs_base, base_history['train_loss'], 'b-',  label='Train Loss')
ax.plot(epochs_base, base_history['val_loss'],   'r-',  label='Val Loss')
if overfit_epoch:
    ax.axvline(x=overfit_epoch, color='orange', linestyle='--',
               label=f'Overfit starts ~ep{overfit_epoch}')
ax.set_title('No Regularisation (overfitting)', fontsize=12)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(); ax.grid(True)

ax2 = axes[0, 1]
ax2.plot(epochs_base, base_history['train_acc'], 'b-',  label='Train Acc')
ax2.plot(epochs_base, base_history['val_acc'],   'r-',  label='Val Acc')
if overfit_epoch:
    ax2.axvline(x=overfit_epoch, color='orange', linestyle='--',
                label=f'Overfit starts ~ep{overfit_epoch}')
ax2.set_title('No Regularisation: Accuracy', fontsize=12)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
ax2.legend(); ax2.grid(True)

# Dropout
ax3 = axes[1, 0]
ax3.plot(epochs_dropout, dropout_history['train_loss'], 'b-',  label='Train Loss (Dropout)')
ax3.plot(epochs_dropout, dropout_history['val_loss'],   'r-',  label='Val Loss (Dropout)')
ax3.plot(epochs_base,    base_history['train_loss'],    'b:',  alpha=0.4, label='Train Loss (No Reg)')
ax3.plot(epochs_base,    base_history['val_loss'],      'r:',  alpha=0.4, label='Val Loss (No Reg)')
ax3.set_title('Dropout vs No Regularisation', fontsize=12)
ax3.set_xlabel('Epoch'); ax3.set_ylabel('Loss')
ax3.legend(); ax3.grid(True)

# Early stopping
ax4 = axes[1, 1]
ax4.plot(epochs_es, es_history['train_loss'], 'b-',  label='Train Loss (Early Stop)')
ax4.plot(epochs_es, es_history['val_loss'],   'r-',  label='Val Loss (Early Stop)')
ax4.axvline(x=stopped_epoch, color='green', linestyle='--', label=f'Stopped at ep{stopped_epoch}')
ax4.plot(epochs_base, base_history['train_loss'], 'b:', alpha=0.4, label='Train Loss (No Reg)')
ax4.plot(epochs_base, base_history['val_loss'],   'r:', alpha=0.4, label='Val Loss (No Reg)')
ax4.set_title('Early Stopping vs No Regularisation', fontsize=12)
ax4.set_xlabel('Epoch'); ax4.set_ylabel('Loss')
ax4.legend(); ax4.grid(True)

plt.suptitle('Task 3: Overfitting Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task3_overfitting.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: task3_overfitting.png")

print("\nDone! All Task 3 figures saved.")

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1   = nn.Linear(32 * 8 * 8, 256)
        self.fc2   = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

print("\n" + "="*55)
print("TRAINING: SmallCNN (should NOT overfit)")
print("="*55)
small_model = SmallCNN().to(device)
small_history, _ = run_training(small_model, NUM_EPOCHS, label="SmallCNN")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs_small = range(1, len(small_history['train_loss']) + 1)
epochs_big   = range(1, len(base_history['train_loss']) + 1)

axes[0].plot(epochs_big,   base_history['train_loss'],  'b-',  label='BigCNN Train')
axes[0].plot(epochs_big,   base_history['val_loss'],    'b--', label='BigCNN Val')
axes[0].plot(epochs_small, small_history['train_loss'], 'r-',  label='SmallCNN Train')
axes[0].plot(epochs_small, small_history['val_loss'],   'r--', label='SmallCNN Val')
axes[0].set_title('Small vs Big Model: Loss')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(True)

axes[1].plot(epochs_big,   base_history['train_acc'],  'b-',  label='BigCNN Train')
axes[1].plot(epochs_big,   base_history['val_acc'],    'b--', label='BigCNN Val')
axes[1].plot(epochs_small, small_history['train_acc'], 'r-',  label='SmallCNN Train')
axes[1].plot(epochs_small, small_history['val_acc'],   'r--', label='SmallCNN Val')
axes[1].set_title('Small vs Big Model: Accuracy')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].legend(); axes[1].grid(True)

plt.suptitle('Task 3: Small vs Big Model Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('task3_small_vs_big.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: task3_small_vs_big.png")
