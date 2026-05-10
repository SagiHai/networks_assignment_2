import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Helper: tensor → displayable image ───────────────────────
def tensor_to_img(t):
    img = t.cpu().numpy().transpose(1, 2, 0)
    img = img * 0.5 + 0.5   # undo normalisation
    return np.clip(img, 0, 1)


# ── Load ONE batch for visualisation (no augmentation yet) ───
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
vis_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                transform=base_transform)
vis_loader = DataLoader(vis_dataset, batch_size=10, shuffle=True)
sample_images, sample_labels = next(iter(vis_loader))
classes = vis_dataset.classes


# ════════════════════════════════════════════════════════════
# PLOT 1: Show each augmentation on the same sample images
# ════════════════════════════════════════════════════════════
aug_transforms = {
    'Original':            transforms.Compose([transforms.ToTensor()]),
    'Random Crop':         transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.ToTensor()]),
    'Horizontal Flip':     transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),
                                                transforms.ToTensor()]),
    'Rotation (±15°)':    transforms.Compose([transforms.RandomRotation(15),
                                                transforms.ToTensor()]),
    'Zoom / Resize':       transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.7, 1.0)),
                                                transforms.ToTensor()]),
}

# Get raw PIL images for augmentation demo
raw_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
n_show = 5
fig, axes = plt.subplots(len(aug_transforms), n_show, figsize=(14, 10))

for row_i, (aug_name, aug_tf) in enumerate(aug_transforms.items()):
    for col_i in range(n_show):
        pil_img, lbl = raw_dataset[col_i]
        tensor_img = aug_tf(pil_img)
        img_np = tensor_img.numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)
        axes[row_i, col_i].imshow(img_np)
        axes[row_i, col_i].axis('off')
        if col_i == 0:
            axes[row_i, col_i].set_ylabel(aug_name, fontsize=11, rotation=0,
                                           ha='right', va='center', labelpad=60)

plt.suptitle('Task 4: Effect of Each Augmentation', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('task4_augmentations_demo.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: task4_augmentations_demo.png")


# ════════════════════════════════════════════════════════════
# TRAIN CNN: without vs with augmentation
# ════════════════════════════════════════════════════════════

# No augmentation
no_aug_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# With augmentation (all combined)
aug_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Datasets
train_no_aug = datasets.CIFAR10(root='./data', train=True,  download=True, transform=no_aug_transform)
train_aug    = datasets.CIFAR10(root='./data', train=True,  download=True, transform=aug_transform)
test_ds      = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

loader_no_aug = DataLoader(train_no_aug, batch_size=128, shuffle=True)
loader_aug    = DataLoader(train_aug,    batch_size=128, shuffle=True)
loader_test   = DataLoader(test_ds,      batch_size=128, shuffle=False)


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
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / len(loader), 100 * correct / total, all_preds, all_labels


def run_training(train_loader, label, num_epochs=15):
    model = ImprovedCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(num_epochs):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        _, v_acc, preds, labels = evaluate(model, loader_test, criterion)
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['test_acc'].append(v_acc)
        print(f"[{label}] Ep {epoch+1:02d} | Train {t_acc:.1f}% | Test {v_acc:.1f}%")

    _, final_acc, final_preds, final_labels = evaluate(model, loader_test, criterion)
    return history, final_preds, final_labels


NUM_EPOCHS = 15

print("\n" + "="*55)
print("TRAINING: No Augmentation")
print("="*55)
history_no_aug, preds_no_aug, labels_no_aug = run_training(loader_no_aug, "No Aug", NUM_EPOCHS)

print("\n" + "="*55)
print("TRAINING: With Augmentation")
print("="*55)
history_aug, preds_aug, labels_aug = run_training(loader_aug, "Aug", NUM_EPOCHS)


# ── Plot 2: Test accuracy over epochs ─────────────────────────
epochs = range(1, NUM_EPOCHS + 1)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, history_no_aug['test_acc'], 'r-',  label='No Augmentation')
ax.plot(epochs, history_aug['test_acc'],    'b-',  label='With Augmentation')
ax.set_title('Test Accuracy: With vs Without Augmentation', fontsize=14)
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy (%)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('task4_accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: task4_accuracy_comparison.png")


# ── Plot 3: Confusion matrices ────────────────────────────────
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("seaborn not found, using basic confusion matrix")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, preds, labels, title in [
    (axes[0], preds_no_aug, labels_no_aug, 'No Augmentation'),
    (axes[1], preds_aug,    labels_aug,    'With Augmentation'),
]:
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if HAS_SEABORN:
        sns.heatmap(cm_norm, annot=True, fmt='.2f', ax=ax,
                    xticklabels=classes, yticklabels=classes,
                    cmap='Blues', vmin=0, vmax=1)
    else:
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(10)); ax.set_xticklabels(classes, rotation=45)
        ax.set_yticks(range(10)); ax.set_yticklabels(classes)
    ax.set_title(f'Confusion Matrix: {title}', fontsize=13)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.suptitle('Task 4: Confusion Matrix Comparison', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('task4_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: task4_confusion_matrices.png")

print(f"\nFinal Test Accuracy without augmentation: {history_no_aug['test_acc'][-1]:.1f}%")
print(f"Final Test Accuracy with augmentation:    {history_aug['test_acc'][-1]:.1f}%")
print("\nDone! All Task 4 figures saved.")
