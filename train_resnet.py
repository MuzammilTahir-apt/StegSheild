import os
import copy
import torch
import numpy as np

from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# ============================================================
# STEGSHIELD - FRESH RESNET18 TRAINING
# ============================================================

print("\n" + "=" * 60)
print("        STEGSHIELD RESNET18 TRAINING")
print("=" * 60)

# ------------------------------------------------------------
# 1. DEVICE
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nDevice: {device}")

if device.type == "cpu":
    print("CPU training enabled.")


# ------------------------------------------------------------
# 2. DATASET PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "data",
    "Stego-pvd-dataset"
)

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_DIR = os.path.join(BASE_DIR, "models")
GRAPH_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


# ------------------------------------------------------------
# 3. IMAGE TRANSFORMS
# ------------------------------------------------------------

# Training augmentation:
#
# These transformations create slightly different versions
# of training images so the model does not simply memorize
# the exact training images.
#
# Validation and testing DO NOT use augmentation.

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------------------------------------------------
# 4. LOAD DATASETS
# ------------------------------------------------------------

print("\nLoading datasets...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=val_test_transform
)


print(f"TRAIN: {len(train_dataset)} images")
print(f"VALIDATION: {len(val_dataset)} images")
print(f"TEST: {len(test_dataset)} images")

print("\nClass mapping:")
print(train_dataset.class_to_idx)


# ------------------------------------------------------------
# 5. DATA LOADERS
# ------------------------------------------------------------

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ------------------------------------------------------------
# 6. LOAD RESNET18
# ------------------------------------------------------------

print("\nLoading ResNet18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# ------------------------------------------------------------
# 7. REPLACE FINAL CLASSIFIER
# ------------------------------------------------------------

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model = model.to(device)


# ------------------------------------------------------------
# 8. LOSS FUNCTION
# ------------------------------------------------------------

criterion = nn.CrossEntropyLoss()


# ------------------------------------------------------------
# 9. OPTIMIZER
# ------------------------------------------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001,
    weight_decay=1e-4
)


# ------------------------------------------------------------
# 10. LEARNING RATE SCHEDULER
# ------------------------------------------------------------

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ------------------------------------------------------------
# 11. TRAINING SETTINGS
# ------------------------------------------------------------

NUM_EPOCHS = 10

best_val_accuracy = 0.0

best_model_weights = copy.deepcopy(
    model.state_dict()
)


train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []


# ------------------------------------------------------------
# 12. TRAINING LOOP
# ------------------------------------------------------------

for epoch in range(NUM_EPOCHS):

    print("\n" + "-" * 60)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 60)

    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        100.0 * correct / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += (
                loss.item() * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_running_loss / val_total
    )

    val_accuracy = (
        100.0 * val_correct / val_total
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)


    # ========================================================
    # LEARNING RATE UPDATE
    # ========================================================

    scheduler.step(val_accuracy)


    current_lr = optimizer.param_groups[0]["lr"]


    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    print(
        f"Learning Rate: {current_lr:.6f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        print("Best model updated.")


# ------------------------------------------------------------
# 13. LOAD BEST MODEL
# ------------------------------------------------------------

print("\nLoading best validation model...")

model.load_state_dict(
    best_model_weights
)


# ------------------------------------------------------------
# 14. TEST MODEL
# ------------------------------------------------------------

print("\nRunning final test...")

model.eval()

all_predictions = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


test_accuracy = (
    100.0 * correct / total
)


# ------------------------------------------------------------
# 15. RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Final Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

print("=" * 60)


# ------------------------------------------------------------
# 16. CLASSIFICATION REPORT
# ------------------------------------------------------------

class_names = test_dataset.classes

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4
    )
)


# ------------------------------------------------------------
# 17. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")

print(cm)


# ------------------------------------------------------------
# 18. SAVE MODEL
# ------------------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "resnet18_stegshield_new.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_to_idx": train_dataset.class_to_idx,
        "classes": train_dataset.classes,
        "best_validation_accuracy":
            best_val_accuracy,
        "test_accuracy":
            test_accuracy
    },
    model_path
)


print("\nModel saved to:")

print(model_path)


# ------------------------------------------------------------
# 19. TRAINING LOSS GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, NUM_EPOCHS + 1),
    train_losses,
    label="Training Loss"
)

plt.plot(
    range(1, NUM_EPOCHS + 1),
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "StegShield - Training vs Validation Loss"
)

plt.legend()

plt.grid(True)

loss_graph = os.path.join(
    GRAPH_DIR,
    "loss_curve.png"
)

plt.savefig(
    loss_graph,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 20. TRAINING ACCURACY GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, NUM_EPOCHS + 1),
    train_accuracies,
    label="Training Accuracy"
)

plt.plot(
    range(1, NUM_EPOCHS + 1),
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "StegShield - Training vs Validation Accuracy"
)

plt.legend()

plt.grid(True)

accuracy_graph = os.path.join(
    GRAPH_DIR,
    "accuracy_curve.png"
)

plt.savefig(
    accuracy_graph,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print("\nGraphs saved:")

print(loss_graph)
print(accuracy_graph)

print("\nTraining completed successfully.")

print("=" * 60)