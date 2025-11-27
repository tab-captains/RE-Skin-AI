# acne/train_resnet_acne.py
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import AcneDataset
from models.resnet50_acne import build_resnet50_acne


DATA_ROOT = "data/acne"   # 서버 기준 경로
BATCH_SIZE = 32
NUM_EPOCHS = 3
LR = 1e-4
FREEZE_BACKBONE = False
SAVE_PATH = "acne_resnet50_best.pth"
VAL_SPLIT_NAME = "valid"  # 폴더 이름이 val이면 "val"로 바꾸기


def get_dataloaders():
    data_root = DATA_ROOT

    # train용에는 augmentation 추가
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = AcneDataset(
        root_dir=data_root, split="train", transform=train_transform
    )
    val_dataset = AcneDataset(
        root_dir=data_root, split=VAL_SPLIT_NAME, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    class_names = list(train_dataset.class_to_idx.keys())

    return train_loader, val_loader, class_names


def train_one_epoch(model, criterion, optimizer, dataloader, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_corrects / total


@torch.no_grad()
def evaluate(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_corrects / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, class_names = get_dataloaders()
    num_classes = len(class_names)
    print("Classes:", class_names)

    model = build_resnet50_acne(
        num_classes=num_classes, freeze_backbone=FREEZE_BACKBONE
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, train_loader, device
        )
        val_loss, val_acc = evaluate(
            model, criterion, val_loader, device
        )
        scheduler.step()

        elapsed = time.time() - start
        print(
            f"[Epoch {epoch+1}/{NUM_EPOCHS}] {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "class_names": class_names,
            }
            torch.save(best_state, SAVE_PATH)
            print(f"  ✓ Best model updated! (val_acc={val_acc:.4f})")

    print("Training finished. Best Val Acc:", best_acc)


if __name__ == "__main__":
    main()
