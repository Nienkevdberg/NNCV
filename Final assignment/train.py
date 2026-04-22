import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode,
    RandomHorizontalFlip,
)

from model import Model



id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])


train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)


def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, h, w = prediction.shape
    out = torch.zeros((batch, 3, h, w), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            out[:, i][mask] = color[i]

    return out


def compute_iou(pred, target, num_classes=19, ignore_index=255):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            continue

        ious.append((intersection / union).item())

    return sum(ious) / len(ious) if len(ious) > 0 else 0.0


def compute_dice(pred, target, num_classes=19, ignore_index=255):
    dices = []
    pred = pred.view(-1)
    target = target.view(-1)

    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds & target_inds).sum().float()
        total = pred_inds.sum().float() + target_inds.sum().float()

        if total == 0:
            continue

        dices.append((2 * intersection / total).item())

    return sum(dices) / len(dices) if len(dices) > 0 else 0.0


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)

        targets = targets.clone()
        valid = targets != self.ignore_index
        targets[~valid] = 0

        onehot = torch.zeros_like(probs)
        onehot.scatter_(1, targets.unsqueeze(1), 1)
        onehot = onehot * valid.unsqueeze(1)

        dims = (0, 2, 3)
        inter = torch.sum(probs * onehot, dims)
        union = torch.sum(probs + onehot, dims)

        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def main(args):
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.experiment_id,
        config=vars(args),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_transform = Compose([
        ToImage(),
        Resize((384, 768)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.485, 0.456, 0.406),
                  (0.229, 0.224, 0.225)),
    ])

    target_transform = Compose([
        ToImage(),
        Resize((384, 768), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),
    ])

    train_dataset = Cityscapes(
        args.data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )

    val_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    model = Model(n_classes=19).to(device)

    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    dice_loss_fn = DiceLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_iou = 0
    best_path = None

    for epoch in range(args.epochs):

        model.train()

        for i, (images, labels) in enumerate(train_loader):

            labels = convert_to_train_id(labels)

            images = images.to(device)
            labels = labels.to(device).squeeze(1)

            #if torch.rand(1) < 0.5:
            #    images = torch.flip(images, dims=[3])
            #    labels = torch.flip(labels, dims=[2])

            optimizer.zero_grad()

            outputs = model(images)

            loss = ce_loss(outputs, labels) + 1.0 * dice_loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            wandb.log({"train_loss": loss.item()})

        model.eval()

        losses, ious, dices = [], [], []

        with torch.no_grad():
            for images, labels in val_loader:

                labels = convert_to_train_id(labels)

                images = images.to(device)
                labels = labels.to(device).squeeze(1)

                outputs = model(images)

                loss = ce_loss(outputs, labels) + 0.5 * dice_loss_fn(outputs, labels)

                preds = torch.argmax(outputs, dim=1)

                losses.append(loss.item())
                ious.append(compute_iou(preds, labels))
                dices.append(compute_dice(preds, labels))

        mean_iou = sum(ious) / len(ious)
        mean_dice = sum(dices) / len(dices)

        wandb.log({
            "val_loss": sum(losses) / len(losses),
            "val_iou": mean_iou,
            "val_dice": mean_dice
        })

        if mean_iou > best_iou:
            best_iou = mean_iou

            if best_path is not None and os.path.exists(best_path):
                    os.remove(best_path)

            best_path = os.path.join(
                "checkpoints",
                f"best_iou_{epoch}.pt"
            )

            torch.save(model.state_dict(), best_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", type=str, default="exp")

    args = parser.parse_args()
    main(args)