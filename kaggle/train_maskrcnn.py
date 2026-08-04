"""Train and run a license-compatible torchvision Mask R-CNN on MAGFiLO.

Designed for a Kaggle GPU notebook. The project code remains MIT licensed and
uses PyTorch/torchvision rather than Ultralytics. Run ``--help`` for modes.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from solarfil.coco_data import (
    greedy_scores_from_matrix,
    select_best_image_records,
    stable_stratified_split,
)
from solarfil.submission import encode_mask


def annotation_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    rles = mask_utils.frPyObjects(annotation["segmentation"], height, width)
    return mask_utils.decode(mask_utils.merge(rles)).astype(np.uint8)


class FilamentDataset(Dataset):
    def __init__(self, root: Path, filenames: list[str], coco: dict, augment: bool = False):
        self.image_dir = root / "train/train_images"
        self.filenames = filenames
        self.augment = augment
        self.images = select_best_image_records(coco)
        self.annotations: dict[int, list[dict]] = defaultdict(list)
        for annotation in coco["annotations"]:
            self.annotations[annotation["image_id"]].append(annotation)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        record = self.images[filename]
        image_np = np.asarray(Image.open(self.image_dir / filename).convert("RGB")).copy()
        masks_np = np.stack([
            annotation_mask(item, record["height"], record["width"])
            for item in self.annotations[record["id"]]
            if not item.get("iscrowd", 0)
        ])
        image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        masks = torch.from_numpy(masks_np).bool()

        if self.augment:
            rotation = random.randrange(4)
            image = torch.rot90(image, rotation, (1, 2))
            masks = torch.rot90(masks, rotation, (1, 2))
            if random.random() < 0.5:
                image, masks = image.flip(2), masks.flip(2)
            if random.random() < 0.5:
                image, masks = image.flip(1), masks.flip(1)

        boxes = []
        keep = []
        for mask_index, mask in enumerate(masks):
            ys, xs = torch.where(mask)
            if len(xs):
                boxes.append(torch.stack((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)))
                keep.append(mask_index)
        masks = masks[keep]
        target = {
            "boxes": torch.stack(boxes).float() if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "masks": masks.to(torch.uint8),
            # MAGFiLO COCO image IDs are strings; torchvision expects a tensor.
            "image_id": torch.tensor(index, dtype=torch.int64),
        }
        return image, target


def collate(batch):
    return tuple(zip(*batch))


def build_model(trainable_backbone_layers: int = 3, pretrained: bool = True):
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None,
        weights_backbone=None,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=800,
        max_size=1600,
        box_detections_per_img=32,
    )
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, 2)
    return model


@torch.inference_mode()
def validate(model, loader, device, score_threshold: float, mask_threshold: float, limit: int):
    model.eval()
    scores = []
    truth_count = prediction_count = 0
    for batch_index, (images, targets) in enumerate(loader):
        if batch_index >= limit:
            break
        outputs = model([image.to(device) for image in images])
        for output, target in zip(outputs, targets):
            selected = output["scores"] >= score_threshold
            predictions = output["masks"][selected, 0] >= mask_threshold
            truths = target["masks"].to(device).bool()
            if len(predictions) and len(truths):
                prediction_areas = predictions.flatten(1).sum(1)
                truth_areas = truths.flatten(1).sum(1)
                intersections = torch.stack([
                    (predictions & truth).flatten(1).sum(1) for truth in truths
                ], dim=1)
                denominators = prediction_areas[:, None] + truth_areas[None, :]
                matrix = (2 * intersections / denominators.clamp_min(1)).cpu().numpy()
                scores.extend(greedy_scores_from_matrix(matrix, len(truths)))
            else:
                scores.extend([0.0] * len(truths))
            prediction_count += len(predictions)
            truth_count += len(truths)
    return {
        "mean_matched_dice": float(np.mean(scores)) if scores else 0.0,
        "instances_predicted": prediction_count,
        "instances_truth": truth_count,
        "prediction_truth_ratio": prediction_count / truth_count if truth_count else 0.0,
    }


def train(args) -> None:
    annotation_path = args.root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = json.loads(annotation_path.read_text())
    records = select_best_image_records(coco)
    train_files, validation_files = stable_stratified_split(list(records), args.val_fraction, args.seed)
    train_data = FilamentDataset(args.root, train_files, coco, augment=True)
    validation_data = FilamentDataset(args.root, validation_files, coco)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=1, shuffle=False,
                                   num_workers=args.workers, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.trainable_backbone_layers, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -1.0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss_map = model(images, targets)
                loss = sum(loss_map.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        metrics = validate(model, validation_loader, device, args.score_threshold,
                           args.mask_threshold, args.validation_limit)
        print(json.dumps({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}))
        if metrics["mean_matched_dice"] > best_score:
            best_score = metrics["mean_matched_dice"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, args.checkpoint)
        scheduler.step()


@torch.inference_mode()
def predict(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.trainable_backbone_layers, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    files = sorted((args.root / "test/test_images").glob("*.jpeg"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for path in files:
            image = torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).permute(2, 0, 1)
            output = model([image.float().to(device) / 255.0])[0]
            selected = output["scores"].cpu() >= args.score_threshold
            masks = output["masks"].cpu()[selected, 0] >= args.mask_threshold
            areas = masks.flatten(1).sum(1)
            masks = masks[areas >= args.min_area]
            for index, mask in enumerate(masks, 1):
                writer.writerow([f"{path.stem}_{index}", encode_mask(mask.numpy())])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "predict"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("maskrcnn-best.pt"))
    parser.add_argument("--output", type=Path, default=Path("submission-maskrcnn.csv"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--validation-limit", type=int, default=64)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=24)
    parser.add_argument("--trainable-backbone-layers", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    train(arguments) if arguments.mode == "train" else predict(arguments)
