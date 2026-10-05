"""
Training Pipeline for Color-Invariant Saree Recognition.
Trains metric backbone + ArcFace head with Mixed Precision (AMP) and Cosine Annealing.
"""
import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from config import Config
from dataset import (
    SareeDataset,
    get_color_invariant_transforms,
    load_saree_data_from_dir,
    split_gallery_query,
    generate_mock_saree_dataset,
)
from models import build_model
from losses import ArcFaceLoss
from evaluate import run_full_evaluation


def train_one_epoch(
    model: nn.Module,
    arcface_head: ArcFaceLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """Trains for a single epoch and returns mean training loss."""
    model.train()
    arcface_head.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                embeddings = model(images, normalize=True)
                loss = arcface_head(embeddings, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            embeddings = model(images, normalize=True)
            loss = arcface_head(embeddings, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, num_batches)


def run_training(cfg: Config, use_mock_if_missing: bool = True):
    """Full training pipeline with validation tracking and best checkpoint preservation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" [INIT] Saree Design Recognition Training")
    print(f"   Device: {device} | Backbone: {cfg.backbone} | Embedding Dim: {cfg.embedding_dim}")
    print(f"=======================================================\n")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Loading
    if not cfg.data_dir.exists() or not any(cfg.data_dir.iterdir()):
        if use_mock_if_missing:
            print(f" [NOTICE] Data directory '{cfg.data_dir}' not found.")
            print(f" [SYNTHESIS] Generating synthetic textile mock dataset at '{cfg.data_dir}'...")
            generate_mock_saree_dataset(cfg.data_dir, num_designs=8, colorways_per_design=6)
        else:
            raise FileNotFoundError(f"Dataset directory '{cfg.data_dir}' does not exist.")

    samples, class_to_idx, idx_to_class = load_saree_data_from_dir(cfg.data_dir)
    num_classes = len(class_to_idx)
    print(f" [DATA] Loaded {len(samples)} images across {num_classes} unique saree design classes.")

    # Split into Gallery and Query sets for realistic validation
    gallery_samples, query_samples = split_gallery_query(
        samples, query_ratio=cfg.gallery_query_split_ratio, seed=cfg.seed
    )
    print(f"   - Gallery (Reference) Samples: {len(gallery_samples)}")
    print(f"   - Query (Test) Samples:        {len(query_samples)}")

    # Data Loaders
    train_transform = get_color_invariant_transforms(
        image_size=cfg.image_size,
        is_training=True,
        grayscale_prob=cfg.grayscale_prob,
        solarize_prob=cfg.solarize_prob,
        hue_jitter=cfg.hue_jitter,
        sat_jitter=cfg.sat_jitter,
    )
    train_dataset = SareeDataset(samples, transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # 2. Build Model & Loss
    model = build_model(
        backbone_name=cfg.backbone,
        embedding_dim=cfg.embedding_dim,
        pretrained=cfg.pretrained,
        dropout=cfg.dropout,
        device=device,
    )
    arcface_head = ArcFaceLoss(
        in_features=cfg.embedding_dim,
        num_classes=num_classes,
        scale=cfg.arcface_scale,
        margin=cfg.arcface_margin,
    ).to(device)

    # Optimizer: Jointly optimizes backbone and ArcFace classification weights
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": cfg.learning_rate},
            {"params": arcface_head.parameters(), "lr": cfg.learning_rate * 2.0},
        ],
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )

    scaler = torch.amp.GradScaler("cuda") if (cfg.use_amp and device.type == "cuda") else None

    # 3. Training & Validation Loop
    best_rank1 = 0.0
    best_checkpoint_path = cfg.output_dir / "best_saree_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(
            model=model,
            arcface_head=arcface_head,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=cfg.use_amp,
        )
        scheduler.step()
        elapsed = time.time() - t0

        # Run Evaluation every 3 epochs or on final epoch
        if epoch % 3 == 0 or epoch == cfg.epochs:
            eval_metrics = run_full_evaluation(
                model=model,
                gallery_samples=gallery_samples,
                query_samples=query_samples,
                device=device,
                image_size=cfg.image_size,
                batch_size=cfg.batch_size,
            )
            r1 = eval_metrics.get("Rank-1", 0.0)
            map_score = eval_metrics.get("mAP", 0.0)
            auc_score = eval_metrics.get("ROC-AUC", 0.0)
            eer_score = eval_metrics.get("EER", 0.0)

            print(
                f"Epoch [{epoch:02d}/{cfg.epochs:02d}] ({elapsed:.1f}s) | "
                f"Loss: {loss:.4f} | "
                f"Rank-1: {r1:5.1f}% | "
                f"mAP: {map_score:5.1f}% | "
                f"AUC: {auc_score:5.1f}% | "
                f"EER: {eer_score:4.1f}%"
            )

            # Save best checkpoint based on Rank-1 Identification Accuracy
            if r1 >= best_rank1:
                best_rank1 = r1
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "arcface_state": arcface_head.state_dict(),
                        "config": cfg,
                        "class_to_idx": class_to_idx,
                        "idx_to_class": idx_to_class,
                        "metrics": eval_metrics,
                    },
                    best_checkpoint_path,
                )
        else:
            print(f"Epoch [{epoch:02d}/{cfg.epochs:02d}] ({elapsed:.1f}s) | Loss: {loss:.4f}")

    print(f"\n[DONE] Training complete! Best Rank-1: {best_rank1:.2f}%")
    print(f"[CHECKPOINT] Checkpoint saved to: {best_checkpoint_path.resolve()}\n")
    return best_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Color-Invariant Saree Recognition Model")
    parser.add_argument("--backbone", type=str, default="convnext_tiny", help="Backbone model")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="./data/saree_corpus", help="Path to saree dataset")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic mock data if dataset missing")
    args = parser.parse_args()

    cfg = Config()
    cfg.backbone = args.backbone
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.data_dir = Path(args.data_dir)

    run_training(cfg, use_mock_if_missing=True)
