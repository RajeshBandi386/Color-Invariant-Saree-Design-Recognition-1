"""
Dataset module for Color-Invariant Saree Design Recognition.
Handles data loading, color-invariance augmentations, and mock dataset synthesis.
"""
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_color_invariant_transforms(
    image_size: int = 224,
    is_training: bool = True,
    grayscale_prob: float = 0.35,
    solarize_prob: float = 0.15,
    hue_jitter: float = 0.4,
    sat_jitter: float = 0.4,
    contrast_jitter: float = 0.3,
    brightness_jitter: float = 0.3,
) -> transforms.Compose:
    """
    Builds data transformations tailored specifically for textile motif recognition.
    Applies aggressive color perturbation during training to force the model to rely
    on surface geometry, edge frequencies, and texture rather than colorway hue.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            # --- Color Invariance Augmentations ---
            transforms.ColorJitter(
                brightness=brightness_jitter,
                contrast=contrast_jitter,
                saturation=sat_jitter,
                hue=hue_jitter,
            ),
            transforms.RandomGrayscale(p=grayscale_prob),
            transforms.RandomSolarize(threshold=128, p=solarize_prob),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


class SareeDataset(Dataset):
    """
    Standard PyTorch dataset for Saree designs.
    Expects samples as a list of (image_path, class_id).
    """
    def __init__(
        self,
        samples: List[Tuple[Union[str, Path], int]],
        transform: Optional[transforms.Compose] = None,
    ):
        self.samples = [(Path(p), int(lbl)) for p, lbl in samples]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(path)


def load_saree_data_from_dir(
    root_dir: Union[str, Path],
    min_samples_per_class: int = 2,
) -> Tuple[List[Tuple[Path, int]], Dict[str, int], Dict[int, str]]:
    """
    Loads dataset organized by folder:
    root_dir/
      ├── motif_peacock/
      │     ├── red.jpg
      │     └── blue.jpg
      ├── motif_temple_border/
      │     ├── green.jpg
      │     └── gold.jpg
    """
    root_dir = Path(root_dir)
    class_folders = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    
    class_to_idx = {}
    idx_to_class = {}
    samples = []
    
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    current_idx = 0
    
    for folder in class_folders:
        img_files = [f for f in folder.iterdir() if f.suffix.lower() in valid_exts]
        if len(img_files) >= min_samples_per_class:
            class_name = folder.name
            class_to_idx[class_name] = current_idx
            idx_to_class[current_idx] = class_name
            for img_p in img_files:
                samples.append((img_p, current_idx))
            current_idx += 1
            
    return samples, class_to_idx, idx_to_class


def split_gallery_query(
    samples: List[Tuple[Path, int]],
    query_ratio: float = 0.5,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Splits samples per motif into Gallery (reference designs) and Query (unseen colorways)
    to emulate realistic 1:N retrieval.
    """
    random.seed(seed)
    class_to_samples: Dict[int, List[Path]] = {}
    for p, lbl in samples:
        class_to_samples.setdefault(lbl, []).append(p)

    gallery_samples = []
    query_samples = []

    for lbl, paths in class_to_samples.items():
        shuffled = paths.copy()
        random.shuffle(shuffled)
        n_query = max(1, int(len(shuffled) * query_ratio))
        # ensure at least 1 in gallery if possible
        if len(shuffled) > 1 and n_query == len(shuffled):
            n_query = len(shuffled) - 1
            
        q_paths = shuffled[:n_query]
        g_paths = shuffled[n_query:]
        if not g_paths:  # fallback
            g_paths = [q_paths.pop()]

        for p in g_paths:
            gallery_samples.append((p, lbl))
        for p in q_paths:
            query_samples.append((p, lbl))

    return gallery_samples, query_samples


# =====================================================================
# Synthetic Saree Generator (Zero-setup testing & verification)
# =====================================================================
def generate_mock_saree_dataset(
    output_dir: Union[str, Path],
    num_designs: int = 8,
    colorways_per_design: int = 5,
    image_size: int = 256,
) -> Path:
    """
    Generates synthetic textile motif patterns in diverse color palettes.
    Useful for immediate unit testing, benchmark verification, and demonstration
    without requiring external multi-gigabyte dataset downloads.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    palettes = [
        # (bg_color, primary_motif_color, secondary_motif_color)
        ((180, 20, 40), (240, 215, 0), (255, 255, 255)),    # Crimson Red & Gold Zari
        ((10, 40, 150), (220, 200, 100), (0, 200, 255)),    # Royal Blue & Metallic Gold
        ((15, 120, 60), (255, 215, 0), (255, 120, 0)),      # Emerald Green & Orange Zari
        ((180, 20, 120), (255, 255, 255), (255, 215, 0)),   # Rani Pink & Silver
        ((220, 160, 10), (120, 20, 20), (255, 255, 255)),   # Mustard Yellow & Maroon
        ((30, 30, 30), (240, 200, 50), (200, 20, 50)),      # Midnight Black & Gold
        ((120, 30, 150), (255, 220, 100), (0, 255, 200)),   # Purple & Champagne
        ((200, 80, 20), (255, 255, 255), (255, 215, 0)),    # Rust Orange & Gold
    ]

    motif_types = [
        "peacock_butta",
        "temple_triangles",
        "ikat_diamonds",
        "paisley_kalka",
        "bandhani_dots",
        "floral_jaal",
        "geometric_chevron",
        "zari_grid_lattice",
    ]

    for d_idx in range(min(num_designs, len(motif_types))):
        motif_name = motif_types[d_idx]
        folder = output_dir / f"design_{d_idx+1:02d}_{motif_name}"
        folder.mkdir(parents=True, exist_ok=True)

        selected_palettes = random.sample(palettes, min(colorways_per_design, len(palettes)))

        for c_idx, (bg, c1, c2) in enumerate(selected_palettes):
            img = Image.new("RGB", (image_size, image_size), bg)
            draw = ImageDraw.Draw(img)

            if motif_name == "peacock_butta":
                # Concentric circles and curved feathers
                for cx in range(32, image_size, 64):
                    for cy in range(32, image_size, 64):
                        draw.ellipse([cx-18, cy-18, cx+18, cy+18], fill=c1, outline=c2, width=2)
                        draw.ellipse([cx-8, cy-8, cx+8, cy+8], fill=c2)

            elif motif_name == "temple_triangles":
                # Stepped zig-zag temple borders
                step = 32
                for y in range(0, image_size, step):
                    for x in range(0, image_size, step):
                        points = [(x, y + step), (x + step // 2, y), (x + step, y + step)]
                        draw.polygon(points, fill=c1, outline=c2)

            elif motif_name == "ikat_diamonds":
                # Diamond lattice pattern
                step = 40
                for y in range(0, image_size + step, step):
                    for x in range(0, image_size + step, step):
                        pts = [(x, y - 18), (x + 18, y), (x, y + 18), (x - 18, y)]
                        draw.polygon(pts, fill=c1, outline=c2)
                        draw.ellipse([x-4, y-4, x+4, y+4], fill=c2)

            elif motif_name == "paisley_kalka":
                # Teardrop paisley curve motifs
                for y in range(30, image_size, 60):
                    for x in range(30, image_size, 60):
                        draw.pieslice([x-20, y-20, x+20, y+20], 0, 240, fill=c1, outline=c2)
                        draw.ellipse([x-6, y-6, x+6, y+6], fill=c2)

            elif motif_name == "bandhani_dots":
                # Clusters of tie-dye dots
                for y in range(16, image_size, 24):
                    for x in range(16, image_size, 24):
                        draw.ellipse([x-4, y-4, x+4, y+4], fill=c1, outline=c2, width=1)

            elif motif_name == "floral_jaal":
                # Interlocking floral vines
                for y in range(20, image_size, 48):
                    draw.line([(0, y), (image_size, y)], fill=c2, width=2)
                    for x in range(24, image_size, 48):
                        draw.ellipse([x-12, y-12, x+12, y+12], fill=c1, outline=c2, width=2)

            elif motif_name == "geometric_chevron":
                # Woven chevron stripes
                for y in range(-image_size, image_size * 2, 28):
                    draw.line([(0, y), (image_size // 2, y + 20), (image_size, y)], fill=c1, width=4)

            elif motif_name == "zari_grid_lattice":
                # Checked grid with center motifs
                for x in range(0, image_size, 32):
                    draw.line([(x, 0), (x, image_size)], fill=c2, width=2)
                for y in range(0, image_size, 32):
                    draw.line([(0, y), (image_size, y)], fill=c2, width=2)
                for x in range(16, image_size, 32):
                    for y in range(16, image_size, 32):
                        draw.rectangle([x-5, y-5, x+5, y+5], fill=c1)

            save_path = folder / f"colorway_{c_idx+1:02d}.jpg"
            img.save(save_path, quality=95)

    return output_dir
