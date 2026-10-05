"""
Configuration settings for Color-Invariant Saree Design Recognition.
Clean, centralized, and customizable without unnecessary complexity.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # --- Experiment & Workspace ---
    project_name: str = "saree_design_recognition"
    seed: int = 42
    output_dir: Path = Path("./outputs")
    
    # --- Data Settings ---
    data_dir: Path = Path("./data/saree_corpus")
    image_size: int = 224
    num_workers: int = 0  # 0 for safe multiplatform execution (Windows/Linux/Colab/Docker)
    
    # --- Model Architecture ---
    backbone: str = "convnext_tiny"  # options: 'convnext_tiny', 'efficientnet_b0', 'resnet50'
    embedding_dim: int = 512
    pretrained: bool = True
    dropout: float = 0.2
    
    # --- Metric Learning & Loss ---
    # ArcFace: Additive Angular Margin Loss for compact hypersphere clusters
    arcface_margin: float = 0.50  # radians (~28.6 degrees)
    arcface_scale: float = 30.0   # temperature scaling factor
    
    # --- Training Hyperparameters ---
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    use_amp: bool = True  # Automatic Mixed Precision
    
    # --- Augmentations for Color Invariance ---
    grayscale_prob: float = 0.35
    solarize_prob: float = 0.15
    hue_jitter: float = 0.4
    sat_jitter: float = 0.4
    contrast_jitter: float = 0.3
    brightness_jitter: float = 0.3
    
    # --- Verification Evaluation Protocol ---
    verification_threshold: float = 0.65  # Cosine similarity cut-off
    gallery_query_split_ratio: float = 0.5 # 50% colorways in gallery, 50% in query
