"""
Inference & Matching Engine for Color-Invariant Saree Recognition.
Supports 1:N Gallery Search and 1:1 Pair Verification.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from config import Config
from dataset import get_color_invariant_transforms
from models import build_model


class SareeMatcher:
    """
    Production-ready matching and verification engine for saree motifs.
    """
    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        backbone: str = "convnext_tiny",
        embedding_dim: int = 512,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.transform = get_color_invariant_transforms(image_size=224, is_training=False)
        self.class_to_idx = {}
        self.idx_to_class = {}

        if weights_path is not None and Path(weights_path).exists():
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            loaded_cfg = checkpoint.get("config", None)
            bb = loaded_cfg.backbone if loaded_cfg else backbone
            emb_d = loaded_cfg.embedding_dim if loaded_cfg else embedding_dim

            self.model = build_model(
                backbone_name=bb,
                embedding_dim=emb_d,
                pretrained=False,
                device=self.device,
            )
            self.model.load_state_dict(checkpoint["model_state"])
            self.class_to_idx = checkpoint.get("class_to_idx", {})
            self.idx_to_class = checkpoint.get("idx_to_class", {})
            print(f" [LOAD] Loaded checkpoint weights from: {weights_path}")
        else:
            print(" [INFO] No checkpoint provided, initializing pretrained backbone directly.")
            self.model = build_model(
                backbone_name=backbone,
                embedding_dim=embedding_dim,
                pretrained=True,
                device=self.device,
            )

        self.model.eval()

    @torch.no_grad()
    def extract_embedding(self, image_input: Union[str, Path, Image.Image]) -> np.ndarray:
        """Extracts a 512-D L2-normalized metric embedding from an image."""
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.model(tensor, normalize=True)
        return embedding.squeeze(0).cpu().numpy()

    def verify_pair(
        self,
        img1: Union[str, Path, Image.Image],
        img2: Union[str, Path, Image.Image],
        threshold: float = 0.65,
    ) -> Dict[str, Union[bool, float, str]]:
        """
        Verifies if two saree images share the same design/motif regardless of color.
        """
        emb1 = self.extract_embedding(img1)
        emb2 = self.extract_embedding(img2)

        cosine_sim = float(np.dot(emb1, emb2))
        is_same_design = cosine_sim >= threshold
        confidence = float(np.clip((cosine_sim + 1.0) / 2.0, 0.0, 1.0))

        return {
            "is_same_design": is_same_design,
            "cosine_similarity": round(cosine_sim, 4),
            "confidence": round(confidence, 4),
            "threshold": threshold,
            "verdict": "MATCH (Same Motif)" if is_same_design else "NO MATCH (Different Motifs)",
        }

    def search_gallery(
        self,
        query_img: Union[str, Path, Image.Image],
        gallery_images: List[Union[str, Path]],
        top_k: int = 5,
    ) -> List[Dict[str, Union[str, float, int]]]:
        """
        Searches a gallery of reference images for the best motif matches.
        """
        query_emb = self.extract_embedding(query_img)
        
        gallery_embs = []
        valid_paths = []
        for path in gallery_images:
            try:
                emb = self.extract_embedding(path)
                gallery_embs.append(emb)
                valid_paths.append(str(path))
            except Exception as e:
                print(f"Skipping {path}: {e}")

        if not gallery_embs:
            return []

        gal_matrix = np.stack(gallery_embs, axis=0)
        sims = np.dot(gal_matrix, query_emb)
        ranked_indices = np.argsort(-sims)[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            results.append({
                "rank": rank,
                "image_path": valid_paths[idx],
                "cosine_similarity": round(float(sims[idx]), 4),
            })

        return results


def main():
    parser = argparse.ArgumentParser(description="Saree Design Inference & Matching")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Verification CLI
    verify_parser = subparsers.add_parser("verify", help="Verify 1:1 pair of images")
    verify_parser.add_argument("--img1", type=str, required=True, help="First saree image")
    verify_parser.add_argument("--img2", type=str, required=True, help="Second saree image")
    verify_parser.add_argument("--weights", type=str, default="./outputs/best_saree_model.pt")
    verify_parser.add_argument("--threshold", type=float, default=0.65)

    # Search CLI
    search_parser = subparsers.add_parser("search", help="1:N Search gallery for query match")
    search_parser.add_argument("--query", type=str, required=True, help="Query image")
    search_parser.add_argument("--gallery_dir", type=str, required=True, help="Directory of gallery images")
    search_parser.add_argument("--weights", type=str, default="./outputs/best_saree_model.pt")
    search_parser.add_argument("--topk", type=int, default=5)

    args = parser.parse_args()
    weights_path = Path(args.weights) if Path(args.weights).exists() else None
    matcher = SareeMatcher(weights_path=weights_path)

    if args.command == "verify":
        res = matcher.verify_pair(args.img1, args.img2, threshold=args.threshold)
        print("\n--- Verification Result ---")
        for k, v in res.items():
            print(f"{k}: {v}")

    elif args.command == "search":
        gal_dir = Path(args.gallery_dir)
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        gal_imgs = [p for p in gal_dir.rglob("*") if p.suffix.lower() in valid_exts]
        
        matches = matcher.search_gallery(args.query, gal_imgs, top_k=args.topk)
        print(f"\n--- Top {args.topk} Gallery Matches ---")
        for m in matches:
            print(f"Rank {m['rank']} | Sim: {m['cosine_similarity']:.4f} | {m['image_path']}")


if __name__ == "__main__":
    main()
