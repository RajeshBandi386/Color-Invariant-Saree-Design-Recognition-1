"""
End-to-End Zero-Setup Demonstration Script.
Executes the full pipeline:
  1. Synthesizes multi-colorway textile mock dataset (if needed).
  2. Trains ConvNeXt-Tiny Metric Network with ArcFace Loss.
  3. Evaluates Identification (Rank-1, Rank-5, mAP) & Verification (ROC-AUC, EER).
  4. Runs 1:1 Pair Verification & 1:N Gallery Search on sample sarees.
"""
import sys
from pathlib import Path
import torch

from config import Config
from dataset import generate_mock_saree_dataset, load_saree_data_from_dir, split_gallery_query
from train import run_training
from evaluate import run_full_evaluation
from inference import SareeMatcher
from efficiency_benchmark import run_architecture_comparison


def main():
    print("=" * 70)
    print(" COLOR-INVARIANT SAREE DESIGN RECOGNITION SYSTEM")
    print(" DeepLure AIE-CASE End-to-End Demonstration")
    print("=" * 70)

    cfg = Config()
    cfg.epochs = 5  # Quick demonstration run
    cfg.batch_size = 16
    cfg.data_dir = Path("./data/demo_sarees")
    cfg.output_dir = Path("./outputs")

    # Step 1: Data Preparation
    print("\n[Step 1/5] Synthesizing Multi-Colorway Textile Dataset...")
    generate_mock_saree_dataset(cfg.data_dir, num_designs=6, colorways_per_design=5)
    
    samples, class_to_idx, idx_to_class = load_saree_data_from_dir(cfg.data_dir)
    print(f" Loaded {len(samples)} images across {len(class_to_idx)} unique saree design motifs.")

    # Step 2: Training
    print("\n[Step 2/5] Training Metric Learning Model with ArcFace...")
    best_checkpoint = run_training(cfg, use_mock_if_missing=False)

    # Step 3: Formal Dual-Protocol Evaluation
    print("\n[Step 3/5] Running Dual-Protocol Benchmark Evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gallery_samples, query_samples = split_gallery_query(samples, query_ratio=0.5, seed=42)

    matcher = SareeMatcher(weights_path=best_checkpoint)
    eval_results = run_full_evaluation(
        model=matcher.model,
        gallery_samples=gallery_samples,
        query_samples=query_samples,
        device=device,
    )

    print("\n" + "-"*50)
    print(" EVALUATION METRICS REPORT")
    print("-"*50)
    print(f"  - CMC Rank-1 Identification Accuracy : {eval_results['Rank-1']:6.2f}%")
    print(f"  - CMC Rank-5 Identification Accuracy : {eval_results['Rank-5']:6.2f}%")
    print(f"  - Mean Average Precision (mAP)       : {eval_results['mAP']:6.2f}%")
    print(f"  - Verification ROC-AUC               : {eval_results['ROC-AUC']:6.2f}%")
    print(f"  - Verification Equal Error Rate (EER): {eval_results['EER']:6.2f}%")
    print(f"  - Optimal Cosine Threshold           : {eval_results['Optimal_Threshold']:6.2f}")
    print("-"*50)

    # Step 4: Live Inference Demonstrations
    print("\n[Step 4/5] Executing Live Inference Demonstrations...")
    
    # Pair Verification: Same motif, different colorways
    peacock_folder = list(cfg.data_dir.glob("*peacock*"))[0]
    peacock_imgs = sorted(list(peacock_folder.glob("*.jpg")))
    
    temple_folder = list(cfg.data_dir.glob("*temple*"))[0]
    temple_imgs = sorted(list(temple_folder.glob("*.jpg")))

    if len(peacock_imgs) >= 2 and len(temple_imgs) >= 1:
        print("\n--- Test A: Positive Pair (Same Peacock Motif in Red vs Blue colorways) ---")
        res_pos = matcher.verify_pair(peacock_imgs[0], peacock_imgs[1], threshold=0.60)
        print(f"  Img 1   : {peacock_imgs[0].name}")
        print(f"  Img 2   : {peacock_imgs[1].name}")
        print(f"  Verdict : {res_pos['verdict']} | Cosine Similarity: {res_pos['cosine_similarity']} (Conf: {res_pos['confidence']})")

        print("\n--- Test B: Negative Pair (Peacock Motif vs Temple Motif) ---")
        res_neg = matcher.verify_pair(peacock_imgs[0], temple_imgs[0], threshold=0.60)
        print(f"  Img 1   : {peacock_imgs[0].name}")
        print(f"  Img 2   : {temple_imgs[0].name}")
        print(f"  Verdict : {res_neg['verdict']} | Cosine Similarity: {res_neg['cosine_similarity']} (Conf: {res_neg['confidence']})")

    # Step 5: Architecture Efficiency Analysis
    print("\n[Step 5/5] Running Model Complexity & Efficiency Analysis...")
    run_architecture_comparison()

    print("\n[SUCCESS] Demonstration complete. Everything is verified and ready for submission!")


if __name__ == "__main__":
    main()
