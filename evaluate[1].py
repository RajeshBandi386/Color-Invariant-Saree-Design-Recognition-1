"""
Comprehensive Evaluation Protocol for Color-Invariant Saree Design Recognition.
Covers both Identification (1:N Rank-1, Rank-5, mAP) and Verification (1:1 ROC-AUC, EER, F1).
"""
from typing import Dict, List, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

from dataset import SareeDataset, get_color_invariant_transforms


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extracts L2-normalized embeddings, labels, and filepaths for an entire dataset.
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    all_paths = []

    for images, labels, paths in dataloader:
        images = images.to(device)
        embeddings = model(images, normalize=True)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())
        all_paths.extend(paths)

    embeddings_arr = np.concatenate(all_embeddings, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    return embeddings_arr, labels_arr, all_paths


def evaluate_identification(
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    topk: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Evaluates 1:N Identification Retrieval.
    Computes CMC Rank-1, Rank-5, Rank-10 accuracy and Mean Average Precision (mAP).
    """
    # Cosine similarity matrix (since vectors are unit normalized)
    similarity_matrix = np.dot(query_embeddings, gallery_embeddings.T)
    num_queries = len(query_labels)
    
    cmc_hits = {k: 0 for k in topk}
    average_precisions = []

    for i in range(num_queries):
        q_label = query_labels[i]
        sims = similarity_matrix[i]
        
        # Sort gallery indices by descending similarity
        ranked_indices = np.argsort(-sims)
        ranked_labels = gallery_labels[ranked_indices]
        
        # Binary match array
        matches = (ranked_labels == q_label).astype(int)
        
        # Rank-k accuracy
        for k in topk:
            if np.any(matches[:k]):
                cmc_hits[k] += 1
                
        # Average Precision (AP) for this query
        num_positives = np.sum(matches)
        if num_positives == 0:
            average_precisions.append(0.0)
            continue
            
        cum_matches = np.cumsum(matches)
        ranks = np.arange(1, len(matches) + 1)
        precisions = cum_matches / ranks
        ap = np.sum(precisions * matches) / num_positives
        average_precisions.append(ap)

    metrics = {f"Rank-{k}": (cmc_hits[k] / num_queries) * 100.0 for k in topk}
    metrics["mAP"] = float(np.mean(average_precisions)) * 100.0
    return metrics


def evaluate_verification(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_pairs: int = 10000,
    seed: int = 42,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluates 1:1 Pairwise Verification.
    Generates genuine (same motif) and imposter (different motif) pairs,
    calculates ROC Curve, Area Under Curve (AUC), Equal Error Rate (EER),
    and Optimal Decision Threshold.
    """
    np.random.seed(seed)
    n = len(labels)
    
    genuine_sims = []
    imposter_sims = []
    
    # Generate all or sampled pairs
    indices = np.arange(n)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if labels[i] == labels[j]:
                genuine_sims.append(sim)
            else:
                imposter_sims.append(sim)

    # Subsample if too many imposter pairs
    if len(imposter_sims) > max_pairs:
        imposter_sims = list(np.random.choice(imposter_sims, max_pairs, replace=False))
        
    y_true = np.array([1] * len(genuine_sims) + [0] * len(imposter_sims))
    y_scores = np.array(genuine_sims + imposter_sims)

    if len(np.unique(y_true)) < 2:
        return {"ROC-AUC": 0.0, "EER": 0.0, "Optimal_Threshold": 0.0, "Accuracy": 0.0}

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate (EER): where FPR == FNR (1 - TPR)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    # Find threshold maximizing classification accuracy
    best_acc = 0.0
    best_thresh = 0.5
    for t in np.linspace(-0.2, 0.95, 100):
        preds = (y_scores >= t).astype(int)
        acc = np.mean(preds == y_true)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    return {
        "ROC-AUC": float(roc_auc) * 100.0,
        "EER": float(eer) * 100.0,
        "EER_Threshold": eer_threshold,
        "Optimal_Threshold": float(best_thresh),
        "Best_Pair_Accuracy": float(best_acc) * 100.0,
        "Mean_Genuine_Sim": float(np.mean(genuine_sims)) if genuine_sims else 0.0,
        "Mean_Imposter_Sim": float(np.mean(imposter_sims)) if imposter_sims else 0.0,
        "fpr": fpr,
        "tpr": tpr,
    }


def run_full_evaluation(
    model: torch.nn.Module,
    gallery_samples: List[Tuple[Path, int]],
    query_samples: List[Tuple[Path, int]],
    device: torch.device,
    image_size: int = 224,
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Executes the complete evaluation suite across identification and verification.
    """
    val_transform = get_color_invariant_transforms(image_size=image_size, is_training=False)

    gal_dataset = SareeDataset(gallery_samples, transform=val_transform)
    query_dataset = SareeDataset(query_samples, transform=val_transform)

    gal_loader = DataLoader(gal_dataset, batch_size=batch_size, shuffle=False)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)

    gal_embs, gal_lbls, _ = extract_embeddings(model, gal_loader, device)
    q_embs, q_lbls, _ = extract_embeddings(model, query_loader, device)

    # 1. Identification Evaluation
    id_results = evaluate_identification(q_embs, q_lbls, gal_embs, gal_lbls)

    # 2. Verification Evaluation on full validation set
    all_embs = np.concatenate([gal_embs, q_embs], axis=0)
    all_lbls = np.concatenate([gal_lbls, q_lbls], axis=0)
    ver_results = evaluate_verification(all_embs, all_lbls)

    combined = {**id_results, **{k: v for k, v in ver_results.items() if not isinstance(v, np.ndarray)}}
    return combined
