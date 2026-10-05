"""
Metric Learning Loss Functions for Color-Invariant Motif Matching.
Includes ArcFace (Additive Angular Margin Loss) and Supervised Contrastive Loss.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss).
    Forces embeddings of the same textile motif into a compact angular cone on a hypersphere.
    
    Formula:
        cos(theta + m) for ground truth target, where theta = arccos(cos_sim(e, W))
    
    Args:
        in_features: Dimension of input embedding (e.g. 512)
        num_classes: Number of distinct saree motifs in training set
        scale: Feature scale factor s (inverse temperature, typically 30.0)
        margin: Angular margin m in radians (typically 0.50 rad ~ 28.6 degrees)
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.50,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Learnable class weight centers on unit hypersphere
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [Batch, Embedding_Dim], already L2 normalized or unnormalized
            labels: [Batch] class indices
        """
        # Normalize weights & embeddings to unit hypersphere
        norm_emb = F.normalize(embeddings, p=2, dim=1)
        norm_w = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: cos(theta) = x * W
        cosine = F.linear(norm_emb, norm_w)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # Compute sin(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0.0, 1.0)

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot mask for target class
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Output logits with scaled margin for ground truth
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale

        loss = F.cross_entropy(logits, labels)
        return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).
    Pulls colorways of the same motif together while pushing different motifs apart.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        norm_emb = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(norm_emb, norm_emb.T) / self.temperature

        # Create mask of positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(labels.shape[0]).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute log-probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)

        # Mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)
        loss = -mean_log_prob_pos.mean()
        return loss
