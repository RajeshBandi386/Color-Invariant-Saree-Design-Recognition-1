"""
Flask Web Server & REST API for Color-Invariant Saree Design Recognition.
Provides interactive verification, gallery retrieval, benchmarks, and dataset tools.
"""
import io
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from flask import Flask, jsonify, request, send_from_directory, send_file
import numpy as np
from PIL import Image
import torch

from config import Config
from dataset import (
    generate_mock_saree_dataset,
    load_saree_data_from_dir,
    split_gallery_query,
)
from inference import SareeMatcher
from efficiency_benchmark import (
    build_model,
    count_parameters,
    estimate_flops,
    benchmark_latency,
)
from evaluate import run_full_evaluation

# Initialize Flask application
app = Flask(__name__, static_folder="static", static_url_path="")

# Initialize matcher instance
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "outputs" / "best_saree_model.pt"
DATA_DIR = BASE_DIR / "data"

matcher: Optional[SareeMatcher] = None


def get_matcher() -> SareeMatcher:
    """Lazy-loads and caches the global SareeMatcher instance."""
    global matcher
    if matcher is None:
        if WEIGHTS_PATH.exists():
            matcher = SareeMatcher(weights_path=WEIGHTS_PATH)
        else:
            matcher = SareeMatcher(weights_path=None, backbone="convnext_tiny")
    return matcher


@app.route("/")
def index():
    """Serves the main single page web application."""
    return send_from_directory("static", "index.html")


@app.route("/data/<path:filename>")
def serve_data_file(filename: str):
    """Serves static image files from the data directory."""
    return send_from_directory(DATA_DIR, filename)


@app.route("/api/status", methods=["GET"])
def get_system_status():
    """Returns runtime configuration, model info, and available datasets."""
    m = get_matcher()
    device_str = str(m.device)
    has_weights = WEIGHTS_PATH.exists()
    
    # Check existing dataset directories
    datasets = []
    if DATA_DIR.exists():
        for d in DATA_DIR.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                sub_count = len([x for x in d.iterdir() if x.is_dir()])
                img_count = len(list(d.rglob("*.jpg"))) + len(list(d.rglob("*.png")))
                datasets.append({
                    "name": d.name,
                    "path": str(d.relative_to(BASE_DIR)),
                    "classes_count": sub_count,
                    "images_count": img_count,
                })

    return jsonify({
        "status": "online",
        "project": "Color-Invariant Saree Design Recognition (DeepLure AIE-CASE)",
        "backbone": "ConvNeXt-Tiny (Metric Projection Head)",
        "embedding_dim": 512,
        "device": device_str,
        "cuda_available": torch.cuda.is_available(),
        "checkpoint_loaded": has_weights,
        "checkpoint_path": str(WEIGHTS_PATH.relative_to(BASE_DIR)) if has_weights else None,
        "datasets": datasets,
    })


@app.route("/api/presets", methods=["GET"])
def get_presets():
    """Returns curated preset image pairs for instant 1-click verification testing."""
    demo_dir = DATA_DIR / "demo_sarees"
    if not demo_dir.exists() or not any(demo_dir.iterdir()):
        generate_mock_saree_dataset(demo_dir, num_designs=6, colorways_per_design=5)

    presets = [
        {
            "id": "preset_peacock_pos",
            "title": "Peacock Butta Motif (Positive Match)",
            "description": "Same peacock circle feather motif across Crimson-Gold vs Royal Blue-Silver colorways.",
            "type": "positive",
            "img1_url": "/data/demo_sarees/design_01_peacock_butta/colorway_01.jpg",
            "img2_url": "/data/demo_sarees/design_01_peacock_butta/colorway_02.jpg",
            "img1_name": "Peacock Butta (Colorway 1)",
            "img2_name": "Peacock Butta (Colorway 2)",
            "expected": "MATCH (Cosine Sim >= 0.70)",
        },
        {
            "id": "preset_temple_pos",
            "title": "Temple Triangles Motif (Positive Match)",
            "description": "Stepped zig-zag temple border motif across Emerald Green vs Rani Pink colorways.",
            "type": "positive",
            "img1_url": "/data/demo_sarees/design_02_temple_triangles/colorway_01.jpg",
            "img2_url": "/data/demo_sarees/design_02_temple_triangles/colorway_03.jpg",
            "img1_name": "Temple Triangles (Colorway 1)",
            "img2_name": "Temple Triangles (Colorway 3)",
            "expected": "MATCH (Cosine Sim >= 0.70)",
        },
        {
            "id": "preset_ikat_pos",
            "title": "Ikat Diamonds Motif (Positive Match)",
            "description": "Diamond lattice geometric motif across contrasting background palettes.",
            "type": "positive",
            "img1_url": "/data/demo_sarees/design_03_ikat_diamonds/colorway_02.jpg",
            "img2_url": "/data/demo_sarees/design_03_ikat_diamonds/colorway_04.jpg",
            "img1_name": "Ikat Diamonds (Colorway 2)",
            "img2_name": "Ikat Diamonds (Colorway 4)",
            "expected": "MATCH (Cosine Sim >= 0.70)",
        },
        {
            "id": "preset_paisley_pos",
            "title": "Paisley Kalka Motif (Positive Match)",
            "description": "Teardrop paisley curve weave pattern in two distinct colorways.",
            "type": "positive",
            "img1_url": "/data/demo_sarees/design_04_paisley_kalka/colorway_01.jpg",
            "img2_url": "/data/demo_sarees/design_04_paisley_kalka/colorway_05.jpg",
            "img1_name": "Paisley Kalka (Colorway 1)",
            "img2_name": "Paisley Kalka (Colorway 5)",
            "expected": "MATCH (Cosine Sim >= 0.70)",
        },
        {
            "id": "preset_neg_peacock_bandhani",
            "title": "Peacock Butta vs Bandhani Dots (Negative Match)",
            "description": "Completely different weave motifs (Circular Feathers vs Clustered Dots).",
            "type": "negative",
            "img1_url": "/data/demo_sarees/design_01_peacock_butta/colorway_01.jpg",
            "img2_url": "/data/demo_sarees/design_05_bandhani_dots/colorway_01.jpg",
            "img1_name": "Peacock Butta",
            "img2_name": "Bandhani Dots",
            "expected": "NO MATCH (Cosine Sim < 0.50)",
        },
        {
            "id": "preset_neg_temple_floral",
            "title": "Temple Border vs Floral Jaal (Negative Match)",
            "description": "Geometric zig-zag border vs Interlocking organic floral vines.",
            "type": "negative",
            "img1_url": "/data/demo_sarees/design_02_temple_triangles/colorway_02.jpg",
            "img2_url": "/data/demo_sarees/design_06_floral_jaal/colorway_02.jpg",
            "img1_name": "Temple Triangles",
            "img2_name": "Floral Jaal",
            "expected": "NO MATCH (Cosine Sim < 0.50)",
        },
    ]
    return jsonify(presets)


@app.route("/api/verify", methods=["POST"])
def verify_pair():
    """
    1:1 Pair Verification Endpoint.
    Accepts either multipart file uploads ('img1', 'img2') or relative paths ('img1_path', 'img2_path').
    """
    t0 = time.perf_counter()
    m = get_matcher()

    threshold = float(request.form.get("threshold", 0.65))

    # Check for file uploads vs URL/paths
    img1 = None
    img2 = None

    if "img1" in request.files and request.files["img1"].filename != "":
        img1 = Image.open(request.files["img1"].stream).convert("RGB")
    elif "img1_path" in request.form:
        rel_p = request.form["img1_path"].lstrip("/")
        if rel_p.startswith("data/"):
            rel_p = rel_p[len("data/"):]
        target = DATA_DIR / rel_p
        if target.exists():
            img1 = Image.open(target).convert("RGB")

    if "img2" in request.files and request.files["img2"].filename != "":
        img2 = Image.open(request.files["img2"].stream).convert("RGB")
    elif "img2_path" in request.form:
        rel_p = request.form["img2_path"].lstrip("/")
        if rel_p.startswith("data/"):
            rel_p = rel_p[len("data/"):]
        target = DATA_DIR / rel_p
        if target.exists():
            img2 = Image.open(target).convert("RGB")

    if img1 is None or img2 is None:
        return jsonify({"error": "Please provide both image 1 and image 2 (files or paths)."}), 400

    # Extract embeddings and compute similarity
    emb1 = m.extract_embedding(img1)
    emb2 = m.extract_embedding(img2)

    cosine_sim = float(np.dot(emb1, emb2))
    is_match = cosine_sim >= threshold
    confidence = float(np.clip((cosine_sim + 1.0) / 2.0, 0.0, 1.0))
    cosine_dist = float(1.0 - cosine_sim)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return jsonify({
        "is_same_design": is_match,
        "verdict": "MATCH (Same Motif)" if is_match else "NO MATCH (Different Motifs)",
        "cosine_similarity": round(cosine_sim, 4),
        "cosine_distance": round(cosine_dist, 4),
        "confidence_percentage": round(confidence * 100.0, 1),
        "threshold": threshold,
        "latency_ms": round(latency_ms, 2),
        "embedding_dimension": len(emb1),
        "norm_img1": round(float(np.linalg.norm(emb1)), 4),
        "norm_img2": round(float(np.linalg.norm(emb2)), 4),
    })


@app.route("/api/search", methods=["POST"])
def search_gallery():
    """
    1:N Gallery Search Endpoint.
    Searches a target gallery folder for the closest motif matches.
    """
    t0 = time.perf_counter()
    m = get_matcher()

    top_k = int(request.form.get("top_k", 5))
    gallery_name = request.form.get("gallery", "demo_sarees")

    target_gallery_dir = DATA_DIR / gallery_name
    if not target_gallery_dir.exists():
        target_gallery_dir = DATA_DIR / "demo_sarees"
        if not target_gallery_dir.exists():
            generate_mock_saree_dataset(target_gallery_dir, num_designs=6, colorways_per_design=5)

    # Resolve Query Image
    query_img = None
    if "query_file" in request.files and request.files["query_file"].filename != "":
        query_img = Image.open(request.files["query_file"].stream).convert("RGB")
    elif "query_path" in request.form:
        rel_p = request.form["query_path"].lstrip("/")
        if rel_p.startswith("data/"):
            rel_p = rel_p[len("data/"):]
        target = DATA_DIR / rel_p
        if target.exists():
            query_img = Image.open(target).convert("RGB")

    if query_img is None:
        return jsonify({"error": "No query image provided."}), 400

    # Collect gallery images
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    gallery_files = [p for p in target_gallery_dir.rglob("*") if p.suffix.lower() in valid_exts]

    if not gallery_files:
        return jsonify({"error": f"No gallery images found in {target_gallery_dir}."}), 400

    query_emb = m.extract_embedding(query_img)

    gallery_embs = []
    valid_paths = []
    for p in gallery_files:
        try:
            emb = m.extract_embedding(p)
            gallery_embs.append(emb)
            valid_paths.append(p)
        except Exception:
            continue

    gal_matrix = np.stack(gallery_embs, axis=0)
    sims = np.dot(gal_matrix, query_emb)
    ranked_indices = np.argsort(-sims)[:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        file_path = valid_paths[idx]
        rel_path = file_path.relative_to(DATA_DIR)
        parent_motif = file_path.parent.name
        results.append({
            "rank": rank,
            "image_url": f"/data/{rel_path.as_posix()}",
            "filename": file_path.name,
            "motif_class": parent_motif.replace("design_", "").replace("_", " ").title(),
            "cosine_similarity": round(float(sims[idx]), 4),
            "score_percentage": round(float(np.clip((sims[idx] + 1.0) / 2.0, 0.0, 1.0)) * 100.0, 1),
            "is_strong_match": bool(sims[idx] >= 0.65),
        })

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return jsonify({
        "query_searched_against": len(valid_paths),
        "top_k": top_k,
        "latency_ms": round(latency_ms, 2),
        "matches": results,
    })


@app.route("/api/benchmark", methods=["GET"])
def get_benchmark():
    """Profiles efficiency across ConvNeXt-Tiny, EfficientNet-B0, and ResNet-50."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = ["convnext_tiny", "efficientnet_b0", "resnet50"]

    results = []
    for name in candidates:
        try:
            model = build_model(backbone_name=name, embedding_dim=512, pretrained=False, device=device)
            total_p, _, size_mb = count_parameters(model)
            flops = estimate_flops(model)
            lat = benchmark_latency(model, batch_size=1, num_runs=25, warmup=5, device=device)

            display_name = {
                "convnext_tiny": "ConvNeXt-Tiny (Our Metric Model)",
                "efficientnet_b0": "EfficientNet-B0 (Edge / Mobile)",
                "resnet50": "ResNet-50 (Standard Baseline)",
            }.get(name, name)

            results.append({
                "model_id": name,
                "display_name": display_name,
                "parameters_m": round(total_p / 1e6, 2),
                "memory_mb": round(size_mb, 1),
                "gflops": round(flops, 2),
                "mean_latency_ms": lat["mean_latency_ms"],
                "median_latency_ms": lat["median_latency_ms"],
                "p95_latency_ms": lat["p95_latency_ms"],
                "throughput_fps": lat["throughput_fps"],
                "embedding_footprint": "512-D (2.048 KB)",
                "device": str(device),
            })
        except Exception as e:
            results.append({"model_id": name, "error": str(e)})

    return jsonify({
        "device": str(device),
        "benchmark_results": results,
        "in_memory_1m_gallery_ram_gb": 2.048,
    })


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Returns the full evaluation metrics across 1:N Identification and 1:1 Verification."""
    demo_dir = DATA_DIR / "demo_sarees"
    if not demo_dir.exists() or not any(demo_dir.iterdir()):
        generate_mock_saree_dataset(demo_dir, num_designs=6, colorways_per_design=5)

    samples, class_to_idx, _ = load_saree_data_from_dir(demo_dir)
    gallery_samples, query_samples = split_gallery_query(samples, query_ratio=0.5, seed=42)

    m = get_matcher()
    device = m.device

    eval_results = run_full_evaluation(
        model=m.model,
        gallery_samples=gallery_samples,
        query_samples=query_samples,
        device=device,
        batch_size=16,
    )

    clean_results = {k: round(float(v), 2) if isinstance(v, (int, float, np.floating)) else v for k, v in eval_results.items()}
    return jsonify({
        "status": "success",
        "dataset_samples": len(samples),
        "unique_motifs": len(class_to_idx),
        "gallery_count": len(gallery_samples),
        "query_count": len(query_samples),
        "metrics": clean_results,
    })


@app.route("/api/generate-mock", methods=["POST"])
def generate_mock():
    """Generates synthetic multi-colorway textile motifs."""
    data = request.get_json(silent=True) or {}
    num_designs = int(data.get("num_designs", 6))
    colorways = int(data.get("colorways", 5))

    target_dir = DATA_DIR / "demo_sarees"
    generate_mock_saree_dataset(target_dir, num_designs=num_designs, colorways_per_design=colorways)

    samples, class_to_idx, _ = load_saree_data_from_dir(target_dir)

    return jsonify({
        "status": "success",
        "message": f"Generated {len(samples)} synthetic textile images across {len(class_to_idx)} motif categories.",
        "total_images": len(samples),
        "total_classes": len(class_to_idx),
        "path": "/data/demo_sarees",
    })


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Start Saree Recognition Web Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print("\n" + "="*70)
    print(" COLOR-INVARIANT SAREE DESIGN RECOGNITION - WEB INTERFACE")
    print(f" Dashboard URL: http://{args.host}:{args.port}")
    print("="*70 + "\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
