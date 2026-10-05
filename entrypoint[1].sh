#!/usr/bin/env bash
set -e

echo "========================================================================"
echo " Starting Saree Design Recognition Container"
echo " Environment: Production (Containerized Architecture)"
echo "========================================================================"

mkdir -p /app/data /app/outputs

case "$1" in
  serve)
    echo ">> Launching Web API & Interface on port ${PORT:-5000}..."
    exec python app.py --host 0.0.0.0 --port "${PORT:-5000}"
    ;;
  demo)
    echo ">> Running End-to-End Demonstration..."
    exec python run_demo.py
    ;;
  train)
    shift
    echo ">> Running Training Pipeline with args: $*..."
    exec python train.py "$@"
    ;;
  benchmark)
    shift
    echo ">> Running Model Complexity & Efficiency Benchmark..."
    exec python efficiency_benchmark.py "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
