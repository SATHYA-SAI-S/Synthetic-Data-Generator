#!/usr/bin/env bash
# scripts/reproduce_end_to_end.sh
# Entrypoint for Kaggle/GPU deployment.

set -e

echo "=== Privacy-Preserving Synthetic Healthcare Data Generation ==="
echo "Ensuring dependencies are installed..."
pip install -r requirements.txt || echo "requirements.txt not found, assuming env is prepped."

echo "Starting epsilon sweeps..."
python scripts/reproduce_end_to_end.py

echo "=== Sweep Completed Successfully ==="
echo "Check outputs/sweep_results/ for synthetic datasets."
