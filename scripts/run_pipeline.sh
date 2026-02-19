#!/bin/bash
# run_pipeline.sh — Master script for SPX options data pipeline
#
# Usage:
#   ./scripts/run_pipeline.sh [--synthetic]  # Use synthetic data
#   ./scripts/run_pipeline.sh [--wrds]       # Download from WRDS (requires credentials)

set -euo pipefail
cd "$(dirname "$0")/.."

# Activate venv
source venv/bin/activate

MODE="${1:---synthetic}"

echo "============================================================"
echo "  SPX Options Data Pipeline"
echo "  Mode: ${MODE}"
echo "  Time: $(date -u)"
echo "============================================================"

if [ "$MODE" = "--wrds" ]; then
    echo ""
    echo "Step 1: Download from WRDS..."
    python3 -u scripts/01_download_wrds_data.py --start 2010-01-01 --end 2024-12-31
elif [ "$MODE" = "--synthetic" ]; then
    echo ""
    echo "Step 1: Generate synthetic data..."
    python3 -u scripts/01b_generate_synthetic_data.py --n-dates 2000
else
    echo "Unknown mode: ${MODE}. Use --synthetic or --wrds"
    exit 1
fi

echo ""
echo "Step 2: Preprocess data..."
python3 -u scripts/02_preprocess_data.py

echo ""
echo "Step 3: Generate descriptive statistics..."
python3 -u scripts/03_descriptive_statistics.py

echo ""
echo "============================================================"
echo "  ✅ Pipeline complete!"
echo "  Time: $(date -u)"
echo ""
echo "  Output files:"
ls -lh data/processed/*.parquet 2>/dev/null
ls -lh data/descriptive_stats/*.csv 2>/dev/null
echo "============================================================"
