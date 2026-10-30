#!/bin/bash
set -e
echo "========================================================"
echo "Reproducing Machine Learning Assignment 2"
echo "Roll Number: M26CSA017 | Seed: 26017"
echo "========================================================"

python scripts/run_all.py
echo "Compiling CVPR Report to PDF..."
python -c "import typst; typst.compile('report/report.typ', output='report/report.pdf')"
echo "[SUCCESS] All experiments reproduced and report/report.pdf generated!"
