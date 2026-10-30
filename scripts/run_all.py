"""
Master reproduction script for Machine Learning Assignment 2:
Bagging, Boosting and Bias-Variance on Face Recognition.
Executes Q1, Q2, and Q3 end-to-end and validates all outputs.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.run_q1 import main as run_q1_main
from scripts.run_q2 import main as run_q2_main
from scripts.run_q3 import main as run_q3_main


def main():
    print("=" * 80)
    print("STARTING FULL REPRODUCTION PIPELINE: ASSIGNMENT 2")
    print("Roll Number: M26CSA017 | Seed: 26017 | Student: Prem Kumar")
    print("=" * 80)
    start_total = time.time()

    print("\n>>> STEP 1: Running Question 1 (Decision Tree Baseline & Bagging from Scratch)...")
    t0 = time.time()
    run_q1_main()
    print(f">>> Question 1 completed in {time.time() - t0:.2f} seconds.")

    print("\n>>> STEP 2: Running Question 2 (Multiclass Boosting from Scratch)...")
    t0 = time.time()
    run_q2_main()
    print(f">>> Question 2 completed in {time.time() - t0:.2f} seconds.")

    print("\n>>> STEP 3: Running Question 3 (Bias and Variance as Measured Quantities)...")
    t0 = time.time()
    run_q3_main()
    print(f">>> Question 3 completed in {time.time() - t0:.2f} seconds.")

    total_time = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"FULL REPRODUCTION COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("Generated Figures in 'figures/':")
    for f in sorted(os.listdir(os.path.join(PROJECT_ROOT, "figures"))):
        print(f"  - figures/{f}")
    print("Generated Logs in 'logs/':")
    for f in sorted(os.listdir(os.path.join(PROJECT_ROOT, "logs"))):
        print(f"  - logs/{f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
