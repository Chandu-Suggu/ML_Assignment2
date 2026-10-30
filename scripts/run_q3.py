"""
Script to execute all Question 3 experiments:
3.(a) Measured Bias^2, Variance, and Test Accuracy (Mean +/- Std) across R=30 resampled refits
      for Selected Single Tree, Bagging (B=100), and Boosting (Selected)
3.(b) Bias-variance decomposition across 3 depths (Shallow d=2, Intermediate d=8, Deep d=None)
      for Single Trees and 100-Tree Bagged Ensembles; Variance removed by Bagging
3.(c) Comprehensive analysis of accuracy improvement vs tree depth
3.(d) Master synthesis table and plots (Clean Acc, Masked Acc, Bias^2, Variance)
"""

import os
import sys
import json
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, "data", "mpl")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score

from src.dataset import load_data, ROLL_SEED, ROLL_NUMBER
from src.bagging import BaggedDecisionTreeEnsemble
from src.boosting import SAMMEClassifier
from src.bias_variance import resample_within_identity, compute_bias_variance


def main():
    print("=" * 70)
    print("QUESTION 3: BIAS AND VARIANCE AS MEASURED QUANTITIES")
    print("=" * 70)

    fig_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "figures"))
    log_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "logs"))
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Load Data
    X_train, y_train, X_val, y_val, X_test_clean, X_test_masked, y_test = load_data()
    N_test = len(y_test)
    K = 40
    R = 30  # 30 repetitions as specified

    # Load frozen configurations from Q1 and Q2
    with open(os.path.join(log_dir, "q1_results.json"), "r") as f:
        q1_data = json.load(f)
    frozen_tree_params = q1_data["best_params"]

    with open(os.path.join(log_dir, "q2_results.json"), "r") as f:
        q2_data = json.load(f)
    boost_depth = q2_data["selected_boosting_model"]["depth"]
    boost_rounds = q2_data["selected_boosting_model"]["rounds"]

    print(f"Configurations:")
    print(f"  Selected Single Tree: {frozen_tree_params}")
    print(f"  Bagging: B=100, Base Tree={frozen_tree_params}")
    print(f"  Boosting: Depth={boost_depth}, Rounds={boost_rounds}")
    print(f"  Repetitions: R={R}, Test Samples: N={N_test}")

    # =========================================================================
    # 3.(a) Bias-Variance across 30 Resampled Fits
    # =========================================================================
    print("\n--- 3.(a) Measured Bias^2 and Variance across 30 Refits ---")
    tree_p_r = np.zeros((R, N_test, K), dtype=np.float64)
    bag_p_r = np.zeros((R, N_test, K), dtype=np.float64)
    boost_p_r = np.zeros((R, N_test, K), dtype=np.float64)

    tree_accs, bag_accs, boost_accs = [], [], []

    for r in range(R):
        # Resample within identity with replacement
        Xr, yr = resample_within_identity(X_train, y_train, n_classes=K, seed=ROLL_SEED + r * 101)

        # 1. Fit Selected Single Tree
        dt = DecisionTreeClassifier(**frozen_tree_params, random_state=ROLL_SEED + r)
        dt.fit(Xr, yr)
        tree_preds = dt.predict(X_test_clean)
        tree_accs.append(accuracy_score(y_test, tree_preds))
        tree_p_r[r] = label_binarize(tree_preds, classes=np.arange(K))

        # 2. Fit Bagged Ensemble (B=100)
        bag = BaggedDecisionTreeEnsemble(base_tree_params=frozen_tree_params, n_estimators=100, random_state=ROLL_SEED + r)
        bag.fit(Xr, yr)
        bag_preds = bag.predict(X_test_clean)
        bag_accs.append(accuracy_score(y_test, bag_preds))
        bag_p_r[r] = label_binarize(bag_preds, classes=np.arange(K))

        # 3. Fit Selected Boosting Model (SAMME)
        boost = SAMMEClassifier(max_depth=boost_depth, n_estimators=boost_rounds, n_classes=K, random_state=ROLL_SEED + r)
        boost.fit(Xr, yr)
        boost_preds = boost.predict(X_test_clean)
        boost_accs.append(accuracy_score(y_test, boost_preds))
        # Prompt explicitly specifies: "For boosting use the normalised weighted vote vector"
        boost_p_r[r] = boost.predict_proba(X_test_clean)

        if (r + 1) % 10 == 0 or r == R - 1:
            print(f"  Completed {r + 1}/{R} refits...")

    tree_bias_sq, tree_var = compute_bias_variance(tree_p_r, y_test, n_classes=K)
    bag_bias_sq, bag_var = compute_bias_variance(bag_p_r, y_test, n_classes=K)
    boost_bias_sq, boost_var = compute_bias_variance(boost_p_r, y_test, n_classes=K)

    print("\n3.(a) Results Summary:")
    print(f"Single Tree | Bias^2: {tree_bias_sq:.4f} | Variance: {tree_var:.4f} | Clean Test Acc: {np.mean(tree_accs):.4f} +/- {np.std(tree_accs):.4f}")
    print(f"Bagging     | Bias^2: {bag_bias_sq:.4f} | Variance: {bag_var:.4f} | Clean Test Acc: {np.mean(bag_accs):.4f} +/- {np.std(bag_accs):.4f}")
    print(f"Boosting    | Bias^2: {boost_bias_sq:.4f} | Variance: {boost_var:.4f} | Clean Test Acc: {np.mean(boost_accs):.4f} +/- {np.std(boost_accs):.4f}")

    # =========================================================================
    # 3.(b) Bias-Variance vs Tree Depth (Shallow, Selected, Deep)
    # =========================================================================
    print("\n--- 3.(b) Bias-Variance across Three Tree Depths ---")
    # Three depths: shallow (d=2), intermediate (d=8), deep (d=None)
    depths_to_test = [2, 8, None]
    results_3b = {}

    for d in depths_to_test:
        d_name = "None" if d is None else str(d)
        print(f"Evaluating Depth = {d_name} across {R} refits...")
        t_pr_d = np.zeros((R, N_test, K), dtype=np.float64)
        b_pr_d = np.zeros((R, N_test, K), dtype=np.float64)
        t_accs_d = []
        b_accs_d = []

        for r in range(R):
            Xr, yr = resample_within_identity(X_train, y_train, n_classes=K, seed=ROLL_SEED + r * 101)

            # Single tree
            dt = DecisionTreeClassifier(max_depth=d, random_state=ROLL_SEED + r)
            dt.fit(Xr, yr)
            dt_p = dt.predict(X_test_clean)
            t_accs_d.append(accuracy_score(y_test, dt_p))
            t_pr_d[r] = label_binarize(dt_p, classes=np.arange(K))

            # 100-tree Bagging
            bag = BaggedDecisionTreeEnsemble(base_tree_params={"max_depth": d}, n_estimators=100, random_state=ROLL_SEED + r)
            bag.fit(Xr, yr)
            bag_p = bag.predict(X_test_clean)
            b_accs_d.append(accuracy_score(y_test, bag_p))
            b_pr_d[r] = label_binarize(bag_p, classes=np.arange(K))

        tb_sq, tvar = compute_bias_variance(t_pr_d, y_test, n_classes=K)
        bb_sq, bvar = compute_bias_variance(b_pr_d, y_test, n_classes=K)
        var_removed = tvar - bvar
        t_acc_mean = float(np.mean(t_accs_d))
        b_acc_mean = float(np.mean(b_accs_d))
        acc_gain = b_acc_mean - t_acc_mean

        results_3b[d_name] = {
            "depth": d_name,
            "tree_bias_sq": tb_sq,
            "tree_var": tvar,
            "tree_acc_mean": t_acc_mean,
            "tree_acc_std": float(np.std(t_accs_d)),
            "bag_bias_sq": bb_sq,
            "bag_var": bvar,
            "bag_acc_mean": b_acc_mean,
            "bag_acc_std": float(np.std(b_accs_d)),
            "variance_removed": var_removed,
            "acc_improvement": acc_gain,
        }
        print(f"  Depth {d_name:4s} | Tree Acc: {t_acc_mean:.4f} -> Bag Acc: {b_acc_mean:.4f} (Gain: {acc_gain:+.4f})")
        print(f"             | Tree: Bias^2={tb_sq:.4f}, Var={tvar:.4f} | Bag: Bias^2={bb_sq:.4f}, Var={bvar:.4f} | Var Removed={var_removed:.4f}")

    # Plot 3.(b) Depth Comparison Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    depth_labels = ["Shallow (d=2)", "Mid (d=8)", "Deep (d=None)"]
    x = np.arange(len(depth_labels))
    width = 0.35

    tree_biases = [results_3b[k]["tree_bias_sq"] for k in ["2", "8", "None"]]
    bag_biases = [results_3b[k]["bag_bias_sq"] for k in ["2", "8", "None"]]
    tree_vars = [results_3b[k]["tree_var"] for k in ["2", "8", "None"]]
    bag_vars = [results_3b[k]["bag_var"] for k in ["2", "8", "None"]]

    # Panel 1: Bias^2 Comparison
    axes[0].bar(x - width/2, tree_biases, width, label="Single Tree", color="#1f77b4")
    axes[0].bar(x + width/2, bag_biases, width, label="100-Tree Bagging", color="#aec7e8")
    axes[0].set_title("Bias Squared across Depths", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(depth_labels, fontsize=10)
    axes[0].set_ylabel("Measured Bias^2", fontsize=10)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")

    # Panel 2: Variance Comparison
    axes[1].bar(x - width/2, tree_vars, width, label="Single Tree", color="#d62728")
    axes[1].bar(x + width/2, bag_vars, width, label="100-Tree Bagging", color="#ff9896")
    axes[1].set_title("Variance across Depths", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(depth_labels, fontsize=10)
    axes[1].set_ylabel("Measured Variance", fontsize=10)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plot_depth_bv_path = os.path.join(fig_dir, "q3_b_depth_bias_variance.png")
    plt.savefig(plot_depth_bv_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Depth Bias-Variance figure saved to {plot_depth_bv_path}")

    # =========================================================================
    # 3.(d) Master Synthesis Table
    # =========================================================================
    print("\n--- 3.(d) Master Synthesis Table ---")
    # Masked Test Evaluations for the selected models on the original training set
    single_tree_orig = DecisionTreeClassifier(**frozen_tree_params, random_state=ROLL_SEED)
    single_tree_orig.fit(X_train, y_train)
    orig_tree_mask_acc = float(accuracy_score(y_test, single_tree_orig.predict(X_test_masked)))

    bag_orig = BaggedDecisionTreeEnsemble(base_tree_params=frozen_tree_params, n_estimators=100, random_state=ROLL_SEED)
    bag_orig.fit(X_train, y_train)
    orig_bag_mask_acc = float(accuracy_score(y_test, bag_orig.predict(X_test_masked)))

    boost_orig = SAMMEClassifier(max_depth=boost_depth, n_estimators=boost_rounds, n_classes=K, random_state=ROLL_SEED)
    boost_orig.fit(X_train, y_train)
    orig_boost_mask_acc = float(accuracy_score(y_test, boost_orig.predict(X_test_masked)))

    master_table = [
        {
            "Model": "Single Tree (Selected)",
            "Clean Test Acc (Mean)": float(np.mean(tree_accs)),
            "Clean Test Acc (Std)": float(np.std(tree_accs)),
            "Masked Test Acc": orig_tree_mask_acc,
            "Measured Bias^2": tree_bias_sq,
            "Measured Variance": tree_var,
        },
        {
            "Model": "Bagging (Selected, B=100)",
            "Clean Test Acc (Mean)": float(np.mean(bag_accs)),
            "Clean Test Acc (Std)": float(np.std(bag_accs)),
            "Masked Test Acc": orig_bag_mask_acc,
            "Measured Bias^2": bag_bias_sq,
            "Measured Variance": bag_var,
        },
        {
            "Model": "Boosting (Selected, SAMME)",
            "Clean Test Acc (Mean)": float(np.mean(boost_accs)),
            "Clean Test Acc (Std)": float(np.std(boost_accs)),
            "Masked Test Acc": orig_boost_mask_acc,
            "Measured Bias^2": boost_bias_sq,
            "Measured Variance": boost_var,
        },
    ]

    df_master = pd.DataFrame(master_table)
    print("\nFinal Bias, Variance, and Accuracy Table:")
    print(df_master.to_string(index=False))

    # Save summary to JSON
    q3_summary = {
        "section_3a": {
            "single_tree": {
                "bias_sq": tree_bias_sq,
                "variance": tree_var,
                "mean_acc": float(np.mean(tree_accs)),
                "std_acc": float(np.std(tree_accs)),
            },
            "bagging": {
                "bias_sq": bag_bias_sq,
                "variance": bag_var,
                "mean_acc": float(np.mean(bag_accs)),
                "std_acc": float(np.std(bag_accs)),
            },
            "boosting": {
                "bias_sq": boost_bias_sq,
                "variance": boost_var,
                "mean_acc": float(np.mean(boost_accs)),
                "std_acc": float(np.std(boost_accs)),
            },
        },
        "section_3b_depths": results_3b,
        "section_3d_master_table": master_table,
    }

    log_path = os.path.join(log_dir, "q3_results.json")
    with open(log_path, "w") as f:
        json.dump(q3_summary, f, indent=2)
    print(f"Results successfully saved to {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
