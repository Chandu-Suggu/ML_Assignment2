"""
Script to execute all Question 2 experiments:
2.(a) Synthetic 3-class toy example logging 1 complete SAMME round
2.(b) SAMME grid search (depths 1..4, rounds 10..100), selection on clean val acc,
      and 4-panel plot of training dynamics over rounds
2.(c) Bagging vs Boosting comparison across 25, 50, 100 learners (Acc & Time),
      plus inspection of heaviest training sample and prediction history
2.(d) Robustness evaluation across Single Tree, Bagging, and Boosting on clean vs masked,
      and deep analysis of a divergent test face
"""

import os
import sys
import time
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, "data", "mpl")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from src.dataset import load_data, ROLL_SEED, ROLL_NUMBER
from src.bagging import BaggedDecisionTreeEnsemble
from src.boosting import SAMMEClassifier


def main():
    print("=" * 70)
    print("QUESTION 2: MULTICLASS BOOSTING (SAMME) FROM SCRATCH")
    print("=" * 70)

    fig_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "figures"))
    log_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "logs"))
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Load Olivetti Face Data
    X_train, y_train, X_val, y_val, X_test_clean, X_test_masked, y_test = load_data()
    N_train = len(X_train)
    K = 40

    # =========================================================================
    # 2.(a) Synthetic 3-Class Toy Example: One Complete SAMME Round
    # =========================================================================
    print("\n--- 2.(a) Synthetic 3-Class Example (One Complete SAMME Round) ---")
    K_toy = 3
    N_toy = 6
    y_true_toy = np.array([0, 0, 1, 1, 2, 2])
    # Synthetic predictions with 3 mistakes (indices 1, 3, 5)
    y_pred_toy = np.array([0, 1, 1, 2, 2, 0])

    w_init_toy = np.ones(N_toy, dtype=np.float64) / N_toy
    mistakes_toy = (y_true_toy != y_pred_toy).astype(np.float64)
    err_t_toy = float(np.sum(w_init_toy * mistakes_toy))

    # SAMME formula: alpha_t = log((1 - err_t) / err_t) + log(K - 1)
    alpha_t_toy = float(np.log((1.0 - err_t_toy) / err_t_toy) + np.log(K_toy - 1))
    w_unnorm_toy = w_init_toy * np.exp(alpha_t_toy * mistakes_toy)
    w_updated_toy = w_unnorm_toy / np.sum(w_unnorm_toy)

    print(f"Initial sample weights: {np.round(w_init_toy, 4).tolist()}")
    print(f"True labels:            {y_true_toy.tolist()}")
    print(f"Weak learner preds:     {y_pred_toy.tolist()}")
    print(f"Mistakes (0/1):         {mistakes_toy.astype(int).tolist()}")
    print(f"Weighted error (eps_t): {err_t_toy:.4f}")
    print(f"Learner weight (alpha): {alpha_t_toy:.4f}")
    print(f"Updated weights:        {np.round(w_updated_toy, 4).tolist()}")

    toy_results = {
        "K": K_toy,
        "N": N_toy,
        "y_true": y_true_toy.tolist(),
        "y_pred": y_pred_toy.tolist(),
        "w_initial": w_init_toy.tolist(),
        "weighted_error": err_t_toy,
        "alpha_t": alpha_t_toy,
        "w_updated": w_updated_toy.tolist(),
    }

    # =========================================================================
    # 2.(b) SAMME Grid Search on Olivetti Faces & Dynamics Over Rounds
    # =========================================================================
    print("\n--- 2.(b) SAMME Grid Search & Dynamics on Olivetti Faces ---")
    depths = [1, 2, 3, 4]
    rounds = [10, 25, 50, 100]

    grid_results = []
    best_val_acc = -1.0
    best_model_params = None

    for d in depths:
        for r in rounds:
            model = SAMMEClassifier(max_depth=d, n_estimators=r, n_classes=K, random_state=ROLL_SEED)
            model.fit(X_train, y_train)

            val_preds = model.predict(X_val)
            val_acc = float(accuracy_score(y_val, val_preds))
            train_preds = model.predict(X_train)
            train_acc = float(accuracy_score(y_train, train_preds))

            grid_results.append({
                "depth": d,
                "rounds": r,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "effective_rounds": len(model.estimators_),
            })
            print(f"Depth: {d} | Rounds: {r:3d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Effective Rounds: {len(model.estimators_)}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_params = (d, r)

    best_depth, best_rounds = best_model_params
    print(f"\nSelected SAMME Model (Val Acc only): Depth={best_depth}, Rounds={best_rounds} with Val Acc={best_val_acc:.4f}")

    # Fit the selected model for deep logging
    selected_boosting = SAMMEClassifier(max_depth=best_depth, n_estimators=best_rounds, n_classes=K, random_state=ROLL_SEED)
    selected_boosting.fit(X_train, y_train)

    # 4-panel plot: weighted error, learner weight, max sample weight, top-10 sample weights
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    rounds_arr = np.arange(1, len(selected_boosting.err_history_) + 1)

    axes[0, 0].plot(rounds_arr, selected_boosting.err_history_, marker="o", markersize=3, color="#1f77b4")
    axes[0, 0].set_title("Weighted Training Error (eps_t)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Round (t)", fontsize=9)
    axes[0, 0].set_ylabel("Error", fontsize=9)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    axes[0, 1].plot(rounds_arr, selected_boosting.alpha_history_, marker="s", markersize=3, color="#ff7f0e")
    axes[0, 1].set_title("Learner Weight (alpha_t)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Round (t)", fontsize=9)
    axes[0, 1].set_ylabel("alpha_t", fontsize=9)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    axes[1, 0].plot(rounds_arr, selected_boosting.max_weight_history_, marker="^", markersize=3, color="#2ca02c")
    axes[1, 0].set_title("Maximum Sample Weight (max_i w_i)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Round (t)", fontsize=9)
    axes[1, 0].set_ylabel("Weight", fontsize=9)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    axes[1, 1].plot(rounds_arr, selected_boosting.top10_weight_history_, marker="d", markersize=3, color="#d62728")
    axes[1, 1].set_title("Total Weight of Top 10 Heaviest Samples", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Round (t)", fontsize=9)
    axes[1, 1].set_ylabel("Combined Weight", fontsize=9)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_dynamics_path = os.path.join(fig_dir, "q2_b_round_dynamics.png")
    plt.savefig(plot_dynamics_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Round dynamics plot saved to {plot_dynamics_path}")

    # =========================================================================
    # 2.(c) Bagging vs Boosting Comparison & Heaviest Sample Analysis
    # =========================================================================
    print("\n--- 2.(c) Comparison Between Bagging and Boosting (25, 50, 100 Learners) ---")
    compare_rounds = [25, 50, 100]
    comparison_results = []

    # Frozen best tree settings from Q1 for bagging
    with open(os.path.join(log_dir, "q1_results.json"), "r") as f:
        q1_data = json.load(f)
    frozen_tree_params = q1_data["best_params"]

    for r in compare_rounds:
        # Time and evaluate Bagging
        t0 = time.time()
        bag_ens = BaggedDecisionTreeEnsemble(base_tree_params=frozen_tree_params, n_estimators=r, random_state=ROLL_SEED)
        bag_ens.fit(X_train, y_train)
        bag_time = time.time() - t0

        bag_tr_acc = accuracy_score(y_train, bag_ens.predict(X_train))
        bag_va_acc = accuracy_score(y_val, bag_ens.predict(X_val))
        bag_te_acc = accuracy_score(y_test, bag_ens.predict(X_test_clean))

        # Time and evaluate Boosting
        t0 = time.time()
        boost_ens = SAMMEClassifier(max_depth=best_depth, n_estimators=r, n_classes=K, random_state=ROLL_SEED)
        boost_ens.fit(X_train, y_train)
        boost_time = time.time() - t0

        boost_tr_acc = accuracy_score(y_train, boost_ens.predict(X_train))
        boost_va_acc = accuracy_score(y_val, boost_ens.predict(X_val))
        boost_te_acc = accuracy_score(y_test, boost_ens.predict(X_test_clean))

        row = {
            "rounds": r,
            "bag_train_acc": float(bag_tr_acc),
            "bag_val_acc": float(bag_va_acc),
            "bag_test_acc": float(bag_te_acc),
            "bag_time": float(bag_time),
            "boost_train_acc": float(boost_tr_acc),
            "boost_val_acc": float(boost_va_acc),
            "boost_test_acc": float(boost_te_acc),
            "boost_time": float(boost_time),
        }
        comparison_results.append(row)
        print(f"Rounds: {r:3d}")
        print(f"  Bagging  | Train: {bag_tr_acc:.4f} | Val: {bag_va_acc:.4f} | Test: {bag_te_acc:.4f} | Time: {bag_time:.3f}s")
        print(f"  Boosting | Train: {boost_tr_acc:.4f} | Val: {boost_va_acc:.4f} | Test: {boost_te_acc:.4f} | Time: {boost_time:.3f}s")

    # Heaviest training sample inspection
    final_w = selected_boosting.final_weights_
    heaviest_idx = int(np.argmax(final_w))
    heaviest_weight = float(final_w[heaviest_idx])
    uniform_weight = 1.0 / N_train

    # Prediction history across first 25 rounds
    history_25 = [
        bool(selected_boosting.estimators_[t].predict(X_train[heaviest_idx].reshape(1, -1))[0] == y_train[heaviest_idx])
        for t in range(min(25, len(selected_boosting.estimators_)))
    ]

    print(f"\nHeaviest Training Face Index: {heaviest_idx}")
    print(f"True Identity: {y_train[heaviest_idx]}")
    print(f"Final Boosting Weight: {heaviest_weight:.5f} (Initial Uniform: {uniform_weight:.5f}, Ratio: {heaviest_weight / uniform_weight:.2f}x)")
    print(f"Prediction History (Correct=True) first 25 rounds: {history_25}")

    # Plot heaviest face
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    ax.imshow(X_train[heaviest_idx].reshape(64, 64), cmap="gray")
    ax.set_title(f"Heaviest Face (Idx: {heaviest_idx}, Class: {y_train[heaviest_idx]})\nFinal Weight: {heaviest_weight:.4f} ({heaviest_weight/uniform_weight:.1f}x uniform)", fontsize=10)
    ax.axis("off")
    heaviest_plot_path = os.path.join(fig_dir, "q2_c_heaviest_face.png")
    plt.savefig(heaviest_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Heaviest face plot saved to {heaviest_plot_path}")

    # =========================================================================
    # 2.(d) Robustness Evaluation & Divergent Response Analysis
    # =========================================================================
    print("\n--- 2.(d) Robustness Evaluation: Single Tree vs Bagging vs Boosting ---")
    # Single Tree Baseline
    single_tree = DecisionTreeClassifier(**frozen_tree_params, random_state=ROLL_SEED)
    single_tree.fit(X_train, y_train)

    tree_clean_preds = single_tree.predict(X_test_clean)
    tree_mask_preds = single_tree.predict(X_test_masked)
    tree_clean_acc = accuracy_score(y_test, tree_clean_preds)
    tree_mask_acc = accuracy_score(y_test, tree_mask_preds)
    tree_clean_conf = 1.0  # single tree leaf winning probability
    tree_mask_conf = 1.0

    # Final Bagged Model (B=100)
    final_bag = BaggedDecisionTreeEnsemble(base_tree_params=frozen_tree_params, n_estimators=100, random_state=ROLL_SEED)
    final_bag.fit(X_train, y_train)
    bag_clean_preds, bag_clean_confs, _, _ = final_bag.predict_with_confidence(X_test_clean)
    bag_mask_preds, bag_mask_confs, _, _ = final_bag.predict_with_confidence(X_test_masked)
    bag_clean_acc = accuracy_score(y_test, bag_clean_preds)
    bag_mask_acc = accuracy_score(y_test, bag_mask_preds)

    # Selected Boosting Model
    boost_clean_preds, boost_clean_confs = selected_boosting.predict_with_confidence(X_test_clean)
    boost_mask_preds, boost_mask_confs = selected_boosting.predict_with_confidence(X_test_masked)
    boost_clean_acc = accuracy_score(y_test, boost_clean_preds)
    boost_mask_acc = accuracy_score(y_test, boost_mask_preds)

    print(f"Single Tree | Clean Acc: {tree_clean_acc:.4f} (Conf: {tree_clean_conf:.4f}) | Masked Acc: {tree_mask_acc:.4f} (Conf: {tree_mask_conf:.4f})")
    print(f"Bagging     | Clean Acc: {bag_clean_acc:.4f} (Conf: {np.mean(bag_clean_confs):.4f}) | Masked Acc: {bag_mask_acc:.4f} (Conf: {np.mean(bag_mask_confs):.4f})")
    print(f"Boosting    | Clean Acc: {boost_clean_acc:.4f} (Conf: {np.mean(boost_clean_confs):.4f}) | Masked Acc: {boost_mask_acc:.4f} (Conf: {np.mean(boost_mask_confs):.4f})")

    # Divergent face: find where Bagging and Boosting respond differently to the missing region
    diverge_indices = np.where(bag_mask_preds != boost_mask_preds)[0]
    div_idx = int(diverge_indices[0]) if len(diverge_indices) > 0 else 0

    print(f"\nDivergent Face Index under Mask: {div_idx}")
    print(f"  True Label:    {y_test[div_idx]}")
    print(f"  Bagging Pred:  {bag_mask_preds[div_idx]} (Vote Fraction: {bag_mask_confs[div_idx]:.4f})")
    print(f"  Boosting Pred: {boost_mask_preds[div_idx]} (Confidence: {boost_mask_confs[div_idx]:.4f})")

    # Plot divergent face (clean vs masked)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    axes[0].imshow(X_test_clean[div_idx].reshape(64, 64), cmap="gray")
    axes[0].set_title(f"Clean Face (Idx: {div_idx})\nTrue Class: {y_test[div_idx]}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(X_test_masked[div_idx].reshape(64, 64), cmap="gray")
    axes[1].set_title(f"Masked Face (Idx: {div_idx})\nBag: {bag_mask_preds[div_idx]} ({bag_mask_confs[div_idx]:.2f}) | Boost: {boost_mask_preds[div_idx]} ({boost_mask_confs[div_idx]:.2f})", fontsize=10)
    axes[1].axis("off")

    plt.tight_layout()
    div_plot_path = os.path.join(fig_dir, "q2_d_divergence_face.png")
    plt.savefig(div_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Divergent face plot saved to {div_plot_path}")

    # Save complete structured summary to JSON
    q2_summary = {
        "toy_example": toy_results,
        "grid_results": grid_results,
        "selected_boosting_model": {
            "depth": best_depth,
            "rounds": best_rounds,
            "val_acc": float(best_val_acc),
        },
        "bag_vs_boost_comparison": comparison_results,
        "heaviest_sample": {
            "index": heaviest_idx,
            "true_identity": int(y_train[heaviest_idx]),
            "final_weight": heaviest_weight,
            "weight_ratio_over_uniform": float(heaviest_weight / uniform_weight),
            "prediction_history_25": history_25,
        },
        "robustness_summary": {
            "single_tree": {
                "clean_acc": float(tree_clean_acc),
                "masked_acc": float(tree_mask_acc),
                "clean_conf": float(tree_clean_conf),
                "masked_conf": float(tree_mask_conf),
            },
            "bagging": {
                "clean_acc": float(bag_clean_acc),
                "masked_acc": float(bag_mask_acc),
                "clean_conf": float(np.mean(bag_clean_confs)),
                "masked_conf": float(np.mean(bag_mask_confs)),
            },
            "boosting": {
                "clean_acc": float(boost_clean_acc),
                "masked_acc": float(boost_mask_acc),
                "clean_conf": float(np.mean(boost_clean_confs)),
                "masked_conf": float(np.mean(boost_mask_confs)),
            },
        },
        "divergent_case": {
            "index": div_idx,
            "true_label": int(y_test[div_idx]),
            "bag_pred": int(bag_mask_preds[div_idx]),
            "bag_conf": float(bag_mask_confs[div_idx]),
            "boost_pred": int(boost_mask_preds[div_idx]),
            "boost_conf": float(boost_mask_confs[div_idx]),
        },
    }

    log_path = os.path.join(log_dir, "q2_results.json")
    with open(log_path, "w") as f:
        json.dump(q2_summary, f, indent=2)
    print(f"Results successfully saved to {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
