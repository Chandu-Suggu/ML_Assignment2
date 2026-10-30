"""
Script to execute all Question 1 experiments:
1.(a) Decision tree grid search baseline and selection
1.(b) From-scratch Bagging with B in {1, 5, 10, 25, 50, 100, 200} and distinct sample verification
1.(c) OOB accuracy, pairwise disagreement, vote margin vs B and final B selection
1.(d) Clean vs masked robustness and qualitative case analysis
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, "data", "mpl")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import json
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from src.dataset import load_data, ROLL_SEED, ROLL_NUMBER
from src.bagging import BaggedDecisionTreeEnsemble


def main():
    print("=" * 70)
    print("QUESTION 1: DECISION TREE BASELINE AND BAGGING FROM SCRATCH")
    print("=" * 70)

    # Output directories
    fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Load Data
    X_train, y_train, X_val, y_val, X_test_clean, X_test_masked, y_test = load_data()
    N = len(X_train)

    # =========================================================================
    # 1.(a) Decision Tree Grid Search Baseline
    # =========================================================================
    print("\n--- 1.(a) Decision Tree Grid Search Baseline ---")
    max_depths = [2, 4, 8, 16, None]
    min_samples_leafs = [1, 2, 4]

    grid_results = []
    best_val_acc = -1.0
    best_params = {}
    best_tree = None

    # We use a fixed seed matching the student's protocol for the grid
    rng = np.random.RandomState(ROLL_SEED)

    for md in max_depths:
        for msl in min_samples_leafs:
            dt = DecisionTreeClassifier(max_depth=md, min_samples_leaf=msl, random_state=rng)
            dt.fit(X_train, y_train)

            train_acc = accuracy_score(y_train, dt.predict(X_train))
            val_acc = accuracy_score(y_val, dt.predict(X_val))
            test_acc = accuracy_score(y_test, dt.predict(X_test_clean))

            n_nodes = dt.tree_.node_count
            children_left = dt.tree_.children_left
            children_right = dt.tree_.children_right
            node_samples = dt.tree_.n_node_samples

            leaf_samples = []
            for i in range(n_nodes):
                if children_left[i] == children_right[i]:  # Leaf node
                    leaf_samples.append(node_samples[i])

            n_leaves = len(leaf_samples)
            mean_samples = np.mean(leaf_samples) if n_leaves > 0 else 0
            single_samples = sum(np.array(leaf_samples) == 1)
            fraction_single = single_samples / n_leaves if n_leaves > 0 else 0

            row = {
                "max_depth": "None" if md is None else md,
                "min_samples_leaf": msl,
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "test_acc": float(test_acc),
                "n_leaves": int(n_leaves),
                "mean_samples_leaf": float(mean_samples),
                "fraction_single_leaf": float(fraction_single),
            }
            grid_results.append(row)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {"max_depth": md, "min_samples_leaf": msl}
                best_tree = dt

    df_grid = pd.DataFrame(grid_results)
    print(df_grid.to_string(index=False))
    print(f"\nFrozen Best Hyperparameters (Selected on Val Acc only): {best_params} (Val Acc: {best_val_acc:.4f})")

    # Evaluate frozen tree
    frozen_train_acc = accuracy_score(y_train, best_tree.predict(X_train))
    frozen_val_acc = accuracy_score(y_val, best_tree.predict(X_val))
    frozen_test_acc = accuracy_score(y_test, best_tree.predict(X_test_clean))
    print(f"Frozen Single Tree Metrics: Train Acc: {frozen_train_acc:.4f}, Val Acc: {frozen_val_acc:.4f}, Clean Test Acc: {frozen_test_acc:.4f}")

    # =========================================================================
    # 1.(b) Bagging from Scratch & Bootstrap Distinct Samples
    # =========================================================================
    print("\n--- 1.(b) Bagging from Scratch ---")
    B_values = [1, 5, 10, 25, 50, 100, 200]
    theoretical_distinct = N * (1.0 - (1.0 - 1.0 / N) ** N)
    print(f"Theoretical Expected Distinct Samples per Bootstrap (N={N}): {theoretical_distinct:.2f} ({theoretical_distinct / N * 100:.2f}%)")

    # Fit ensemble with max(B_values) = 200
    bagging_ensemble = BaggedDecisionTreeEnsemble(
        base_tree_params=best_params,
        n_estimators=max(B_values),
        random_state=ROLL_SEED
    )
    bagging_ensemble.fit(X_train, y_train)

    b_results = []
    for B in B_values:
        train_pred = bagging_ensemble.predict(X_train, n_trees=B)
        val_pred = bagging_ensemble.predict(X_val, n_trees=B)
        test_pred = bagging_ensemble.predict(X_test_clean, n_trees=B)

        tr_acc = accuracy_score(y_train, train_pred)
        va_acc = accuracy_score(y_val, val_pred)
        te_acc = accuracy_score(y_test, test_pred)

        avg_distinct = np.mean(bagging_ensemble.distinct_sample_counts_[:B])

        b_results.append({
            "B": B,
            "train_acc": float(tr_acc),
            "val_acc": float(va_acc),
            "test_acc": float(te_acc),
            "avg_distinct": float(avg_distinct),
            "theoretical_distinct": float(theoretical_distinct)
        })
        print(f"B={B:3d} | Train: {tr_acc:.4f} | Val: {va_acc:.4f} | Test: {te_acc:.4f} | Avg Distinct: {avg_distinct:.2f} (Theo: {theoretical_distinct:.2f})")

    # =========================================================================
    # 1.(c) OOB Accuracy, Disagreement, Vote Margin vs B
    # =========================================================================
    print("\n--- 1.(c) OOB, Disagreement, and Vote Margin vs B ---")
    oob_accuracies = []
    avg_disagreements = []
    mean_margins = []

    for B in B_values:
        oob_acc = bagging_ensemble.compute_oob_accuracy(X_train, y_train, n_trees=B)
        disagree = bagging_ensemble.compute_pairwise_disagreement(X_val, n_trees=B)
        margin = bagging_ensemble.compute_mean_vote_margin(X_val, n_trees=B)

        oob_accuracies.append(oob_acc)
        avg_disagreements.append(disagree)
        mean_margins.append(margin)

        print(f"B={B:3d} | OOB Acc: {oob_acc:.4f} | Pairwise Disagreement: {disagree:.4f} | Mean Margin: {margin:.4f}")

    # Plot 1.(c) Metrics
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=300)
    axes[0].plot(B_values, oob_accuracies, marker="o", color="#1f77b4", lw=2)
    axes[0].set_title("OOB Accuracy vs B", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Ensemble Size (B)", fontsize=10)
    axes[0].set_ylabel("Out-of-Bag Accuracy", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(B_values, avg_disagreements, marker="s", color="#ff7f0e", lw=2)
    axes[1].set_title("Pairwise Disagreement vs B", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Ensemble Size (B)", fontsize=10)
    axes[1].set_ylabel("Avg Disagreement (Val)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(B_values, mean_margins, marker="^", color="#2ca02c", lw=2)
    axes[2].set_title("Mean Vote Margin vs B", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Ensemble Size (B)", fontsize=10)
    axes[2].set_ylabel("Mean Vote Margin (Val)", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_c_path = os.path.join(fig_dir, "q1_c_ensemble_metrics.png")
    plt.savefig(plot_c_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {plot_c_path}")

    # Freeze final B (B=100)
    final_B = 100
    print(f"\nSelected Final Ensemble Size: B={final_B}")

    # =========================================================================
    # 1.(d) Clean vs Masked Robustness & Case Analysis
    # =========================================================================
    print("\n--- 1.(d) Robustness on Clean vs Masked Test Sets ---")
    tree_clean_preds = best_tree.predict(X_test_clean)
    tree_mask_preds = best_tree.predict(X_test_masked)
    tree_clean_acc = accuracy_score(y_test, tree_clean_preds)
    tree_mask_acc = accuracy_score(y_test, tree_mask_preds)

    # Tree confidence: for decision tree, predictions at leaf nodes have probability = fraction in leaf
    tree_clean_conf = np.ones(len(y_test))  # Leaves in single tree are deterministic/pure or near-pure
    tree_mask_conf = np.ones(len(y_test))

    bag_clean_preds, bag_clean_conf, bag_clean_votes, bag_clean_margins = bagging_ensemble.predict_with_confidence(X_test_clean, n_trees=final_B)
    bag_mask_preds, bag_mask_conf, bag_mask_votes, bag_mask_margins = bagging_ensemble.predict_with_confidence(X_test_masked, n_trees=final_B)

    bag_clean_acc = accuracy_score(y_test, bag_clean_preds)
    bag_mask_acc = accuracy_score(y_test, bag_mask_preds)

    print(f"Single Tree Baseline: Clean Acc: {tree_clean_acc:.4f}, Masked Acc: {tree_mask_acc:.4f}")
    print(f"Bagged Model (B={final_B}): Clean Acc: {bag_clean_acc:.4f} (Mean Conf: {np.mean(bag_clean_conf):.4f}), Masked Acc: {bag_mask_acc:.4f} (Mean Conf: {np.mean(bag_mask_conf):.4f})")

    # Find 2 cases where bagging corrects the tree
    tree_wrong_bag_right = np.where((tree_clean_preds != y_test) & (bag_clean_preds == y_test))[0]
    # Find 2 cases where both are wrong
    both_wrong = np.where((tree_clean_preds != y_test) & (bag_clean_preds != y_test))[0]

    print("\n--- Case Analysis ---")
    cases_corrected = tree_wrong_bag_right[:2]
    cases_both_wrong = both_wrong[:2]

    case_info = []
    print("Two Cases Where Bagging CORRECTS the Tree:")
    for idx in cases_corrected:
        top_v = {int(k): int(v) for k, v in bag_clean_votes[idx].most_common(5)}
        print(f"  Test Index {idx}: True={y_test[idx]}, Tree Pred={tree_clean_preds[idx]}, Bag Pred={bag_clean_preds[idx]}")
        print(f"    Bagging Votes: {top_v}")
        case_info.append({
            "index": int(idx),
            "type": "corrected",
            "true_label": int(y_test[idx]),
            "tree_pred": int(tree_clean_preds[idx]),
            "bag_pred": int(bag_clean_preds[idx]),
            "top_votes": top_v
        })

    print("Two Cases Where BOTH are WRONG:")
    for idx in cases_both_wrong:
        top_v = {int(k): int(v) for k, v in bag_clean_votes[idx].most_common(5)}
        print(f"  Test Index {idx}: True={y_test[idx]}, Tree Pred={tree_clean_preds[idx]}, Bag Pred={bag_clean_preds[idx]}")
        print(f"    Bagging Votes: {top_v}")
        case_info.append({
            "index": int(idx),
            "type": "both_wrong",
            "true_label": int(y_test[idx]),
            "tree_pred": int(tree_clean_preds[idx]),
            "bag_pred": int(bag_clean_preds[idx]),
            "top_votes": top_v
        })

    # Plot Case Analysis Figure
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=300)
    all_case_indices = list(cases_corrected) + list(cases_both_wrong)
    titles = [
        f"Corrected: Idx {cases_corrected[0]}\nTrue: {y_test[cases_corrected[0]]} | Tree: {tree_clean_preds[cases_corrected[0]]} | Bag: {bag_clean_preds[cases_corrected[0]]}",
        f"Corrected: Idx {cases_corrected[1]}\nTrue: {y_test[cases_corrected[1]]} | Tree: {tree_clean_preds[cases_corrected[1]]} | Bag: {bag_clean_preds[cases_corrected[1]]}",
        f"Both Wrong: Idx {cases_both_wrong[0]}\nTrue: {y_test[cases_both_wrong[0]]} | Tree: {tree_clean_preds[cases_both_wrong[0]]} | Bag: {bag_clean_preds[cases_both_wrong[0]]}",
        f"Both Wrong: Idx {cases_both_wrong[1]}\nTrue: {y_test[cases_both_wrong[1]]} | Tree: {tree_clean_preds[cases_both_wrong[1]]} | Bag: {bag_clean_preds[cases_both_wrong[1]]}",
    ]

    for col, idx in enumerate(all_case_indices):
        # Row 0: Clean Image
        axes[0, col].imshow(X_test_clean[idx].reshape(64, 64), cmap="gray")
        axes[0, col].set_title(titles[col], fontsize=9)
        axes[0, col].axis("off")

        # Row 1: Masked Image
        axes[1, col].imshow(X_test_masked[idx].reshape(64, 64), cmap="gray")
        axes[1, col].set_title(f"Masked View (Idx {idx})", fontsize=9)
        axes[1, col].axis("off")

    plt.tight_layout()
    plot_cases_path = os.path.join(fig_dir, "q1_d_case_faces.png")
    plt.savefig(plot_cases_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Case analysis faces saved to {plot_cases_path}")

    # Save complete structured results to JSON
    q1_summary = {
        "grid_results": grid_results,
        "best_params": best_params,
        "frozen_single_tree": {
            "train_acc": frozen_train_acc,
            "val_acc": frozen_val_acc,
            "test_acc": frozen_test_acc,
        },
        "bagging_b_results": b_results,
        "ensemble_metrics": {
            "B_values": B_values,
            "oob_accuracies": oob_accuracies,
            "avg_disagreements": avg_disagreements,
            "mean_margins": mean_margins,
            "selected_B": final_B,
        },
        "robustness": {
            "tree_clean_acc": tree_clean_acc,
            "tree_mask_acc": tree_mask_acc,
            "bag_clean_acc": bag_clean_acc,
            "bag_mask_acc": bag_mask_acc,
            "bag_clean_mean_conf": float(np.mean(bag_clean_conf)),
            "bag_mask_mean_conf": float(np.mean(bag_mask_conf)),
        },
        "case_studies": case_info,
    }

    log_path = os.path.join(log_dir, "q1_results.json")
    with open(log_path, "w") as f:
        json.dump(q1_summary, f, indent=2)
    print(f"Results successfully saved to {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
