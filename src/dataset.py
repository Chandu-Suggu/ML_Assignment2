"""
Dataset utilities for Olivetti Faces experiments.
Handles reproducible stratified splitting (6 train, 2 val, 2 test per identity)
using roll number seed (26017) and creates the center-occluded masked test set.
"""

import os
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

ROLL_NUMBER = "M26CSA017"
ROLL_SEED = 26017
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
SAVED_SPLIT_PATH = os.path.join(DATA_DIR, "saved_split.npz")


def apply_mask(images_flat, block_start=24, block_end=40):
    """
    Sets the centered 16x16 block (rows and columns 24 to 39 inclusive) to zero.
    images_flat: numpy array of shape (N, 4096)
    Returns: masked numpy array of shape (N, 4096)
    """
    masked = images_flat.copy()
    for i in range(len(masked)):
        img = masked[i].reshape(64, 64)
        img[block_start:block_end, block_start:block_end] = 0.0
        masked[i] = img.flatten()
    return masked


def create_and_save_split(save_path=SAVED_SPLIT_PATH):
    """
    Creates the official train / val / test split and saves it to disk.
    Each of the 40 identities has exactly 6 train, 2 val, and 2 test images.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rng = np.random.RandomState(ROLL_SEED)
    faces = fetch_olivetti_faces(data_home=DATA_DIR, shuffle=True, random_state=rng)
    X = faces.data
    y = faces.target
    images = faces.images

    train_idx, val_idx, test_idx = [], [], []
    for i in range(40):
        identity_indices = np.where(y == i)[0]
        rng.shuffle(identity_indices)
        train_idx.extend(identity_indices[:6])
        val_idx.extend(identity_indices[6:8])
        test_idx.extend(identity_indices[8:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test_clean, y_test = X[test_idx], y[test_idx]
    X_test_masked = apply_mask(X_test_clean)

    np.savez(
        save_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test_clean=X_test_clean,
        X_test_masked=X_test_masked,
        y_test=y_test,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        roll_seed=ROLL_SEED,
        roll_number=ROLL_NUMBER,
    )
    print(f"Dataset split successfully saved to {save_path}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Clean Test: {X_test_clean.shape}, Masked Test: {X_test_masked.shape}")
    return X_train, y_train, X_val, y_val, X_test_clean, X_test_masked, y_test


def load_data(save_path=SAVED_SPLIT_PATH):
    """
    Loads the saved split from disk. If not found, generates and saves it.
    """
    if not os.path.exists(save_path):
        return create_and_save_split(save_path)

    data = np.load(save_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test_clean = data["X_test_clean"]
    X_test_masked = data["X_test_masked"]
    y_test = data["y_test"]
    return X_train, y_train, X_val, y_val, X_test_clean, X_test_masked, y_test


if __name__ == "__main__":
    create_and_save_split()
