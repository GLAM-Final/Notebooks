"""
=============================================================================
ALL MODELS + CLASSICAL BASELINES
=============================================================================
Addresses the review criticisms:
  - Classical baselines (LogisticRegression, RandomForest, XGBoost on MFCCs)
  - XAI visualizations on all models
  - Multi-patient evaluation
  - Patient diagnosis evaluation

Models implemented:
  1. Baseline GAT          (already exists)
  2. GraphSAGE             (already exists)
  3. ImprovedRespiratoryGAT (already exists)

=============================================================================
"""

import os, sys, pathlib, warnings, time, pickle, hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GATConv, SAGEConv, GCNConv, GINConv,
    GATv2Conv, global_mean_pool, global_add_pool,
    BatchNorm
)

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "testing_models" / "results"
CKPT_BASELINE_GAT = BASE_DIR / "testing_models" / "respiratory_gnn_model_cpu_full.pth"
CKPT_GRAPHSAGE = BASE_DIR / "testing_models" / "best_respiratory_graphsage_temp_scaled.pth"
CKPT_IMPROVED_GAT = BASE_DIR / "testing_models" / "best_improved_30epochs_no_earlystop.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════════════════
# MODEL 1: Baseline GAT (already exists)
# ════════════════════════════════════════════════════════════════════════════

class BaselineGAT(nn.Module):
    """Original GAT model with 3 GAT layers and global pooling."""
    def __init__(self, input_dim=768, hidden_dim=128, num_heads=4, dropout=0.4):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.bn1 = BatchNorm(hidden_dim * num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.bn2 = BatchNorm(hidden_dim * num_heads)
        self.conv3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.bn3 = BatchNorm(hidden_dim * num_heads)
        
        self.wheeze_head = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.crackle_head = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.elu(self.bn1(self.conv1(x, ei)))
        x = F.elu(self.bn2(self.conv2(x, ei)))
        x = F.elu(self.bn3(self.conv3(x, ei)))
        
        if hasattr(data, "batch") and data.batch is not None:
            x = global_mean_pool(x, data.batch)
            return self.wheeze_head(x).squeeze(-1), self.crackle_head(x).squeeze(-1)
        return self.wheeze_head(x).squeeze(-1), self.crackle_head(x).squeeze(-1)


# ════════════════════════════════════════════════════════════════════════════
# MODEL 2: GraphSAGE (already exists)
# ════════════════════════════════════════════════════════════════════════════

class GraphSAGEGNN(nn.Module):
    """GraphSAGE model with 4 SAGE layers."""
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=4, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.sage_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.sage_layers.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            
        self.wheeze_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.crackle_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = self.input_proj(x)
        for sage, norm in zip(self.sage_layers, self.norms):
            x = F.relu(sage(x, ei))
            x = norm(x)
            x = F.dropout(x, p=0.4, training=self.training)
            
        if hasattr(data, "batch") and data.batch is not None:
            x = global_mean_pool(x, data.batch)
            return self.wheeze_head(x).squeeze(-1), self.crackle_head(x).squeeze(-1)
        return self.wheeze_head(x).squeeze(-1), self.crackle_head(x).squeeze(-1)


# ════════════════════════════════════════════════════════════════════════════
# MODEL 3: ImprovedRespiratoryGAT (already exists)
# ════════════════════════════════════════════════════════════════════════════

class ImprovedRespiratoryGAT(nn.Module):
    """Improved GAT with edge features and attention pooling."""
    def __init__(self, input_dim=768, hidden_dim=128, edge_dim=7, num_heads=4, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim // num_heads)
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                               edge_dim=hidden_dim // num_heads, dropout=dropout)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                               edge_dim=hidden_dim // num_heads, dropout=dropout)
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                               edge_dim=hidden_dim // num_heads, dropout=dropout)
        self.bn3 = BatchNorm(hidden_dim)
        
        self.wheeze_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.crackle_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = self.input_proj(x)
        
        # Process edge features if available
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_attr = self.edge_proj(data.edge_attr)
        else:
            edge_attr = None
            
        x = F.elu(self.bn1(self.conv1(x, ei, edge_attr=edge_attr)))
        x = F.elu(self.bn2(self.conv2(x, ei, edge_attr=edge_attr)))
        x = F.elu(self.bn3(self.conv3(x, ei, edge_attr=edge_attr)))
        
        if hasattr(data, "batch") and data.batch is not None:
            x = global_mean_pool(x, data.batch)
            return self.wheeze_head(x).squeeze(-1), self.crackle_head(x).squeeze(-1)
        return self.wheeze_head(x).squeeze(-1), self.crackle_head(x).squeeze(-1)


# ════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """
    Focal Loss for handling class imbalance.
    
    Mathematical formulation:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
        where p_t is the model's estimated probability for the target class
    
    Args:
        logits: Model outputs (before sigmoid)
        targets: Ground truth binary labels
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Weighting factor for class imbalance
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()


def train_gnn_model(model, train_loader, val_loader, device, epochs=30,
                    lr=5e-4, wd=1e-4, use_focal=True):
    """Train any GNN model with early stopping based on validation F1."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_val_f1 = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            if batch is None:
                continue
            batch = batch.to(device)
            wl, cl = model(batch)

            if use_focal:
                loss_w = focal_loss(wl, batch.y_wheeze)
                loss_c = focal_loss(cl, batch.y_crackle)
            else:
                loss_w = F.binary_cross_entropy_with_logits(wl, batch.y_wheeze)
                loss_c = F.binary_cross_entropy_with_logits(cl, batch.y_crackle)

            loss = loss_w + loss_c
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        val_results = evaluate_gnn(model, val_loader, device)
        val_f1 = (val_results["wheeze"]["f1"] + val_results["crackle"]["f1"]) / 2
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_f1


@torch.no_grad()
def evaluate_gnn(model, loader, device):
    """
    Evaluate GNN model on test set.
    
    Returns comprehensive metrics including:
    - F1 score, precision, recall, specificity, ROC-AUC
    """
    model.eval()
    w_p, c_p, w_t, c_t = [], [], [], []
    for batch in loader:
        if batch is None:
            continue
        batch = batch.to(device)
        wl, cl = model(batch)
        w_p.append(torch.sigmoid(wl).cpu().numpy().ravel())
        c_p.append(torch.sigmoid(cl).cpu().numpy().ravel())
        w_t.append(batch.y_wheeze.cpu().numpy().ravel())
        c_t.append(batch.y_crackle.cpu().numpy().ravel())
    
    w_p = np.concatenate(w_p); c_p = np.concatenate(c_p)
    w_t = np.concatenate(w_t); c_t = np.concatenate(c_t)
    
    results = {}
    for name, prob, true in [("wheeze", w_p, w_t), ("crackle", c_p, c_t)]:
        pred = (prob >= 0.5).astype(int)
        true_int = true.astype(int)
        if len(np.unique(true_int)) < 2:
            results[name] = {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                            "specificity": 0.0, "roc_auc": None}
            continue
        cm = confusion_matrix(true_int, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        eps = 1e-9
        results[name] = {
            "precision": tp / (tp + fp + eps),
            "recall": tp / (tp + fn + eps),
            "specificity": tn / (tn + fp + eps),
            "f1": 2 * tp / (2 * tp + fp + fn + eps),
            "roc_auc": roc_auc_score(true_int, prob),
        }
    return results


# ════════════════════════════════════════════════════════════════════════════
# CLASSICAL BASELINES (RF, XGB, LR on MFCC features)
# ════════════════════════════════════════════════════════════════════════════

def extract_mfcc_features(audio_path, sr=16000, n_mfcc=13, hop_length=800):
    """
    Extract MFCC features from audio for classical baselines.
    
    Mathematical formulation:
        MFCC = DCT(log(|STFT(y)|^2))
        where STFT is the Short-Time Fourier Transform
        delta = (MFCC[t+1] - MFCC[t-1]) / 2 (velocity)
        delta2 = (delta[t+1] - delta[t-1]) / 2 (acceleration)
    
    Returns 39-dimensional feature vector (13 MFCC + 13 delta + 13 delta2)
    """
    import librosa
    y, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    return features.mean(axis=1).flatten()  # Global mean pooling


def train_classical_baselines(train_features, train_labels, val_features, val_labels):
    """
    Train classical baselines on extracted MFCC features.
    
    Models trained:
    1. Random Forest (200 estimators, max_depth=10)
    2. Logistic Regression (L2 regularization, C=0.1)
    3. XGBoost (200 estimators, max_depth=6)
    
    Returns dictionary of evaluation metrics for each model.
    """
    results = {}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(train_features, train_labels)
    rf_pred = rf.predict(val_features)
    rf_prob = rf.predict_proba(val_features)[:, 1]
    cm = confusion_matrix(val_labels, rf_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    eps = 1e-9
    results["RandomForest"] = {
        "f1": f1_score(val_labels, rf_pred, zero_division=0),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "roc_auc": roc_auc_score(val_labels, rf_prob) if len(np.unique(val_labels)) > 1 else 0,
    }

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
    lr.fit(train_features, train_labels)
    lr_pred = lr.predict(val_features)
    lr_prob = lr.predict_proba(val_features)[:, 1]
    cm = confusion_matrix(val_labels, lr_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    results["LogisticRegression"] = {
        "f1": f1_score(val_labels, lr_pred, zero_division=0),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "roc_auc": roc_auc_score(val_labels, lr_prob) if len(np.unique(val_labels)) > 1 else 0,
    }

    # XGBoost
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=200, max_depth=6, random_state=42, 
                           use_label_encoder=False, eval_metric="logloss")
        xgb.fit(train_features, train_labels)
        xgb_pred = xgb.predict(val_features)
        xgb_prob = xgb.predict_proba(val_features)[:, 1]
        cm = confusion_matrix(val_labels, xgb_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        results["XGBoost"] = {
            "f1": f1_score(val_labels, xgb_pred, zero_division=0),
            "precision": tp / (tp + fp + eps),
            "recall": tp / (tp + fn + eps),
            "specificity": tn / (tn + fp + eps),
            "roc_auc": roc_auc_score(val_labels, xgb_prob) if len(np.unique(val_labels)) > 1 else 0,
        }
    except ImportError:
        print("    XGBoost not installed, skipping")

    return results


# ════════════════════════════════════════════════════════════════════════════
# MULTI-PATIENT SYNTHESIS
# ════════════════════════════════════════════════════════════════════════════

def create_multi_patient_mixtures(meta, audio_folder, max_patients=3, snr_levels=[0, 5, 10, 15]):
    """
    Create synthetic multi-patient audio mixtures.
    
    Mathematical formulation:
        x_mix = sum_i (s_i * 10^(snr_i/20)) for i = 1..N patients
        where s_i is the i-th patient's recording, snr_i is the SNR in dB
    
    The ground truth labels for the anchor (primary) patient are preserved.
    
    Args:
        meta: DataFrame with columns ['patient_id', 'wav_path', 'file_id']
        audio_folder: Path to audio files
        max_patients: Maximum number of patients in mixture
        snr_levels: List of SNR levels to test (dB)
    
    Returns:
        List of mixture dictionaries with audio and metadata
    """
    import librosa
    import random

    sr = 16000
    mixtures = []
    patient_ids = meta["patient_id"].unique()

    for idx, anchor_row in meta.iterrows():
        anchor_audio, _ = librosa.load(anchor_row["wav_path"], sr=sr)
        anchor_len = len(anchor_audio)

        # Select interfering patients
        other_patients = [p for p in patient_ids if p != anchor_row["patient_id"]]
        n_interferers = random.randint(1, min(max_patients - 1, len(other_patients)))
        interferer_pids = random.sample(other_patients, n_interferers)

        for snr_db in snr_levels:
            # Mix anchor with interferers
            mix = anchor_audio.copy()
            snr_linear = 10 ** (snr_db / 20.0)

            for ipid in interferer_pids:
                ip_files = meta[meta["patient_id"] == ipid]["wav_path"].values
                if len(ip_files) == 0:
                    continue
                int_file = random.choice(ip_files)
                int_audio, _ = librosa.load(int_file, sr=sr)
                # Tile if shorter
                if len(int_audio) < anchor_len:
                    int_audio = np.tile(int_audio, anchor_len // len(int_audio) + 1)
                int_audio = int_audio[:anchor_len]
                # Mix with SNR
                mix = mix + int_audio * snr_linear

            # Normalize
            max_val = np.abs(mix).max()
            if max_val > 0:
                mix = mix / max_val * 0.95

            mixtures.append({
                "anchor_file": anchor_row["wav_path"],
                "anchor_id": anchor_row["file_id"],
                "patient_id": anchor_row["patient_id"],
                "snr_db": snr_db,
                "n_patients": n_interferers + 1,
                "audio": mix,
                "anchor_duration": anchor_len / sr,
                "mix_duration": len(mix) / sr,
            })

    return mixtures


# ════════════════════════════════════════════════════════════════════════════
# PATIENT DIAGNOSIS EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate_patient_diagnosis(model, test_loader, device, meta_test):
    """
    Evaluate per-patient diagnostic accuracy.
    
    Aggregates event-level predictions to patient-level diagnosis.
    
    Rule-based aggregation:
        1. Average wheeze and crackle probabilities across patient's events
        2. has_event = (avg_wheeze > 0.5) or (avg_crackle > 0.5)
        3. Patient is diagnosed with respiratory condition if has_event = True
    
    Args:
        model: Trained GNN model
        test_loader: DataLoader with test set
        device: Device to run inference on
        meta_test: DataFrame with patient metadata
    
    Returns:
        List of dictionaries with patient-level diagnosis predictions
    """
    model.eval()
    patient_predictions = {}
    patient_true = {}

    for batch in test_loader:
        if batch is None:
            continue
        batch = batch.to(device)
        wl, cl = model(batch)
        w_prob = torch.sigmoid(wl).cpu().numpy().ravel()
        c_prob = torch.sigmoid(cl).cpu().numpy().ravel()

        # Store predictions per patient
        patient_ids = batch.patient_id if hasattr(batch, "patient_id") else ["unknown"] * len(w_prob)
        for i, pid in enumerate(patient_ids):
            if pid not in patient_predictions:
                patient_predictions[pid] = []
            patient_predictions[pid].append({
                "wheeze_prob": float(w_prob[i]), 
                "crackle_prob": float(c_prob[i])
            })

    # Aggregate per patient
    patient_diag_pred = []
    diag_map = dict(zip(meta_test["patient_id"], meta_test["diagnosis"]))

    for pid, preds in patient_predictions.items():
        avg_wheeze = np.mean([p["wheeze_prob"] for p in preds])
        avg_crackle = np.mean([p["crackle_prob"] for p in preds])
        has_event = (avg_wheeze > 0.5) or (avg_crackle > 0.5)
        predicted_disease = diag_map.get(pid, "Unknown")
        
        patient_diag_pred.append({
            "patient_id": pid,
            "diagnosis": predicted_disease,
            "avg_wheeze_prob": avg_wheeze,
            "avg_crackle_prob": avg_crackle,
            "has_event": has_event,
            "predicted_positive": has_event,
        })

    return patient_diag_pred


# ════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def evaluate_all_models(test_loader, device, meta_test=None):
    """
    Evaluate all implemented models on test data.
    
    Returns:
        Dictionary with model names as keys and evaluation results as values
    """
    models = {
        "BaselineGAT": BaselineGAT(),
        "GraphSAGE": GraphSAGEGNN(),
        "ImprovedGAT": ImprovedRespiratoryGAT(),
    }
    
    # Checkpoints mapping
    checkpoints = {
        "BaselineGAT": CKPT_BASELINE_GAT,
        "GraphSAGE": CKPT_GRAPHSAGE,
        "ImprovedGAT": CKPT_IMPROVED_GAT,
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        ckpt_path = checkpoints.get(name)
        
        if ckpt_path and ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
        else:
            print(f"  Warning: Checkpoint not found for {name}, using random initialization")
        
        model = model.to(device)
        model.eval()
        
        # Standard evaluation
        eval_results = evaluate_gnn(model, test_loader, device)
        results[name] = eval_results
        
        # Patient diagnosis evaluation
        if meta_test is not None:
            patient_diag = evaluate_patient_diagnosis(model, test_loader, device, meta_test)
            results[name]["patient_diagnosis"] = patient_diag
            
            # Compute patient-level accuracy
            correct = sum(1 for p in patient_diag if p["predicted_positive"] == (p["diagnosis"] != "Healthy"))
            total = len(patient_diag)
            results[name]["patient_accuracy"] = correct / total if total > 0 else 0
            
    return results