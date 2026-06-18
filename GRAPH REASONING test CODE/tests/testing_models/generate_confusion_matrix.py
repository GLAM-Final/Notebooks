"""
ImprovedRespiratoryGAT Confusion Matrix Generator
=================================================
Loads the trained model from testing_models/best_improved_30epochs_no_earlystop.pt,
runs inference on the test set, and generates comprehensive confusion matrices
with detailed metrics for both wheeze and crackle detection.
"""

import os, sys, pathlib, warnings, pickle, hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score,
    roc_auc_score, accuracy_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ── Paths ──
BASE = pathlib.Path(__file__).resolve().parent.parent
RESULTS = BASE / "testing_models" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
CACHE = BASE / "cache" / "wav2vec2_embeddings"
CKPT_PATH = BASE / "testing_models" / "best_improved_30epochs_no_earlystop.pt"
AUDIO = BASE / "ICBHI_final_database"
DIAG = AUDIO / "important" / "ICBHI_Challenge_diagnosis.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(BASE))
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, SAGEConv, BatchNorm


# ════════════════════════════════════════════════════════════════════════════
# MODEL: ImprovedRespiratoryGAT (matching model_comparisons/gnn_improved_model.py)
# ════════════════════════════════════════════════════════════════════════════

class ImprovedRespiratoryGAT(nn.Module):
    """Multi-head GAT + SAGE hybrid with residuals and attention pooling.
    Architecture matches model_comparisons/gnn_improved_model.py exactly."""

    def __init__(self, input_dim=768, hidden_dim=256, num_layers=4, num_heads=4, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                conv = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True, dropout=dropout)
            else:
                conv = SAGEConv(hidden_dim, hidden_dim)
            self.gat_layers.append(conv)
            self.norms.append(BatchNorm(hidden_dim))

        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, max(4, hidden_dim // 4)),
            nn.Tanh(),
            nn.Linear(max(4, hidden_dim // 4), 1),
        )
        self.wheeze_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.crackle_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.log_var_wheeze = nn.Parameter(torch.tensor(0.0))
        self.log_var_crackle = nn.Parameter(torch.tensor(0.0))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") else None
        x = self.input_proj(x)
        residuals = []
        for i, (conv, norm) in enumerate(zip(self.gat_layers, self.norms)):
            x_new = conv(x, edge_index)
            x_new = F.elu(x_new)
            x_new = norm(x_new)
            if i > 0 and i % 2 == 0:
                x_new = x_new + residuals[-1]
            x = F.dropout(x_new, p=self.dropout, training=self.training)
            residuals.append(x)

        if batch is not None:
            attn_scores = self.attention_pool(x).squeeze(-1)
            x_graph = []
            for b in torch.unique(batch):
                mask = batch == b
                scores = attn_scores[mask]
                weights = torch.softmax(scores, dim=0).unsqueeze(-1)
                x_graph.append((x[mask] * weights).sum(dim=0))
            x = torch.stack(x_graph, dim=0)
        else:
            attn_scores = self.attention_pool(x).squeeze(-1)
            weights = torch.softmax(attn_scores, dim=0).unsqueeze(-1)
            x = (x * weights).sum(dim=0, keepdim=True)

        w_logits = self.wheeze_head(x).squeeze(-1)
        c_logits = self.crackle_head(x).squeeze(-1)
        return w_logits, c_logits


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING (matching comprehensive_eval.py)
# ════════════════════════════════════════════════════════════════════════════

def temporal_edges(n):
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long)
    e = [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]
    return torch.tensor(e, dtype=torch.long).t().contiguous()


class FeatureCache:
    def __init__(self):
        self.cache_dir = CACHE
        self._model = None
        self._processor = None

    def _lazy_init(self):
        if self._model is None:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            self._processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base-960h", local_files_only=True
            )
            self._model = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base-960h", local_files_only=True
            )
            self._model.eval()

    def _cache_key(self, audio_path):
        mtime = os.path.getmtime(audio_path)
        return hashlib.md5(f"{audio_path}_{mtime}_0.5".encode()).hexdigest()

    def get_embeddings(self, audio_path):
        key = self._cache_key(audio_path)
        cpath = self.cache_dir / f"{key}.pkl"
        if cpath.exists():
            return pickle.load(open(cpath, "rb"))["embeddings"]
        self._lazy_init()
        import librosa
        y, _ = librosa.load(audio_path, sr=16000, mono=True)
        fl = int(0.5 * 16000)
        nf = len(y) // fl
        frames = [y[i * fl:(i + 1) * fl] for i in range(nf)]
        embs = []
        with torch.no_grad():
            for i in range(0, len(frames), 24):
                batch = frames[i:i + 24]
                inputs = self._processor(
                    batch, sampling_rate=16000, return_tensors="pt", padding=True
                )
                out = self._model(inputs.input_values)
                embs.append(out.last_hidden_state.mean(dim=1).cpu().numpy())
        emb = np.concatenate(embs, axis=0) if embs else np.zeros((0, 768), dtype=np.float32)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        with open(cpath, "wb") as f:
            pickle.dump(
                {"embeddings": emb.astype(np.float32), "audio_duration": len(y) / 16000, "num_frames": nf},
                f,
            )
        return emb


FEAT_CACHE = FeatureCache()


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, meta, anns, max_frames=20):
        self.meta = meta.reset_index(drop=True)
        self.anns = anns
        self.max_frames = max_frames

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        try:
            emb = FEAT_CACHE.get_embeddings(row["wav_path"])
        except Exception:
            return None
        if emb.shape[0] == 0:
            return None
        frames = min(self.max_frames, emb.shape[0])
        x = emb[:frames]
        if x.shape[0] < self.max_frames:
            x = np.concatenate([x, np.zeros((self.max_frames - x.shape[0], 768), dtype=np.float32)])
        a = self.anns.get(row["wav_path"])
        if a is None:
            return None
        w = np.zeros(self.max_frames, dtype=np.float32)
        c = np.zeros(self.max_frames, dtype=np.float32)
        for i in range(self.max_frames):
            fs = i * 0.5
            fe = fs + 0.5
            for _, r in a[(a["start"] < fe) & (a["end"] > fs)].iterrows():
                ov = max(0.0, min(fe, r["end"]) - max(fs, r["start"]))
                if ov / 0.5 >= 0.3:
                    if int(r["crackle"]) == 1:
                        c[i] = 1.0
                    if int(r["wheeze"]) == 1:
                        w[i] = 1.0
        return Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=temporal_edges(x.shape[0]),
            y_wheeze=torch.tensor(w, dtype=torch.float32),
            y_crackle=torch.tensor(c, dtype=torch.float32),
        )


def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    return Batch.from_data_list(batch) if batch else None


# ════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_predictions(model, loader, device):
    """Collect all predictions and ground truth from the test loader.
    
    The ImprovedRespiratoryGAT produces graph-level predictions (one per graph)
    via attention pooling, but the dataset stores per-node labels (20 frames per graph).
    We aggregate per-node labels to graph-level: a graph is positive if ANY frame is positive.
    """
    model.eval()
    all_w_probs, all_c_probs = [], []
    all_w_trues, all_c_trues = [], []

    for batch in loader:
        if batch is None:
            continue
        batch = batch.to(device)
        w_logits, c_logits = model(batch)

        # Graph-level predictions: shape [num_graphs]
        w_prob = torch.sigmoid(w_logits).cpu().numpy()
        c_prob = torch.sigmoid(c_logits).cpu().numpy()

        # Per-node labels: shape [num_nodes_total] = [num_graphs * frames_per_graph]
        w_true_nodes = batch.y_wheeze.cpu().numpy()
        c_true_nodes = batch.y_crackle.cpu().numpy()

        # Aggregate to graph-level: max over frames per graph (any positive frame → graph positive)
        batch_vector = batch.batch.cpu().numpy()
        num_graphs = int(batch_vector.max()) + 1
        for g in range(num_graphs):
            mask = batch_vector == g
            w_true_graph = int(w_true_nodes[mask].max() > 0.5)
            c_true_graph = int(c_true_nodes[mask].max() > 0.5)
            all_w_trues.append(w_true_graph)
            all_c_trues.append(c_true_graph)

        all_w_probs.extend(w_prob.tolist())
        all_c_probs.extend(c_prob.tolist())

    return (
        np.array(all_w_probs),
        np.array(all_c_probs),
        np.array(all_w_trues),
        np.array(all_c_trues),
    )


def compute_metrics(y_true, y_prob, threshold=0.5, task_name="Task"):
    """Compute comprehensive metrics from confusion matrix and probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_true.astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    balanced_acc = (recall + specificity) / 2.0

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = 0.0

    return {
        "task": task_name,
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_acc),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


def plot_confusion_matrix(cm, task_name, threshold, save_path):
    """Plot a single confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 16, "weight": "bold"},
    )
    plt.title(f"{task_name} Confusion Matrix\n(threshold = {threshold:.2f})", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_combined_confusion_matrices(cm_w, cm_c, threshold, save_path):
    """Plot both confusion matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Wheeze
    sns.heatmap(
        cm_w,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 14, "weight": "bold"},
        ax=axes[0],
    )
    axes[0].set_title(f"Wheeze Confusion Matrix\n(threshold = {threshold:.2f})", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("True Label", fontsize=11)
    axes[0].set_xlabel("Predicted Label", fontsize=11)

    # Crackle
    sns.heatmap(
        cm_c,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 14, "weight": "bold"},
        ax=axes[1],
    )
    axes[1].set_title(f"Crackle Confusion Matrix\n(threshold = {threshold:.2f})", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("True Label", fontsize=11)
    axes[1].set_xlabel("Predicted Label", fontsize=11)

    plt.suptitle(
        f"ImprovedRespiratoryGAT — Test Set Evaluation\n"
        f"Checkpoint: best_improved_30epochs_no_earlystop.pt",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_and_pr_curves(w_trues, w_probs, c_trues, c_probs, save_path):
    """Plot ROC and Precision-Recall curves for both tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    try:
        fpr_w, tpr_w, _ = roc_curve(w_trues, w_probs)
        auc_w = roc_auc_score(w_trues, w_probs)
        axes[0].plot(fpr_w, tpr_w, "b-", linewidth=2, label=f"Wheeze AUC = {auc_w:.3f}")
    except Exception:
        pass
    try:
        fpr_c, tpr_c, _ = roc_curve(c_trues, c_probs)
        auc_c = roc_auc_score(c_trues, c_probs)
        axes[0].plot(fpr_c, tpr_c, "r-", linewidth=2, label=f"Crackle AUC = {auc_c:.3f}")
    except Exception:
        pass
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.5)")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].set_title("ROC Curves", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # PR
    try:
        prec_w, rec_w, _ = precision_recall_curve(w_trues, w_probs)
        ap_w = average_precision_score(w_trues, w_probs)
        axes[1].plot(rec_w, prec_w, "b-", linewidth=2, label=f"Wheeze AP = {ap_w:.3f}")
    except Exception:
        pass
    try:
        prec_c, rec_c, _ = precision_recall_curve(c_trues, c_probs)
        ap_c = average_precision_score(c_trues, c_probs)
        axes[1].plot(rec_c, prec_c, "r-", linewidth=2, label=f"Crackle AP = {ap_c:.3f}")
    except Exception:
        pass
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("ImprovedRespiratoryGAT — ROC & PR Curves (Test Set)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("ImprovedRespiratoryGAT — Confusion Matrix Generation")
    print("=" * 70)
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Device: {DEVICE}")

    # ── 1. Load metadata and annotations ──
    print("\n[1/6] Loading metadata and annotations...")
    diag = pd.read_csv(DIAG, sep="\t", header=None, names=["pid", "diagnosis"])
    diag["pid"] = diag["pid"].astype(str)
    dm = dict(zip(diag["pid"], diag["diagnosis"]))

    wav_map, ann_map = {}, {}
    for f in os.listdir(AUDIO):
        fp = os.path.join(AUDIO, f)
        if not os.path.isfile(fp):
            continue
        b, e = os.path.splitext(f)
        if e.lower() == ".wav":
            wav_map[b] = fp
        elif e.lower() == ".txt":
            ann_map[b] = fp

    rows = []
    for b in sorted(wav_map):
        if b not in ann_map:
            continue
        pid = b.split("_")[0]
        rows.append({
            "file_id": b,
            "patient_id": pid,
            "diagnosis": dm.get(pid, "Unknown"),
            "wav_path": wav_map[b],
            "ann_path": ann_map[b],
        })
    meta = pd.DataFrame(rows)
    print(f"  Total files: {len(meta)}")

    # ── 2. Patient-wise train/val/test split ──
    print("\n[2/6] Splitting data (patient-wise 70/15/15)...")
    pats = meta["patient_id"].unique()
    tp, tmp = train_test_split(pats, test_size=0.3, random_state=42)
    vp, tp2 = train_test_split(tmp, test_size=0.5, random_state=42)
    test_m = meta[meta["patient_id"].isin(tp2)].reset_index(drop=True)
    print(f"  Test patients: {len(tp2)}, Test files: {len(test_m)}")

    # ── 3. Load annotations ──
    print("\n[3/6] Loading annotations...")
    anns = {}
    for _, row in meta.iterrows():
        anns[row["wav_path"]] = pd.read_csv(
            row["ann_path"], sep="\t", header=None,
            names=["start", "end", "crackle", "wheeze"]
        )

    # ── 4. Build test dataset and loader ──
    print("\n[4/6] Building test dataset (loading cached embeddings)...")
    ds_test_raw = TestDataset(test_m, anns)
    valid_idx = [i for i in range(len(ds_test_raw)) if ds_test_raw[i] is not None]
    ds_test = torch.utils.data.Subset(ds_test_raw, valid_idx)
    print(f"  Valid test samples: {len(ds_test)}")
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # ── 5. Load model ──
    print("\n[5/6] Loading ImprovedRespiratoryGAT model...")
    model = ImprovedRespiratoryGAT(
        input_dim=768, hidden_dim=256, num_layers=4, num_heads=4, dropout=0.4
    )
    sd = torch.load(str(CKPT_PATH), map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    mk = set(model.state_dict().keys())
    sd = {k: v for k, v in sd.items() if k in mk}
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    model.eval()
    print(f"  Loaded checkpoint: {CKPT_PATH.name}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 6. Collect predictions ──
    print("\n[6/6] Running inference on test set...")
    w_probs, c_probs, w_trues, c_trues = collect_predictions(model, test_loader, DEVICE)
    print(f"  Total predictions: {len(w_probs)}")
    print(f"  Wheeze positives: {int(w_trues.sum())}/{len(w_trues)} ({w_trues.mean()*100:.1f}%)")
    print(f"  Crackle positives: {int(c_trues.sum())}/{len(c_trues)} ({c_trues.mean()*100:.1f}%)")

    # ── Compute metrics at threshold 0.5 ──
    threshold = 0.5
    w_metrics = compute_metrics(w_trues, w_probs, threshold=threshold, task_name="Wheeze")
    c_metrics = compute_metrics(c_trues, c_probs, threshold=threshold, task_name="Crackle")

    # ── Print results ──
    print("\n" + "=" * 70)
    print(f"CONFUSION MATRIX RESULTS (threshold = {threshold})")
    print("=" * 70)

    for metrics in [w_metrics, c_metrics]:
        print(f"\n{'─' * 50}")
        print(f"  {metrics['task']} Detection")
        print(f"{'─' * 50}")
        print(f"  Confusion Matrix:")
        print(f"    TN = {metrics['tn']:>6,}    FP = {metrics['fp']:>6,}")
        print(f"    FN = {metrics['fn']:>6,}    TP = {metrics['tp']:>6,}")
        print(f"  Metrics:")
        print(f"    Precision:        {metrics['precision']:.4f}")
        print(f"    Recall (Sens.):   {metrics['recall']:.4f}")
        print(f"    Specificity:      {metrics['specificity']:.4f}")
        print(f"    F1 Score:         {metrics['f1']:.4f}")
        print(f"    Accuracy:         {metrics['accuracy']:.4f}")
        print(f"    Balanced Acc.:    {metrics['balanced_accuracy']:.4f}")
        print(f"    FPR:              {metrics['fpr']:.4f}")
        print(f"    FNR:              {metrics['fnr']:.4f}")
        print(f"    ROC-AUC:          {metrics['roc_auc']:.4f}")
        print(f"    PR-AUC:           {metrics['pr_auc']:.4f}")

    # ── Generate plots ──
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    cm_w = confusion_matrix(w_trues.astype(int), (w_probs >= threshold).astype(int), labels=[0, 1])
    cm_c = confusion_matrix(c_trues.astype(int), (c_probs >= threshold).astype(int), labels=[0, 1])

    plot_combined_confusion_matrices(cm_w, cm_c, threshold, RESULTS / "improved_gat_confusion_matrix.png")
    plot_confusion_matrix(cm_w, "Wheeze", threshold, RESULTS / "improved_gat_wheeze_confusion.png")
    plot_confusion_matrix(cm_c, "Crackle", threshold, RESULTS / "improved_gat_crackle_confusion.png")
    plot_roc_and_pr_curves(w_trues, w_probs, c_trues, c_probs, RESULTS / "improved_gat_roc_pr.png")

    # ── Debug: print prediction distributions ──
    print(f"\n  Wheeze prediction stats: min={w_probs.min():.4f}, max={w_probs.max():.4f}, "
          f"mean={w_probs.mean():.4f}, median={np.median(w_probs):.4f}")
    print(f"  Crackle prediction stats: min={c_probs.min():.4f}, max={c_probs.max():.4f}, "
          f"mean={c_probs.mean():.4f}, median={np.median(c_probs):.4f}")

    # ── Save summary report ──
    report_path = RESULTS / "improved_gat_confusion_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("ImprovedRespiratoryGAT - Confusion Matrix Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint: {CKPT_PATH}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Test samples: {len(w_probs)}\n")
        f.write(f"Test patients: {len(tp2)}\n\n")
        f.write(f"Wheeze pred stats: min={w_probs.min():.4f}, max={w_probs.max():.4f}, "
                f"mean={w_probs.mean():.4f}\n")
        f.write(f"Crackle pred stats: min={c_probs.min():.4f}, max={c_probs.max():.4f}, "
                f"mean={c_probs.mean():.4f}\n\n")

        for metrics in [w_metrics, c_metrics]:
            f.write(f"\n{'-' * 50}\n")
            f.write(f"  {metrics['task']} Detection\n")
            f.write(f"{'-' * 50}\n")
            f.write(f"  Confusion Matrix:\n")
            f.write(f"    TN = {metrics['tn']:>6,}    FP = {metrics['fp']:>6,}\n")
            f.write(f"    FN = {metrics['fn']:>6,}    TP = {metrics['tp']:>6,}\n")
            f.write(f"  Metrics:\n")
            for key in ["precision", "recall", "specificity", "f1", "accuracy",
                        "balanced_accuracy", "fpr", "fnr", "roc_auc", "pr_auc"]:
                f.write(f"    {key:20s}: {metrics[key]:.4f}\n")

    print(f"\n  Report saved: {report_path}")
    print("\nDone!")

    return w_metrics, c_metrics


if __name__ == "__main__":
    main()