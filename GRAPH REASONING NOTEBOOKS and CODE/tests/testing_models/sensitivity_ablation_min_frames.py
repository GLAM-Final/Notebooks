"""
=============================================================================
SENSITIVITY ABLATION: Temporal Postprocessing Window Size
=============================================================================
Empirically justifies the choice of temporal smoothing window used in
chunk-level temporal postprocessing. Applies median filtering (as described
in Section V of the manuscript) with window sizes {1, 2, 3, 5, 7, 10} to
frame-level predictions, then evaluates chunk-level Precision, Recall, F1,
and Specificity for both wheeze and crackle detection.

This directly addresses the reviewer critique on Layer 5 Robustness:
the "magic number" heuristic for temporal postprocessing is validated
through a systematic sensitivity ablation.

Runs for both BaselineGAT and GraphSAGE models (both produce node-level
per-frame predictions).

Generates:
  - temporal_postprocessing_ablation.csv  (raw data for all models)
  - temporal_postprocessing_ablation.png  (sensitivity ablation graph)
=============================================================================
"""

import os, sys, pathlib, warnings, pickle, hashlib, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

BASE = pathlib.Path(__file__).resolve().parent.parent
RESULTS = BASE / "testing_models" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
CACHE_DIR = BASE / "cache" / "wav2vec2_embeddings"
AUDIO_FOLDER = BASE / "ICBHI_final_database"
DIAG_FILE = AUDIO_FOLDER / "important" / "ICBHI_Challenge_diagnosis.txt"

CKPT_BASELINE = BASE / "testing_models" / "respiratory_gnn_model_cpu_full.pth"
CKPT_GRAPHSAGE = BASE / "testing_models" / "best_respiratory_graphsage_temp_scaled.pth"

TARGET_SR = 16000
FRAME_SECONDS = 0.5
CHUNK_SECONDS = 5.0
FRAMES_PER_CHUNK = int(CHUNK_SECONDS / FRAME_SECONDS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ========================================================
# MODEL DEFINITIONS
# ========================================================

class BaselineGAT(nn.Module):
    """3 GATConv layers, node-level outputs (one prediction per frame)."""
    def __init__(self, input_dim=768, hidden_dim=256, num_heads=4,
                 num_layers=3, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(
                hidden_dim, hidden_dim // num_heads,
                heads=num_heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.wheeze_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
        self.crackle_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = F.elu(x_new)
            x_new = norm(x_new)
            x = x + x_new
            x = F.dropout(x, p=0.1, training=self.training)
        w = self.wheeze_head(x).squeeze(-1)
        c = self.crackle_head(x).squeeze(-1)
        return w, c


class GraphSAGEModel(nn.Module):
    """3 SAGEConv layers, node-level outputs (one prediction per frame)."""
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=3, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.wheeze_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
        self.crackle_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = F.relu(x_new)
            x_new = norm(x_new)
            x = x + x_new
            x = F.dropout(x, p=self.dropout, training=self.training)
        w = self.wheeze_head(x).squeeze(-1)
        c = self.crackle_head(x).squeeze(-1)
        return w, c


# ========================================================
# DATA HELPERS (reused from thorough_evaluation.py)
# ========================================================

def load_diagnosis_file(path):
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["patient_id", "diagnosis"])
    df["patient_id"] = df["patient_id"].astype(str)
    return df


def scan_icbhi_pairs(audio_root):
    wav_map, ann_map = {}, {}
    for fname in os.listdir(audio_root):
        fpath = os.path.join(audio_root, fname)
        if not os.path.isfile(fpath):
            continue
        base, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext == ".wav":
            wav_map[base] = fpath
        elif ext == ".txt":
            ann_map[base] = fpath
    paired = [(b, wav_map[b], ann_map[b])
              for b in sorted(wav_map) if b in ann_map]
    return paired


def extract_patient_id(base):
    return base.split("_")[0]


def build_metadata(paired, diag_df):
    diag_map = dict(zip(diag_df["patient_id"], diag_df["diagnosis"]))
    rows = []
    for base, wav_path, ann_path in paired:
        rows.append({
            "file_id": base,
            "patient_id": extract_patient_id(base),
            "diagnosis": diag_map.get(extract_patient_id(base), "Unknown"),
            "wav_path": wav_path, "ann_path": ann_path
        })
    return pd.DataFrame(rows)


def patient_wise_split(meta_df, seed=42):
    patients = meta_df["patient_id"].unique()
    train_p, temp_p = train_test_split(
        patients, test_size=0.30, random_state=seed)
    val_p, test_p = train_test_split(
        temp_p, test_size=0.50, random_state=seed)
    test_df = meta_df[
        meta_df["patient_id"].isin(test_p)].reset_index(drop=True)
    return test_df


# ========================================================
# FEATURE CACHE
# ========================================================

class Wav2Vec2FeatureCache:
    def __init__(self, cache_dir=CACHE_DIR, sr=TARGET_SR,
                 frame_seconds=FRAME_SECONDS):
        self.cache_dir = str(cache_dir)
        self.sr = sr
        self.frame_seconds = frame_seconds
        self.frame_samples = int(sr * frame_seconds)

    def _cache_key(self, audio_path):
        mtime = os.path.getmtime(audio_path)
        h = hashlib.md5(
            f"{audio_path}_{mtime}_{self.frame_seconds}".encode()
        ).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.pkl")

    def compute_and_cache_file(self, audio_path):
        cache_path = self._cache_key(audio_path)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        raise FileNotFoundError(f"Cache miss: {audio_path}")

    def get_full_embeddings(self, audio_path):
        cache = self.compute_and_cache_file(audio_path)
        return cache["embeddings"], cache["audio_duration"]


# ========================================================
# GRAPH CONSTRUCTION
# ========================================================

def build_simple_temporal_edges(num_nodes):
    """Chronological chaining: each frame i connects to i+1 bidirectionally.
    This defines the adjacency matrix A where A_{i,i+1} = A_{i+1,i} = 1."""
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long)
    edges = [[i, i + 1] for i in range(num_nodes - 1)]
    edges += [[i + 1, i] for i in range(num_nodes - 1)]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def get_chunk_labels(ann_df, chunk_start):
    """Get frame-level labels for a chunk."""
    c_lbl = np.zeros(FRAMES_PER_CHUNK, dtype=np.float32)
    w_lbl = np.zeros(FRAMES_PER_CHUNK, dtype=np.float32)
    for i in range(FRAMES_PER_CHUNK):
        fs = chunk_start + i * FRAME_SECONDS
        fe = fs + FRAME_SECONDS
        overlaps = ann_df[
            (ann_df["start"] < fe) & (ann_df["end"] > fs)]
        for _, row in overlaps.iterrows():
            overlap = max(0.0, min(fe, row["end"]) - max(fs, row["start"]))
            if overlap / FRAME_SECONDS >= 0.3:
                if int(row["crackle"]) == 1:
                    c_lbl[i] = 1.0
                if int(row["wheeze"]) == 1:
                    w_lbl[i] = 1.0
    return w_lbl, c_lbl


# ========================================================
# MAIN
# ========================================================

def run_model_inference(model, device, test_meta, cache):
    """Run inference on all test files. Returns frame-level probabilities,
    labels, and chunk IDs."""
    all_probs_w, all_probs_c = [], []
    all_true_w, all_true_c = [], []
    all_cids = []
    chunk_offset = 0
    processed, errors = 0, 0

    for _, row in test_meta.iterrows():
        wav_path = row["wav_path"]
        ann_path = row["ann_path"]
        try:
            embeddings, duration = cache.get_full_embeddings(wav_path)
            ann_df = pd.read_csv(
                ann_path, sep="\t", header=None,
                names=["start", "end", "crackle", "wheeze"])
        except Exception:
            errors += 1
            continue

        num_chunks = max(1, int(duration / CHUNK_SECONDS))
        for ci in range(num_chunks):
            start_sec = ci * CHUNK_SECONDS
            if start_sec + CHUNK_SECONDS > duration and ci > 0:
                break
            start_frame = int(start_sec / FRAME_SECONDS)
            end_frame = min(
                start_frame + FRAMES_PER_CHUNK, embeddings.shape[0])
            if start_frame >= embeddings.shape[0]:
                continue
            chunk_emb = embeddings[start_frame:end_frame]
            n = chunk_emb.shape[0]
            if n < FRAMES_PER_CHUNK:
                pad = np.zeros(
                    (FRAMES_PER_CHUNK - n, 768), dtype=np.float32)
                chunk_emb = np.concatenate([chunk_emb, pad], axis=0)
                n = FRAMES_PER_CHUNK

            edge_index = build_simple_temporal_edges(n)
            from torch_geometric.data import Data
            data = Data(
                x=torch.tensor(chunk_emb, dtype=torch.float32),
                edge_index=edge_index,
            ).to(device)

            with torch.no_grad():
                w_logits, c_logits = model(data)
                w_prob = torch.sigmoid(w_logits).cpu().numpy().ravel()
                c_prob = torch.sigmoid(c_logits).cpu().numpy().ravel()

            w_true, c_true = get_chunk_labels(ann_df, start_sec)

            all_probs_w.extend(w_prob[:len(w_true)])
            all_probs_c.extend(c_prob[:len(c_true)])
            all_true_w.extend(w_true[:len(w_prob)])
            all_true_c.extend(c_true[:len(c_prob)])
            all_cids.extend([chunk_offset] * len(w_true[:len(w_prob)]))
            chunk_offset += 1

        processed += 1
        if processed % 50 == 0:
            print(f"    Processed {processed}/{len(test_meta)} "
                  f"({chunk_offset} chunks)...")

    return (np.array(all_probs_w), np.array(all_probs_c),
            np.array(all_true_w), np.array(all_true_c),
            np.array(all_cids), processed, errors)


def sweep_window_sizes(probs_w, probs_c, true_w, true_c, cids,
                       model_name, results_rows):
    """Sweep median filter windows and aggregate to chunk level."""
    window_sizes = [1, 2, 3, 5, 7, 10]
    frame_threshold = 0.5

    for ws in window_sizes:
        for task, probs, true in [
            ("wheeze", probs_w, true_w),
            ("crackle", probs_c, true_c),
        ]:
            smoothed = median_filter(probs, size=ws)
            chunk_true, chunk_pred = [], []
            for cid in np.unique(cids):
                mask = cids == cid
                chunk_true.append(int(np.max(true[mask]) > 0.5))
                chunk_pred.append(int(
                    np.sum(smoothed[mask] >= frame_threshold) >= 1))

            ct = np.array(chunk_true)
            cp = np.array(chunk_pred)

            eps = 1e-9
            tp = int(np.sum((cp == 1) & (ct == 1)))
            fp = int(np.sum((cp == 1) & (ct == 0)))
            tn = int(np.sum((cp == 0) & (ct == 0)))
            fn = int(np.sum((cp == 0) & (ct == 1)))
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            spec = tn / (tn + fp + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)

            results_rows.append({
                "model": model_name, "window_size": ws, "task": task,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "specificity": round(spec, 4),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "chunk_total": int(len(ct)),
                "chunk_positive": int(ct.sum()),
            })
            print(f"  [{model_name}] ws={ws:2d} | {task:8s} | "
                  f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
                  f"Spec={spec:.3f}")


def load_checkpoint(model, ckpt_path):
    """Load a checkpoint into a model, handling various formats."""
    sd = torch.load(str(ckpt_path), map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    elif "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    mk = set(model.state_dict().keys())
    sd = {k: v for k, v in sd.items() if k in mk}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def plot_ablation_results(df, plot_path):
    """Generate two-panel sensitivity ablation graph."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

    for ax_idx, (ax, task) in enumerate([(axes[0], "wheeze"),
                                          (axes[1], "crackle")]):
        colors = {"BaselineGAT": ("#E74C3C", "#3498DB", "#2ECC71"),
                  "GraphSAGE": ("#E67E22", "#9B59B6", "#1ABC9C")}
        window_sizes = [1, 2, 3, 5, 7, 10]

        for model_name, (c_p, c_r, c_f) in colors.items():
            sub = df[(df["task"] == task) & (df["model"] == model_name)]
            if sub.empty:
                continue
            sub = sub.sort_values("window_size")
            x = sub["window_size"].values

            ls = "--" if model_name == "GraphSAGE" else "-"
            ax.plot(x, sub["precision"].values, marker="o", linewidth=2,
                    markersize=7, color=c_p, label=f"{model_name} P",
                    linestyle=ls, zorder=5)
            ax.plot(x, sub["recall"].values, marker="s", linewidth=2,
                    markersize=7, color=c_r, label=f"{model_name} R",
                    linestyle=ls, zorder=5)
            ax.plot(x, sub["f1"].values, marker="^", linewidth=1.8,
                    markersize=6, color=c_f, label=f"{model_name} F1",
                    linestyle=":", zorder=4)

        # Chosen annotation
        ax.axvline(x=3, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.annotate("Chosen\n(window=3)", xy=(3, 0.45), fontsize=9,
                    ha="center", va="center", color="gray",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="lightyellow",
                              edgecolor="gray", alpha=0.8))

        ax.set_xlabel("Median Filter Window Size (frames)", fontsize=11)
        ax.set_xticks(window_sizes)
        ax.set_xlim(0.5, 11)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{task.capitalize()} Detection", fontsize=13,
                     fontweight="bold")
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Metric Score", fontsize=11)
    fig.suptitle(
        "Sensitivity Ablation: Temporal Postprocessing Window Size\n"
        "(Median filter on frame-level probs \u2192 chunk-level aggregation)",
        fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 70)
    print("SENSITIVITY ABLATION: Temporal Postprocessing Window Size")
    print("Runs for BaselineGAT and GraphSAGE")
    print("=" * 70)

    # 1. Load metadata
    print("\n[1/5] Loading metadata...")
    diag_df = load_diagnosis_file(str(DIAG_FILE))
    paired = scan_icbhi_pairs(str(AUDIO_FOLDER))
    meta = build_metadata(paired, diag_df)
    test_meta = patient_wise_split(meta)
    print(f"  Test files: {len(test_meta)}")
    cache = Wav2Vec2FeatureCache()

    # 2. Model configs
    model_configs = [
        ("BaselineGAT", BaselineGAT(), CKPT_BASELINE),
        ("GraphSAGE", GraphSAGEModel(), CKPT_GRAPHSAGE),
    ]

    all_results = []

    for model_name, model, ckpt_path in model_configs:
        print(f"\n{'=' * 70}")
        print(f"Processing model: {model_name}")
        print(f"{'=' * 70}")

        # Load checkpoint
        print(f"\n[load] Loading {ckpt_path.name}...")
        model = load_checkpoint(model, ckpt_path)
        model.to(DEVICE)
        print(f"  Loaded successfully")

        # Run inference
        print(f"\n[infer] Running inference...")
        (pw, pc, tw, tc, cids,
         processed, errors) = run_model_inference(model, DEVICE,
                                                   test_meta, cache)
        print(f"  {processed} files, {errors} errors, "
              f"{len(np.unique(cids))} chunks, {len(pw)} frames")
        print(f"  Wheeze+ frames: {int(tw.sum())} ({100*tw.mean():.1f}%)")
        print(f"  Crackle+ frames: {int(tc.sum())} ({100*tc.mean():.1f}%)")

        # Sweep window sizes
        print(f"\n[sweep] Sweeping window sizes...")
        sweep_window_sizes(pw, pc, tw, tc, cids, model_name, all_results)

    # 4. Save CSV
    df = pd.DataFrame(all_results)
    csv_path = RESULTS / "temporal_postprocessing_ablation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # 5. Generate graph
    print("\n[plot] Generating sensitivity ablation graph...")
    plot_path = RESULTS / "temporal_postprocessing_ablation.png"
    plot_ablation_results(df, plot_path)
    print(f"  Saved: {plot_path}")

    # 6. Summary
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    for model_name in ["BaselineGAT", "GraphSAGE"]:
        for task in ["wheeze", "crackle"]:
            sub = df[(df["model"] == model_name) & (df["task"] == task)]
            if sub.empty:
                continue
            best = sub.loc[sub["f1"].idxmax()]
            chosen = sub[sub["window_size"] == 3]
            print(f"\n  {model_name} | {task.upper()}:")
            print(f"    Best F1: ws={int(best['window_size'])} "
                  f"(F1={best['f1']:.3f}, P={best['precision']:.3f}, "
                  f"R={best['recall']:.3f})")
            if not chosen.empty:
                c = chosen.iloc[0]
                print(f"    Chosen(3): F1={c['f1']:.3f}, "
                      f"P={c['precision']:.3f}, R={c['recall']:.3f}")
            f1s = sub["f1"].values
            print(f"    F1 range: {f1s.min():.3f} \u2013 {f1s.max():.3f} "
                  f"(\u0394={f1s.max() - f1s.min():.3f})")

    print(f"\nFiles saved to: {RESULTS}")
    print(f"  - temporal_postprocessing_ablation.csv")
    print(f"  - temporal_postprocessing_ablation.png")


if __name__ == "__main__":
    main()