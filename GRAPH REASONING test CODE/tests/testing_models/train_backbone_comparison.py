"""
=============================================================================
MULTI-BACKBONE GNN TRAINING & EVALUATION
=============================================================================
For each acoustic backbone, trains all three GNN architectures and evaluates
them on the held-out test set. Produces a comparison table.

Backbones: wav2vec2_base, wav2vec2_large, hubert_base, hubert_large
GNN Models: Baseline GAT, GraphSAGE, Improved GAT

Results saved to testing_models/results/backbone_comparison/
=============================================================================
"""

import os, sys, pathlib, pickle, hashlib, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, SAGEConv, BatchNorm

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"
RESULTS_DIR = BASE_DIR / "testing_models" / "results" / "backbone_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DIAGNOSIS_FILE = BASE_DIR / "ICBHI_final_database" / "important" / "ICBHI_Challenge_diagnosis.txt"
AUDIO_FOLDER = BASE_DIR / "ICBHI_final_database"

SR = 16000
FRAME_SECONDS = 0.5
FRAME_LEN = int(SR * FRAME_SECONDS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING FROM CACHE
# ════════════════════════════════════════════════════════════════════════════

def load_features_from_cache(backbone_name):
    """Load cached features and build metadata."""
    cache_dir = CACHE_DIR / f"features_{backbone_name}"
    meta = pd.read_csv(cache_dir / "metadata.csv")
    meta["patient_id"] = meta["patient_id"].astype(str)

    # Load embeddings
    features = {}
    for i, row in meta.iterrows():
        # Find the pkl file for this wav_path
        wav_path = row["wav_path"]
        mtime = os.path.getmtime(wav_path)
        key = hashlib.md5(f"{wav_path}_{backbone_name}_{mtime}".encode()).hexdigest()
        cpath = cache_dir / f"{key}.pkl"
        if cpath.exists():
            with open(cpath, "rb") as f:
                features[wav_path] = pickle.load(f)

    print(f"  Loaded {len(features)} cached feature files for {backbone_name}")
    return meta, features, cache_dir


def patient_wise_split(meta, seed=42):
    patients = meta["patient_id"].unique()
    train_p, temp_p = train_test_split(patients, test_size=0.30, random_state=seed)
    val_p, test_p = train_test_split(temp_p, test_size=0.50, random_state=seed)
    return (
        meta[meta["patient_id"].isin(train_p)].reset_index(drop=True),
        meta[meta["patient_id"].isin(val_p)].reset_index(drop=True),
        meta[meta["patient_id"].isin(test_p)].reset_index(drop=True),
    )


# ════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ════════════════════════════════════════════════════════════════════════════

def build_simple_temporal_edges(n):
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long)
    edges = [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_similarity_edges(x_np, k=4, temporal_hops=2):
    x = torch.tensor(x_np, dtype=torch.float32)
    x = F.normalize(x, dim=1)
    sim = x @ x.T
    n = sim.shape[0]
    edges = []
    for i in range(n):
        _, idx = torch.topk(sim[i], k=min(int(k) + 1, n))
        for j in idx.tolist():
            if i != j:
                edges.extend([[i, j], [j, i]])
    if int(temporal_hops) > 0:
        te = build_simple_temporal_edges(n)
        if te.shape[1] > 0:
            edges.extend(te.t().tolist())
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.unique(torch.tensor(edges, dtype=torch.long).t().contiguous(), dim=1)


# ════════════════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════════════════

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, meta_df, features, edge_type="temporal", chunk_seconds=10.0,
                 frame_seconds=0.5, k_edges=4, temporal_hops=2):
        self.meta = meta_df.reset_index(drop=True)
        self.features = features
        self.edge_type = edge_type
        self.chunk_seconds = chunk_seconds
        self.frame_seconds = frame_seconds
        self.frames_per_chunk = int(chunk_seconds / frame_seconds)
        self.k_edges = k_edges
        self.temporal_hops = temporal_hops

        self.annotations = {}
        for _, row in self.meta.iterrows():
            ann = pd.read_csv(row["ann_path"], sep="\t", header=None,
                              names=["start", "end", "crackle", "wheeze"])
            self.annotations[row["wav_path"]] = ann

    def __len__(self):
        return len(self.meta)

    def _labels_for_chunk(self, ann_df, chunk_start):
        c_lbl = np.zeros(self.frames_per_chunk, dtype=np.float32)
        w_lbl = np.zeros(self.frames_per_chunk, dtype=np.float32)
        for i in range(self.frames_per_chunk):
            fs = chunk_start + i * self.frame_seconds
            fe = fs + self.frame_seconds
            overlaps = ann_df[(ann_df["start"] < fe) & (ann_df["end"] > fs)]
            for _, row in overlaps.iterrows():
                overlap = max(0.0, min(fe, row["end"]) - max(fs, row["start"]))
                if overlap / self.frame_seconds >= 0.3:
                    if int(row["crackle"]) == 1:
                        c_lbl[i] = 1.0
                    if int(row["wheeze"]) == 1:
                        w_lbl[i] = 1.0
        return c_lbl, w_lbl

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        wav_path = row["wav_path"]
        if wav_path not in self.features:
            return None
        cache = self.features[wav_path]
        emb = cache["embeddings"]
        duration = cache["audio_duration"]
        start_sec = 0.0
        if duration > self.chunk_seconds:
            start_sec = (duration - self.chunk_seconds) / 2.0

        start_frame = int(start_sec / self.frame_seconds)
        end_frame = min(start_frame + self.frames_per_chunk, emb.shape[0])
        if start_frame >= emb.shape[0]:
            x_np = np.zeros((self.frames_per_chunk, emb.shape[1]), dtype=np.float32)
        else:
            x_np = emb[start_frame:end_frame]
            if x_np.shape[0] < self.frames_per_chunk:
                pad = np.zeros((self.frames_per_chunk - x_np.shape[0], emb.shape[1]), dtype=np.float32)
                x_np = np.concatenate([x_np, pad], axis=0)

        ann_df = self.annotations[wav_path]
        c_lbl, w_lbl = self._labels_for_chunk(ann_df, start_sec)

        if self.edge_type == "similarity":
            edge_index = build_similarity_edges(x_np, k=self.k_edges, temporal_hops=self.temporal_hops)
        else:
            edge_index = build_simple_temporal_edges(x_np.shape[0])

        data = Data(
            x=torch.tensor(x_np, dtype=torch.float32),
            edge_index=edge_index,
            y_wheeze=torch.tensor(w_lbl, dtype=torch.float32),
            y_crackle=torch.tensor(c_lbl, dtype=torch.float32),
        )
        data.file_id = row["file_id"]
        data.patient_id = row["patient_id"]
        return data


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


# ════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

def make_baseline_gat(input_dim):
    class BaselineGAT(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, 256)
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(3):
                self.convs.append(GATConv(256, 64, heads=4, dropout=0.3))
                self.norms.append(nn.LayerNorm(256))
            self.wheeze_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
            self.crackle_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
        def forward(self, data):
            x, ei = data.x, data.edge_index
            x = self.input_proj(x)
            for conv, norm in zip(self.convs, self.norms):
                x = norm(F.elu(conv(x, ei))) + x
                x = F.dropout(x, p=0.1, training=self.training)
            return self.wheeze_head(x).squeeze(-1), self.crackle_head(x).squeeze(-1)
    return BaselineGAT()


def make_graphsage(input_dim):
    class GraphSAGE(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, 256)
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(3):
                self.convs.append(SAGEConv(256, 256))
                self.norms.append(nn.BatchNorm1d(256))
            self.wheeze_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1))
            self.crackle_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1))
        def forward(self, data):
            x, ei = data.x, data.edge_index
            x = self.input_proj(x)
            for conv, norm in zip(self.convs, self.norms):
                x = norm(F.relu(conv(x, ei))) + x
                x = F.dropout(x, p=0.5, training=self.training)
            return self.wheeze_head(x).squeeze(-1), self.crackle_head(x).squeeze(-1)
    return GraphSAGE()


def make_improved_gat(input_dim):
    class ImprovedGAT(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Sequential(nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.4))
            self.gat_layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            for i in range(4):
                if i % 2 == 0:
                    self.gat_layers.append(GATConv(256, 64, heads=4, concat=True, dropout=0.4))
                else:
                    self.gat_layers.append(SAGEConv(256, 256))
                self.norms.append(BatchNorm(256))
            self.attn_pool = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1))
            self.wheeze_head = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.4), nn.Linear(128, 1))
            self.crackle_head = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.4), nn.Linear(128, 1))
            self.dropout = 0.4
        def forward(self, data):
            x, ei = data.x, data.edge_index
            batch = data.batch if hasattr(data, "batch") else None
            x = self.input_proj(x)
            res = []
            for i, (conv, norm) in enumerate(zip(self.gat_layers, self.norms)):
                xn = norm(F.elu(conv(x, ei)))
                if i > 0 and i % 2 == 0:
                    xn = xn + res[-1]
                x = F.dropout(xn, p=self.dropout, training=self.training)
                res.append(x)
            if batch is not None:
                attn = self.attn_pool(x).squeeze(-1)
                xg = []
                for b in torch.unique(batch):
                    m = batch == b
                    w = torch.softmax(attn[m], dim=0).unsqueeze(-1)
                    xg.append((x[m] * w).sum(dim=0))
                x = torch.stack(xg, dim=0)
            else:
                attn = self.attn_pool(x).squeeze(-1)
                w = torch.softmax(attn, dim=0).unsqueeze(-1)
                x = (x * w).sum(dim=0, keepdim=True)
            return self.wheeze_head(x).squeeze(-1), self.crackle_head(x).squeeze(-1)
    return ImprovedGAT()


# ════════════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════════════

def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()


@torch.no_grad()
def evaluate(model, loader, device, graph_level=False):
    model.eval()
    w_p, c_p, w_t, c_t = [], [], [], []
    for batch in loader:
        if batch is None:
            continue
        batch = batch.to(device)
        wl, cl = model(batch)
        w_prob = torch.sigmoid(wl).cpu().numpy()
        c_prob = torch.sigmoid(cl).cpu().numpy()
        w_l = batch.y_wheeze.cpu().numpy()
        c_l = batch.y_crackle.cpu().numpy()

        if graph_level or (wl.shape[0] != w_l.shape[0] if w_l.ndim > 1 else wl.shape[0] < w_l.shape[0]):
            # Graph-level: aggregate labels
            if hasattr(batch, "batch") and batch.batch is not None:
                local = batch.batch.cpu().numpy()
                for g in np.unique(local):
                    m = local == g
                    w_p.append([float(np.max(w_prob[local == g]))])
                    c_p.append([float(np.max(c_prob[local == g]))])
                    w_t.append([float(np.max(w_l[local == g])) if w_l.ndim > 1 else float(np.max(w_l[m]))])
                    c_t.append([float(np.max(c_l[local == g])) if c_l.ndim > 1 else float(np.max(c_l[m]))])
            else:
                w_p.append(w_prob)
                c_p.append(c_prob)
                w_t.append(w_l)
                c_t.append(c_l)
        else:
            w_p.append(w_prob.ravel())
            c_p.append(c_prob.ravel())
            w_t.append(w_l.ravel())
            c_c = c_l.ravel()
            c_p[-1] = c_prob.ravel()
            c_t.append(c_l.ravel())

    w_p = np.concatenate(w_p)
    c_p = np.concatenate(c_p)
    w_t = np.concatenate(w_t)
    c_t = np.concatenate(c_t)

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


def train_one(model, train_loader, val_loader, device, epochs=30, lr=5e-4, wd=1e-4, graph_level=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_val_f1 = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batch = 0
        for batch in train_loader:
            if batch is None:
                continue
            batch = batch.to(device)
            wl, cl = model(batch)
            w_target = batch.y_wheeze
            c_target = batch.y_crackle

            if graph_level and hasattr(batch, "batch") and batch.batch is not None:
                # Graph-level: aggregate to per-graph labels
                local = batch.batch.cpu().numpy()
                w_graph, c_graph = [], []
                for g in np.unique(local):
                    m = local == g
                    if w_target.dim() > 1:
                        w_graph.append(float(w_target[m].max()))
                        c_graph.append(float(c_target[m].max()))
                    else:
                        w_graph.append(float(w_target[m].max()))
                        c_graph.append(float(c_target[m].max()))
                w_target = torch.tensor(w_graph, device=device)
                c_target = torch.tensor(c_graph, device=device)
                wl = wl.squeeze()
                cl = cl.squeeze()

            loss_w = focal_loss(wl, w_target)
            loss_c = focal_loss(cl, c_target)
            loss = loss_w + loss_c

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        scheduler.step()

        # Validate
        val_results = evaluate(model, val_loader, device, graph_level)
        val_f1 = (val_results["wheeze"]["f1"] + val_results["crackle"]["f1"]) / 2

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_f1


# ════════════════════════════════════════════════════════════════════════════
# MAIN COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  MULTI-BACKBONE GNN TRAINING & EVALUATION")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    backbones = ["wav2vec2_base", "wav2vec2_large", "hubert_base", "hubert_large"]
    gnn_models = {
        "Baseline_GAT": lambda dim: make_baseline_gat(dim),
        "GraphSAGE": lambda dim: make_graphsage(dim),
        "Improved_GAT": lambda dim: make_improved_gat(dim),
    }

    all_results = []

    for bb_name in backbones:
        bb_cache = CACHE_DIR / f"features_{bb_name}"
        if not bb_cache.exists():
            print(f"\n  SKIP: {bb_name} — cache dir not found at {bb_cache}")
            continue

        print(f"\n{'='*70}")
        print(f"  BACKBONE: {bb_name}")
        print(f"{'='*70}")

        try:
            meta, features, cache_dir = load_features_from_cache(bb_name)
        except Exception as e:
            print(f"  ERROR loading features: {e}")
            continue

        # Determine embedding dim from first feature
        first_key = list(features.keys())[0]
        embed_dim = features[first_key]["embeddings"].shape[1]
        print(f"  Embedding dim: {embed_dim}")
        print(f"  Files with features: {len(features)}")

        # Filter meta to only files with features
        meta = meta[meta["wav_path"].isin(features.keys())].reset_index(drop=True)
        train_meta, val_meta, test_meta = patient_wise_split(meta)
        print(f"  Train: {len(train_meta)}, Val: {len(val_meta)}, Test: {len(test_meta)}")

        # Determine chunk duration based on model expectations
        chunk_secs = [10.0]  # use 10s chunks for all

        for gnn_name, gnn_fn in gnn_models.items():
            print(f"\n  --- Training {gnn_name} on {bb_name} ---")

            is_graph = (gnn_name == "Improved_GAT")
            edge_type = "similarity" if gnn_name == "GraphSAGE" else "temporal"
            chunk_s = 5.0 if gnn_name == "GraphSAGE" else 10.0

            ds_train = CachedDataset(train_meta, features, edge_type=edge_type, chunk_seconds=chunk_s)
            ds_val = CachedDataset(val_meta, features, edge_type=edge_type, chunk_seconds=chunk_s)
            ds_test = CachedDataset(test_meta, features, edge_type=edge_type, chunk_seconds=chunk_s)

            train_loader = DataLoader(ds_train, batch_size=1, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(ds_val, batch_size=1, shuffle=False, collate_fn=collate_fn)
            test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

            model = gnn_fn(embed_dim).to(DEVICE)
            t0 = time.time()
            model, val_f1 = train_one(model, train_loader, val_loader, DEVICE, epochs=30, graph_level=is_graph)
            train_time = time.time() - t0

            test_results = evaluate(model, test_loader, DEVICE, graph_level=is_graph)

            result = {
                "backbone": bb_name,
                "gnn": gnn_name,
                "embed_dim": embed_dim,
                "val_f1": val_f1,
                "train_time_s": train_time,
                "wheeze_f1": test_results["wheeze"]["f1"],
                "wheeze_precision": test_results["wheeze"]["precision"],
                "wheeze_recall": test_results["wheeze"]["recall"],
                "wheeze_roc_auc": test_results["wheeze"].get("roc_auc"),
                "crackle_f1": test_results["crackle"]["f1"],
                "crackle_precision": test_results["crackle"]["precision"],
                "crackle_recall": test_results["crackle"]["recall"],
                "crackle_roc_auc": test_results["crackle"].get("roc_auc"),
            }
            all_results.append(result)

            print(f"    Val F1: {val_f1:.4f}")
            print(f"    Test Wheeze F1: {test_results['wheeze']['f1']:.4f} (P={test_results['wheeze']['precision']:.3f}, R={test_results['wheeze']['recall']:.3f})")
            print(f"    Test Crackle F1: {test_results['crackle']['f1']:.4f} (P={test_results['crackle']['precision']:.3f}, R={test_results['crackle']['recall']:.3f})")
            print(f"    Train time: {train_time:.1f}s")

    # Summary
    print(f"\n{'='*70}")
    print("  BACKBONE COMPARISON SUMMARY")
    print(f"{'='*70}")

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "backbone_comparison.csv", index=False)

    headers = ["Backbone", "GNN", "Dim", "Val F1", "Wheeze F1", "Crackle F1", "Wheeze AUC", "Crackle AUC", "Train (s)"]
    rows = []
    for r in all_results:
        rows.append([
            r["backbone"], r["gnn"], r["embed_dim"],
            f"{r['val_f1']:.4f}",
            f"{r['wheeze_f1']:.4f}",
            f"{r['crackle_f1']:.4f}",
            f"{r.get('wheeze_roc_auc', 0) or 0:.3f}",
            f"{r.get('crackle_roc_auc', 0) or 0:.3f}",
            f"{r['train_time_s']:.1f}",
        ])
    summary_df = pd.DataFrame(rows, columns=headers)
    print(summary_df.to_string(index=False))

    # Markdown report
    lines = [
        "# Backbone Comparison Report",
        "",
        f"**Device**: {DEVICE}",
        "",
        "## Results",
        "",
        "| Backbone | GNN | Dim | Val F1 | Wheeze F1 | Crackle F1 | Wheeze AUC | Crackle AUC | Train (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in all_results:
        lines.append(
            f"| {r['backbone']} | {r['gnn']} | {r['embed_dim']} | {r['val_f1']:.4f} | "
            f"{r['wheeze_f1']:.4f} | {r['crackle_f1']:.4f} | "
            f"{r.get('wheeze_roc_auc', 0) or 0:.3f} | {r.get('crackle_roc_auc', 0) or 0:.3f} | "
            f"{r['train_time_s']:.1f} |"
        )

    # Find best
    if all_results:
        best = max(all_results, key=lambda x: x["val_f1"])
        lines += [
            "",
            "## Best Configuration",
            "",
            f"**{best['backbone']} + {best['gnn']}** with Val F1 = {best['val_f1']:.4f}",
        ]

    with open(RESULTS_DIR / "backbone_comparison_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {RESULTS_DIR / 'backbone_comparison.csv'}")
    print(f"  Saved: {RESULTS_DIR / 'backbone_comparison_report.md'}")


if __name__ == "__main__":
    main()