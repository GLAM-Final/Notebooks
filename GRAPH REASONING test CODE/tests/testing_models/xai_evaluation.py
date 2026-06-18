"""
=============================================================================
XAI EVALUATION: Integrated Gradients + Grad×Input for All Three GNN Models
=============================================================================
Runs attribution methods on all three models using a test audio sample,
generates XAI plots (bar charts, heatmap overlays), and saves results
to testing_models/results/
=============================================================================
"""

import os, sys, pathlib, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "testing_models" / "results"
CKPT_BASELINE_GAT = BASE_DIR / "testing_models" / "respiratory_gnn_model_cpu_full.pth"
CKPT_GRAPHSAGE    = BASE_DIR / "testing_models" / "best_respiratory_graphsage_temp_scaled.pth"
CKPT_IMPROVED_GAT = BASE_DIR / "testing_models" / "best_improved_30epochs_no_earlystop.pt"
AUDIO_FOLDER = BASE_DIR / "ICBHI_final_database"

from torch_geometric.nn import GATConv, SAGEConv, BatchNorm
from torch_geometric.data import Data

SR = 16000
FRAME_SECONDS = 0.5
FRAME_LEN = int(FRAME_SECONDS * SR)


# ════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

class BaselineGAT(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.wheeze_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1))
        self.crackle_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1))

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
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=3, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.wheeze_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1))
        self.crackle_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1))

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


class ImprovedGATModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=4, num_heads=4, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                self.gat_layers.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True, dropout=dropout))
            else:
                self.gat_layers.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))
        self.attention_pool = nn.Sequential(nn.Linear(hidden_dim, max(4, hidden_dim // 4)), nn.Tanh(), nn.Linear(max(4, hidden_dim // 4), 1))
        self.wheeze_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
        self.crackle_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
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
# XAI ATTRIBUTION METHODS
# ════════════════════════════════════════════════════════════════════════════

def integrated_gradients(model, data, target="wheeze", steps=25):
    """Integrated Gradients attribution at node level. Returns [num_nodes] array."""
    model.eval()
    x = data.x.detach().clone()
    baseline = torch.zeros_like(x)
    grads_sum = torch.zeros_like(x)
    for alpha in np.linspace(0.0, 1.0, steps):
        x_scaled = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        d = Data(x=x_scaled, edge_index=data.edge_index)
        w_logits, c_logits = model(d)
        out = w_logits if target == "wheeze" else c_logits
        out_scalar = out.view(-1)[0]
        grad = torch.autograd.grad(out_scalar, x_scaled, retain_graph=False, create_graph=False)[0]
        grads_sum += grad
    avg_grads = grads_sum / float(steps)
    attributions = (x - baseline) * avg_grads
    return attributions.detach().cpu().norm(p=2, dim=1).numpy()


def grad_times_input(model, data, target="wheeze"):
    """Grad × Input attribution at node level. Returns [num_nodes] array."""
    model.eval()
    x = data.x.detach().clone().requires_grad_(True)
    d = Data(x=x, edge_index=data.edge_index)
    w_logits, c_logits = model(d)
    out = w_logits if target == "wheeze" else c_logits
    out_scalar = out.view(-1)[0]
    grad = torch.autograd.grad(out_scalar, x, retain_graph=False, create_graph=False)[0]
    return (grad * x).detach().cpu().norm(p=2, dim=1).numpy()


def get_attention_weights(model, data):
    """Extract attention pooling weights per node (Improved GAT only)."""
    model.eval()
    x = data.x.detach()
    with torch.no_grad():
        x_proj = model.input_proj(x)
        residuals = []
        for i, (conv, norm) in enumerate(zip(model.gat_layers, model.norms)):
            x_new = conv(x_proj, data.edge_index)
            x_new = F.elu(x_new)
            x_new = norm(x_new)
            if i > 0 and i % 2 == 0:
                x_new = x_new + residuals[-1]
            x_proj = F.dropout(x_new, p=model.dropout, training=False)
            residuals.append(x_proj)
        attn_scores = model.attention_pool(x_proj).squeeze(-1)
        weights = torch.softmax(attn_scores, dim=0).cpu().numpy()
    return weights


# ════════════════════════════════════════════════════════════════════════════
# DATA + FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def load_test_audio_and_labels(audio_folder):
    """Load one test audio file with annotations for XAI demo."""
    # Find a file with both wheeze and crackle if possible
    import pandas as pd
    wav_files = sorted([f for f in os.listdir(audio_folder) if f.endswith(".wav")])
    for fname in wav_files[:50]:  # check first 50
        base = os.path.splitext(fname)[0]
        ann_path = os.path.join(audio_folder, base + ".txt")
        if not os.path.exists(ann_path):
            continue
        ann = pd.read_csv(ann_path, sep="\t", header=None, names=["start", "end", "crackle", "wheeze"])
        has_wheeze = ann["wheeze"].sum() > 0
        has_crackle = ann["crackle"].sum() > 0
        if has_wheeze and has_crackle:
            return os.path.join(audio_folder, fname), ann
    # Fallback: just use the first file
    fname = wav_files[0]
    base = os.path.splitext(fname)[0]
    ann_path = os.path.join(audio_folder, base + ".txt")
    ann = pd.read_csv(ann_path, sep="\t", header=None, names=["start", "end", "crackle", "wheeze"])
    return os.path.join(audio_folder, fname), ann


def extract_wav2vec2_embeddings(audio_path, sr=SR, frame_seconds=FRAME_SECONDS):
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", local_files_only=True)
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", local_files_only=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    frame_len = int(frame_seconds * sr)
    num_frames = len(y) // frame_len
    frames = [y[i*frame_len:(i+1)*frame_len] for i in range(num_frames)]

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(frames), 24):
            batch = frames[i:i+24]
            inputs = processor(batch, sampling_rate=sr, return_tensors="pt", padding=True)
            out = model(inputs.input_values)
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(emb)
    emb_arr = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 768), dtype=np.float32)
    return emb_arr.astype(np.float32), y, num_frames


def make_labels(ann_df, num_frames, frame_seconds=FRAME_SECONDS):
    w_lbl = np.zeros(num_frames, dtype=np.float32)
    c_lbl = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        fs = i * frame_seconds
        fe = fs + frame_seconds
        overlaps = ann_df[(ann_df["start"] < fe) & (ann_df["end"] > fs)]
        for _, row in overlaps.iterrows():
            overlap = max(0.0, min(fe, row["end"]) - max(fs, row["start"]))
            if overlap / frame_seconds >= 0.3:
                if int(row["crackle"]) == 1:
                    c_lbl[i] = 1.0
                if int(row["wheeze"]) == 1:
                    w_lbl[i] = 1.0
    return w_lbl, c_lbl


def build_temporal_edges(n):
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long)
    edges = [[i, i+1] for i in range(n-1)] + [[i+1, i] for i in range(n-1)]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# ════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════════════

def plot_xai_bar_chart(ig_attr, gxi_attr, w_true, model_name, save_dir, target="wheeze"):
    """Bar chart of IG and Grad×Input attributions with ground truth overlay."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x = np.arange(len(ig_attr))

    # Normalize attributions for visualization
    ig_norm = ig_attr / (ig_attr.max() + 1e-9)
    gxi_norm = gxi_attr / (gxi_attr.max() + 1e-9)

    axes[0].bar(x, ig_norm, color=np.where(w_true > 0.5, "red", "steelblue"), alpha=0.8)
    axes[0].set_ylabel("Normalised IG Attribution")
    axes[0].set_title(f"{model_name} — Integrated Gradients ({target})")
    axes[0].grid(alpha=0.3)

    axes[1].bar(x, gxi_norm, color=np.where(w_true > 0.5, "red", "steelblue"), alpha=0.8)
    axes[1].set_ylabel("Normalised Grad×Input")
    axes[1].set_title(f"{model_name} — Grad×Input ({target})")
    axes[1].set_xlabel("Frame Index (0.5s each)")
    axes[1].grid(alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="red", label="Ground truth positive"),
                       Patch(facecolor="steelblue", label="Ground truth negative")]
    axes[0].legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_xai_bar_{target}.png", dpi=150)
    plt.close()
    print(f"  Saved: {model_name}_xai_bar_{target}.png")


def plot_xai_combined(ig_w, ig_c, gxi_w, gxi_c, attn, w_true, c_true, model_name, save_dir):
    """Combined 4-panel XAI figure."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    x = np.arange(len(ig_w))

    ig_w_n = ig_w / (ig_w.max() + 1e-9)
    ig_c_n = ig_c / (ig_c.max() + 1e-9)
    gxi_w_n = gxi_w / (gxi_w.max() + 1e-9)
    gxi_c_n = gxi_c / (gxi_c.max() + 1e-9)

    for ax, data, title in zip(axes, [ig_w_n, ig_c_n, gxi_w_n, gxi_c_n],
                               ["IG Attribution (Wheeze)", "IG Attribution (Crackle)",
                                "Grad×Input (Wheeze)", "Grad×Input (Crackle)"]):
        colors = []
        for i in range(len(data)):
            if w_true[i] > 0.5 and c_true[i] > 0.5:
                colors.append("darkred")
            elif w_true[i] > 0.5:
                colors.append("red")
            elif c_true[i] > 0.5:
                colors.append("orange")
            else:
                colors.append("steelblue")
        ax.bar(x, data, color=colors, alpha=0.8)
        ax.set_ylabel("Normalised")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Frame Index (0.5s each)")
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="red", label="Wheeze"),
                       Patch(facecolor="orange", label="Crackle"),
                       Patch(facecolor="darkred", label="Both"),
                       Patch(facecolor="steelblue", label="Neither")]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_xai_combined.png", dpi=150)
    plt.close()
    print(f"  Saved: {model_name}_xai_combined.png")


def plot_xai_heatmap_on_spectrogram(ig_attr, audio, sr, model_name, save_dir, target="wheeze"):
    """Heatmap overlay on mel spectrogram."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, fmax=sr//2)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[0])
    axes[0].set_title(f"{model_name} — Mel Spectrogram")

    # Plot IG as line overlay
    ig_norm = ig_attr / (ig_attr.max() + 1e-9)
    frames = np.arange(len(ig_norm))
    time_axis = np.arange(len(ig_norm)) * FRAME_SECONDS
    axes[0].plot(time_axis, ig_norm * np.max(S_db), color="red", linewidth=1.5, alpha=0.8, label="IG Attribution")
    axes[0].legend(loc="upper right")

    # Attribution-only bar
    axes[1].bar(frames, ig_norm, color="red", alpha=0.7)
    axes[1].set_title(f"{model_name} — {target.upper()} IG Attribution (Normalised)")
    axes[1].set_xlabel("Frame Index")
    axes[1].set_ylabel("Attribution")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_xai_heatmap_{target}.png", dpi=150)
    plt.close()
    print(f"  Saved: {model_name}_xai_heatmap_{target}.png")


# ════════════════════════════════════════════════════════════════════════════
# MAIN XAI EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  XAI EVALUATION: All Three GNN Models")
    print("=" * 70)

    # Load test audio
    print("\n[1] Loading test audio with annotations...")
    audio_path, ann_df = load_test_audio_and_labels(AUDIO_FOLDER)
    print(f"  File: {os.path.basename(audio_path)}")

    # Extract features
    print("\n[2] Extracting Wav2Vec2 embeddings...")
    emb, audio, num_frames = extract_wav2vec2_embeddings(audio_path)
    w_true, c_true = make_labels(ann_df, num_frames)
    print(f"  Frames: {num_frames}, Wheeze frames: {int(w_true.sum())}, Crackle frames: {int(c_true.sum())}")

    # Build graph
    edge_index = build_temporal_edges(num_frames)
    x = torch.tensor(emb, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index)

    # Models config
    models_config = [
        ("Baseline_GAT", BaselineGAT, CKPT_BASELINE_GAT),
        ("GraphSAGE", GraphSAGEModel, CKPT_GRAPHSAGE),
        ("Improved_GAT", ImprovedGATModel, CKPT_IMPROVED_GAT),
    ]

    all_xai_results = []

    for name, model_cls, ckpt_path in models_config:
        print(f"\n{'='*60}")
        print(f"  XAI: {name}")
        print(f"{'='*60}")

        save_dir = RESULTS_DIR / name
        save_dir.mkdir(exist_ok=True)

        # Load model
        model = model_cls()
        sd = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model_keys = set(model.state_dict().keys())
        sd = {k: v for k, v in sd.items() if k in model_keys}
        model.load_state_dict(sd, strict=False)
        model.eval()

        # Get model predictions
        with torch.no_grad():
            w_logits, c_logits = model(data)
            w_prob = torch.sigmoid(w_logits).numpy()
            c_prob = torch.sigmoid(c_logits).numpy()

        print(f"  Wheeze prob: min={w_prob.min():.4f}, max={w_prob.max():.4f}, mean={w_prob.mean():.4f}")
        print(f"  Crackle prob: min={c_prob.min():.4f}, max={c_prob.max():.4f}, mean={c_prob.mean():.4f}")

        # For node-level models (Baseline GAT, GraphSAGE), attributions are per-node
        # For graph-level models (Improved GAT), attributions are on the full graph but output is scalar
        is_graph_level = (name == "Improved_GAT")

        # Compute attributions
        print("  Computing Integrated Gradients (wheeze)...")
        ig_w = integrated_gradients(model, data, target="wheeze", steps=25)
        print("  Computing Integrated Gradients (crackle)...")
        ig_c = integrated_gradients(model, data, target="crackle", steps=25)
        print("  Computing Grad×Input (wheeze)...")
        gxi_w = grad_times_input(model, data, target="wheeze")
        print("  Computing Grad×Input (crackle)...")
        gxi_c = grad_times_input(model, data, target="crackle")

        # Attention weights (Improved GAT only)
        attn = None
        if is_graph_level:
            try:
                attn = get_attention_weights(model, data)
                print(f"  Attention weights: min={attn.min():.4f}, max={attn.max():.4f}, entropy={-(attn * np.log(attn + 1e-9)).sum():.3f}")
            except Exception:
                attn = None

        # Generate plots
        print("  Generating XAI plots...")
        plot_xai_bar_chart(ig_w, gxi_w, w_true, name, save_dir, target="wheeze")
        plot_xai_bar_chart(ig_c, gxi_c, c_true, name, save_dir, target="crackle")
        plot_xai_combined(ig_w, ig_c, gxi_w, gxi_c, attn, w_true, c_true, name, save_dir)
        plot_xai_heatmap_on_spectrogram(ig_w, audio, SR, name, save_dir, target="wheeze")
        plot_xai_heatmap_on_spectrogram(ig_c, audio, SR, name, save_dir, target="crackle")

        # Compute attribution overlap with ground truth
        w_frames = np.where(w_true > 0.5)[0]
        c_frames = np.where(c_true > 0)[0]
        ig_w_pos = ig_w[w_frames].mean() if len(w_frames) > 0 else 0
        ig_w_neg = ig_w[np.where(w_true < 0.5)[0]].mean()
        ig_c_pos = ig_c[c_frames].mean() if len(c_frames) > 0 else 0
        ig_c_neg = ig_c[np.where(c_true < 0.5)[0]].mean()
        signal_noise_ratio_w = ig_w_pos / (ig_w_neg + 1e-9)
        signal_noise_ratio_c = ig_c_pos / (ig_c_neg + 1e-9)

        # Top-K attributed frames
        ig_w_top5 = np.argsort(ig_w)[-5:][::-1].tolist()
        ig_c_top5 = np.argsort(ig_c)[-5:][::-1].tolist()
        gxi_w_top5 = np.argsort(gxi_w)[-5:][::-1].tolist()
        gxi_c_top5 = np.argsort(gxi_c)[-5:][::-1].tolist()

        # Overlap: how many of top-5 IG frames are actually positive?
        if len(w_frames) > 0:
            ig_w_top5_overlap = len(set(ig_w_top5) & set(w_frames.tolist())) / 5
        else:
            ig_w_top5_overlap = 0
        if len(c_frames) > 0:
            ig_c_top5_overlap = len(set(ig_c_top5) & set(c_frames.tolist())) / 5
        else:
            ig_c_top5_overlap = 0

        result = {
            "model": name,
            "wheeze_prob_range": [float(w_prob.min()), float(w_prob.max()), float(w_prob.mean())],
            "crackle_prob_range": [float(c_prob.min()), float(c_prob.max()), float(c_prob.mean())],
            "ig_signal_noise_ratio_wheeze": float(signal_noise_ratio_w),
            "ig_signal_noise_ratio_crackle": float(signal_noise_ratio_c),
            "ig_top5_overlap_wheeze": float(ig_w_top5_overlap),
            "ig_top5_overlap_crackle": float(ig_c_top5_overlap),
            "ig_wheeze_top5_frames": ig_w_top5,
            "ig_crackle_top5_frames": ig_c_top5,
        }
        all_xai_results.append(result)

        print(f"  IG Signal-to-Noise (wheeze): {signal_noise_ratio_w:.3f}")
        print(f"  IG Signal-to-Noise (crackle): {signal_noise_ratio_c:.3f}")
        print(f"  Top-5 IG overlap with ground truth (wheeze): {ig_w_top5_overlap:.0%}")
        print(f"  Top-5 IG overlap with ground truth (crackle): {ig_c_top5_overlap:.0%}")

    # Save XAI summary
    import pandas as pd
    xai_df = pd.DataFrame(all_xai_results)
    xai_df.to_csv(RESULTS_DIR / "xai_evaluation_results.csv", index=False)
    print(f"\n  Saved: xai_evaluation_results.csv")

    # Generate XAI comparison markdown
    lines = [
        "# XAI Evaluation Results",
        "",
        "## Attribution Methods",
        "",
        "- **Integrated Gradients** (IG): Attributes prediction to input features by integrating gradients along a path from zero baseline to input. Provides theoretically principled attribution.",
        "- **Grad×Input**: Element-wise product of gradient and input. Simple single-pass attribution, computationally cheaper than IG.",
        "- **Attention Weights**: Learned per-node importance weights from the attention pooling layer (Improved GAT only).",
        "",
        "## Quantitative XAI Metrics",
        "",
        "| Model | Wheeze Prob Range | Crackle Prob Range | IG SNR (Wheeze) | IG SNR (Crackle) | Top-5 IG Overlap (Wheeze) | Top-5 IG Overlap (Crackle) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in all_xai_results:
        wp = r["wheeze_prob_range"]
        cp = r["crackle_prob_range"]
        lines.append(
            f"| {r['model']} | {wp[0]:.4f}–{wp[1]:.4f} (μ={wp[2]:.4f}) | "
            f"{cp[0]:.4f}–{cp[1]:.4f} (μ={cp[2]:.4f}) | "
            f"{r['ig_signal_noise_ratio_wheeze']:.3f} | {r['ig_signal_noise_ratio_crackle']:.3f} | "
            f"{r['ig_top5_overlap_wheeze']:.0%} | {r['ig_top5_overlap_crackle']:.0%} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "### Signal-to-Noise Ratio (SNR)",
        "The IG SNR measures whether attributions are concentrated on ground-truth positive frames (SNR > 1) or uniformly distributed (SNR ≈ 1).",
        "",
        "### Top-K Overlap",
        "The fraction of the 5 most attributed frames that overlap with ground-truth positive frames. Higher overlap indicates the model is correctly focusing on clinically relevant segments.",
        "",
        "## Visual Outputs",
        "",
        "For each model, the following XAI visualisations are generated:",
        "1. **XAI Bar Chart (Wheeze)**: Frame-level IG and Grad×Input attributions with ground truth overlay",
        "2. **XAI Bar Chart (Crackle)**: Same for crackle detection",
        "3. **Combined XAI**: 4-panel view of IG and Grad×Input for both tasks",
        "4. **Spectrogram Heatmap (Wheeze)**: Mel spectrogram with IG overlay",
        "5. **Spectrogram Heatmap (Crackle)**: Same for crackle",
    ]

    with open(RESULTS_DIR / "xai_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: xai_evaluation_report.md")

    print(f"\n{'='*70}")
    print("  XAI EVALUATION COMPLETE")
    print(f"  Files in {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()