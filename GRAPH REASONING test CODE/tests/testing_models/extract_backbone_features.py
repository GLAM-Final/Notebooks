"""
=============================================================================
MULTI-BACKBONE FEATURE EXTRACTION
=============================================================================
Extracts embeddings from multiple acoustic backbones for comparison:
  1. Wav2Vec2-Base-960h  (768-dim, 95M params)   — already cached
  2. Wav2Vec2-Large-960h (1024-dim, 304M params)  — NEW
  3. HuBERT-Large-Librispeech (1024-dim, 304M params) — NEW
  4. HuBERT-Base-Librispeech (768-dim, 96M params) — NEW

All features are cached per backbone. The GNN training pipeline can then
be run on any backbone's features to compare performance.

Usage:  python testing_models/extract_backbone_features.py --backbone all
=============================================================================
"""

import os, sys, pathlib, pickle, hashlib, argparse, time, warnings
import numpy as np
import pandas as pd
import librosa
import torch

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"
AUDIO_FOLDER = BASE_DIR / "ICBHI_final_database"
DIAGNOSIS_FILE = BASE_DIR / "ICBHI_final_database" / "important" / "ICBHI_Challenge_diagnosis.txt"

SR = 16000
FRAME_SECONDS = 0.5
FRAME_LEN = int(SR * FRAME_SECONDS)


# ════════════════════════════════════════════════════════════════════════════
# BACKBONE DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

BACKBONES = {
    "wav2vec2_base": {
        "model_name": "facebook/wav2vec2-base-960h",
        "dim": 768,
        "class_name": "Wav2Vec2Model",
        "processor_name": "Wav2Vec2Processor",
    },
    "wav2vec2_large": {
        "model_name": "facebook/wav2vec2-large-960h",
        "dim": 1024,
        "class_name": "Wav2Vec2Model",
        "processor_name": "Wav2Vec2Processor",
    },
    "hubert_base": {
        "model_name": "facebook/hubert-base-ls960",
        "dim": 768,
        "class_name": "HubertModel",
        "processor_name": "Wav2Vec2Processor",  # HuBERT uses same processor
    },
    "hubert_large": {
        "model_name": "facebook/hubert-large-ls960-ft",
        "dim": 1024,
        "class_name": "HubertModel",
        "processor_name": "Wav2Vec2Processor",
    },
}


def get_model_and_processor(backbone_name):
    """Lazy-load model and processor."""
    cfg = BACKBONES[backbone_name]
    from transformers import Wav2Vec2Processor

    if cfg["class_name"] == "Wav2Vec2Model":
        from transformers import Wav2Vec2Model
        processor = Wav2Vec2Processor.from_pretrained(
            cfg["model_name"], local_files_only=True)
        model = Wav2Vec2Model.from_pretrained(
            cfg["model_name"], local_files_only=True)
    elif cfg["class_name"] == "HubertModel":
        from transformers import HubertModel
        processor = Wav2Vec2Processor.from_pretrained(
            cfg["model_name"], local_files_only=True)
        model = HubertModel.from_pretrained(
            cfg["model_name"], local_files_only=True)
    else:
        raise ValueError(f"Unknown class: {cfg['class_name']}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return processor, model, cfg["dim"]


# ════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def cache_key(audio_path, backbone_name):
    mtime = os.path.getmtime(audio_path)
    return hashlib.md5(f"{audio_path}_{backbone_name}_{mtime}".encode()).hexdigest()


def extract_file(audio_path, processor, model, embed_dim, cache_dir):
    """Extract embeddings for one file, using cache if available."""
    key = cache_key(audio_path, str(cache_dir))
    cpath = cache_dir / f"{key}.pkl"

    if cpath.exists():
        with open(cpath, "rb") as f:
            return pickle.load(f)

    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    num_frames = len(y) // FRAME_LEN
    if num_frames == 0:
        return np.zeros((0, embed_dim), dtype=np.float32), len(y) / SR, 0

    frames = [y[i * FRAME_LEN:(i + 1) * FRAME_LEN] for i in range(num_frames)]
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(frames), 32):
            batch = frames[i:i + 32]
            inputs = processor(batch, sampling_rate=SR, return_tensors="pt", padding=True)
            out = model(inputs.input_values)
            # Mean-pool over sequence dimension
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(emb)
    emb_arr = np.concatenate(embeddings, axis=0).astype(np.float32)
    payload = {"embeddings": emb_arr, "audio_duration": len(y) / SR, "num_frames": num_frames}

    cpath.parent.mkdir(parents=True, exist_ok=True)
    with open(cpath, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def scan_pairs(audio_root):
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
    return [(b, wav_map[b], ann_map[b]) for b in sorted(wav_map) if b in ann_map]


def extract_patient_id(base):
    return base.split("_")[0]


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="all",
                        choices=list(BACKBONES.keys()) + ["all"],
                        help="Which backbone to extract features for")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Load metadata
    diag = pd.read_csv(DIAGNOSIS_FILE, sep="\t", header=None,
                       names=["patient_id", "diagnosis"])
    diag["patient_id"] = diag["patient_id"].astype(str)
    pairs = scan_pairs(AUDIO_FOLDER)

    meta_rows = []
    diag_map = dict(zip(diag["patient_id"], diag["diagnosis"]))
    for base, wav, ann in pairs:
        pid = extract_patient_id(base)
        meta_rows.append({
            "file_id": base, "patient_id": pid,
            "diagnosis": diag_map.get(pid, "Unknown"),
            "wav_path": wav, "ann_path": ann,
        })
    meta = pd.DataFrame(meta_rows)
    print(f"Total files: {len(meta)}")

    backbones = list(BACKBONES.keys()) if args.backbone == "all" else [args.backbone]

    for bb_name in backbones:
        print(f"\n{'='*70}")
        print(f"  EXTRACTING: {bb_name} ({BACKBONES[bb_name]['dim']}-dim)")
        print(f"{'='*70}")

        cache_dir = CACHE_DIR / f"features_{bb_name}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        processor, model, embed_dim = get_model_and_processor(bb_name)
        print(f"  Model: {BACKBONES[bb_name]['model_name']}")
        print(f"  Embedding dim: {embed_dim}")
        print(f"  Cache dir: {cache_dir}")

        t_start = time.time()
        for i, row in meta.iterrows():
            result = extract_file(row["wav_path"], processor, model, embed_dim, cache_dir)
            if (i + 1) % 100 == 0 or i == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{len(meta)}] {elapsed:.1f}s ({rate:.1f} files/s)")

        elapsed = time.time() - t_start
        print(f"  Done: {len(meta)} files in {elapsed:.1f}s")

        # Save metadata alongside features
        meta.to_csv(cache_dir / "metadata.csv", index=False)
        print(f"  Saved metadata to {cache_dir / 'metadata.csv'}")

    print(f"\n{'='*70}")
    print("  FEATURE EXTRACTION COMPLETE")
    print(f"  Cached feature dirs:")
    for bb_name in backbones:
        d = CACHE_DIR / f"features_{bb_name}"
        n_files = len(list(d.glob("*.pkl")))
        print(f"    {bb_name}: {n_files} files, {d}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()