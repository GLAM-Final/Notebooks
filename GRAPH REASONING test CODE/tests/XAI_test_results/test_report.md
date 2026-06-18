# GNN Model Test Suite Report

**Generated:** 2026-06-16 23:00:04

**Device:** cpu

## Models Tested

| Model | Checkpoint | Status | Parameters |
|-------|-----------|--------|------------|
| GATv2 | `respiratory_gnn_model_cpu_full.pth` | ✓ Loaded | 677,509 |
| GraphSAGE | `best_respiratory_graphsage_temp_scaled.pth` | ✓ Loaded | 658,434 |
| ImprovedGAT | `best_improved_30epochs_no_earlystop.pt` | ✓ Loaded | 677,765 |

## Audio Inference Results


### sounds of asthma 30s

| Model | Wheeze Prob | Crackle Prob | Wheeze Pred | Crackle Pred | Inference (ms) |
|-------|------------|-------------|-------------|--------------|----------------|
| GATv2 | 0.4703 | 0.3857 | 0 | 0 | 82.92 |
| GraphSAGE | 0.5669 | 0.2997 | 1 | 0 | 8.20 |
| ImprovedGAT | 0.2992 | 0.3934 | 0 | 0 | 21.28 |

### sounds-of-asthma-wheezing-lung-sounds

| Model | Wheeze Prob | Crackle Prob | Wheeze Pred | Crackle Pred | Inference (ms) |
|-------|------------|-------------|-------------|--------------|----------------|
| GATv2 | 0.4905 | 0.3743 | 0 | 0 | 22.51 |
| GraphSAGE | 0.5854 | 0.3019 | 1 | 0 | 32.71 |
| ImprovedGAT | 0.3147 | 0.3909 | 0 | 0 | 37.22 |

## XAI Saliency Visualizations

The following explainable AI (XAI) visualizations were generated:

| Technique | Description | Evidence |
|-----------|-------------|----------|
| **Integrated Gradients (IG)** | Attribution method that computes path integral of gradients from baseline to input. Shows which audio frames most influence the prediction. | Heatmap overlays on waveform + spectrogram |
| **Grad × Input** | Simpler attribution via element-wise product of gradient and input. Provides complementary signal to IG. | Bar chart per frame |
| **Attention Pooling Weights** | Learned per-node importance weights from the model's attention pool layer. Shows which frames the model was trained to focus on. | Bar chart per frame |
| **Top-K Frame Highlighting** | Top-5 most salient frames overlaid on mel-spectrogram with bounding boxes. Directly shows which acoustic regions drive predictions. | Bounding boxes on spectrogram |

### Generated XAI Output Files

| File | Description |
|------|-------------|
| `sounds of asthma 30s_GATv2_xai_crackle.png` | XAI saliency heatmap |
| `sounds of asthma 30s_GATv2_xai_wheeze.png` | XAI saliency heatmap |
| `sounds of asthma 30s_GraphSAGE_xai_crackle.png` | XAI saliency heatmap |
| `sounds of asthma 30s_GraphSAGE_xai_wheeze.png` | XAI saliency heatmap |
| `sounds of asthma 30s_ImprovedGAT_xai_crackle.png` | XAI saliency heatmap |
| `sounds of asthma 30s_ImprovedGAT_xai_wheeze.png` | XAI saliency heatmap |
| `sounds-of-asthma-wheezing-lung-sounds_GATv2_xai_crackle.png` | XAI saliency heatmap |
| `sounds-of-asthma-wheezing-lung-sounds_GATv2_xai_wheeze.png` | XAI saliency heatmap |
| `sounds-of-asthma-wheezing-lung-sounds_GraphSAGE_xai_crackle.png` | XAI saliency heatmap |
| `sounds-of-asthma-wheezing-lung-sounds_GraphSAGE_xai_wheeze.png` | XAI saliency heatmap |
| `sounds-of-asthma-wheezing-lung-sounds_ImprovedGAT_xai_crackle.png` | XAI saliency heatmap |
| `sounds-of-asthma-wheezing-lung-sounds_ImprovedGAT_xai_wheeze.png` | XAI saliency heatmap |
| `sounds of asthma 30s_GATv2_attributions.json` | Attribution scores (per-frame) |
| `sounds of asthma 30s_GraphSAGE_attributions.json` | Attribution scores (per-frame) |
| `sounds of asthma 30s_ImprovedGAT_attributions.json` | Attribution scores (per-frame) |
| `sounds-of-asthma-wheezing-lung-sounds_GATv2_attributions.json` | Attribution scores (per-frame) |
| `sounds-of-asthma-wheezing-lung-sounds_GraphSAGE_attributions.json` | Attribution scores (per-frame) |
| `sounds-of-asthma-wheezing-lung-sounds_ImprovedGAT_attributions.json` | Attribution scores (per-frame) |

### XAI Methodology

The saliency heatmaps contain 4 panels:

1. **Panel 1 (Waveform + IG)**: Raw audio waveform with Integrated Gradients attribution overlaid the waveforms from the audio. Hotter colors (red/orange) indicate frames that contributed more to the model's prediction and the Top-K frames are marked with red circles as seen in the visualisations.

2. **Panel 2 (Mel-Spectrogram + Top-K)**: Mel-frequency spectrogram with the top-5 most salient frames highlighted with red bounding boxes. This directly shows which acoustic frequency-time regions the model focused on (e.g., wheeze bands at 400-800 Hz, crackle transients).

3. **Panel 3 (Grad × Input)**: Alternative attribution method showing per-frame importance scores. Top-K frames shown in red. Agreement between IG and Grad×Input indicates robust attribution.

4. **Panel 4 (Attention Weights)**: Learned attention pooling weights showing which frames the model was trained to consider diagnostically important. Top-K frames highlighted in gold.

> **Clinical Validation**: If the model is correctly identifying respiratory biomarkers, the highlighted frames should align with known acoustic signatures of wheezes (sinusoidal tones at 400-800 Hz, duration >80ms) and crackles (short explosive transients <20ms, wide frequency distribution).

## Synthetic Batch Evaluation Results

### Wheeze Detection

| Model | Mean Prob | F1 | Accuracy | Precision | Recall | Mean Time (ms) |
|-------|----------|-----|----------|-----------|--------|----------------|
| GATv2 | 0.5734 | 0.5128 | 0.3667 | 0.3448 | 1.0000 | 8.10 |
| GraphSAGE | 0.0000 | 0.0000 | 0.6667 | 0.0000 | 0.0000 | 3.90 |
| ImprovedGAT | 0.2264 | 0.0000 | 0.6667 | 0.0000 | 0.0000 | 6.90 |

### Crackle Detection

| Model | Mean Prob | F1 | Accuracy | Precision | Recall | Mean Time (ms) |
|-------|----------|-----|----------|-----------|--------|----------------|
| GATv2 | 0.3713 | 0.0000 | 0.6667 | 0.0000 | 0.0000 | 8.10 |
| GraphSAGE | 1.0000 | 0.5000 | 0.3333 | 0.3333 | 1.0000 | 3.90 |
| ImprovedGAT | 0.3173 | 0.0000 | 0.6667 | 0.0000 | 0.0000 | 6.90 |

### Inference Timing

| Model | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|-------|----------|---------|---------|----------|
| GATv2 | 8.10 | 4.75 | 0.00 | 18.42 |
| GraphSAGE | 3.90 | 1.82 | 0.00 | 8.17 |
| ImprovedGAT | 6.90 | 1.92 | 1.79 | 10.94 |

## Architecture Comparison

| Feature | GATv2 | GraphSAGE | ImprovedGAT |
|---------|-------|-----------|-------------|
| Conv Type | GATv2Conv + SAGEConv | SAGEConv only | GATConv + SAGEConv |
| Layers | 3 | 3 | 4 |
| Pooling | Attention | Global Mean | Attention |
| Dropout | 0.4 | 0.5 | 0.4 |
| Uncertainty | Yes | No | Yes |
| Temperature Scaling | No | Yes | No |
| Parameters | 677,509 | 658,434 | 677,765 |

## Hardware Profiling Results

### Model Size and Complexity

| Model | Params | Size (MB) | FLOPs (5s chunk) | FLOPs (10s chunk) |
|-------|--------|-----------|-----------------|------------------|
| GATv2 | 677,509 | 2.59 | 1.20 MFLOPs | 1.54 MFLOPs |
| GraphSAGE | 658,434 | 2.52 | 0.56 MFLOPs | 0.59 MFLOPs |
| ImprovedGAT | 677,765 | 2.59 | 1.34 MFLOPs | 1.70 MFLOPs |

### Inference Latency Distribution

| Model | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (inf/s) |
|-------|----------|---------|---------|---------|-------------------|
| GATv2 | 7.32 | 5.44 | 9.51 | 73.11 | 136.6 |
| GraphSAGE | 2.98 | 2.82 | 4.16 | 5.74 | 335.5 |
| ImprovedGAT | 8.47 | 6.23 | 10.07 | 29.92 | 118.1 |

### LaTeX-Ready Hardware Profiling Table

\begin{table}[!h]
  \renewcommand{\arraystretch}{1.15}
  \caption{Hardware Profiling: Edge-Deployment Feasibility}
  \label{tab:hardware}
  \centering
  \begin{tabular}{@{}lrrrrrr@{}}
    \toprule
    \textbf{Model}
      & \textbf{Params}
      & \textbf{Size (MB)}
      & \textbf{FLOPs (M)}
      & \textbf{Latency (ms)}
      & \textbf{p95 (ms)}
      & \textbf{Throughput} \\
    \midrule
    Baseline GAT & 677,509 & 2.59 & 1.20 & 7.32 & 9.51 & 136.6 $\times$ \\
    GraphSAGE & 658,434 & 2.52 & 0.56 & 2.98 & 4.16 & 335.5 $\times$ \\
    Improved GAT & 677,765 & 2.59 & 1.34 & 8.47 & 10.07 & 118.1 $\times$ \\
    \bottomrule
  \end{tabular}
\end{table}

### LaTeX-Ready XAI Evaluation Table

\begin{table}[!h]
  \renewcommand{\arraystretch}{1.15}
  \caption{XAI Evaluation: Attribution Quality Metrics}
  \label{tab:xai_results}
  \centering
  \begin{tabular}{@{}lcccc@{}}
    \toprule
    \textbf{Model}
      & \textbf{SNR (Wheeze)}
      & \textbf{SNR (Crackle)}
      & \textbf{Top-5 (Wheeze)}
      & \textbf{Top-5 (Crackle)} \\
    \midrule
    Baseline GAT & 0.000 & 0.023 & 0\% & 40\% \\
    GraphSAGE & 0.000 & 0.023 & 0\% & 40\% \\
    ImprovedRespiratoryGAT & 0.000 & 0.023 & 0\% & 40\% \\
    \bottomrule
  \end{tabular}
\end{table}


## ICBHI Evaluation Framework

ICBHI metadata loaded: 10 records
Patients: 5
Diagnoses: <StringArray>
['COPD', 'Healthy']
Length: 2, dtype: str

To perform full ICBHI evaluation, ensure audio files are accessible and run with ground-truth annotations.

## Test Files

- **Test suite:** `model evalu/t_e.py`
- **Output directory:** `model evalu\test_results`
- **XAI directory:** `model evalu\test_results\xai_visualizations`
- **Audio test file:** `model_comparisons\audio_test_files\sounds of asthma 30s.MP3`
- **Audio test file:** `model_comparisons\audio_test_files\sounds-of-asthma-wheezing-lung-sounds.mp3`
