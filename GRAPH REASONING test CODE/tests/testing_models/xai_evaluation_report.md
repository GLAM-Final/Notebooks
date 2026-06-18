# XAI Evaluation Results

## Attribution Methods

- **Integrated Gradients** (IG): Attributes prediction to input features by integrating gradients along a path from zero baseline to input. Provides theoretically principled attribution.
- **Grad×Input**: Element-wise product of gradient and input. Simple single-pass attribution, computationally cheaper than IG.
- **Attention Weights**: Learned per-node importance weights from the attention pooling layer (Improved GAT only).

## Quantitative XAI Metrics

| Model | Wheeze Prob Range | Crackle Prob Range | IG SNR (Wheeze) | IG SNR (Crackle) | Top-5 IG Overlap (Wheeze) | Top-5 IG Overlap (Crackle) |
|---|---|---|---|---|---|---|
| Baseline_GAT | 0.2683–0.5555 (μ=0.4356) | 0.4807–0.5449 (μ=0.5019) | 0.000 | 0.023 | 0% | 40% |
| GraphSAGE | 0.4275–0.6138 (μ=0.5186) | 0.3381–0.5747 (μ=0.4253) | 0.000 | 0.029 | 0% | 40% |
| Improved_GAT | 0.3494–0.3494 (μ=0.3494) | 0.3780–0.3780 (μ=0.3780) | 1.009 | 1.102 | 60% | 80% |

## Interpretation

### Signal-to-Noise Ratio (SNR)
The IG SNR measures whether attributions are concentrated on ground-truth positive frames (SNR > 1) or uniformly distributed (SNR ≈ 1).

### Top-K Overlap
The fraction of the 5 most attributed frames that overlap with ground-truth positive frames. Higher overlap indicates the model is correctly focusing on clinically relevant segments.

## Visual Outputs

For each model, the following XAI visualisations are generated:
1. **XAI Bar Chart (Wheeze)**: Frame-level IG and Grad×Input attributions with ground truth overlay
2. **XAI Bar Chart (Crackle)**: Same for crackle detection
3. **Combined XAI**: 4-panel view of IG and Grad×Input for both tasks
4. **Spectrogram Heatmap (Wheeze)**: Mel spectrogram with IG overlay
5. **Spectrogram Heatmap (Crackle)**: Same for crackle