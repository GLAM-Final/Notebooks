"""Deployment utilities: load model checkpoint and run inference from numpy arrays.

Simple helpers for a minimal deployment/test harness used by the notebook.
"""

import os
import numpy as np
import torch
import time
from typing import Optional, Tuple, Dict, Any
from collections import deque
import math


def load_deployment_model(checkpoint_path: str, device: str = "cpu"):
    """Load `ImprovedRespiratoryGAT` from `model_comparisons` and map to `device`.

    Args:
        checkpoint_path: path to a state_dict saved with `torch.save(model.state_dict(), path)`.
        device: device string ("cpu" or "cuda").

    Returns:
        model: the loaded PyTorch model in eval() mode.
    """
    from model_comparisons.gnn_improved_model import ImprovedRespiratoryGAT

    map_loc = torch.device(device)
    model = ImprovedRespiratoryGAT(input_dim=768, hidden_dim=256, num_layers=4, num_heads=4, dropout=0.4)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=map_loc)

    # Handle common wrapped dict formats (module. prefixes or nested state_dict)
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    if isinstance(state, dict) and "state_dict" in state:
        s = state["state_dict"]
        if any(k.startswith("module.") for k in s.keys()):
            s = {k.replace("module.", ""): v for k, v in s.items()}
        state = s

    model.load_state_dict(state)
    model.to(map_loc)
    model.eval()
    return model


def predict_from_arrays(
    model,
    x_array: np.ndarray,
    edge_index_array: np.ndarray,
    device: str = "cpu",
    return_logits: bool = False,
):
    """Run inference for a single graph given node features and edge_index arrays.

    Args:
        model: PyTorch model (ImprovedRespiratoryGAT) in eval mode.
        x_array: float32 array of shape [num_nodes, feat_dim].
        edge_index_array: int array of shape [2, num_edges].
        device: device string.
        return_logits: if True, also return raw logits (before sigmoid).

    Returns:
        If return_logits is False: (w_prob, c_prob) as numpy floats or arrays.
        If return_logits is True: (w_prob, c_prob, w_logits, c_logits)
    """
    from torch_geometric.data import Data

    dev = torch.device(device)
    x = torch.tensor(x_array, dtype=torch.float32, device=dev)
    edge_index = torch.tensor(edge_index_array, dtype=torch.long, device=dev)

    data = Data(x=x, edge_index=edge_index)
    with torch.no_grad():
        w_logits, c_logits = model(data)
        w_prob = torch.sigmoid(w_logits).cpu().numpy()
        c_prob = torch.sigmoid(c_logits).cpu().numpy()

    if return_logits:
        return w_prob, c_prob, w_logits.cpu().numpy(), c_logits.cpu().numpy()
    return w_prob, c_prob


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def fit_platt_scaler(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Fit a simple Platt-scaling (logistic) calibrator on provided probs.

    Returns a small param dict {'coef': a, 'intercept': b} where calibrated = sigmoid(a*logit(p) + b).
    Requires scikit-learn to be installed.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        raise RuntimeError("scikit-learn required for Platt scaling. Install sklearn and retry.") from e

    p = np.asarray(probs).ravel()
    y = np.asarray(y_true).ravel().astype(int)
    if len(np.unique(y)) == 1:
        raise ValueError("y_true contains only one class; cannot fit Platt scaler.")

    X = _logit(p).reshape(-1, 1)
    clf = LogisticRegression(solver="lbfgs", C=1e6, max_iter=2000)
    clf.fit(X, y)
    return {"coef": float(clf.coef_.ravel()[0]), "intercept": float(clf.intercept_.ravel()[0])}


def apply_platt_scaler(probs: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    p = np.asarray(probs).ravel()
    logits = _logit(p)
    a = float(params.get("coef", 1.0))
    b = float(params.get("intercept", 0.0))
    return _sigmoid(a * logits + b)


def find_optimal_threshold_by_f1(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Return threshold that maximizes F1 score on (y_true, probs)."""
    try:
        from sklearn.metrics import precision_recall_curve
    except Exception:
        raise RuntimeError("scikit-learn required to compute optimal threshold by F1.")

    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(probs).ravel()
    if len(np.unique(y)) == 1:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    # thr has length len(prec)-1
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    # align f1 to thresholds: drop the last prec/rec which don't have thresholds
    f1 = f1[:-1]
    if len(f1) == 0:
        return 0.5
    idx = int(np.nanargmax(f1))
    return float(thr[idx])


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(probs).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_p = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / float(len(p))) * abs(acc - avg_p)
    return float(ece)


class PatientStateManager:
    """Manage per-patient baselines, trends and map probs to nurse-facing states.

    States per axis: 'green' (within baseline), 'orange' (near limit), 'red' (out-of-range),
    'establishing' (not enough history to form baseline).

    Usage:
        mgr = PatientStateManager()
        state = mgr.update_and_get_state(patient_id, wheeze_prob, crackle_prob, timestamp)
    """

    def __init__(self, ema_alpha: float = 0.1, low_delta: float = 0.10, high_delta: float = 0.25,
                 min_samples_for_baseline: int = 5, trend_window: int = 5, max_history: int = 100,
                 force_established_after_s: Optional[float] = None):
        self.ema_alpha = float(ema_alpha)
        self.low_delta = float(low_delta)
        self.high_delta = float(high_delta)
        self.min_samples = int(min_samples_for_baseline)
        self.trend_window = int(trend_window)
        self.max_history = int(max_history)
        self.force_established_after_s = None if force_established_after_s is None else float(force_established_after_s)
        self._store: Dict[str, Dict[str, Any]] = {}

    def _init_patient(self, pid: str):
        self._store[pid] = {
            'ema_w': None,
            'ema_c': None,
            'count': 0,
            'recent_w': deque(maxlen=self.max_history),
            'recent_c': deque(maxlen=self.max_history),
            'timestamps': deque(maxlen=self.max_history),
            'first_ts': None,
        }

    def _update_ema(self, prev: Optional[float], value: float) -> float:
        if prev is None:
            return float(value)
        return float(prev * (1.0 - self.ema_alpha) + float(value) * self.ema_alpha)

    def _compute_trend(self, recent: deque) -> float:
        # Simple linear trend: diff between last and first over window length
        if len(recent) < 2:
            return 0.0
        return float(recent[-1]) - float(recent[0])

    def _axis_state(self, baseline: Optional[float], value: float) -> Tuple[str, float]:
        if baseline is None:
            return 'establishing', 0.0
        delta = float(value) - float(baseline)
        ad = abs(delta)
        if ad <= self.low_delta:
            return 'green', delta
        if ad <= self.high_delta:
            return 'orange', delta
        return 'red', delta

    def update_and_get_state(self, patient_id: str, wheeze_prob: float, crackle_prob: float,
                             timestamp: Optional[float] = None) -> Dict[str, Any]:
        if patient_id is None:
            patient_id = '__anon__'
        if patient_id not in self._store:
            self._init_patient(patient_id)

        entry = self._store[patient_id]
        now_ts = float(timestamp) if timestamp is not None else time.time()
        entry['count'] += 1
        entry['recent_w'].append(float(wheeze_prob))
        entry['recent_c'].append(float(crackle_prob))
        entry['timestamps'].append(now_ts)
        if entry['first_ts'] is None:
            entry['first_ts'] = now_ts

        # update EMA baselines
        entry['ema_w'] = self._update_ema(entry['ema_w'], float(wheeze_prob))
        entry['ema_c'] = self._update_ema(entry['ema_c'], float(crackle_prob))

        force_established = False
        if self.force_established_after_s is not None and entry['first_ts'] is not None:
            if (now_ts - entry['first_ts']) >= self.force_established_after_s:
                force_established = True

        baseline_w = entry['ema_w'] if (entry['count'] >= self.min_samples or force_established) else None
        baseline_c = entry['ema_c'] if (entry['count'] >= self.min_samples or force_established) else None

        state_w, delta_w = self._axis_state(baseline_w, float(wheeze_prob))
        state_c, delta_c = self._axis_state(baseline_c, float(crackle_prob))

        trend_w = self._compute_trend(list(entry['recent_w'])[-self.trend_window:])
        trend_c = self._compute_trend(list(entry['recent_c'])[-self.trend_window:])

        # overall severity: red > orange > green > establishing
        severity_rank = {'establishing': 0, 'green': 1, 'orange': 2, 'red': 3}
        overall = 'green'
        # choose worst of two axes
        if severity_rank[state_w] >= severity_rank[state_c]:
            overall = state_w
        else:
            overall = state_c

        reason = {
            'wheeze': {
                'baseline': baseline_w,
                'value': float(wheeze_prob),
                'delta': float(delta_w),
                'trend': float(trend_w),
                'state': state_w,
            },
            'crackle': {
                'baseline': baseline_c,
                'value': float(crackle_prob),
                'delta': float(delta_c),
                'trend': float(trend_c),
                'state': state_c,
            },
            'count': int(entry['count'])
        }

        return {
            'overall_state': overall,
            'reason': reason,
        }


def predict_with_metadata(
    model,
    x_array: np.ndarray,
    edge_index_array: np.ndarray,
    device: str = "cpu",
    calibrator: Optional[Dict[str, Any]] = None,
    thresholds: Tuple[float, float] = (0.5, 0.5),
    audio_id: Optional[str] = None,
    audio_duration_s: Optional[float] = None,
    model_version: Optional[str] = None,
    patient_id: Optional[str] = None,
    state_manager: Optional[PatientStateManager] = None,
) -> Dict[str, Any]:
    """High-level inference that returns calibrated probs, preds, uncertainties and metadata.

    calibrator: either None or a dict with keys 'wheeze' and/or 'crackle' mapping to Platt params (coef/intercept).
    thresholds: (wheeze_threshold, crackle_threshold)
    """
    dev = torch.device(device)
    start = time.time()
    w_prob, c_prob, w_logits, c_logits = predict_from_arrays(model, x_array, edge_index_array, device=device, return_logits=True)
    infer_ms = (time.time() - start) * 1000.0

    # flatten to scalars if single-value
    w_prob = float(np.asarray(w_prob).ravel()[0])
    c_prob = float(np.asarray(c_prob).ravel()[0])

    # apply calibrator if provided
    if calibrator is not None:
        if isinstance(calibrator, dict):
            if 'wheeze' in calibrator:
                w_prob = float(apply_platt_scaler(np.array([w_prob]), calibrator['wheeze'])[0])
            if 'crackle' in calibrator:
                c_prob = float(apply_platt_scaler(np.array([c_prob]), calibrator['crackle'])[0])
        elif callable(calibrator):
            w_prob, c_prob = calibrator(w_prob, c_prob)

    w_thr, c_thr = thresholds
    w_pred = int(w_prob >= float(w_thr))
    c_pred = int(c_prob >= float(c_thr))

    # uncertainty from learned log-variance (if present)
    w_unc = None
    c_unc = None
    w_conf = None
    c_conf = None
    try:
        if hasattr(model, 'log_var_wheeze'):
            w_logvar = float(model.log_var_wheeze.detach().cpu().item())
            w_std = float(np.exp(0.5 * w_logvar))
            w_unc = w_std
            w_conf = float(1.0 / (1.0 + w_std))
        if hasattr(model, 'log_var_crackle'):
            c_logvar = float(model.log_var_crackle.detach().cpu().item())
            c_std = float(np.exp(0.5 * c_logvar))
            c_unc = c_std
            c_conf = float(1.0 / (1.0 + c_std))
    except Exception:
        pass

    meta = {
        'audio_id': audio_id,
        'audio_duration_s': audio_duration_s,
        'model_version': model_version,
        'inference_time_ms': float(infer_ms),
    }

    # patient state reasoning (nurse-facing): uses PatientStateManager if provided
    patient_state = None
    patient_state_reason = None
    try:
        if state_manager is not None and callable(getattr(state_manager, 'update_and_get_state', None)):
            ts = time.time()
            st = state_manager.update_and_get_state(patient_id, w_prob, c_prob, timestamp=ts)
            patient_state = st.get('overall_state')
            patient_state_reason = st.get('reason')
    except Exception:
        # fail-safe: do not break inference
        patient_state = None
        patient_state_reason = None

    return {
        'wheeze_prob': w_prob,
        'crackle_prob': c_prob,
        'wheeze_pred': w_pred,
        'crackle_pred': c_pred,
        'wheeze_uncertainty': w_unc,
        'crackle_uncertainty': c_unc,
        'wheeze_confidence': w_conf,
        'crackle_confidence': c_conf,
        'metadata': meta,
        'patient_state': patient_state,
        'patient_state_reason': patient_state_reason,
    }
