"""
GNN-EMG — Signal Utilities
Preprocesamiento EMG + extracción de features por ventana.

Funciones principales:
  bandpass_filter       → Butterworth 20-150 Hz
  notch_filter          → Notch 60 Hz (o 50 Hz)
  extract_features_win  → vector de 8 features por canal/ventana
  compute_corr_matrix   → matriz de correlación entre canales (→ aristas del grafo)
  normalize_channels    → Z-score por canal
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, iirnotch, sosfiltfilt


# ── Filtros ──────────────────────────────────────────────────────────────────

def bandpass_filter(
    signal: np.ndarray,  # (T,) o (T, C)
    fs: float = 200.0,
    low: float = 20.0,
    high: float = 150.0,
    order: int = 4,
) -> np.ndarray:
    """Filtro Butterworth pasa-banda. Opera sobre el eje 0 (tiempo)."""
    nyq = 0.5 * fs
    lo  = max(low  / nyq, 1e-4)
    hi  = min(high / nyq, 1 - 1e-4)
    sos = butter(order, [lo, hi], btype="band", output="sos")
    if signal.ndim == 1:
        return sosfiltfilt(sos, signal)
    # (T, C) → filtrar cada canal
    return np.stack([sosfiltfilt(sos, signal[:, c]) for c in range(signal.shape[1])], axis=1)


def notch_filter(
    signal: np.ndarray,
    fs: float = 200.0,
    f0: float = 60.0,
    Q: float = 30.0,
) -> np.ndarray:
    """Filtro notch para eliminar interferencia de línea eléctrica."""
    b, a = iirnotch(f0 / (0.5 * fs), Q)
    if signal.ndim == 1:
        return np.convolve(signal, b, mode="same")   # aproximación simple
    return np.stack([np.convolve(signal[:, c], b, mode="same")
                     for c in range(signal.shape[1])], axis=1)


# ── Extracción de features por ventana ───────────────────────────────────────

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))

def _mav(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))

def _wl(x: np.ndarray) -> float:
    """Waveform Length — proxy de frecuencia."""
    return float(np.sum(np.abs(np.diff(x))))

def _zc(x: np.ndarray, threshold: float = 0.0) -> float:
    """Zero Crossings con umbral de ruido."""
    xc = x - np.mean(x)
    signs = np.sign(xc)
    crossings = np.where(np.diff(signs))[0]
    if threshold > 0:
        crossings = [i for i in crossings if abs(xc[i + 1] - xc[i]) > threshold]
    return float(len(crossings))

def _tkeo(x: np.ndarray) -> float:
    """Teager-Kaiser Energy Operator (valor medio sobre la ventana)."""
    if len(x) < 3:
        return 0.0
    e = x[1:-1] ** 2 - x[:-2] * x[2:]
    return float(np.mean(np.maximum(e, 0)))

def _ssc(x: np.ndarray) -> float:
    """Slope Sign Changes."""
    d = np.diff(x)
    return float(np.sum(np.diff(np.sign(d)) != 0))

def _kurt(x: np.ndarray) -> float:
    """Kurtosis — detecta artefactos e impulsos."""
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-9:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 4))

def _skew(x: np.ndarray) -> float:
    """Skewness — asimetría de la distribución de amplitudes."""
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-9:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 3))


FEATURE_NAMES = ["RMS", "MAV", "WL", "ZC", "TKEO", "KURT", "SKEW", "SSC"]
N_FEATURES    = len(FEATURE_NAMES)   # 8


def extract_features_window(window: np.ndarray) -> np.ndarray:
    """
    Extrae el vector de 8 features EMG de una ventana 1-D (un solo canal).

    Parameters
    ----------
    window : np.ndarray  shape (T,)
        Señal EMG de un canal en la ventana.

    Returns
    -------
    np.ndarray  shape (8,)
        [RMS, MAV, WL, ZC, TKEO, KURT, SKEW, SSC]
    """
    w = np.asarray(window, dtype=np.float64)
    return np.array([
        _rms(w), _mav(w), _wl(w), _zc(w),
        _tkeo(w), _kurt(w), _skew(w), _ssc(w),
    ], dtype=np.float32)


def extract_features_multichannel(
    emg: np.ndarray,  # (T, C)
) -> np.ndarray:
    """
    Extrae features de una ventana multi-canal.

    Returns
    -------
    np.ndarray  shape (C, 8)
        Matriz de features — una fila por canal, una columna por feature.
    """
    C = emg.shape[1] if emg.ndim == 2 else 1
    if emg.ndim == 1:
        emg = emg[:, np.newaxis]
    feats = np.zeros((C, N_FEATURES), dtype=np.float32)
    for c in range(C):
        feats[c] = extract_features_window(emg[:, c])
    return feats


# ── Correlación inter-canales → aristas del grafo ────────────────────────────

def compute_correlation_matrix(
    emg: np.ndarray,      # (T, C)
    threshold: float = 0.3,
    absolute: bool  = True,
) -> np.ndarray:
    """
    Calcula la matriz de correlación de Pearson entre canales EMG y
    aplica un umbral para obtener la matriz de adyacencia del grafo.

    Parameters
    ----------
    emg       : (T, C) señal multi-canal en la ventana
    threshold : correlación mínima para crear arista
    absolute  : si True, usa |correlación| (correlación negativa también conecta)

    Returns
    -------
    adj : np.ndarray  shape (C, C)  — matriz de adyacencia 0/1
    wts : np.ndarray  shape (C, C)  — pesos de las aristas (|corr|)
    """
    if emg.ndim == 1:
        return np.ones((1, 1), dtype=np.float32), np.ones((1, 1), dtype=np.float32)

    C = emg.shape[1]
    # Normalizar antes de correlación
    std = np.std(emg, axis=0, keepdims=True)
    std = np.where(std < 1e-9, 1.0, std)
    normed = (emg - np.mean(emg, axis=0, keepdims=True)) / std

    corr = (normed.T @ normed) / max(emg.shape[0] - 1, 1)
    corr = np.clip(corr, -1.0, 1.0)

    wts  = np.abs(corr) if absolute else corr
    adj  = (wts >= threshold).astype(np.float32)
    np.fill_diagonal(adj, 1.0)   # auto-aristas siempre

    return adj, wts.astype(np.float32)


# ── Normalización ─────────────────────────────────────────────────────────────

def normalize_channels(
    features: np.ndarray,   # (C, F) o (N_windows, C, F)
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Z-score por canal (eje C).
    Si shape es (C, F): normaliza sobre F (por feature dentro del canal).
    """
    mu  = features.mean(axis=-1, keepdims=True)
    sig = features.std(axis=-1, keepdims=True) + eps
    return (features - mu) / sig


# ── Sliding window ────────────────────────────────────────────────────────────

def sliding_windows(
    emg: np.ndarray,   # (T, C) o (T,)
    win_samples: int,
    stride_samples: int,
    labels: np.ndarray | None = None,  # (T,)
) -> list[dict]:
    """
    Genera ventanas deslizantes sobre la señal EMG.

    Returns
    -------
    list of dicts con keys: 'emg' (win, C), 'label' (int o -1)
    """
    T = emg.shape[0]
    if emg.ndim == 1:
        emg = emg[:, np.newaxis]

    windows = []
    for start in range(0, T - win_samples + 1, stride_samples):
        end   = start + win_samples
        chunk = emg[start:end]

        # Etiqueta: clase mayoritaria en la ventana
        if labels is not None:
            lbl_chunk = labels[start:end]
            # Modo de la ventana, ignorando -1 (sin gesto)
            valid = lbl_chunk[lbl_chunk >= 0]
            if len(valid) == 0:
                lbl = -1
            else:
                vals, counts = np.unique(valid, return_counts=True)
                lbl = int(vals[np.argmax(counts)])
        else:
            lbl = -1

        windows.append({"emg": chunk, "label": lbl})

    return windows
