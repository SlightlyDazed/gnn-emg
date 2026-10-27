"""
GNN-EMG — Sichiray Dataset Loader
Carga los CSVs de tu captura personal (ESP32 + MediaPipe) y construye
grafos PyG con NODOS VIRTUALES para el GATv2 autoencoder.

Estrategia de grafización (1 canal EMG → multi-nodo):
  Agrupamos las features en 5 "nodos funcionales":
    0: RAW_EMG    → [emg_raw, emg_smooth, emg_filt]
    1: ENERGY     → [emg_tkeo, emg_rms, emg_mav]
    2: COMPLEXITY → [emg_wl, emg_zc, emg_envelope, emg_rect]
    3: THRESHOLD  → [muscle_active, dt_active, active_pct, snr_db]
    4: HAND_KINE  → [f_thumb, f_index, f_middle, f_ring, f_pinky, pinch, hand_speed]

  Las aristas conectan nodos con alta correlación de sus features.
  Esto crea un "grafo de interacciones señal-gesto" que el GATv2 puede
  aprender a decodificar en un embedding de gesto.

Domain gap con NinaPro:
  - NinaPro: 16 nodos reales (canales EMG)   features (16, 8)
  - Sichiray: 5 nodos virtuales              features (5, max_feat)
  - Se usa una capa de proyección en el modelo para mapear a dimensión uniforme.
"""
from __future__ import annotations

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch.utils.data import random_split

from ..utils.signal_utils import compute_correlation_matrix


# ─────────────────────────────────────────────────────────────────────────────
#  Grupos de columnas → nodos virtuales
# ─────────────────────────────────────────────────────────────────────────────

VIRTUAL_NODES = {
    "raw_emg":    ["emg_raw", "emg_smooth", "emg_filt"],
    "energy":     ["emg_tkeo", "emg_rms", "emg_mav"],
    "complexity": ["emg_wl", "emg_zc", "emg_envelope", "emg_rect"],
    "threshold":  ["muscle_active", "dt_active", "active_pct", "snr_db"],
    "hand_kine":  ["f_thumb", "f_index", "f_middle", "f_ring", "f_pinky",
                   "pinch", "hand_speed"],
}

NODE_NAMES  = list(VIRTUAL_NODES.keys())
N_VNODES    = len(NODE_NAMES)               # 5

# Máximo features por nodo (para padding)
MAX_FEAT_PER_NODE = max(len(v) for v in VIRTUAL_NODES.values())  # 7

GESTURE_MAP = {
    "ABIERTA":   0, "PUNIO":     1, "INDICE":  2,
    "PAZ":       3, "OK":        4, "PULGAR_UP": 5,
    "PULGAR_DN": 6, "TRES":      7, "PINZA":   8,
    "TELEFONO":  9, "OTRO":      10,
}


# ─────────────────────────────────────────────────────────────────────────────

def _normalize_row(arr: np.ndarray) -> np.ndarray:
    """Normaliza un vector 1-D por Z-score."""
    mu  = arr.mean()
    sig = arr.std() + 1e-8
    return (arr - mu) / sig


def _build_node_features(row: pd.Series) -> np.ndarray:
    """
    Construye la matriz de features de nodo (5, MAX_FEAT_PER_NODE)
    para una sola fila del CSV sichiray.

    Features con menos columnas se rellenan con 0 (padding).
    """
    feats = np.zeros((N_VNODES, MAX_FEAT_PER_NODE), dtype=np.float32)
    for i, (node_name, cols) in enumerate(VIRTUAL_NODES.items()):
        vals = []
        for c in cols:
            # Enmascarar (poner a cero) los features de la cámara en la entrada
            # para obligar al modelo a aprender la relación EMG -> Gesto.
            if node_name == "hand_kine":
                vals.append(0.0)
            elif c in row.index:
                v = row[c]
                vals.append(float(v) if not np.isnan(float(v)) else 0.0)
            else:
                vals.append(0.0)
        arr = np.array(vals, dtype=np.float32)
        # Normalización local por nodo
        mu  = arr.mean()
        sig = arr.std() + 1e-8
        arr = (arr - mu) / sig
        feats[i, :len(arr)] = arr
    return feats


def _build_virtual_graph(
    feat_matrix: np.ndarray,   # (5, MAX_FEAT_PER_NODE)
    corr_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye aristas entre nodos virtuales usando correlación de filas
    (cada fila = vector de features de un nodo).
    """
    # Correlación entre vectores de nodos: (5, 5)
    C  = feat_matrix  # (N_VNODES, F)
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    Cn   = C / norms
    corr = Cn @ Cn.T    # (5, 5)
    corr = np.clip(corr, -1.0, 1.0)

    adj  = (np.abs(corr) >= corr_threshold).astype(np.float32)
    np.fill_diagonal(adj, 1.0)   # auto-aristas

    return adj, np.abs(corr).astype(np.float32)


def _rows_to_graph(
    rows: pd.DataFrame,
    corr_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Agrega N filas del CSV en un único grafo:
    - Promedia los valores de features a lo largo de la ventana
    - Construye la matriz de features de nodo con los promedios
    - Calcula aristas por correlación entre nodos virtuales
    """
    # Promediar columnas numéricas en la ventana temporal
    numeric = rows.select_dtypes(include=[np.number]).mean()

    feat_matrix = _build_node_features(numeric)            # (5, 7)
    adj, wts    = _build_virtual_graph(feat_matrix, corr_threshold)

    src, dst    = np.where(adj > 0)
    edge_weights = wts[src, dst]

    return feat_matrix, np.stack([src, dst]), edge_weights


# ─────────────────────────────────────────────────────────────────────────────

def load_sichiray_file(
    path: str,
    gesture_map: dict      = GESTURE_MAP,
    win_rows: int          = 40,       # filas = muestras EMG @ 200Hz → 200ms
    stride_rows: int       = 20,
    corr_threshold: float  = 0.3,
    min_confidence: float  = 0.0,      # 0 = aceptar todo
    exclude_otro: bool     = True,     # excluir gesto "OTRO" (poco confiable)
) -> list[Data]:
    """
    Carga un CSV sichiray y devuelve lista de grafos PyG.

    Returns
    -------
    list[Data]  donde cada Data tiene:
        x          : (5, 7)   features por nodo virtual
        edge_index : (2, E)   aristas por correlación
        edge_attr  : (E, 1)   pesos de aristas
        y          : (1,)     clase de gesto (GESTURE_MAP)
        source     : 'sichiray'
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  ✗ Error leyendo {os.path.basename(path)}: {e}")
        return []

    # Verificar columnas mínimas
    required = ["emg_raw", "gesture"]
    if not all(c in df.columns for c in required):
        print(f"  ✗ {os.path.basename(path)}: columnas faltantes")
        return []

    # Filtrar filas con gesto inválido
    df = df[df["gesture"].isin(gesture_map.keys())]
    if exclude_otro:
        df = df[df["gesture"] != "OTRO"]
    if "confidence" in df.columns and min_confidence > 0:
        df = df[df["confidence"] >= min_confidence]

    df = df.reset_index(drop=True)
    if len(df) < win_rows:
        return []

    graphs = []
    for start in range(0, len(df) - win_rows + 1, stride_rows):
        end   = start + win_rows
        chunk = df.iloc[start:end]

        # Clase mayoritaria en la ventana
        gest_counts = chunk["gesture"].value_counts()
        if gest_counts.empty:
            continue
        gest_name = gest_counts.index[0]
        lbl = gesture_map.get(gest_name, -1)
        if lbl < 0:
            continue

        feat_matrix, edge_idx, edge_wts = _rows_to_graph(chunk, corr_threshold)

        x          = torch.tensor(feat_matrix, dtype=torch.float32)
        edge_index = torch.tensor(edge_idx,    dtype=torch.long)
        edge_attr  = torch.tensor(edge_wts[:, np.newaxis], dtype=torch.float32)
        y          = torch.tensor([lbl],        dtype=torch.long)

        graphs.append(Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
            source="sichiray",
            gesture_name=gest_name,
        ))

    return graphs


# ─────────────────────────────────────────────────────────────────────────────

class SichirayDataset(Dataset):
    """
    Dataset PyG para los datos personales capturados con sichiray.

    Carga todos los CSVs del directorio `csv_root` que superen `min_rows`.
    """

    def __init__(
        self,
        csv_root: str | Path,
        win_ms: int         = 200,
        overlap_pct: float  = 0.5,
        fs: float           = 200.0,
        corr_thr: float     = 0.3,
        min_rows: int       = 500,
        exclude_otro: bool  = True,
        gesture_map: dict   = GESTURE_MAP,
    ):
        super().__init__()
        self.csv_root    = Path(csv_root)
        self.gesture_map = gesture_map

        win_rows    = int(fs * win_ms / 1000)
        stride_rows = max(1, int(win_rows * (1 - overlap_pct)))

        csv_files = sorted(self.csv_root.glob("*.csv"))
        print(f"[Sichiray] {len(csv_files)} CSVs encontrados en {self.csv_root}")

        self._graphs: list[Data] = []
        for path in csv_files:
            # Descartar archivos muy pequeños
            if path.stat().st_size < min_rows * 50:   # ~50 bytes por fila
                print(f"  ⚠ {path.name}: muy pequeño, ignorado")
                continue
            gs = load_sichiray_file(
                str(path),
                gesture_map=gesture_map,
                win_rows=win_rows,
                stride_rows=stride_rows,
                corr_threshold=corr_thr,
                exclude_otro=exclude_otro,
            )
            self._graphs.extend(gs)
            print(f"  ✓ {path.name:45s} → {len(gs):5d} grafos")

        self.n_classes       = len(gesture_map)
        self.n_node_features = MAX_FEAT_PER_NODE  # 7

        # Distribución de clases
        class_counts = {}
        for g in self._graphs:
            lbl = int(g.y.item())
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

        print(f"\n[Sichiray] Dataset listo: {len(self._graphs):,} grafos | "
              f"{self.n_classes} clases posibles")
        if class_counts:
            inv_map = {v: k for k, v in gesture_map.items()}
            for lbl, cnt in sorted(class_counts.items()):
                print(f"  {inv_map.get(lbl,'?'):12s} (clase {lbl:2d}): {cnt:5d} grafos")

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> Data:
        return self._graphs[idx]

    def split(self, val_pct: float = 0.20, seed: int = 42):
        n_val   = max(1, int(len(self) * val_pct))
        n_train = len(self) - n_val
        return random_split(self, [n_train, n_val],
                            generator=torch.Generator().manual_seed(seed))

    @property
    def class_weights(self) -> torch.Tensor:
        """Pesos inversamente proporcionales a la frecuencia de cada clase."""
        counts = torch.zeros(self.n_classes)
        for g in self._graphs:
            counts[g.y.item()] += 1
        counts = torch.where(counts == 0, torch.ones_like(counts), counts)
        return (1.0 / counts) / (1.0 / counts).sum()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "../sichiray"
    ds = SichirayDataset(csv_root=csv_path)
    if len(ds) > 0:
        g = ds[0]
        print(f"\nEjemplo de grafo Sichiray:")
        print(f"  x.shape        : {g.x.shape}")      # (5, 7)
        print(f"  edge_index     : {g.edge_index.shape}")
        print(f"  y              : {g.y.item()} ({g.gesture_name})")
