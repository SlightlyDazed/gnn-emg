"""
GNN-EMG — NinaPro DB5 Dataset Loader
Carga archivos .mat de NinaPro DB5 y construye grafos PyG para el GATv2 autoencoder.

Estructura DB5:
  - 10 sujetos sanos
  - 16 canales EMG (2× Myo armband)
  - 200 Hz
  - Ejercicios E1 (12 gestos basicos), E2 (8 prono-sup), E3 (9 agarre)
  - Variables: emg, stimulus, restimulus, repetition, rerepetition, glove, acc

Cada muestra del dataset = un grafo:
  - Nodos: 16 canales EMG
  - Features de nodo: 8 features por canal [RMS, MAV, WL, ZC, TKEO, KURT, SKEW, SSC]
  - Aristas: correlación inter-canal (|Pearson| >= threshold)
  - y: clase de gesto (restimulus)
"""
from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio
import torch
from torch_geometric.data import Data, Dataset
from torch.utils.data import random_split

from ..utils.signal_utils import (
    bandpass_filter, notch_filter,
    extract_features_multichannel,
    compute_correlation_matrix,
    sliding_windows,
)


# ─────────────────────────────────────────────────────────────────────────────

def _load_mat(path: str) -> dict:
    """Carga un archivo .mat (NinaPro) de forma robusta."""
    try:
        mat = sio.loadmat(path, simplify_cells=True)
        return mat
    except Exception as e:
        raise IOError(f"Error leyendo {path}: {e}")


def _get_subject_files(db_root: str, exercises: list[int]) -> list[str]:
    """Devuelve todos los .mat válidos para los ejercicios pedidos."""
    files = []
    ex_tags = [f"E{e}" for e in exercises]
    for tag in ex_tags:
        pattern = os.path.join(db_root, "**", f"*_{tag}_*.mat")
        found   = glob.glob(pattern, recursive=True)
        # Filtrar archivos ocultos (._*.mat) que causan errores
        found   = [f for f in found if not os.path.basename(f).startswith("._")]
        files.extend(found)
    return sorted(files)


def load_ninapro_file(
    path: str,
    fs: float          = 200.0,
    win_samples: int   = 40,
    stride_samples: int= 20,
    corr_threshold: float = 0.3,
    use_restimulus: bool  = True,
    exclude_rest: bool    = False,
) -> list[Data]:
    """
    Carga un único archivo .mat de NinaPro y genera una lista de grafos PyG.

    Returns
    -------
    list[Data]  — cada Data tiene:
        x          : Tensor (16, 8)   features de nodo (8 features por canal)
        edge_index : Tensor (2, E)    índices de aristas
        edge_attr  : Tensor (E, 1)    peso de arista (|correlación|)
        y          : Tensor (1,)      clase de gesto
        subject    : int
        exercise   : int
    """
    mat = _load_mat(path)

    emg_raw = np.array(mat["emg"], dtype=np.float64)          # (T, 16)
    stim_key = "restimulus" if (use_restimulus and "restimulus" in mat) else "stimulus"
    labels  = np.array(mat[stim_key], dtype=np.int32).ravel() # (T,)
    subject  = int(mat.get("subject", -1))
    exercise = int(mat.get("exercise", -1))

    # ── Preprocesamiento EMG ──────────────────────────────────────────────────
    emg_filt = bandpass_filter(emg_raw, fs=fs, low=20.0, high=95.0)
    # Nota: Myo @ 200Hz → Nyquist = 100Hz; limite alto a 95Hz

    # ── Ventanas deslizantes ──────────────────────────────────────────────────
    wins = sliding_windows(emg_filt, win_samples, stride_samples, labels)

    graphs = []
    for w in wins:
        lbl = w["label"]
        if exclude_rest and lbl == 0:
            continue
        if lbl < 0:
            continue   # ventana sin gesto claro → descartar

        emg_win = w["emg"]  # (win_samples, 16)

        # ── Features de nodo: EMG CRUDO ──────────────────────────────────────
        # Cada nodo (canal EMG) recibe las 40 muestras crudas de la ventana
        # como features directamente. El GNN aprenderá sus propias features
        # temporales via atención — sin bottleneck de 8 features estadísticas.
        node_feats = emg_win.T.astype(np.float32)  # (16, win_samples=40)

        # Normalización GLOBAL: preserva amplitudes relativas entre canales
        # (canal activo debe verse diferente de canal inactivo — eso es la señal)
        global_std = node_feats.std() + 1e-8
        global_mean = node_feats.mean()
        node_feats = (node_feats - global_mean) / global_std

        # Positional encoding del canal
        n_ch = node_feats.shape[0]
        chan_id = np.arange(n_ch, dtype=np.float32).reshape(-1, 1) / max(n_ch - 1, 1)
        node_feats = np.concatenate([node_feats, chan_id], axis=1)  # (16, 41)

        # ── Grafo: aristas por correlación ───────────────────────────────────
        adj, wts = compute_correlation_matrix(emg_win, threshold=corr_threshold)
        src, dst = np.where(adj > 0)
        edge_weights = wts[src, dst]

        x          = torch.tensor(node_feats, dtype=torch.float32)
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        edge_attr  = torch.tensor(edge_weights[:, np.newaxis], dtype=torch.float32)
        y          = torch.tensor([lbl], dtype=torch.long)

        graphs.append(Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
            subject=subject, exercise=exercise,
        ))

    return graphs


# ─────────────────────────────────────────────────────────────────────────────

class NinaProDataset(Dataset):
    """
    Dataset PyG para NinaPro DB5 (o cualquier DB con estructura similar).

    Carga todos los archivos .mat del directorio `db_root` y construye
    el conjunto de grafos EMG en memoria.

    Parámetros
    ----------
    db_root      : str | Path  — ruta a la carpeta de la DB (ej: ../NinaPro/db5)
    exercises    : list[int]   — ejercicios a incluir [1,2,3]
    win_ms       : int         — duración de ventana en ms (default 200)
    overlap_pct  : float       — solapamiento 0..1 (default 0.5)
    fs           : float       — frecuencia de muestreo Hz
    corr_thr     : float       — umbral de correlación para aristas
    use_restim   : bool        — usar restimulus (más limpio)
    exclude_rest : bool        — excluir clase 0 (reposo)
    subjects     : list[int] | None  — None = todos
    """

    def __init__(
        self,
        db_root: str | Path,
        exercises: list[int]    = [1, 2, 3],
        win_ms: int             = 200,
        overlap_pct: float      = 0.5,
        fs: float               = 200.0,
        corr_thr: float         = 0.3,
        use_restim: bool        = True,
        exclude_rest: bool      = False,
        subjects: list[int] | None = None,
    ):
        super().__init__()
        self.db_root    = Path(db_root)
        self.fs         = fs
        win_samples     = int(fs * win_ms / 1000)
        stride_samples  = max(1, int(win_samples * (1 - overlap_pct)))

        print(f"[NinaPro] Escaneando: {self.db_root}")
        mat_files = _get_subject_files(str(self.db_root), exercises)
        print(f"[NinaPro] {len(mat_files)} archivos .mat encontrados")

        self._graphs: list[Data] = []
        for path in mat_files:
            # Filtrar por sujeto si se especificó
            if subjects is not None:
                bname = os.path.basename(path)
                # extrae número de sujeto del nombre (S1_, S10_, etc.)
                try:
                    snum = int(bname.split("_")[0].lstrip("Ss"))
                except ValueError:
                    snum = -1
                if snum not in subjects:
                    continue

            try:
                gs = load_ninapro_file(
                    path, fs=fs,
                    win_samples=win_samples,
                    stride_samples=stride_samples,
                    corr_threshold=corr_thr,
                    use_restimulus=use_restim,
                    exclude_rest=exclude_rest,
                )
                self._graphs.extend(gs)
                print(f"  ✓ {os.path.basename(path):30s} → {len(gs):5d} grafos")
            except Exception as e:
                print(f"  ✗ {os.path.basename(path):30s} — Error: {e}")

        # ── Remapear etiquetas a rango [0, N-1] ──────────────────────────────
        all_labels = sorted(set(int(g.y.item()) for g in self._graphs))
        self._label_map = {orig: new for new, orig in enumerate(all_labels)}
        for g in self._graphs:
            g.y = torch.tensor([self._label_map[int(g.y.item())]], dtype=torch.long)

        self.n_classes       = len(all_labels)
        self.n_node_features = self._graphs[0].x.shape[1] if self._graphs else 8

        print(f"\n[NinaPro] Dataset listo: {len(self._graphs):,} grafos | "
              f"{self.n_classes} clases | "
              f"features/nodo: {self.n_node_features}")

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> Data:
        return self._graphs[idx]

    def split(self, val_pct: float = 0.15, seed: int = 42):
        """Divide en train / val por porcentaje."""
        n_val   = max(1, int(len(self) * val_pct))
        n_train = len(self) - n_val
        return random_split(self, [n_train, n_val],
                            generator=torch.Generator().manual_seed(seed))


# ─────────────────────────────────────────────────────────────────────────────
#  Utilidad rápida para verificar que el dataset se carga bien
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "../NinaPro/db5"
    ds = NinaProDataset(
        db_root=db_path,
        exercises=[1],    # solo E1 para prueba rápida
        subjects=[1, 2],  # solo 2 sujetos
    )
    g = ds[0]
    print(f"\nEjemplo de grafo:")
    print(f"  x.shape        : {g.x.shape}")
    print(f"  edge_index.shape: {g.edge_index.shape}")
    print(f"  edge_attr.shape : {g.edge_attr.shape}")
    print(f"  y              : {g.y.item()} (gesto clase {g.y.item()})")
    print(f"  subject / exer : {g.subject} / {g.exercise}")
