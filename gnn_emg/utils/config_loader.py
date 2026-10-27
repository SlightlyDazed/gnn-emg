"""
GNN-EMG — Config Loader
Lee configs/config.yaml y expone como namespace accesible.
"""
from __future__ import annotations
import os
import yaml
from pathlib import Path
from types import SimpleNamespace


def _to_namespace(d):
    """Recursivamente convierte dict → SimpleNamespace para acceso con punto."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    return d


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    """
    Carga el archivo YAML de configuración.
    Si path es None, busca configs/config.yaml relativo al paquete gnn_emg/.
    """
    if path is None:
        # Busca el config relativo a este archivo (utils/config_loader.py)
        pkg_root = Path(__file__).parent.parent          # gnn_emg/
        path = pkg_root / "configs" / "config.yaml"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config no encontrado: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = _to_namespace(raw)
    return cfg
