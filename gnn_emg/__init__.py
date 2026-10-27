"""GNN-EMG — Paquete principal."""
from .models  import EMGGraphAutoencoder
from .data    import NinaProDataset, SichirayDataset, GESTURE_MAP
from .train   import pretrain, finetune
from .inference import run_realtime

__version__ = "0.1.0"
__all__ = [
    "EMGGraphAutoencoder",
    "NinaProDataset", "SichirayDataset", "GESTURE_MAP",
    "pretrain", "finetune",
    "run_realtime",
]
