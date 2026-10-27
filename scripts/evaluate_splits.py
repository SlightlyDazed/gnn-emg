"""Evaluate the Sichiray gesture classifier with three protocols.

1. The released checkpoint on the original random validation split, against the majority class.
2. The same fine-tuning procedure (fixed number of epochs) with a random split and with a
   5-fold blocked-in-time split (contiguous blocks, overlapping windows at the edges removed).
3. The released checkpoint fed with the features that gnn_emg/inference/realtime.py computes.

Usage (from the repository root):
    python scripts/evaluate_splits.py
"""
import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gnn_emg.data import SichirayDataset, GESTURE_MAP            # noqa: E402
from gnn_emg.data import sichiray_dataset as sd                  # noqa: E402
from gnn_emg.models import EMGGraphAutoencoder                   # noqa: E402
from gnn_emg.train.pretrain_ninapro import _run_epoch            # noqa: E402
from gnn_emg.utils import load_config                            # noqa: E402

torch.manual_seed(0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG = load_config()
CFG_MAP = dict(CFG.sichiray.gesture_map.__dict__)
CKPT_DIR = ROOT / "gnn_emg" / "checkpoints"
INV = {v: k for k, v in GESTURE_MAP.items()}
N_NINA = torch.load(CKPT_DIR / "sich_best.pt", map_location="cpu",
                    weights_only=False)["model"]["nina_gesture_head.net.3.weight"].shape[0]


def build_model():
    m = CFG.model
    return EMGGraphAutoencoder(
        nina_n_nodes=CFG.ninapro.n_channels, nina_in_feat=m.in_channels_nina, sich_n_nodes=5,
        sich_in_feat=m.in_channels_sich, hidden_channels=m.hidden_channels, latent_dim=m.latent_dim,
        num_layers=m.num_layers, heads=m.num_heads, dropout=m.dropout, use_bn=m.use_batch_norm,
        n_gestures_nina=N_NINA, n_gestures_sich=len(GESTURE_MAP), decoder_hidden=list(m.decoder_hidden),
    ).to(DEV)


def load(name):
    model = build_model()
    model.load_state_dict(torch.load(CKPT_DIR / name, map_location=DEV, weights_only=False)["model"], strict=False)
    return model


@torch.no_grad()
def predict(model, graphs):
    model.eval()
    return torch.cat([model(b.to(DEV), domain="sichiray")["logits"].argmax(-1).cpu()
                      for b in DataLoader(graphs, batch_size=512)]).numpy()


def report(name, y, p):
    bal = np.mean([(p[y == c] == c).mean() for c in np.unique(y)])
    dist = dict(collections.Counter(INV[i] for i in p).most_common())
    print(f"  {name:46s} acc={np.mean(p == y):.3f}  balanced={bal:.3f}  predicted={dist}")
    return np.mean(p == y), bal


def finetune(train_g, epochs=50, freeze=5):
    """Same two-phase procedure as gnn_emg/train/finetune_sichiray.py, with a fixed epoch count."""
    model = build_model()
    model.load_state_dict(torch.load(CKPT_DIR / "nina_best.pt", map_location=DEV, weights_only=False)["model"],
                          strict=False)
    loader = DataLoader(train_g, batch_size=CFG.finetune.batch_size, shuffle=True)
    lr, wd = float(CFG.finetune.lr), float(CFG.finetune.weight_decay)
    model.freeze_encoder()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    for _ in range(freeze):
        _run_epoch(model, loader, opt, DEV, "sichiray", 1.0, 1.0, 0.0, True)
    model.unfreeze_encoder()
    opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.5, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - freeze, eta_min=1e-7)
    for _ in range(epochs - freeze):
        _run_epoch(model, loader, opt, DEV, "sichiray", 1.0, 1.0, 0.0, True)
        sched.step()
    return model


SOS = butter(4, [20 / 100, 95 / 100], btype="band", output="sos")


def realtime_features(w):
    """Literal copy of GesturePredictor._emg_to_features in gnn_emg/inference/realtime.py."""
    w = w.astype(np.float64)
    filt = sosfiltfilt(SOS, w)
    rms = float(np.sqrt(np.mean(filt ** 2)))
    rect = float(np.mean(np.abs(filt)))
    return pd.Series({
        "emg_raw": float(np.mean(w)), "emg_smooth": float(np.convolve(w, np.ones(10) / 10, mode="valid").mean()),
        "emg_filt": float(np.mean(filt)),
        "emg_tkeo": float(np.mean(np.maximum(filt[1:-1] ** 2 - filt[:-2] * filt[2:], 0))),
        "emg_rms": rms, "emg_mav": rect, "emg_wl": float(np.sum(np.abs(np.diff(filt)))),
        "emg_zc": float(np.sum(np.diff(np.sign(filt)) != 0)), "emg_envelope": rect, "emg_rect": rect,
        "muscle_active": float(rms > 50), "dt_active": float(rms > 50), "active_pct": 50.0, "snr_db": 10.0,
        "f_thumb": 0, "f_index": 0, "f_middle": 0, "f_ring": 0, "f_pinky": 0, "pinch": 0.5, "hand_speed": 0.0,
    })


def main():
    ds = SichirayDataset(csv_root=ROOT / "dataset", win_ms=CFG.ninapro.window_ms, overlap_pct=0.5,
                         fs=CFG.sichiray.fs, corr_thr=CFG.graph.corr_threshold, gesture_map=CFG_MAP)
    graphs = [ds[i] for i in range(len(ds))]
    y = np.array([int(g.y) for g in graphs])
    tr, va = ds.split(val_pct=CFG.finetune.val_split, seed=42)
    tr_i, va_i = np.array(tr.indices), np.array(va.indices)

    print("\n1. Released checkpoint on the original random split")
    report("sich_best.pt", y[va_i], predict(load("sich_best.pt"), [graphs[i] for i in va_i]))
    majority = collections.Counter(y[tr_i]).most_common(1)[0][0]
    report("majority class (no model)", y[va_i], np.full(len(va_i), majority))

    print("\n2. Same fine-tuning, different validation")
    model = finetune([graphs[i] for i in tr_i])
    report("random split (overlapping windows)", y[va_i], predict(model, [graphs[i] for i in va_i]))
    idx, gap, accs, bals = np.arange(len(graphs)), 2, [], []
    for k, block in enumerate(np.array_split(idx, 5)):
        test = block[gap:-gap]
        train = np.setdiff1d(idx, np.arange(block[0] - gap, block[-1] + gap + 1))
        model = finetune([graphs[i] for i in train])
        a, b = report(f"time block {k + 1}/5", y[test], predict(model, [graphs[i] for i in test]))
        accs.append(a)
        bals.append(b)
    print(f"  blocked-in-time, mean of 5 folds: acc={np.mean(accs):.3f} ± {np.std(accs):.3f}  "
          f"balanced={np.mean(bals):.3f}")

    print("\n3. Released checkpoint with the features computed by realtime.py")
    csv = sorted((ROOT / "dataset").glob("*.csv"))[0]
    df = pd.read_csv(csv, low_memory=False)
    df = df[df["gesture"].isin(CFG_MAP) & (df["gesture"] != "OTRO")].reset_index(drop=True)
    rt_graphs, rt_y = [], []
    for s in range(0, len(df) - 40 + 1, 20):
        chunk = df.iloc[s:s + 40]
        label = CFG_MAP.get(chunk["gesture"].value_counts().index[0], -1)
        if label < 0:
            continue
        feat = sd._build_node_features(realtime_features(chunk["emg_raw"].to_numpy(float)))
        adj, wts = sd._build_virtual_graph(feat)
        src, dst = np.where(adj > 0)
        rt_graphs.append(Data(x=torch.tensor(feat), edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
                              edge_attr=torch.tensor(wts[src, dst, None])))
        rt_y.append(label)
    model = load("sich_best.pt")
    report("training features (all windows)", y, predict(model, graphs))
    report("realtime.py features (same windows)", np.array(rt_y), predict(model, rt_graphs))


if __name__ == "__main__":
    main()
