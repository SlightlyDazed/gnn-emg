"""
GNN-EMG — Fase 1: Pre-entrenamiento en NinaPro DB5
════════════════════════════════════════════════════

Uso:
    python -m gnn_emg.train.pretrain_ninapro
    python -m gnn_emg.train.pretrain_ninapro --resume          # continúa desde nina_best.pt
    python -m gnn_emg.train.pretrain_ninapro --resume --epochs 120
"""
from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gnn_emg.data   import NinaProDataset
from gnn_emg.models import EMGGraphAutoencoder
from gnn_emg.utils  import load_config


# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopper:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait      = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait      = 0
        else:
            self.wait += 1
        return self.wait >= self.patience


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        dev  = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"[Device] GPU: {name}")
    else:
        dev = torch.device("cpu")
        print("[Device] CPU")
    return dev


def _run_epoch(model, loader, optimizer, device, domain,
               lambda_recon, lambda_clf, lambda_contrast, is_train,
               class_weights=None):
    model.train(is_train)
    total     = {"loss": 0., "loss_recon": 0., "loss_clf": 0., "accuracy": 0.}
    n_batches = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch  = batch.to(device)
            out    = model(batch, domain=domain)
            losses = model.compute_loss(
                out, batch, domain=domain,
                lambda_recon=lambda_recon,
                lambda_clf=lambda_clf,
                lambda_contrast=lambda_contrast,
                class_weights=class_weights,
            )
            if is_train:
                optimizer.zero_grad()
                if torch.isnan(losses["loss"]) or torch.isinf(losses["loss"]):
                    print(f"  [WARN] NaN — batch saltado "
                          f"r={losses['loss_recon']:.4f} c={losses['loss_clf']:.4f}")
                    continue
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            for k in total:
                total[k] += losses[k].item() if torch.is_tensor(losses[k]) else losses[k]
            n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in total.items()}


def _save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path):
    torch.save({
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "val_loss": val_loss,
    }, path)


def _load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))


# ─────────────────────────────────────────────────────────────────────────────

def pretrain(cfg_path=None, resume=False, extra_epochs=None):
    cfg    = load_config(cfg_path)
    device = _get_device()

    pkg_root = Path(__file__).parent.parent
    nina_root= (pkg_root / cfg.paths.ninapro_root / cfg.paths.ninapro_db).resolve()
    ckpt_dir = (pkg_root / cfg.paths.checkpoints).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "nina_best.pt"
    last_ckpt = ckpt_dir / "nina_last.pt"

    total_epochs = extra_epochs or cfg.pretrain.epochs

    print(f"\n{'═'*60}")
    print(f"  GNN-EMG  |  Fase 1: Pre-entrenamiento NinaPro DB5")
    print(f"{'═'*60}")
    print(f"  DB path  : {nina_root}")
    print(f"  Batch    : {cfg.pretrain.batch_size}  |  LR: {cfg.pretrain.lr}")
    print(f"  Epochs   : {total_epochs}  |  Resume: {resume}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    exercises = list(cfg.ninapro.exercises)
    subjects  = list(cfg.ninapro.subjects) if cfg.ninapro.subjects else None

    dataset = NinaProDataset(
        db_root      = nina_root,
        exercises    = exercises,
        win_ms       = cfg.ninapro.window_ms,
        overlap_pct  = cfg.ninapro.overlap_pct,
        fs           = cfg.ninapro.fs,
        corr_thr     = cfg.graph.corr_threshold,
        use_restim   = cfg.ninapro.use_restimulus,
        exclude_rest = cfg.ninapro.exclude_rest,
        subjects     = subjects,
    )
    if len(dataset) == 0:
        print("[ERROR] Dataset vacío."); return

    n_classes = dataset.n_classes
    train_ds, val_ds = dataset.split(val_pct=cfg.pretrain.val_split, seed=42)

    nw = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(train_ds, batch_size=cfg.pretrain.batch_size,
                              shuffle=True,  num_workers=nw, pin_memory=True,
                              persistent_workers=nw > 0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.pretrain.batch_size,
                              shuffle=False, num_workers=nw, pin_memory=True,
                              persistent_workers=nw > 0)

    print(f"\n  Train: {len(train_ds):,} | Val: {len(val_ds):,} | "
          f"Clases: {n_classes} | Workers: {nw}")

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = EMGGraphAutoencoder(
        nina_n_nodes    = cfg.ninapro.n_channels,
        nina_in_feat    = cfg.model.in_channels_nina,
        sich_n_nodes    = 5,
        sich_in_feat    = cfg.model.in_channels_sich,
        hidden_channels = cfg.model.hidden_channels,
        latent_dim      = cfg.model.latent_dim,
        num_layers      = cfg.model.num_layers,
        heads           = cfg.model.num_heads,
        dropout         = cfg.model.dropout,
        use_bn          = cfg.model.use_batch_norm,
        n_gestures_nina = n_classes,
        n_gestures_sich = cfg.model.n_gestures_sich,
        decoder_hidden  = list(cfg.model.decoder_hidden),
        temporal_encoder= getattr(cfg.model, "temporal_encoder", "conv"),
    ).to(device)

    optimizer = AdamW(model.parameters(),
                      lr=float(cfg.pretrain.lr),
                      weight_decay=float(cfg.pretrain.weight_decay))

    if cfg.pretrain.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    elif cfg.pretrain.scheduler == "cosine_warm":
        T0 = getattr(cfg.pretrain, "scheduler_T0", 80)
        Tmult = getattr(cfg.pretrain, "scheduler_Tmult", 2)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=Tmult, eta_min=1e-6)
    elif cfg.pretrain.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        scheduler = None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val    = float("inf")
    stopper     = EarlyStopper(patience=50)   # ~13min sin mejora → parar (sujeto 1)

    resume_path = best_ckpt if best_ckpt.exists() else (
                  last_ckpt if last_ckpt.exists() else None)

    if resume and resume_path:
        # NO cargar scheduler state — puede ser de otro tipo
        start_epoch, best_val = _load_checkpoint(
            model, optimizer, None, resume_path, device)
        start_epoch   += 1
        stopper.best_loss = best_val
        print(f"\n[Resume] {resume_path.name} → época {start_epoch} | best_val={best_val:.4f}")
    elif resume:
        print("[Resume] Sin checkpoint previo — empezando desde cero")

    print(f"\n{model}")
    print(f"\n{'─'*60}")
    print(f"  Épocas {start_epoch} → {total_epochs}")
    print(f"{'─'*60}\n")

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()

        train_m = _run_epoch(model, train_loader, optimizer, device, "ninapro",
                             cfg.pretrain.loss_recon_weight,
                             cfg.pretrain.loss_clf_weight,
                             float(cfg.pretrain.loss_contrast_wt), True)
        val_m   = _run_epoch(model, val_loader, None, device, "ninapro",
                             cfg.pretrain.loss_recon_weight,
                             cfg.pretrain.loss_clf_weight,
                             float(cfg.pretrain.loss_contrast_wt), False)

        if scheduler:
            scheduler.step()

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["val_acc"].append(val_m["accuracy"])

        flag = ""
        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            _save_checkpoint(model, optimizer, scheduler, epoch, best_val, best_ckpt)
            flag = " ★ BEST"

        dt = time.time() - t0
        print(f"Epoch {epoch:3d}/{total_epochs} "
              f"| train={train_m['loss']:.4f} "
              f"(r={train_m['loss_recon']:.4f} c={train_m['loss_clf']:.4f}) "
              f"| val={val_m['loss']:.4f} acc={val_m['accuracy']:.3f} "
              f"| {dt:.1f}s{flag}")

        if stopper.step(val_m["loss"]):
            print(f"\n[EarlyStopping] {stopper.patience} épocas sin mejora.")
            break

    _save_checkpoint(model, optimizer, scheduler, epoch, val_m["loss"], last_ckpt)
    np.save(ckpt_dir / "nina_history.npy", history)

    print(f"\n{'═'*60}")
    print(f"  Listo | mejor val_loss={best_val:.4f} | {best_ckpt}")
    print(f"{'═'*60}\n")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-EMG Fase 1: NinaPro")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Continuar desde checkpoint guardado")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Total de épocas (override del config)")
    args = parser.parse_args()
    pretrain(cfg_path=args.config, resume=args.resume, extra_epochs=args.epochs)
