"""
GNN-EMG — Fase 2: Fine-tuning en datos Sichiray (tus capturas personales)
══════════════════════════════════════════════════════════════════════════

Uso:
    python -m gnn_emg.train.finetune_sichiray [--config configs/config.yaml]
                                              [--ckpt checkpoints/nina_best.pt]

Flujo:
  1. Carga el checkpoint pre-entrenado en NinaPro
  2. Carga tus CSVs de sichiray → grafos con nodos virtuales (5 nodos)
  3. Fase A (primeras freeze_encoder_epochs épocas): SOLO entrena
     los adaptadores sichiray (input_proj + decoder + gesture_head)
     manteniendo el encoder GATv2 congelado.
  4. Fase B (resto): entrena TODO con LR menor.
  5. Guarda checkpoint en checkpoints/sich_best.pt
"""
from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gnn_emg.data     import SichirayDataset, GESTURE_MAP
from gnn_emg.models   import EMGGraphAutoencoder
from gnn_emg.utils    import load_config
from gnn_emg.train.pretrain_ninapro import (
    EarlyStopper, _run_epoch, _save_checkpoint, _get_device
)


# ─────────────────────────────────────────────────────────────────────────────

def finetune(cfg_path: str | None = None, ckpt_path: str | None = None):
    cfg    = load_config(cfg_path)
    device = _get_device()

    pkg_root = Path(__file__).parent.parent
    sich_root= (pkg_root / cfg.paths.sichiray_root).resolve()
    ckpt_dir = (pkg_root / cfg.paths.checkpoints).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if ckpt_path is None:
        ckpt_path = ckpt_dir / "nina_best.pt"

    best_ckpt = ckpt_dir / "sich_best.pt"

    print(f"\n{'═'*60}")
    print(f"  GNN-EMG  |  Fase 2: Fine-tuning Sichiray")
    print(f"{'═'*60}")
    print(f"  CSV root : {sich_root}")
    print(f"  Pretrain : {ckpt_path}")

    # ── Dataset Sichiray ──────────────────────────────────────────────────────
    dataset = SichirayDataset(
        csv_root    = sich_root,
        win_ms      = cfg.ninapro.window_ms,
        overlap_pct = 0.5,
        fs          = cfg.sichiray.fs,
        corr_thr    = cfg.graph.corr_threshold,
        gesture_map = dict(cfg.sichiray.gesture_map.__dict__),
    )

    if len(dataset) == 0:
        print("[ERROR] Sin datos Sichiray. Verifica la ruta.")
        return

    train_ds, val_ds = dataset.split(val_pct=cfg.finetune.val_split, seed=42)

    train_loader = DataLoader(train_ds, batch_size=cfg.finetune.batch_size,
                              shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.finetune.batch_size,
                              shuffle=False, num_workers=2)

    class_weights = dataset.class_weights.to(device)
    print(f"\n  Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # ── Modelo — cargar pesos pre-entrenados ──────────────────────────────────
    # Necesitamos saber cuántas clases usó NinaPro para instanciar el modelo
    # Esto está guardado en el checkpoint
    ckpt = torch.load(ckpt_path, map_location=device) if Path(ckpt_path).exists() else None

    # Para poder cargar el state_dict, el modelo debe tener las MISMAS clases NinaPro
    # Leemos n_gestures_nina del checkpoint
    n_gestures_nina = cfg.model.n_gestures_nina
    if ckpt:
        # Inferir desde el shape del weight de la gesture head
        try:
            n_gestures_nina = ckpt["model"][
                "nina_gesture_head.net.3.weight"
            ].shape[0]
        except KeyError:
            pass

    model = EMGGraphAutoencoder(
        nina_n_nodes     = cfg.ninapro.n_channels,
        nina_in_feat     = cfg.model.in_channels_nina,
        sich_n_nodes     = 5,
        sich_in_feat     = cfg.model.in_channels_sich,
        hidden_channels  = cfg.model.hidden_channels,
        latent_dim       = cfg.model.latent_dim,
        num_layers       = cfg.model.num_layers,
        heads            = cfg.model.num_heads,
        dropout          = cfg.model.dropout,
        use_bn           = cfg.model.use_batch_norm,
        n_gestures_nina  = n_gestures_nina,
        n_gestures_sich  = len(GESTURE_MAP),
        decoder_hidden   = list(cfg.model.decoder_hidden),
    ).to(device)

    if ckpt:
        # Cargar con strict=False para ignorar capas sichiray (no pre-entrenadas)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"[Checkpoint] Pesos cargados — missing: {len(missing)} | "
              f"unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing: {missing[:5]}{'...' if len(missing)>5 else ''}")
    else:
        print("[WARNING] Checkpoint NinaPro no encontrado. Entrenando desde cero.")

    print(f"\n{model}")

    # ── FASE A: encoder congelado ─────────────────────────────────────────────
    freeze_epochs = cfg.finetune.freeze_encoder_epochs

    print(f"\n{'─'*60}")
    print(f"  Fase A: encoder CONGELADO ({freeze_epochs} épocas)")
    print(f"{'─'*60}")

    model.freeze_encoder()

    # Solo parámetros descongelados
    params_a = [p for p in model.parameters() if p.requires_grad]
    optim_a  = AdamW(params_a, lr=float(cfg.finetune.lr), weight_decay=float(cfg.finetune.weight_decay))
    sched_a  = CosineAnnealingLR(optim_a, T_max=freeze_epochs, eta_min=1e-6)

    best_val = float("inf")
    history  = {"train_loss": [], "val_loss": [], "val_acc": [], "phase": []}

    for epoch in range(1, freeze_epochs + 1):
        t0 = time.time()
        train_m = _run_epoch(
            model, train_loader, optim_a, device, "sichiray",
            cfg.finetune.loss_recon_weight, cfg.finetune.loss_clf_weight, 0.0, True,
        )
        val_m = _run_epoch(
            model, val_loader, None, device, "sichiray",
            cfg.finetune.loss_recon_weight, cfg.finetune.loss_clf_weight, 0.0, False,
        )
        sched_a.step()

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["val_acc"].append(val_m["accuracy"])
        history["phase"].append("A")

        flag = ""
        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            _save_checkpoint(model, optim_a, sched_a, epoch, best_val, best_ckpt)
            flag = " ★"

        print(f"[A] Epoch {epoch:3d}/{freeze_epochs} "
              f"| L={train_m['loss']:.4f} | valL={val_m['loss']:.4f} "
              f"| acc={val_m['accuracy']:.3f} | {time.time()-t0:.1f}s{flag}")

    # ── FASE B: encoder descongelado, LR menor ────────────────────────────────
    model.unfreeze_encoder()
    total_b = cfg.finetune.epochs - freeze_epochs
    stopper = EarlyStopper(patience=10)

    optim_b = AdamW(
        model.parameters(),
        lr           = float(cfg.finetune.lr) * 0.5,
        weight_decay = float(cfg.finetune.weight_decay),
    )
    sched_b = CosineAnnealingLR(optim_b, T_max=max(total_b, 1), eta_min=1e-7)

    print(f"\n{'─'*60}")
    print(f"  Fase B: entrenamiento COMPLETO ({total_b} épocas)")
    print(f"{'─'*60}")

    for epoch in range(1, total_b + 1):
        t0 = time.time()
        train_m = _run_epoch(
            model, train_loader, optim_b, device, "sichiray",
            cfg.finetune.loss_recon_weight, cfg.finetune.loss_clf_weight, 0.0, True,
        )
        val_m = _run_epoch(
            model, val_loader, None, device, "sichiray",
            cfg.finetune.loss_recon_weight, cfg.finetune.loss_clf_weight, 0.0, False,
        )
        sched_b.step()

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["val_acc"].append(val_m["accuracy"])
        history["phase"].append("B")

        flag = ""
        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            _save_checkpoint(model, optim_b, sched_b, epoch + freeze_epochs,
                             best_val, best_ckpt)
            flag = " ★"

        print(f"[B] Epoch {epoch:3d}/{total_b} "
              f"| L={train_m['loss']:.4f} | valL={val_m['loss']:.4f} "
              f"| acc={val_m['accuracy']:.3f} | {time.time()-t0:.1f}s{flag}")

        if stopper.step(val_m["loss"]):
            print(f"\n[EarlyStopping]")
            break

    np.save(ckpt_dir / "sich_history.npy", history)

    print(f"\n{'═'*60}")
    print(f"  Fine-tuning completo | Mejor val_loss: {best_val:.4f}")
    print(f"  Checkpoint: {best_ckpt}")
    print(f"{'═'*60}\n")

    return model, history


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-EMG Fase 2: fine-tuning sichiray")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ckpt",   type=str, default=None,
                        help="Ruta al checkpoint NinaPro (default: checkpoints/nina_best.pt)")
    args = parser.parse_args()
    finetune(cfg_path=args.config, ckpt_path=args.ckpt)
