"""
compare_temporal.py — Comparación Conv1d vs BiLSTM en NinaPro

Corre ambos modelos N épocas con los mismos datos/seed y reporta
cuál aprende más rápido la relación EMG→gesto.

Uso:
    python -m gnn_emg.train.compare_temporal --epochs 20
"""
import sys, time, argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn_emg.utils import load_config
from gnn_emg.data import NinaProDataset
from gnn_emg.models import EMGGraphAutoencoder


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config",  default="gnn_emg/configs/config.yaml")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--variant", choices=["conv", "lstm", "both"], default="both",
                    help="Qué variante correr: conv, lstm, o both (default)")
args = parser.parse_args()

# ── Configuración ─────────────────────────────────────────────────────────────
cfg     = load_config(args.config)
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCHS = args.epochs

# Resolver paths igual que pretrain_ninapro.py (relativo al paquete)
pkg_root  = Path(__file__).parent.parent
nina_root = (pkg_root / cfg.paths.ninapro_root / cfg.paths.ninapro_db).resolve()

print(f"\n{'═'*60}")
print(f"  COMPARACIÓN: Conv1d vs BiLSTM  |  {N_EPOCHS} épocas")
print(f"  Dispositivo: {device}")
print(f"  DB path    : {nina_root}")
print(f"{'═'*60}\n")

# ── Dataset (compartido) ──────────────────────────────────────────────────────
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
    print("[ERROR] Dataset vacío — revisa la ruta en config.yaml"); raise SystemExit(1)

n_classes = dataset.n_classes
train_ds, val_ds = dataset.split(val_pct=cfg.pretrain.val_split, seed=42)

nw = min(8, __import__("os").cpu_count() or 4)
train_loader = DataLoader(train_ds, batch_size=cfg.pretrain.batch_size,
                          shuffle=True,  num_workers=nw, pin_memory=True,
                          persistent_workers=nw > 0)
val_loader   = DataLoader(val_ds,   batch_size=cfg.pretrain.batch_size,
                          shuffle=False, num_workers=nw, pin_memory=True,
                          persistent_workers=nw > 0)

print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Clases: {n_classes}\n")


# ── Función de entrenamiento ──────────────────────────────────────────────────
def run_one_epoch(model, loader, optimizer, device, is_train):
    model.train(is_train)
    total_loss, total_acc, n = 0., 0., 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            out   = model(batch, domain="ninapro")
            loss_dict = model.compute_loss(
                out, batch, "ninapro",
                lambda_recon = cfg.pretrain.loss_recon_weight,
                lambda_clf   = cfg.pretrain.loss_clf_weight,
            )
            if is_train:
                optimizer.zero_grad()
                if not (torch.isnan(loss_dict["loss"]) or torch.isinf(loss_dict["loss"])):
                    loss_dict["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            total_loss += loss_dict["loss"].item()
            total_acc  += loss_dict["accuracy"].item()
            n += 1
    return total_loss / max(n,1), total_acc / max(n,1)


def build_model(temporal_encoder: str):
    m = EMGGraphAutoencoder(
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
        temporal_encoder= temporal_encoder,
    ).to(device)
    return m


# ── Entrenar ambos modelos ────────────────────────────────────────────────────
results = {}

variants = ["conv", "lstm"] if args.variant == "both" else [args.variant]
for variant in variants:
    name = "Conv1d (v2a)" if variant == "conv" else "BiLSTM  (v2b)"
    print(f"{'─'*60}")
    print(f"  Modelo: {name}  ({N_EPOCHS} épocas)")
    print(f"{'─'*60}")

    model = build_model(variant)
    opt   = AdamW(model.parameters(), lr=float(cfg.pretrain.lr),
                  weight_decay=float(cfg.pretrain.weight_decay))
    sched = CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-6)

    print(f"  Parámetros: {model.count_parameters():,}")

    history = []
    best_acc = 0.

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_one_epoch(model, train_loader, opt, device, True)
        va_loss, va_acc = run_one_epoch(model, val_loader,   None, device, False)
        sched.step()
        dt = time.time() - t0
        best_acc = max(best_acc, va_acc)
        star = "★" if va_acc >= best_acc else ""
        print(f"  Época {epoch:3d}/{N_EPOCHS} | "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.3f} | "
              f"{dt:.1f}s {star}")
        history.append({"epoch": epoch, "val_acc": va_acc, "val_loss": va_loss})

    results[variant] = {"name": name, "best_acc": best_acc, "history": history}
    print()

# ── Resumen final ─────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print(f"  RESULTADO FINAL  ({N_EPOCHS} épocas)")
print(f"{'═'*60}")
for v, r in results.items():
    print(f"  {r['name']} → best val_acc = {r['best_acc']:.4f}  ({r['best_acc']*100:.1f}%)")

winner = max(results, key=lambda v: results[v]["best_acc"])
print(f"\n  🏆  GANADOR: {results[winner]['name']}")
print(f"\n  → Para usar en entrenamiento completo, cambia en config.yaml:")
print(f"       temporal_encoder: \"{winner}\"")
print(f"{'═'*60}\n")
