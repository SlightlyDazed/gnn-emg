"""
GNN-EMG — GATv2 Graph Autoencoder con cabeza de clasificación de gesto
═══════════════════════════════════════════════════════════════════════

ARQUITECTURA:
─────────────────────────────────────────────────────────────────────
                       ┌─── Entrada ───┐
               (N, F_in) tensor de features de nodo
                            │
                     ┌──────▼──────┐
                     │  InputProj  │  ← proyecta F_in → hidden_dim
                     │   Linear    │    (maneja domain gap nina/sich)
                     └──────┬──────┘
                            │
                   ┌────────▼────────┐
                   │  GATv2Encoder   │
                   │  3× GATv2Conv   │  ← atención por arista
                   │  + BatchNorm    │    aprende qué músculo importa
                   │  + Dropout      │    para cada gesto
                   └────────┬────────┘
                            │ (N, hidden)
                   ┌────────▼────────┐
                   │ global_mean_pool│  ← colapsa nodos → embedding gráfico
                   └────────┬────────┘
                            │ (B, hidden)
              ┌─────────────┴─────────────┐
              │                           │
      ┌───────▼───────┐         ┌─────────▼──────────┐
      │  LatentProj   │         │                    │
      │  hidden→latent│         │   GestureHead      │
      │  (z embedding)│         │   latent→n_classes │
      └───────┬───────┘         │   (softmax)        │
              │                 └─────────┬──────────┘
              │                           │
      ┌───────▼───────┐         gesture logits (→ CrossEntropy)
      │    Decoder    │
      │  latent→hidden│
      │  →n_nodes×F_in│  ← reconstruye features de nodo
      └───────┬───────┘
              │
       x_hat (reconstruida)
       L_recon = MSE(x_hat, x)
─────────────────────────────────────────────────────────────────────

PÉRDIDAS:
  L_total = λ_r × L_recon + λ_c × L_clf [+ λ_k × L_contrast]

  L_recon:   MSE entre x_hat y x normalizado (reconstrucción autoencoderiana)
  L_clf:     CrossEntropy en gesture logits
  L_contrast: NT-Xent contrastivo sobre z (opcional, mejora separabilidad)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    global_mean_pool,
    global_max_pool,
    BatchNorm,
)
from torch_geometric.data import Data, Batch


# ─────────────────────────────────────────────────────────────────────────────
#  Bloques auxiliares
# ─────────────────────────────────────────────────────────────────────────────

class TemporalConvProj(nn.Module):
    """
    1D-CNN que extrae features temporales aprendidas de EMG crudo por nodo.

    Flujo:  (N, 40+1) → split raw/chan_id → Conv1d stack → pool → concat → proj
    
    Aprende automáticamente qué patrones temporales importan para cada gesto
    (análogo aprendido de RMS, ZC, WL, pero end-to-end optimizado).
    """
    def __init__(self, n_samples: int = 40, hidden_channels: int = 128):
        super().__init__()
        self.n_samples = n_samples
        self.conv = nn.Sequential(
            # (N, 1, 40) → (N, 32, 40)
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            # (N, 32, 40) → (N, 64, 40)
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            # (N, 64, 40) → (N, 128, 40)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            # Pool temporal → (N, 128, 1)
            nn.AdaptiveAvgPool1d(1),
        )
        # 128 from CNN + 1 chan_id → hidden_channels
        self.proj = nn.Sequential(
            nn.Linear(128 + 1, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N_total, n_samples + 1)  — 40 muestras raw + 1 canal ID
        returns : (N_total, hidden_channels)
        """
        raw    = x[:, :self.n_samples]    # (N, 40)
        chan_id = x[:, -1:]               # (N, 1)

        # Rectificación fisiológica: evitar que el promedio temporal converja a cero
        raw_rectified = torch.abs(raw)
        raw_in = raw_rectified.unsqueeze(1)  # (N, 1, 40) — 1 "canal" de Conv1d
        feat = self.conv(raw_in)             # (N, 128, 1)
        feat = feat.squeeze(-1)              # (N, 128)

        h = torch.cat([feat, chan_id], dim=-1)  # (N, 129)
        h = self.proj(h)                  # (N, hidden_channels)
        return h


class TemporalLSTMProj(nn.Module):
    """
    BiLSTM que extrae features temporales de EMG crudo por nodo.

    Flujo: (N, 40+1) → split → BiLSTM(40 timesteps) → last hidden → proj
    
    Captura dependencias temporales de largo alcance en la señal EMG
    (útil para gestos con activación progresiva o sostenida).
    """
    def __init__(self, n_samples: int = 40, hidden_channels: int = 128,
                 lstm_hidden: int = 64, lstm_layers: int = 2):
        super().__init__()
        self.n_samples = n_samples
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            # dropout=0.1  ← deshabilitado: ROCm/MIOpen no tiene rocrand_xorwow en gfx1201
        )
        # BiLSTM output: 2 * lstm_hidden + 1 chan_id → hidden_channels
        self.proj = nn.Sequential(
            nn.Linear(2 * lstm_hidden + 1, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N_total, n_samples + 1)
        returns : (N_total, hidden_channels)
        """
        raw    = x[:, :self.n_samples]    # (N, 40)
        chan_id = x[:, -1:]               # (N, 1)

        raw = raw.unsqueeze(-1)           # (N, 40, 1) — secuencia temporal
        _, (h_n, _) = self.lstm(raw)      # h_n: (num_layers*2, N, lstm_hidden)

        # Tomar las últimas hidden states de ambas direcciones
        h_fwd = h_n[-2]                   # (N, lstm_hidden)
        h_bwd = h_n[-1]                   # (N, lstm_hidden)
        feat  = torch.cat([h_fwd, h_bwd], dim=-1)  # (N, 2*lstm_hidden)

        h = torch.cat([feat, chan_id], dim=-1)
        h = self.proj(h)                  # (N, hidden_channels)
        return h

class ResidualGATv2Block(nn.Module):
    """
    Un bloque GATv2 con conexión residual (si dim_in == dim_out).
    BatchNorm → GATv2Conv → ELU → Dropout → (+ skip si dims coinciden)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int       = 4,
        dropout: float   = 0.2,
        edge_dim: int    = 1,
        use_bn: bool     = True,
    ):
        super().__init__()
        self.heads   = heads
        self.out_dim = out_channels

        # GATv2Conv produce heads × out_channels → luego proyectamos a out_channels
        self.conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=True,             # concatenar cabezas
            share_weights=False,
        )
        # Proyección de cabezas concatenadas → out_channels
        self.proj = nn.Linear(heads * out_channels, out_channels, bias=False)
        self.bn   = BatchNorm(out_channels) if use_bn else nn.Identity()
        self.drop = nn.Dropout(dropout)

        # Skip connection si dims no coinciden
        self.skip = (
            nn.Linear(in_channels, out_channels, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv(x, edge_index, edge_attr=edge_attr)   # (N, heads*out)
        out = self.proj(out)                                    # (N, out)
        out = self.bn(out)
        out = F.elu(out)
        out = self.drop(out)
        return out + residual


# ─────────────────────────────────────────────────────────────────────────────
#  Encoder
# ─────────────────────────────────────────────────────────────────────────────

class GATv2Encoder(nn.Module):
    """
    Encoder GATv2 multi-capa.
    Mapea (N, F_in) → z ∈ (B, latent_dim) via pooling global.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        latent_dim: int      = 32,
        num_layers: int      = 3,
        heads: int           = 4,
        dropout: float       = 0.2,
        use_bn: bool         = True,
    ):
        super().__init__()
        # Proyección inicial al espacio oculto
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ELU(),
        )

        # Capas GATv2
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ResidualGATv2Block(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                heads=heads,
                dropout=dropout,
                edge_dim=1,
                use_bn=use_bn,
            ))

        # Pooling combinado: mean + max → 2×hidden
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ELU(),
        )

        # Proyección al espacio latente
        self.to_latent = nn.Linear(hidden_channels, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns
        -------
        z : (B, latent_dim)   embedding de gesto por grafo
        node_h : (N, hidden)  representaciones de nodo (para decoder)
        """
        h = self.input_proj(x)                        # (N, hidden)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)       # (N, hidden)

        node_h = h                                    # guardar para decoder

        # Pooling global diferenciable (mean + max)
        g_mean = global_mean_pool(h, batch)           # (B, hidden)
        g_max  = global_max_pool(h, batch)            # (B, hidden)
        g      = torch.cat([g_mean, g_max], dim=-1)   # (B, 2·hidden)
        g      = self.pool_proj(g)                    # (B, hidden)

        z = self.to_latent(g)                         # (B, latent_dim)
        return z, node_h


# ─────────────────────────────────────────────────────────────────────────────
#  Decoder — reconstrucción de features de nodo
# ─────────────────────────────────────────────────────────────────────────────

class SignalDecoder(nn.Module):
    """
    Decoder MLP a nivel de nodo: node_h (N_total, hidden_channels) -> x_hat (N_total, out_features)

    Expande la representación local de cada nodo de vuelta a sus features de entrada.
    Esto permite reconstruir la señal localmente sin sobrecargar el bottleneck global 'z'.
    """
    def __init__(
        self,
        hidden_dim: int,
        out_features: int,
        hidden_sizes: list[int] = [64, 128],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.out_features = out_features
        layers = []
        prev = hidden_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.mlp = nn.Sequential(*layers)

    def forward(self, node_h: torch.Tensor) -> torch.Tensor:
        """
        node_h : (N_total, hidden_dim)

        Returns
        -------
        x_hat : (N_total, out_features)
            Reconstrucción directa de las features originales de entrada.
        """
        return self.mlp(node_h)


# ─────────────────────────────────────────────────────────────────────────────
#  Gesture Head — clasificación de gesto desde z
# ─────────────────────────────────────────────────────────────────────────────

class GestureHead(nn.Module):
    """
    Cabeza de clasificación: z → gesture logits.
    Capa simple pero con temperatura aprendible para calibración.
    """
    def __init__(self, latent_dim: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, n_classes),
        )
        # Temperatura aprendible (calibración de confianza)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.net(z)
        return logits / self.temperature.clamp(min=0.1)


# ─────────────────────────────────────────────────────────────────────────────
#  Modelo completo
# ─────────────────────────────────────────────────────────────────────────────

class EMGGraphAutoencoder(nn.Module):
    """
    GATv2 Graph Autoencoder para señales EMG y reconocimiento de gestos.

    Soporta dos modos de dominio vía `domain`:
      - 'ninapro'  : nodos = 16 canales EMG reales    , F_in = 8
      - 'sichiray' : nodos = 5 nodos virtuales        , F_in = 7

    El encoder es COMPARTIDO entre dominios (transfer learning).
    Cada dominio tiene su propia capa de proyección de entrada (InputProj)
    para manejar el gap de features.
    """

    def __init__(
        self,
        # Geometría del grafo por dominio
        nina_n_nodes: int    = 16,
        nina_in_feat: int    = 8,
        sich_n_nodes: int    = 5,
        sich_in_feat: int    = 7,
        # Arquitectura compartida
        hidden_channels: int = 64,
        latent_dim: int      = 32,
        num_layers: int      = 3,
        heads: int           = 4,
        dropout: float       = 0.2,
        use_bn: bool         = True,
        # Cabezas de clasificación
        n_gestures_nina: int = 53,
        n_gestures_sich: int = 11,
        # Decoder
        decoder_hidden: list[int] = [64, 128],
        # Extractor temporal: 'conv' o 'lstm'
        temporal_encoder: str = "conv",
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.latent_dim      = latent_dim
        self.nina_n_nodes    = nina_n_nodes
        self.nina_in_feat    = nina_in_feat
        self.sich_n_nodes    = sich_n_nodes
        self.sich_in_feat    = sich_in_feat
        self.temporal_type   = temporal_encoder

        # ── Encoder compartido ────────────────────────────────────────────────
        self.encoder = GATv2Encoder(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            use_bn=use_bn,
        )

        # ── Proyecciones de entrada por dominio (domain adapters) ─────────────
        n_raw = nina_in_feat - 1  # quitar chan_id
        if temporal_encoder == "lstm":
            self.nina_input_proj = TemporalLSTMProj(
                n_samples=n_raw, hidden_channels=hidden_channels,
            )
        else:  # "conv" por defecto
            self.nina_input_proj = TemporalConvProj(
                n_samples=n_raw, hidden_channels=hidden_channels,
            )
        # Sichiray: Linear para features pre-computadas
        self.sich_input_proj = nn.Sequential(
            nn.Linear(sich_in_feat, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
        )

        # ── Decoders por dominio ──────────────────────────────────────────────
        self.nina_decoder = SignalDecoder(
            hidden_dim=hidden_channels,
            out_features=nina_in_feat,  # Reconstruye la entrada original para el autoencoder
            hidden_sizes=decoder_hidden,
            dropout=dropout,
        )
        self.sich_decoder = SignalDecoder(
            hidden_dim=hidden_channels,
            out_features=sich_in_feat,
            hidden_sizes=decoder_hidden,
            dropout=dropout,
        )

        # ── Cabezas de clasificación por dominio ──────────────────────────────
        self.nina_gesture_head = GestureHead(latent_dim, n_gestures_nina, dropout)
        self.sich_gesture_head = GestureHead(latent_dim, n_gestures_sich, dropout)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        batch: Batch,
        domain: str = "ninapro",   # 'ninapro' | 'sichiray'
    ) -> dict:
        """
        Forward pass completo.

        Returns dict con:
          z       : (B, latent_dim)    — embedding de gesto
          logits  : (B, n_classes)     — logits de clasificación
          x_hat   : (B·N, F_in)        — señal reconstruida
          node_h  : (N_total, hidden)  — representaciones intermedias de nodo
        """
        x          = batch.x            # (N_total, F_in)
        edge_index = batch.edge_index   # (2, E)
        edge_attr  = batch.edge_attr    # (E, 1)
        b          = batch.batch        # (N_total,)

        # ── Proyección de dominio ─────────────────────────────────────────────
        if domain == "ninapro":
            h = self.nina_input_proj(x)          # (N, hidden)
        else:
            h = self.sich_input_proj(x)          # (N, hidden)

        # ── Encoder GATv2 ─────────────────────────────────────────────────────
        z, node_h = self.encoder(h, edge_index, b, edge_attr)  # z: (B, latent)

        # ── Decoder a nivel de nodo ───────────────────────────────────────────
        if domain == "ninapro":
            x_hat  = self.nina_decoder(node_h)   # (N_total_nina, F_nina)
            logits = self.nina_gesture_head(z)   # (B, n_gestures_nina)
        else:
            x_hat  = self.sich_decoder(node_h)   # (N_total_sich, F_sich)
            logits = self.sich_gesture_head(z)   # (B, n_gestures_sich)

        return {
            "z":      z,
            "logits": logits,
            "x_hat":  x_hat,
            "node_h": node_h,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Pérdidas
    # ─────────────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        out: dict,
        batch: Batch,
        domain: str,
        lambda_recon: float   = 1.0,
        lambda_clf: float     = 0.5,
        lambda_contrast: float= 0.0,
        class_weights: torch.Tensor | None = None,
    ) -> dict:
        """
        Calcula todas las pérdidas y retorna un dict con:
          loss        : pérdida total (escalar)
          loss_recon  : pérdida de reconstrucción (MSE)
          loss_clf    : pérdida de clasificación (CrossEntropy)
          loss_contrast: pérdida contrastiva NT-Xent (opcional)
          accuracy    : accuracy de clasificación en el batch
        """
        x      = batch.x        # (N_total, F_in)
        y      = batch.y.view(-1)    # (B,)
        z      = out["z"]        # (B, latent)
        logits = out["logits"]   # (B, n_classes)
        x_hat  = out["x_hat"]   # (B·N, F)

        # ── Reconstrucción MSE ────────────────────────────────────────────────
        if domain == "ninapro":
            # Target es la envolvente (Mean Absolute Value) por canal: media absoluta de las 40 muestras
            target = torch.abs(x[:, :40]).mean(dim=-1, keepdim=True)
        else:
            # Sichiray: ya son features estadísticas pre-calculadas (7 features)
            target = x

        loss_recon = F.mse_loss(x_hat, target)

        # ── Clasificación CrossEntropy ────────────────────────────────────────
        loss_clf = F.cross_entropy(logits, y, weight=class_weights)

        # ── Contrastivo NT-Xent (opcional) ───────────────────────────────────
        loss_contrast = torch.tensor(0.0, device=z.device)
        if lambda_contrast > 0:
            loss_contrast = self._nt_xent_loss(z, y)

        loss = (lambda_recon  * loss_recon
              + lambda_clf    * loss_clf
              + lambda_contrast * loss_contrast)

        # Accuracy
        preds = logits.argmax(dim=-1)
        acc   = (preds == y).float().mean()

        return {
            "loss":          loss,
            "loss_recon":    loss_recon,
            "loss_clf":      loss_clf,
            "loss_contrast": loss_contrast,
            "accuracy":      acc,
        }

    @staticmethod
    def _nt_xent_loss(
        z: torch.Tensor,        # (B, D)
        labels: torch.Tensor,   # (B,)
        temperature: float = 0.5,   # subido de 0.1 → más estable
    ) -> torch.Tensor:
        """
        NT-Xent contrastivo supervisado — versión numéricamente estable.
        Atrae embeddings de la misma clase, repele los de otras.
        """
        B = z.size(0)
        if B < 2:
            return torch.tensor(0.0, device=z.device)

        # Normalizar
        z_norm = F.normalize(z, dim=1)                             # (B, B)
        # Clamp temperatura para evitar overflow
        temp   = max(temperature, 0.07)
        sim    = (z_norm @ z_norm.T) / temp                        # (B, B)

        # Máscara positiva: misma clase, excluir diagonal
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)     # (B, B)
        self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
        pos_mask  = labels_eq & ~self_mask                         # (B, B)

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)

        # Máscara negativa (todo excepto la diagonal)
        neg_mask  = ~self_mask

        # Log-sum-exp estable sobre negativos (por fila)
        # Usamos logsumexp en lugar de log(softmax) para evitar -inf
        sim_masked = sim.clone()
        sim_masked = sim_masked.masked_fill(self_mask, float("-inf"))

        # Para cada par positivo, pérdida = -sim_pos + logsumexp(sim_todos_sin_diag)
        logsumexp_row = torch.logsumexp(sim_masked, dim=1)         # (B,)

        # Pérdida por positivo
        losses = []
        for i in range(B):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_indices) == 0:
                continue
            for j in pos_indices:
                losses.append(-sim[i, j] + logsumexp_row[i])

        if not losses:
            return torch.tensor(0.0, device=z.device)

        loss = torch.stack(losses).mean()
        # Clamp para evitar NaN residual
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=z.device)
        return loss

    # ─────────────────────────────────────────────────────────────────────────
    #  Inferencia: predicción de gesto desde embedding z
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_gesture(
        self,
        batch: Batch,
        domain: str = "sichiray",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicción de gesto en modo inferencia.

        Returns
        -------
        probs   : (B, n_classes)  probabilidades softmax
        classes : (B,)            índice de clase predicha
        """
        self.eval()
        out    = self.forward(batch, domain=domain)
        probs  = F.softmax(out["logits"], dim=-1)
        classes= probs.argmax(dim=-1)
        return probs, classes

    # ─────────────────────────────────────────────────────────────────────────
    #  Utilidades
    # ─────────────────────────────────────────────────────────────────────────

    def freeze_encoder(self):
        """Congela los pesos del encoder (útil al inicio del fine-tuning)."""
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self):
        """Descongela el encoder para fine-tuning completo."""
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"EMGGraphAutoencoder(\n"
            f"  encoder     = GATv2 ({len(self.encoder.layers)} layers, "
            f"hidden={self.hidden_channels}, latent={self.latent_dim})\n"
            f"  domains     = ninapro ({self.nina_n_nodes}N×{self.nina_in_feat}F) | "
            f"sichiray ({self.sich_n_nodes}N×{self.sich_in_feat}F)\n"
            f"  parameters  = {self.count_parameters():,}\n"
            f")"
        )
