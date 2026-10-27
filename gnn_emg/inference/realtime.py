"""
GNN-EMG — Inferencia en tiempo real con GUI
────────────────────────────────────────────────────────────────────────────
Captura señal EMG serial → construye grafo → predice gesto en vivo →
anima una MANO VIRTUAL (21 landmarks estilo MediaPipe) según el gesto
predicho. Ya no depende de cámara: los mismos puntos que antes se dibujaban
sobre el video ahora se mueven exactamente según la señal EMG capturada.

Uso:
    python -m gnn_emg.inference.realtime [--port /dev/ttyUSB0] [--ckpt checkpoints/sich_best.pt]
    python -m gnn_emg.inference.realtime --demo   # señal sintética sin hardware
"""
from __future__ import annotations

import sys, os, time, math, argparse, collections, threading
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, iirnotch

from .hand_pose import HAND_CONNECTIONS, GESTURE_COLORS, HandAnimator

# ─── Import del modelo GNN-EMG (opcional) ──────────────────────────────────
# En la máquina de entrenamiento esto siempre existe. Si por algún motivo no
# está disponible (paquete no instalado, torch ausente, etc.) caemos a un
# predictor "dummy" reactivo a la señal, para que la GUI se pueda igual
# levantar y probar (por ejemplo en otra máquina, o para depurar la interfaz
# sin GPU). Apenas el paquete esté disponible, se usa el modelo real sin
# tocar nada más del script.
try:
    import torch
    from torch_geometric.data import Data, Batch

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from gnn_emg.models   import EMGGraphAutoencoder
    from gnn_emg.utils    import load_config
    from gnn_emg.data     import GESTURE_MAP, N_VNODES, MAX_FEAT_PER_NODE
    from gnn_emg.data.sichiray_dataset import VIRTUAL_NODES, _build_node_features, _build_virtual_graph
    from gnn_emg.utils.signal_utils import bandpass_filter, notch_filter, extract_features_window
    _HAS_MODEL = True
except ImportError as e:
    _HAS_MODEL = False
    _IMPORT_ERR = e
    # Mapa de respaldo — mismos nombres de gesto usados en todo el proyecto
    GESTURE_MAP = {g: i for i, g in enumerate(
        ['ABIERTA', 'PUNIO', 'INDICE', 'PAZ', 'TRES', 'OK',
         'PULGAR_UP', 'PULGAR_DN', 'PINZA', 'TELEFONO', 'OTRO'])}

INV_GESTURE = {v: k for k, v in GESTURE_MAP.items()}

FS          = 200
WIN_SAMPLES = 40    # 200ms @ 200Hz
STRIDE      = 20    # 100ms stride
STRIDE_S    = STRIDE / FS

F_LOW, F_HIGH, F_NOTCH, FILT_ORDER = 20.0, 95.0, 60.0, 4
K_THRESHOLD, K_THRESHOLD_LO = 1.5, 1.0

# Paleta (consistente con monitor_emg_gesture.py)
BG_DARK, BG_SURF, BG_SURF2 = '#171614', '#1c1b19', '#22211f'
TEXT_C, DIVIDER = '#cdccca', '#393836'
C_RAW, C_ENV, C_THRESH = '#4f98a3', '#fdc451', '#e87050'
G_GRAY = '#7a7876'

# ─────────────────────────────────────────────────────────────────────────
#  Buffer de señal — ventana corta (inferencia) + historial largo (display)
# ─────────────────────────────────────────────────────────────────────────
class SignalHub:
    """Buffer circular thread-safe: guarda la señal cruda tanto para la
    ventana de inferencia (corta) como para el trazo visual (larga)."""

    def __init__(self, win_len: int = WIN_SAMPLES * 10, disp_seconds: float = 5.0):
        self._win  = collections.deque(maxlen=win_len)
        self._disp = collections.deque(maxlen=int(FS * disp_seconds))
        self._lock = threading.Lock()

    def push(self, val: float):
        with self._lock:
            self._win.append(val)
            self._disp.append(val)

    def get_window(self, n: int) -> np.ndarray | None:
        with self._lock:
            if len(self._win) < n:
                return None
            return np.array(list(self._win)[-n:], dtype=np.float32)

    def get_display(self) -> np.ndarray:
        with self._lock:
            return np.array(self._disp, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────
#  Filtro online + umbral adaptativo (para el panel de "activación EMG")
# ─────────────────────────────────────────────────────────────────────────
class EmgActivation:
    """Réplica simplificada del pipeline de filtrado/umbral de
    monitor_emg_gesture.py, solo para alimentar la lectura visual
    (no participa en la predicción del modelo, que usa su propio
    pipeline de features)."""

    def __init__(self):
        nyq = 0.5 * FS
        self.bp_b, self.bp_a = butter(FILT_ORDER, [F_LOW / nyq, F_HIGH / nyq], btype='band')
        self.nt_b, self.nt_a = iirnotch(F_NOTCH / nyq, 30)
        self._bp_zi = np.zeros(max(len(self.bp_a), len(self.bp_b)) - 1)
        self._nt_zi = np.zeros(max(len(self.nt_a), len(self.nt_b)) - 1)
        self.baseline_buf = collections.deque([2048.0] * (FS * 2), maxlen=FS * 2)
        self.env_hist = collections.deque([0.0] * 30, maxlen=30)
        self.baseline_mean = 2048.0
        self.baseline_std  = 50.0
        self.threshold_hi  = 2500.0
        self.active        = False
        self._n = 0

    def _filter(self, x: float) -> float:
        bp_b, bp_a, zi = self.bp_b, self.bp_a, self._bp_zi
        y_bp = bp_b[0] * x + zi[0]
        for k in range(len(zi) - 1):
            zi[k] = bp_b[k+1] * x - bp_a[k+1] * y_bp + zi[k+1]
        zi[-1] = bp_b[-1] * x - bp_a[-1] * y_bp
        nt_b, nt_a, nzi = self.nt_b, self.nt_a, self._nt_zi
        y_nt = nt_b[0] * y_bp + nzi[0]
        for k in range(len(nzi) - 1):
            nzi[k] = nt_b[k+1] * y_bp - nt_a[k+1] * y_nt + nzi[k+1]
        nzi[-1] = nt_b[-1] * y_bp - nt_a[-1] * y_nt
        return y_nt

    def push(self, raw: float):
        self._n += 1
        filt = self._filter(float(raw))
        self.env_hist.append(abs(filt))
        envelope = sum(self.env_hist) / len(self.env_hist)
        if not self.active:
            self.baseline_buf.append(float(raw))
        if self._n % (FS // 2) == 0 and len(self.baseline_buf) > FS:
            arr = np.array(self.baseline_buf)
            self.baseline_mean = float(np.mean(arr))
            self.baseline_std  = max(float(np.std(arr)), 10.0)
        self.threshold_hi = self.baseline_mean + K_THRESHOLD * self.baseline_std
        self.active = envelope > (K_THRESHOLD * self.baseline_std)
        return envelope


# ─────────────────────────────────────────────────────────────────────────
#  Predictor: modelo GNN real, o alternativa "dummy" reactiva a la señal
# ─────────────────────────────────────────────────────────────────────────
if _HAS_MODEL:
    class GesturePredictor:
        """Wrapper de inferencia: señal EMG → gesto predicho (modelo GNN real)."""

        def __init__(self, model, device, smoothing: int = 5, confidence_min: float = 0.50):
            self.model    = model
            self.device   = device
            self.hist     = collections.deque(maxlen=smoothing)
            self.conf_min = confidence_min

        def _emg_to_features(self, raw_window: np.ndarray):
            import pandas as pd
            from scipy.signal import sosfiltfilt

            w = raw_window.astype(np.float64)
            nyq = 0.5 * FS
            sos = butter(4, [20 / nyq, 95 / nyq], btype='band', output='sos')
            filt = sosfiltfilt(sos, w)

            rms  = float(np.sqrt(np.mean(filt ** 2)))
            mav  = float(np.mean(np.abs(filt)))
            wl   = float(np.sum(np.abs(np.diff(filt))))
            zc   = float(np.sum(np.diff(np.sign(filt)) != 0))
            tkeo = float(np.mean(np.maximum(filt[1:-1] ** 2 - filt[:-2] * filt[2:], 0)))
            rect = float(np.mean(np.abs(filt)))
            raw_v = float(np.mean(w))
            smo  = float(np.convolve(w, np.ones(10) / 10, mode='valid').mean())

            return pd.Series({
                "emg_raw": raw_v, "emg_smooth": smo, "emg_filt": float(np.mean(filt)),
                "emg_tkeo": tkeo, "emg_rms": rms, "emg_mav": mav,
                "emg_wl": wl, "emg_zc": zc, "emg_envelope": rect, "emg_rect": rect,
                "muscle_active": float(rms > 50), "dt_active": float(rms > 50),
                "active_pct": 50.0, "snr_db": 10.0,
                "f_thumb": 0, "f_index": 0, "f_middle": 0, "f_ring": 0, "f_pinky": 0,
                "pinch": 0.5, "hand_speed": 0.0,
            })

        def _row_to_pyg(self, row):
            feat_matrix = _build_node_features(row)
            adj, wts    = _build_virtual_graph(feat_matrix)
            src, dst    = np.where(adj > 0)
            x          = torch.tensor(feat_matrix, dtype=torch.float32)
            edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
            edge_attr  = torch.tensor(wts[src, dst, None], dtype=torch.float32)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        @torch.no_grad()
        def predict(self, raw_window: np.ndarray) -> dict:
            # ── Umbral de silencio (Reposo absoluto) ──
            # Calculamos la energía (RMS) de la ventana quitando el baseline DC (~2048)
            rms_raw = float(np.sqrt(np.mean((raw_window - raw_window.mean()) ** 2)))
            
            if rms_raw < 40.0:
                cls_abierta = GESTURE_MAP.get("ABIERTA", 0)
                self.hist.append(cls_abierta)
                probs = np.zeros(len(GESTURE_MAP))
                probs[cls_abierta] = 1.0
                return {"gesture": "ABIERTA", "confidence": 1.0,
                        "class_idx": cls_abierta, "probs": probs}

            # ── Inferencia GNN ──
            row  = self._emg_to_features(raw_window)
            data = self._row_to_pyg(row).to(self.device)
            batch = Batch.from_data_list([data])

            probs, cls = self.model.predict_gesture(batch, domain="sichiray")
            probs_np  = probs[0].cpu().numpy()
            pred_cls  = int(cls[0].item())

            self.hist.append(pred_cls)
            smooth_cls = collections.Counter(self.hist).most_common(1)[0][0]
            smooth_conf = float(probs_np[smooth_cls])

            gesture = INV_GESTURE.get(smooth_cls, "OTRO")
            if smooth_conf < self.conf_min:
                gesture = "OTRO"

            return {"gesture": gesture, "confidence": smooth_conf,
                    "class_idx": smooth_cls, "probs": probs_np}


class DummyPredictor:
    """Se usa solo si el paquete gnn_emg / torch no está disponible en esta
    máquina. Reacciona a la envolvente EMG real (contracción → PUNIO, reposo
    → ABIERTA) para poder probar la GUI de punta a punta sin el modelo."""

    def __init__(self, smoothing: int = 5, confidence_min: float = 0.50):
        self.hist = collections.deque(maxlen=smoothing)
        self.conf_min = confidence_min
        self.n_classes = len(GESTURE_MAP)

    def predict(self, raw_window: np.ndarray) -> dict:
        rms = float(np.sqrt(np.mean((raw_window - raw_window.mean()) ** 2)))
        active = rms > 120
        cls = GESTURE_MAP['PUNIO'] if active else GESTURE_MAP['ABIERTA']
        self.hist.append(cls)
        smooth_cls = collections.Counter(self.hist).most_common(1)[0][0]
        probs = np.full(self.n_classes, 0.02)
        conf = min(0.55 + rms / 400.0, 0.98)
        probs[smooth_cls] = conf
        gesture = INV_GESTURE.get(smooth_cls, "OTRO")
        return {"gesture": gesture, "confidence": conf,
                "class_idx": smooth_cls, "probs": probs}


# ─────────────────────────────────────────────────────────────────────────
#  Hilos de lectura de señal
# ─────────────────────────────────────────────────────────────────────────
def _serial_reader(port: str, baud: int, hub: SignalHub, stop: threading.Event):
    import serial
    ser = None
    try:
        ser = serial.Serial(port, baud, timeout=1.0)
        time.sleep(1.0)
        ser.reset_input_buffer()
        print(f"[Serial] Conectado a {port} @ {baud}")
        while not stop.is_set():
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            try:
                if "RAW:" in line:
                    raw = int(line.split("RAW:")[1].split(",")[0])
                elif "," in line:
                    raw = int(line.split(",")[0])
                else:
                    raw = int(line)
                hub.push(float(raw))
            except (ValueError, IndexError):
                pass
    except Exception as e:
        print(f"[Serial] Error: {e}")
    finally:
        if ser is not None and ser.is_open:
            ser.close()


def _demo_reader(hub: SignalHub, stop: threading.Event):
    """Genera señal EMG sintética sin hardware (con ráfagas de 'contracción')."""
    print("[DEMO] Generando señal EMG sintética...")
    t = 0
    while not stop.is_set():
        t += 1
        noise = np.random.normal(0, 30)
        burst = 0.0
        if (t % 400) < 80:
            burst = np.random.normal(0, 300) * np.sin(np.pi * (t % 400) / 80)
        hub.push(2048 + noise + burst)
        time.sleep(1.0 / FS)


# ─────────────────────────────────────────────────────────────────────────
#  GUI — EMG (izquierda) + Mano virtual grande (derecha)
# ─────────────────────────────────────────────────────────────────────────
def build_gui(hub: SignalHub, predictor, smoothing: int, stride_s: float = STRIDE_S):
    fig = plt.figure(figsize=(14, 7.5), facecolor=BG_DARK)
    fig.canvas.manager.set_window_title('Sichiray — GNN-EMG  Mano Virtual')

    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.35], wspace=0.06,
                            left=0.05, right=0.97, top=0.85, bottom=0.07)
    gs_left = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs[0], height_ratios=[2.2, 0.6, 1.4], hspace=0.45)

    # ── Panel 1: señal EMG cruda ────────────────────────────────────────
    ax_sig = fig.add_subplot(gs_left[0])
    ax_sig.set_facecolor(BG_SURF)
    for spine in ax_sig.spines.values():
        spine.set_color(DIVIDER)
    ax_sig.set_title('Señal EMG (cruda)', color=TEXT_C, fontsize=11, loc='left')
    ax_sig.tick_params(colors=TEXT_C, labelsize=8)
    ln_raw, = ax_sig.plot([], [], color=C_RAW, lw=1.0)
    ln_thr  = ax_sig.axhline(2500, color=C_THRESH, lw=0.8, ls='--', alpha=0.7)
    ax_sig.set_xlim(0, 5)
    ax_sig.set_ylim(1800, 2400)

    # ── Panel 2: barra de activación ─────────────────────────────────────
    ax_act = fig.add_subplot(gs_left[1])
    ax_act.set_facecolor(BG_SURF)
    for spine in ax_act.spines.values():
        spine.set_color(DIVIDER)
    ax_act.set_xlim(0, 1); ax_act.set_ylim(0, 1)
    ax_act.set_xticks([]); ax_act.set_yticks([])
    act_bg = ax_act.barh(0.5, 1.0, height=0.6, color=BG_SURF2)
    act_fill = ax_act.barh(0.5, 0.0, height=0.6, color=C_ENV)[0]
    act_label = ax_act.text(0.5, 0.5, 'Activación EMG', color=TEXT_C,
                             ha='center', va='center', fontsize=9)

    # ── Panel 3: lectura numérica + info de inferencia ──────────────────
    ax_txt = fig.add_subplot(gs_left[2])
    ax_txt.set_facecolor(BG_SURF)
    ax_txt.axis('off')
    txt_raw    = ax_txt.text(0.03, 0.90, '', color=TEXT_C, fontsize=10, transform=ax_txt.transAxes)
    txt_base   = ax_txt.text(0.03, 0.74, '', color=TEXT_C, fontsize=10, transform=ax_txt.transAxes)
    txt_thr    = ax_txt.text(0.03, 0.58, '', color=TEXT_C, fontsize=10, transform=ax_txt.transAxes)
    txt_model  = ax_txt.text(0.03, 0.38, '', color=TEXT_C, fontsize=9,  transform=ax_txt.transAxes)
    txt_status = ax_txt.text(0.03, 0.16, '', color=G_GRAY, fontsize=9,  transform=ax_txt.transAxes)

    # ── Panel grande: mano virtual ──────────────────────────────────────
    ax_hand = fig.add_subplot(gs[1])
    ax_hand.set_facecolor(BG_SURF)
    for spine in ax_hand.spines.values():
        spine.set_color(DIVIDER)
    ax_hand.set_xlim(-50, 190)
    ax_hand.set_ylim(-45, 205)
    ax_hand.set_aspect('equal')
    ax_hand.set_xticks([]); ax_hand.set_yticks([])

    hand_lines = [ax_hand.plot([], [], '-', lw=3, color=G_GRAY,
                                solid_capstyle='round')[0] for _ in HAND_CONNECTIONS]
    hand_dots  = ax_hand.scatter([], [], s=28, color=G_GRAY, zorder=5)
    gest_title = ax_hand.text(0.5, 1.10, '—', transform=ax_hand.transAxes,
                               ha='center', va='bottom', fontsize=22, color=G_GRAY,
                               fontweight='bold')
    gest_conf  = ax_hand.text(0.5, 1.02, '', transform=ax_hand.transAxes,
                               ha='center', va='bottom', fontsize=11, color=TEXT_C)

    animator = HandAnimator()
    state = {'last_infer_t': 0.0, 'result': None, 'prev_t': time.time()}

    def update(frame):
        now = time.time()
        dt = max(now - state['prev_t'], 1e-3)
        state['prev_t'] = now

        # ── Inferencia (cada stride_s) ──────────────────────────────────
        if now - state['last_infer_t'] >= stride_s:
            window = hub.get_window(WIN_SAMPLES)
            if window is not None:
                try:
                    result = predictor.predict(window)
                    state['result'] = result
                except Exception as e:
                    txt_status.set_text(f'✘ Error inferencia: {e}')
            state['last_infer_t'] = now

        result = state['result']
        if result is not None:
            animator.set_gesture(result['gesture'], result['confidence'])

        animator.step(dt)
        pts = animator.full_points()
        for line, (a, b) in zip(hand_lines, HAND_CONNECTIONS):
            gcolor = GESTURE_COLORS.get(animator.gesture, G_GRAY)
            line.set_data([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]])
            line.set_color(gcolor)
        hand_dots.set_offsets(pts)
        hand_dots.set_color(GESTURE_COLORS.get(animator.gesture, G_GRAY))

        gcolor = GESTURE_COLORS.get(animator.gesture, G_GRAY)
        gest_title.set_text(animator.gesture)
        gest_title.set_color(gcolor)
        gest_conf.set_text(f'Confianza: {animator.confidence:.0%}')
        gest_conf.set_color(gcolor)

        # ── Panel EMG ────────────────────────────────────────────────────
        disp = hub.get_display()
        if len(disp) > 1:
            t_axis = np.linspace(-len(disp) / FS, 0, len(disp))
            ln_raw.set_data(t_axis, disp)
            ax_sig.set_xlim(t_axis[0], 0)
            lo, hi = np.percentile(disp, [1, 99])
            pad = max((hi - lo) * 0.3, 20)
            ax_sig.set_ylim(lo - pad, hi + pad)

            rms = float(np.sqrt(np.mean((disp[-40:] - disp[-40:].mean()) ** 2))) if len(disp) >= 40 else 0.0
            act_norm = min(rms / 300.0, 1.0)
            act_fill.set_width(act_norm)
            act_fill.set_color(C_THRESH if act_norm > 0.6 else C_ENV)
            act_label.set_text(f'Activación EMG — RMS {rms:.0f}')

            txt_raw.set_text(f'EMG raw actual:  {disp[-1]:.0f}')
            txt_base.set_text(f'Media ventana:   {disp.mean():.0f}')
            txt_thr.set_text(f'RMS (últ. 40 ms): {rms:.1f}')

        if result is not None:
            txt_model.set_text(f'Clase: {result["gesture"]}   |   '
                                f'conf: {result["confidence"]:.0%}')
        if not _HAS_MODEL:
            txt_status.set_text('⚠ Modelo GNN no cargado — modo demo reactivo a RMS')
            txt_status.set_color('#e8c450')
        else:
            txt_status.set_text('✔ Modelo GNN-EMG activo (sich_best.pt)')
            txt_status.set_color('#45aa6d')

        return (*hand_lines, hand_dots, gest_title, gest_conf, ln_raw,
                act_fill, act_label, txt_raw, txt_base, txt_thr, txt_model, txt_status)

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()
    return ani


def auto_detect_port():
    import os
    import serial.tools.list_ports
    if os.path.exists('/dev/ttyACM0'):
        return '/dev/ttyACM0'
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if 'ttyS' in p.device and 'n/a' in (p.description or '').lower():
            continue
        desc = (p.description or '').lower()
        dev = (p.device or '').lower()
        if any(kw in desc for kw in ['cp210', 'ch340', 'ft232', 'uart', 'serial', 'esp32', 'jtag']) or any(kw in dev for kw in ['ttyusb', 'ttyacm']):
            return p.device
    for p in ports:
        if 'ttyS' not in p.device:
            return p.device
    return None

# ─────────────────────────────────────────────────────────────────────────
#  Entry point principal
# ─────────────────────────────────────────────────────────────────────────
def run_realtime(
    ckpt_path: str | None = None,
    port: str | None      = None,
    baud: int             = 115200,
    demo: bool            = False,
    smoothing: int        = 5,
    cfg_path: str | None  = None,
):
    predictor = None

    if _HAS_MODEL:
        cfg    = load_config(cfg_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pkg_root = Path(__file__).parent.parent
        ckpt_dir = (pkg_root / cfg.paths.checkpoints).resolve()

        if ckpt_path is None:
            sich_ckpt = ckpt_dir / "sich_best.pt"
            nina_ckpt = ckpt_dir / "nina_best.pt"
            if sich_ckpt.exists():
                ckpt_path = sich_ckpt
                print(f"[Checkpoint] Usando sichiray: {sich_ckpt}")
            elif nina_ckpt.exists():
                ckpt_path = nina_ckpt
                print(f"[Checkpoint] Usando ninapro: {nina_ckpt}")
            else:
                print("[WARNING] Sin checkpoint. Modelo inicializado aleatoriamente.")

        ckpt = None
        n_gestures_nina = cfg.model.n_gestures_nina
        if ckpt_path and Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            try:
                n_gestures_nina = ckpt["model"]["nina_gesture_head.net.3.weight"].shape[0]
            except KeyError:
                pass

        model = EMGGraphAutoencoder(
            nina_n_nodes    = cfg.ninapro.n_channels,
            nina_in_feat    = cfg.model.in_channels_nina,
            sich_n_nodes    = 5,
            sich_in_feat    = 7,
            hidden_channels = cfg.model.hidden_channels,
            latent_dim      = cfg.model.latent_dim,
            num_layers      = cfg.model.num_layers,
            heads           = cfg.model.num_heads,
            dropout         = 0.0,
            n_gestures_nina = n_gestures_nina,
            n_gestures_sich = cfg.model.n_gestures_sich,
            decoder_hidden  = list(cfg.model.decoder_hidden),
        ).to(device)

        if ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
            print(f"[Modelo] Checkpoint cargado (epoch {ckpt.get('epoch', '?')})")

        model.eval()
        predictor = GesturePredictor(
            model, device, smoothing=smoothing,
            confidence_min=cfg.inference.confidence_min,
        )
        stride_s = cfg.inference.stride_ms / 1000.0
    else:
        print(f"[AVISO] gnn_emg / torch no disponible ({_IMPORT_ERR}).")
        print("[AVISO] Iniciando con predictor de respaldo (reactivo a RMS) solo para probar la GUI.")
        predictor = DummyPredictor(smoothing=smoothing)
        stride_s = STRIDE_S

    hub  = SignalHub()
    stop = threading.Event()

    if not demo and port is None:
        port = auto_detect_port()
        if port:
            print(f"[Auto-Detect] Puerto encontrado: {port}")
        else:
            print("[AVISO] No se encontró ESP32/placa serial. Iniciando modo demo...")
            demo = True

    if demo or port is None:
        t = threading.Thread(target=_demo_reader, args=(hub, stop), daemon=True)
    else:
        t = threading.Thread(target=_serial_reader, args=(port, baud, hub, stop), daemon=True)
    t.start()

    print(f"[Init] Esperando {WIN_SAMPLES} muestras iniciales...")
    while hub.get_window(WIN_SAMPLES) is None:
        time.sleep(0.05)

    print("\n[GUI] Iniciando ventana — cierra la ventana para salir.\n")
    try:
        build_gui(hub, predictor, smoothing, stride_s=stride_s)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        t.join(timeout=2.0)
        print("\n[Inferencia] Detenida.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-EMG — Inferencia en tiempo real (GUI)")
    parser.add_argument("--ckpt",      type=str, default=None)
    parser.add_argument("--port",      type=str, default=None, help="Puerto serial, ej: /dev/ttyUSB0")
    parser.add_argument("--baud",      type=int, default=115200)
    parser.add_argument("--demo",      action="store_true", help="Modo demo sin hardware")
    parser.add_argument("--smoothing", type=int, default=5,  help="Frames para votación")
    parser.add_argument("--config",    type=str, default=None)
    args = parser.parse_args()

    run_realtime(
        ckpt_path = args.ckpt,
        port      = args.port,
        baud      = args.baud,
        demo      = args.demo,
        smoothing = args.smoothing,
        cfg_path  = args.config,
    )
