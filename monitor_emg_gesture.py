#!/usr/bin/env python3
"""
=============================================================
  SICHIRAY EMG + GESTURE MONITOR — Python Real-Time Dashboard
  EMG: lectura serial + procesamiento avanzado de señal EMG
  Gesto: MediaPipe Tasks API 0.10+ — reconocimiento de manos

  Integración: panel de gesto de mano en dashboard EMG
  Grabación CSV continua (estilo NinaPro) — seg. offline
=============================================================
"""

import sys, os, time, math, argparse, threading, collections, csv
from datetime import datetime
import numpy as np
import serial
import serial.tools.list_ports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from scipy.signal import butter, iirnotch

try:
    import cv2
    import mediapipe as mp
    import gesto_mano
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import HandLandmarksConnections
    _HAS_GESTURE = True
except ImportError:
    _HAS_GESTURE = False

# ─── PARÁMETROS CONFIGURABLES — EMG ────────────────────────────────────────────
FS          = 200
BAUD        = 115200
WINDOW_S    = 5
N_DISPLAY   = FS * WINDOW_S
BASELINE_S  = 2
N_BASELINE  = FS * BASELINE_S
K_THRESHOLD     = 1.5   # bajado de 3.5 → más sensible (flexores antebrazo palmar)
K_THRESHOLD_LO  = 1.0   # bajado de 2.0 → umbral de relajación
DEBOUNCE_MS     = 150
F_LOW    = 20.0
F_HIGH   = 150.0
F_NOTCH  = 60.0
FILT_ORDER = 4
TKEO_SMOOTHING = 15

# Colores EMG
C_RAW       = '#4f98a3'
C_SMOOTH    = '#fdc451'
C_FILT      = '#a86fdf'
C_TKEO      = '#fd6e6e'
C_THRESH_HI = '#e87050'
C_THRESH_LO = '#e8c450'
C_PULSE     = '#6daa45'

BG_DARK  = '#171614'
BG_SURF  = '#1c1b19'
BG_SURF2 = '#22211f'
TEXT_C   = '#cdccca'
DIVIDER  = '#393836'

# Colores gesto
G_GREEN   = '#45aa6d'
G_RED     = '#fd6e6e'
G_YELLOW  = '#fdc451'
G_ACCENT  = '#a86fdf'
G_GRAY    = '#7a7876'
G_CYAN    = '#4f98a3'

GESTURE_COLORS = {
    'ABIERTA':   G_GREEN,  'PUNIO':     G_RED,
    'INDICE':    G_YELLOW, 'PAZ':       G_ACCENT,
    'OK':        G_GREEN,  'PULGAR_UP': G_GREEN,
    'PULGAR_DN': G_RED,    'TRES':      G_YELLOW,
    'PINZA':     G_ACCENT, 'TELEFONO':  G_CYAN,
    'OTRO':      G_GRAY,
}

# ─── BUFFERS CIRCULARES — EMG ──────────────────────────────────────────────────
raw_buf      = collections.deque([2048]*N_DISPLAY, maxlen=N_DISPLAY)
smooth_buf   = collections.deque([2048]*N_DISPLAY, maxlen=N_DISPLAY)
filt_buf     = collections.deque([0.0]*N_DISPLAY, maxlen=N_DISPLAY)
tkeo_buf     = collections.deque([0.0]*N_DISPLAY, maxlen=N_DISPLAY)
rect_buf     = collections.deque([0.0]*N_DISPLAY, maxlen=N_DISPLAY)
envelope_buf = collections.deque([0.0]*N_DISPLAY, maxlen=N_DISPLAY)
pulse_buf    = collections.deque([0]*N_DISPLAY, maxlen=N_DISPLAY)

_raw_hist     = collections.deque([2048.0]*64, maxlen=64)
_tkeo_hist    = collections.deque([0.0]*TKEO_SMOOTHING, maxlen=TKEO_SMOOTHING)
_env_hist     = collections.deque([0.0]*30, maxlen=30)
baseline_buf  = collections.deque([2048.0]*N_BASELINE, maxlen=N_BASELINE)
baseline_ready = False
_smooth_buf_local = collections.deque([2048]*10, maxlen=10)

lock = threading.Lock()
metrics = {
    'raw': 0, 'smooth': 0, 'filt': 0.0, 'tkeo': 0.0,
    'rms': 0.0, 'mav': 0.0, 'wl': 0.0, 'zc': 0,
    'pulse_count': 0, 'active_pct': 0.0,
    'baseline_mean': 2048.0, 'baseline_std': 50.0,
    'threshold_hi': 2500.0, 'threshold_lo': 2300.0,
    'muscle_active': False, 'dt_active': False,
    'elapsed_s': 0.0, 'samples_total': 0,
    'snr_db': 0.0, 'baseline_ready': False,
}
serial_running = False
ser_obj = None
t_start = time.time()

# ─── FILTROS DIGITALES — EMG ───────────────────────────────────────────────────
def _butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low  = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def _notch_filter(f0, fs, Q=30):
    b, a = iirnotch(f0 / (0.5 * fs), Q)
    return b, a

bp_b, bp_a = _butter_bandpass(F_LOW, F_HIGH, FS, FILT_ORDER)
nt_b, nt_a = _notch_filter(F_NOTCH, FS)
_bp_zi   = np.zeros(max(len(bp_a), len(bp_b)) - 1)
_nt_zi   = np.zeros(max(len(nt_a), len(nt_b)) - 1)

def filter_sample_online(x):
    global _bp_zi, _nt_zi
    y_bp = bp_b[0] * x + _bp_zi[0]
    for k in range(len(_bp_zi) - 1):
        _bp_zi[k] = bp_b[k+1] * x - bp_a[k+1] * y_bp + _bp_zi[k+1]
    _bp_zi[-1] = bp_b[-1] * x - bp_a[-1] * y_bp
    y_nt = nt_b[0] * y_bp + _nt_zi[0]
    for k in range(len(_nt_zi) - 1):
        _nt_zi[k] = nt_b[k+1] * y_bp - nt_a[k+1] * y_nt + _nt_zi[k+1]
    _nt_zi[-1] = nt_b[-1] * y_bp - nt_a[-1] * y_nt
    return y_nt

# ─── TKEO ──────────────────────────────────────────────────────────────────────
_tkeo_prev2 = 2048.0
_tkeo_prev1 = 2048.0

def tkeo_sample(x_curr):
    global _tkeo_prev2, _tkeo_prev1
    tkeo_val = max(_tkeo_prev1**2 - _tkeo_prev2 * x_curr, 0.0)
    _tkeo_prev2 = _tkeo_prev1
    _tkeo_prev1 = x_curr
    return tkeo_val

def moving_avg(x, n=10):
    _smooth_buf_local.append(x)
    return sum(_smooth_buf_local) / len(_smooth_buf_local)

# ─── UMBRAL ADAPTATIVO — EMG ───────────────────────────────────────────────────
_dt_counter = 0
_dt_threshold_n = int(FS * 0.025)
_last_pulse_t   = 0.0
_active_samples = 0
_total_samples  = 0

def update_baseline(raw_val):
    global baseline_ready
    if not metrics['muscle_active']:
        baseline_buf.append(float(raw_val))
    if len(baseline_buf) >= N_BASELINE and not baseline_ready:
        baseline_ready = True
        metrics['baseline_ready'] = True

def compute_threshold():
    arr = np.array(baseline_buf)
    mu  = np.mean(arr)
    sd  = np.std(arr)
    sd  = max(sd, 10.0)
    metrics['baseline_mean'] = mu
    metrics['baseline_std']  = sd
    metrics['threshold_hi']  = mu + K_THRESHOLD    * sd
    metrics['threshold_lo']  = mu + K_THRESHOLD_LO * sd

# ─── PARSEO SERIAL — EMG ───────────────────────────────────────────────────────
def parse_line(line: str):
    line = line.strip()
    if not line:
        return None, None
    try:
        if 'RAW:' in line:
            parts = dict(p.split(':') for p in line.split(',') if ':' in p)
            raw = int(parts.get('RAW', 2048))
            smo = int(parts.get('SUAVIZADO', raw))
            return raw, smo
        elif ',' in line:
            vals = line.split(',')
            return int(vals[0]), int(vals[1]) if len(vals) > 1 else int(vals[0])
        else:
            v = int(line)
            return v, v
    except (ValueError, KeyError):
        return None, None

# ─── PROCESAMIENTO DE MUESTRA — EMG ────────────────────────────────────────────
def process_sample(raw, smo):
    global _dt_counter, _last_pulse_t, _active_samples, _total_samples, _csv_row_count
    now = time.time()
    _total_samples += 1
    filt = filter_sample_online(float(raw))
    raw_tkeo = tkeo_sample(filt)
    _tkeo_hist.append(raw_tkeo)
    tkeo_smooth = sum(_tkeo_hist) / len(_tkeo_hist)
    rect_val = abs(filt)
    _env_hist.append(rect_val)
    envelope = sum(_env_hist) / len(_env_hist)
    update_baseline(raw)
    if _total_samples % (FS // 2) == 0:
        compute_threshold()
    thr_hi = metrics['threshold_hi']
    thr_lo = metrics['threshold_lo']
    filt_thr_hi = K_THRESHOLD    * metrics['baseline_std']
    filt_thr_lo = K_THRESHOLD_LO * metrics['baseline_std']
    if envelope > filt_thr_hi:
        _dt_counter += 1
    else:
        _dt_counter = max(0, _dt_counter - 1)
    dt_active = _dt_counter >= _dt_threshold_n
    was_active = metrics['dt_active']
    new_pulse = False
    if dt_active and not was_active:
        dt_ms = (now - _last_pulse_t) * 1000
        if dt_ms > DEBOUNCE_MS:
            metrics['pulse_count'] += 1
            _last_pulse_t = now
            new_pulse = True
    if dt_active:
        _active_samples += 1
    with lock:
        raw_buf.append(raw)
        smooth_buf.append(smo)
        filt_buf.append(filt)
        tkeo_buf.append(tkeo_smooth)
        rect_buf.append(rect_val)
        envelope_buf.append(envelope)
        pulse_buf.append(800 if dt_active else 0)
        metrics['raw']           = raw
        metrics['smooth']        = smo
        metrics['filt']          = filt
        metrics['tkeo']          = tkeo_smooth
        metrics['muscle_active'] = raw > thr_hi
        metrics['dt_active']     = dt_active
        metrics['elapsed_s']     = now - t_start
        metrics['samples_total'] = _total_samples
        metrics['active_pct']    = 100.0 * _active_samples / max(_total_samples, 1)
    if _total_samples % 20 == 0:
        recent_filt = list(filt_buf)[-40:]
        arr = np.array(recent_filt, dtype=float)
        rms = float(np.sqrt(np.mean(arr**2)))
        mav = float(np.mean(np.abs(arr)))
        wl  = float(np.sum(np.abs(np.diff(arr))))
        zc  = int(np.sum(np.diff(np.sign(arr)) != 0))
        base_std = metrics['baseline_std']
        snr_db = 20.0 * np.log10(rms / max(base_std, 1e-6)) if rms > 0 else 0.0
        with lock:
            metrics['rms'] = rms
            metrics['mav'] = mav
            metrics['wl']  = wl
            metrics['zc']  = zc
            metrics['snr_db'] = snr_db

    # ── Grabación CSV continua (cada muestra EMG = 1 fila) ──────────────────
    if _recording and _hand_seen and _csv_writer is not None:
        with gesture_lock:
            g = dict(gesture_state)
        fingers = g.get('fingers', (False,)*5)
        row = [
            f'{now - t_start:.5f}',
            raw, smo,
            f'{filt:.4f}',
            f'{tkeo_smooth:.4f}',
            f'{metrics["rms"]:.4f}',
            f'{metrics["mav"]:.4f}',
            f'{metrics["wl"]:.4f}',
            metrics['zc'],
            f'{envelope:.4f}',
            f'{rect_val:.4f}',
            int(raw > thr_hi),
            int(dt_active),
            metrics['pulse_count'],
            f'{100.0 * _active_samples / max(_total_samples, 1):.2f}',
            f'{metrics["snr_db"]:.2f}',
            f'{metrics["baseline_mean"]:.2f}',
            f'{metrics["baseline_std"]:.2f}',
            f'{thr_hi:.2f}',
            f'{thr_lo:.2f}',
            g.get('gesture', '—'),
            f'{g.get("confidence", 0.0):.2f}',
            g.get('hand_label', '—'),
            int(fingers[0]), int(fingers[1]), int(fingers[2]),
            int(fingers[3]), int(fingers[4]),
            f'{g.get("pinch", 0.0):.4f}',
            g.get('wrist_x', 0),
            g.get('wrist_y', 0),
            f'{g.get("speed", 0.0):.2f}',
            g.get('swipe') or '',
        ]
        with _csv_lock:
            _csv_writer.writerow(row)
            _csv_row_count += 1  # contador de filas por sesión

# ─── HILO SERIAL — EMG ─────────────────────────────────────────────────────────
def serial_thread(port, baud):
    global serial_running, ser_obj
    try:
        ser_obj = serial.Serial(port, baud, timeout=1.0)
        time.sleep(1.5)
        ser_obj.reset_input_buffer()
        print(f"[OK] Conectado a {port} @ {baud} baud")
        serial_running = True
        while serial_running:
            try:
                line = ser_obj.readline().decode('utf-8', errors='ignore')
                raw, smo = parse_line(line)
                if raw is not None:
                    process_sample(raw, smo)
            except serial.SerialException:
                break
    except serial.SerialException as e:
        print(f"[ERROR] No se pudo abrir {port}: {e}")
        serial_running = False
    finally:
        if ser_obj and ser_obj.is_open:
            ser_obj.close()

# ─── MODO DEMO — EMG ───────────────────────────────────────────────────────────
def demo_thread():
    global serial_running
    serial_running = True
    t = 0
    print("[DEMO] Sin puerto serial — generando señal EMG sintética...")
    while serial_running:
        t += 1
        noise  = np.random.normal(0, 30)
        burst  = 0.0
        if (t % 400) < 80:
            burst = np.random.normal(0, 250) * np.sin(np.pi * (t % 400) / 80)
        raw_v = int(2048 + noise + burst)
        raw_v = max(0, min(4095, raw_v))
        process_sample(raw_v, raw_v)
        time.sleep(1.0 / FS)

# ────────────────────────────────────────────────────────────────────────────────
#  GESTO — RECONOCIMIENTO DE MANOS (MediaPipe Tasks API)
# ────────────────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'hand_landmarker.task')

PINCH_THRESH = 0.07
SWIPE_THRESH = 80
SWIPE_FRAMES = 12
TRAIL_LEN    = 40

# Índices de landmarks
WRIST=0; THUMB_TIP=4; INDEX_TIP=8; INDEX_PIP=6; INDEX_MCP=5
MIDDLE_TIP=12; MIDDLE_PIP=10; MIDDLE_MCP=9
RING_TIP=16; RING_PIP=14; RING_MCP=13
PINKY_TIP=20; PINKY_PIP=18; PINKY_MCP=17
THUMB_MCP=2

# Estado compartido del gesto
gesture_state = {
    'gesture': '—', 'confidence': 0.0, 'hand_label': '—',
    'fingers': (False, False, False, False, False),
    'pinch': 0.0, 'speed': 0.0, 'swipe': None,
    'fps': 0.0, 'active': False, 'camera_ok': False,
    'wrist_x': 0, 'wrist_y': 0,
}
gesture_lock = threading.Lock()
gesture_running = False

# ─── GRABACIÓN CSV (NinaPro style) ────────────────────────────────────────────
# Columnas: timestamp_s | emg_raw | emg_smooth | emg_filt | emg_tkeo | emg_rms
#           emg_mav | emg_wl | emg_zc | emg_envelope | emg_rect
#           muscle_active | dt_active | pulse_count | active_pct | snr_db
#           baseline_mean | baseline_std | threshold_hi | threshold_lo
#           gesture | confidence | hand_label
#           f_thumb | f_index | f_middle | f_ring | f_pinky
#           pinch | wrist_x | wrist_y | hand_speed | swipe
CSV_HEADER = [
    'timestamp_s',
    'emg_raw', 'emg_smooth', 'emg_filt', 'emg_tkeo',
    'emg_rms', 'emg_mav', 'emg_wl', 'emg_zc',
    'emg_envelope', 'emg_rect',
    'muscle_active', 'dt_active',
    'pulse_count', 'active_pct', 'snr_db',
    'baseline_mean', 'baseline_std', 'threshold_hi', 'threshold_lo',
    'gesture', 'confidence', 'hand_label',
    'f_thumb', 'f_index', 'f_middle', 'f_ring', 'f_pinky',
    'pinch', 'wrist_x', 'wrist_y', 'hand_speed', 'swipe',
]

_csv_file     = None   # archivo abierto
_csv_writer   = None   # csv.writer
_csv_lock     = threading.Lock()
_recording    = False  # True = CSV abierto y listo
_hand_seen    = False  # True = se detectó mano → graba
_csv_path     = ''     # ruta del archivo actual
_csv_row_count = 0     # filas escritas en la sesión actual
_session_num  = 0      # contador de sesiones — garantiza unicidad

def _unique_csv_path(out_dir):
    """Genera ruta única usando timestamp con microsegundos + contador de sesión.
    Nunca sobreescribe: si el archivo ya existe (colisión extrema), agrega sufijo."""
    global _session_num
    _session_num += 1
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')  # microsegundos incluidos
    base = f'sichiray_s{_session_num:02d}_{ts}.csv'
    path = os.path.join(out_dir, base)
    # Protección extra: si por alguna razón colisiona, agrega sufijo numérico
    counter = 0
    while os.path.exists(path):
        counter += 1
        path = os.path.join(out_dir, f'sichiray_s{_session_num:02d}_{ts}__{counter}.csv')
    return path

def init_csv_recording(out_dir='.'):
    """Abre un CSV nuevo con nombre único. No escribe hasta que se detecte mano."""
    global _csv_file, _csv_writer, _recording, _hand_seen, _csv_path, _csv_row_count
    close_csv_recording()  # cierra sesión anterior si hay una abierta
    os.makedirs(out_dir, exist_ok=True)
    _csv_path = _unique_csv_path(out_dir)
    # 'x' = exclusive creation — lanza FileExistsError si el archivo ya existe
    # (no puede pasar por _unique_csv_path, pero es la red de seguridad final)
    _csv_file     = open(_csv_path, 'x', newline='', buffering=1)
    _csv_writer   = csv.writer(_csv_file)
    _csv_writer.writerow(CSV_HEADER)
    _csv_row_count = 0
    _recording    = True
    _hand_seen    = False
    print(f'[REC] Sesión #{_session_num} lista → {_csv_path}')
    print('[REC] Esperando detección de mano para comenzar grabación...')
    return _csv_path

def close_csv_recording():
    """Cierra el CSV actual y reporta cuántas filas se grabaron."""
    global _csv_file, _recording, _hand_seen
    was_recording = _recording
    _recording = False
    _hand_seen = False
    if _csv_file and not _csv_file.closed:
        _csv_file.flush()
        _csv_file.close()
        if was_recording:
            print(f'[REC] ✔ Sesión #{_session_num} cerrada — {_csv_row_count:,} filas → {_csv_path}')

def lm_to_array(hand_landmarks, img_w, img_h):
    pts = []
    for lm in hand_landmarks:
        pts.append([lm.x * img_w, lm.y * img_h, lm.z * img_w])
    return np.array(pts, dtype=np.float32)

def dist2d(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def finger_extended(pts, tip, pip):
    return dist2d(pts[tip], pts[WRIST]) > dist2d(pts[pip], pts[WRIST]) * 1.15

def thumb_extended(pts, hand_label):
    tip_x, mcp_x = pts[THUMB_TIP][0], pts[THUMB_MCP][0]
    return tip_x < mcp_x - 15 if hand_label == 'Right' else tip_x > mcp_x + 15

def get_finger_states(pts, hand_label):
    t = thumb_extended(pts, hand_label)
    i = finger_extended(pts, INDEX_TIP,  INDEX_PIP)
    m = finger_extended(pts, MIDDLE_TIP, MIDDLE_PIP)
    r = finger_extended(pts, RING_TIP,   RING_PIP)
    p = finger_extended(pts, PINKY_TIP,  PINKY_PIP)
    return t, i, m, r, p

def pinch_dist(pts):
    hand_size = dist2d(pts[WRIST], pts[MIDDLE_MCP])
    return dist2d(pts[THUMB_TIP], pts[INDEX_TIP]) / max(hand_size, 1e-6)

def thumb_angle(pts):
    dx = pts[THUMB_TIP][0] - pts[THUMB_MCP][0]
    dy = pts[THUMB_TIP][1] - pts[THUMB_MCP][1]
    return math.degrees(math.atan2(-dy, dx))

def classify_gesture(pts, hand_label):
    t, i, m, r, p = get_finger_states(pts, hand_label)
    pd = pinch_dist(pts)
    ta = thumb_angle(pts)
    if pd < PINCH_THRESH:              return 'PINZA',     0.90
    if not any([t,i,m,r,p]):           return 'PUNIO',     0.95
    if all([t,i,m,r,p]):              return 'ABIERTA',   0.95
    if t and not i and not m and not r and not p:
        if ta > 45:                    return 'PULGAR_UP', 0.90
        if ta < -45:                   return 'PULGAR_DN', 0.90
    if not t and i and not m and not r and not p: return 'INDICE',   0.92
    if not t and i and m and not r and not p:     return 'PAZ',      0.92
    if not t and i and m and r and not p:         return 'TRES',     0.90
    if t and not i and not m and not r and p:     return 'TELEFONO', 0.88
    if t and not i and m and r and p:             return 'OK',       0.85
    return 'OTRO', 0.50

class HandTracker:
    def __init__(self):
        self.trail    = collections.deque(maxlen=TRAIL_LEN)
        self.vel_hist = collections.deque(maxlen=8)
        self.last_pos = None
        self.last_t   = None
        self.swipe    = None
        self.swipe_ttl = 0
        self.gest_hist = collections.deque(maxlen=10)

    def update(self, pos):
        now = time.time()
        self.trail.append(pos)
        vx = vy = spd = 0.0
        if self.last_pos and self.last_t:
            dt = now - self.last_t
            if dt > 0:
                vx = (pos[0]-self.last_pos[0]) / dt
                vy = (pos[1]-self.last_pos[1]) / dt
                spd = math.hypot(vx, vy)
        self.vel_hist.append((vx, vy, spd))
        self._detect_swipe()
        self.swipe_ttl = max(0, self.swipe_ttl - 1)
        if self.swipe_ttl == 0:
            self.swipe = None
        self.last_pos, self.last_t = pos, now
        return vx, vy, spd

    def _detect_swipe(self):
        if len(self.trail) < SWIPE_FRAMES:
            return
        rec = list(self.trail)[-SWIPE_FRAMES:]
        dx = rec[-1][0] - rec[0][0]
        dy = rec[-1][1] - rec[0][1]
        if abs(dx) < SWIPE_THRESH and abs(dy) < SWIPE_THRESH:
            return
        new = ('RIGHT' if dx > 0 else 'LEFT') if abs(dx) > abs(dy) \
              else ('DOWN' if dy > 0 else 'UP')
        if new != self.swipe:
            self.swipe = new
            self.swipe_ttl = 20

    def smooth(self, g):
        self.gest_hist.append(g)
        return collections.Counter(self.gest_hist).most_common(1)[0][0]

    @property
    def avg_speed(self):
        return np.mean([v[2] for v in self.vel_hist]) if self.vel_hist else 0.0

_gesture_tracker = HandTracker()

SWIPE_LABELS = {'LEFT':'← IZQ', 'RIGHT':'→ DER', 'UP':'↑ ARR', 'DOWN':'↓ ABA'}
FINGER_NAMES = ['Pulgar','Indice','Medio','Anular','Menique']

# ─── HILO DE CÁMARA + GESTO ────────────────────────────────────────────────────
def gesture_thread(cam_index=0, mirror=True):
    global gesture_running
    if not _HAS_GESTURE or not os.path.exists(MODEL_PATH):
        print(f"[GESTO] Dependencias faltantes o modelo no encontrado: {MODEL_PATH}")
        print("[GESTO] Gestos desactivados — instala con:")
        print("  pip install opencv-python mediapipe")
        print("  python3 -c \"import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task','hand_landmarker.task')\"")
        gesture_running = False
        return

    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[GESTO] ERROR: No se pudo abrir cámara {cam_index}")
        gesture_running = False
        return

    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    print(f"[GESTO] Cámara {cam_index} iniciada — reconociendo gestos")
    cv2.namedWindow('Sichiray — Gesture Monitor  (Q=salir)', cv2.WINDOW_NORMAL)
    gesture_running = True
    fps_buf = collections.deque(maxlen=30)
    frame_ms = 0

    while gesture_running:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        if mirror:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ms += 33
        result = landmarker.detect_for_video(mp_img, frame_ms)

        gest = '—'
        conf = 0.0
        label = '—'
        fingers = (False, False, False, False, False)
        pd_val = 0.0
        spd = 0.0
        swipe = None
        wx = wy = 0
        active = False
        hand_list = []

        if result.hand_landmarks:
            for hand_lm, handedness in zip(result.hand_landmarks, result.handedness):
                lbl = handedness[0].display_name
                pts = lm_to_array(hand_lm, w, h)
                gesture_name, gesture_conf = classify_gesture(pts, lbl)
                wx = int(pts[WRIST][0])
                wy = int(pts[WRIST][1])
                _gesture_tracker.update((wx, wy))
                gesture_name = _gesture_tracker.smooth(gesture_name)
                fst = get_finger_states(pts, lbl)
                pd_val = pinch_dist(pts)
                spd = _gesture_tracker.avg_speed
                swipe = _gesture_tracker.swipe
                gest = gesture_name
                conf = gesture_conf
                label = lbl
                fingers = fst
                active = True

                # ── Disparar grabación al ver la primera mano ──
                global _hand_seen
                if _recording and not _hand_seen:
                    _hand_seen = True
                    print('[REC] ✅ Mano detectada — grabación iniciada.')
                
                gc = gesto_mano.GESTURE_COLORS.get(gest, gesto_mano.C_GRAY)
                gesto_mano.draw_landmarks_custom(frame, pts, gc)
                gesto_mano.draw_trail(frame, _gesture_tracker.trail, gc)
                if swipe:
                    gesto_mano.draw_swipe(frame, swipe, wx, wy)
                
                lx = max(0, wx-60)
                ly = max(30, wy-30)
                cv2.putText(frame, gest, (lx, ly), gesto_mano.FONT, 0.7, gc, 2, cv2.LINE_AA)
                cv2.putText(frame, lbl, (lx, ly+20), gesto_mano.FONT, 0.4, gesto_mano.C_GRAY, 1, cv2.LINE_AA)
                
                hand_list.append({'label':lbl, 'gesture':gest, 'conf':conf,
                                  'fingers':fst, 'pinch':pd_val, 'speed':spd, 'swipe':swipe})
                break  # solo primera mano

        t1 = time.time()
        fps_buf.append(1.0 / max(t1 - t0, 1e-6))
        fps = float(np.mean(fps_buf))
        cv2.putText(frame, f'FPS:{fps:.1f}', (w-110,25), gesto_mano.FONT, 0.5, gesto_mano.C_ACCENT, 1, cv2.LINE_AA)

        panel = gesto_mano.build_panel(h, hand_list)
        fw    = gesto_mano.DISPLAY_W - gesto_mano.PANEL_W
        scale = fw / w
        disp  = cv2.resize(frame, (fw, int(h*scale)))
        pan   = cv2.resize(panel, (gesto_mano.PANEL_W, disp.shape[0]))
        out   = np.hstack([disp, pan])
        cv2.imshow('Sichiray — Gesture Monitor  (Q=salir)', out)

        with gesture_lock:
            gesture_state.update({
                'gesture': gest, 'confidence': conf, 'hand_label': label,
                'fingers': fingers, 'pinch': pd_val, 'speed': spd,
                'swipe': swipe, 'fps': fps, 'active': active,
                'camera_ok': True, 'wrist_x': wx, 'wrist_y': wy,
            })

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            gesture_running = False
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("[GESTO] Cámara detenida.")

# ─── DASHBOARD MATPLOTLIB ──────────────────────────────────────────────────────
def build_dashboard():
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 9), facecolor=BG_DARK)
    fig.canvas.manager.set_window_title('Sichiray EMG + Gesture Monitor | ESP32 + MediaPipe')

    gs = gridspec.GridSpec(4, 4, figure=fig,
                           left=0.04, right=0.98,
                           top=0.93, bottom=0.06,
                           hspace=0.45, wspace=0.30)

    ax_main  = fig.add_subplot(gs[0:2, 0:2])
    ax_filt  = fig.add_subplot(gs[2, 0:2])
    ax_tkeo  = fig.add_subplot(gs[3, 0:2])
    ax_info  = fig.add_subplot(gs[0:2, 2])
    ax_gesture = fig.add_subplot(gs[0:2, 3])
    ax_hist  = fig.add_subplot(gs[2, 2:4])
    ax_bar   = fig.add_subplot(gs[3, 2:4])

    for ax in [ax_main, ax_filt, ax_tkeo, ax_info, ax_gesture, ax_hist, ax_bar]:
        ax.set_facecolor(BG_SURF)
        for spine in ax.spines.values():
            spine.set_edgecolor(DIVIDER)
        ax.tick_params(colors=TEXT_C, labelsize=7)
        ax.xaxis.label.set_color(TEXT_C)
        ax.yaxis.label.set_color(TEXT_C)

    t_ax = np.linspace(-WINDOW_S, 0, N_DISPLAY)

    # ── Señal principal ──
    ax_main.set_title('Señal EMG — RAW / Suavizado / Umbrales adaptativos',
                       color=TEXT_C, fontsize=9, pad=4)
    ax_main.set_ylabel('ADC (12-bit)', fontsize=8)
    ax_main.set_xlim(-WINDOW_S, 0)
    ax_main.set_ylim(0, 4095)
    ln_raw,   = ax_main.plot(t_ax, list(raw_buf),    color=C_RAW,   lw=0.8, alpha=0.5, label='RAW')
    ln_smo,   = ax_main.plot(t_ax, list(smooth_buf), color=C_SMOOTH, lw=1.2, label='Suavizado (ESP32 MA×10)')
    ln_thr_hi = ax_main.axhline(metrics['threshold_hi'], color=C_THRESH_HI, lw=1, ls='--', alpha=0.8, label=f'Umbral AT ({K_THRESHOLD}σ)')
    ln_thr_lo = ax_main.axhline(metrics['threshold_lo'], color=C_THRESH_LO, lw=1, ls=':', alpha=0.7, label=f'Umbral DT ({K_THRESHOLD_LO}σ)')
    ln_pul,   = ax_main.plot(t_ax, list(pulse_buf),  color=C_PULSE, lw=0, marker='None', alpha=0)
    ax_main.fill_between(t_ax, [800 if p else 0 for p in pulse_buf],
                         alpha=0.15, color=C_PULSE)
    ax_main.legend(loc='upper left', fontsize=7, framealpha=0.3,
                   facecolor=BG_SURF2, edgecolor=DIVIDER, labelcolor=TEXT_C)

    # ── Señal filtrada + envelope ──
    ax_filt.set_title('Filtrada (Butterworth 20-150Hz + Notch 60Hz) + Envelope',
                       color=TEXT_C, fontsize=9, pad=4)
    ax_filt.set_ylabel('Amplitud', fontsize=8)
    ax_filt.set_xlim(-WINDOW_S, 0)
    ln_filt, = ax_filt.plot(t_ax, list(filt_buf),     color=C_FILT, lw=0.8, alpha=0.7, label='Filtrada')
    ln_env,  = ax_filt.plot(t_ax, list(envelope_buf), color=C_SMOOTH, lw=1.5, label='Envelope (MAV LP)')
    ln_fth   = ax_filt.axhline(0, color=C_THRESH_HI, lw=1, ls='--', alpha=0.8)
    ax_filt.legend(loc='upper left', fontsize=7, framealpha=0.3,
                   facecolor=BG_SURF2, edgecolor=DIVIDER, labelcolor=TEXT_C)

    # ── TKEO ──
    ax_tkeo.set_title('TKEO — Teager-Kaiser Energy Operator',
                       color=TEXT_C, fontsize=9, pad=4)
    ax_tkeo.set_xlabel('Tiempo (s)', fontsize=8)
    ax_tkeo.set_ylabel('Energía TKEO', fontsize=8)
    ax_tkeo.set_xlim(-WINDOW_S, 0)
    ln_tkeo, = ax_tkeo.plot(t_ax, list(tkeo_buf), color=C_TKEO, lw=0.9, label='TKEO suavizado')
    ax_tkeo.legend(loc='upper left', fontsize=7, framealpha=0.3,
                   facecolor=BG_SURF2, edgecolor=DIVIDER, labelcolor=TEXT_C)

    # ── Panel de métricas — EMG ──
    ax_info.axis('off')
    ax_info.set_title('Métricas EMG', color=TEXT_C, fontsize=9, pad=4)
    emg_metrics = [
        ('ADC Raw / Smooth',  'raw_smooth', '', '#cdccca'),
        ('RMS (filtrada)',    'rms',        '', C_FILT),
        ('SNR estimado',      'snr_db',     ' dB', C_PULSE),
        ('Pulsos detectados', 'pulse_count','',   C_PULSE),
        ('Tiempo activo',     'active_pct', ' %', C_PULSE),
        ('Baseline μ / σ',    'baseline',   '',   C_THRESH_HI),
        ('Umbral hi (AT)',    'threshold_hi','',  C_THRESH_HI),
    ]
    info_texts = []
    for i, (label, key, unit, color) in enumerate(emg_metrics):
        y = 0.94 - i * 0.10
        ax_info.text(0.03, y, label, transform=ax_info.transAxes,
                     color='#7a7876', fontsize=7, va='top')
        t_obj = ax_info.text(0.65, y, '—', transform=ax_info.transAxes,
                             color=color, fontsize=8, fontweight='bold', va='top',
                             fontfamily='monospace')
        info_texts.append((t_obj, key, unit))

    baseline_status = ax_info.text(0.03, 0.04, '[*] Calibrando baseline...',
                                    transform=ax_info.transAxes,
                                    color=C_THRESH_HI, fontsize=7, va='bottom')

    muscle_patch = mpatches.FancyBboxPatch((0.02, -0.08), 0.96, 0.06,
                                           boxstyle='round,pad=0.01',
                                           facecolor=BG_SURF2, edgecolor=DIVIDER,
                                           transform=ax_info.transAxes)
    ax_info.add_patch(muscle_patch)
    muscle_text = ax_info.text(0.50, -0.05, '● REPOSO', transform=ax_info.transAxes,
                                ha='center', va='center', fontsize=10,
                                fontweight='bold', color='#7a7876')

    # ── Panel de gesto ──
    ax_gesture.axis('off')
    ax_gesture.set_title('Reconocimiento de Gesto (MediaPipe)',
                          color=TEXT_C, fontsize=9, pad=4)

    gest_header  = ax_gesture.text(0.50, 0.94, '—', transform=ax_gesture.transAxes,
                                   ha='center', va='top', fontsize=16,
                                   fontweight='bold', color=G_GRAY)
    gest_conf    = ax_gesture.text(0.50, 0.78, '—', transform=ax_gesture.transAxes,
                                   ha='center', va='top', fontsize=9, color=G_GRAY)
    gest_hand    = ax_gesture.text(0.50, 0.68, '—', transform=ax_gesture.transAxes,
                                   ha='center', va='top', fontsize=8, color=G_GRAY)
    gest_fingers = ax_gesture.text(0.10, 0.56, '—', transform=ax_gesture.transAxes,
                                   va='top', fontsize=7, color=G_GRAY,
                                   fontfamily='monospace')
    gest_pinch   = ax_gesture.text(0.10, 0.40, '—', transform=ax_gesture.transAxes,
                                   va='top', fontsize=7, color=G_GRAY)
    gest_swipe   = ax_gesture.text(0.10, 0.30, '—', transform=ax_gesture.transAxes,
                                   va='top', fontsize=7, color=G_GRAY)
    gest_speed   = ax_gesture.text(0.10, 0.20, '—', transform=ax_gesture.transAxes,
                                   va='top', fontsize=7, color=G_GRAY)
    gest_fps     = ax_gesture.text(0.10, 0.10, '—', transform=ax_gesture.transAxes,
                                   va='top', fontsize=7, color=G_GRAY)
    gest_status  = ax_gesture.text(0.50, 0.02, '—', transform=ax_gesture.transAxes,
                                   ha='center', va='bottom', fontsize=7, color=G_GRAY)

    gest_objs = (gest_header, gest_conf, gest_hand, gest_fingers,
                 gest_pinch, gest_swipe, gest_speed, gest_fps, gest_status)

    # ── Histograma ADC ──
    ax_hist.set_title('Distribución ADC (últimas 1000 muestras)',
                       color=TEXT_C, fontsize=9, pad=4)
    ax_hist.set_xlabel('Valor ADC', fontsize=8)
    ax_hist.set_ylabel('Frecuencia', fontsize=8)
    hist_data = [0] * 32
    hist_bars = ax_hist.bar(range(32), hist_data, color=C_RAW, alpha=0.7,
                             edgecolor='none', width=0.9)
    ax_hist.set_xlim(-0.5, 31.5)

    # ── Barra de actividad ──
    ax_bar.set_title('Nivel de activación muscular', color=TEXT_C, fontsize=9, pad=4)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, 1)
    ax_bar.axis('off')
    act_bg   = mpatches.FancyBboxPatch((0.05, 0.15), 0.90, 0.45,
                                        boxstyle='round,pad=0.01',
                                        facecolor=BG_SURF2, edgecolor=DIVIDER)
    act_fill = mpatches.FancyBboxPatch((0.05, 0.15), 0.01, 0.45,
                                        boxstyle='round,pad=0.005',
                                        facecolor=C_PULSE, edgecolor='none')
    ax_bar.add_patch(act_bg)
    ax_bar.add_patch(act_fill)
    act_text = ax_bar.text(0.50, 0.72, '0.0 %', ha='center', va='bottom',
                           color=TEXT_C, fontsize=11, fontweight='bold')
    act_method = ax_bar.text(0.50, 0.05, 'Double Threshold + Debounce',
                              ha='center', va='bottom', color='#7a7876', fontsize=7)
    baseline_bar = ax_bar.text(0.50, -0.05, 'K=3.5σ  |  Debounce=150ms',
                                ha='center', va='bottom', color='#7a7876', fontsize=7)

    # Título global
    fig.text(0.50, 0.97, 'SICHIRAY EMG + GESTURE MONITOR  |  ESP32 + MediaPipe',
             ha='center', va='top', color=TEXT_C, fontsize=11, fontweight='bold')
    fig.text(0.50, 0.955, f'EMG: BP {F_LOW}-{F_HIGH}Hz | Notch {F_NOTCH}Hz | TKEO | AT ({K_THRESHOLD}σ) | FS={FS}Hz    |    Gesto: MediaPipe HandLandmarker',
             ha='center', va='top', color='#7a7876', fontsize=8)

    # ── Botón [■ DETENER GRABACIÓN] ───────────────────────────────────────────
    # Eje del botón: esquina inferior derecha [left, bottom, width, height]
    ax_btn    = fig.add_axes([0.775, 0.005, 0.21, 0.042])
    btn_stop  = Button(ax_btn, '● REC  —  click para detener',
                       color='#5a1010', hovercolor='#8a1a1a')
    ax_btn.set_facecolor('#5a1010')
    for sp in ax_btn.spines.values():
        sp.set_edgecolor('#fd6e6e')
    btn_stop.label.set_color('#fd6e6e')
    btn_stop.label.set_fontsize(8)
    btn_stop.label.set_fontweight('bold')

    # Texto contador de filas — izquierda del botón
    rec_info = fig.text(0.555, 0.014, '',
                        ha='left', va='center', color='#fd6e6e',
                        fontsize=7.5, fontfamily='monospace')

    _btn_state = {'stopped': False, 'rows': 0, 'out_dir': './dataset'}

    def _set_btn_recording():
        """Pone el botón en estado REC (rojo)."""
        btn_stop.label.set_text('● REC  —  click para detener')
        btn_stop.label.set_color('#fd6e6e')
        ax_btn.set_facecolor('#5a1010')
        for sp in ax_btn.spines.values():
            sp.set_edgecolor('#fd6e6e')
        rec_info.set_color('#fd6e6e')

    def _set_btn_stopped(rows, path):
        """Pone el botón en estado GUARDADO (verde) + opción de nueva sesión."""
        short = os.path.basename(path) if path else ''
        btn_stop.label.set_text(f'✔ {rows:,} filas guardadas  —  click para nueva sesión')
        btn_stop.label.set_color('#6daa45')
        ax_btn.set_facecolor('#0e2e10')
        for sp in ax_btn.spines.values():
            sp.set_edgecolor('#6daa45')
        rec_info.set_text(f'✔ {short}')
        rec_info.set_color('#6daa45')

    def on_stop(event):
        if not _btn_state['stopped']:
            # ── Detener sesión actual ──
            _btn_state['stopped'] = True
            rows = _csv_row_count
            path = _csv_path
            close_csv_recording()
            _set_btn_stopped(rows, path)
        else:
            # ── Iniciar NUEVA sesión ──
            _btn_state['stopped'] = False
            init_csv_recording(_btn_state['out_dir'])
            _set_btn_recording()
        fig.canvas.draw_idle()

    btn_stop.on_clicked(on_stop)

    # ─── FUNCIÓN DE ACTUALIZACIÓN ───────────────────────────────────────────
    def update(frame):
        with lock:
            m = dict(metrics)
            raw_data  = list(raw_buf)
            smo_data  = list(smooth_buf)
            filt_data = list(filt_buf)
            tkeo_data = list(tkeo_buf)
            env_data  = list(envelope_buf)
            pul_data  = list(pulse_buf)

        with gesture_lock:
            g = dict(gesture_state)

        # Señal principal
        ln_raw.set_ydata(raw_data)
        ln_smo.set_ydata(smo_data)
        ln_thr_hi.set_ydata([m['threshold_hi']] * 2)
        ln_thr_lo.set_ydata([m['threshold_lo']] * 2)
        lo = max(0, m['baseline_mean'] - 6 * m['baseline_std'])
        hi = min(4095, m['baseline_mean'] + 6 * m['baseline_std'])
        if hi > lo + 100:
            ax_main.set_ylim(lo, hi)

        # Señal filtrada
        ln_filt.set_ydata(filt_data)
        ln_env.set_ydata(env_data)
        filt_thr = K_THRESHOLD * m['baseline_std']
        ln_fth.set_ydata([filt_thr, filt_thr])
        filt_range = max(np.max(np.abs(filt_data)), filt_thr * 1.5, 100)
        ax_filt.set_ylim(-filt_range, filt_range)

        # TKEO
        ln_tkeo.set_ydata(tkeo_data)
        tkeo_max = max(np.max(tkeo_data) * 1.1, 1.0)
        ax_tkeo.set_ylim(0, tkeo_max)

        # Métricas EMG
        raw_smooth = f"{m['raw']:4d} / {m['smooth']:4d}"
        baseline  = f"{m['baseline_mean']:.0f} / {m['baseline_std']:.1f}"
        fmt = {
            'raw_smooth':  raw_smooth,
            'rms':         f"{m['rms']:.1f}",
            'snr_db':      f"{m['snr_db']:.1f}",
            'pulse_count': f"{m['pulse_count']:4d}",
            'active_pct':  f"{m['active_pct']:.1f}",
            'baseline':    baseline,
            'threshold_hi':f"{m['threshold_hi']:.0f}",
        }
        for t_obj, key, unit in info_texts:
            t_obj.set_text(fmt.get(key, '—') + unit)

        if m['baseline_ready']:
            baseline_status.set_text('✔ Baseline calibrado')
            baseline_status.set_color(C_PULSE)
        else:
            pct = min(100, int(100 * m['samples_total'] / N_BASELINE))
            baseline_status.set_text(f'[*] Calibrando baseline... {pct}%')

        if m['dt_active']:
            muscle_text.set_text('⚡ CONTRACCIÓN')
            muscle_text.set_color(C_TKEO)
            muscle_patch.set_facecolor('#3a1a1a')
        else:
            muscle_text.set_text('● REPOSO')
            muscle_text.set_color('#7a7876')
            muscle_patch.set_facecolor(BG_SURF2)

        # Histograma
        hist_vals = np.array(raw_data[-1000:])
        counts, _ = np.histogram(hist_vals, bins=32, range=(0, 4095))
        for bar, h in zip(hist_bars, counts):
            bar.set_height(h)
        ax_hist.set_ylim(0, max(counts.max() * 1.1, 1))

        # Barra de activación
        rms_norm = min(m['rms'] / max(m['baseline_std'] * K_THRESHOLD * 2, 1), 1.0)
        act_fill.set_width(0.90 * rms_norm)
        act_text.set_text(f"{m['active_pct']:.1f} %")
        act_color = C_TKEO if m['dt_active'] else (C_SMOOTH if rms_norm > 0.3 else C_PULSE)
        act_fill.set_facecolor(act_color)

        # ── Actualizar panel de gesto ──
        if g['camera_ok'] and g['active']:
            gc = GESTURE_COLORS.get(g['gesture'], G_GRAY)
            gest_header.set_text(g['gesture'])
            gest_header.set_color(gc)
            gest_conf.set_text(f"Confianza: {g['confidence']:.0%}")
            gest_conf.set_color(gc)
            gest_hand.set_text(f"Mano: {g['hand_label']}")
            gest_hand.set_color(TEXT_C)

            finger_chars = ['●' if f else '○' for f in g['fingers']]
            finger_line = '  '.join(f'{n}: {c}' for n, c in zip(FINGER_NAMES, finger_chars))
            gest_fingers.set_text(f'Dedos:  {finger_line}')
            gest_fingers.set_color(TEXT_C)

            gest_pinch.set_text(f"Pinza: {g['pinch']:.2f}")
            gest_pinch.set_color(G_ACCENT)

            sw = g['swipe']
            if sw:
                gest_swipe.set_text(f"Swipe: {SWIPE_LABELS.get(sw, sw)}")
                gest_swipe.set_color(G_YELLOW)
            else:
                gest_swipe.set_text('Swipe: —')
                gest_swipe.set_color(G_GRAY)

            gest_speed.set_text(f"Velocidad: {g['speed']:.0f} px/s")
            gest_speed.set_color(G_CYAN)
            gest_fps.set_text(f"FPS cámara: {g['fps']:.1f}")
            gest_fps.set_color(G_CYAN)
            gest_status.set_text('✔ Cámara activa')
            gest_status.set_color(G_GREEN)
        elif g['camera_ok'] and not g['active']:
            gest_header.set_text('—')
            gest_conf.set_text('Sin mano detectada')
            gest_conf.set_color(G_GRAY)
            gest_status.set_text('✔ Cámara activa — esperando mano...')
            gest_status.set_color(G_YELLOW)
        else:
            gest_header.set_text('—')
            gest_conf.set_text('Cámara no disponible')
            gest_conf.set_color(G_RED)
            gest_status.set_text('✘ Cámara desconectada')
            gest_status.set_color(G_RED)

        return (ln_raw, ln_smo, ln_filt, ln_env, ln_tkeo,
                *[t for t, _, _ in info_texts],
                muscle_text, baseline_status, act_fill, act_text,
                *gest_objs)

    # Guardar out_dir en btn_state para que on_stop pueda re-iniciar sesiones
    # (se sobreescribe después del bloque build_dashboard desde main)
    _btn_state['out_dir'] = './dataset'

    # Actualizar contador de filas en el botón cada ciclo de animación
    _orig_update = update
    def update_with_rec(frame):
        artists = _orig_update(frame)
        if not _btn_state['stopped']:
            if _recording and _hand_seen:
                rec_info.set_text(f'● REC  {_csv_row_count:,} filas  |  {metrics.get("elapsed_s", 0):.0f}s')
            elif _recording and not _hand_seen:
                rec_info.set_text('⏳ Esperando mano...')
            else:
                rec_info.set_text('')
        return artists

    ani = FuncAnimation(fig, update_with_rec, interval=50, blit=False, cache_frame_data=False)
    plt.show()
    return ani

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def list_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No se encontraron puertos seriales.")
    else:
        print("Puertos disponibles:")
        for p in ports:
            print(f"  {p.device:20s} — {p.description}")

def auto_detect_port():
    import os
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

def main():
    global serial_running, gesture_running
    parser = argparse.ArgumentParser(description='Sichiray EMG + Gesture Monitor')
    parser.add_argument('--port', '-p', help='Puerto serial (ej: /dev/ttyUSB0, COM3)')
    parser.add_argument('--baud', '-b', type=int, default=BAUD, help=f'Baudrate (default: {BAUD})')
    parser.add_argument('--list', '-l', action='store_true', help='Lista puertos disponibles')
    parser.add_argument('--demo', action='store_true', help='Modo demo sin hardware')
    parser.add_argument('--cam', type=int, default=0, help='Índice de cámara (default: 0)')
    parser.add_argument('--no-gesture', action='store_true', help='Desactivar reconocimiento de gestos')
    parser.add_argument('--no-mirror', action='store_true', help='No espejar cámara')
    # ── Grabación CSV ──
    parser.add_argument('--no-record', action='store_true',
                        help='Desactivar grabación CSV (por defecto siempre graba)')
    parser.add_argument('--out-dir', default='./dataset',
                        help='Directorio de salida para CSVs (default: ./dataset)')
    args = parser.parse_args()

    # Grabación activa por defecto, a menos que se pase --no-record
    if not args.no_record:
        init_csv_recording(args.out_dir)

    if args.list:
        list_ports()
        return

    # Iniciar EMG
    if args.demo:
        t = threading.Thread(target=demo_thread, daemon=True)
        t.start()
    else:
        port = args.port or auto_detect_port()
        if port is None:
            print("[AVISO] No se encontró ESP32. Iniciando modo demo...")
            t = threading.Thread(target=demo_thread, daemon=True)
            t.start()
        else:
            print(f"[INFO] Intentando conectar a {port}...")
            t = threading.Thread(target=serial_thread, args=(port, args.baud), daemon=True)
            t.start()
            for _ in range(25):
                if serial_running or not t.is_alive():
                    break
                time.sleep(0.1)
            if not serial_running:
                print("[AVISO] No se pudo conectar. Iniciando modo demo...")
                t = threading.Thread(target=demo_thread, daemon=True)
                t.start()

    # Iniciar gesto
    if not _HAS_GESTURE:
        print("[GESTO] opencv-python / mediapipe no instalados — gestos desactivados.")
        print("  Para activar: pip install opencv-python mediapipe")
    elif not args.no_gesture:
        gt = threading.Thread(target=gesture_thread,
                              args=(args.cam, not args.no_mirror),
                              daemon=True)
        gt.start()
        time.sleep(3.0)
        if not gesture_running:
            print("[GESTO] Cámara no disponible — gestos desactivados.")
    else:
        print("[GESTO] Reconocimiento de gestos desactivado.")

    try:
        ani = build_dashboard()
    except KeyboardInterrupt:
        pass
    finally:
        serial_running = False
        gesture_running = False
        close_csv_recording()
        print("\n[INFO] Monitor cerrado.")

if __name__ == '__main__':
    main()
