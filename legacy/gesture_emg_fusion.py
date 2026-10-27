#!/usr/bin/env python3
"""
SICHIRAY GESTURE + EMG FUSION MONITOR
Combina reconocimiento de gestos (MediaPipe) con señal EMG (ESP32)
Permite ver qué gesto genera qué señal EMG y grabar sesiones
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['OPENCV_VIDEOIO_PRIORITY_LIST'] = 'V4L2'

import cv2
import mediapipe as mp
import numpy as np
import time, math, collections, threading, argparse, csv
from datetime import datetime
from pathlib import Path

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ── Colors BGR ──
C_BG      = (23,  22,  20)
C_PANEL   = (28,  27,  25)
C_ACCENT  = (163, 152,  79)
C_GREEN   = ( 69, 170,  69)
C_RED     = ( 72, 110, 253)
C_YELLOW  = ( 81, 196, 253)
C_TEAL    = (163, 152,  79)
C_WHITE   = (205, 204, 202)
C_GRAY    = (120, 120, 118)
C_LMRK    = (200, 180, 100)
C_CONN    = ( 80, 130, 200)
C_DIVIDER = ( 57,  56,  54)
C_EMG_TEAL= ( 79, 152, 163)

GESTURE_COLORS = {
    'ABIERTA':   C_GREEN,  'PUNIO':     C_RED,
    'INDICE':    C_YELLOW, 'PAZ':       C_ACCENT,
    'OK':        C_GREEN,  'PULGAR_UP': C_GREEN,
    'PULGAR_DN': C_RED,    'TRES':      C_YELLOW,
    'PINZA':     C_ACCENT, 'TELEFONO':  C_TEAL,
    'OTRO':      C_GRAY,
}

# ── Config ──
CAM_INDEX    = 0
CAM_W        = 1280
CAM_H        = 720
DISPLAY_W    = 1600
PANEL_W      = 400
MAX_HANDS    = 2
PINCH_THRESH = 0.07
TRAIL_LEN    = 30
FONT         = cv2.FONT_HERSHEY_SIMPLEX

WRIST=0; THUMB_TIP=4; INDEX_TIP=8; INDEX_PIP=6; INDEX_MCP=5
MIDDLE_TIP=12; MIDDLE_PIP=10; MIDDLE_MCP=9
RING_TIP=16; RING_PIP=14; RING_MCP=13
PINKY_TIP=20; PINKY_PIP=18; PINKY_MCP=17
THUMB_MCP=2

# ── EMG Config ──
FS          = 200
BAUD        = 115200
K_THRESHOLD = 3.5

# ── EMG shared state ──
emg_lock = threading.Lock()
emg_state = {
    'raw': 2048, 'filt': 0.0, 'rms': 0.0, 'mav': 0.0,
    'wl': 0.0, 'zc': 0, 'pulse_count': 0,
    'muscle_active': False,
    'baseline_std': 50.0, 'snr_db': 0.0,
    'connected': False, 'running': False,
}

# ── Recording state ──
recording = False
record_dir = None
csv_writer = None
record_file = None
record_start = 0.0

# ─────────────────────────────────────────────────────────
# GESTURE RECOGNITION
# ─────────────────────────────────────────────────────────
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
    i = finger_extended(pts, INDEX_TIP, INDEX_PIP)
    m = finger_extended(pts, MIDDLE_TIP, MIDDLE_PIP)
    r = finger_extended(pts, RING_TIP, RING_PIP)
    p = finger_extended(pts, PINKY_TIP, PINKY_PIP)
    return t, i, m, r, p

def pinch_dist(pts):
    hs = dist2d(pts[WRIST], pts[MIDDLE_MCP])
    return dist2d(pts[THUMB_TIP], pts[INDEX_TIP]) / max(hs, 1e-6)

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

# ─────────────────────────────────────────────────────────
# HAND TRACKER
# ─────────────────────────────────────────────────────────
class HandTracker:
    def __init__(self):
        self.trail = collections.deque(maxlen=TRAIL_LEN)
        self.gest_hist = collections.deque(maxlen=10)
        self.last_pos = None
    def update(self, pos):
        self.trail.append(pos)
        self.last_pos = pos
    def smooth(self, g):
        self.gest_hist.append(g)
        return collections.Counter(self.gest_hist).most_common(1)[0][0]

trackers = [HandTracker(), HandTracker()]

# ─────────────────────────────────────────────────────────
# EMG PROCESSING
# ─────────────────────────────────────────────────────────
def parse_emg_line(line):
    line = line.strip()
    if not line: return None, None
    try:
        if 'RAW:' in line:
            parts = dict(p.split(':') for p in line.split(',') if ':' in p)
            return int(parts.get('RAW', 2048)), int(parts.get('SUAVIZADO', parts.get('RAW', 2048)))
        elif ',' in line:
            vals = line.split(',')
            return int(vals[0]), int(vals[1]) if len(vals) > 1 else int(vals[0])
        else:
            v = int(line)
            return v, v
    except:
        return None, None

_emg_history = collections.deque(maxlen=200)

def process_emg_sample(raw, smo=None):
    if smo is None: smo = raw
    with emg_lock:
        s = emg_state
        s['raw'] = raw
        filt = float(raw - 2048) * 0.3
        s['filt'] = filt
        _emg_history.append(filt)
        recent = list(_emg_history)
        n = len(recent)
        if n >= 10:
            arr = np.array(recent[-40:], dtype=float)
            s['rms'] = float(np.sqrt(np.mean(arr**2)))
            s['mav'] = float(np.mean(np.abs(arr)))
            s['wl'] = float(np.sum(np.abs(np.diff(arr)))) if len(arr) > 1 else 0.0
            s['zc'] = int(np.sum(np.diff(np.sign(arr)) != 0)) if len(arr) > 1 else 0
        thr = K_THRESHOLD * max(s['baseline_std'], 10.0)
        s['muscle_active'] = s['rms'] > thr

def emg_serial_thread(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=1.0)
        time.sleep(1.5)
        ser.reset_input_buffer()
        with emg_lock:
            emg_state['connected'] = True
            emg_state['running'] = True
        print(f"[EMG] Conectado a {port} @ {baud} baud")
        while True:
            with emg_lock:
                if not emg_state['running']: break
            raw, smo = parse_emg_line(ser.readline().decode('utf-8', errors='ignore'))
            if raw is not None:
                process_emg_sample(raw, smo)
    except Exception as e:
        print(f"[EMG] Error: {e}")
        with emg_lock:
            emg_state['connected'] = False
            emg_state['running'] = False
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

def emg_demo_thread():
    with emg_lock:
        emg_state['running'] = True
    print("[EMG] Modo demo — señal sintética")
    t = 0
    while True:
        with emg_lock:
            if not emg_state['running']: break
        t += 1
        noise = np.random.normal(0, 30)
        burst = 0.0
        if (t % 400) < 80:
            burst = np.random.normal(0, 250) * np.sin(np.pi * (t % 400) / 80)
        raw_v = int(2048 + noise + burst)
        raw_v = max(0, min(4095, raw_v))
        process_emg_sample(raw_v)
        time.sleep(1.0/FS)

def start_emg(port=None, demo=False):
    if demo or not HAS_SERIAL:
        t = threading.Thread(target=emg_demo_thread, daemon=True)
        t.start()
        return t
    if port is None:
        try:
            ports = serial.tools.list_ports.comports()
            for p in ports:
                d = (p.description or '').lower()
                if any(k in d for k in ['cp210','ch340','ft232','uart','serial','esp32']):
                    port = p.device; break
            if port is None:
                for p in ports:
                    if 'ttyS' not in p.device:
                        port = p.device; break
        except:
            pass
    if port:
        t = threading.Thread(target=emg_serial_thread, args=(port, BAUD), daemon=True)
        t.start()
        return t
    print("[EMG] No se encontró ESP32, iniciando demo")
    return start_emg(demo=True)

# ─────────────────────────────────────────────────────────
# RECORDING
# ─────────────────────────────────────────────────────────
def toggle_recording(gesture, emg_data):
    global recording, record_dir, csv_writer, record_file, record_start
    if not recording:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        record_dir = Path.home() / 'sichiray_sessions' / timestamp
        record_dir.mkdir(parents=True, exist_ok=True)
        record_file = open(record_dir / 'session.csv', 'w', newline='')
        csv_writer = csv.writer(record_file)
        csv_writer.writerow(['timestamp', 'gesture', 'hand_label', 'conf',
                             'emg_raw', 'emg_filt', 'emg_rms', 'emg_mav',
                             'emg_wl', 'emg_zc', 'muscle_active', 'pulse_count'])
        recording = True
        record_start = time.time()
        print(f"[REC] Grabando en {record_dir}")
    else:
        recording = False
        if record_file:
            record_file.close()
            record_file = None
        csv_writer = None
        print(f"[REC] Grabación guardada en {record_dir}")

def record_sample(gesture, hand_label, conf, emg_data):
    global csv_writer
    if csv_writer:
        csv_writer.writerow([
            time.time() - record_start,
            gesture, hand_label, f'{conf:.2f}',
            emg_data.get('raw', 0), emg_data.get('filt', 0),
            f"{emg_data.get('rms', 0):.1f}", f"{emg_data.get('mav', 0):.1f}",
            f"{emg_data.get('wl', 0):.1f}", emg_data.get('zc', 0),
            1 if emg_data.get('muscle_active') else 0,
            emg_data.get('pulse_count', 0)
        ])

# ─────────────────────────────────────────────────────────
# RENDERING
# ─────────────────────────────────────────────────────────
HAND_CONNECTIONS = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]

def draw_landmarks(frame, pts, color):
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, (int(pts[s][0]), int(pts[s][1])),
                        (int(pts[e][0]), int(pts[e][1])), C_CONN, 2, cv2.LINE_AA)
    for idx, p in enumerate(pts):
        x, y = int(p[0]), int(p[1])
        r = 7 if idx in [4,8,12,16,20] else 5
        cv2.circle(frame, (x, y), r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r, C_BG, 1, cv2.LINE_AA)

def draw_trail(frame, trail, color):
    pts = list(trail)
    for i in range(1, len(pts)):
        a = i / len(pts)
        c = tuple(int(ch*a) for ch in color)
        cv2.line(frame, (int(pts[i-1][0]), int(pts[i-1][1])),
                        (int(pts[i][0]), int(pts[i][1])), c, max(1, int(3*a)))

def build_emg_panel(h, emg):
    panel = np.full((h, PANEL_W, 3), C_PANEL, dtype=np.uint8)
    y = 20

    # Title
    cv2.putText(panel, 'EMG MONITOR', (10, y), FONT, 0.55, C_ACCENT, 1, cv2.LINE_AA)
    y += 25

    # Connection status
    status = 'CONECTADO' if emg['connected'] else 'DEMO'
    status_c = C_GREEN if emg['connected'] else C_YELLOW
    cv2.putText(panel, status, (10, y), FONT, 0.4, status_c, 1, cv2.LINE_AA)
    y += 25

    # Muscle activity indicator
    act_color = C_GREEN if emg['muscle_active'] else C_GRAY
    act_label = 'ACTIVO' if emg['muscle_active'] else 'REPOSO'
    cv2.rectangle(panel, (10, y-10), (PANEL_W-20, y+10), act_color, -1)
    cv2.putText(panel, act_label, (20, y+4), FONT, 0.5, C_BG, 2, cv2.LINE_AA)
    y += 25

    # Divider
    cv2.line(panel, (10, y), (PANEL_W-10, y), C_DIVIDER, 1)
    y += 15

    # Metrics
    metrics = [
        ('ADC Raw',   f"{emg['raw']:4d}",      C_WHITE),
        ('Filtrada',  f"{emg['filt']:+.1f}",    C_EMG_TEAL),
        ('RMS',       f"{emg['rms']:.1f}",      C_GREEN),
        ('MAV',       f"{emg['mav']:.1f}",      C_ACCENT),
        ('WL',        f"{emg['wl']:.0f}",       C_YELLOW),
        ('ZC',        f"{emg['zc']:3d}",        C_TEAL),
    ]
    for label, val, color in metrics:
        cv2.putText(panel, label, (15, y), FONT, 0.4, C_GRAY, 1, cv2.LINE_AA)
        cv2.putText(panel, val,  (PANEL_W//2, y), FONT, 0.45, color, 1, cv2.LINE_AA)
        y += 22

    # Activity bar
    y += 5
    rms_norm = min(emg['rms'] / max(emg['baseline_std'] * K_THRESHOLD * 2, 1), 1.0)
    cv2.putText(panel, 'Activacion', (15, y), FONT, 0.4, C_GRAY, 1, cv2.LINE_AA)
    y += 12
    bar_w = PANEL_W - 40
    cv2.rectangle(panel, (15, y), (15+bar_w, y+18), C_BG, -1)
    fill_w = int(bar_w * rms_norm)
    if fill_w > 0:
        bar_c = C_GREEN if emg['muscle_active'] else C_EMG_TEAL
        cv2.rectangle(panel, (15, y), (15+fill_w, y+18), bar_c, -1)
    cv2.putText(panel, f'{rms_norm*100:.0f}%', (bar_w-40, y+13), FONT, 0.4, C_WHITE, 1, cv2.LINE_AA)
    y += 35

    # Pulse count
    cv2.putText(panel, 'Pulsos', (15, y), FONT, 0.4, C_GRAY, 1, cv2.LINE_AA)
    cv2.putText(panel, str(emg['pulse_count']), (PANEL_W//2, y), FONT, 0.5, C_GREEN, 1, cv2.LINE_AA)
    y += 25

    # SNR
    cv2.putText(panel, 'SNR', (15, y), FONT, 0.4, C_GRAY, 1, cv2.LINE_AA)
    cv2.putText(panel, f"{emg['snr_db']:.1f} dB", (PANEL_W//2, y), FONT, 0.45, C_ACCENT, 1, cv2.LINE_AA)
    y += 25

    # Divider
    cv2.line(panel, (10, y), (PANEL_W-10, y), C_DIVIDER, 1)
    y += 20

    # Gesture display (large)
    return panel

def build_gesture_panel(panel, h, hand_data, emg_active):
    # Overlay gesture info on the lower part of the panel
    y = h - 200
    cv2.putText(panel, 'GESTO DETECTADO', (10, y), FONT, 0.45, C_ACCENT, 1, cv2.LINE_AA)
    y += 10

    if not hand_data:
        cv2.putText(panel, 'Sin mano', (20, y+40), FONT, 0.7, C_GRAY, 1, cv2.LINE_AA)
        return panel

    hr = hand_data[0]
    gesture = hr['gesture']
    gc = GESTURE_COLORS.get(gesture, C_GRAY)

    # Large gesture name
    cv2.putText(panel, gesture, (15, y+35), FONT, 0.9, gc, 2, cv2.LINE_AA)
    y += 50
    cv2.putText(panel, f"{hr['label']}  {hr['conf']:.0%}", (15, y), FONT, 0.4, C_GRAY, 1, cv2.LINE_AA)

    # EMG correlation indicator
    y += 25
    if emg_active:
        cv2.putText(panel, '<< EMG ACTIVA >>', (15, y), FONT, 0.5, C_GREEN, 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, 'EMG en reposo', (15, y), FONT, 0.4, C_GRAY, 1, cv2.LINE_AA)

    return panel

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    global recording, record_dir, csv_writer, record_file, record_start

    parser = argparse.ArgumentParser(description='Sichiray Gesture + EMG Fusion Monitor')
    parser.add_argument('--cam', '-c', type=int, default=0)
    parser.add_argument('--port', '-p', help='Puerto serial EMG')
    parser.add_argument('--demo', action='store_true', help='Modo demo EMG')
    parser.add_argument('--no-mirror', action='store_true')
    args = parser.parse_args()

    # Start EMG thread
    emg_thread = start_emg(args.port, args.demo)
    time.sleep(0.5)

    # Load MediaPipe model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')
    if not os.path.exists(model_path):
        print(f"[ERROR] Modelo no encontrado: {model_path}")
        return

    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    # Camera
    cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir cámara")
        return

    mirror = not args.no_mirror
    w, h = int(cap.get(3)), int(cap.get(4))
    print(f"[OK] Camara {w}x{h}")

    cv2.namedWindow('Sichiray — Gesture + EMG Fusion', cv2.WINDOW_NORMAL)

    fps_buf = collections.deque(maxlen=30)
    frame_ms = 0
    prev_gesture = ''
    shot_n = 0

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret: continue
        if mirror:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ms += 33
        result = landmarker.detect_for_video(mp_img, frame_ms)

        hand_data = []
        if result.hand_landmarks:
            for hi, (hand_lm, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
                label = handedness[0].display_name
                pts = lm_to_array(hand_lm, w, h)
                gesture, conf = classify_gesture(pts, label)
                tk = trackers[hi % 2]
                wrist = (int(pts[WRIST][0]), int(pts[WRIST][1]))
                tk.update(wrist)
                gesture = tk.smooth(gesture)
                gc = GESTURE_COLORS.get(gesture, C_GRAY)
                draw_landmarks(frame, pts, gc)
                draw_trail(frame, tk.trail, gc)

                # Gesture label on camera
                lx, ly = max(10, wrist[0]-60), max(30, wrist[1]-30)
                cv2.putText(frame, gesture, (lx, ly), FONT, 0.7, gc, 2, cv2.LINE_AA)
                cv2.putText(frame, label, (lx, ly+20), FONT, 0.4, C_GRAY, 1, cv2.LINE_AA)

                hand_data.append({'gesture': gesture, 'label': label, 'conf': conf})

        # FPS
        t1 = time.time()
        fps_buf.append(1.0/max(t1-t0, 1e-6))
        fps = float(np.mean(fps_buf))
        cv2.putText(frame, f'{fps:.0f} FPS', (w-100, 25), FONT, 0.5, C_ACCENT, 1, cv2.LINE_AA)

        # EMG read
        with emg_lock:
            emg = dict(emg_state)

        # Build panel
        fw = DISPLAY_W - PANEL_W
        scale = fw / w
        disp_h = int(h * scale)
        disp = cv2.resize(frame, (fw, disp_h))

        panel = build_emg_panel(disp_h, emg)
        build_gesture_panel(panel, disp_h, hand_data, emg['muscle_active'])

        # Combine
        out = np.hstack([disp, panel])

        # Recording
        if recording:
            elapsed = time.time() - record_start
            mins, secs = int(elapsed//60), int(elapsed%60)
            rec_text = f"REC {mins:02d}:{secs:02d}"
            cv2.rectangle(out, (5, 5), (140, 30), C_RED, -1)
            cv2.putText(out, rec_text, (12, 22), FONT, 0.45, C_WHITE, 1, cv2.LINE_AA)

            gesture_name = hand_data[0]['gesture'] if hand_data else 'SIN_MANO'
            hand_label = hand_data[0]['label'] if hand_data else ''
            conf_val = hand_data[0]['conf'] if hand_data else 0
            record_sample(gesture_name, hand_label, conf_val, emg)

            if gesture_name != prev_gesture:
                prev_gesture = gesture_name
                ss_path = record_dir / f'gesture_{gesture_name}_{shot_n:03d}.png'
                cv2.imwrite(str(ss_path), out)
                shot_n += 1

        cv2.imshow('Sichiray — Gesture + EMG Fusion', out)

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            toggle_recording(hand_data[0]['gesture'] if hand_data else 'SIN_MANO', emg)
        elif key == ord('s'):
            fname = f'captura_{shot_n:03d}.png'
            cv2.imwrite(fname, out)
            print(f'[OK] Captura: {fname}')
            shot_n += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    if recording:
        toggle_recording('', {})
    with emg_lock:
        emg_state['running'] = False
    print('[INFO] Monitor cerrado.')

if __name__ == '__main__':
    main()
