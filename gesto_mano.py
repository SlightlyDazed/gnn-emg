#!/usr/bin/env python3
"""
================================================================
  SICHIRAY GESTURE MONITOR — MediaPipe Tasks API 0.10+ | Python
  Reconocimiento de gestos y movimientos de la mano en tiempo real

  Gestos detectados:
    ABIERTA, PUNIO, INDICE, PAZ, OK, PULGAR_UP, PULGAR_DN,
    TRES, PINZA, TELEFONO

  Movimientos:  Swipe ←→↑↓ | Velocidad | Rastro animado

  Integración EMG:
    shared_state (dict) actualizado cada frame — thread-safe
================================================================
"""

import os
# Forzar Qt a usar XCB (XWayland) — el plugin nativo 'wayland' de OpenCV no funciona
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_LOGGING_RULES'] = '*=false'  # Suprimir TODO el spam de Qt (fonts, etc)
os.environ['OPENCV_VIDEOIO_PRIORITY_LIST'] = 'V4L2'  # Evitar que GStreamer se cuelgue en Wayland

import cv2
import mediapipe as mp
import numpy as np
import time, math, collections, threading, argparse

# ── MediaPipe Tasks API (0.10+) ───────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

# ── Modelo .task (mismo directorio que el script) ─────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'hand_landmarker.task')

# ── Paleta BGR ────────────────────────────────────────────────
C_BG      = (23,  22,  20)
C_PANEL   = (28,  27,  25)
C_ACCENT  = (163, 152,  79)
C_GREEN   = ( 69, 170,  69)
C_RED     = ( 72, 110, 253)
C_YELLOW  = ( 81, 196, 253)
C_TEAL    = (163, 152,  79)
C_WHITE   = (205, 204, 202)
C_GRAY    = (120, 120, 118)
C_LMRK   = (200, 180, 100)
C_CONN    = ( 80, 130, 200)
C_DIVIDER = ( 57,  56,  54)

GESTURE_COLORS = {
    'ABIERTA':   C_GREEN,  'PUNIO':     C_RED,
    'INDICE':    C_YELLOW, 'PAZ':       C_ACCENT,
    'OK':        C_GREEN,  'PULGAR_UP': C_GREEN,
    'PULGAR_DN': C_RED,    'TRES':      C_YELLOW,
    'PINZA':     C_ACCENT, 'TELEFONO':  C_TEAL,
    'OTRO':      C_GRAY,
}

# ── Config ────────────────────────────────────────────────────
CAM_INDEX    = 0
CAM_W        = 1280
CAM_H        = 720
DISPLAY_W    = 1600
PANEL_W      = 300
MAX_HANDS    = 2
PINCH_THRESH = 0.07
SWIPE_THRESH = 80
SWIPE_FRAMES = 12
TRAIL_LEN    = 40
FONT         = cv2.FONT_HERSHEY_SIMPLEX

# Índices de landmarks (igual en nueva API)
WRIST=0; THUMB_TIP=4; INDEX_TIP=8; INDEX_PIP=6; INDEX_MCP=5
MIDDLE_TIP=12; MIDDLE_PIP=10; MIDDLE_MCP=9
RING_TIP=16; RING_PIP=14; RING_MCP=13
PINKY_TIP=20; PINKY_PIP=18; PINKY_MCP=17
THUMB_MCP=2

# ── Estado compartido ─────────────────────────────────────────
shared_state = {
    'gestures': [], 'landmarks': [], 'wrist_pos': [],
    'wrist_vel': [], 'swipe': None, 'pinch_dist': [],
    'hand_labels': [], 'fps': 0.0, 'timestamp': 0.0, 'active': False,
}
state_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────
# RECONOCIMIENTO DE GESTOS
# ─────────────────────────────────────────────────────────────

def lm_to_array(hand_landmarks, img_w, img_h):
    """Convierte lista de NormalizedLandmark a numpy (21,3) en píxeles."""
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

# ─────────────────────────────────────────────────────────────
# TRACKER DE MOVIMIENTO
# ─────────────────────────────────────────────────────────────

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

trackers = [HandTracker(), HandTracker()]

# ─────────────────────────────────────────────────────────────
# RENDERING
# ─────────────────────────────────────────────────────────────

HAND_CONNECTIONS = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]

def draw_landmarks_custom(frame, pts, color):
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, (int(pts[s][0]),int(pts[s][1])),
                        (int(pts[e][0]),int(pts[e][1])), C_CONN, 1, cv2.LINE_AA)
    for idx, p in enumerate(pts):
        x, y = int(p[0]), int(p[1])
        is_tip = idx in [4,8,12,16,20]
        c = color if is_tip else C_LMRK
        r = 6 if is_tip else 4
        cv2.circle(frame, (x,y), r, c, -1, cv2.LINE_AA)
        cv2.circle(frame, (x,y), r, C_BG, 1, cv2.LINE_AA)

def draw_trail(frame, trail, color):
    pts = list(trail)
    n = len(pts)
    for i in range(1, n):
        a = i / n
        c = tuple(int(ch*a) for ch in color)
        cv2.line(frame, (int(pts[i-1][0]),int(pts[i-1][1])),
                        (int(pts[i][0]),  int(pts[i][1])), c, max(1,int(3*a)))

def draw_swipe(frame, swipe, cx, cy):
    if not swipe:
        return
    labels = {'LEFT':'<','RIGHT':'>','UP':'^','DOWN':'v'}
    offsets = {'LEFT':(-70,0),'RIGHT':(70,0),'UP':(0,-70),'DOWN':(0,70)}
    lbl = labels.get(swipe,'')
    dx, dy = offsets.get(swipe, (0,0))
    tx = max(40, min(cx+dx, frame.shape[1]-40))
    ty = max(40, min(cy+dy, frame.shape[0]-40))
    ov = frame.copy()
    cv2.circle(ov, (tx,ty), 40, C_ACCENT, -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, lbl, (tx-14,ty+12), FONT, 1.4, C_BG, 3, cv2.LINE_AA)

def draw_finger_bars(panel, y0, states):
    names = ['Pulgar','Indice','Medio','Anular','Menique']
    for i,(n,s) in enumerate(zip(names, states)):
        y = y0 + i*22
        cv2.rectangle(panel, (20,y), (100,y+16), C_GREEN if s else C_RED, -1)
        cv2.putText(panel, n, (105,y+12), FONT, 0.36, C_WHITE, 1, cv2.LINE_AA)

def build_panel(h, hand_list):
    panel = np.full((h, PANEL_W, 3), C_PANEL, dtype=np.uint8)
    cv2.putText(panel, 'SICHIRAY GESTURE', (10,26), FONT, 0.5, C_ACCENT, 1, cv2.LINE_AA)
    cv2.line(panel, (10,34), (PANEL_W-10,34), C_ACCENT, 1)
    y = 50
    if not hand_list:
        cv2.putText(panel, 'Sin manos', (10,y), FONT, 0.4, C_GRAY, 1, cv2.LINE_AA)
        return panel
    for hr in hand_list:
        gesture = hr['gesture']
        gc = GESTURE_COLORS.get(gesture, C_GRAY)
        cv2.putText(panel, f"Mano: {hr['label']}", (10,y), FONT, 0.4, C_WHITE, 1, cv2.LINE_AA)
        y += 18
        cv2.rectangle(panel, (10,y), (PANEL_W-10, y+22), gc, -1)
        cv2.putText(panel, f"{gesture} {hr['conf']:.0%}", (14,y+16),
                    FONT, 0.48, C_BG, 2, cv2.LINE_AA)
        y += 30
        draw_finger_bars(panel, y, hr['fingers'])
        y += 5*22 + 4
        # Pinza bar
        pw = int((1 - min(hr['pinch'],1)) * (PANEL_W-30))
        cv2.rectangle(panel, (10,y), (PANEL_W-20,y+10), C_BG, -1)
        cv2.rectangle(panel, (10,y), (10+pw,y+10), C_ACCENT, -1)
        cv2.putText(panel, f"Pinza:{hr['pinch']:.2f}", (10,y+22),
                    FONT, 0.34, C_GRAY, 1, cv2.LINE_AA)
        y += 30
        cv2.putText(panel, f"Vel:{hr['speed']:.0f}px/s", (10,y),
                    FONT, 0.37, C_YELLOW, 1, cv2.LINE_AA)
        y += 18
        if hr['swipe']:
            sv={'LEFT':'< IZQ','RIGHT':'> DER','UP':'^ ARR','DOWN':'v ABA'}
            cv2.putText(panel, f"Swipe:{sv.get(hr['swipe'],'')}",
                        (10,y), FONT, 0.4, C_ACCENT, 1, cv2.LINE_AA)
        y += 20
        cv2.line(panel, (10,y), (PANEL_W-10,y), C_DIVIDER, 1)
        y += 10
    return panel

# ─────────────────────────────────────────────────────────────
# BUCLE PRINCIPAL
# ─────────────────────────────────────────────────────────────

def run(cam_index=CAM_INDEX, mirror=True):
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Modelo no encontrado: {MODEL_PATH}")
        print("Ejecuta primero:")
        print("  python3 -c \"import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task','hand_landmarker.task')\"")
        return

    # Configurar HandLandmarker (nueva Tasks API)
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    # FORZAR MJPG: La cámara (USB2.0 HD) se cuelga si se pide 720p@30fps en YUYV crudo.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir cámara {cam_index}")
        return

    print(f"[OK] Cámara {cam_index} — {int(cap.get(3))}x{int(cap.get(4))}")
    print("[INFO]  Q / ESC = salir  |  S = captura")

    cv2.namedWindow('Sichiray — Gesture Monitor  (Q=salir)', cv2.WINDOW_NORMAL)
    fps_buf = collections.deque(maxlen=30)
    shot_n  = 0
    frame_ms = 0

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            continue
        if mirror:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # MediaPipe necesita imagen RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ms += 33   # timestamp incremental en ms
        result = landmarker.detect_for_video(mp_img, frame_ms)

        hand_list = []
        gestures_out, wrist_out, vel_out, lm_out, pd_out, lbl_out = [],[],[],[],[],[]
        global_swipe = None

        if result.hand_landmarks:
            for hi, (hand_lm, handedness) in enumerate(
                    zip(result.hand_landmarks, result.handedness)):
                label = handedness[0].display_name  # 'Left' / 'Right'
                pts   = lm_to_array(hand_lm, w, h)
                gesture, conf = classify_gesture(pts, label)

                tk = trackers[hi % 2]
                wrist_px = (int(pts[WRIST][0]), int(pts[WRIST][1]))
                vx, vy, spd = tk.update(wrist_px)
                gesture = tk.smooth(gesture)
                swipe   = tk.swipe
                if swipe:
                    global_swipe = swipe

                pd  = pinch_dist(pts)
                fst = get_finger_states(pts, label)
                gc  = GESTURE_COLORS.get(gesture, C_GRAY)

                draw_landmarks_custom(frame, pts, gc)
                draw_trail(frame, tk.trail, gc)
                if swipe:
                    draw_swipe(frame, swipe, wrist_px[0], wrist_px[1])

                lx = max(0, wrist_px[0]-60)
                ly = max(30, wrist_px[1]-30)
                cv2.putText(frame, gesture, (lx, ly),   FONT, 0.7, gc,     2, cv2.LINE_AA)
                cv2.putText(frame, label,   (lx, ly+20), FONT, 0.4, C_GRAY, 1, cv2.LINE_AA)

                hand_list.append({'label':label,'gesture':gesture,'conf':conf,
                                  'fingers':fst,'pinch':pd,'speed':spd,'swipe':swipe})
                gestures_out.append(gesture); wrist_out.append(wrist_px)
                vel_out.append((vx,vy,spd));  lm_out.append(pts.tolist())
                pd_out.append(pd);            lbl_out.append(label)

        # FPS
        t1 = time.time()
        fps_buf.append(1.0/max(t1-t0,1e-6))
        fps = float(np.mean(fps_buf))
        cv2.putText(frame, f'FPS:{fps:.1f}', (w-110,25), FONT, 0.5, C_ACCENT, 1, cv2.LINE_AA)

        # Composición
        panel = build_panel(h, hand_list)
        fw    = DISPLAY_W - PANEL_W
        scale = fw / w
        disp  = cv2.resize(frame, (fw, int(h*scale)))
        pan   = cv2.resize(panel, (PANEL_W, disp.shape[0]))
        out   = np.hstack([disp, pan])
        cv2.imshow('Sichiray — Gesture Monitor  (Q=salir)', out)

        with state_lock:
            shared_state.update({
                'gestures':gestures_out,'landmarks':lm_out,
                'wrist_pos':wrist_out,'wrist_vel':vel_out,
                'swipe':global_swipe,'pinch_dist':pd_out,
                'hand_labels':lbl_out,'fps':fps,
                'timestamp':t1,'active':len(gestures_out)>0,
            })

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            fname = f'captura_{shot_n:03d}.png'
            cv2.imwrite(fname, out)
            print(f'[OK] Guardado: {fname}')
            shot_n += 1

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print('[INFO] Monitor cerrado.')


def main():
    global MAX_HANDS
    p = argparse.ArgumentParser(description='Sichiray Gesture Monitor')
    p.add_argument('--cam',       '-c', type=int, default=CAM_INDEX)
    p.add_argument('--no-mirror', action='store_true')
    p.add_argument('--max-hands', type=int, default=MAX_HANDS)
    args = p.parse_args()
    MAX_HANDS = args.max_hands
    run(cam_index=args.cam, mirror=not args.no_mirror)


if __name__ == '__main__':
    main()
