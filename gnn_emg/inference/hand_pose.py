"""
hand_pose.py
────────────────────────────────────────────────────────────────────────────
Mano virtual sintética (21 landmarks, misma topología que MediaPipe Hands)
cuya pose se define por gesto (no por cámara). Se usa para animar en la GUI
la mano según el gesto que predice el modelo GNN-EMG.

Índices de landmark (idénticos a MediaPipe):
    0  WRIST
    1-4   THUMB   (CMC, MCP, IP, TIP)
    5-8   INDEX   (MCP, PIP, DIP, TIP)
    9-12  MIDDLE  (MCP, PIP, DIP, TIP)
    13-16 RING    (MCP, PIP, DIP, TIP)
    17-20 PINKY   (MCP, PIP, DIP, TIP)
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import math
import numpy as np

# ── Conexiones del esqueleto (idénticas a mediapipe.HAND_CONNECTIONS) ──────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),          # índice
    (5, 9), (9, 10), (10, 11), (11, 12),     # medio
    (9, 13), (13, 14), (14, 15), (15, 16),   # anular
    (13, 17), (17, 18), (18, 19), (19, 20),  # meñique
    (0, 17),                                 # base de la palma
]

FINGER_TIP_IDX = {'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}

# ── Punto fijo de la muñeca (no cambia entre poses) ─────────────────────────
WRIST = (100.0, 15.0)

# ── Cadenas de referencia por dedo: [base, joint2, joint3, tip] ─────────────
# 'open'  = dedo extendido      'fist' = dedo cerrado (puño)
# 'up'/'down'/'pinch' = variantes especiales solo para el pulgar/índice
_THUMB_OPEN  = [(72, 35), (58, 52), (46, 72), (36, 90)]
_THUMB_FIST  = [(72, 35), (65, 48), (76, 58), (92, 66)]
_THUMB_UP    = [(72, 35), (70, 58), (68, 90), (66, 125)]
_THUMB_DOWN  = [(72, 35), (68, 18), (64, -8), (60, -35)]
_THUMB_PINCH = [(72, 35), (66, 58), (78, 95), (92, 132)]

_INDEX_OPEN  = [(78, 85), (75, 125), (73, 155), (71, 178)]
_INDEX_FIST  = [(78, 85), (68, 98), (70, 84), (80, 74)]
_INDEX_PINCH = [(78, 85), (80, 110), (86, 128), (92, 136)]

_MIDDLE_OPEN = [(100, 92), (100, 135), (100, 165), (100, 188)]
_MIDDLE_FIST = [(100, 92), (90, 106), (94, 90), (103, 78)]

_RING_OPEN   = [(122, 87), (124, 128), (125, 157), (126, 178)]
_RING_FIST   = [(122, 87), (114, 102), (117, 86), (125, 76)]

_PINKY_OPEN  = [(142, 75), (147, 108), (150, 131), (152, 150)]
_PINKY_FIST  = [(142, 75), (134, 88), (136, 74), (143, 66)]

FINGER_CHAINS = {
    'thumb':  {'open': _THUMB_OPEN,  'fist': _THUMB_FIST,
               'up': _THUMB_UP, 'down': _THUMB_DOWN, 'pinch': _THUMB_PINCH},
    'index':  {'open': _INDEX_OPEN,  'fist': _INDEX_FIST, 'pinch': _INDEX_PINCH},
    'middle': {'open': _MIDDLE_OPEN, 'fist': _MIDDLE_FIST},
    'ring':   {'open': _RING_OPEN,   'fist': _RING_FIST},
    'pinky':  {'open': _PINKY_OPEN,  'fist': _PINKY_FIST},
}

# 'half' = pose relajada/neutral → punto medio entre abierta y cerrada
for _chains in FINGER_CHAINS.values():
    _o = np.array(_chains['open'], dtype=float)
    _f = np.array(_chains['fist'], dtype=float)
    _chains['half'] = [tuple(p) for p in (_o + _f) / 2.0]

# ── Pose objetivo por gesto: qué cadena usa cada dedo ───────────────────────
GESTURE_POSES = {
    'ABIERTA':   dict(thumb='open',  index='open',  middle='open',  ring='open',  pinky='open'),
    'PUNIO':     dict(thumb='fist',  index='fist',  middle='fist',  ring='fist',  pinky='fist'),
    'INDICE':    dict(thumb='fist',  index='open',  middle='fist',  ring='fist',  pinky='fist'),
    'PAZ':       dict(thumb='fist',  index='open',  middle='open',  ring='fist',  pinky='fist'),
    'TRES':      dict(thumb='fist',  index='open',  middle='open',  ring='open',  pinky='fist'),
    'OK':        dict(thumb='fist',  index='fist',  middle='open',  ring='open',  pinky='open'),
    'PULGAR_UP': dict(thumb='up',    index='fist',  middle='fist',  ring='fist',  pinky='fist'),
    'PULGAR_DN': dict(thumb='down',  index='fist',  middle='fist',  ring='fist',  pinky='fist'),
    'PINZA':     dict(thumb='pinch', index='pinch', middle='fist',  ring='fist',  pinky='fist'),
    'TELEFONO':  dict(thumb='up',    index='fist',  middle='fist',  ring='fist',  pinky='open'),
    'OTRO':      dict(thumb='half',  index='half',  middle='half',  ring='half',  pinky='half'),
}

GESTURE_ORDER = ['ABIERTA', 'PUNIO', 'INDICE', 'PAZ', 'TRES', 'OK',
                  'PULGAR_UP', 'PULGAR_DN', 'PINZA', 'TELEFONO', 'OTRO']

GESTURE_COLORS = {
    'ABIERTA': '#45aa6d', 'PUNIO': '#fd6e6e', 'INDICE': '#fdc451',
    'PAZ': '#a86fdf', 'OK': '#45aa6d', 'PULGAR_UP': '#45aa6d',
    'PULGAR_DN': '#fd6e6e', 'TRES': '#fdc451', 'PINZA': '#a86fdf',
    'TELEFONO': '#4f98a3', 'OTRO': '#7a7876',
}


def get_pose_targets(gesture: str) -> np.ndarray:
    """Devuelve (20,2): posiciones landmark 1..20 (sin la muñeca) para el gesto dado."""
    spec = GESTURE_POSES.get(gesture, GESTURE_POSES['OTRO'])
    pts = []
    for finger in ('thumb', 'index', 'middle', 'ring', 'pinky'):
        key = spec[finger]
        pts.extend(FINGER_CHAINS[finger][key])
    return np.array(pts, dtype=float)


class HandAnimator:
    """Interpola suavemente la mano virtual (20 puntos + muñeca fija) hacia la
    pose del gesto detectado, para que el movimiento se vea natural y no
    'salte' entre poses de un frame a otro."""

    def __init__(self, tau: float = 0.15):
        self.tau = tau
        self.current = get_pose_targets('OTRO')
        self.target = self.current.copy()
        self.gesture = 'OTRO'
        self.confidence = 0.0

    def set_gesture(self, gesture: str, confidence: float):
        if gesture not in GESTURE_POSES:
            gesture = 'OTRO'
        if gesture != self.gesture:
            self.target = get_pose_targets(gesture)
        self.gesture = gesture
        self.confidence = confidence

    def step(self, dt: float):
        alpha = 1.0 - math.exp(-dt / self.tau)
        self.current = self.current + (self.target - self.current) * alpha

    def full_points(self) -> np.ndarray:
        """(21,2): muñeca + 20 landmarks de dedos, en orden MediaPipe."""
        return np.vstack([[WRIST], self.current])
