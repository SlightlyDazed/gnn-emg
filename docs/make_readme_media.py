"""
make_readme_media.py — Genera las animaciones y figuras del README en docs/images/.

  banner.png          título con un tramo real de la señal EMG
  system.png          cadena de adquisición: Sichiray → ESP32 → monitor ← webcam → CSV → GATv2
  emg_live.gif        15 s reales de la grabación: mano vista por la cámara + EMG + etiqueta
  virtual_graph.gif   los 4 nodos virtuales del grafo encendiéndose con sus features en el mismo tramo

Todo sale de dataset/sichiray_s01_20260711_010750_782003_fixed.csv (tramo 464–479 s, los cuatro gestos).
Necesita matplotlib, pandas, pillow y ffmpeg. Uso: python docs/make_readme_media.py
"""
import io
import os
import shutil
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Circle
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "images")
CSV = os.path.join(ROOT, "dataset", "sichiray_s01_20260711_010750_782003_fixed.csv")
T0, T1 = 462.0, 479.0          # tramo con abierta, índice, paz y puño
FPS = 10
WINDOW = 5.0                   # segundos visibles en la señal que se desplaza

BG = "#0d1117"
PANEL = "#161b22"
EDGE = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
EMG = "#ff6b4a"
GESTURES = {   # mismos colores que las figuras de resultados
    "ABIERTA": ("Open hand", "#4c9be8"),
    "INDICE": ("Index", "#8e6bd8"),
    "PAZ": ("Peace", "#3db39e"),
    "PUNIO": ("Fist", "#e8604c"),
    "—": ("No hand", "#6e7681"),
}
NODES = [  # nodos virtuales de gnn_emg/data/sichiray_dataset.py y la columna que los representa
    ("raw EMG", "emg_filt", "#58a6ff"),
    ("energy", "emg_rms", "#ff6b4a"),
    ("complexity", "emg_wl", "#d2a8ff"),
    ("threshold", "muscle_active", "#3fb950"),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def load():
    df = pd.read_csv(CSV, low_memory=False)
    df["t"] = df["timestamp_s"] - df["timestamp_s"].iloc[0]
    return df


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def frames_to_gif(frames, path, fps=FPS):
    tmp = tempfile.mkdtemp()
    try:
        for i, im in enumerate(frames):
            im.save(os.path.join(tmp, f"f_{i:04d}.png"))
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", os.path.join(tmp, "f_%04d.png"),
            "-filter_complex",
            "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=none:diff_mode=rectangle",
            "-loop", "0", path], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Mano dibujada a partir de los dedos que detectó la cámara ─────────────────
FINGERS = [  # (base x, base y, ángulo°, largo) en coordenadas de la mano
    ("f_thumb", -0.62, 0.15, 145, 0.75),
    ("f_index", -0.36, 0.62, 98, 1.0),
    ("f_middle", -0.08, 0.68, 90, 1.1),
    ("f_ring", 0.2, 0.64, 83, 1.0),
    ("f_pinky", 0.45, 0.52, 74, 0.8),
]


def draw_hand(ax, row, color):
    ax.add_patch(FancyBboxPatch((-0.62, -0.75), 1.2, 1.4, boxstyle="round,pad=0.02,rounding_size=0.35",
                                fc=color, ec="none", alpha=0.9))
    ax.add_patch(FancyBboxPatch((-0.42, -1.25), 0.8, 0.6, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc=color, ec="none", alpha=0.6))
    for col, x, y, ang, length in FINGERS:
        up = row[col] >= 0.5
        L = length if up else 0.28
        a = np.radians(ang if up else ang - 40)
        ax.plot([x, x + L * np.cos(a)], [y, y + L * np.sin(a)], color=color, lw=17,
                solid_capstyle="round", alpha=0.9 if up else 0.55)


def emg_live(df):
    w = df[(df.t >= T0 - WINDOW) & (df.t < T1)].copy()
    frames = []
    for k in range(int((T1 - T0) * FPS)):
        now = T0 + k / FPS
        seg = w[(w.t > max(now - WINDOW, T0)) & (w.t <= now + 0.05)]
        row = seg.iloc[-1]
        name, color = GESTURES.get(row.gesture, GESTURES["—"])

        fig = plt.figure(figsize=(9.6, 4.0), dpi=100, facecolor=BG)
        # Mano
        axh = fig.add_axes([0.0, 0.12, 0.27, 0.8])
        axh.set_xlim(-1.5, 1.5)
        axh.set_ylim(-1.5, 2.1)
        axh.set_aspect("equal")
        axh.axis("off")
        draw_hand(axh, row, color)
        fig.text(0.135, 0.9, "CAMERA LABEL", ha="center", fontsize=9, color=MUTED, family="DejaVu Sans Mono")
        fig.text(0.135, 0.08, name, ha="center", fontsize=17, weight="bold", color=color)

        # Señal que se desplaza
        ax = fig.add_axes([0.32, 0.3, 0.65, 0.58], facecolor=BG)
        x = seg.t - now
        ax.plot(x, seg.emg_filt, color="#c9d1d9", lw=0.6, alpha=0.75)
        ax.plot(x, seg.emg_envelope, color=EMG, lw=2.2)
        ax.set_xlim(-WINDOW, 0)
        ax.set_ylim(-500, 500)
        ax.set_xticks([])
        ax.set_ylabel("ADC units", fontsize=9)
        for s in ["top", "right", "bottom"]:
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(EDGE)
        active = row.muscle_active >= 0.5
        ax.text(-WINDOW, 470, "● MUSCLE ACTIVE" if active else "○ rest", ha="left", va="top", fontsize=11,
                weight="bold", color=EMG if active else MUTED, family="DejaVu Sans Mono")
        fig.text(0.97, 0.93, "filtered EMG", ha="right", fontsize=9, color="#c9d1d9")
        fig.text(0.97, 0.885, "envelope", ha="right", fontsize=9, color=EMG)

        # Banda de etiquetas
        axb = fig.add_axes([0.32, 0.17, 0.65, 0.08], facecolor=BG)
        g = seg.gesture.values
        edges = np.flatnonzero(g[1:] != g[:-1]) + 1
        starts = np.r_[0, edges]
        ends = np.r_[edges, len(g)]
        tt = x.values
        for s, e in zip(starts, ends):
            axb.axvspan(tt[s], tt[e - 1], color=GESTURES.get(g[s], GESTURES["—"])[1], lw=0)
        axb.set_xlim(-WINDOW, 0)
        axb.set_yticks([])
        axb.set_xticks([-5, -4, -3, -2, -1, 0])
        axb.set_xticklabels(["-5 s", "-4", "-3", "-2", "-1", "now"], fontsize=8)
        for s in axb.spines.values():
            s.set_visible(False)
        fig.text(0.32, 0.93, "Sichiray + ESP32 · 200 Hz · labelled live by the webcam", fontsize=10,
                 color=MUTED, family="DejaVu Sans Mono")
        frames.append(fig_to_image(fig))
        plt.close(fig)
    frames_to_gif(frames, os.path.join(OUT, "emg_live.gif"))


def virtual_graph(df):
    w = df[(df.t >= T0 - 1) & (df.t < T1)].copy()
    w["emg_filt"] = w["emg_filt"].abs()
    # Normalización robusta por nodo para la intensidad
    norm = {}
    for _, col, _ in NODES:
        lo, hi = np.percentile(w[col], 5), np.percentile(w[col], 98)
        norm[col] = (lo, max(hi - lo, 1e-9))
    pos = [(-1.0, 0.75), (1.0, 0.75), (1.0, -0.75), (-1.0, -0.75)]
    frames = []
    for k in range(int((T1 - T0) * FPS)):
        now = T0 + k / FPS
        seg = w[(w.t > now - 0.25) & (w.t <= now)]
        lv = []
        for _, col, _ in NODES:
            lo, span = norm[col]
            lv.append(float(np.clip((seg[col].mean() - lo) / span, 0, 1)))
        row = seg.iloc[-1]
        name, gcol = GESTURES.get(row.gesture, GESTURES["—"])

        fig = plt.figure(figsize=(6.4, 4.0), dpi=100, facecolor=BG)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-1.75, 1.75)
        ax.set_aspect("equal")
        ax.axis("off")
        # Aristas (grafo completo entre nodos virtuales) más gruesas cuanto más activos
        for i in range(4):
            for j in range(i + 1, 4):
                s = (lv[i] + lv[j]) / 2
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="#58a6ff",
                        lw=0.8 + 4 * s, alpha=0.15 + 0.6 * s, zorder=1)
        for (label, _, c), (x, y), v in zip(NODES, pos, lv):
            ax.add_patch(Circle((x, y), 0.28 + 0.22 * v, color=c, alpha=0.15 + 0.25 * v, zorder=2))
            ax.add_patch(Circle((x, y), 0.26, fc=PANEL, ec=c, lw=2 + 2 * v, zorder=3))
            ax.add_patch(Circle((x, y), 0.2 * v + 0.02, color=c, zorder=4))
            ax.text(x, y - 0.55 if y < 0 else y + 0.5, label, ha="center", va="center", fontsize=11,
                    color=c, weight="bold")
        ax.add_patch(FancyBboxPatch((-0.62, -0.38), 1.24, 0.82, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=PANEL, ec=EDGE, lw=1.2, zorder=5))
        ax.text(0, 0.12, "GATv2", ha="center", fontsize=15, weight="bold", color=TEXT, zorder=6)
        ax.text(0, -0.2, "4 virtual nodes", ha="center", fontsize=9, color=MUTED, zorder=6)
        ax.text(0, -1.6, f"camera: {name}", ha="center", fontsize=11, color=gcol, weight="bold")
        frames.append(fig_to_image(fig))
        plt.close(fig)
    frames_to_gif(frames, os.path.join(OUT, "virtual_graph.gif"))


def banner(df):
    w = df[(df.t >= 464) & (df.t < 479)]
    fig = plt.figure(figsize=(16, 4.6), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    # Señal real de fondo, a la derecha
    x = 8.3 + (w.t.values - 464) / 15 * 7.5
    ax.plot(x, 2.3 + w.emg_filt.values / 700, color="#c9d1d9", lw=0.5, alpha=0.35)
    ax.plot(x, 2.3 + w.emg_envelope.values / 260, color=EMG, lw=2.2)
    g = w.gesture.values
    edges = np.flatnonzero(g[1:] != g[:-1]) + 1
    for s, e in zip(np.r_[0, edges], np.r_[edges, len(g)]):
        ax.add_patch(plt.Rectangle((x[s], 0.55), x[e - 1] - x[s], 0.12, lw=0,
                                   color=GESTURES.get(g[s], GESTURES["—"])[1]))
    ax.text(0.9, 2.85, "GNN-EMG", fontsize=74, weight="bold", color=TEXT)
    ax.text(0.95, 2.1, "Low-cost forearm EMG, labelled by a webcam", fontsize=19, color="#c9d1d9")
    ax.add_patch(plt.Rectangle((0.95, 1.55), 0.8, 0.05, color=EMG, zorder=3))
    ax.text(0.95, 0.98, "SICHIRAY + ESP32  ·  MEDIAPIPE LABELS  ·  GATv2 GRAPH MODEL",
            fontsize=13, color=EMG, family="DejaVu Sans Mono")
    fig.savefig(os.path.join(OUT, "banner.png"), facecolor=BG)
    plt.close(fig)


def system():
    fig = plt.figure(figsize=(16, 4.2), dpi=110, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    def box(x, y, w, h, title, body, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=PANEL, ec=c, lw=1.6))
        ax.text(x + w / 2, y + h - 0.3, title, ha="center", va="top", fontsize=12.5, weight="bold", color=c)
        ax.text(x + w / 2, y + h - 0.78, body, ha="center", va="top", fontsize=9.6, color=TEXT,
                linespacing=1.45)

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6, mutation_scale=16))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label, ha="center", fontsize=8.5, color=MUTED,
                    family="DejaVu Sans Mono")

    w, h = 2.4, 1.75
    box(0.3, 2.1, w, h, "Sichiray", "single-channel EMG\n3 electrodes on the\nforearm flexors", EMG)
    box(3.45, 2.1, w, h, "ESP32 SuperMini", "ADC at ~200 Hz\nRAW / SUAVIZADO\nover USB serial", "#f0b429")
    box(6.6, 2.1, w, h, "Monitor", "band-pass + 60 Hz notch\nTKEO · RMS · MAV · WL\nadaptive threshold", "#58a6ff")
    box(6.6, 0.15, w, 1.5, "Webcam", "MediaPipe hand pose\n→ gesture label", "#3db39e")
    box(9.75, 2.1, w, h, "Labelled CSV", "every EMG sample +\ngesture, fingers,\nhand speed", "#d2a8ff")
    box(12.9, 2.1, w + 0.5, h, "GATv2 autoencoder", "pre-trained on NinaPro DB5\nfine-tuned on own data\n+ gesture head", "#3fb950")
    y = 2.1 + h / 2
    arrow(0.3 + w, y, 3.45, y, "analog")
    arrow(3.45 + w, y, 6.6, y, "115200")
    arrow(6.6 + w, y, 9.75, y)
    arrow(9.75 + w, y, 12.9, y)
    arrow(6.6 + w / 2, 1.65, 6.6 + w / 2, 2.1)
    fig.savefig(os.path.join(OUT, "system.png"), facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data = load()
    banner(data)
    system()
    emg_live(data)
    virtual_graph(data)
    print(f"Media written to {OUT}")
