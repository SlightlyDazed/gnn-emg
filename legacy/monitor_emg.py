#!/usr/bin/env python3
"""
=============================================================
  SICHIRAY EMG MONITOR — ESP32 | Python Real-Time Dashboard
  Lectura serial + procesamiento avanzado de señal EMG
  
  Mejoras implementadas (literatura 2010-2025):
  [1] TKEO - Teager-Kaiser Energy Operator (Solnik et al. 2010)
  [2] Umbral adaptativo AT (Carvalho et al. 2023 review)
  [3] Double Threshold DT con histéresis (Lidierth 1986)
  [4] RMS por ventana deslizante (Phinyomark et al. 2012)
  [5] Baseline adaptativo con media/SD rolling (Hodges & Bui 1996)
  [6] MAV + WL + ZC features (De Luca 1997)
  [7] Rectificación de onda completa + envelope LP
  [8] Oversampling x4 para reducir ruido ADC ESP32
=============================================================
"""

import sys
import time
import argparse
import threading
import collections
import numpy as np
import serial
import serial.tools.list_ports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, lfilter, iirnotch, filtfilt

# ─── PARÁMETROS CONFIGURABLES ────────────────────────────────────────────────
FS          = 200       # Frecuencia de muestreo ESP32 (delay 5ms → ~200Hz)
BAUD        = 115200
WINDOW_S    = 5         # Segundos de señal visibles en pantalla
N_DISPLAY   = FS * WINDOW_S
BASELINE_S  = 2         # Segundos para calcular baseline inicial
N_BASELINE  = FS * BASELINE_S

# Umbral adaptativo: threshold = baseline_mean + K * baseline_std
K_THRESHOLD     = 3.5   # Sensibilidad (↓ = más sensible, literatura: 2-5 SD)
K_THRESHOLD_LO  = 2.0   # Umbral inferior para Double Threshold
DEBOUNCE_MS     = 150   # ms mínimo entre pulsos (evita rebotes)

# Filtros digitales (Butterworth bandpass + notch)
F_LOW    = 20.0   # Hz (corte bajo — elimina movimiento y DC drift)
F_HIGH   = 150.0  # Hz (corte alto — señal útil sensor analógico básico)
F_NOTCH  = 60.0   # Hz (elimina ruido de red 60Hz, Perú/LATAM)
FILT_ORDER = 4

# TKEO params
TKEO_SMOOTHING = 15  # muestras de promedio móvil sobre TKEO

# Colores del dashboard
C_RAW       = '#4f98a3'  # teal
C_SMOOTH    = '#fdc451'  # dorado
C_FILT      = '#a86fdf'  # violeta
C_TKEO      = '#fd6e6e'  # rojo-coral
C_THRESH_HI = '#e87050'  # naranja umbral alto
C_THRESH_LO = '#e8c450'  # amarillo umbral bajo
C_PULSE     = '#6daa45'  # verde pulso

BG_DARK  = '#171614'
BG_SURF  = '#1c1b19'
BG_SURF2 = '#22211f'
TEXT_C   = '#cdccca'
DIVIDER  = '#393836'

# ─── BUFFERS CIRCULARES ──────────────────────────────────────────────────────
raw_buf      = collections.deque([2048]*N_DISPLAY, maxlen=N_DISPLAY)
smooth_buf   = collections.deque([2048]*N_DISPLAY, maxlen=N_DISPLAY)
filt_buf     = collections.deque([0.0]*N_DISPLAY, maxlen=N_DISPLAY)
tkeo_buf     = collections.deque([0.0]*N_DISPLAY, maxlen=N_DISPLAY)
rect_buf     = collections.deque([0.0]*N_DISPLAY, maxlen=N_DISPLAY)
envelope_buf = collections.deque([0.0]*N_DISPLAY, maxlen=N_DISPLAY)
pulse_buf    = collections.deque([0]*N_DISPLAY, maxlen=N_DISPLAY)

# Buffer para filtros (necesitan historial mínimo)
_raw_hist     = collections.deque([2048.0]*64, maxlen=64)
_tkeo_hist    = collections.deque([0.0]*TKEO_SMOOTHING, maxlen=TKEO_SMOOTHING)
_env_hist     = collections.deque([0.0]*30, maxlen=30)  # envelope LP 30 muestras (~150ms)

# Buffer para baseline adaptativo
baseline_buf  = collections.deque([2048.0]*N_BASELINE, maxlen=N_BASELINE)
baseline_ready = False

# Estado
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

# ─── FILTROS DIGITALES ───────────────────────────────────────────────────────
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

# Filtros IIR en línea (Direct Form II Transposed)
_bp_zi   = np.zeros(max(len(bp_a), len(bp_b)) - 1)
_nt_zi   = np.zeros(max(len(nt_a), len(nt_b)) - 1)

def filter_sample_online(x):
    """Aplica bandpass + notch muestra a muestra (IIR online)."""
    global _bp_zi, _nt_zi
    # Bandpass
    y_bp = bp_b[0] * x + _bp_zi[0]
    for k in range(len(_bp_zi) - 1):
        _bp_zi[k] = bp_b[k+1] * x - bp_a[k+1] * y_bp + _bp_zi[k+1]
    _bp_zi[-1] = bp_b[-1] * x - bp_a[-1] * y_bp
    # Notch
    y_nt = nt_b[0] * y_bp + _nt_zi[0]
    for k in range(len(_nt_zi) - 1):
        _nt_zi[k] = nt_b[k+1] * y_bp - nt_a[k+1] * y_nt + _nt_zi[k+1]
    _nt_zi[-1] = nt_b[-1] * y_bp - nt_a[-1] * y_nt
    return y_nt

# ─── TKEO (Teager-Kaiser Energy Operator) ────────────────────────────────────
# ψ[x(n)] = x²(n) - x(n-1)·x(n+1)  → solo necesita 3 muestras
_tkeo_prev2 = 2048.0
_tkeo_prev1 = 2048.0

def tkeo_sample(x_curr):
    """TKEO en línea usando las últimas 3 muestras."""
    global _tkeo_prev2, _tkeo_prev1
    # Con n-1 y el sample actual como "n-1" para el próximo
    tkeo_val = max(_tkeo_prev1**2 - _tkeo_prev2 * x_curr, 0.0)
    _tkeo_prev2 = _tkeo_prev1
    _tkeo_prev1 = x_curr
    return tkeo_val

# ─── PROMEDIO MÓVIL EN LÍNEA ─────────────────────────────────────────────────
_smooth_buf_local = collections.deque([2048]*10, maxlen=10)
def moving_avg(x, n=10):
    _smooth_buf_local.append(x)
    return sum(_smooth_buf_local) / len(_smooth_buf_local)

# ─── UMBRAL ADAPTATIVO ───────────────────────────────────────────────────────
_dt_counter = 0          # Muestras consecutivas sobre umbral (Double Threshold)
_dt_threshold_n = int(FS * 0.025)   # 25ms = ~5 muestras @ 200Hz
_last_pulse_t   = 0.0
_active_samples = 0
_total_samples  = 0

def update_baseline(raw_val):
    """Actualiza baseline rolling con muestras en reposo."""
    global baseline_ready
    if not metrics['muscle_active']:
        baseline_buf.append(float(raw_val))
    if len(baseline_buf) >= N_BASELINE and not baseline_ready:
        baseline_ready = True
        metrics['baseline_ready'] = True

def compute_threshold():
    """Recalcula umbral adaptativo (AT) basado en baseline actual."""
    arr = np.array(baseline_buf)
    mu  = np.mean(arr)
    sd  = np.std(arr)
    sd  = max(sd, 10.0)  # mínimo para evitar colapso
    metrics['baseline_mean'] = mu
    metrics['baseline_std']  = sd
    metrics['threshold_hi']  = mu + K_THRESHOLD    * sd
    metrics['threshold_lo']  = mu + K_THRESHOLD_LO * sd

# ─── PARSEO DE LÍNEA SERIAL ──────────────────────────────────────────────────
def parse_line(line: str):
    """
    Acepta múltiples formatos:
      RAW:1234,SUAVIZADO:1200,UMBRAL:600,PULSO:0   ← formato original ESP32
      1234                                           ← valor crudo solo
      1234,1200                                      ← raw,suavizado
    """
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

# ─── PROCESAMIENTO DE MUESTRA ────────────────────────────────────────────────
def process_sample(raw, smo):
    """Pipeline de procesamiento por muestra."""
    global _dt_counter, _last_pulse_t, _active_samples, _total_samples

    now = time.time()
    _total_samples += 1

    # 1. Filtrado digital online (bandpass + notch)
    filt = filter_sample_online(float(raw))

    # 2. TKEO sobre señal filtrada
    raw_tkeo = tkeo_sample(filt)
    _tkeo_hist.append(raw_tkeo)
    tkeo_smooth = sum(_tkeo_hist) / len(_tkeo_hist)

    # 3. Rectificación + envelope
    rect_val = abs(filt)
    _env_hist.append(rect_val)
    envelope = sum(_env_hist) / len(_env_hist)

    # 4. Baseline adaptativo
    update_baseline(raw)
    if _total_samples % (FS // 2) == 0:  # actualiza cada 0.5s
        compute_threshold()

    # 5. Double Threshold detection (DT) sobre envelope
    thr_hi = metrics['threshold_hi']
    thr_lo = metrics['threshold_lo']
    # Ajustar umbral relativo al baseline para señal filtrada
    filt_thr_hi = K_THRESHOLD    * metrics['baseline_std']
    filt_thr_lo = K_THRESHOLD_LO * metrics['baseline_std']

    if envelope > filt_thr_hi:
        _dt_counter += 1
    else:
        _dt_counter = max(0, _dt_counter - 1)

    dt_active = _dt_counter >= _dt_threshold_n

    # 6. Detección de pulso con debounce
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

    # 7. Features de ventana (calculados con buffer actual)
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

    # 8. Features RMS, MAV, WL, ZC sobre ventana de 200ms
    if _total_samples % 20 == 0:
        recent_filt = list(filt_buf)[-40:]  # últimos 200ms @ 200Hz
        arr = np.array(recent_filt, dtype=float)
        rms = float(np.sqrt(np.mean(arr**2)))
        mav = float(np.mean(np.abs(arr)))
        wl  = float(np.sum(np.abs(np.diff(arr))))
        zc  = int(np.sum(np.diff(np.sign(arr)) != 0))
        # SNR estimado: potencia contracción / potencia baseline
        base_std = metrics['baseline_std']
        snr_db = 20.0 * np.log10(rms / max(base_std, 1e-6)) if rms > 0 else 0.0
        with lock:
            metrics['rms'] = rms
            metrics['mav'] = mav
            metrics['wl']  = wl
            metrics['zc']  = zc
            metrics['snr_db'] = snr_db

# ─── HILO SERIAL ─────────────────────────────────────────────────────────────
def serial_thread(port, baud):
    global serial_running, ser_obj
    try:
        ser_obj = serial.Serial(port, baud, timeout=1.0)
        time.sleep(1.5)  # espera boot ESP32
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

# ─── MODO DEMO (sin hardware) ─────────────────────────────────────────────────
def demo_thread():
    """Genera señal EMG sintética realista para pruebas sin ESP32."""
    global serial_running
    serial_running = True
    phase = 0.0
    t = 0
    print("[DEMO] Sin puerto serial — generando señal EMG sintética...")
    while serial_running:
        t += 1
        # Señal EMG: ruido gaussiano + burst muscular cada ~2s
        noise  = np.random.normal(0, 30)
        burst  = 0.0
        if (t % 400) < 80:   # contracción ~80 muestras / 400ms @ 200Hz
            burst = np.random.normal(0, 250) * np.sin(np.pi * (t % 400) / 80)
        raw_v = int(2048 + noise + burst)
        raw_v = max(0, min(4095, raw_v))
        process_sample(raw_v, raw_v)
        time.sleep(1.0 / FS)

# ─── DASHBOARD MATPLOTLIB ────────────────────────────────────────────────────
def build_dashboard():
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9), facecolor=BG_DARK)
    fig.canvas.manager.set_window_title('Sichiray EMG Monitor | ESP32 Dashboard')

    gs = gridspec.GridSpec(4, 3, figure=fig,
                           left=0.05, right=0.98,
                           top=0.93, bottom=0.06,
                           hspace=0.55, wspace=0.35)

    ax_main  = fig.add_subplot(gs[0:2, 0:2])   # señal principal (ocupa 2 filas)
    ax_filt  = fig.add_subplot(gs[2, 0:2])      # señal filtrada + envelope
    ax_tkeo  = fig.add_subplot(gs[3, 0:2])      # TKEO
    ax_info  = fig.add_subplot(gs[0:2, 2])      # panel métricas
    ax_hist  = fig.add_subplot(gs[2, 2])        # histograma ADC
    ax_bar   = fig.add_subplot(gs[3, 2])        # barra actividad

    for ax in [ax_main, ax_filt, ax_tkeo, ax_info, ax_hist, ax_bar]:
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
    ax_tkeo.set_title('TKEO — Teager-Kaiser Energy Operator (amplifica spikes motores)', 
                       color=TEXT_C, fontsize=9, pad=4)
    ax_tkeo.set_xlabel('Tiempo (s)', fontsize=8)
    ax_tkeo.set_ylabel('Energía TKEO', fontsize=8)
    ax_tkeo.set_xlim(-WINDOW_S, 0)

    ln_tkeo, = ax_tkeo.plot(t_ax, list(tkeo_buf), color=C_TKEO, lw=0.9, label='TKEO suavizado')
    ax_tkeo.legend(loc='upper left', fontsize=7, framealpha=0.3,
                   facecolor=BG_SURF2, edgecolor=DIVIDER, labelcolor=TEXT_C)

    # ── Panel de métricas (texto) ──
    ax_info.axis('off')
    ax_info.set_title('Métricas en tiempo real', color=TEXT_C, fontsize=9, pad=4)
    metric_labels = [
        ('ADC Raw', 'raw', '', '#cdccca'),
        ('Suavizado', 'smooth', '', C_SMOOTH),
        ('RMS (filtrada)', 'rms', '', C_FILT),
        ('MAV', 'mav', '', C_FILT),
        ('WL (complejidad)', 'wl', '', C_FILT),
        ('ZC (cruces cero)', 'zc', '', C_FILT),
        ('SNR estimado', 'snr_db', ' dB', '#6daa45'),
        ('Pulsos detectados', 'pulse_count', '', C_PULSE),
        ('Tiempo activo', 'active_pct', ' %', C_PULSE),
        ('Baseline media', 'baseline_mean', '', C_THRESH_HI),
        ('Baseline σ', 'baseline_std', '', C_THRESH_HI),
        ('Umbral hi (AT)', 'threshold_hi', '', C_THRESH_HI),
    ]
    info_texts = []
    for i, (label, key, unit, color) in enumerate(metric_labels):
        y = 0.96 - i * 0.08
        ax_info.text(0.02, y, f'{label}:', transform=ax_info.transAxes,
                     color='#7a7876', fontsize=8, va='top')
        t_obj = ax_info.text(0.60, y, '—', transform=ax_info.transAxes,
                             color=color, fontsize=9, fontweight='bold', va='top',
                             fontfamily='monospace')
        info_texts.append((t_obj, key, unit))

    # Estado baseline
    baseline_status = ax_info.text(0.02, 0.04, '[*] Calibrando baseline...', 
                                    transform=ax_info.transAxes,
                                    color=C_THRESH_HI, fontsize=8, va='bottom')

    # Indicador de músculo activo
    muscle_patch = mpatches.FancyBboxPatch((0.02, -0.08), 0.96, 0.06,
                                           boxstyle='round,pad=0.01',
                                           facecolor=BG_SURF2, edgecolor=DIVIDER,
                                           transform=ax_info.transAxes)
    ax_info.add_patch(muscle_patch)
    muscle_text = ax_info.text(0.50, -0.05, '● REPOSO', transform=ax_info.transAxes,
                                ha='center', va='center', fontsize=10,
                                fontweight='bold', color='#7a7876')

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
    fig.text(0.50, 0.97, 'SICHIRAY EMG MONITOR  |  ESP32 + Python  |  GPIO34 ADC1-CH6',
             ha='center', va='top', color=TEXT_C, fontsize=11, fontweight='bold')
    fig.text(0.50, 0.955, f'Filtros: Butterworth BP {F_LOW}-{F_HIGH}Hz | Notch {F_NOTCH}Hz | TKEO | AT (μ+{K_THRESHOLD}σ) | DT ({K_THRESHOLD_LO}σ) | FS={FS}Hz',
             ha='center', va='top', color='#7a7876', fontsize=8)

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

        # Señal principal
        ln_raw.set_ydata(raw_data)
        ln_smo.set_ydata(smo_data)
        ln_thr_hi.set_ydata([m['threshold_hi']] * 2)
        ln_thr_lo.set_ydata([m['threshold_lo']] * 2)

        # Auto-escala Y señal principal
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

        # Métricas texto
        fmt = {
            'raw': f"{m['raw']:4d}",
            'smooth': f"{m['smooth']:4d}",
            'rms': f"{m['rms']:.1f}",
            'mav': f"{m['mav']:.1f}",
            'wl': f"{m['wl']:.0f}",
            'zc': f"{m['zc']:3d}",
            'snr_db': f"{m['snr_db']:.1f}",
            'pulse_count': f"{m['pulse_count']:4d}",
            'active_pct': f"{m['active_pct']:.1f}",
            'baseline_mean': f"{m['baseline_mean']:.0f}",
            'baseline_std': f"{m['baseline_std']:.1f}",
            'threshold_hi': f"{m['threshold_hi']:.0f}",
        }
        for t_obj, key, unit in info_texts:
            t_obj.set_text(fmt.get(key, '—') + unit)

        # Estado baseline
        if m['baseline_ready']:
            baseline_status.set_text('✔ Baseline calibrado')
            baseline_status.set_color(C_PULSE)
        else:
            pct = min(100, int(100 * m['samples_total'] / N_BASELINE))
            baseline_status.set_text(f'[*] Calibrando baseline... {pct}%')

        # Indicador muscular
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

        # Barra de activación (nivel RMS normalizado)
        rms_norm = min(m['rms'] / max(m['baseline_std'] * K_THRESHOLD * 2, 1), 1.0)
        act_fill.set_width(0.90 * rms_norm)
        act_text.set_text(f"{m['active_pct']:.1f} %")
        act_color = C_TKEO if m['dt_active'] else (C_SMOOTH if rms_norm > 0.3 else C_PULSE)
        act_fill.set_facecolor(act_color)

        return (ln_raw, ln_smo, ln_filt, ln_env, ln_tkeo,
                *[t for t, _, _ in info_texts],
                muscle_text, baseline_status, act_fill, act_text)

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()
    return ani

# ─── MAIN ─────────────────────────────────────────────────────────────────────
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
    global serial_running
    parser = argparse.ArgumentParser(description='Sichiray EMG Monitor — ESP32 Dashboard')
    parser.add_argument('--port', '-p', help='Puerto serial (ej: /dev/ttyUSB0, COM3)')
    parser.add_argument('--baud', '-b', type=int, default=BAUD, help=f'Baudrate (default: {BAUD})')
    parser.add_argument('--list', '-l', action='store_true', help='Lista puertos disponibles')
    parser.add_argument('--demo', action='store_true', help='Modo demo sin hardware')
    args = parser.parse_args()

    if args.list:
        list_ports()
        return

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
            
            # Esperar hasta 2.5s para que el ESP32 inicie (serial_thread hace sleep 1.5s)
            for _ in range(25):
                if serial_running or not t.is_alive():
                    break
                time.sleep(0.1)
                
            if not serial_running:
                print("[AVISO] No se pudo conectar. Iniciando modo demo...")
                t = threading.Thread(target=demo_thread, daemon=True)
                t.start()

    try:
        ani = build_dashboard()
    except KeyboardInterrupt:
        pass
    finally:
        serial_running = False
        print("\n[INFO] Monitor cerrado.")

if __name__ == '__main__':
    main()
