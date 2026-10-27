# GNN-EMG — Low-Cost EMG with Automatic Gesture Labeling

A single-channel surface-EMG system built from an inexpensive **Sichiray** muscle sensor and an
**ESP32 SuperMini**, with a Python monitor that processes the signal in real time and uses the
**webcam (MediaPipe)** to label every EMG sample with the hand gesture being made. The labeled
recordings train a **GATv2 graph autoencoder**, pre-trained on the public NinaPro DB5 dataset and
fine-tuned on my own data.

```
Sichiray EMG ──► ESP32 SuperMini ──serial 115200──► monitor_emg_gesture.py ──► labeled CSV ──► GATv2 model
                                                          ▲
                                     webcam + MediaPipe ──┘ (gesture label for each sample)
```

## What it does

- **Acquisition:** the ESP32 samples the sensor at ~200 Hz and streams `RAW:<value>,SUAVIZADO:<value>`
  lines over USB serial (115 200 baud).
- **Real-time signal processing** (`monitor_emg_gesture.py`): band-pass and 60 Hz notch filtering,
  Teager–Kaiser energy, RMS, MAV, waveform length, zero crossings, envelope, adaptive
  activation threshold (baseline + kσ) with debounce, SNR.
- **Automatic labeling:** `gesto_mano.py` recognises the hand pose from the webcam
  (open hand, fist, index, peace, OK, thumbs up/down, three, pinch, phone) and every EMG row is saved
  together with the gesture, finger states and hand speed. Recording a dataset is just making
  gestures in front of the camera.
- **Model** (`gnn_emg/`): GATv2 graph autoencoder with a gesture head. NinaPro DB5 channels are graph
  nodes in pre-training; for the single-channel Sichiray signal, feature groups (raw, energy,
  complexity, threshold) become virtual nodes. Camera features are masked at the model input so the
  classifier cannot read the label from them.
- **Real-time inference** (`gnn_emg/inference/realtime.py`) with a virtual-hand display.

## Hardware

| Part | Notes |
|---|---|
| Sichiray single-channel EMG module | 3 electrodes: two over the muscle, one reference on bone |
| ESP32 SuperMini | Analog output of the module → an ADC pin; USB serial to the PC |
| Webcam | Only needed for labeling |

Electrodes on the **forearm flexors** (palm side, 3–4 finger widths below the elbow crease), the two
active ones 2–3 cm apart along the muscle, reference on the elbow or wrist bone. The ESP32
firmware is not included in this repository; any sketch that prints the line format above works.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# MediaPipe hand model (needed by the monitor)
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task','hand_landmarker.task')"
```

On Linux the serial port needs permissions: `sudo usermod -aG dialout $USER` (Debian/Ubuntu) or
`uucp` (Arch), then log in again.

## Usage

```bash
python monitor_emg_gesture.py --list                      # find the ESP32 port
python monitor_emg_gesture.py --port /dev/ttyACM0         # live EMG + camera, records CSV to ./dataset
python monitor_emg_gesture.py --demo                      # without hardware

python -m gnn_emg.train.finetune_sichiray                 # fine-tune on ./dataset
python -m gnn_emg.inference.realtime --port /dev/ttyACM0  # real-time inference (--demo without hardware)
python scripts/evaluate_splits.py                         # evaluation below
```

Pre-training (`python -m gnn_emg.train.pretrain_ninapro`) expects NinaPro DB5 `.mat` files in
`NinaPro/db5/`.

## Data

| Path | Content |
|---|---|
| `dataset/` | The recording used for fine-tuning (523 s, 104 352 samples; open hand, fist, index, peace) |
| `data/raw/` | All original recordings, gzip-compressed, with the labels exactly as the camera produced them |

In the training file, the rows the camera labeled `OTRO` (no gesture recognised) were relabeled as
`ABIERTA` (open hand): in that session the thumb was never detected as extended (the thumb test in
`gesto_mano.py` depends on handedness and image orientation), so an open hand was never recognised.
About 79 % of those rows show four extended fingers; the rest are other poses.

## Results

`scripts/evaluate_splits.py` evaluates the classifier three ways:

| Evaluation | Accuracy | Balanced accuracy |
|---|---|---|
| Released model, random validation split | 52.0 % | 46.1 % |
| Always predicting the majority class | 40.3 % | 25.0 % |
| Same training, validation on separate time blocks (5 folds) | 11.6 % | 22.8 % |

- With a **random split**, validation windows overlap (50 %) and sit next to training windows from the
  same recording, which inflates the score. On **unseen time blocks** the model does not generalize
  across the four gestures. Part of the drop also comes from the class mix changing between blocks,
  because gestures were recorded in runs.
- **One channel cannot separate finger gestures.** The fist stands out clearly (mean RMS ≈ 180 vs
  24–38 for the other gestures); index, peace and open hand use muscles that a single electrode pair
  barely distinguishes. Multi-class gesture recognition usually uses 8–16 channels (NinaPro DB5 has 16).
- **Real-time inference computes its features differently** from the recordings (other filter, fixed
  SNR and activity values). With those features the model predicts *fist* for every window, so in
  practice the live display separates **rest from contraction** through its RMS threshold.

What works reliably: the acquisition chain, the real-time processing and activation detection, and
the camera-based labeling pipeline. Rest vs. contraction is the realistic target for one channel.

## Next steps

- Compute features with one shared function for recording, training and real-time inference.
- Validate across sessions (train on some recordings, test on others).
- Make the thumb test independent of image mirroring.
- Add 2–4 Sichiray modules on different forearm muscles (more ESP32 ADC pins) for finger gestures.

## Repository structure

```
monitor_emg_gesture.py   real-time EMG + camera monitor and recorder
gesto_mano.py            MediaPipe hand-gesture recognition
gnn_emg/                 model, datasets, training, real-time inference, config, checkpoints, logs
scripts/                 evaluation
dataset/, data/raw/      recordings
legacy/                  earlier versions of the monitor
```

## License

[MIT](LICENSE). The MediaPipe hand model is distributed by Google under Apache 2.0; NinaPro is
available from its authors under its own terms.
