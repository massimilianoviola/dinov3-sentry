# Unsupervised video anomaly detector with DINOv3🦕

🚨DINOv3-Sentry🚨 is a monitoring system using the state-of-the-art **DINOv3** [[1]](#references) backbone to perform unsupervised anomaly detection on video data.

Its features are augmentation invariant and robust to environmental changes (see demo), so the system only triggers when something semantically different appears in the scene.
The detector stores a set of normal feature patterns at each location, and it flags any patch that is significantly different from the calibration set.

## 💡 The idea

This can be used to automatically analyze footage from a camera and flag any anomalous intervals within it.

1. **Calibration phase**: the system currently observes the first N frames of a video or live stream to build the memory bank of normal feature vectors for every patch in the grid.
2. **Anomaly detection**: new patches are compared against this memory bank. If a patch is significantly different from everything seen during calibration, it is marked as anomalous. Two anomaly detection methods are supported: `Cosine similarity` and `Mahalanobis distance`.
3. **Area-based trigger**: an event is only reported if a significant fraction (e.g., >3%) of the scene is anomalous simultaneously, filtering out small artifacts or noise. The interval and size of the anomaly are reported in the terminal.
    ```
    [00:03.62] Anomaly #1 started
    [00:07.79] [ANOMALY] █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   3.2% (51/1590)
    [00:07.83] Anomaly #1 ended (4.2s)
    ```

## 📺 Demo

![DINOv3-Sentry demo](data/demo.mp4)

*Anomaly detection identifying a deer in a forest scene, changing season in the background.*

## 🛠️ Environment setup

### DINOv3 prerequisite
The DINOv3 model is hosted on Hugging Face. To use it, you must:
1. Have a Hugging Face account.
2. Accept the terms of the **facebook/dinov3-vits16** repository.
3. Generate an [Access Token](https://huggingface.co/settings/tokens).

### Option 1: Conda
```bash
conda create -n sentry python=3.10 -y
conda activate sentry
pip install -r requirements.txt
hf auth login  # Use your Access Token here
```

### Option 2: venv
```bash
python -m venv sentry
source sentry/bin/activate
pip install -r requirements.txt
hf auth login
```

## 🚀 Usage

1. **Configure**: edit `configs/settings.yaml`:
   - Set `stream.path` to a local file, YouTube URL, or `0` for webcam.
   - Adjust `min_anomaly_fraction` for sensitivity and `calibration_frames` for calibration duration.
   - (Optional) Adjust anomaly method in `main.py` or thresholds in `src/detector/`.
2. **Run**: `python main.py`
3. **Automated analysis**:
   - The app will automatically calibrate on the first `config.detector.min_train_frames` frames.
   - Real-time logging of anomaly start/end times will be printed in the terminal.
   - Visual display shows the original feed and a red binary mask overlay of detected anomalies.

## References
[1] DINOv3: https://arxiv.org/abs/2508.10104