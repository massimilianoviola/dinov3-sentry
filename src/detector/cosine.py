import numpy as np
from tqdm import tqdm


class CosineSimilarityDetector:
    """Anomaly detector based on cosine similarity to a memory bank.
    For each patch location, it stores a library of normal vectors, and an anomaly is detected if the
    best cosine similarity to the memory bank for that specific location is below the threshold.
    """

    def __init__(self):
        self.memory_bank = None
        self.threshold_map = None
        self.is_calibrated = False
        self.grid_shape = None

    def _L2_normalize(self, x):
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / (norm + 1e-8)

    def _get_cosine_scores(self, feat_map, exclude_index=None):
        h, w, dim = feat_map.shape
        feats_flat = self._L2_normalize(feat_map.reshape(-1, dim))

        # Optimized dot product between self.memory_bank (N_frames, H*W, dim) and feats_flat (H*W, dim)
        sims = np.einsum("nih,ih->ni", self.memory_bank, feats_flat)
        if exclude_index is not None:
            sims[exclude_index, :] = -np.inf

        # We need the max similarity for each of the H*W patches across N_frames
        best_sim = np.max(sims, axis=0)  # (H*W,)
        # Anomaly score: 0.0 = perfect match, 1.0 = orthogonal, 2.0 = opposite direction
        scores = 1.0 - best_sim
        return scores.reshape(h, w)

    def fit(self, feature_stack):
        H, W, dim = feature_stack[0].shape
        self.grid_shape = (H, W)
        print(f"[Detector] Calibrating on grid {H}x{W} with {len(feature_stack)} frames...")
        self.memory_bank = np.array([self._L2_normalize(f.reshape(-1, dim)) for f in feature_stack])

        # Calculate anomaly scores for all training frames to compute thresholds
        print("[Detector] Calculating patch-level thresholds...")
        train_scores = []
        for i, frame in enumerate(tqdm(feature_stack, desc="Frames processed")):
            train_scores.append(self._get_cosine_scores(frame, exclude_index=i))
        train_scores = np.array(train_scores)

        # Use the max plus 5 std as the threshold
        max_scores = np.max(train_scores, axis=0)
        std_scores = np.std(train_scores, axis=0)
        self.threshold_map = np.maximum(max_scores, 0.2) + (5 * std_scores)
        print(f"[Detector] Ready. Global threshold avg: {np.mean(self.threshold_map):.4f}")

        self.is_calibrated = True
        del train_scores

    def predict(self, feat_map):
        if not self.is_calibrated:
            return None, None
        scores = self._get_cosine_scores(feat_map)
        binary_mask = (scores > self.threshold_map).astype(np.uint8)
        return scores, binary_mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    H, W, DIM = 30, 40, 384
    N_TRAIN = 30

    print("[Test] Generating training data...")
    feature_stack = []
    for _ in range(N_TRAIN):
        # Base features with mean 0.5 + random noise
        frame = np.random.normal(loc=0.5, scale=0.05, size=(H, W, DIM))
        feature_stack.append(frame.astype(np.float32))

    detector = CosineSimilarityDetector()
    detector.fit(feature_stack)

    # Generate a test frame with anomalies in 5x5 squares
    test_frame = np.random.normal(loc=0.5, scale=0.05, size=(H, W, DIM)).astype(np.float32)
    anomaly_size = 5
    test_frame[10 : 10 + anomaly_size, 10 : 10 + anomaly_size, : DIM // 2] *= -0.5
    test_frame[20 : 20 + anomaly_size, 20 : 20 + anomaly_size, :] -= 1.5

    print("[Test] Predicting anomaly scores and mask...")
    anomaly_map, detection_mask = detector.predict(test_frame)

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.title("Test frame (mean of channels)")
    plt.imshow(np.mean(test_frame, axis=-1), cmap="Spectral")

    plt.subplot(1, 3, 2)
    plt.title(
        f"Raw scores (max: {anomaly_map.max():.4f}, mean threshold: {np.mean(detector.threshold_map):.4f})"
    )
    plt.imshow(anomaly_map, cmap="Spectral")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Patch-level detection mask")
    plt.imshow(detection_mask, cmap="gray")

    plt.tight_layout()
    plt.show()
