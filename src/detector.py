import numpy as np
from sklearn.covariance import LedoitWolf
from tqdm import tqdm


class MahalanobisDetector:
    """Mahalanobis distance detector for anomaly detection in feature maps.
    Each patch location is considered independently, with its own threshold.
    """

    def __init__(self):
        self.mean_map = None
        self.precision_map = None
        self.threshold_map = None
        self.is_calibrated = False
        self.grid_shape = None

    def _get_mahalanobis_scores(self, feat_map):
        h, w, dim = feat_map.shape
        feats_flat = feat_map.reshape(-1, dim)
        diff = feats_flat - self.mean_map
        # Vectorized Mahalanobis distance calculation
        dist_sq = np.einsum("ij,ijk,ik->i", diff, self.precision_map, diff)
        return np.sqrt(dist_sq).reshape(h, w)

    def fit(self, feature_stack):
        # feature_stack is a list of N feature maps of shape (H, W, dim)
        H, W, dim = feature_stack[0].shape
        self.grid_shape = (H, W)
        print(f"[Detector] Calibrating on grid {H}x{W} with {len(feature_stack)} frames...")

        # Compute mean feature vector for each patch
        X_stack = np.stack(feature_stack, axis=0)  # (N, H, W, dim)
        X_flat = X_stack.reshape(len(feature_stack), -1, dim)  # (N, H*W, dim)
        n_frames, n_patches, _ = X_flat.shape
        self.mean_map = np.mean(X_flat, axis=0)  # (H*W, dim)

        # Compute inverse covariance matrix for each patch
        self.precision_map = []
        for i in tqdm(range(n_patches), desc="Patches processed"):
            obs = X_flat[:, i, :]
            try:
                lw = LedoitWolf(store_precision=True)
                lw.fit(obs)
                self.precision_map.append(lw.precision_)
            except:
                print(f"[Detector] Patch {i} failed to compute precision matrix. Using identity.")
                self.precision_map.append(np.eye(dim))
        self.precision_map = np.array(self.precision_map)
        del X_stack, X_flat

        # Calculate anomaly scores for all training frames to compute thresholds
        print("[Detector] Calculating patch-level thresholds...")
        train_scores = []
        for frame in tqdm(feature_stack, desc="Frames processed"):
            s = self._get_mahalanobis_scores(frame)
            train_scores.append(s)
        train_scores = np.array(train_scores)  # (N, H, W)

        # Use the max plus 5 std as the threshold
        max_scores = np.max(train_scores, axis=0)
        std_scores = np.std(train_scores, axis=0)
        self.threshold_map = max_scores + (5 * std_scores)
        print(f"[Detector] Ready. Global threshold avg: {np.mean(self.threshold_map):.2f}")

        self.is_calibrated = True
        del train_scores, max_scores, std_scores

    def predict(self, feat_map):
        if not self.is_calibrated:
            return None, None
        scores = self._get_mahalanobis_scores(feat_map)
        binary_mask = (scores > self.threshold_map).astype(np.uint8)
        return scores, binary_mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    H, W, DIM = 30, 40, 384
    N_TRAIN = 100

    print("[Test] Generating training data...")
    feature_stack = []
    for _ in range(N_TRAIN):
        # Base features with mean 0.5 + random noise
        frame = np.random.normal(loc=0.5, scale=0.05, size=(H, W, DIM))
        feature_stack.append(frame.astype(np.float32))

    detector = MahalanobisDetector()
    detector.fit(feature_stack)

    # Generate a test frame with anomalies in 5x5 squares
    test_frame = np.random.normal(loc=0.5, scale=0.05, size=(H, W, DIM)).astype(np.float32)
    anomaly_size = 5
    test_frame[10 : 10 + anomaly_size, 10 : 10 + anomaly_size, :] += 0.2
    test_frame[20 : 20 + anomaly_size, 20 : 20 + anomaly_size, :] -= 0.05

    print("[Test] Predicting anomaly scores and mask...")
    anomaly_map, detection_mask = detector.predict(test_frame)

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.title("Test frame (mean of channels)")
    plt.imshow(np.mean(test_frame, axis=-1), cmap="Spectral")

    plt.subplot(1, 3, 2)
    plt.title(
        f"Raw scores (max: {anomaly_map.max():.1f}, mean threshold: {np.mean(detector.threshold_map):.1f})"
    )
    plt.imshow(anomaly_map, cmap="Spectral")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Patch-level detection mask")
    plt.imshow(detection_mask, cmap="gray")

    plt.tight_layout()
    plt.show()
