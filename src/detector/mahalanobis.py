import numpy as np
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from tqdm import tqdm


def _fit_patch(obs, dim):
    """Fit LedoitWolf for a single patch. Returns precision matrix."""
    try:
        lw = LedoitWolf(store_precision=True)
        lw.fit(obs)
        return lw.precision_
    except:
        return np.eye(dim)


def _score_frame(feat_map, mean_map, precision_map):
    """Compute Mahalanobis scores for a single frame."""
    h, w, dim = feat_map.shape
    feats_flat = feat_map.reshape(-1, dim)
    diff = feats_flat - mean_map
    # Vectorized Mahalanobis distance calculation
    dist_sq = np.einsum("ij,ijk,ik->i", diff, precision_map, diff)
    return np.sqrt(dist_sq).reshape(h, w)


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
        return _score_frame(feat_map, self.mean_map, self.precision_map)

    def fit(self, feature_stack):
        # feature_stack is a list of N feature maps of shape (H, W, dim)
        H, W, dim = feature_stack[0].shape
        self.grid_shape = (H, W)
        n_frames = len(feature_stack)
        print(f"[Detector] Calibrating on grid {H}x{W} with {n_frames} frames...")

        # Compute mean feature vector for each patch
        X_stack = np.stack(feature_stack, axis=0)  # (N, H, W, dim)
        X_flat = X_stack.reshape(n_frames, -1, dim)  # (N, H*W, dim)
        n_patches = X_flat.shape[1]
        self.mean_map = np.mean(X_flat, axis=0)  # (H*W, dim)

        # Parallel LedoitWolf fitting to compute inverse covariance matrix for each patch
        print(f"[Detector] Fitting {n_patches} patches in parallel...")
        precision_list = Parallel(n_jobs=-1, backend="loky")(
            delayed(_fit_patch)(X_flat[:, i, :], dim)
            for i in tqdm(range(n_patches), desc="Patches")
        )
        self.precision_map = np.array(precision_list)
        del X_stack, X_flat

        # Calculate anomaly scores for all training frames to compute thresholds
        print("[Detector] Calculating patch-level thresholds in parallel...")
        train_scores = Parallel(n_jobs=-1, backend="loky")(
            delayed(_score_frame)(f, self.mean_map, self.precision_map)
            for f in tqdm(feature_stack, desc="Frames processed")
        )
        train_scores = np.array(train_scores)  # (N, H, W)

        # Use the max plus chi-squared std as the threshold
        max_scores = np.max(train_scores, axis=0)
        chi2_margin = np.sqrt(2 * dim)
        self.threshold_map = max_scores + chi2_margin
        print(f"[Detector] Ready. Global threshold avg: {np.mean(self.threshold_map):.4f}")

        self.is_calibrated = True
        del train_scores, max_scores

    def predict(self, feat_map):
        if not self.is_calibrated:
            return None, None
        scores = self._get_mahalanobis_scores(feat_map)
        binary_mask = (scores > self.threshold_map).astype(np.uint8)
        return scores, binary_mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt

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
    test_frame[20 : 20 + anomaly_size, 20 : 20 + anomaly_size, :] -= 0.15

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
