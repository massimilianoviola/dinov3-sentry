import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DinoFeatureExtractor:
    """Extract patch-level features and the global CLS token from an image.
    Optionally run PCA on the extracted features to create a semantic RGB visualization.
    """

    def __init__(self, model_repo, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(
            f"[Backbone] Using device: {self.device}"
            + (f" ⚠️ Requested device: {device}" if f"{device}" != f"{self.device}" else "")
        )

        # Load processor and model from Hugging Face
        print(f"[Backbone] Loading {model_repo} from Hugging Face...")
        self.processor = AutoImageProcessor.from_pretrained(model_repo)
        self.model = AutoModel.from_pretrained(model_repo)
        self.model.to(self.device)
        self.model.eval()

        # Calculate model size
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Backbone] Model Size: {total_params / 1e6:.1f}M parameters")

        # Get patch size from model config
        self.patch_size = getattr(self.model.config, "patch_size")
        print(f"[Backbone] Detected patch_size: {self.patch_size}")

    def preprocess(self, img_pil):
        # Resize to nearest multiple of patch_size
        w, h = img_pil.size
        new_w = (w // self.patch_size) * self.patch_size
        new_h = (h // self.patch_size) * self.patch_size
        resized = img_pil.resize((new_w, new_h), resample=Image.LANCZOS)

        # Use processor to convert to tensor/normalize but disable resizing/cropping
        inputs = self.processor(
            images=resized, return_tensors="pt", do_resize=False, do_center_crop=False
        )
        return inputs["pixel_values"].to(self.device)

    def extract(self, frame_bgr):
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        pixel_values = self.preprocess(img_pil)  # (1, 3, H, W)

        with torch.inference_mode():
            outputs = self.model(pixel_values)
            last_hidden_state = outputs.last_hidden_state

            # Position 0: [CLS] token
            cls_token = last_hidden_state[:, 0, :].cpu().numpy()

            # Following are [Registers] tokens
            num_registers = getattr(self.model.config, "num_register_tokens", 0)

            # Skip [CLS] and [Registers] to get patch tokens
            patch_tokens = last_hidden_state[:, 1 + num_registers :, :]

            # Reshape tokens back to spatial grid
            B, N, D = patch_tokens.shape
            h_img, w_img = pixel_values.shape[2], pixel_values.shape[3]
            H_grid = h_img // self.patch_size
            W_grid = w_img // self.patch_size

            feat_map = patch_tokens.view(H_grid, W_grid, D).cpu().numpy()

        return feat_map, cls_token

    def visualize(self, frame_bgr):
        from sklearn.decomposition import PCA

        feat_map, _ = self.extract(frame_bgr)
        H, W, D = feat_map.shape
        h_orig, w_orig = frame_bgr.shape[:2]

        # PCA to 3 components
        pca = PCA(n_components=3)
        pca_proj = pca.fit_transform(feat_map.reshape(-1, D))

        # Normalize to 0-255
        p_min, p_max = pca_proj.min(axis=0), pca_proj.max(axis=0)
        pca_norm = (pca_proj - p_min) / (p_max - p_min + 1e-8)
        pca_rgb = (pca_norm.reshape(H, W, 3) * 255).astype(np.uint8)

        # Upscale to original size for side-by-side
        pca_rgb = cv2.resize(pca_rgb, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        return img_rgb, pca_rgb


if __name__ == "__main__":
    from pathlib import Path
    from urllib.request import Request, urlopen

    import yaml

    def load_image_from_url(url: str) -> np.ndarray:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = np.frombuffer(urlopen(req).read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Could not decode image from {url}")
        img_bgr = cv2.resize(img_bgr, (1280, 800), interpolation=cv2.INTER_LANCZOS4)
        return img_bgr

    url = "https://images.pexels.com/photos/414612/pexels-photo-414612.jpeg"
    frame = load_image_from_url(url)

    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "settings.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_repo = config["model"]["model_repo"]
    device = config["model"]["device"]

    print(f"[Test] Initializing DinoFeatureExtractor with {model_repo} on {device}...")
    backbone = DinoFeatureExtractor(model_repo=model_repo, device=device)

    # Visualize image and (upscaled) feature map
    img_rgb, pca_rgb = backbone.visualize(frame)
    feat_map, _ = backbone.extract(frame)
    print(f"[Test] Feature map shape: {feat_map.shape}")
    combined = np.hstack([img_rgb, pca_rgb])
    Image.fromarray(combined).show()

    print("[Test] Done.")
