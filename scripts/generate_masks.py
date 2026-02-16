"""Modified generate_masks.py - defensive SAM usage with checkpoint and debug"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import traceback
import argparse

# Add project root to Python path (adjust if needed)
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class MaskGenerator:
    def __init__(self, model_type="vit_b", checkpoint_path=None,
                 device=None, points_per_side=32, pred_iou_thresh=0.88,
                 stability_score_thresh=0.92, min_mask_region_area=100, debug=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        self.debug = debug
        self.mask_generator = None
        self._load_model()

    def _load_model(self):
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path is required. Provide via --checkpoint_path")
        ckpt = Path(self.checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"SAM checkpoint not found at {ckpt}. Please download & place it there.")
        print(f"[SAM] Loading {self.model_type} from {ckpt} on {self.device}...")
        
        # Load model with weights
        sam = sam_model_registry[self.model_type](checkpoint=str(ckpt))
        sam.to(device=self.device)
        sam.eval()
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True

        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=self.min_mask_region_area,
        )
        print("[SAM] Model loaded.")

    def _read_image_safe(self, image_path):
        # Read file robustly (handles unicode paths)
        raw = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        return img

    def process_image(self, image_path, output_path=None):
        image_path = Path(image_path)
        try:
            image = self._read_image_safe(image_path)
            if image is None:
                if self.debug: print(f"[WARN] Could not decode image: {image_path}")
                return None

            # ensure non-empty dims
            if image.size == 0:
                if self.debug: print(f"[WARN] Empty image: {image_path} shape {image.shape}")
                return None

            # Convert to RGB (handle grayscale and alpha)
            if image.ndim == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h, w = image_rgb.shape[:2]
            if min(h, w) < 50:
                if self.debug: print(f"[WARN] Skipping small image {image_path} -> {image_rgb.shape}")
                return None

            if self.debug:
                print(f"[INFO] Generating mask for {image_path} shape={image_rgb.shape} dtype={image_rgb.dtype}")

            # Generate masks
            try:
                with torch.no_grad():
                    masks = self.mask_generator.generate(image_rgb)
            except Exception:
                print(f"[ERROR] Exception during mask generation for {image_path}:")
                traceback.print_exc()
                return None

            if not masks:
                if self.debug: print(f"[INFO] No masks returned for {image_path}")
                return None

            # pick largest region by area
            masks = sorted(masks, key=lambda x: x.get('area', 0), reverse=True)
            combined_mask = masks[0]['segmentation'].astype(np.uint8) * 255

            # morphological clean-up
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            if output_path:
                outp = Path(output_path)
                outp.parent.mkdir(parents=True, exist_ok=True)
                # encode as PNG to buffer then write to file to avoid path issues
                success, buffer = cv2.imencode('.png', combined_mask)
                if success:
                    with open(outp, 'wb') as f:
                        buffer.tofile(f)
                else:
                    print(f"[ERROR] Failed to encode mask for {image_path}")
                    return None

            return combined_mask

        except Exception:
            print(f"[ERROR] Unhandled exception while processing {image_path}:")
            traceback.print_exc()
            return None


def process_directory(input_dir, output_dir, model_type, checkpoint_path,
                      points_per_side=32, debug=False, **kwargs):
    generator = MaskGenerator(model_type=model_type, checkpoint_path=checkpoint_path,
                              points_per_side=points_per_side, debug=debug, **kwargs)

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"[ERROR] Input directory missing: {input_path}")
        return

    # Recursively find image files (case-insensitive)
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    image_files = [p for p in input_path.rglob('*') if p.suffix.lower() in exts]

    print(f"[INFO] Found {len(image_files)} images under {input_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed = 0
    for p in tqdm(image_files, desc=f"Generating masks ({input_dir})"):
        try:
            mask_filename = f"{p.stem}.png"
            mask_path = output_path / mask_filename

            if mask_path.exists() and mask_path.stat().st_size > 0:
                # skip existing mask
                continue

            res = generator.process_image(p, mask_path)
            if res is not None:
                processed += 1

        except Exception:
            print(f"[ERROR] Error handling file {p}:")
            traceback.print_exc()
            continue

    print(f"[DONE] Processed {processed} / {len(image_files)} images for {input_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate masks using SAM (defensive)')
    parser.add_argument('--data_dir', required=True, help='Root data directory containing train/val/test')
    parser.add_argument('--model_type', default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'])
    parser.add_argument('--checkpoint_path', required=True, help='Path to SAM .pth checkpoint')
    parser.add_argument('--points_per_side', type=int, default=16, help='points_per_side (lower => faster)')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    args = parser.parse_args()

    for split in ['train', 'val', 'test']:
        in_dir = Path(args.data_dir) / split / 'images'
        out_dir = Path(args.data_dir) / split / 'masks'
        if in_dir.exists() and any(in_dir.iterdir()):
            print(f"\n[INFO] Processing split: {split}")
            process_directory(str(in_dir), str(out_dir),
                              model_type=args.model_type,
                              checkpoint_path=args.checkpoint_path,
                              points_per_side=args.points_per_side,
                              debug=args.debug)
        else:
            print(f"[INFO] Skipping {split} - no images found at {in_dir}")

if __name__ == "__main__":
    main()
