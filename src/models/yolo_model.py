"""
src/models/yolo_model.py
-------------------------
YOLO wrapper for product localization and cropping.

Detects the primary clothing item in an image and returns a tight crop,
reducing background clutter before downstream embedding.

Model is used frozen (no fine-tuning) per assignment spec.
Any YOLOv8 variant can be used (yolov8n/s/m/l/x).
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO


# ------------------------------------------------------------------ #
#  Clothing-relevant COCO class ids                                    #
#  person=0  (full body provides good clothing context)               #
#  tie=27, backpack=24, handbag=26, suitcase=28 excluded              #
# ------------------------------------------------------------------ #
CLOTHING_CLASSES = {0}   # person class; YOLO on DeepFashion mostly detects persons


class YOLODetector:
    """
    YOLO-based product region detector.

    For each input image the detector:
      1. Runs inference to get bounding boxes.
      2. Selects the highest-confidence box (optionally filtered by class).
      3. Crops and returns the ROI.

    If no detection is found, the full image is returned unchanged.
    """

    def __init__(
        self,
        model_name: str = "yolov8m.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None,
        device: Optional[str] = None,
    ):
        print(f"[YOLO] Loading model: {model_name}")
        self.model = YOLO(model_name)
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.target_classes = target_classes  # None = accept all classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[YOLO] Ready on device: {self.device}")

    # -------------------------------------------------------------- #
    #  Single image                                                    #
    # -------------------------------------------------------------- #

    def detect_and_crop(
        self,
        image: Union[Image.Image, np.ndarray],
        padding: float = 0.05,
    ) -> Tuple[Image.Image, Optional[Tuple[int, int, int, int]]]:
        """
        Detect the primary clothing item and return the cropped image.

        Args:
            image:   PIL Image or numpy array (H, W, 3)
            padding: fractional padding to add around the detected box

        Returns:
            (cropped_image, bbox) where bbox is (x1, y1, x2, y2) in pixels,
            or (original_image, None) if no detection found.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        W, H = image.size

        results = self.model.predict(
            source=np.array(image),
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        best_box = self._select_best_box(results, W, H)
        if best_box is None:
            return image, None

        x1, y1, x2, y2 = best_box
        # Add padding
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(W, x2 + pad_w)
        y2 = min(H, y2 + pad_h)

        crop = image.crop((x1, y1, x2, y2))
        return crop, (x1, y1, x2, y2)

    def _select_best_box(
        self,
        results,
        W: int,
        H: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Select the highest-confidence bounding box.
        Optionally filter by target_classes.

        Returns (x1, y1, x2, y2) in integer pixel coords, or None.
        """
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            best_conf = -1
            best_box = None

            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())

                if self.target_classes and cls_id not in self.target_classes:
                    continue

                if conf > best_conf:
                    best_conf = conf
                    xyxy = box.xyxy[0].cpu().numpy()
                    best_box = (
                        int(xyxy[0]), int(xyxy[1]),
                        int(xyxy[2]), int(xyxy[3]),
                    )

            if best_box:
                return best_box

        return None

    # -------------------------------------------------------------- #
    #  Batch processing                                                #
    # -------------------------------------------------------------- #

    def batch_detect_and_crop(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        padding: float = 0.05,
    ) -> List[Tuple[Image.Image, Optional[Tuple]]]:
        """
        Run detect_and_crop on a list of images.

        Returns:
            List of (cropped_image, bbox_or_None) tuples
        """
        return [self.detect_and_crop(img, padding=padding) for img in images]
