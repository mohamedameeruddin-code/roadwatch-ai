"""
Reusable Pothole Inference Engine for RoadWatch AI.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from src.preprocessing import draw_bounding_boxes, validate_image_file


@dataclass
class DetectionResult:
    """Data transfer object containing inference outputs and metadata."""
    annotated_image: Optional[np.ndarray] = None
    num_detections: int = 0
    boxes: List[List[int]] = field(default_factory=list)  # [xmin, ymin, xmax, ymax]
    confidences: List[float] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    inference_time_ms: float = 0.0
    is_success: bool = True
    error_message: Optional[str] = None

    @property
    def highest_confidence(self) -> float:
        return max(self.confidences) if self.confidences else 0.0

    @property
    def average_confidence(self) -> float:
        return (sum(self.confidences) / len(self.confidences)) if self.confidences else 0.0


class PotholeDetector:
    """
    High-level YOLOv8n object detection engine specialized for road potholes.
    Designed for reuse across CLI, test scripts, and the Streamlit UI.
    """

    def __init__(
        self,
        model_path: str = "models/best.pt",
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.is_ready = False
        self.status_message = ""

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Attempts to load the YOLOv8 model weights cleanly."""
        if not ULTRALYTICS_AVAILABLE:
            self.is_ready = False
            self.status_message = "Ultralytics package is not installed. Please run `pip install -r requirements.txt`."
            return

        if not self.model_path.is_file():
            self.is_ready = False
            self.status_message = (
                f"Model weights not found at '{self.model_path}'. "
                "Please train the model via `python src/train.py` or place pretrained weights in `models/best.pt`."
            )
            return

        try:
            # Instantiate Ultralytics YOLO
            self.model = YOLO(str(self.model_path))
            self.is_ready = True
            self.status_message = f"Model loaded successfully from {self.model_path.name}."
        except Exception as e:
            self.is_ready = False
            self.status_message = f"Failed to initialize model from '{self.model_path}': {str(e)}"

    def predict(
        self,
        image_input: Union[str, bytes, np.ndarray, Image.Image],
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ) -> DetectionResult:
        """
        Executes pothole detection on the provided image.

        Args:
            image_input: File path, raw bytes, numpy array (RGB/BGR), or PIL Image.
            conf_threshold: Detection confidence cutoff (0.05 to 1.00).
            iou_threshold: Non-Maximum Suppression (NMS) IoU threshold.

        Returns:
            DetectionResult object.
        """
        if not self.is_ready or self.model is None:
            return DetectionResult(
                annotated_image=None,
                num_detections=0,
                is_success=False,
                error_message=self.status_message,
            )

        # Convert input to numpy RGB array for drawing and inference
        try:
            if isinstance(image_input, (str, bytes)):
                is_valid, msg, pil_img = validate_image_file(image_input)
                if not is_valid or pil_img is None:
                    return DetectionResult(
                        annotated_image=None,
                        num_detections=0,
                        is_success=False,
                        error_message=msg,
                    )
                rgb_image = np.array(pil_img)
            elif isinstance(image_input, Image.Image):
                rgb_image = np.array(image_input.convert("RGB"))
            elif isinstance(image_input, np.ndarray):
                # Ensure 3-channel
                if len(image_input.shape) == 2:
                    rgb_image = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
                else:
                    rgb_image = image_input.copy()
            else:
                return DetectionResult(
                    annotated_image=None,
                    num_detections=0,
                    is_success=False,
                    error_message="Unsupported image input type.",
                )

            # Record inference latency
            t_start = time.perf_counter()

            # Execute forward inference using Ultralytics YOLOv8
            results = self.model.predict(
                source=rgb_image,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=640,
                device=self.device,
                verbose=False,
            )

            latency_ms = (time.perf_counter() - t_start) * 1000.0

            boxes_list: List[List[int]] = []
            confidences_list: List[float] = []
            classes_list: List[str] = []

            if results and len(results) > 0:
                yolo_boxes = results[0].boxes
                if yolo_boxes is not None and len(yolo_boxes) > 0:
                    for b in yolo_boxes:
                        # Extract xyxy bounding box in absolute pixels
                        xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
                        conf = float(b.conf[0].cpu().numpy())
                        cls_idx = int(b.cls[0].cpu().numpy())
                        cls_name = self.model.names.get(cls_idx, "pothole")

                        boxes_list.append(xyxy)
                        confidences_list.append(conf)
                        classes_list.append(cls_name)

            # Draw annotations using our forest green palette
            annotated_rgb = draw_bounding_boxes(
                image=rgb_image,
                boxes=boxes_list,
                confidences=confidences_list,
                class_names=classes_list,
                box_color_bgr=(36, 75, 60),  # Primary forest green #244B3C
                text_color_bgr=(255, 255, 255),
                line_thickness=3,
            )

            return DetectionResult(
                annotated_image=annotated_rgb,
                num_detections=len(boxes_list),
                boxes=boxes_list,
                confidences=confidences_list,
                classes=classes_list,
                inference_time_ms=round(latency_ms, 1),
                is_success=True,
                error_message=None,
            )

        except Exception as e:
            return DetectionResult(
                annotated_image=None,
                num_detections=0,
                is_success=False,
                error_message=f"Inference execution failed: {str(e)}",
            )
