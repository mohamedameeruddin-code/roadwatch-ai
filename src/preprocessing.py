"""
Image Preprocessing and Computer Vision Utilities for RoadWatch AI.
"""

from io import BytesIO
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image


def validate_image_file(
    file_input: Union[str, bytes, BytesIO],
    max_size_mb: float = 15.0,
) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Validates that the provided input is a valid, readable road image
    conforming to supported formats (JPG, JPEG, PNG) and size constraints.

    Returns:
        (is_valid, message, PIL Image instance if valid)
    """
    try:
        # Check size if bytes or BytesIO
        if isinstance(file_input, bytes):
            size_mb = len(file_input) / (1024 * 1024)
            if size_mb > max_size_mb:
                return False, f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {max_size_mb} MB.", None
            img = Image.open(BytesIO(file_input))
        elif isinstance(file_input, BytesIO):
            file_input.seek(0, 2)
            size_mb = file_input.tell() / (1024 * 1024)
            file_input.seek(0)
            if size_mb > max_size_mb:
                return False, f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {max_size_mb} MB.", None
            img = Image.open(file_input)
        elif isinstance(file_input, str):
            img = Image.open(file_input)
        else:
            return False, "Unsupported file input type provided.", None

        # Format validation
        img_format = (img.format or "").upper()
        if img_format not in ["JPEG", "JPG", "PNG"]:
            return False, f"Unsupported format '{img_format}'. Only JPG, JPEG, and PNG images are supported.", None

        # Dimensions validation
        width, height = img.size
        if width < 32 or height < 32:
            return False, f"Image resolution ({width}x{height}) is too small for pothole inspection (minimum 32x32).", None

        # Convert to RGB (in case of RGBA, Grayscale, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")

        return True, "Image validation successful.", img

    except Exception as e:
        return False, f"Failed to decode image file. File may be corrupt. Details: {str(e)}", None


def letterbox_image(
    image: np.ndarray,
    target_shape: Tuple[int, int] = (640, 640),
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resizes an image with padding to fit the target_shape while preserving aspect ratio.

    Args:
        image: Original image as BGR or RGB numpy array (H, W, C)
        target_shape: (target_width, target_height)
        pad_color: Padding color value (default standard grey 114)

    Returns:
        padded_image, scale_ratio, (pad_w, pad_h)
    """
    orig_h, orig_w = image.shape[:2]
    target_w, target_h = target_shape

    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * ratio))
    new_h = int(round(orig_h * ratio))

    dw = (target_w - new_w) / 2
    dh = (target_h - new_h) / 2

    # Resize
    if (orig_w, orig_h) != (new_w, new_h):
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image.copy()

    # Add borders
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
    )

    return padded, ratio, (dw, dh)


def denormalize_box_coordinates(
    boxes: List[List[float]],
    scale_ratio: float,
    dwdh: Tuple[float, float],
    original_shape: Tuple[int, int],
) -> List[List[int]]:
    """
    Reverts letterboxed coordinates back to original image dimensions.
    """
    orig_h, orig_w = original_shape
    dw, dh = dwdh
    adjusted_boxes = []

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        xmin = (xmin - dw) / scale_ratio
        ymin = (ymin - dh) / scale_ratio
        xmax = (xmax - dw) / scale_ratio
        ymax = (ymax - dh) / scale_ratio

        # Clip to original image boundaries
        xmin = max(0, min(int(round(xmin)), orig_w - 1))
        ymin = max(0, min(int(round(ymin)), orig_h - 1))
        xmax = max(0, min(int(round(xmax)), orig_w - 1))
        ymax = max(0, min(int(round(ymax)), orig_h - 1))

        adjusted_boxes.append([xmin, ymin, xmax, ymax])

    return adjusted_boxes


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: List[List[int]],
    confidences: List[float],
    class_names: Optional[List[str]] = None,
    box_color_bgr: Tuple[int, int, int] = (60, 75, 36),  # #244B3C in BGR
    text_color_bgr: Tuple[int, int, int] = (255, 255, 255),
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draws professional bounding boxes and confidence score labels onto a copy of the image.

    Args:
        image: RGB or BGR numpy array
        boxes: List of [xmin, ymin, xmax, ymax] in absolute pixel coordinates
        confidences: List of float confidence values (0.0 to 1.0)
        class_names: List of class labels (defaults to 'pothole')
        box_color_bgr: BGR color for borders and label badges
        text_color_bgr: BGR text color
        line_thickness: Box outline thickness

    Returns:
        Annotated image numpy array
    """
    annotated = image.copy()
    if class_names is None:
        class_names = ["pothole"] * len(boxes)

    for box, conf, cls_name in zip(boxes, confidences, class_names):
        xmin, ymin, xmax, ymax = box

        # Draw main bounding box
        cv2.rectangle(
            annotated,
            (xmin, ymin),
            (xmax, ymax),
            box_color_bgr,
            line_thickness,
            lineType=cv2.LINE_AA,
        )

        # Label text
        label_text = f"{cls_name}: {conf * 100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, font, font_scale, font_thickness
        )

        # Badge rectangle position (above box, or inside if too close to top edge)
        badge_y1 = max(0, ymin - text_h - 8)
        badge_y2 = ymin if ymin - text_h - 8 >= 0 else ymin + text_h + 8
        badge_x2 = min(annotated.shape[1] - 1, xmin + text_w + 10)

        # Draw filled label badge
        cv2.rectangle(
            annotated,
            (xmin, badge_y1),
            (badge_x2, badge_y2),
            box_color_bgr,
            cv2.FILLED,
        )

        # Draw label text
        text_baseline_y = (
            ymin - 4 if ymin - text_h - 8 >= 0 else ymin + text_h + 4
        )
        cv2.putText(
            annotated,
            label_text,
            (xmin + 5, text_baseline_y),
            font,
            font_scale,
            text_color_bgr,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated
