import os

import cv2
import numpy as np


_MODEL_CACHE = {}


def _order_corners(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)


def _resolve_weights_path(weights_path):
    if weights_path and os.path.exists(weights_path):
        return weights_path
    local_default = os.path.join(os.path.dirname(__file__), "weights", "whiteboard_best.pt")
    if os.path.exists(local_default):
        return local_default
    return None


def _load_model(weights_path):
    from ultralytics import YOLO

    if weights_path not in _MODEL_CACHE:
        _MODEL_CACHE[weights_path] = YOLO(weights_path)
    return _MODEL_CACHE[weights_path]


def detect_board_ml(img, weights_path=None, conf=0.30, min_mask_area_ratio=0.01):
    """
    Return ordered board corners from a segmentation model, or None when unavailable.
    Falls back cleanly when ultralytics/weights are missing.
    """
    resolved = _resolve_weights_path(weights_path)
    if resolved is None:
        return None

    try:
        model = _load_model(resolved)
    except Exception:
        return None

    try:
        result = model.predict(source=img, conf=conf, verbose=False)[0]
    except Exception:
        return None

    if result.masks is None or result.masks.data is None:
        return None

    masks = result.masks.data
    if hasattr(masks, "cpu"):
        masks = masks.cpu().numpy()
    img_area = img.shape[0] * img.shape[1]
    min_area = max(1, int(min_mask_area_ratio * img_area))

    best_corners = None
    best_score = -1.0
    for idx, mask in enumerate(masks):
        bin_mask = (mask > 0.5).astype(np.uint8) * 255
        area = int((bin_mask > 0).sum())
        if area < min_area:
            continue

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area:
            continue

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        corners = _order_corners(box)
        confidence = 1.0
        if result.boxes is not None and result.boxes.conf is not None and idx < len(result.boxes.conf):
            conf_val = result.boxes.conf[idx]
            confidence = float(conf_val.cpu().item()) if hasattr(conf_val, "cpu") else float(conf_val)
        score = area * confidence
        if score > best_score:
            best_score = score
            best_corners = corners

    return best_corners
