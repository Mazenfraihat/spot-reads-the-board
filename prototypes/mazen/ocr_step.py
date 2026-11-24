import os
import re
import shutil

import cv2
import numpy as np


def _resolve_tesseract_cmd():
    """Find a usable tesseract executable from PATH or common Windows install paths."""
    found = shutil.which("tesseract")
    if found:
        return found
    common_paths = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    )
    for path in common_paths:
        if os.path.exists(path):
            return path
    return None


def _prepare_ocr_image(flat, normalized):
    """Build an OCR-friendly high-contrast image from the flattened board."""
    source = normalized if normalized is not None else cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    source = cv2.resize(source, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
    binary = cv2.adaptiveThreshold(
        source,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        15,
    )
    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    denoised = cv2.medianBlur(cleaned, 3)
    return cv2.bitwise_not(denoised)


def _build_color_ink_mask(flat):
    """Separate likely marker ink by color (black/blue/red/green) in HSV space."""
    hsv = cv2.cvtColor(flat, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([90, 35, 20]), np.array([140, 255, 255]))
    red1 = cv2.inRange(hsv, np.array([0, 35, 20]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 35, 20]), np.array([180, 255, 255]))
    green = cv2.inRange(hsv, np.array([40, 30, 20]), np.array([90, 255, 255]))
    black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 120, 110]))
    mask = blue | red1 | red2 | green | black
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))


def _build_cluster_mask(flat, normalized):
    """Create a merged text mask for clustering from threshold + color-separated ink."""
    gray = normalized if normalized is not None else cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    color = _build_color_ink_mask(flat)
    merged = cv2.bitwise_or(thr, color)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    merged = cv2.dilate(merged, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)), iterations=1)
    return merged


def _find_text_clusters(flat, normalized):
    """Detect text clusters (regions) and return sorted region boxes in reading order."""
    mask = _build_cluster_mask(flat, normalized)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    min_area = max(250, int(0.001 * h * w))
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < min_area:
            continue
        if bw < 20 or bh < 10:
            continue
        # Drop likely board/frame lines that are too flat or too skinny.
        if bw > int(0.85 * w) and bh < int(0.06 * h):
            continue
        if bw / max(bh, 1) > 25:
            continue
        boxes.append((x, y, bw, bh))

    if not boxes:
        return []

    # Group rows by y-center, then sort left-to-right inside each row.
    boxes = sorted(boxes, key=lambda b: (b[1] + b[3] // 2, b[0]))
    rows = []
    row_tol = max(18, int(h * 0.03))
    for box in boxes:
        cy = box[1] + box[3] // 2
        placed = False
        for row in rows:
            if abs(cy - row["cy"]) <= row_tol:
                row["boxes"].append(box)
                row["cy"] = int(0.7 * row["cy"] + 0.3 * cy)
                placed = True
                break
        if not placed:
            rows.append({"cy": cy, "boxes": [box]})

    ordered = []
    for row in sorted(rows, key=lambda r: r["cy"]):
        for box in sorted(row["boxes"], key=lambda b: b[0]):
            ordered.append(box)
    return ordered


def _extract_lines_from_data(data, min_conf):
    grouped = {}
    n = len(data.get("text", []))
    for i in range(n):
        word = (data["text"][i] or "").strip()
        if not word:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            continue
        if conf < min_conf:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        grouped.setdefault(key, []).append((word, conf))

    lines = []
    for key in sorted(grouped.keys()):
        words = [w for w, _ in grouped[key]]
        confs = [c for _, c in grouped[key]]
        line = " ".join(words).strip()
        if line:
            lines.append({"text": line, "avg_conf": sum(confs) / len(confs)})
    return lines


def _line_quality(line):
    """Filter out obvious OCR garbage while keeping short meaningful phrases."""
    cleaned = " ".join(line.split()).strip()
    if len(cleaned) < 3:
        return False
    alnum_count = sum(ch.isalnum() for ch in cleaned)
    if alnum_count < 3:
        return False
    alpha_ratio = alnum_count / max(len(cleaned), 1)
    if alpha_ratio < 0.45:
        return False
    return True


def _canonical_line(line):
    return re.sub(r"[^a-z0-9]+", "", line.lower())


def _postprocess_lines(cluster_lines):
    """Normalize spacing and remove duplicates/low-quality lines."""
    cleaned = []
    seen = set()
    for raw in cluster_lines:
        line = " ".join(raw.split()).strip()
        if not _line_quality(line):
            continue
        key = _canonical_line(line)
        if len(key) < 3:
            continue
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(line)
    return cleaned


def _ocr_candidate(pytesseract, ocr_img, psm, min_conf):
    cfg = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(
        ocr_img,
        config=cfg,
        lang="eng",
        output_type=pytesseract.Output.DICT,
    )
    lines = _extract_lines_from_data(data, min_conf=min_conf)
    if not lines:
        return {"text": "", "score": 0.0, "avg_conf": 0.0, "line_count": 0, "psm": psm}
    text = "\n".join(line["text"] for line in lines)
    avg_conf = sum(line["avg_conf"] for line in lines) / len(lines)
    alpha_chars = sum(ch.isalnum() for ch in text)
    score = alpha_chars + (avg_conf * 2.0) + (len(lines) * 8.0)
    return {"text": text, "score": score, "avg_conf": avg_conf, "line_count": len(lines), "psm": psm}


def _ocr_cluster_roi(pytesseract, roi):
    """OCR one clustered region and return best candidate text + confidence."""
    ocr_img = _prepare_ocr_image(roi, normalized=None)
    candidates = [
        _ocr_candidate(pytesseract, ocr_img, psm=6, min_conf=38),
        _ocr_candidate(pytesseract, ocr_img, psm=7, min_conf=35),
        _ocr_candidate(pytesseract, ocr_img, psm=11, min_conf=33),
    ]
    return max(candidates, key=lambda c: c["score"])


def extract_text(flat, normalized=None):
    """Run OCR and return (text, status_note) without raising hard failures to the pipeline."""
    try:
        import pytesseract
    except ImportError:
        return "", "pytesseract missing (pip install pytesseract)"

    tesseract_cmd = _resolve_tesseract_cmd()
    if tesseract_cmd is None:
        return "", "tesseract executable missing (install tesseract-ocr)"
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        clusters = _find_text_clusters(flat, normalized)
        cluster_lines = []
        conf_values = []
        for x, y, w, h in clusters:
            pad = 4
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(flat.shape[1], x + w + pad)
            y1 = min(flat.shape[0], y + h + pad)
            roi = flat[y0:y1, x0:x1]
            best = _ocr_cluster_roi(pytesseract, roi)
            if best["text"]:
                cluster_lines.extend(best["text"].splitlines())
                conf_values.append(best["avg_conf"])

        # Fallback to whole-board OCR if clusters are too sparse.
        if len(cluster_lines) < 2:
            ocr_img = _prepare_ocr_image(flat, normalized)
            fallback = max(
                [
                    _ocr_candidate(pytesseract, ocr_img, psm=6, min_conf=40),
                    _ocr_candidate(pytesseract, ocr_img, psm=11, min_conf=35),
                ],
                key=lambda c: c["score"],
            )
            if fallback["text"]:
                cluster_lines = fallback["text"].splitlines()
                conf_values = [fallback["avg_conf"]]
                mode = f"whole-board psm={fallback['psm']}"
            else:
                mode = "clustered"
        else:
            mode = "clustered"
    except pytesseract.TesseractNotFoundError:
        return "", "tesseract executable not found in PATH"
    except pytesseract.TesseractError as err:
        return "", f"tesseract error: {err}"

    cleaned_lines = _postprocess_lines(cluster_lines)
    text = "\n".join(cleaned_lines)
    if not text:
        return "", "no text detected"
    avg_conf = (sum(conf_values) / len(conf_values)) if conf_values else 0.0
    return text, f"ok mode={mode} conf={avg_conf:.1f} clusters={len(clusters)} lines={len(cleaned_lines)}"
