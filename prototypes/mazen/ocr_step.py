import os
import shutil

import cv2


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

    ocr_img = _prepare_ocr_image(flat, normalized)
    try:
        candidates = [
            _ocr_candidate(pytesseract, ocr_img, psm=6, min_conf=35),
            _ocr_candidate(pytesseract, ocr_img, psm=11, min_conf=30),
        ]
    except pytesseract.TesseractNotFoundError:
        return "", "tesseract executable not found in PATH"
    except pytesseract.TesseractError as err:
        return "", f"tesseract error: {err}"

    best = max(candidates, key=lambda c: c["score"])
    text = best["text"]
    if not text:
        return "", "no text detected"
    return text, f"ok psm={best['psm']} conf={best['avg_conf']:.1f} lines={best['line_count']}"
