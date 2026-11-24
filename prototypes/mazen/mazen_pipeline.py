"""
Spot Whiteboard - Mini 1 starter prototype (conventional CV, no ML)
Pipeline: detect board -> flatten (homography) -> reduce glare -> find writing

Usage:
    pip install -r requirements.txt
    python whiteboard_starter.py data/phone_photos results

Each photo produces 4 images in results/:
    <name>_1_edges.jpg    Canny edges used to find the board
    <name>_2_board.jpg    detected board outline (green)
    <name>_3_flat.jpg     board flattened to a straight-on view
    <name>_4_writing.jpg  boxes around detected writing (red)
plus results/summary.csv with one row per photo (bring this to the meeting).

To improve a step: copy this file to prototypes/<yourname>/ and change ONE function.
"""
import sys, os, glob, csv, textwrap
import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)  # allow importing sibling helper modules (e.g., ocr_step.py)

from ocr_step import extract_text  # Step 5 OCR helper kept modular in a separate file
from board_detector_ml import detect_board_ml

DEFAULT_ML_WEIGHTS = os.environ.get("WHITEBOARD_ML_WEIGHTS")


def order_corners(pts):
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype("float32")
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype="float32")


# STEP 1 - Detect the board   *** MAZEN'S IMPROVEMENT: "whitest big quad" ***
#
# Why the starter fails:
#   (a) fixed Canny thresholds (50/150) miss the frame in dim rooms      -> "not found"
#   (b) the 15%-of-image area rule throws away boards seen from far away -> "not found"
#   (c) it takes the BIGGEST 4-sided shape, so a far board loses to the
#       wall around it                                                    -> wrong quad
# Fix:
#   1. CLAHE + auto-Canny: thresholds follow the image's own brightness
#   2. collect EVERY convex 4-sided shape >= 3% of the image, nested ones too
#   3. score each one by how white its inside is (bright + unsaturated),
#      because a whiteboard is the whitest flat thing in a classroom
#   4. among near-ties (within 6%), keep the largest -> the whole board,
#      not a box someone drew on it

MIN_AREA = 0.03      # board must cover at least 3% of the photo
TIE = 0.94           # candidates within 6% of the best whiteness count as ties


def auto_canny(gray, sigma=0.33):
    """Canny with thresholds set from the median brightness (adapts to dim/bright rooms)."""
    v = float(np.median(gray))
    return cv2.Canny(gray, int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v)))


def quad_candidates(edges, img_area):
    """Every convex quadrilateral outline in the edge map (outer AND nested contours)."""
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA * img_area:
            continue
        hull = cv2.convexHull(c)
        for eps in (0.01, 0.02, 0.04):
            approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quads.append(approx.reshape(4, 2))
                break
    return quads


def whiteness(quad, gray, sat):
    """Median brightness x (1 - mean saturation) inside the quad."""
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    inside = mask > 0
    return float(np.median(gray[inside])) * (1 - float(sat[inside].mean()) / 255)


def detect_board(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)   # boost contrast in dim rooms
    edges = auto_canny(cv2.GaussianBlur(eq, (5, 5), 0))
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)     # close gaps in the frame

    # Try trained segmentation-based detector first; fallback to CV heuristic when unavailable.
    ml_corners = detect_board_ml(img, weights_path=DEFAULT_ML_WEIGHTS, conf=0.30)
    if ml_corners is not None:
        return ml_corners, edges

    img_area = img.shape[0] * img.shape[1]
    quads = quad_candidates(edges, img_area)
    if not quads:
        return None, edges
    scores = [whiteness(q, gray, sat) for q in quads]
    best = max(scores)
    ties = [q for q, s in zip(quads, scores) if s >= TIE * best]
    board = max(ties, key=lambda q: cv2.contourArea(q.astype(np.float32)))
    return order_corners(board), edges


# STEP 2 - Flatten the board (perspective warp / homography)
def flatten(img, corners):
    tl, tr, br, bl = corners
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    H = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, H, (w, h))


# STEP 3 - Reduce glare / uneven lighting (background division)
def normalize_lighting(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 51)  # estimate the board's lighting
    return cv2.divide(gray, background, scale=255)                              # flatten bright/dark spots


# STEP 4 - Find the writing (adaptive threshold + morphology)
def binarize(norm):
    """Ink pixels = 255, board = 0. evaluate.py scores this mask against ground truth."""
    return cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 25, 15)


def find_writing(norm, flat):
    binary = binarize(norm)
    merged = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7)))  # merge letters into lines
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = flat.copy()
    boxes = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 300:                       # skip tiny specks
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
            boxes += 1
    return out, boxes


def render_ocr_preview(flat, text):
    """Create a side-by-side preview image with the flattened board and OCR text panel."""
    h, w = flat.shape[:2]
    panel = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.putText(panel, "OCR DETECTED TEXT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    y = 80
    wrapped = []
    for line in text.splitlines():
        wrapped.extend(textwrap.wrap(line, width=42) or [""])
    if not wrapped:
        wrapped = ["(no text detected)"]
    for line in wrapped:
        if y > h - 20:
            break
        cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        y += 24
    return np.hstack([flat, panel])


def format_readable_ocr(text, ocr_note):
    """Return a human-readable OCR report with numbered phrases."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    out = [f"OCR status: {ocr_note}", "", "Readable phrases:"]
    if not lines:
        out.append("- (no text detected)")
        return "\n".join(out)
    for idx, line in enumerate(lines, start=1):
        out.append(f"{idx:02d}. {line}")
    return "\n".join(out)


def run(path, out_dir):
    """Run the full pipeline on one photo. Returns a summary row (dict)."""
    name = os.path.splitext(os.path.basename(path))[0]
    row = {"photo": name, "board_found": "", "writing_regions": "", "ocr_chars": "", "note": "", "ocr_note": ""}
    img = cv2.imread(path)
    if img is None:
        print(f"[skip] can't read {path}")
        row["note"] = "unreadable file"
        return row
    scale = 1200 / max(img.shape[:2])         # resize for speed and consistent parameters
    img = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img

    corners, edges = detect_board(img)
    cv2.imwrite(f"{out_dir}/{name}_1_edges.jpg", edges)
    if corners is None:
        print(f"[FAIL] {name}: board not found")
        row["board_found"] = "no"
        return row

    vis = img.copy()
    cv2.polylines(vis, [corners.astype(int)], True, (0, 255, 0), 4)
    cv2.imwrite(f"{out_dir}/{name}_2_board.jpg", vis)

    flat = flatten(img, corners)
    cv2.imwrite(f"{out_dir}/{name}_3_flat.jpg", flat)

    normalized = normalize_lighting(flat)  # reuse glare-corrected board for both writing boxes and OCR
    boxed, n = find_writing(normalized, flat)
    cv2.imwrite(f"{out_dir}/{name}_4_writing.jpg", boxed)

    # STEP 5 - OCR text extraction on the flattened board.
    text, ocr_note = extract_text(flat, normalized)
    # Persist OCR output per image so it can be inspected independently of CSV metrics.
    with open(f"{out_dir}/{name}_5_text.txt", "w", encoding="utf-8") as f:
        f.write(f"OCR status: {ocr_note}\n\n")
        f.write(text if text else "(no text detected)")
    with open(f"{out_dir}/{name}_5_text_readable.txt", "w", encoding="utf-8") as f:
        f.write(format_readable_ocr(text, ocr_note))
    preview = render_ocr_preview(flat, text)
    cv2.imwrite(f"{out_dir}/{name}_6_ocr_preview.jpg", preview)

    print(f"[OK]   {name}: board found, {n} writing regions, OCR={ocr_note}")
    if text:
        print(f"       OCR text ({name}):")
        for i, line in enumerate(text.splitlines(), start=1):
            print(f"       {i:02d}. {line}")
    else:
        print("       OCR text: (no text detected)")
    row.update(board_found="yes", writing_regions=n, ocr_chars=len(text), ocr_note=ocr_note)
    return row


if __name__ == "__main__":
    in_dir = sys.argv[1] if len(sys.argv) > 1 else "data/phone_photos"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    os.makedirs(out_dir, exist_ok=True)
    photos = sorted({p for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")
                     for p in glob.glob(f"{in_dir}/*.{ext}")})
    if not photos:
        print(f"No photos found in {in_dir}/ - download them from the shared folder first.")
        sys.exit(1)

    rows = [run(p, out_dir) for p in photos]
    with open(f"{out_dir}/summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["photo", "board_found", "writing_regions", "ocr_chars", "note", "ocr_note"])
        writer.writeheader()
        writer.writerows(rows)

    found = sum(r["board_found"] == "yes" for r in rows)
    print(f"\nDone: board found in {found}/{len(photos)} photos -> {out_dir}/ (see summary.csv)")
