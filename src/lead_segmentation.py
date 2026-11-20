import cv2
import numpy as np

# --------------------------
# Helpers
# --------------------------

def subdivide_large_box(box, rows=3, cols=4):
    """Split one large bounding box into a 3×4 grid (12 leads)."""
    x, y, w, h = box
    out = []
    for r in range(rows):
        for c in range(cols):
            rx = x + int(c * w / cols)
            ry = y + int(r * h / rows)
            rw = int(w / cols)
            rh = int(h / rows)
            out.append((rx, ry, rw, rh))
    return out


def force_central_split(img_gray, rows=3, cols=4):
    """Split the central region of the ECG into 3×4 leads."""
    H, W = img_gray.shape

    top_margin = int(0.12 * H)
    bottom_margin = int(0.08 * H)

    usable_h = H - top_margin - bottom_margin
    usable_w = W

    leads = []
    for r in range(rows):
        for c in range(cols):
            rx = int(c * usable_w / cols)
            ry = top_margin + int(r * usable_h / rows)
            rw = int(usable_w / cols)
            rh = int(usable_h / rows)
            leads.append((rx, ry, rw, rh))

    return leads


# --------------------------
# Main Detector
# --------------------------

def find_lead_regions(img_gray, debug=False):
    """
    Detect ECG lead regions.
    Always returns (boxes, debug_imgs) when debug=True.
    Otherwise returns boxes only.
    """

    H, W = img_gray.shape
    debug_imgs = {}

    # --------------------------
    # Step 1: Binarize
    # --------------------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)

    th = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=41, C=12
    )

    debug_imgs["binary"] = th.copy()

    # --------------------------
    # Step 2: Morph Closing to Highlight Lead Strips
    # --------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    debug_imgs["closed"] = closed.copy()

    # --------------------------
    # Step 3: Contour Detection
    # --------------------------
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    min_area = (H * W) * 0.02  # must be large enough to be a real panel

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w * h < min_area:
            continue

        boxes.append((x, y, w, h))

    # --------------------------
    # Step 4: Evaluate box count
    # --------------------------
    if len(boxes) == 12:
        # Perfect detection
        if debug:
            return boxes, debug_imgs
        return boxes

    if len(boxes) == 1:
        # One giant rectangle → split it
        big = boxes[0]
        leads = subdivide_large_box(big, rows=3, cols=4)

        if debug:
            debug_imgs["single_big_box"] = closed.copy()
            return leads, debug_imgs
        return leads

    # Fallback: use fixed central crop split
    leads = force_central_split(img_gray, rows=3, cols=4)

    if debug:
        debug_imgs["fallback_split"] = closed.copy()
        return leads, debug_imgs

    return leads
