# src/preprocessing.py
import cv2
import numpy as np

def load_image(path):
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        img = cv2.imread(path)
    return img

def to_grayscale(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()

def enhance_contrast(img_gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(img_gray)

def remove_grid_strong(img_gray, debug=False):
    """
    Stronger grid removal for clinical ECG sheets.
    Steps:
      - Large-scale adaptive thresholding to identify grid lines
      - Morphological extraction of horizontal and vertical grid lines
      - Inpainting to remove grid from the image (preserve trace)
    """
    # slight blur to reduce noise
    blur = cv2.GaussianBlur(img_gray, (3,3), 0)

    # adaptive threshold to highlight grid + trace
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 10)

    # extract horizontal lines
    horiz = th.copy()
    cols = horiz.shape[1]
    horizontal_size = max(1, cols // 40)
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size,1))
    horiz = cv2.erode(horiz, horizontal_structure, iterations=1)
    horiz = cv2.dilate(horiz, horizontal_structure, iterations=1)

    # extract vertical lines
    vert = th.copy()
    rows = vert.shape[0]
    vertical_size = max(1, rows // 40)
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vert = cv2.erode(vert, vertical_structure, iterations=1)
    vert = cv2.dilate(vert, vertical_structure, iterations=1)

    grid = cv2.bitwise_or(horiz, vert)

    # mask for inpainting must have grid lines but not thick trace
    # dilate grid slightly to ensure removal
    grid_dil = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)

    # attempt to preserve the trace by subtracting a smoothed version
    # large-opening to remove trace from grid mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    grid_only = cv2.morphologyEx(grid_dil, cv2.MORPH_OPEN, kernel, iterations=1)

    # create inpaint mask where grid_only is present
    mask = grid_only

    # inpaint on original grayscale to remove gridlines
    inpainted = cv2.inpaint(img_gray, mask, 5, cv2.INPAINT_TELEA)

    # enhance contrast again
    cleaned = enhance_contrast(inpainted)

    # return cleaned image
    if debug:
        return cleaned, mask, inpainted
    return cleaned

def preprocess_image(path, debug=False):
    img = load_image(path)
    gray = to_grayscale(img)
    enhanced = enhance_contrast(gray)
    cleaned = remove_grid_strong(enhanced, debug=debug)
    return {
        "orig": img,
        "gray": gray,
        "enhanced": enhanced,
        "cleaned": cleaned
    }
