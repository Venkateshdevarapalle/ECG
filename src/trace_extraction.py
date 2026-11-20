# src/trace_extraction.py
"""
Trace extraction utilities for ECG image digitization.

Provides:
- binarize_for_trace(gray)
- extract_trace_by_edges(gray) -> xs, ys, skel_img
- extract_trace_by_contour(gray) -> xs, ys

The functions are defensive and return empty lists if extraction fails.
They aim to produce (x, y) pixel coordinates ready for pixels_to_signal().
"""

import cv2
import numpy as np
from scipy.ndimage import median_filter


def binarize_for_trace(gray, tile_grid=(15, 15), c=8):
    """
    Adaptive binarization that emphasizes thin black ECG traces and reduces grid noise.
    Returns (binary_inv, enhanced_gray)
    - binary_inv: binary image where trace/grid are white on black (uint8 0/255)
    - enhanced_gray: contrast-enhanced grayscale (uint8)
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # Contrast limited adaptive histogram equalization
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
    except Exception:
        enhanced = gray.copy()

    # Adaptive threshold (mean) — tune blockSize and C for typical ECG scans
    block = 41 if max(gray.shape) > 1000 else 31
    th = cv2.adaptiveThreshold(enhanced, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV,
                               blockSize=block, C=c)

    # Morphological open to remove tiny dot noise (grid dots)
    k = 3 if max(gray.shape) < 1500 else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    # Optional stronger remove for faint grid: apply a small closing then open
    if max(gray.shape) > 1500:
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel2, iterations=1)
        opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kernel2, iterations=1)

    return opened, enhanced


def _fill_gaps_in_trace(xs, ys, max_gap_px=10):
    """
    Fill small gaps in trace by linear interpolation.
    xs, ys are lists (x increasing). Return numpy arrays.
    """
    if len(xs) == 0:
        return np.array([]), np.array([])

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    # ensure sorted by x
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    full_x = np.arange(xs[0], xs[-1] + 1, dtype=int)
    full_y = np.full_like(full_x, fill_value=np.nan, dtype=float)

    # map known points
    idx = xs - full_x[0]
    full_y[idx] = ys

    # linear interpolate small gaps
    nans = np.isnan(full_y)
    if np.any(nans):
        # index locations
        good = np.where(~nans)[0]
        if len(good) >= 2:
            from_idx = good[0]
            to_idx = good[-1]
            # interpolate between known values
            full_y[from_idx:to_idx+1] = np.interp(
                np.arange(from_idx, to_idx+1),
                good,
                full_y[good]
            )

    # remove remaining NaNs (at edges) by forward/backward fill
    nans = np.isnan(full_y)
    if np.any(nans):
        # forward fill
        for i in range(len(full_y)):
            if np.isnan(full_y[i]):
                if i > 0:
                    full_y[i] = full_y[i-1]
        # backward fill
        for i in range(len(full_y)-1, -1, -1):
            if np.isnan(full_y[i]):
                if i < len(full_y)-1:
                    full_y[i] = full_y[i+1]

    # downsample back to columns present: return full_x and full_y
    return full_x.astype(float), full_y.astype(float)


def extract_trace_by_edges(gray):
    """
    Primary trace extraction using Canny edges + column-wise aggregation.

    Returns:
      xs: list of x pixel coordinates
      ys: list of y pixel coordinates
      skel_img: binary edge image useful for debug (uint8 0/255)
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    bin_img, enhanced = binarize_for_trace(gray, tile_grid=(15, 15), c=8)

    # Edge detection - Canny on enhanced image improves sensitivity
    # compute Canny thresholds adaptively
    v = np.median(enhanced)
    sigma = 0.33
    lower = int(max(5, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v * 2))
    edges = cv2.Canny(enhanced, lower, upper, L2gradient=True)

    # combine with binary mask to emphasize trace pixels
    comb = cv2.bitwise_and(edges, bin_img)

    h, w = comb.shape
    xs = []
    ys = []

    # For each column, pick the central tendency of edge pixels (median)
    for x in range(w):
        col = comb[:, x]
        ys_idx = np.where(col > 0)[0]
        if ys_idx.size == 0:
            # try using bin_img column (trace pixels)
            ys_idx = np.where(bin_img[:, x] > 0)[0]
            if ys_idx.size == 0:
                continue
        # choose median y (more robust to vertical artifacts)
        y_med = int(np.median(ys_idx))
        xs.append(x)
        ys.append(y_med)

    # If too few points found, try a relaxed approach using bin_img rows
    if len(xs) < (w * 0.05):
        # fallback: use bin_img connected component projection
        xs2 = []
        ys2 = []
        for x in range(w):
            col = bin_img[:, x]
            ys_idx = np.where(col > 0)[0]
            if ys_idx.size == 0:
                continue
            # choose center of white cluster
            # take median again
            y_med = int(np.median(ys_idx))
            xs2.append(x)
            ys2.append(y_med)
        xs, ys = xs2, ys2
        comb = bin_img.copy()

    # fill small gaps and smooth the trace
    full_x, full_y = _fill_gaps_in_trace(xs, ys)

    if full_x.size == 0:
        return [], [], comb

    # Apply light median filter to reduce jitter
    try:
        yf = median_filter(full_y, size=3)
    except Exception:
        yf = full_y

    # convert to python lists
    xs_out = full_x.tolist()
    ys_out = yf.tolist()

    return xs_out, ys_out, comb


def extract_trace_by_contour(gray):
    """
    Fallback trace extraction using the largest vertical-extending contour.

    Returns:
      xs, ys lists (x increasing). May be coarse but helps when edge method fails.
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    bin_img, enhanced = binarize_for_trace(gray, tile_grid=(15, 15), c=8)

    # Dilate slightly so the trace becomes a continuous blob for contour detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], []

    # Choose contour with maximal bounding box height (ECG trace is long horizontally with small vertical range)
    best = max(contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3])

    # Extract x,y from contour and aggregate by x
    pts = best.reshape(-1, 2)
    # group by x coordinate (use dictionary)
    groups = {}
    for (x, y) in pts:
        groups.setdefault(x, []).append(y)

    xs = []
    ys = []
    for x in sorted(groups.keys()):
        ys_list = groups[x]
        y_med = int(np.median(ys_list))
        xs.append(int(x))
        ys.append(int(y_med))

    full_x, full_y = _fill_gaps_in_trace(xs, ys)
    if full_x.size == 0:
        return [], []
    # slight smoothing
    try:
        yf = median_filter(full_y, size=3)
    except Exception:
        yf = full_y

    return full_x.tolist(), yf.tolist()
