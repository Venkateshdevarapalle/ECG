# app/streamlit_app.py
import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile
from datetime import datetime

from src.preprocessing import preprocess_image
from src.lead_segmentation import find_lead_regions
from src.trace_extraction import extract_trace
from src.signal_reconstruction import detect_grid_spacing, pixels_to_signal
from src.evaluation import heart_rate_from_signal
from src.utils import save_csv, zip_files

st.set_page_config(page_title="ECG Image Digitizer (Multi-lead)", layout="wide")

st.title("ECG Image Digitizer — Multi-lead")
st.write("Upload a full ECG sheet (12-lead) and the system will extract each lead into a CSV.")

uploaded = st.file_uploader("Upload ECG image (PNG/JPG)", type=["png","jpg","jpeg"])

with st.sidebar:
    st.header("Settings")
    sampling_rate = st.number_input("Output sampling rate (Hz)", value=360, step=1)
    paper_speed_mm_s = st.selectbox("Paper speed (mm/s)", [25.0, 50.0], index=0)
    apply_smoothing = st.checkbox("Apply smoothing (median filter)", value=True)
    show_debug = st.checkbox("Show debug images (lead boxes, cleaned)", value=False)

def safe_imdecode(file_bytes):
    try:
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            tmp = NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(file_bytes)
            tmp.flush()
            tmp.close()
            img = cv2.imread(tmp.name)
            try:
                os.remove(tmp.name)
            except Exception:
                pass
        return img
    except Exception:
        return None

if uploaded is not None:
    # load image
    img_bgr = safe_imdecode(uploaded.getvalue())
    if img_bgr is None:
        st.error("Failed to load image.")
        st.stop()
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # find leads
    boxes = find_lead_regions(img_gray)
    st.sidebar.write(f"Detected {len(boxes)} lead regions (boxes)")

    # show original
    st.subheader("Original (preview)")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

    lead_files = []
    all_meta = []
    os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)

    # process each box
    for i, (x,y,w,h) in enumerate(boxes):
        # crop with margin
        padx = max(5, int(0.03 * w))
        pady = max(5, int(0.03 * h))
        cx1, cy1 = max(0,x-padx), max(0,y-pady)
        cx2, cy2 = min(img_bgr.shape[1], x+w+padx), min(img_bgr.shape[0], y+h+pady)
        crop = img_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        # preprocess crop
        tmp_path = None
        try:
            # save temp crop and run preprocess_image to reuse existing code
            tmpf = NamedTemporaryFile(delete=False, suffix=".png")
            cv2.imwrite(tmpf.name, crop)
            tmpf.close()
            res = preprocess_image(tmpf.name)
            cleaned = res["cleaned"]
        finally:
            try:
                os.remove(tmpf.name)
            except Exception:
                pass

        if show_debug:
            st.image(cleaned, width=400, caption=f"Lead {i+1} cleaned")

        # extract trace
        xs, ys, skel = extract_trace(cleaned)
        if len(xs) == 0:
            st.warning(f"Lead {i+1}: no trace found (skipping)")
            continue

        px_grid = detect_grid_spacing(res["enhanced"])
        if px_grid is None:
            px_grid = 6

        t, v, fs = pixels_to_signal(xs, ys, int(px_grid), sampling_rate=int(sampling_rate), paper_speed_mm_s=float(paper_speed_mm_s))

        if apply_smoothing:
            try:
                from scipy.signal import medfilt
                v = medfilt(v, kernel_size=3)
            except Exception:
                pass

        # compute HR
        hr_bpm, peaks = heart_rate_from_signal(v, fs)

        # save CSV per-lead
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lead_name = f"lead_{i+1}_{timestamp}.csv"
        out_path = os.path.join(ROOT_DIR, "results", lead_name)
        save_csv(t, v, out_path)
        lead_files.append(out_path)

        meta = {
            "lead_index": i+1,
            "duration_s": float(t.max() - t.min()) if len(t)>1 else 0.0,
            "sampling_rate": int(fs),
            "hr_bpm": float(hr_bpm) if hr_bpm else None,
            "csv_file": lead_name
        }
        all_meta.append(meta)

        # show small plot and metadata
        st.subheader(f"Lead {i+1} — metadata")
        st.write(meta)
        fig, ax = plt.subplots(figsize=(8,2.2))
        ax.plot(t, v, linewidth=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (mV)")
        ax.set_xlim(t.min() if len(t)>0 else 0, t.max() if len(t)>0 else 1)
        st.pyplot(fig)

    if not lead_files:
        st.error("No leads were successfully processed.")
        st.stop()

    # zip all lead CSVs
    zip_name = f"results/leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = zip_files(lead_files, os.path.join(ROOT_DIR, zip_name))
    with open(zip_path, "rb") as f:
        st.download_button("Download all leads (zip)", data=f, file_name=os.path.basename(zip_path), mime="application/zip")

    st.success(f"Processed {len(lead_files)} leads. Individual CSVs saved in results/")

    # show full JSON metadata
    st.json(all_meta)
else:
    st.info("Upload a full ECG sheet image to extract all leads automatically.")
