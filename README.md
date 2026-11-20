# ECG
# 🫀 ECG Image Digitizer

A complete system to convert scanned / photographed ECG paper reports into **digital ECG signals** (CSV/WFDB), with preprocessing, grid removal, trace extraction, signal reconstruction, quality scoring, visualization, and a Streamlit-based web app.

This project uses **real ECG images** (Mendeley), **digital ECG signals** (MIT-BIH, PTB-XL), and **synthetic WFDB signals** for robust validation.

---
## 🚀 Features

- Upload any ECG paper image → get **clean digital signal**
- Automatic preprocessing:
  - Grid detection + removal
  - Contrast enhancement
- Trace extraction (skeletonization + fallback contour extraction)
- Pixel-to-signal conversion:
  - Time axis calibration
  - Voltage scaling (mV)
- Signal smoothing & reconstruction
- Heart rate estimation (BPM)
- Quality score calculation
- **CSV output** (`time, voltage`)
- Streamlit UI for end-to-end demo
- Supports custom HTML templates

---
## 📁 Project Structure

ECG-Digitizer/
│
├── data/
│ ├── images/ # ECG paper images (gwbz3fsgp8 dataset)
│ ├── ptbxl/ # PTB-XL (WFDB or CSV)
│ ├── mitbih/ # MIT-BIH & other CSV signal datasets
│ ├── synthetic_wfdb/ # rrest-syn WFDB signals (optional)
│ └── metadata/ # additional metadata files
│
├── src/
│ ├── preprocessing.py
│ ├── trace_extraction.py
│ ├── signal_reconstruction.py
│ ├── evaluation.py
│ └── utils.py
│
├── app/
│ ├── streamlit_app.py
│ └── template.html # custom UI (optional)
│
├── results/
│ └── extracted_signal.csv
│
├── notebooks/
│ └── validation.ipynb # optional accuracy evaluation
│
├── run_test.py
├── requirements.txt
└── README.md

yaml
Copy code

---
## 🛠️ Installation

```bash
python -m venv venv
source venv/bin/activate     # Windows → venv\Scripts\activate
pip install -r requirements.txt
📥 Add Datasets
Place files as follows:

kotlin
Copy code
data/images/                 ← Mendeley ECG images (gwbz3fsgp8)
data/ptbxl/                  ← PTB-XL signals
data/mitbih/                 ← MIT-BIH & other CSV signal datasets
data/synthetic_wfdb/         ← rrest-syn synthetic WFDB signals
data/metadata/               ← data.json, metadata summary.csv
▶️ Quick Test (CLI)
Run this to test digitization on a single ECG image:

bash
Copy code
python run_test.py
This produces:
bash
Copy code
results/test_extracted.csv
Format:

python-repl
Copy code
time,voltage
0.000,0.12
0.004,0.15
...
🌐 Run Web App (Streamlit)
bash
Copy code
streamlit run app/streamlit_app.py
Features:

Upload ECG image

View cleaned image & skeleton trace

View reconstructed ECG waveform plot

Download CSV

View metadata: duration, sampling rate, HR, quality score

If you add a custom UI template, save it as:
arduino
Copy code
app/template.html
It loads automatically.

📊 Validation (Optional)
Use MIT-BIH/PTB-XL signals to compute:

RMSE (signal shape error)

DTW similarity

HR difference

Example notebook stored in:

bash
Copy code
notebooks/validation.ipynb
📌 Notes
Grid detection is heuristic — adjust thresholds for different images.

Median filtering smooths noisy signals; disable in sidebar if needed.

For advanced ECG analysis, use Pan–Tompkins (biosppy ECG module).

✔️ Output Summary
Your web app returns:

extracted_signal.csv

Waveform plot (PNG)

Metadata JSON:

json
Copy code
{
    "duration_s": 2.45,
    "sampling_rate": 360,
    "hr_bpm": 76.4,
    "quality_score": 87
}
🏁 End
This project converts real-world ECG paper images into usable digital signals suitable for ML models, analysis, and clinical research.

vbnet
Copy code

---

# **2️⃣ What is `run_test.py` and what do YOU do with it?**  
You asked:

> what is these  
> what i want to do with it

### 🔍 **What is run_test.py?**  
It is a **simple command-line script** designed to:

- Take **one sample ECG image**  
- Run your full pipeline **without** Streamlit  
- Output a CSV file  

This is mainly for:

- Testing  
- Debugging  
- Verifying your code works before running the web app  

### 🧠 **Why do we need it?**

Because the Streamlit app is larger — if anything breaks, it’s easier to debug in a **small script**, not the full web UI.

### ✔️ **What YOU do with run_test.py:**

1. Go to your project root.  
2. Put a test ECG image like:

data/images/example1.jpg

markdown
Copy code

3. Run:

python run_test.py

markdown
Copy code
4. It will generate:

results/test_extracted.csv

markdown
Copy code

5. Open the CSV — you will see:

time,voltage
0.000,0.15
0.004,0.17
...

markdown
Copy code

That’s all!  
It is just a quick **debug/test tool**.

---

# ✅ If you want, I can also generate:
- A **premium advanced README** with screenshots & dataset citations  
- A **project logo/banner**  
- A **final GitHub-ready ZIP** with all folders pre-created  
- A **Dockerfile** for deployment  
- A **colab notebook** for validation  

Just tell me,  
👉 **“Give me GitHub-ready package”**  
or  
👉 **“Add screenshots to README”**
