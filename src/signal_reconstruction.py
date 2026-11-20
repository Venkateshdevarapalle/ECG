# src/signal_reconstruction.py
import numpy as np
from scipy import signal
from scipy.signal import resample

def detect_grid_spacing(img_gray):
    # measure vertical pattern using column mean and find periodic peaks
    col_sum = np.mean(img_gray, axis=0)
    col_s = signal.medfilt(col_sum, kernel_size=51)
    # invert for grid valleys
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(-col_s, distance=8, prominence=0.5*np.std(col_s))
    if len(peaks) < 2:
        # try smaller kernel
        peaks, _ = find_peaks(-col_sum, distance=6)
    if len(peaks) < 2:
        return None
    spacings = np.diff(peaks)
    px_per_small = int(max(1, np.median(spacings)))
    return px_per_small

def pixels_to_signal(xs, ys, px_per_small_grid, sampling_rate=360, paper_speed_mm_s=25.0):
    if len(xs) == 0 or len(ys) == 0:
        return np.array([0.0]), np.array([0.0]), sampling_rate

    # convert x pixels to time: first normalize xs across range
    px_per_s = paper_speed_mm_s * px_per_small_grid  # px per second
    t = (xs - xs.min()) / px_per_s  # seconds, start at zero

    # voltage mapping: baseline is median of ys
    px_per_mV = px_per_small_grid * 10.0  # heuristic: 10 small boxes = 1mV -> adjust if needed
    baseline = np.median(ys)
    volt_mV = (baseline - ys) / px_per_mV

    # resample to uniform sampling_rate
    duration = t.max() - t.min() if len(t) > 1 else 0.0
    if duration <= 0:
        return t, volt_mV, sampling_rate
    n_samples = max(2, int(round(duration * sampling_rate)))
    t_uniform = np.linspace(t.min(), t.max(), n_samples)
    volt_uniform = np.interp(t_uniform, t, volt_mV)

    # remove slow baseline wander using high-pass (cutoff 0.5 Hz)
    try:
        b, a = signal.butter(2, 0.5/(0.5*sampling_rate), btype='highpass')
        volt_uniform = signal.filtfilt(b, a, volt_uniform)
    except Exception:
        pass

    return t_uniform, volt_uniform, sampling_rate
