# src/evaluation.py
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    n = min(len(a), len(b))
    if n == 0:
        return None
    return np.sqrt(np.mean((a[:n] - b[:n])**2))

def correlation(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    n = min(len(a), len(b))
    if n < 2:
        return None
    return np.corrcoef(a[:n], b[:n])[0,1]

def dtw_distance(a, b):
    distance, _ = fastdtw(a, b, dist=euclidean)
    return distance

def pan_tompkin_rpeaks(sig, fs):
    """
    Simplified Pan-Tompkins style pipeline:
    - bandpass 5-15 Hz
    - derivative
    - squaring
    - moving average integration
    - peak detection
    Returns R-peaks indices (in samples)
    """
    sig = np.asarray(sig)
    if len(sig) < 10:
        return []

    # bandpass 5-15 Hz
    nyq = 0.5 * fs
    low = 5.0 / nyq
    high = 15.0 / nyq
    try:
        b, a = butter(2, [low, high], btype='band')
        filt = filtfilt(b, a, sig)
    except Exception:
        filt = sig

    # derivative
    diff = np.diff(filt, prepend=filt[0])
    diff2 = np.abs(diff)

    # squaring
    sq = diff2 ** 2

    # moving average window ~150 ms
    window_len = max(1, int(0.150 * fs))
    w = np.ones(window_len) / window_len
    integr = np.convolve(sq, w, mode='same')

    # dynamic threshold
    thresh = np.mean(integr) + 0.5 * np.std(integr)
    peaks, props = find_peaks(integr, distance=int(0.2*fs), height=thresh)
    # refine peaks on original filtered signal: local maxima within +-40ms
    rpeaks = []
    for p in peaks:
        left = max(0, p - int(0.04*fs))
        right = min(len(filt)-1, p + int(0.04*fs))
        local_max = np.argmax(filt[left:right+1]) + left
        rpeaks.append(int(local_max))
    # unique sorted
    rpeaks = sorted(list(set(rpeaks)))
    return rpeaks

def heart_rate_from_signal(sig, fs):
    rpeaks = pan_tompkin_rpeaks(sig, fs)
    if len(rpeaks) < 2:
        return None, rpeaks
    rr = np.diff(rpeaks) / float(fs)
    hr = 60.0 / np.mean(rr)
    return hr, rpeaks
