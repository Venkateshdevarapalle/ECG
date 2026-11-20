# src/utils.py
import pandas as pd
import os
import zipfile

def save_csv(time, voltage, out_path):
    df = pd.DataFrame({"time": time, "voltage": voltage})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path

def zip_files(file_list, zip_path):
    os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in file_list:
            z.write(f, arcname=os.path.basename(f))
    return zip_path
