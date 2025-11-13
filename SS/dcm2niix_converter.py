# import required libraries
import os
import subprocess
from pathlib import Path
from tqdm import tqdm

# root and output directories
root_dir = Path(r"D:\projects\research\HighResHippocampus\ADNI")
output_root = Path(r"D:\projects\research\Converted_NIfTI")

output_root.mkdir(parents=True, exist_ok=True)

# get list of subject directories
subject_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
print(f"Found {len(subject_dirs)} subjects.")

# iterate over each subject directory and convert DICOM to NIfTI
for subj_dir in tqdm(subject_dirs):
    # locate the DICOM folder (last level with .dcm files)
    dicom_folder = next(subj_dir.rglob("*.dcm")).parent  # get first dcm, then parent folder
    subj_id = subj_dir.name
    subj_output = output_root / subj_id
    subj_output.mkdir(exist_ok=True)

    cmd = [
        "dcm2niix",
        "-z", "y",
        "-f", subj_id,
        "-o", str(subj_output),
        str(dicom_folder)
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("All conversions done!")