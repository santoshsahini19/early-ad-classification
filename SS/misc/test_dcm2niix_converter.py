import os
import subprocess
from pathlib import Path
import nibabel as nib

# folder that contains DICOM files and output folder for NIfTI files
dicom_folder = Path(r"D:\projects\research\HighResHippocampus\ADNI\002_S_0413\HighResHippocampus\2017-06-21_13_23_38.0\I863061")
output_folder = Path(r"D:\projects\research\Converted_NIfTI")

# Create the output folder if it doesn't exist
output_folder.mkdir(parents=True, exist_ok=True)

# command to run dcm2niix 
cmd = [
    "dcm2niix",   # This will work if dcm2niix is in your PATH
    "-z", "y",     # compress the output (.nii.gz)
    "-f", "011_S_7112",  # output filename
    "-o", str(output_folder),  # where to save
    str(dicom_folder)  # input folder containing DICOMs
]

print("Running command:")
print(" ".join(cmd))
# execute the command which converts DICOM to NIfTI
result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

print("\n--- dcm2niix output ---")
print(result.stdout)
print(result.stderr)

# check if a .nii.gz file was created
nifti_files = list(output_folder.glob("*.nii*"))
if len(nifti_files) == 0:
    print("No NIfTI file created. Check paths or dcm2niix installation.")
else:
    print(f"NIfTI file created: {nifti_files[0]}")

    # check its properties of the created NIfTI file
    img = nib.load(str(nifti_files[0]))
    print("Shape:", img.shape)
    print("Voxel sizes:", img.header.get_zooms())