import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from lungmask import LMInferer

def generate_and_save_lung_mask(patient_id, base_folder, output_name="lungmask.nii.gz"):
    """
    Generate lung mask for a single patient using LMInferer and save as NIfTI using nibabel.
    """
    ct_path = os.path.join(base_folder, patient_id, "CT-Plan-image.nii.gz")
    output_mask_path = os.path.join(base_folder, patient_id, output_name)

    if not os.path.exists(ct_path):
        print(f"❌ CT file not found for {patient_id}: {ct_path}")
        return

    try:
        # Read CT with SimpleITK (required by LMInferer)
        ct_sitk = sitk.ReadImage(ct_path)
        inferer = LMInferer()
        lung_mask_np = inferer.apply(ct_sitk)  # (z, y, x)
        print(f"✅ Lung mask generated for {patient_id}.")

        # Binary and transpose to (x, y, z)
        lung_mask_np = (lung_mask_np > 0).astype(np.uint8)
        lung_mask_np_t = np.transpose(lung_mask_np, (2, 1, 0))

        # Use original CT for affine/header
        ct_nib = nib.load(ct_path)
        affine = ct_nib.affine
        header = ct_nib.header.copy()

        # Save mask
        lung_mask_nifti = nib.Nifti1Image(lung_mask_np_t, affine=affine, header=header)
        nib.save(lung_mask_nifti, output_mask_path)
        print(f"💾 Lung mask saved: {output_mask_path}")

    except Exception as e:
        print(f"❗ Error processing {patient_id}: {e}")

def process_patients(patient_list, base_folder):
    for patient_id in patient_list:
        print(f"\n🚀 Processing patient: {patient_id}")
        generate_and_save_lung_mask(patient_id, base_folder)

# ---------------------------
# 🎯 Example usage
base_folder = "/home/ilkin/Documents/2024PHD/data/Planning-CTs"
patients = ["2011HURU"]  # Replace with your actual list
process_patients(patients, base_folder)
