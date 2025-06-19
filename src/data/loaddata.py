import os
import glob
import random
from functions import printandsave
from sklearn.model_selection import KFold

def get_train_val_test_files_5fold_planning(dataset, data_dir, train_rt, val_rt, test_rt, output_dir, seg, modified=False, num_folds=5):

    folder_name = '*'

    # Get patient directories
    patients = sorted(glob.glob(os.path.join(data_dir, folder_name)))
    if not patients:
        raise ValueError(f"No patients found in directory: {os.path.join(data_dir, folder_name)}")

    filtered_patients = []

    for patient in patients:
        ct_path = os.path.join(patient, 'CT-Plan-image.nii.gz')  # assuming mask folder is where CTs are kept
        gtv_path = os.path.join(patient, 'CT-Plan-mask-clinical-gtv.nii.gz')  # assuming image folder is where GTVs are kept

        # Check if both CT and GTV files exist in the directory
        if os.path.exists(ct_path) and os.path.exists(gtv_path):
            filtered_patients.append(patient)

    patients = filtered_patients

    seed = 42
    random.seed(seed)
    random.shuffle(patients)

    numOfPatients = len(patients)
    print(f"Total number of patients: {numOfPatients}")
    
    def get_files(patient_list):
        masks, images, patient_ids = [], [], []
        for patient in patient_list:
            patient_id = os.path.basename(patient)
            image_np = os.path.join(patient, 'CT-Plan-image.nii.gz')
            label_dir = os.path.join(patient, 'CT-Plan-mask-clinical-gtv.nii.gz')
            masks.append(label_dir)
            images.append(image_np)
            patient_ids.append(patient_id)
        return masks, images, patient_ids

    test_num = int(numOfPatients * test_rt)
    val_num = int(numOfPatients * val_rt)
    train_num = numOfPatients - test_num - val_num

    test_patients = patients[train_num + val_num:]  # Use the last part of the data as test (same as old code)
    remaining_patients = patients[:train_num + val_num]  # Remaining patients for train/validation
    
    test_masks, test_images, test_ids = get_files(test_patients)
    test_files = [{"image": img, "mask": msk, "patient_id": pid} for img, msk, pid in zip(test_images, test_masks, test_ids)]

    print(f"Fixed Test Patients: {len(test_patients)}")

    # If only one fold, use train/val/test ratios
    if num_folds == 1:

        fold_dir = os.path.join(output_dir, f"fold_1")
        os.makedirs(fold_dir, exist_ok=True)

        train_patients = remaining_patients[:train_num]
        val_patients = remaining_patients[train_num:]

        train_masks, train_images, train_ids = get_files(train_patients)
        val_masks, val_images, val_ids = get_files(val_patients)

        train_files = [{"image": img, "mask": msk, "patient_id": pid} for img, msk, pid in zip(train_images, train_masks, train_ids)]
        val_files = [{"image": img, "mask": msk, "patient_id": pid} for img, msk, pid in zip(val_images, val_masks, val_ids)]

        print(f"1-Fold Split:")
        print(f"  Training Patients: {len(train_patients)}")
        print(f"  Validation Patients: {len(val_patients)}")
        print(f"  Test Patients: {len(test_patients)}")


        return [{
            "fold": 1,
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files,
        }]

    # For multiple folds, use KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    fold_splits = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(remaining_patients)):
        # Create fold-specific directory
        fold_dir = os.path.join(output_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)

        # Split train and validation patients
        train_patients = [remaining_patients[i] for i in train_idx]
        val_patients = [remaining_patients[i] for i in val_idx]

        train_masks, train_images, train_ids = get_files(train_patients)
        val_masks, val_images, val_ids = get_files(val_patients)

        train_files = [{"image": img, "mask": msk, "patient_id": pid} for img, msk, pid in zip(train_images, train_masks, train_ids)]
        val_files = [{"image": img, "mask": msk, "patient_id": pid} for img, msk, pid in zip(val_images, val_masks, val_ids)]

        fold_splits.append({
            "fold": fold + 1,
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files,  # Same test set across all folds
        })

        # Print the number of patients in each split
        print(f"Fold {fold + 1}:")
        print(f"  Training Patients: {len(train_patients)}")
        print(f"  Validation Patients: {len(val_patients)}")
        print(f"  Test Patients: {len(test_patients)}")

    #print(fold_splits)
    return fold_splits


def get_train_val_test_files_5fold_planning_withlungmask(dataset, data_dir, train_rt, val_rt, test_rt, output_dir, seg, modified=False, num_folds=5, existing_lung_mask=False):

    folder_name = '*'

    # Get patient directories
    patients = sorted(glob.glob(os.path.join(data_dir, folder_name)))
    if not patients:
        raise ValueError(f"No patients found in directory: {os.path.join(data_dir, folder_name)}")

    filtered_patients = []

    for patient in patients:
        ct_path = os.path.join(patient, 'CT-Plan-image.nii.gz')  # assuming mask folder is where CTs are kept
        gtv_path = os.path.join(patient, 'CT-Plan-mask-clinical-gtv.nii.gz')  # assuming image folder is where GTVs are kept
        #lung_path = os.path.join(patient, 'CT-Plan-mask-clinical-rlung_llung.nii.gz')
        if existing_lung_mask:
            lung_path = os.path.join(patient, 'lung-mask-manual.nii.gz')  # assuming image folder is where lung masks are kept
        else:
            lung_path = os.path.join(patient, 'lung-mask.nii.gz')  # assuming image folder is where lung masks are kept
        # Check if both CT and GTV files exist in the directory
        if os.path.exists(ct_path) and os.path.exists(gtv_path):
            filtered_patients.append(patient)

    patients = filtered_patients

    seed = 42
    random.seed(seed)
    random.shuffle(patients)

    numOfPatients = len(patients)
    print(f"Total number of patients: {numOfPatients}")
    
    def get_files(patient_list):
        masks, images, lungs, patient_ids = [], [], [], []
        for patient in patient_list:
            patient_id = os.path.basename(patient)
            image_np = os.path.join(patient, 'CT-Plan-image.nii.gz')
            label_dir = os.path.join(patient, 'CT-Plan-mask-clinical-gtv.nii.gz')
            if existing_lung_mask:
                print("Using existing_lung_mask...")
                lung_dir = os.path.join(patient, 'lung-mask-manual.nii.gz')
            else:
                lung_dir = os.path.join(patient, 'lung-mask.nii.gz')
            #lung_dir = os.path.join(patient, 'CT-Plan-mask-clinical-rlung_llung.nii.gz')
            masks.append(label_dir)
            images.append(image_np)
            lungs.append(lung_dir)
            patient_ids.append(patient_id)
        return masks, images, lungs, patient_ids

    test_num = int(numOfPatients * test_rt)
    val_num = int(numOfPatients * val_rt)
    train_num = numOfPatients - test_num - val_num

    test_patients = patients[train_num + val_num:]  # Use the last part of the data as test (same as old code)
    remaining_patients = patients[:train_num + val_num]  # Remaining patients for train/validation
    
    test_masks, test_images, test_lungs, test_ids = get_files(test_patients)
    test_files = [{"image": img, "mask": msk, "lung":lng, "patient_id": pid} for img, msk, lng, pid in zip(test_images, test_masks, test_lungs, test_ids)]

    print(f"Fixed Test Patients: {len(test_patients)}")

    # If only one fold, use train/val/test ratios
    if num_folds == 1:

        fold_dir = os.path.join(output_dir, f"fold_1")
        os.makedirs(fold_dir, exist_ok=True)

        train_patients = remaining_patients[:train_num]
        val_patients = remaining_patients[train_num:]

        train_masks, train_images, train_lungs, train_ids = get_files(train_patients)
        val_masks, val_images, val_lungs, val_ids = get_files(val_patients)

        train_files = [{"image": img, "mask": msk, "lung":lng, "patient_id": pid} for img, msk, lng, pid in zip(train_images, train_masks, train_lungs, train_ids)]
        val_files = [{"image": img, "mask": msk, "lung":lng, "patient_id": pid} for img, msk, lng, pid in zip(val_images, val_masks, val_lungs, val_ids)]

        print(f"1-Fold Split:")
        print(f"  Training Patients: {len(train_patients)}")
        print(f"  Validation Patients: {len(val_patients)}")
        print(f"  Test Patients: {len(test_patients)}")


        return [{
            "fold": 1,
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files,
        }]

    # For multiple folds, use KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    fold_splits = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(remaining_patients)):
        # Create fold-specific directory
        fold_dir = os.path.join(output_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)

        # Split train and validation patients
        train_patients = [remaining_patients[i] for i in train_idx]
        val_patients = [remaining_patients[i] for i in val_idx]

        train_masks, train_images, train_lungs, train_ids = get_files(train_patients)
        val_masks, val_images, val_lungs, val_ids = get_files(val_patients)

        train_files = [{"image": img, "mask": msk, "lung": lng, "patient_id": pid} for img, msk, lng, pid in zip(train_images, train_masks, train_lungs, train_ids)]
        val_files = [{"image": img, "mask": msk, "lung": lng, "patient_id": pid} for img, msk, lng, pid in zip(val_images, val_masks, val_lungs, val_ids)]

        fold_splits.append({
            "fold": fold + 1,
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files,  # Same test set across all folds
        })

        # Print the number of patients in each split
        print(f"Fold {fold + 1}:")
        print(f"  Training Patients: {len(train_patients)}")
        print(f"  Validation Patients: {len(val_patients)}")
        print(f"  Test Patients: {len(test_patients)}")

    #print(fold_splits)
    return fold_splits

