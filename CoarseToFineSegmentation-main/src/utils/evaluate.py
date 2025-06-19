#LIBRARIES
import os
import json
import csv
import numpy as np
import nibabel as nib
import torch
from datetime import datetime
from scipy.ndimage import distance_transform_edt
from joblib import Parallel, delayed
from torch.utils.tensorboard import SummaryWriter
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader
import functions
import loaddata
import transforms2
import argparse
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion

# Initialize TensorBoard
writer = SummaryWriter()
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Compute segmentation metrics")
parser.add_argument("--data", type=str, required=True, help="Dataset name (e.g., 'OH-GLLES-3D')")
parser.add_argument("--experiment", type=str, required=True, help="Experiment name (e.g., 's4')")
parser.add_argument("--model_path", type=str, default="best_metric_model.pth", help="Model path (default: 'best_model')")
parser.add_argument("--ROI", type=str, default=False, help="Model path (default: 'False')")
parser.add_argument("--onlydice", type=str, default=False, help="MCalculate only DICE (default: 'False')")
args = parser.parse_args()

experiment = args.experiment  # Get the value from command-line
data = args.data  # Get the value from command-line
model_path = args.model_path
roi = args.ROI
onlydice = args.onlydice
print(roi)

print(f"Running experiment: {experiment} for {data}")

# Experiment setup
#experiment = "s4"
experiment_dir = f"/home/ilkin/Documents/2024PHD/segmentation/swinunetr/tests/{data}/{experiment}"
json_path = f"{experiment_dir}/hyperparameters.json"

print(json_path)
# Load hyperparameters
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        hyperparams = json.load(f)

# Retrieve parameters
lossfunc = functions.get_loss_func(hyperparams["diceCE"])
label_names, numberofclasses = functions.get_classes(hyperparams["dataset"], hyperparams["seg"])
dimension, batchsize = functions.get_dimension_and_bs(hyperparams["dataset"])
dice_roi, aug_roi = functions.get_rois(hyperparams["dataset"], hyperparams["architecture"])

# Update and save hyperparameters
hyperparams.update({
    "experiment": experiment,
    "number_of_classes": numberofclasses,
    "dimension": dimension,
    "batch_size": batchsize,
    "dice_roi": dice_roi,
    "aug_roi": aug_roi,
    "date_time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    "loss_function": f"{lossfunc} ({hyperparams['wdice']}-{hyperparams['wce']})",
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "patience": 10
})

with open(json_path, "w") as f:
    json.dump(hyperparams, f, indent=4)
print("Hyperparameters updated and saved.")

# Set dataset path
data_dir = os.path.join("/data/ayc9699/dataset" if hyperparams["server"] else "/home/ilkin/Documents/2024PHD/data", hyperparams["dataset"])

# Load data splits
fold_splits = loaddata.get_train_val_test_files_5fold_planning(
    dataset=hyperparams["dataset"],
    data_dir=data_dir,
    train_rt=hyperparams["train_rt"],
    val_rt=hyperparams["val_rt"],
    test_rt=hyperparams["test_rt"],
    output_dir=experiment_dir,
    seg=hyperparams["seg"],
    modified=False,
    num_folds=hyperparams["num_folds"],
)

test_transforms = transforms2.getTestTransform(-175, 250)
print(roi)

if roi=="roi":
    print("ROI")
    test_transforms = transforms2.getTestTransformROI(-175, 250)

# Define model
model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=numberofclasses+1,
    feature_size=48,
    use_checkpoint=True,
).to(device)

# Load model
fold_idx = 0
fold_dir = os.path.join(experiment_dir, f"fold_{fold_idx+1}")

if model_path == 'best_metric_model.pth':
    model.load_state_dict(torch.load(os.path.join(fold_dir, model_path)))
else:
    checkpoint = torch.load(os.path.join(fold_dir, model_path))
    model.load_state_dict(checkpoint["model_state_dict"])

# Define test loader
test_loader = DataLoader(
    CacheDataset(data=fold_splits[fold_idx]["test_files"], transform=test_transforms, cache_rate=0.1, num_workers=4),
    batch_size=batchsize, shuffle=False, num_workers=4
)

def save_evaluation_results(eval_loader, model, output_dir, device, eval_files, roi_size, overlap=0.5):
    model.eval()
    with torch.no_grad():
        print("\nStarting evaluation and saving results...\n")
        for i, eval_data in enumerate(eval_loader):
            pt_name = eval_files[i]["image"].split("/")[-2]
            print(f"Processing Patient: {pt_name}")
            
            original_nifti = nib.load(eval_files[i]["image"])
            affine, header = original_nifti.affine, original_nifti.header
            
            eval_inputs = eval_data["image"].to(device)
            eval_outputs = sliding_window_inference(eval_inputs, roi_size, sw_batch_size=4, predictor=model, overlap=overlap)
            
            image, mask = eval_data["image"][0, 0].cpu().numpy(), eval_data["mask"][0, 0].cpu().numpy()
            predicted = torch.argmax(eval_outputs, dim=1).detach().cpu().numpy()[0]
            
            patient_output_dir = os.path.join(output_dir, pt_name)
            os.makedirs(patient_output_dir, exist_ok=True)
            
            nib.save(nib.Nifti1Image(image, affine, header), os.path.join(patient_output_dir, f"{pt_name}_image.nii"))
            nib.save(nib.Nifti1Image(mask, affine, header), os.path.join(patient_output_dir, f"{pt_name}_mask.nii"))
            nib.save(nib.Nifti1Image(predicted.astype(np.int16), affine, header), os.path.join(patient_output_dir, f"{pt_name}_predicted.nii"))
            
            print(f"Results saved for {pt_name} at {patient_output_dir}")

evaluation_results_dir = os.path.join(experiment_dir, f"fold_{fold_idx+1}", "evaluation_results")
if not os.path.exists(evaluation_results_dir):
    print(f"Directory '{evaluation_results_dir}' does not exist. Running evaluation and saving results.")
    save_evaluation_results(test_loader, model, evaluation_results_dir, device, fold_splits[0]["test_files"], (64, 64, 64), 0.5)
else:
    print(f"Directory '{evaluation_results_dir}' already exists. Skipping evaluation.")

def get_all_patients(evaluation_results_dir):
    patients = {}
    for item in os.listdir(evaluation_results_dir):
        patient_dir = os.path.join(evaluation_results_dir, item)
        if os.path.isdir(patient_dir):
            gt_path, pred_path = None, None
            for filename in os.listdir(patient_dir):
                if "_mask.nii" in filename:
                    gt_path = os.path.join(patient_dir, filename)
                elif "_predicted.nii" in filename:
                    pred_path = os.path.join(patient_dir, filename)
            if gt_path and pred_path:
                patients[item] = {"gt": gt_path, "pred": pred_path}
    return patients

def dice_per_class(truth, prediction, class_label):
    truth_class, pred_class = (truth == class_label), (prediction == class_label)
    intersection = np.sum(truth_class & pred_class)
    return round(2.0 * intersection / (np.sum(truth_class) + np.sum(pred_class)) if (np.sum(truth_class) + np.sum(pred_class)) else 1.0,3)

def fast_hd95(truth, prediction, class_label):
    # Binary masks for the class
    truth_mask, pred_mask = (truth == class_label).astype(np.uint8), (prediction == class_label).astype(np.uint8)
    # If both masks are empty, define HD95 as 0 (no boundary error)
    if np.sum(truth_mask) == 0 and np.sum(pred_mask) == 0: return 0.0
    if np.sum(truth_mask) == 0 or np.sum(pred_mask) == 0: return np.nan  # No valid boundary for comparison
    # Compute distance maps (Euclidean Distance Transform)
    truth_dist, pred_dist = distance_transform_edt(1 - truth_mask), distance_transform_edt(1 - pred_mask)
    # Get distances from prediction boundary to ground truth
    pred_to_truth, truth_to_pred = np.max(truth_dist[pred_mask > 0]), np.max(pred_dist[truth_mask > 0])
    # HD95 is the 95th percentile of distances
    return round(np.percentile([pred_to_truth, truth_to_pred], 95), 3)

def compute_metrics_old(patients, metric_func, metric_name):
    output_txt = os.path.join(experiment_dir, f"fold_{fold_idx+1}", f"{metric_name}_scores_output.txt")
    output_csv = os.path.join(experiment_dir, f"fold_{fold_idx+1}", f"{metric_name}_scores_output.csv")
    results = {}
    for pt, files in patients.items():
        gt_data, pred_data = nib.load(files["gt"]).get_fdata().astype(np.uint8), nib.load(files["pred"]).get_fdata().astype(np.uint8)
        results[pt] = {f"Background {metric_name}": metric_func(gt_data, pred_data, 0), f"Tumor {metric_name}": metric_func(gt_data, pred_data, 1)}
    
    with open(output_txt, "w") as f_txt:
        for pt, data in results.items():
            f_txt.write(f"{pt}: Background {metric_name} = {data[f'Background {metric_name}']}, Tumor {metric_name} = {data[f'Tumor {metric_name}']}\n")
    
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Patient", f"Background {metric_name}", f"Tumor {metric_name}"])
        for pt, data in results.items():
            writer.writerow([pt, data[f"Background {metric_name}"], data[f"Tumor {metric_name}"]])


# Define Boundary Dice metric
def boundary_dice(truth, prediction, class_label, dilation_radius=1):
    truth_mask = (truth == class_label).astype(np.uint8)
    pred_mask = (prediction == class_label).astype(np.uint8)
    if np.sum(truth_mask) == 0 and np.sum(pred_mask) == 0:
        return 1.0
    if np.sum(truth_mask) == 0 or np.sum(pred_mask) == 0:
        return 0.0
    truth_boundary = binary_dilation(truth_mask, iterations=dilation_radius) ^ binary_erosion(truth_mask, iterations=dilation_radius)
    pred_boundary = binary_dilation(pred_mask, iterations=dilation_radius) ^ binary_erosion(pred_mask, iterations=dilation_radius)
    intersection = np.sum(truth_boundary & pred_boundary)
    union = np.sum(truth_boundary) + np.sum(pred_boundary)
    return round(2.0 * intersection / union if union > 0 else 1.0, 3)

def compute_metrics_old(patients, metric_func, metric_name):
    output_txt = os.path.join(experiment_dir, f"fold_{fold_idx+1}", f"{metric_name}_scores_output.txt")
    output_csv = os.path.join(experiment_dir, f"fold_{fold_idx+1}", f"{metric_name}_scores_output.csv")
    results = {}
    
    for pt, files in patients.items():
        gt_data, pred_data = nib.load(files["gt"]).get_fdata().astype(np.uint8), nib.load(files["pred"]).get_fdata().astype(np.uint8)
        results[pt] = {
            f"Background {metric_name}": metric_func(gt_data, pred_data, 0),
            f"Tumor {metric_name}": metric_func(gt_data, pred_data, 1)
        }
    
    with open(output_txt, "w") as f_txt:
        for pt, data in results.items():
            f_txt.write(f"{pt}: Background {metric_name} = {data[f'Background {metric_name}']}, Tumor {metric_name} = {data[f'Tumor {metric_name}']}\n")
    
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Patient", f"Background {metric_name}", f"Tumor {metric_name}"])
        for pt, data in results.items():
            writer.writerow([pt, data[f"Background {metric_name}"], data[f"Tumor {metric_name}"]])
    
    # Calculate overall mean metric for each class across patients
    all_bg = np.array([results[pt][f"Background {metric_name}"] for pt in results])
    all_tumor = np.array([results[pt][f"Tumor {metric_name}"] for pt in results])
    
    mean_bg = np.round(np.nanmean(all_bg), 3)  # Use nanmean to ignore NaN values
    mean_tumor = np.round(np.nanmean(all_tumor), 3)
    overall_mean = np.round(np.nanmean([mean_bg, mean_tumor]), 3)
    
    print(f"Overall Mean Background {metric_name}:", mean_bg)
    print(f"Overall Mean Tumor {metric_name}:", mean_tumor)
    
    with open(output_txt, "a") as f_txt:
        f_txt.write("\nOverall Scores:\n")
        f_txt.write(f"Overall Mean Background {metric_name}: {mean_bg}\n")
        f_txt.write(f"Overall Mean Tumor {metric_name}: {mean_tumor}\n")    


# Compute and save metrics
def compute_metrics(patients, metric_func, metric_name):
    output_txt = os.path.join(experiment_dir, f"fold_{fold_idx+1}", f"{metric_name}_scores_output.txt")
    output_csv = os.path.join(experiment_dir, f"fold_{fold_idx+1}", f"{metric_name}_scores_output.csv")
    results = {}

    for pt, files in patients.items():
        gt_data, pred_data = nib.load(files["gt"]).get_fdata().astype(np.uint8), nib.load(files["pred"]).get_fdata().astype(np.uint8)
        results[pt] = {
            f"Background {metric_name}": metric_func(gt_data, pred_data, 0),
            f"Tumor {metric_name}": metric_func(gt_data, pred_data, 1)
        }

    with open(output_txt, "w") as f_txt:
        for pt, data in results.items():
            f_txt.write(f"{pt}: Background {metric_name} = {data[f'Background {metric_name}']}, Tumor {metric_name} = {data[f'Tumor {metric_name}']}\n")

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Patient", f"Background {metric_name}", f"Tumor {metric_name}"])
        for pt, data in results.items():
            writer.writerow([pt, data[f"Background {metric_name}"], data[f"Tumor {metric_name}"]])

    # Calculate and save overall mean
    all_bg = np.array([results[pt][f"Background {metric_name}"] for pt in results])
    all_tumor = np.array([results[pt][f"Tumor {metric_name}"] for pt in results])

    mean_bg = np.round(np.nanmean(all_bg), 3)
    mean_tumor = np.round(np.nanmean(all_tumor), 3)

    print(f"Overall Mean Background {metric_name}:", mean_bg)
    print(f"Overall Mean Tumor {metric_name}:", mean_tumor)

    with open(output_txt, "a") as f_txt:
        f_txt.write("\nOverall Scores:\n")
        f_txt.write(f"Overall Mean Background {metric_name}: {mean_bg}\n")
        f_txt.write(f"Overall Mean Tumor {metric_name}: {mean_tumor}\n")    

evaluation_results_dir = os.path.join(experiment_dir, f"fold_{fold_idx+1}", "evaluation_results")
patients = get_all_patients(evaluation_results_dir)

compute_metrics(patients, dice_per_class, "dice")

if onlydice=='False':
    compute_metrics(patients, fast_hd95, "hd95")
    compute_metrics(patients, boundary_dice, "boundary_dice")
