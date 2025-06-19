from monai.transforms import SpatialCropd, Spacingd, ScaleIntensityRanged ,Compose, LoadImaged, CropForegroundd, Orientationd, EnsureChannelFirstd, EnsureTyped
from skimage.measure import label, regionprops
#LIBRARIES
import os
import json
import csv
import pandas as pd
import numpy as np
import nibabel as nib
import torch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset
import functions
import src.data.loaddata as loaddata
import src.data.transforms2 as transforms2
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, binary_dilation, find_objects
from skimage.measure import label
import numpy as np
from skimage.measure import regionprops
from scipy.ndimage import distance_transform_edt
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.measure import regionprops
from monai.networks.nets import UNet, UNETR, SwinUNETR

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Compute before-after postprocessing metrics")
parser.add_argument("--data", type=str, required=True, help="Dataset name (e.g., 'OH-GLLES-3D')")
parser.add_argument("--model_path", type=str, default="best_metric_model.pth", help="Model path (default: 'best_model')")
parser.add_argument("--architecture", type=str, required=True, help="Model arcitecture (e.g., 'swinunetr', 'unet', 'unetr')")
parser.add_argument("--full_model", type=str, required=True, help="FULL model (e.g., 's3')")
parser.add_argument("--roi_model", type=str, required=True, help="ROI model (e.g., 's3', 's4', 's5')")
parser.add_argument("--experiment", type=str, required=True, help="Experiment name (e.g., oh-s4-k1)")
parser.add_argument("--k", type=int, default=None, help="Top k ROIs to be returned")
parser.add_argument("--use_existing_lung_mask", action="store_true", default=False, help="Use existing lung masks if available")
parser.add_argument("--roi_size", type=lambda s: tuple(map(int, s.strip("()").split(","))), default=(64, 64, 64), help="ROI size in the format '(D,H,W)', e.g., '(64,64,64)'")
parser.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap fraction")
parser.add_argument("--fold_idx", type=int, default=0, help="Fold index to use for cross-validation")
args = parser.parse_args()

data = args.data  # Get the value from command-line
model_path = args.model_path
architecture = args.architecture
full_model = args.full_model  # Get the value from command-line
roi_model = args.roi_model  # Get the value from command-line
experiment = args.experiment
k = args.k
use_existing_lung_mask = args.use_existing_lung_mask
roi_size = args.roi_size
print(roi_size)
overlap = args.overlap
fold_idx = args.fold_idx
if data=='OH-GLLES-3D':
    data_dir = os.path.join("/home/ilkin/Documents/2024PHD/data", "Planning-CTs")

post_label = AsDiscrete(to_onehot=1+1)
post_pred = AsDiscrete(argmax=True, to_onehot=1+1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_segmentation_pipeline_with_roi_unc(ct_image, gt_mask, pred_initial, pred_post, roi_ct, uncertainty_map, bbox, save_path, title=None):
    """
    Creates a 5-panel visualization:
    1. Full CT
    2. Full GT overlay
    3. Full initial prediction
    4. Full post-processed prediction
    5. Zoomed-in ROI + Uncertainty
    """
    z = get_middle_tumor_slice(gt_mask)

    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    cmap_mask = plt.cm.Reds
    cmap_unc = plt.cm.viridis

    axs[0].imshow(ct_image[:, :, z], cmap='gray')
    axs[0].set_title("CT Image")
    axs[0].axis('off')

    axs[1].imshow(ct_image[:, :, z], cmap='gray')
    axs[1].imshow(gt_mask[:, :, z], cmap=cmap_mask, alpha=0.5)
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')

    axs[2].imshow(ct_image[:, :, z], cmap='gray')
    axs[2].imshow(pred_initial[:, :, z], cmap=cmap_mask, alpha=0.5)
    axs[2].set_title("Initial Prediction")
    axs[2].axis('off')

    axs[3].imshow(ct_image[:, :, z], cmap='gray')
    axs[3].imshow(pred_post[:, :, z], cmap=cmap_mask, alpha=0.5)
    axs[3].set_title("Post-Processed")
    axs[3].axis('off')

    # Last panel: Zoomed-in ROI uncertainty
    y1, y2, x1, x2, z1, z2 = bbox
    z_roi = get_middle_tumor_slice(uncertainty_map)

    axs[4].imshow(roi_ct[:, :, z_roi], cmap='gray')
    im = axs[4].imshow(uncertainty_map[:, :, z_roi], cmap=cmap_unc, alpha=0.6, vmin=0.0, vmax=0.2)
    axs[4].set_title("Zoomed ROI + Uncertainty")
    axs[4].axis('off')

    cbar = fig.colorbar(im, ax=axs[4], fraction=0.046, pad=0.04)
    cbar.set_label("Uncertainty (Variance)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Figure saved to {save_path}")

def compute_uncertainty_map(model, input_tensor, roi_size, n_iter=10):
    model.train()  # Enable dropout
    predictions = []

    with torch.no_grad():
        for _ in range(n_iter):
            output = sliding_window_inference(input_tensor, roi_size, sw_batch_size=1, predictor=model)
            pred = torch.softmax(output, dim=1).cpu().numpy()[0, 1]  # Take softmax prob of tumor class
            predictions.append(pred)

    model.eval()
    stacked = np.stack(predictions, axis=0)
    return np.var(stacked, axis=0)  # Variance over dropout runs

def get_middle_tumor_slice(mask):
    """Find the middle slice of the tumor region."""
    #print(mask.shape)
    tumor_slices = np.where(mask.sum(axis=(0, 1)) > 0)[0]  # Find slices containing tumor
    if len(tumor_slices) > 0:
        return tumor_slices[len(tumor_slices) // 2]  # Return middle tumor slice
    return mask.shape[2] // 2  # Default to middle slice if no tumor detected

def setup_json(experiment_dir):
    json_path = f"{experiment_dir}/hyperparameters.json"
    # Load hyperparameters
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            hyperparams = json.load(f)
    # Retrieve parameters
    lossfunc = functions.get_loss_func(hyperparams["diceCE"])
    label_names, numberofclasses = functions.get_classes(hyperparams["dataset"], hyperparams["seg"])
    dimension, batchsize = functions.get_dimension_and_bs(hyperparams["dataset"])
    dice_roi, aug_roi = functions.get_rois(hyperparams["dataset"], hyperparams["architecture"])
    train_rt, val_rt, test_rt = hyperparams["train_rt"], hyperparams["val_rt"], hyperparams["test_rt"]
    return lossfunc, label_names, dimension, batchsize, dice_roi, aug_roi, train_rt, val_rt, test_rt, hyperparams["seg"], hyperparams["num_folds"]

def get_dataloader(fold_splits, batchsize):
    test_transforms = transforms2.getTestTransform_withlungmask(-175, 250)
    return DataLoader(
        CacheDataset(data=fold_splits[fold_idx]["test_files"], transform=test_transforms, cache_rate=0.1, num_workers=4),
        batch_size=batchsize, shuffle=False, num_workers=4
    )

def setup(architecture, experiment, existing_lung_mask=False, roi='full'):
    experiment_dir = f"/home/ilkin/Documents/2024PHD/segmentation/swinunetr/tests/{data}/{architecture}/{experiment}"
    evaluation_results_dir = os.path.join(experiment_dir, f"fold_{fold_idx+1}", "evaluation_results_temp")
    fold_dir = os.path.join(experiment_dir, f"fold_{fold_idx+1}")
    lossfunc, label_names, dimension, batchsize, dice_roi, aug_roi, train_rt, val_rt, test_rt, seg, num_folds = setup_json(experiment_dir)
    print(f"Running experiment: {experiment} for {data}")

    # Load data splits
    fold_splits = loaddata.get_train_val_test_files_5fold_planning_withlungmask(
        dataset=data,
        data_dir=data_dir,
        train_rt=train_rt,
        val_rt=val_rt,
        test_rt=test_rt,
        output_dir=experiment_dir,
        seg=seg,
        modified=False,
        num_folds=num_folds,
        existing_lung_mask=existing_lung_mask,
    )
    test_loader = get_dataloader(fold_splits, batchsize)
    model = get_model(architecture, roi)
    model.load_state_dict(torch.load(os.path.join(fold_dir, model_path)))
    return experiment_dir, evaluation_results_dir, fold_splits, test_loader, model

def visualize_and_returnbb(image, mask, predicted, patient_output_dir, pt_name, slice_index=None):
    # If no specific slice index is given, default to middle tumor slice
    z_gt = get_middle_tumor_slice(mask) if slice_index is None else slice_index
    #z_predicted = get_middle_tumor_slice(predicted) if slice_index is None else slice_index

    # Make sure predicted is binary
    predicted_mask = (predicted > 0).astype(np.uint8)

    # Get 3D bounding box (no need to transpose since all inputs are cropped consistently)
    bbox = find_objects(predicted_mask)

    if bbox and bbox[0] is not None:
        y_slice, x_slice, z_slice = bbox[0]
        z1, z2 =  z_slice.start, z_slice.stop
        y1, y2 = y_slice.start, y_slice.stop
        x1, x2 = x_slice.start, x_slice.stop
    else:
        z1 = z2 = y1 = y2 = x1 = x2 = None

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    center_z = (z1 + z2) // 2

    size_x = x2 - x1
    size_y = y2 - y1
    size_z = z2 - z1

    print(f"Bounding Box:")
    print(f"  X (width)  → x1: {x1}, x2: {x2}, size: {size_x}")
    print(f"  Y (height) → y1: {y1}, y2: {y2}, size: {size_y}")
    print(f"  Z (depth)  → z1: {z1}, z2: {z2}, size: {size_z}")
    print(f"Center: (x={center_x}, y={center_y}, z={center_z})")

    '''
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image[:, :, z_gt], cmap="gray")
    axs[0].set_title(f"CT - Slice {z_gt} (shape: {image[:, :, z_gt].shape})")
    axs[0].set_xlabel("X-axis (width)")
    axs[0].set_ylabel("Y-axis (height)")
    if z1 is not None and z1 <= z_gt < z2:
        axs[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', lw=2))

    axs[1].imshow(image[:, :, z_gt], cmap="gray")  # CT as background
    axs[1].imshow(mask[:, :, z_gt], cmap="Reds", alpha=0.5)  # GT mask overlay
    axs[1].set_title(f"GT - Slice {z_gt} (shape: {mask[:, :, z_gt].shape})")
    axs[1].set_xlabel("X-axis (width)")
    axs[1].set_ylabel("Y-axis (height)")

    axs[2].imshow(image[:, :, z_gt], cmap="gray")  # CT as background
    axs[2].imshow(predicted[:, :, z_predicted], cmap="Reds", alpha=0.5)  # GT mask overlay
    axs[2].set_title(f"Prediction - Slice {z_predicted} (shape: {predicted[:, :, z_predicted].shape})")
    axs[2].set_xlabel("X-axis (width)")
    axs[2].set_ylabel("Y-axis (height)")
    if z1 is not None and z1 <= z_predicted < z2:
        axs[2].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='lime', facecolor='none', lw=2))
        axs[2].axhline(y=(y1 + y2) // 2, color='cyan', ls='--')
        axs[2].axvline(x=(x1 + x2) // 2, color='cyan', ls='--')

    plt.tight_layout()
    plt.savefig(os.path.join(patient_output_dir, f"{pt_name}_vis_slice{z_gt}.png"))
    plt.show()
    
    print(f"✅ Saved visualization at slice {z_gt} to {pt_name}_vis_slice{z_gt}.png")
    '''
    return x1,x2,y1,y2, z1,z2

def apply_lung_mask(pred_mask, lung_mask):
    """
    Apply lung mask to remove false positive predictions outside the lung.
    """
    return pred_mask * lung_mask

def get_valid_tumor_components_old(pred_mask, lung_mask, dilation_kernel=(2, 2, 2), lung_overlap_threshold=0.8, top_k=None):
    """
    Merges close components, filters out those mostly outside the lung, and returns remaining component masks.
    Optionally keeps only the top_k largest valid components.

    Args:
        pred_mask (np.ndarray): 3D binary prediction mask.
        lung_mask (np.ndarray): 3D binary lung mask.
        dilation_kernel (tuple): Structure for binary dilation.
        lung_overlap_threshold (float): Minimum percentage of voxels that must be inside the lung.
        top_k (int or None): If set, return only the top_k largest components after filtering.

    Returns:
        List of component masks (each a 3D numpy array of 0s and 1s), and total original count.
    """
    initial_num_components = label(pred_mask, return_num=True)[1]
    merged = binary_dilation(pred_mask, structure=np.ones(dilation_kernel)).astype(np.uint8)
    labeled, num_components = label(merged, return_num=True)

    print(f"🔍 Found {num_components} merged components.")

    if not use_existing_lung_mask:
        print("🫁 Dilating lung mask before filtering...")
        lung_mask = binary_dilation(lung_mask, structure=np.ones((3, 3, 3))).astype(np.uint8)

    valid_components = []
    for region in regionprops(labeled):
        comp_mask = (labeled == region.label).astype(np.uint8)
        total_voxels = np.sum(comp_mask)
        lung_voxels = np.sum(comp_mask * lung_mask)
        inside_ratio = lung_voxels / total_voxels if total_voxels > 0 else 0

        print(f"Component {region.label} → Total voxels: {total_voxels}, Inside lung: {lung_voxels} ({inside_ratio*100:.1f}%)")

        if inside_ratio >= lung_overlap_threshold:
            valid_components.append((comp_mask, total_voxels))
        else:
            print(f"❌ Component {region.label} removed (below threshold).")

    print(f"✅ {len(valid_components)} components kept after lung filtering.")

    # If top_k is specified, return only the largest ones
    if top_k is not None and len(valid_components) > top_k:
        valid_components = sorted(valid_components, key=lambda x: x[1], reverse=True)[:top_k]

    # Remove voxel count, return only masks
    valid_component_masks = [c[0] for c in valid_components]
    return valid_component_masks, initial_num_components, num_components, len(valid_components)

def get_valid_tumor_components(
    pred_mask,
    lung_mask,
    dilation_kernel=(2, 2, 2),
    lung_overlap_threshold=0.8,
    top_k=None,
    min_voxel_threshold=50,
    distance_threshold=5,
    use_existing_lung_mask=False
):
    """
    Filters predicted tumor components by lung overlap, size, and distance to lung.
    Enforces stricter overlap for components in central XY region (mediastinal zone).

    Returns:
        - List of binary masks for valid components.
        - Original component count before merging.
        - Count after merging.
        - Final kept component count.
    """
    initial_num_components = label(pred_mask, return_num=True)[1]

    # Merge nearby regions with dilation
    merged = binary_dilation(pred_mask, structure=np.ones(dilation_kernel)).astype(np.uint8)
    labeled, num_components = label(merged, return_num=True)
    print(f"🔍 Found {num_components} merged components.")

    # Dilate lung if not GT
    if not use_existing_lung_mask:
        print("🫁 Dilating lung mask before filtering...")
        lung_mask = binary_dilation(lung_mask, structure=np.ones((3, 3, 3))).astype(np.uint8)

    # Distance to lung
    distance_to_lung = distance_transform_edt(1 - lung_mask)

    # --- Define central XY zone based on lung mask bounding box
    lung_coords = np.argwhere(lung_mask == 1)
    if lung_coords.size == 0:
        print("⚠️ Empty lung mask detected. Skipping all filtering.")
        return [], initial_num_components, num_components, 0

    y_min, x_min = np.min(lung_coords[:, 1]), np.min(lung_coords[:, 2])
    y_max, x_max = np.max(lung_coords[:, 1]), np.max(lung_coords[:, 2])

    x_center_min = int(x_min + 0.25 * (x_max - x_min))
    x_center_max = int(x_min + 0.75 * (x_max - x_min))
    print(x_center_min, x_center_max)
    valid_components = []
    for region in regionprops(labeled):
        comp_mask = (labeled == region.label).astype(np.uint8)
        total_voxels = np.sum(comp_mask)
        lung_voxels = np.sum(comp_mask * lung_mask)
        inside_ratio = lung_voxels / total_voxels if total_voxels > 0 else 0
        min_distance = np.min(distance_to_lung[comp_mask == 1])

        # Component center in X/Y
        comp_coords = np.argwhere(comp_mask == 1)
        comp_center_y = np.mean(comp_coords[:, 1])
        comp_center_x = np.mean(comp_coords[:, 2])
        x_vals = comp_coords[:, 2]
        y_vals = comp_coords[:, 1]
        x_min_comp, x_max_comp = int(np.min(x_vals)), int(np.max(x_vals))
        y_min_comp, y_max_comp = int(np.min(y_vals)), int(np.max(y_vals))

        in_central_xy = (
            x_center_min <= comp_center_x <= x_center_max
        )

        print(f"Component {region.label} → Voxels: {total_voxels}, In-lung: {inside_ratio:.2f}, "
            f"MinDist: {min_distance:.2f}, InCentralXY: {in_central_xy}, "
            f"X: [{x_min_comp}-{x_max_comp}], Y: [{y_min_comp}-{y_max_comp}]")

        # Reject if 100% outside lung
        if lung_voxels == 0:
            print(f"🚫 Component {region.label} rejected — 100% outside lung.")
            continue

        # Enforce stricter rule in central XY zone
        if in_central_xy and inside_ratio < lung_overlap_threshold:
            print(f"❌ Component {region.label} is in central XY zone but not enough overlap — rejected.")
            continue

        # Accept if inside ratio is high enough
        if inside_ratio >= lung_overlap_threshold:
            valid_components.append((comp_mask, total_voxels))
            continue

        # Allow peripheral ones near lung
        if total_voxels >= min_voxel_threshold and min_distance <= distance_threshold:
            print(f"⚠️ Component {region.label} mostly outside lung but plausible — keeping.")
            valid_components.append((comp_mask, total_voxels))
        else:
            print(f"❌ Component {region.label} rejected — not enough overlap or too far from lung.")

    print(f"✅ {len(valid_components)} components kept after filtering.")

    # Keep top-k largest if needed
    if top_k is not None and len(valid_components) > top_k:
        valid_components = sorted(valid_components, key=lambda x: x[1], reverse=True)[:top_k]

    return [c[0] for c in valid_components], initial_num_components, num_components, len(valid_components)

def fast_dice(gt, pred, class_label=1):
    gt_bin = (gt == class_label).astype(np.uint8)
    pred_bin = (pred == class_label).astype(np.uint8)
    intersection = np.sum(gt_bin & pred_bin)
    denominator = np.sum(gt_bin) + np.sum(pred_bin)
    return 2.0 * intersection / denominator if denominator > 0 else 1.0

def add_canny_edges(pred_mask, ct_image, dilation_kernel=(2, 2, 2)):
    """
    Apply Sobel-based edge detection and refine prediction mask.
    """
    pred_mask = (pred_mask > 0).astype(np.uint8)  # Ensure binary

    # Apply 3D Sobel edge detection
    sobel_x = sobel(ct_image, axis=1)
    sobel_y = sobel(ct_image, axis=2)
    edges = np.hypot(sobel_x, sobel_y)  # Compute gradient magnitude

    # Normalize and threshold edges
    edge_mask = (edges > np.percentile(edges, 95)).astype(np.uint8)

    # Dilate the prediction mask
    dilated_mask = binary_dilation(pred_mask, structure=np.ones(dilation_kernel)).astype(np.uint8)

    # Add edge pixels touching the dilated mask
    modified_mask = np.logical_or(pred_mask, np.logical_and(edge_mask, dilated_mask)).astype(np.uint8)

    return modified_mask

def get_model(architecture, roi):
    if architecture=="swinunetr":
        if roi =='full':
            model = SwinUNETR(
                img_size=(96, 96, 96),
                in_channels=1,
                out_channels=2,
                feature_size=48,
                use_checkpoint=True)
        else:
            model = SwinUNETR(
                img_size=(96, 96, 96),
                in_channels=1,
                out_channels=2,
                feature_size=48,
                drop_rate=0.2,            # ✅ Dropout in MLP layers
                attn_drop_rate=0.2,       # ✅ Dropout in attention layers
                dropout_path_rate=0.2,    # ✅ Dropout in residual connections
                use_checkpoint=True
            )
    elif architecture == "unetr":
        model = UNETR(
            in_channels=1,
            out_channels=2,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.1,
        )
    elif architecture == "unet":
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="instance",
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model.to("cuda")

experiment_dir_full, evaluation_results_dir_full, fold_splits_full, test_loader_full, model_full = setup(architecture, full_model, existing_lung_mask=use_existing_lung_mask)
experiment_dir_roi = f"/home/ilkin/Documents/2024PHD/segmentation/swinunetr/tests/{data}/{architecture}/{roi_model}"
fold_dir = os.path.join(experiment_dir_roi, f"fold_{fold_idx+1}")

model_roi = get_model(architecture, 'roi')
model_roi.load_state_dict(torch.load(os.path.join(fold_dir, model_path)))

pt_list = []
dice_scores = []
improved_dice_scores = []
total_components_before = []
total_components_after_merging = []
total_valid_components = []

model_full.eval()
with torch.no_grad():
    print("\nStarting evaluation...\n")
    for i, eval_data in enumerate(test_loader_full):
        pt_name = fold_splits_full[0]["test_files"][i]["image"].split("/")[-2]
        pt_list.append(pt_name)
        print(f"Processing Patient: {pt_name}")

        original_ct_path = fold_splits_full[0]["test_files"][i]["image"]
        original_gt_path = fold_splits_full[0]["test_files"][i]["image"]

        eval_inputs, eval_labels, lung_mask = eval_data["image"].to(device), eval_data["mask"].to(device), eval_data["lung"].to(device)
        ct_array = eval_inputs.squeeze().cpu().numpy()  # Extract CT scan as NumPy
        gt_mask = eval_labels.squeeze().cpu().numpy().astype(np.uint8)  # Ground truth mask
        lung_mask_np = (lung_mask.squeeze().cpu().numpy() > 0).astype(np.uint8)

        eval_outputs = sliding_window_inference(eval_inputs, roi_size, sw_batch_size=4, predictor=model_full, overlap=overlap)
        image, mask = eval_data["image"][0, 0].cpu().numpy(), eval_data["mask"][0, 0].cpu().numpy()
        predicted = torch.argmax(eval_outputs, dim=1).detach().cpu().numpy()[0]
        patient_output_dir = os.path.join(evaluation_results_dir_full, pt_name)
        os.makedirs(patient_output_dir, exist_ok=True)

        dice_score = fast_dice(gt_mask, predicted)
        dice_scores.append(dice_score)
        print(f"############Initial Tumor Dice for {pt_name}: {dice_score:.4f}############")

        print("🔹 gt - 1s:", np.sum(gt_mask == 1))
        print("🔹 predicted - 1s:", np.sum(predicted == 1))
        
        component_masks, components_before, components_after_merging, valid_components = get_valid_tumor_components(predicted, 
                                                                                                lung_mask_np, 
                                                                                                top_k=k)
        total_components_before.append(components_before)
        total_components_after_merging.append(components_after_merging)
        total_valid_components.append(valid_components)

        combined_prediction = np.zeros_like(gt_mask, dtype=np.uint8)
        for i, comp_mask in enumerate(component_masks):
            # Bounding box of this component
            coords = np.array(np.where(comp_mask))
            y1, x1, z1 = coords.min(axis=1)
            y2, x2, z2 = coords.max(axis=1) + 1
            if roi_model=='s5':
                y1, y2, x1, x2, z1, z2 = y1-16, y2+16, x1-16, x2+16, z1-16, z2+16
            bbox_start = [y1, x1, z1]
            bbox_end = [y2, x2, z2]

            # Prepare ROI data dict
            data_dict = {"image": original_ct_path}
            transforms_roi = Compose([
                LoadImaged(keys=["image"]),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                EnsureChannelFirstd(keys=["image"]),
                CropForegroundd(keys=["image"], source_key="image"),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                SpatialCropd(keys=["image"], roi_start=bbox_start, roi_end=bbox_end),
                EnsureTyped(keys=["image"])
            ])

            roi_ds = Dataset(data=[data_dict], transform=transforms_roi)
            roi_loader = DataLoader(roi_ds, batch_size=1)
            roi_batch = next(iter(roi_loader))
            cropped_image = roi_batch["image"][0, 0].numpy()
            input_tensor = torch.tensor(cropped_image[None, None]).float().to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output = sliding_window_inference(input_tensor, roi_size=roi_size, sw_batch_size=1, predictor=model_roi)
                    pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            dropout_variance_map = compute_uncertainty_map(model_roi, input_tensor, roi_size=roi_size, n_iter=10)
            
            # Paste prediction into combined volume
            combined_prediction[y1:y2, x1:x2, z1:z2] = np.logical_or(
                combined_prediction[y1:y2, x1:x2, z1:z2], pred_mask
            ).astype(np.uint8)

            # 1. Get same slice index used earlier (center slice of GT tumor)
            z_roi = get_middle_tumor_slice(gt_mask[y1:y2, x1:x2, z1:z2])

            # 2. Create zoomed-in uncertainty figure (only last panel style)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            z_roi = get_middle_tumor_slice(gt_mask[y1:y2, x1:x2, z1:z2])

            # Show ROI slice with uncertainty
            ax.imshow(cropped_image[:, :, z_roi], cmap="gray")
            im = ax.imshow(dropout_variance_map[:, :, z_roi], cmap="viridis", alpha=0.6, vmin=0.0, vmax=0.2)
            ax.set_title("Zoomed-in ROI + Uncertainty", fontsize=12)
            ax.axis("off")

            # Add fixed-scale colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Uncertainty (Variance)", fontsize=10)
            cbar.ax.tick_params(labelsize=8)

            plt.tight_layout()
            plt.savefig(os.path.join(patient_output_dir, f"roi_uncertainty_zoom_{pt_name}.png"), dpi=300)
            plt.close()

        # Final cleanup
        print(f"Combined prediction voxels: {np.sum(combined_prediction==1)}")

        if use_existing_lung_mask:
            combined_prediction = apply_lung_mask(combined_prediction, lung_mask_np)
            print(f"Final combined prediction voxels (after lung masking): {np.sum(combined_prediction==1)}")

        # Reconstruct prediction and evaluate
        reconstructed_pred = combined_prediction  # Already full-size

        vis_path = os.path.join(patient_output_dir, f"ieee_pipeline_{pt_name}.png")
        plot_segmentation_pipeline_with_roi_unc(
        ct_image=ct_array,
            gt_mask=gt_mask,
            pred_initial=predicted,
            pred_post=reconstructed_pred,
            roi_ct=cropped_image,
            uncertainty_map=dropout_variance_map,
            bbox=(y1, y2, x1, x2, z1, z2),
            save_path=vis_path,
            title=f"Segmentation Pipeline – {pt_name}"
        )

        # Save prediction and GT as NIfTI files
        pt_output_dir = os.path.join(f"/home/ilkin/Documents/2024PHD/segmentation/swinunetr/postprocessing_results/OH-GLLES-3D/{experiment}", pt_name)
        os.makedirs(pt_output_dir, exist_ok=True)

        # Load original GT NIfTI to reuse metadata
        gt_nii = nib.load(original_gt_path)
        affine = gt_nii.affine
        header = gt_nii.header

        # Save GT mask
        nib.save(nib.Nifti1Image(gt_mask.astype(np.uint8), affine, header), os.path.join(pt_output_dir, "gt_mask.nii.gz"))
        # Save reconstructed prediction
        nib.save(nib.Nifti1Image(reconstructed_pred.astype(np.uint8), affine, header), os.path.join(pt_output_dir, "reconstructed_pred.nii.gz"))

        dice_score = fast_dice(gt_mask, reconstructed_pred)
        print(f"############Final ROI Dice Score: {dice_score:.4f}############")
        improved_dice_scores.append(dice_score)

# Convert to NumPy arrays
initial_dices = np.array(dice_scores)
improved_dices = np.array(improved_dice_scores)
pt_list = np.array(pt_list)
total_components_before = np.array(total_components_before)
total_components_after = np.array(total_components_after_merging)
total_valid_components = np.array(total_valid_components)

# Compute per-patient difference
differences = improved_dices - initial_dices
# Compute means
mean_initial = np.mean(initial_dices)
mean_improved = np.mean(improved_dices)
mean_diff = np.mean(differences)

# Print per-patient comparison
print("Per-patient Dice changes:")
for i, (init, improved, diff) in enumerate(zip(initial_dices, improved_dices, differences)):
    symbol = "⬆️" if diff > 0 else ("⬇️" if diff < 0 else "—")
    print(f"{pt_list[i]}: {init:.4f} → {improved:.4f}  ({diff:+.4f}) {symbol}")
# Print summary
print("\nSummary:")
print(f"Initial Mean Dice:   {mean_initial:.4f}")
print(f"Improved Mean Dice:  {mean_improved:.4f}")
print(f"Average Change:      {mean_diff:+.4f}")

# Create DataFrame
df = pd.DataFrame({
    "Patient": pt_list,
    "Initial_Dice": initial_dices,
    "Improved_Dice": improved_dices,
    "Total_Components_Before": total_components_before,
    "Total_Components_After": total_components_after,
    "Total_Valid_Components": total_valid_components,
})

df = pd.DataFrame({
    "Patient": pt_list,
    "Total_Components_Before": total_components_before,
    "Total_Components_After": total_components_after,
    "Total_Valid_Components": total_valid_components,
})
# Save to CSV
csv_path = f"/home/ilkin/Documents/2024PHD/segmentation/swinunetr/postprocessing_results/{data}/{architecture}/{experiment}.csv"
df.to_csv(csv_path, index=False)

