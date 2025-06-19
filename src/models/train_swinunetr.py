#LIBRARIES
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,4,5,6"
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import random
import nibabel as nib
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data import Dataset, CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    Compose,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    CropForegroundd,
)
from monai.networks.nets import SwinUNETR
import segmentation_models_pytorch as smp
import functions
import loaddata
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import transforms2
import torch.nn.functional as F
import torch.nn as nn
from scipy.ndimage import distance_transform_edt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet, UNETR, SwinUNETR
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize TensorBoard
writer = SummaryWriter()
torch.cuda.empty_cache()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train segmentation model")
parser.add_argument("--data", type=str, required=True, help="Dataset name (e.g., 'OH-GLLES-3D')")
parser.add_argument("--architecture", type=str, required=True, help="Architecture (e.g., 'unet, unetr, swinunetr')")
parser.add_argument("--loss", type=str, required=True, help="Loss Function (e.g., 'dice_ce, uncertainty_guided, uncertainty_lovasz', 'boundary_contrast')")
parser.add_argument("--roi", action="store_true", default=False, help="ROI model")
parser.add_argument("--model_path", type=str, default="best_metric_model.pth", help="Model path (default: 'best_model')")
parser.add_argument("--experiment", type=str, required=True, help="Experiment name (e.g., unet_full)")
args = parser.parse_args()

data = args.data  # Get the value from command-line
model_path = args.model_path
architecture = args.architecture
selected_loss = args.loss
roi = args.roi
experiment = args.experiment
if data=='OH-GLLES-3D':
    data_dir = os.path.join("/home/ilkin/Documents/2024PHD/data", "Planning-CTs")
experiment_dir = f"/home/ilkin/Documents/2024PHD/segmentation/swinunetr/tests/{data}/{architecture}/{experiment}"
print(experiment_dir)
json_path = f"{experiment_dir}/hyperparameters.json"

print(selected_loss)
print(json_path)
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        hyperparams = json.load(f)

# Functions to get additional parameters
lossfunc = functions.get_loss_func(hyperparams["diceCE"])
label_names, numberofclasses = functions.get_classes(hyperparams["dataset"], hyperparams["seg"])
dimension, batchsize = functions.get_dimension_and_bs(hyperparams["dataset"])
dice_roi, aug_roi = functions.get_rois(hyperparams["dataset"], hyperparams["architecture"])

# Update hyperparameters
hyperparams.update({
    "experiment": experiment,
    "number_of_classes": numberofclasses,
    "dimension": dimension,
    "batch_size": batchsize,
    "dice_roi": dice_roi,
    "aug_roi": aug_roi,
    "date_time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),  # Log timestamp
    "loss_function": f"{selected_loss} ({hyperparams['wdice']}-{hyperparams['wce']})",
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "patience": 10
})

# Save the updated hyperparameters back to the JSON file
with open(json_path, "w") as f:
    json.dump(hyperparams, f, indent=4)
print("Hyperparameters updated and saved.")

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

print(aug_roi)
# Get Transforms
if roi:
    print("ROI Segmentation")
    train_transforms = transforms2.getTrainTransformROI(dimension, aug_roi)
    val_transforms = transforms2.getValTransformROI(dimension, aug_roi)
    test_transforms = transforms2.getTestTransformROI(-175, 250)
else:
    print("Full CT Segmentation")
    train_transforms = transforms2.getTrainTransform(dimension, aug_roi)
    val_transforms = transforms2.getValTransform(dimension, aug_roi)
    test_transforms = transforms2.getTestTransform(-175, 250)    

def validation(epoch_iterator_val):
    model.eval()
    meandiceperclass = np.zeros([numberofclasses+1])
    total_val_loss = 0  # ✅ Track validation loss

    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["mask"].cuda())

            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, aug_roi, 4, model, overlap=0.5)

            # Keep val_labels in its original form for post-processing
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]  # ✅ FIXED

            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            

            if selected_loss=="dice_ce":
                val_loss = loss_function(val_outputs, val_labels)
            else:
                # One-hot encode val_labels ONLY for loss computation
                val_labels_onehot = F.one_hot(val_labels.squeeze(1).long(), num_classes=numberofclasses+1).permute(0, 4, 1, 2, 3).float()

                # Compute uncertainty maps and distance transform maps if needed
                uncertainty_map = get_uncertainty_map(model, val_inputs) if selected_loss in ["uncertainty_guided", "uncertainty_lovasz"] else None
                distance_map = compute_distance_transform(val_labels) if selected_loss == "distance_transform" else None

                # Compute validation loss based on selected loss function
                if selected_loss == "uncertainty_guided":
                    val_loss = loss_function(val_outputs, val_labels_onehot, uncertainty_map)
                elif selected_loss == "distance_transform":
                    val_loss = loss_function(val_outputs, val_labels_onehot, distance_map)
                elif selected_loss == "uncertainty_lovasz":
                    val_loss = loss_function(val_outputs, val_labels_onehot, uncertainty_map)
                elif selected_loss == "boundary_contrast":
                    feature_maps = val_outputs  # Assume feature maps are extracted from the model
                    val_loss = loss_function(feature_maps, distance_map)
            
            total_val_loss += val_loss.item()

            dm = dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
            meandiceperclass = meandiceperclass + dm[0]

        print(len(epoch_iterator_val))
        meandiceperclass = meandiceperclass / (len(epoch_iterator_val) + 1)
        print(meandiceperclass)
        mean_dice_val = dice_metric.aggregate().item()
        print(mean_dice_val)

        dice_metric.reset()

    return meandiceperclass, mean_dice_val, total_val_loss / len(epoch_iterator_val)

def train(global_step, train_loader, dice_val_best, global_step_best, patience, max_iterations, fold_dir, 
          recent_dice_scores, no_improvement_count):
    model.train()
    epoch_loss = 0
    step = 0
    
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["mask"].cuda())

        with torch.cuda.amp.autocast():
            logit_map = model(x)  # This contains raw logits

            if selected_loss=="dice_ce":
                loss = loss_function(logit_map, y)
            else:
                # Ensure y is in correct shape before encoding
                y = y.squeeze(1)  # Convert (B, 1, H, W, D) → (B, H, W, D)

                # Convert ground truth mask to one-hot encoding
                y_onehot = F.one_hot(y.long(), num_classes=numberofclasses+1).permute(0, 4, 1, 2, 3).float()
                #print(f"🚀 DEBUG: Input shape before padding: {x.shape}")

                # Compute uncertainty maps or distance transform maps if needed
                uncertainty_map = get_uncertainty_map(model, x) if selected_loss in ["uncertainty_guided", "uncertainty_lovasz"] else None
                distance_map = compute_distance_transform(y) if selected_loss == "distance_transform" else None

                # Compute loss based on selected loss function
                if selected_loss == "uncertainty_guided":
                    loss = loss_function(logit_map, y_onehot, uncertainty_map)
                elif selected_loss == "distance_transform":
                    loss = loss_function(logit_map, y_onehot, distance_map)
                elif selected_loss == "uncertainty_lovasz":
                    loss = loss_function(logit_map, y_onehot, uncertainty_map)
                elif selected_loss == "boundary_contrast":
                    feature_maps = logit_map
                    loss = loss_function(feature_maps, distance_map)

        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Save checkpoint every 5000 iterations
        if global_step % 5000 == 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
                "dice_val_best": dice_val_best,
            }
            checkpoint_path = os.path.join(fold_dir, f"checkpoint_step_{global_step}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at step {global_step} to {checkpoint_path}")

        epoch_iterator.set_description(f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})")

#       if (global_step % hyperparams["eval_num"] == 0 and global_step != 0) or global_step == max_iterations:
        if (global_step % hyperparams["eval_num"] == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val_cb, dice_val, val_loss = validation(epoch_iterator_val)
            epoch_loss /= step
            metrics["train_loss"].append(epoch_loss)
            metrics["val_dice"].append(dice_val)
            metrics["val_loss"].append(val_loss)  # ✅ Track validation loss
            scheduler.step(dice_val)
            metrics["val_dice_per_class"].append(dice_val_cb)
            writer.add_scalar('Loss/train', epoch_loss, global_step)
            writer.add_scalar('Loss/val', val_loss, global_step)  # ✅ Log validation loss
            writer.add_scalar('Dice/mean', dice_val, global_step)
            writer.add_scalar('Dice/tumor', dice_val_cb[1], global_step)
            writer.add_scalar('Dice/background', dice_val_cb[0], global_step)

            # ✅ Track the last 10 Dice scores
            recent_dice_scores.append(dice_val_cb[1])
            print("recent dice scores", recent_dice_scores)
            print("no_improvement_count", no_improvement_count)

            if len(recent_dice_scores) > patience:
                recent_dice_scores.pop(0)  # Keep only the last 10 scores

            # ✅ Early stopping condition: if current Dice is not the best in the last 10
            if dice_val_cb[1] >= max(recent_dice_scores):
                dice_val_best = dice_val_cb[1]
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(fold_dir, "best_metric_model.pth" ))
                print(f"Model Saved! Best Tumor Dice: {dice_val_best:.4f}, Current Tumor Dice: {dice_val_cb[1]:.4f}")
                no_improvement_count = 0  # ✅ Reset counter because we improved!
            else:
                no_improvement_count += 1  # ✅ Increment counter if no improvement
                print(f"Model Not Saved. Best Tumor Dice in last {patience}: {max(recent_dice_scores):.4f}, Current Tumor Dice: {dice_val_cb[1]:.4f}")
            
            # ✅ Stop if Dice is not the highest in the last 10 evaluations, but only if at least 10 exist
            if no_improvement_count >= patience:  
                print(f"No improvement in last {patience} evaluations. Stopping training.")
                return global_step, dice_val_best, global_step_best, recent_dice_scores, no_improvement_count, True  # ✅ Stop training

        global_step += 1
    
    return global_step, dice_val_best, global_step_best, recent_dice_scores, no_improvement_count, False

class UncertaintyGuidedAdaptiveLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(UncertaintyGuidedAdaptiveLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target, uncertainty_map):
        # Compute Dice Loss
        pred_sigmoid = torch.sigmoid(pred)  # Ensure input to Dice Loss is in probability range
        intersection = torch.sum(pred_sigmoid * target)
        dice_loss = 1 - (2. * intersection + self.smooth) / (torch.sum(pred_sigmoid) + torch.sum(target) + self.smooth)
        #What it does? Measures how much the predicted segmentation overlaps with the ground truth.
        #Why use sigmoid? Converts logits into probabilities between 0 and 1.
        #Why use smooth? Prevents division by zero when intersection is very small.

        # Use BCEWithLogits instead of BCE
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
        #What it does? Treats segmentation as a classification problem for each pixel.
        #Why use binary_cross_entropy_with_logits? Avoids numerical instability by directly applying sigmoid inside BCE.
        
        # Compute Uncertainty-Guided Weighting
        alpha = torch.exp(-uncertainty_map)
        weighted_loss = alpha * dice_loss + (1 - alpha) * ce_loss
        #What it does? Adjusts how much Dice or BCE contributes based on uncertainty.
        #How?
        #If uncertainty is high, the model trusts BCE loss more (focuses on classification).
        #If uncertainty is low, the model trusts Dice loss more (focuses on segmentation quality).
        #Why use torch.exp(-uncertainty_map)?
        #Ensures higher uncertainty gives more weight to BCE loss.
        #Prevents extreme dominance of one loss function.

        return weighted_loss.mean()
        #Computes the mean weighted loss across all pixels in the batch.
        #Helps stabilize learning by dynamically adjusting segmentation and classification loss.

class DistanceTransformUncertaintyLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Distance Transform-Based Uncertainty Loss.

        Args:
            smooth (float): Smoothing factor to prevent division by zero in Dice loss.
        """
        super(DistanceTransformUncertaintyLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target, distance_map):
        """
        Compute the loss, weighting the pixel-wise BCE loss using the distance transform.

        Args:
            pred (torch.Tensor): Predicted segmentation map (B, 1, H, W).
            target (torch.Tensor): Ground truth binary mask (B, 1, H, W).
            distance_map (torch.Tensor): Normalized distance transform map (B, 1, H, W).

        Returns:
            torch.Tensor: Weighted loss value.
        """
        # Binary Cross-Entropy loss
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')

        # Dice loss computation
        intersection = torch.sum(pred * target)
        dice_loss = 1 - (2. * intersection + self.smooth) / (torch.sum(pred) + torch.sum(target) + self.smooth)

        # Apply distance map weighting to BCE loss
        weighted_ce_loss = distance_map * ce_loss

        # Final loss: weighted BCE + Dice
        return weighted_ce_loss.mean() + dice_loss

def lovasz_hinge(logits, labels):
    signs = 2. * labels - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    grad = torch.arange(len(errors), dtype=torch.float32).to(logits.device)
    loss = torch.dot(F.relu(errors_sorted), grad) / grad.sum()
    return loss

class UncertaintyWeightedLovaszLoss(nn.Module):
    def __init__(self):
        super(UncertaintyWeightedLovaszLoss, self).__init__()

    def forward(self, pred, target, uncertainty_map):
        lovasz_loss = lovasz_hinge(pred, target)
        uncertainty_weighted_loss = (1 + uncertainty_map) * lovasz_loss
        return uncertainty_weighted_loss.mean()

class BoundaryAwareContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(BoundaryAwareContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, features, boundary_map):
        dist_matrix = torch.cdist(features, features, p=2)
        pos_mask = boundary_map.unsqueeze(1) * boundary_map.unsqueeze(2)
        neg_mask = 1 - pos_mask
        pos_loss = torch.mean(pos_mask * dist_matrix)
        neg_loss = torch.mean(neg_mask * F.relu(self.margin - dist_matrix))
        return pos_loss + neg_loss
        
def compute_distance_transform(mask):
    """
    Compute distance transform where object boundaries are emphasized.
    Supports batch processing.
    """
    mask_np = mask.cpu().numpy().astype(np.uint8)  # Convert to NumPy (batch size, H, W)
    batch_size = mask_np.shape[0]
    
    distance_maps = np.zeros_like(mask_np, dtype=np.float32)
    
    for i in range(batch_size):
        distance_maps[i] = distance_transform_edt(mask_np[i] == 0)  # Compute DT for background
        if np.max(distance_maps[i]) > 0:
            distance_maps[i] /= np.max(distance_maps[i])  # Normalize

    return torch.tensor(distance_maps, dtype=torch.float32).unsqueeze(1).to(mask.device)  # Shape: (B, 1, H, W)

def get_uncertainty_map(model, x, dropout_iters=3):
    model.train()  # ✅ Keep dropout active
    with torch.no_grad(), torch.cuda.amp.autocast():  # ✅ Lower precision & disable gradients
        preds = torch.stack([F.softmax(model(x), dim=1) for _ in range(dropout_iters)], dim=0)
    
    uncertainty_map = preds.var(dim=0).mean(dim=1, keepdim=True)
    model.eval()
    return uncertainty_map

def get_model(architecture, pre_training=False):
    if architecture=="swinunetr":
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=numberofclasses+1,
            feature_size=48,
            drop_rate=0.2,            # ✅ Dropout in MLP layers
            attn_drop_rate=0.2,       # ✅ Dropout in attention layers
            dropout_path_rate=0.2,    # ✅ Dropout in residual connections
            use_checkpoint=False
        )
        if pre_training:
            try:
                weight = torch.load("./model_swinvit.pt")
                model.load_from(weights=weight)
                print("Using pretrained self-supervied Swin UNETR backbone weights !")
            except FileNotFoundError:
                print("Pretrained weights not found. Training from scratch.")

    elif architecture == "unetr":
        model = UNETR(
            in_channels=1,
            out_channels=numberofclasses + 1,
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
            out_channels=numberofclasses + 1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="instance",
        )

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model.to("cuda")

model = get_model(architecture, hyperparams["pretraining"])

# Loss, optimizer, scheduler
loss_functions = {
    "dice_ce": DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=hyperparams["wdice"], lambda_ce=hyperparams["wce"]),
    "uncertainty_guided": UncertaintyGuidedAdaptiveLoss(),
    "distance_transform": DistanceTransformUncertaintyLoss(),
    "uncertainty_lovasz": UncertaintyWeightedLovaszLoss(),
    "boundary_contrast": BoundaryAwareContrastiveLoss()
}
loss_function = loss_functions[selected_loss]

#loss_function = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=hyperparams["wdice"], lambda_ce=hyperparams["wce"])
optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["learning_rate"], weight_decay=hyperparams["weight_decay"])
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=10, verbose=True)
scaler = torch.cuda.amp.GradScaler()

# Post-transforms and metrics
post_label = AsDiscrete(to_onehot=numberofclasses + 1)
post_pred = AsDiscrete(argmax=True, to_onehot=numberofclasses + 1)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

# Function to recursively convert ndarrays in the metrics dictionary to lists
def convert_ndarray_to_list(data):
    if isinstance(data, dict):
        return {key: convert_ndarray_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data
    
# Training loop for all folds
torch.cuda.empty_cache()
for fold_idx, fold_split in enumerate(fold_splits):
    #fold_idx = fold_idx+1
    if fold_idx == 0:
        fold_dir = os.path.join(experiment_dir, f"fold_{fold_idx+1}")

        print(f"Starting training for Fold {fold_idx + 1} / {len(fold_splits)}")
        
        # Set up DataLoaders for this fold
        train_loader = DataLoader(
            CacheDataset(data=fold_split["train_files"], transform=train_transforms, cache_rate=0.1, num_workers=4),
            batch_size=batchsize, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            CacheDataset(data=fold_split["val_files"], transform=val_transforms, cache_rate=0.1, num_workers=4),
            batch_size=1, num_workers=4
        )
        
        # Reset training state variables for this fold
        global_step, dice_val_best, global_step_best = 0, 0.0, 0
        patience = hyperparams["patience"]  # Adjust as needed
        max_iterations = hyperparams["max_iterations"]
        early_stopping = False
        
        # Metrics for this fold
        metrics = {
            "fold_idx": fold_idx + 1,            # Fold number
            "train_loss": [],                   # List of training losses per epoch
            "val_loss": [],                     # List of validation losses per epoch
            "val_dice": [],                     # Global Dice score across all classes
            "val_dice_per_class": [],           # Per-class Dice scores (e.g., [background, tumor])
            "best_tumor_dice": 0.0,             # Best tumor Dice score
            "best_iteration": 0,                # Iteration where best tumor Dice occurred
            "learning_rate": [],                # Learning rate per epoch
        }

        torch.backends.cudnn.benchmark = True
        
        no_improvement_count = 0  # ✅ Make sure this is tracked across calls
        recent_dice_scores = []  # ✅ Track last {patience} Dice scores

        while global_step < max_iterations:
            global_step, dice_val_best, global_step_best, recent_dice_scores, no_improvement_count, early_stopping = train(
                global_step,
                train_loader,
                dice_val_best,
                global_step_best,
                patience=patience,  # ✅ Set patience threshold
                max_iterations=max_iterations,
                fold_dir=fold_dir,
                recent_dice_scores=recent_dice_scores,
                no_improvement_count=no_improvement_count  # ✅ Keep persistent count
            )
            # Log current learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            metrics["learning_rate"].append(current_lr)
            
            if early_stopping:
                print(f"Early stopping triggered for Fold {fold_idx + 1} at iteration {global_step}.")
                metrics["early_stopping"] = {
                    "iteration": global_step,
                    "reason": "No improvement for {} consecutive evaluations.".format(patience),
                }
                break
        
        # Save best model path for this fold
        best_model_path = os.path.join(fold_dir, "best_metric_model.pth")
        print(f"Training completed for Fold {fold_idx + 1}. Best Tumor Dice: {dice_val_best:.4f}")

        # Save the final model as a checkpoint
        final_checkpoint_path = os.path.join(fold_dir, "final_checkpoint_model.pth")
        torch.save(model.state_dict(), final_checkpoint_path)
        print(f"Final checkpoint saved to {final_checkpoint_path}")

        # Log metrics for this fold
        metrics["best_tumor_dice"] = dice_val_best
        metrics["best_iteration"] = global_step_best
        metrics["final_iteration"] = global_step  # Log the final iteration
        metrics_serializable = convert_ndarray_to_list(metrics)
        metrics_path = os.path.join(fold_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_serializable, f)

        print(f"Metrics saved to {metrics_path}")

    # Log all fold-level results at the end of the training
    if fold_idx == len(fold_splits) - 1:
        fold_results_path = os.path.join(experiment_dir, "fold_results.json")
        fold_results = {f"fold_{fold['fold']}": convert_ndarray_to_list(metrics) for fold in fold_splits}

        with open(fold_results_path, "w") as f:
            json.dump(fold_results, f)
        print(f"Fold-level results saved to {fold_results_path}")

def plot_metrics(metrics, save_path):
    plt.figure(figsize=(10, 6))

    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Dice scores
    plt.subplot(2, 1, 2)
    plt.plot(metrics["val_dice"], label="Validation Dice")
    if "val_dice_per_class" in metrics:
        val_dice_per_class = metrics["val_dice_per_class"]
        if len(val_dice_per_class) > 0:
            plt.plot([c[1] for c in val_dice_per_class], label="Tumor Dice")
    plt.title("Dice Score Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.legend()

    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=300)
    print(f"Metrics plot saved at: {save_path}")
    plt.close()

# Load metrics.json for a specific fold
fold_idx = 0  # Replace with the desired fold index
experiment_fold_dir = f"{experiment_dir}/fold_{fold_idx+1}"
metrics_file = os.path.join(experiment_fold_dir, "metrics.json")
plot_save_path = os.path.join(experiment_fold_dir, "metrics_plot.png")

with open(metrics_file, "r") as f:
    metrics = json.load(f)
plot_metrics(metrics, plot_save_path)
