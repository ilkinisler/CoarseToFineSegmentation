from monai.transforms import (
    AsDiscrete,
    Compose,
    CenterSpatialCropd,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    Resized,
    SpatialPadd,
    EnsureChannelFirstd,
    Transform
)
import numpy as np
import torch
from monai.transforms import Lambda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_samples = 16


class PrintSizes:
    def __init__(self, message=""):
        self.message = message

    def __call__(self, data):
        for key, value in data.items():
            print(f"{self.message} - Print Sizes for {key}: {value.shape}")
        return data


def getTrainTransform(dimension, aug_roi):
    def print_shapes(data, msg):
        print(f"For {data['patient_id']} - {msg} - Image shape: {data['image'].shape}, Mask shape: {data['mask'].shape}")
        return data  # It's important to return the data for the next transform

    train_transforms = Compose([
        LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
        #Lambda(lambda x: print_shapes(x, "After Loading")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        #Lambda(lambda x: print_shapes(x, "After ScaleIntensityRanged")),
        #SpatialPadd(
        #    keys=["image", "mask"], 
        #    spatial_size=(512, 512, 128), 
        #    method="symmetric", 
        #    mode='constant', 
        #    constant_values=0
        #),  # Center and pad
        #Lambda(lambda x: print_shapes(x, "After SpatialPadd")),
        CropForegroundd(keys=["image", "mask"], source_key="image"),
        #Lambda(lambda x: print_shapes(x, "After CropForegroundd")),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        #Lambda(lambda x: print_shapes(x, "After Orientationd")),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        #Lambda(lambda x: print_shapes(x, "After Spacingd")),
        EnsureTyped(keys=["image", "mask"], track_meta=False),
        #Lambda(lambda x: print_shapes(x, "After EnsureTyped")),
        RandCropByPosNegLabeld(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=aug_roi,
            pos=1, neg=1, num_samples=num_samples,
            image_key="image", image_threshold=0
        ),
        #Lambda(lambda x: print_shapes(x, "After RandCropByPosNegLabeld")),
        RandFlipd(keys=["image", "mask"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "mask"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "mask"], spatial_axis=[2], prob=0.10),
        #Lambda(lambda x: print_shapes(x, "After RandFlipd")),
        RandRotate90d(keys=["image", "mask"], prob=0.10, max_k=3),
        #Lambda(lambda x: print_shapes(x, "After RandRotate90d")),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
        #Lambda(lambda x: print_shapes(x, "After RandShiftIntensityd")),

    ])
    return train_transforms

def getValTransform(dimension, aug_roi):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "mask"], source_key="image"),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            Spacingd(
                keys=["image", "mask"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "mask"], track_meta=True),
        ]
    )
    return val_transforms

def getValTransformOLD():
    val_transforms = Compose([
        LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "mask"], source_key="image"),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),

        # ✅ NEW: Pad the image so it's divisible by 32
        SpatialPadd(keys=["image", "mask"], spatial_size=[(d + 31) // 32 * 32 for d in (128, 128, 128)], method="symmetric"),

        # ✅ NEW: Crop a fixed ROI for validation (ensures correct input size)
        RandCropByPosNegLabeld(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=(128, 128, 128),  # ✅ Ensure this is a multiple of 32
            pos=1, neg=1, num_samples=1,  # ✅ One sample ensures deterministic behavior in validation
            image_key="image",
            image_threshold=0,
        ),
    ])
    return val_transforms



def getTestTransform(a_min, a_max):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),            
            CropForegroundd(keys=["image", "mask"], source_key="image"),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            Spacingd(
                keys=["image", "mask"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "mask"], track_meta=True),
        ]
    )
    return val_transforms

def getTestTransform_withlungmask(a_min, a_max):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask", "lung"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,

                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),            
            #EnsureChannelFirstd(keys=["image", "mask"]), #added this later
            CropForegroundd(keys=["image", "mask", "lung"], source_key="image"),
            Orientationd(keys=["image", "mask", "lung"], axcodes="RAS"),
            Spacingd(
                keys=["image", "mask", "lung"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest", "nearest"),
            ),
            EnsureTyped(keys=["image", "mask", "lung"], track_meta=True),
        ]
    )
    return val_transforms



def getTrainTransformROI(dimension, aug_roi):
    def print_shapes(data, msg):
        print(f"For {data['patient_id']} - {msg} - Image shape: {data['image'].shape}, Mask shape: {data['mask'].shape}")
        return data  # It's important to return the data for the next transform

    train_transforms = Compose([
        LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "mask"], source_key="mask"),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        EnsureTyped(keys=["image", "mask"], track_meta=False),
        SpatialPadd(keys=["image", "mask"], spatial_size=aug_roi, method="symmetric", mode="constant", constant_values=0),
        RandCropByPosNegLabeld(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=aug_roi,
            pos=1, neg=1, num_samples=num_samples,
            image_key="image", image_threshold=0
        ),
        RandFlipd(keys=["image", "mask"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "mask"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "mask"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "mask"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    ])

    return train_transforms


def getValTransformROI(dimension, aug_roi):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "mask"], source_key="mask"),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            Spacingd(
                keys=["image", "mask"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            SpatialPadd(keys=["image", "mask"], spatial_size=aug_roi, method="symmetric", mode="constant", constant_values=0),
            CenterSpatialCropd(keys=["image", "mask"], roi_size=aug_roi),
            EnsureTyped(keys=["image", "mask"], track_meta=True),
        ]
    )
    return val_transforms


def getTestTransformROI(a_min=-175, a_max=250):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),            
            CropForegroundd(keys=["image", "mask"], source_key="mask"),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            Spacingd(
                keys=["image", "mask"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            SpatialPadd(keys=["image", "mask"], spatial_size=(64, 64, 64), method="symmetric", mode="constant", constant_values=0),
            EnsureTyped(keys=["image", "mask"], track_meta=True),
        ]
    )
    return val_transforms



def getTrainTransformROI16(dimension, aug_roi):
    def print_shapes(data, msg):
        print(f"For {data['patient_id']} - {msg} - Image shape: {data['image'].shape}, Mask shape: {data['mask'].shape}")
        return data  # It's important to return the data for the next transform

    train_transforms = Compose([
        LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "mask"], source_key="mask", margin=(16, 16, 8)),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),
        EnsureTyped(keys=["image", "mask"], track_meta=False),
        SpatialPadd(keys=["image", "mask"], spatial_size=(64, 64, 64), method="symmetric", mode="constant", constant_values=0),
        RandCropByPosNegLabeld(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=(64, 64, 64),
            pos=1, neg=1, num_samples=num_samples,
            image_key="image", image_threshold=0
        ),
        RandFlipd(keys=["image", "mask"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "mask"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "mask"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "mask"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    ])

    return train_transforms


def getValTransformROI16():
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "mask"], source_key="mask",  margin=(16, 16, 8)),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            Spacingd(
                keys=["image", "mask"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            SpatialPadd(keys=["image", "mask"], spatial_size=(64, 64, 64), method="symmetric", mode="constant", constant_values=0),
            EnsureTyped(keys=["image", "mask"], track_meta=True),
        ]
    )
    return val_transforms


def getTestTransformROI16(a_min=-175, a_max=250):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),            
            CropForegroundd(keys=["image", "mask"], source_key="mask",  margin=(16, 16, 16)),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            Spacingd(
                keys=["image", "mask"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            SpatialPadd(keys=["image", "mask"], spatial_size=(64, 64, 64), method="symmetric", mode="constant", constant_values=0),
            EnsureTyped(keys=["image", "mask"], track_meta=True),
        ]
    )
    return val_transforms
