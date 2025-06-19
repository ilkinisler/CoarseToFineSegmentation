import gc
import numpy as np
import os
import time
import pickle
import glob
import nibabel as nib
from os.path import isfile, join
import sys
import matplotlib.pyplot as plt
import torch
import pandas as pd
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, compute_hausdorff_distance, HausdorffDistanceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.utils import first, set_determinism
from monai.data.utils import partition_dataset
import segmentation_models_pytorch as smp
from monai.metrics.utils import do_metric_reduction, ignore_background
import scipy.ndimage as ndimage
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
    Activations,
)
from typing import Optional, Union

import numpy as np
import torch

from monai.metrics.utils import do_metric_reduction, get_mask_edges, get_surface_distance, ignore_background

def printandsave(string, logFile):
    str = string
    print(str)
    logFile.write(f"{str}\n")

def compute_meandice(val_outputs, val_labels, include_background):
    if not include_background:
        val_outputs, val_labels = ignore_background(y_pred=val_outputs,y=val_labels)
    n_len = len(val_outputs.shape)
    reduce_axis = list(range(2, n_len))
    intersection = torch.sum(val_labels * val_outputs, dim=reduce_axis)
    y_o = torch.sum(val_labels, reduce_axis) #gt
    y_pred_o = torch.sum(val_outputs, dim=reduce_axis) #predicted
    denominator = y_o + y_pred_o
    value = torch.where(y_o > 0, (2.0 * intersection) / denominator, torch.tensor(float(0), device=y_o.device))
    return value

def avg_loss_metric_graph(epoch_loss_values, metric_values, eval_num, loss_metric_path):
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Iteration Average Loss")
    x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [eval_num * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.savefig(loss_metric_path)
    #plt.show()
    plt.savefig(loss_metric_path, bbox_inches='tight')

def metric_graphs_cb(labels, numberofclasses, dice_values_cb, hd_values_cb, dice_path, hd_path, val_interval):

    plt.figure("DICE", (numberofclasses*6, 6))
    plt.title("class based DICE results")
    colors = ['blue','purple','green','pink','orange','red','yellow']
    for i in range(numberofclasses):
        plt.subplot(1, numberofclasses, i+1)
        plt.title(labels[i])
        x = [val_interval * (i + 1) for i in range(len(dice_values_cb[i]))]
        y = dice_values_cb[i]
        plt.xlabel("epoch")
        plt.plot(x, y, color=colors[i])
    
    #plt.show()
    plt.savefig(dice_path, bbox_inches='tight')

    plt.figure("HD", (numberofclasses*6, 6))
    plt.title("class based HD results")
    colors = ['blue','purple','green','pink','orange','red','yellow']
    for i in range(numberofclasses):
        plt.subplot(1, numberofclasses, i+1)
        plt.title(labels[i])
        x = [val_interval * (i + 1) for i in range(len(hd_values_cb[i]))]
        y = hd_values_cb[i]
        plt.xlabel("epoch")
        plt.plot(x, y, color=colors[i])
    
    #plt.show()
    plt.savefig(hd_path, bbox_inches='tight')

#def rotateImg(img):
   #return ndimage.rotate(img, -90)

def plotSegmentation(image, label, tag, test_dir, trial, pt_name):
  plt.title(f"CT & {tag} for {pt_name}")
  plt.imshow(image, cmap='gray', interpolation='none')
  plt.imshow(label, cmap='jet', alpha=0.5, interpolation='none')
  plt.savefig(os.path.join(test_dir, 'evaluation', f"{trial}_{pt_name}_result.png"), bbox_inches='tight')


def save_evaluation_results(dataset, eval_loader, model, test_dir, trial, device, eval_files, aug_roi):
    with torch.no_grad():
        for i, eval_data in enumerate(eval_loader):
            pt_name = eval_files[i]["image"].split("/")[-3]
            print("patient:", eval_files[i]["image"].split("/")[-3], "/", eval_files[i]["image"].split("/")[-1])
            
            if dataset.endswith('3D'):
                roi_size = (64,256,256)
            if dataset.endswith('2D'):
                roi_size = (256,256)

            sw_batch_size = 4
            eval_outputs = sliding_window_inference(
                eval_data["image"].to(device), aug_roi, sw_batch_size, model
            )
            plt.figure("check", (18, 6))

            if dataset.endswith('2D'):
                image = eval_data["image"][0, 0, :, :]
                mask = eval_data["mask"][0, 0, :, :]
                predicted = torch.argmax(eval_outputs, dim=1).detach().cpu()[0, :, :]
            if dataset.endswith('3D'):
                image = eval_data["image"][0, 0, 70, :, :]
                mask = eval_data["mask"][0, 0, 70, :, :]
                predicted = torch.argmax(eval_outputs, dim=1).detach().cpu()[0, 70, :, :]
            
            #print(type(predicted), type(predicted.numpy()),type(torch.argmax(eval_outputs, dim=1).numpy()))

            eval_images_path = os.path.join(test_dir,'evaluation')
            if not os.path.isdir(eval_images_path):
                os.mkdir(eval_images_path)
            eval_images_path = os.path.join(eval_images_path, pt_name)
            if not os.path.isdir(eval_images_path):
                os.mkdir(eval_images_path)

            np.save(os.path.join(eval_images_path, f"{pt_name}_image"), eval_data["image"].numpy())
            np.save(os.path.join(eval_images_path, f"{pt_name}_mask"), eval_data["mask"].numpy())
            np.save(os.path.join(eval_images_path, f"{pt_name}_predicted"), torch.argmax(eval_outputs, dim=1).detach().cpu().numpy())

            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            plt.imshow(image, cmap="gray")

            plt.subplot(1, 3, 2)
            plt.title(f"mask {i}")
            plt.imshow(mask)
            plotSegmentation(image, mask, 'Mask', test_dir, trial, pt_name)

            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")
            plotSegmentation(image, predicted, 'Output', test_dir, trial, pt_name)
            plt.imshow(predicted)

            plt.show(block=False)
            plt.close()


def get_loss_func(diceCE):
    if diceCE:
        return 'DiceCE Loss'
    else:    
        return 'Dice Loss'

def get_classes(dataset,seg):
    if seg == 'GTV' or seg=='CGTV':
        label_names = ['GTV']
        numberofclasses = 1   
        return label_names, numberofclasses
    elif dataset.startswith('NSCLC-LLSEG') or dataset.startswith('OH-GLLES'):
        label_names = ['Right Lung', 'Left Lung', 'Spinal Cord', 'Esophagus']
        numberofclasses = 4


    elif dataset.startswith('OpenKBP-BSMPP'):
        label_names = ['Brainstem', 'Spinal Cord', 'Right Parotid', 'Left Parotid', 'Mandible']
        numberofclasses = 5
    elif dataset.startswith('NSCLC-LLS-3D'):
        label_names = ['Right Lung', 'Left Lung', 'Spinal Cord']
        numberofclasses = 3
    elif dataset.startswith('PDDCA-BCOOPP'):
        label_names = ['Brainstem', 'Chiasm', 'Optic Nerve Left', 'Optic Nerve Right', 'Parotid Left', 'Parotid Right']
        numberofclasses = 6
    elif dataset.startswith('OH-LLG') or dataset.startswith('OHC-LLG'):
        label_names = ['GTV', 'Right Lung', 'Left Lung']
        numberofclasses = 3
    elif dataset.startswith('OHC-G'):
        label_names = ['GTV']
        numberofclasses = 1
        numberofclasses = 5       
    elif dataset =='IPMNCYST':
        label_names = ['Cyst']
        numberofclasses = 1  
    #elif dataset.startswith('OH-GLLES') or dataset.startswith('IPMN'):
    #    label_names = ['GTV', 'Right Lung', 'Left Lung', 'Esophagus', 'Spinal Cord']
    
    print(label_names, numberofclasses)
    return label_names, numberofclasses

def get_dimension_and_bs(dataset):
    
    if dataset.endswith('2D'):
        dimension = 2
        batchsize = 24
    elif dataset.endswith('3D'):
        dimension = 3
        batchsize = 1
    else:
        dimension = 3
        batchsize = 1
    return dimension,batchsize

def get_rois(dataset, architecture):
    '''
    if dataset.endswith('2D'):
        dice_roi = (512, 512)
        aug_roi = (128, 128)

    if dataset.endswith('3D'):
        dice_roi = (64, 512, 512)
        aug_roi = (64, 64, 64)
        if architecture == 'unetr' or  architecture == 'swinunetr':
            aug_roi = (64, 64, 64)

    if dataset.startswith('OpenKBP-BSMPP'):
        if dataset.endswith('3D'):
            dice_roi = (128, 128, 128)
            #dice_roi = (64, 64, 64)
        if dataset.endswith('2D'):
            dice_roi = (128, 128)
    if dataset.startswith('OHC-G'):
        aug_roi = (32, 64, 64)

    if dataset=='NSCLC-LLS-3D':
        aug_roi = (80, 80, 80)
        dice_roi = (200, 512, 512)
        
   
    if dataset == 'NSCLC-LLSEG-3D': #75'''
    aug_roi = (64, 64, 64)
    dice_roi = (512, 512, 200)
    if architecture=='UNETR':
        aug_roi = (96, 96, 96)

    '''if dataset == 'OH-GLLES-3D': #92
        aug_roi = (64, 64, 64)
        dice_roi = (512, 512, 200)    '''


    return dice_roi,aug_roi

def getChannelsOpenKBP(test_labels, cond):
    labels = test_labels.detach().cpu()
    channel0= np.where(labels!= 0, 0, labels)
    channel1= np.where(labels != 1, 0, labels)
    channel2= np.where(labels != 2, 0, labels)
    channel3= np.where(labels != 3, 0, labels)
    channel4= np.where(labels != 4, 0, labels)
    channel5= np.where(labels != 5, 0, labels)
    if cond:
        channel0 = channel0[0,0,:,:,:]
        channel1 = channel1[0,0,:,:,:]
        channel2 = channel2[0,0,:,:,:]
        channel3 = channel3[0,0,:,:,:]
        channel4 = channel4[0,0,:,:,:]
        channel5 = channel5[0,0,:,:,:]
    else:
        channel0 = channel0[0,:,:,:]
        channel1 = channel1[0,:,:,:]
        channel2 = channel2[0,:,:,:]
        channel3 = channel3[0,:,:,:]
        channel4 = channel4[0,:,:,:]
        channel5 = channel5[0,:,:,:]
    new = np.stack((channel0,channel1,channel2,channel3,channel4,channel5))[None,:,:,:,:]
    #print(new.shape)
    new[new != 0] = 1
    return new

def getChannelsNSCLCorOH(test_labels, cond):
    labels = test_labels.detach().cpu()
    channel0= np.where(labels!= 0, 0, labels)
    channel1= np.where(labels != 1, 0, labels)
    channel2= np.where(labels != 2, 0, labels)
    channel3= np.where(labels != 3, 0, labels)
    if cond:
        channel0 = channel0[0,0,:,:,:]
        channel1 = channel1[0,0,:,:,:]
        channel2 = channel2[0,0,:,:,:]
        channel3 = channel3[0,0,:,:,:]
    else:
        channel0 = channel0[0,:,:,:]
        channel1 = channel1[0,:,:,:]
        channel2 = channel2[0,:,:,:]
        channel3 = channel3[0,:,:,:]
    new = np.stack((channel0,channel1,channel2,channel3))[None,:,:,:,:]
    #print(new.shape)
    new[new != 0] = 1
    return new

def calculate_dice(test_inputs,test_labels,dice_roi,model,data):
    roi_size = dice_roi
    sw_batch_size = 1
    test_outputs = sliding_window_inference(
        test_inputs, roi_size, sw_batch_size, model, device=torch.device("cpu")
    )
    post_label = AsDiscrete(argmax = True, to_onehot=False)
    predictions = post_label(test_outputs.detach().cpu()[0,:,:,:,:])
    
    if data=='OpenKBP-BSMPP-3D':
        test_labels_channels = getChannelsOpenKBP(test_labels, True)
        test_outputs_channels = getChannelsOpenKBP(predictions, False)
    elif data=='NSCLC-LLS-3D' or data=='OH-LLG-3D':
        test_labels_channels = getChannelsNSCLCorOH(test_labels, True)
        test_outputs_channels = getChannelsNSCLCorOH(predictions, False)    

    dice_metric = DiceMetric(include_background=True, reduction="mean_channel")
    dicevalue = dice_metric(torch.tensor(test_outputs_channels), torch.tensor(test_labels_channels))
    return dicevalue[0].cpu().detach().numpy()

def calculate_hd95(test_inputs,test_labels,dice_roi,model,data):
    roi_size = dice_roi
    sw_batch_size = 1
    test_outputs = sliding_window_inference(
        test_inputs, roi_size, sw_batch_size, model, device=torch.device("cpu")
    )
    post_label = AsDiscrete(argmax = True, to_onehot=False)
    predictions = post_label(test_outputs.detach().cpu()[0,:,:,:,:])
    
    if data=='OpenKBP-BSMPP-3D':
        test_labels_channels = getChannelsOpenKBP(test_labels, True)
        test_outputs_channels = getChannelsOpenKBP(predictions, False)
    elif data=='NSCLC-LLS-3D' or data=='OH-LLG-3D':
        test_labels_channels = getChannelsNSCLCorOH(test_labels, True)
        test_outputs_channels = getChannelsNSCLCorOH(predictions, False)   

    hdvalue = compute_hausdorff_distance(test_outputs_channels, test_labels_channels, percentile=95)
    return hdvalue.cpu().detach().numpy()[0]

def convertto1hot(array):
    array = array.astype(int)
    n_values = np.max(array) + 1
    onehot = np.eye(n_values)[array]
    onehot = np.transpose(onehot, (3, 0, 1, 2))

    return onehot

def predictive_entropy(predictions,classification=False):
    epsilon = sys.float_info.min
    if classification:
        predictive_entropy = -np.sum( np.mean(predictions, axis=0) * np.log(np.mean(predictions, axis=0) + epsilon),
            axis=-1)
    else:
        cpredictions =[]
        for i in range(len(predictions)):
            cpredictions.append(convertto1hot(predictions[i]))
        predictive_entropy = -np.sum( np.mean(cpredictions, axis=0) * np.log(np.mean(cpredictions, axis=0) + epsilon),
            axis=-1)
    return predictive_entropy