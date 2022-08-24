import pickle
import utils #import serialize, deserialize, str2bool
import numpy as np
import argparse
import os
import gc
import copy
import pdb
from predict import image_accuracies, plot_roc_prc

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors
import seaborn as sns
from scipy.signal import medfilt
from sklearn.metrics import auc, roc_curve, average_precision_score
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from sklearn.preprocessing import scale
import skimage
import imutils
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage import morphology
from pyscagnostics import scagnostics
from scipy.spatial import distance

from dataloader import DataLoader
from train import Flatten
from preprocess import inflate_2by2
import pdb
import math
from timeout import timeout
import errno

Z95=1.96
POOL = 7
USE_GPU = True
dtype = torch.float32

# background-filtered
problem_images_vggatt_md = ["reg72", "reg90"]
problem_images_vgg19_gsp = ["reg70"]
problem_images_vggatt_gsp = ["reg70", "reg66"]
problem_images_vggatt_dsp = ["reg71", "reg85"]
problem_images_vggatt_fm = ["reg79", "reg65", "reg77", "reg67", "reg59", "reg95", "reg93", "reg69", "reg41"]
problem_images_vgg19_mdsp = ["S11100006-P235-subject212", "S11100006-P391-subject337", "S11100003-P235-subject212", "S11100003-P407-subject420", "S11100003-P083-subject315"]
problem_images_vggatt_mdsp = ["S11100003-P311-subject320"]

#none-filtered / background-preserved
problem_images_vgg19_mdsp_none = ["S11100006-P235-subject212", "S11100003-P143-subject266", "S11100003-P203-subject302", "S11100006-P363-subject248", "S11100003-P367-subject259", "S11100006-P359-subject163",
                                  "S11100003-P331-subject258", "S11100006-P491-subject297", "S11100003-P283-subject345", "S11100006-P423-subject322", "S11100003-P263-subject279", "S11100006-P391-subject337",
                                  "S11100006-P127-subject222", "S11100006-P451-subject239", "S11100003-P127-subject222", "S11100003-P235-subject212", "S11100003-P407-subject420", "S11100003-P083-subject315"]
problem_images_vggatt_mdsp_none = ["S11100003-P411-subject410"]
problem_images_vggatt_fm_none = ["reg64", "reg16"]


def compute_saliency_maps(X, y, model, model_class, channel_dim=1):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, C, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Make input tensor require gradient
    X.requires_grad_()

    if model_class == "VGG_att":
        [s, c1, c2, c3] = model.forward(X)
    else: #VGG19 and VGG19_bn
        s = model.forward(X)

    sg = s.gather(1, y.view(-1, 1)).squeeze()
    loss = torch.sum(sg)
    loss.backward()

    # saliency map
    saliency = X.grad

    if channel_dim > 1:
        # absolute max of values -- not taking into account all info
        max_values, _ = torch.max(saliency, 1)
        argmax_values = torch.argmax(saliency, 1)

        # L2 norm of ReLU, only positive values
        # can find more options at: https://pytorch.org/docs/master/nn.functional.html
        pos_vals = torch.nn.functional.relu(saliency)   
        norm_values = torch.norm(pos_vals, p=2, dim=1)

        return norm_values #argmax_values, max_values
    else:
        return saliency


def compute_attention_maps(X, y, model, model_class, channel_dim=1):
    # Make sure the model is in "test" mode
    model.eval()

    # Make input tensor require gradient
    X.requires_grad_()

    if model_class == "VGG_att":
        [_, c1, c2, c3] = model.forward(X)
    else: #VGG19 and VGG19_bn
        print("Warning: Skipping attention maps -- non-attention model")
        return None
    if channel_dim == 1:
        # we use 2nd layer of attention for before max-pooling model
        return c2
    elif channel_dim > 1:
        norm_values = torch.norm(c2, p=2, dim=1) # L2?
        return norm_values
    else:
        return None

def compute_gradcam_maps(X, y, model, model_class, channel_dim=1):
    if channel_dim <= 3:
        if model_class.startswith("VGG19"):
            pdb.set_trace()
            target_layers = [model.layer4[-1]]
        elif model_class == "VGG_att":
            pdb.set_trace()
        elif model_class =="ViT"
            pdb.set_trace()

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        targets = [ClassifierOutputTarget(1)]
        grayscale_cam = cam(input_tensor=X, targets=targets)
        return grayscale_cam
    else: 
        return None

def show_explanation_maps(X, y, model, args, plot_flag=False):
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
            
    X_tensor = X_tensor.to(device=args.device, dtype=dtype)  # move to device, e.g. GPU
    y_tensor = y_tensor.to(device=args.device, dtype=torch.long)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model, args.model_class, args.channel_dim)
    attention = compute_attention_maps(X_tensor, y_tensor, model, args.model_class, args.channel_dim)
    gradcam = compute_gradcam_maps(X_tensor, y_tensor, model, args.model_class, args.channel_dim):

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together of whole batch.
    saliency = saliency.cpu().numpy()

    if attention is not None:
        attention = attention.detach().cpu().numpy() 

    if plot_flag == True:
        N = X.shape[0]
        for i in range(N):
            plt.subplot(2, N, i + 1)
            plt.axis('off')
            plt.subplot(2, N, N + i + 1)
            plt.imshow(saliency[i], cmap=plt.cm.hot)
            plt.axis('off')
            plt.gcf().set_size_inches(12, 5)
        plt.show()
        return None
    else:
        return saliency, attention, gradcam
    

def stitch_expmaps(saliency_loader_stitch, model_pt, args, regs_normal, regs_50, imgdim_dict, pred_dict):
    mode = args.saliency_resolution

    # instantiate all giant arrays based on sizes
    saliency_dict = {}
    attention_dict = {}
    ps = args.patch_size

    for regi in imgdim_dict.keys():
        [rows, cols] = imgdim_dict[regi]
        if mode == "patch":
            saliency_dict[regi] = [np.zeros((rows+1, cols+1)), np.zeros((rows+1, cols+1))]
            attention_dict[regi] = [np.zeros((rows+1, cols+1)), np.zeros((rows+1, cols+1))]
        elif mode == "pixels":
            saliency_dict[regi] = [np.zeros((rows*ps, cols*ps)), np.zeros((rows*ps, cols*ps))]
            attention_dict[regi] = [np.zeros((rows*ps, cols*ps)), np.zeros((rows*ps, cols*ps))]

    # now check through all patches
    for i, (fxy, X, y) in enumerate(saliency_loader_stitch):

        patch_sal_map, patch_att_map = show_explanation_maps(X, y, model_pt, args, plot_flag=False)

        for exmap_idx, exmap in enumerate([patch_sal_map, patch_att_map]):
            if torch.is_tensor(exmap) == False and isinstance(exmap, np.ndarray) == False: # if attention is empty/None for VGG19
                if exmap is None:
                # exmap == []: 
                    continue
            if exmap_idx == 0:
                exdict = saliency_dict
            elif exmap_idx == 1:
                exdict = attention_dict

            for j, pn in enumerate(fxy):
                contents = pn.split("_")
                regi = contents[0]
                patchnum = int(contents[1].split("patch")[1])
                coords = contents[2]
                shift = contents[3]
                aug = contents[4].split(".npy")[0]
                if aug != "noaug":
                    continue # only interested in non-augmented patches; used as a sanity check
            
                if mode == "patch": # calculate abs-mean of saliency/attention
                    row = int(coords.split("-")[0].split("coords")[1])
                    col = int(coords.split("-")[1])
                    if shift == "noshift":
                        exdict[regi][0][row, col] = np.mean(np.abs(exmap[j,:,:].squeeze()))
                    elif shift == "50shift":
                        exdict[regi][1][row, col] = np.mean(np.abs(exmap[j,:,:].squeeze()))

                elif mode == "pixels":
                    [x1x2,y1y2] = coords.split("-")[2].strip('][').split(',')
                    x1,x2 = [int(el) for el in x1x2.split(":")]
                    y1,y2 = [int(el) for el in y1y2.split(":")]
                    if shift == "noshift":
                        exdict[regi][0][x1:x2, y1:y2] = exmap[j,:,:].squeeze()
                    elif shift == "50shift":
                        exdict[regi][1][x1:x2, y1:y2] = exmap[j,:,:].squeeze()            

    # merge overlaps
    for exdict in [saliency_dict, attention_dict]:
        if exdict == {}:
            continue
        for regi, sal_arri in exdict.items():
            sal_arri_noshift = exdict[regi][0]
            sal_arri_50shift = exdict[regi][1]

            if mode == "patch":
                sal_arri_noshift_inflate = inflate_2by2(sal_arri_noshift)
                sal_arri_50shift_inflate = inflate_2by2(sal_arri_50shift)

                regh = sal_arri_noshift_inflate.shape[0]
                regw = sal_arri_noshift_inflate.shape[1]
                s50h = sal_arri_50shift_inflate.shape[0]
                s50w = sal_arri_50shift_inflate.shape[1]
                maxh = np.max([regh, s50h])
                maxw = np.max([regw, s50w])
                h = maxh + 1 
                w = maxw + 1 

            elif mode == "pixels":
                regh = sal_arri_noshift.shape[0]
                regw = sal_arri_noshift.shape[1]
                s50h = sal_arri_50shift.shape[0]
                s50w = sal_arri_50shift.shape[1]
                maxh = np.max([regh, s50h])
                maxw = np.max([regw, s50w])
                h = maxh + ps//2 
                w = maxw + ps//2 

            sal_arri = np.zeros((h,w))
            sal_arri[0:regh,0:regw] += sal_arri_noshift_inflate
            sal_arri[1:s50h+1,1:s50w+1] += sal_arri_50shift_inflate
            sal_arri = sal_arri / 2 # avg

            # overwrite with overlap/overlay
            exdict[regi] = sal_arri

    return saliency_dict, attention_dict



#====================
# notebook functions
#====================

def plot_ssms(sal_dict, mode="demo"):
    for regi in sal_dict.keys():
        sm = sal_dict[regi]

        plt.figure(figsize=(7, 5)) 
        ax = plt.imshow(sm)
        cbar = plt.colorbar(ax)
        cbar.set_label('Patch saliency', rotation=270, labelpad=15)
        plt.title('Stitched saliency map (SSM) for ' + regi)
        plt.axis('off')
        plt.show()

        if mode == "demo":
            break


def dice(im1, im2, empty_score=1.0, beta2=1):
    # F_beta (commonly F_1)
    # https://gist.github.com/brunodoamaral/e130b4e97aa4ebc468225b7ce39b3137
    # https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/base/functional.py
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    # print(im1, im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    tp = np.logical_and(im1, im2).sum() # intersection
    fp = im1.sum() - tp
    fn = im2.sum() - tp

    return ((1+beta2) * tp) / (((1+beta2) * tp) + (beta2 * fn) + fp)


def jaccard(im1, im2):

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)
    union = np.logical_or(im1, im2)

    return intersection.sum() / union.sum()


def overlap(im1, im2, empty_score=1.0):

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)
    A = im1.sum() 
    B = im2.sum()
    minAB = np.min([A,B])
    
    if minAB == 0:
        return 0
    if A + B == 0:
        return empty_score

    return intersection.sum() / minAB


def sensitivity(im1, im2):
    # TPR or recall
    # adapted from: https://github.com/Issam28/Brain-tumor-segmentation/blob/master/evaluation_metrics.py
    im1 = np.asarray(im1).astype(int)
    im2 = np.asarray(im2).astype(int)
    # print(im1, im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    num = np.sum(np.multiply(im2, im1))
    denom = np.sum(im2)
    if denom == 0:
        return 1
    else:
        return num / denom

def specificty(im1, im2):
    # TNR
    # im2: ground truth
    # adapted from: https://github.com/Issam28/Brain-tumor-segmentation/blob/master/evaluation_metrics.py
    im1 = np.asarray(im1).astype(int)
    im2 = np.asarray(im2).astype(int)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    num = np.sum(np.multiply(im2==0, im1==0))
    denom = np.sum(im2==0)
    if denom == 0:
        return 1
    else:
        return num / denom


def percent_difference(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    num_differences = np.sum(np.logical_xor(im1, im2))
    # print(num_differences / im1.size)
    return num_differences / im1.size


def scagnostics_helper(x1, y1, x2, y2):
    measures2, _ = scagnostics(x2, y2, remove_outliers=True)
    del x2, y2

    try: # empty prediction
        measures1, _ = scagnostics(x1, y1, remove_outliers=True)
        del x1, y1
    except ValueError:
        dist = 1.0
        return dist

    list1, list2 = [], []
    for key in measures1.keys():
        list1.append(measures1[key])
        list2.append(measures2[key])

    del measures1, measures2
    gc.collect()

    vec1 = np.array(list1)
    vec2 = np.array(list2)
    del list1, list2

    dist = np.abs(distance.cosine(vec1, vec2))

    del vec1,vec2
    gc.collect()

    return dist



# @timeout(1, os.strerror(errno.ETIMEDOUT)) # abort on a given calc if taking > 1 second
def scagnostics_cos(im1, im2, reduce_flag=False):
    im1 = np.asarray(im1).astype(int)
    im2 = np.asarray(im2).astype(int)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    if reduce_flag == "block_mean":
        # make less computationally expensive by downsampling images
        print("Downsampling activated; mean-pool reducing image from:", im1.shape)
        im1 = np.array(skimage.measure.block_reduce(im1, (POOL,POOL), np.mean) >= 0.5, dtype=int)
        im2 = np.array(skimage.measure.block_reduce(im2, (POOL,POOL), np.mean) >= 0.5, dtype=int)
        print("...To size:", im1.shape)
    elif reduce_flag == "block_max":
        # make less computationally expensive by downsampling images
        print("Downsampling activated; max-pool reducing image from:", im1.shape)
        im1 = np.array(skimage.measure.block_reduce(im1, (POOL,POOL), np.max), dtype=int)
        im2 = np.array(skimage.measure.block_reduce(im2, (POOL,POOL), np.max), dtype=int)
        print("...To size:", im1.shape)

    coords1 = np.array(np.where(im1==1))
    x1, y1 = coords1[0], coords1[1]
    del im1,coords1 # memory clearance
    gc.collect()

    coords2 = np.array(np.where(im2==1))
    x2, y2 = coords2[0], coords2[1]
    del im2,coords2 # memory clearance
    gc.collect()

    dist = scagnostics_helper(x1, y1, x2, y2)
    del x1,y1,x2,y2
    gc.collect()
   
    return dist


def f_measure(im1, im2):
    beta2 = 0.3
    return dice(im1, im2, empty_score=1.0, beta2=beta2)


def e_measure(im1, im2, num=255):
    # https://github.com/Hanqer/Evaluate-SOD/blob/master/evaluator.py
    im1 = np.asarray(im1).astype(int)
    im2 = np.asarray(im2).astype(int)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    y_pred, y = im1, im2

    score = np.zeros(num)
    thlist = np.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_pred_th = (y_pred >= thlist[i])
        fm = y_pred_th - y_pred_th.mean()
        gt = y - y.mean()
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
        score[i] = np.sum(enhanced) / (y.size - 1 + 1e-20)
    
    return score

#---------------------------------------------------------
# https://github.com/Hanqer/Evaluate-SOD/blob/master/evaluator.py
def S_object(pred, gt):
    fg = np.where(gt==0, np.zeros_like(pred), pred)
    bg = np.where(gt==1, np.zeros_like(pred), 1-pred)
    # pdb.set_trace()
    o_fg = object(fg, gt)
    o_bg = object(bg, 1-gt)
    u = gt.mean()
    Q = u * o_fg + (1-u) * o_bg
    # print("S_object",Q)
    return Q

def object(pred, gt):
    temp = pred[gt == 1]
    # pdb.set_trace()
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
    return score

def S_region(pred, gt):
    X, Y = centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = divideGT(gt, X, Y)
    p1, p2, p3, p4 = dividePrediction(pred, X, Y)
    Q1 = ssim_manual(p1, gt1)
    Q2 = ssim_manual(p2, gt2)
    Q3 = ssim_manual(p3, gt3)
    Q4 = ssim_manual(p4, gt4)
    Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
    # print("S_region", Q)
    return Q
    
def centroid(gt):
    rows, cols = gt.shape[-2:]
    if gt.sum() == 0:
        X = np.round(cols / 2)
        Y = np.round(rows / 2)
    else:
        total = gt.sum()
        i = np.arange(0,cols)
        j = np.arange(0,rows)
        X = np.round((gt.sum(axis=0)*i).sum() / total)
        Y = np.round((gt.sum(axis=1)*j).sum() / total)
    return int(X), int(Y)
    
def divideGT(gt, X, Y):
    h, w = gt.shape[-2:]
    area = h*w
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4

def dividePrediction(pred, X, Y):
    h, w = pred.shape[-2:]
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB

def ssim_manual(pred, gt):
    h, w = pred.shape[-2:]
    N = h*w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x)*(pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y)*(gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x)*(gt - y)).sum() / (N - 1 + 1e-20)
    
    aplha = 4 * x * y * sigma_xy
    beta = (x*x + y*y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q

#---------------------------------------------------------------

def s_measure(im1, im2):
    # https://github.com/Hanqer/Evaluate-SOD/blob/master/evaluator.py
    im1 = np.asarray(im1).astype(np.float32)
    im2 = np.asarray(im2).astype(int)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    alpha = 0.5
    pred, gt = im1, im2

    y = gt.mean()
    if y == 0:
        x = pred.mean()
        Q = 1.0 - x
    elif y == 1:
        x = pred.mean()
        Q = x
    else:
        Q = (alpha * S_object(pred, gt)) + ((1-alpha) * S_region(pred, gt))
        if Q < 0:
            Q = 0.0
    return Q


def mae(im1, im2):
    # https://github.com/Hanqer/Evaluate-SOD/blob/master/evaluator.py
    im1 = np.asarray(im1).astype(np.float32)
    im2 = np.asarray(im2).astype(np.float32)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.abs(im1 - im2).mean()


def map_eval_mse(a, b):
    a = np.asarray(a).astype(np.float32)
    b = np.asarray(b).astype(np.float32)

    if a.shape != b.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.mean((a - b)**2)


def map_eval_ssim(a, b):
    a = np.asarray(a).astype(np.float32)
    b = np.asarray(b).astype(np.float32)

    if a.shape != b.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return ssim(a,b) #ssim(a,b)


def map_eval(a, b, mode, reduce_flag=False, y=None):
    if mode == "dice":
        rule = dice
    elif mode == "jaccard":
        rule = jaccard
    elif mode == "overlap":
        rule = overlap
    elif mode == "difference": 
        rule = percent_difference 
    elif mode == "sensitivity":
        rule = sensitivity
    elif mode == "specificty":
        rule = specificty  
    elif mode == "scagnostics_cos":
        rule = scagnostics_cos
    elif mode == "mae":
        rule = mae
    elif mode == "f-measure": 
        rule = f_measure
    elif mode == "s-measure":
        rule = s_measure
    elif mode == "e-measure":
        rule = e_measure
    elif mode == "ssim":
        rule = ssim
    else:
        print("Error: please choose a valid eval metric!")

    # print(mode)
    if mode == "scagnostics_cos":
        score = rule(a,b, reduce_flag)
    else:
        score = rule(a, b)

    gc.collect() # memory clearance

    return score


def threshold_sal(raw_sal, gt=None, mode="abs"):
    a = copy.deepcopy(raw_sal)
    
    # absolute value approach
    if mode == "abs":
        a = np.abs(raw_sal)

    # threshold
    p = (2 / (a.shape[0] * a.shape[1])) * a.sum()  # Borji et al
    a = np.array(a > p, dtype=int)

    # if not absolute value, must flip saliency maps that are 1-border
    if mode == "sign":
        a_borders_1s = a[0,:].sum() + a[-1,:].sum() + a[:,0].sum() + a[:,-1].sum()  
        a_borders = a[0,:].size + a[-1,:].size + a[:,0].size + a[:,-1].size
        prop_border_1s = a_borders_1s / a_borders

        if prop_border_1s > 0.5 and raw_sal.mean() < 0: # row 0's don't match 
            a = np.array(1 - np.array(a, dtype=bool), dtype=int) # flip
            print("\t\tflipping SSM/SAM (p-threshold, proportion of 1 borders):", p, prop_border_1s)
            if gt is not None:
                print("\t\trough check: top row of SSM/SAM and GT are < 95 percent similar?", np.sum(a[0,:] == gt[0,:]) / a[0,:].size < 0.95)
    return a



def map_accuracy(scenario, specs, example_1, example_0, label_dict, att_dict, sal_dict, ppm_targets, ppm_probs, ppmgt_dict, reference_path, model_name, num_examples=1, reduce_flag=False):
    modes = ["dice", "jaccard", "overlap", "sensitivity", "specificty", "difference", "scagnostics_cos", "mae", "f-measure", "s-measure", "e-measure", "ssim"]
    analysis = ["ppm_confidence", "ppm_values", "ssm", "sam"]

    if scenario not in ["extreme_value_pixels", "distribution_shifted_pixels", "morphological_differences", "extreme_value_superpixels", "guilty_superpixels", "fractal_morphologies", "morphological_differences_superpixels"]:
        print("Error: please input a supported scenario")
        exit()

    filter_status = specs.split("_")[-2]
    TOL=0.25

    fig, axs = plt.subplots(4, 5, figsize=(10, 10))
    axs[0,0].axis('off')
    axs[0,1].axis('off')
    axs[0,2].axis('off')
    axs[0,3].axis('off')
    axs[0,4].axis('off')
    axs[1,0].axis('off')
    axs[2,0].axis('off')
    axs[1,4].axis('off')
    axs[2,4].axis('off')
    axs[3,0].axis('off')
    axs[3,4].axis('off')
    fig.text(0.1, 0.61, 'Patch\nPrediction', va='center', rotation='vertical', weight="bold")
    fig.text(0.1, 0.41, 'Patch\nConfidence', va='center', rotation='vertical', weight="bold")
    fig.text(0.1, 0.215, 'Saliency & Attention', va='center', rotation='vertical', weight="bold")
    fig.text(0.315, 0.72, 'Output', va='center', weight="bold")
    fig.text(0.455, 0.72, 'Ground truth', va='center', weight="bold")
    fig.text(0.625, 0.72, 'Difference', va='center', weight="bold")
    fig.text(0.45, 0.85, "Scenario: " + scenario.replace("_", " ").lower(), va='center', weight="bold")
    fig.text(0.45, 0.83, "Filtering: " + filter_status, va="center", weight="bold")
    fig.text(0.45, 0.81, "Model: " + model_name, va='center', weight="bold")

    print("Test dataset:\n" + "="*50)
    print(label_dict)

    all_scores = []
    for an in analysis:
        if an == "ppm_confidence": 
            print("\nCalculating PCM statistics...\n")
            maes = []
            ssims = []
            
            print("\nPlease choose example images from the test set:\n" + "="*60)

            for i, regi in enumerate(ppm_targets.keys()):                
                img_list = os.listdir(reference_path)

                if "subject" in regi: # hard code for pathology controls
                    regi_underscore = "_".join(regi.split("-")) 
                    print(regi_underscore)
                else:
                    regi_underscore = regi

                img_match = [x for x in img_list if regi_underscore in x][0]
                # img_match = [x for x in img_list if x.startswith(regi+"-")][0]
                print(img_match)
                lab = label_dict[regi]

                a = ppm_probs[regi] # 0=bg, [-1,1]=fg
                if i < 5:
                    print("output probs:", a)

                if lab == 0:
                    a = -a # -P(y=0) ==> +P(y=0)
                a = np.where(a > 0, a, 0) # will get only the proper probabilities by label
                    
                #averages so they need tolerance
                gt = ppmgt_dict[regi] # -1=bg, [0,1]=fg
                if i < 5:
                    print("raw gt:", gt)

                if lab == 0:
                    if scenario == "extreme_value_pixels" or scenario == "distribution_shifted_pixels" or scenario == "extreme_value_superpixels":
                        b = np.where((gt < 0.5) & (gt >= 0), gt, 0)
                    elif scenario == "guilty_superpixels":
                        b = np.where(gt > 0, gt, 0) 
                    elif scenario == "morphological_differences_superpixels":
                        b = np.where(gt > 0, gt, 0) 
                    else:
                        b = np.where(gt > 0.5, gt, 0) 
                elif lab == 1:
                    if scenario == "guilty_superpixels":
                        b = np.where(gt > 0, gt, 0) 
                    elif scenario == "morphological_differences_superpixels":
                        b = np.where(gt > 0, gt, 0) 
                    else:
                        b = np.where(gt > 0.5, gt, 0)

                if i < 5:
                    print("preview of maps")
                    print("pcm:", a)
                    print("gt:", b)

                pred_mae = map_eval(a, b, "mae")
                pred_ssim = map_eval_ssim(a, b)
               
                maes.append(pred_mae)
                ssims.append(pred_ssim)


                if example_1 == img_match: 

                    if "regi" in img_match:
                        fig.text(0.45, 0.79, "Image: " + img_match.split("-")[0] + "-" + img_match.split("-")[1] , va='center', weight="bold")
                    elif "subject" in img_match:
                        fig.text(0.45, 0.79, "Image: " + img_match.split("_")[2] , va='center', weight="bold")

                    mask = np.load(reference_path + "/" + example_1)
                    
                    if "regi" in img_match:
                        axs[0,0].imshow(mask[:,:,0], cmap="gray")
                        axs[0,0].set_title("Class-1\ntest image")
                        mask = np.load(reference_path + "/" + example_0)
                        axs[0,1].imshow(mask[:,:,0], cmap="gray")
                        axs[0,1].set_title("Corresponding class-0\ntest image")
                    elif "subject" in img_match:
                        axs[0,0].imshow(mask[:,:], cmap="gray")
                        axs[0,0].set_title("Class-1\ntest image")
                        mask = np.load(reference_path + "/" + example_0)
                        axs[0,1].imshow(mask[:,:], cmap="gray")
                        axs[0,1].set_title("Corresponding class-0\ntest image")

                    axs[2,1].imshow(a, cmap="viridis")
                    axs[2,1].set_title("PCM")
                    axs[2,1].axis('off')

                    if scenario != "guilty_superpixels":
                        axs[2,2].imshow(b, cmap="viridis")
                        axs[2,2].set_title("Patch means")

                        differences = np.abs(a - b)
                        axs[2,3].imshow(differences, cmap="viridis")
                        axs[2,3].set_title("Absolute\nDifference")

                    axs[2,2].axis('off')
                    axs[2,3].axis('off')

                del a,b,gt
                gc.collect()

            avg_score_mae = np.mean(maes)
            se_mae = np.std(maes) / np.sqrt(len(maes)) # standard dev / sqrt(n)
            ci_mae = Z95 * se_mae

            avg_score_ssim = np.mean(ssims)
            se_ssim = np.std(ssims) / np.sqrt(len(ssims)) # standard dev / sqrt(n)
            ci_ssim = Z95 * se_ssim
 
            skip_string = ""
            if scenario == "guilty_superpixels":
                skip_string = " --- NOT CALCULATED FOR Guilty Superpixels (GSP)!"
                all_scores.append("Average test "+an+"-mae = "+skip_string)
                all_scores.append("Average test "+an+"-ssim = "+skip_string)
            else:
                all_scores.append("Average test "+an+"-mae (CI) = "+str('%.3f'%avg_score_mae)+" ("+str('%.3f'%ci_mae)+")")
                all_scores.append("Average test "+an+"-ssim (CI) = "+str('%.3f'%avg_score_ssim)+" ("+str('%.3f'%ci_ssim)+")")
        
            print("done with PCM")
        #------------------------
        # PPM, SSM, SAM analyses
        #------------------------
        else:
            if an == "ssm":
                print("\nCalculating SSM statistics...\n")
                print("\tnote: performing MAE,SSIM,S-measure on non-binarized SSM vs GT")
            elif an == "sam":
                print("\nCalculating SAM statistics...\n")
                print("\tnote: performing MAE,SSIM,S-measure on non-binarized SAM vs GT")
            elif an == "ppm_values":
                print("\nCalculating PPM statistics...\n")

            metrics_dict = {}
            sal_visited = False

            guilty_patches = 0

            for i, regi in enumerate(ppm_targets.keys()):

                img_list = os.listdir(reference_path)

                if "subject" in regi: # hard code for pathology controls
                    regi_underscore = "_".join(regi.split("-")) 
                    print(regi_underscore)
                else:
                    regi_underscore = regi

                # img_match = [x for x in img_list if x.startswith(regi+"-")][0]
                img_match = [x for x in img_list if regi_underscore in x][0]

                lab = label_dict[regi]
                print("\t", i, regi, "label=", lab)

                if i > 0 and i % 10 == 0: 
                    print("\tCompleted analyses for", i, "control images")

                skip_string = ""
                if model_name == "VGG19" and an == "sam":
                    skip_string = " --- NOT CALCULATED FOR VGG19!"

                #--------------------
                # SAM or SSM Anlayses
                #--------------------
                # only checking class-1
                if ((an == "ssm") or (an == "sam" and model_name == "VGG_att")) and lab == 1:
                    if an == "ssm":
                        expmap = sal_dict
                    elif an == "sam":
                        expmap = att_dict
                        if expmap == None:
                            continue
                    
                    gt = ppmgt_dict[regi] # -1=bg, [0,1]=fg
                    if i < 5:
                        print("raw gt:", gt)

                    if scenario == "guilty_superpixels": # just 0-val patches exist
                        paired_0label_id = "reg" + str(int(regi.split("reg")[1]) - 1) # 0-label paired image always the ID before
                        paired_0label_gt = ppmgt_dict[paired_0label_id]

                        diff = np.array(np.abs(paired_0label_gt - gt), dtype=bool)  # elementwise for scipy arrays
                        
                        thresholded = np.where((gt < 0.5) & (gt >= 0), 1, 0)
                        thresholded = np.array(thresholded, dtype=bool)

                        all_guilty = np.array(np.logical_and(diff, thresholded), dtype=bool)

                        b = np.array(all_guilty, dtype=int)

                        guilty_patches += b.sum() 
                    elif scenario == "morphological_differences_superpixels":
                        # b = np.where(gt > 0.175, 1, 0) 
                        b = np.where(gt > 0, 1, 0) 
                    else:
                        b = np.where(gt > 0.5, 1, 0)

                    # binarization via adaptive thresholding (2xmean)
                    raw_sal = expmap[regi] # 0=bg, [0,inf)=fg
                    a = threshold_sal(raw_sal, b, mode="abs")
                    a_copy = copy.deepcopy(a)

                    if i < 5:
                        print("preview of maps")
                        print("raw sal:", raw_sal)
                        print("bin sal:", a_copy)
                        print("gt:", b)

                    # calc statistics
                    #----------------      
                    for mode in modes:
                        print("ssm/sam", mode)

                        if sal_visited == False:
                            metrics_dict[mode] = []

                        if mode == "mae" or mode == "ssim" or mode == "s-measure":
                            a = raw_sal
                        else:
                            a = a_copy # reset

                        if filter_status == "background":
                            if reduce_flag == "manual" and mode == "scagnostics_cos":
                                if scenario == "morphological_differences" and model_name == "VGG_att" and regi in problem_images_vggatt_md:
                                    print("skipping image", regi, "due to known scagnostics complexity issues")
                                    continue
                                elif scenario == "morphological_differences_superpixels" and model_name == "VGG_att" and regi in problem_images_vggatt_mdsp:
                                    print("skipping image", regi, "due to known scagnostics complexity issues")
                                    continue
                                elif scenario == "guilty_superpixels" and model_name == "VGG19" and regi in problem_images_vgg19_gsp:
                                    print("skipping image", regi, "due to known scagnostics complexity issues")
                                    continue
                                elif scenario == "guilty_superpixels" and model_name == "VGG_att" and regi in problem_images_vggatt_gsp:
                                    print("skipping image", regi, "due to known scagnostics complexity issues")
                                    continue
                                elif scenario == "distribution_shifted_pixels" and model_name == "VGG_att" and regi in problem_images_vggatt_dsp:
                                    print("skipping image", regi, "due to known scagnostics complexity issues")
                                    continue
                                elif scenario == "fractal_morphologies" and model_name == "VGG_att" and regi in problem_images_vggatt_fm:
                                    print("skipping image", regi, "due to known scagnostics complexity issues")
                                    continue
                        elif filter_status == "none":
                            if reduce_flag == "manual" and mode =="scagnostics_cos":
                                if scenario == "morphological_differences_superpixels" and model_name == "VGG_att" and regi in problem_images_vggatt_mdsp_none:
                                    print("skipping image", regi, "due to known scagnostics complexity issues")
                                    continue
                                elif scenario == "fractal_morphologies" and model_name == "VGG_att" and regi in problem_images_vggatt_fm_none:
                                    print("skipping image", regi, "due to known scagnostics complexity issues")
                                    continue

                        pred = map_eval(a, b, mode, reduce_flag=reduce_flag, y=lab)
                        metrics_dict[mode].append(pred)

                    sal_visited = True

                    if example_1 == img_match: 
                        # enforce same data type
                        a = np.array(a_copy, dtype=bool)
                        b = np.array(b, dtype=bool)

                        # Color for F/T
                        cmap_sal = matplotlib.colors.ListedColormap(['black', 'orangered'])

                        if an == "ssm":
                            axs[3,1].imshow(a, cmap=cmap_sal)
                            axs[3,1].set_title("Binarized SSM")
                            axs[3,1].axis('off')
                        elif an == "sam":
                            axs[3,0].imshow(a, cmap=cmap_sal)
                            axs[3,0].set_title("Binarized SAM")
                            axs[3,0].axis('off')

                        axs[3,2].imshow(b, cmap=cmap_sal)
                        if scenario == "guilty_superpixels":
                            axs[3,2].set_title("Thresholded\nguilty patches")
                        else:
                            axs[3,2].set_title("Binarized\npatch means")
                        axs[3,2].axis('off')

                        if an == "ssm":
                            differences = np.logical_xor(a, b)
                            axs[3,3].imshow(differences, cmap=cmap_sal)
                            axs[3,3].set_title("XOR SSM")
                            axs[3,3].axis('off')
                        elif an == "sam":
                            differences = np.logical_xor(a, b)
                            axs[3,4].imshow(differences, cmap=cmap_sal)
                            axs[3,4].set_title("XOR SAM")
                            axs[3,4].axis('off')

                        if an == "ssm" and model_name == "VGG19":
                            cmap_raw="magma"
                            raw_toplot = scale(np.abs(raw_sal), with_mean=True, with_std=True, copy=True)
                            im1 = axs[3,0].imshow(raw_toplot, cmap=cmap_raw, vmin=0, vmax=1)
                            plt.colorbar(im1, ax=axs[3,0], orientation="horizontal", pad=0.05)
                            axs[3,0].set_title("Abs-Normalized\nSSM")
                            axs[3,0].axis('off')

                    del a,b,gt
                    gc.collect()


                elif an == "ppm_values":

                    a = ppm_targets[regi] # -0.0=bg, [-0.5,0.5]=fg
                    if i < 5:
                        print("output targets:", a)

                    if lab == 0:
                        a = np.where(a < 0, 1, 0)
                    elif lab == 1:
                        a = np.where(a > 0, 1, 0) 

                    
                    #averages so they need tolerance
                    gt = ppmgt_dict[regi] # -1=bg, [0,1]=fg
                    # if i < 5:
                    #     print("raw gt:", gt)

                    if lab == 0:
                        if scenario == "extreme_value_pixels" or scenario == "distribution_shifted_pixels" or scenario == "extreme_value_superpixels":
                            b = np.where((gt < 0.5) & (gt >= 0), 1, 0)
                        elif scenario == "guilty_superpixels":
                            b = np.where(gt > 0, 1, 0) 
                        elif scenario == "morphological_differences_superpixels":
                            b = np.where(gt > 0, 1, 0) 
                        else:
                            b = np.where(gt > 0.5, 1, 0)
                    elif lab == 1:
                        if scenario == "guilty_superpixels":
                            b = np.where(gt > 0, 1, 0)
                        elif scenario == "morphological_differences_superpixels":
                            # b = np.where(gt > 0.175, 1, 0) 
                            b = np.where(gt > 0, 1, 0) 
                        else:
                            b = np.where(gt > 0.5, 1, 0)

                    # if i < 5:
                    #     print("preview of maps")
                    #     print("ppm:", a)
                    #     print("num votes:", a.sum())
                    #     print("gt:", b)
                    #     print("num gt:", b.sum())


                    # calc statistics
                    #----------------
                    print(regi)
                    pre0_skip_flag = False
                    # if filter_status == "background":
                    #     if reduce_flag == "manual" and scenario == "morphological_differences_superpixels" and model_name == "VGG19" and regi in problem_images_vgg19_mdsp:
                    #         print("skipping image", regi, "due to known complexity issues")
                    #         continue
                    #     if reduce_flag == "manual" and scenario == "morphological_differences_superpixels" and model_name == "VGG_att" and regi in problem_images_vggatt_mdsp:
                    #         print("skipping image", regi, "due to known complexity issues")
                    #         continue
                    # elif filter_status == "none":
                    #     if reduce_flag == "manual" and scenario == "morphological_differences_superpixels" and model_name == "VGG19" and regi in problem_images_vgg19_mdsp_none:
                    #         print("skipping image", regi, "due to known complexity issues")
                    #         if i == 0:
                    #             pre0_skip_flag = True
                    #         continue
                    #     if reduce_flag == "manual" and scenario == "morphological_differences_superpixels" and model_name == "VGG_att" and regi in problem_images_vggatt_mdsp_none:
                    #         print("skipping image", regi, "due to known complexity issues")
                    #         if i == 0:
                    #             pre0_skip_flag = True
                    #         continue

                    for mode in modes:
                        print("ppm",mode)

                        if mode == "s-measure" or mode == "ssim":
                            continue
                        if i == 0 or pre0_skip_flag == True:
                            metrics_dict[mode] = []

                        if filter_status == "background":
                            if reduce_flag == "manual" and mode == "scagnostics_cos":
                                if scenario == "morphological_differences" and model_name == "VGG_att" and regi in problem_images_vggatt_md:
                                    print("skipping image", regi, "due to known scagnostics complexity issues")
                                    continue
                                elif scenario == "guilty_superpixels" and model_name == "VGG19" and regi in problem_images_vgg19_gsp:
                                    print("skipping image", regi, "due to known scagnostics complexity issues")
                                    continue
                                elif scenario == "guilty_superpixels" and model_name == "VGG_att" and regi in problem_images_vggatt_gsp:
                                    print("skipping image", regi, "due to known scagnostics complexity issues")
                                    continue
                                elif scenario == "distribution_shifted_pixels" and model_name == "VGG_att" and regi in problem_images_vggatt_dsp:
                                    print("skipping image", regi, "due to known scagnostics complexity issues")
                                    continue
                                elif scenario == "fractal_morphologies" and model_name == "VGG_att" and regi in problem_images_vggatt_fm:
                                    print("skipping image", regi, "due to known scagnostics complexity issues")
                                    continue
                                elif scenario == "morphological_differences_superpixels" and model_name == "VGG19" and regi in problem_images_vgg19_mdsp:
                                    print("skipping image", regi, "due to known scagnostics complexity issues")
                                    continue
                                elif scenario == "morphological_differences_superpixels" and model_name == "VGG_att" and regi in problem_images_vggatt_mdsp:
                                    print("skipping image", regi, "due to known scagnostics complexity issues")
                                    continue

                        elif filter_status == "none":
                            if reduce_flag == "manual" and mode == "scagnostics_cos":
                                if scenario == "morphological_differences_superpixels" and model_name == "VGG19" and regi in problem_images_vgg19_mdsp_none:
                                    print("skipping image", regi, "due to known scagnostics complexity issues")
                                    if i == 0:
                                        pre0_skip_flag = True
                                    continue
                                elif scenario == "morphological_differences_superpixels" and model_name == "VGG_att" and regi in problem_images_vggatt_mdsp_none:
                                    print("skipping image", regi, "due to known scagnostics complexity issues")
                                    if i == 0:
                                        pre0_skip_flag = True
                                    continue

                        pred = map_eval(a, b, mode, reduce_flag=reduce_flag, y=lab)
                        metrics_dict[mode].append(pred)

                    if example_1 == img_match: 
                        # enforce same data type
                        a = np.array(a, dtype=bool)
                        b = np.array(b, dtype=bool)

                        axs[1,1].imshow(a, cmap="gray")
                        axs[1,1].set_title("PPM")
                        axs[1,1].axis('off')

                        if scenario != "guilty_superpixels":
                            axs[1,2].imshow(b, cmap="gray")
                            axs[1,2].set_title("Binarized\npatch means")
          
                            differences = np.logical_xor(a, b)
                            axs[1,3].imshow(differences, cmap="gray")
                            axs[1,3].set_title("XOR")
                        
                        axs[1,2].axis('off')
                        axs[1,3].axis('off')

                    del a,b,gt
                    gc.collect()
                
            # final stats calculations
            #-------------------------
            for mode in modes:

                if scenario == "guilty_superpixels" and (an == "sam"):
                    print("Approximate guilty region #patches:", guilty_patches)

                if scenario == "guilty_superpixels" and an == "ppm_values":
                    skip_string = "0.000 --- NOT CALCULATED FOR Guilty Superpixels (GSP)!"
                    all_scores.append("Average test " + an + "-" + mode + " = " + skip_string)
                else:
                    if metrics_dict == {}:
                        continue 
                    else:
                        try:
                            preds = metrics_dict[mode]
                            print("\tFinalizing average:", mode, "(n="+str(len(preds))+" measurements)")
                        except KeyError:
                            continue

                    if mode == "e-measure":
                        try:
                            summed = np.zeros([1,len(preds[0])])
                            k_dict = {}
                            for j,pred_array in enumerate(preds):
                                summed += pred_array
                                # ---dict of all measurements by index---
                                for k,k_el in enumerate(list(pred_array)):
                                    if j == 0:
                                        k_dict[k] = [k_el]
                                    else:
                                        k_dict[k].append(k_el)
                                #----------------------------------------
                        
                            avgs = summed / (j+1) # avg
                            Q = avgs.max() # check against dictionary method for CIs

                            # get E confidence intervals
                            for k in k_dict.keys():
                                k_vals = k_dict[k]
                                k_entries = np.count_nonzero(~np.isnan(np.array(k_vals)))
                                k_mean = np.nanmean(k_vals)
                                k_se = np.nanstd(k_vals) / np.sqrt(k_entries) 
                                k_dict[k] = (k_mean, k_se) # update

                            Q_cands = k_dict.values()
                            Q_mean_ci = max(Q_cands, key=lambda i: i[0])
                            Q_mean = Q_mean_ci[0]
                            Q_ci = Q_mean_ci[1]
                            # print("MATCH?", Q, Q_mean, Q_ci)
                            all_scores.append("Average test "+an+"-"+mode+" (CI) = "+str('%.3f'%Q_mean)+" ("+str('%.3f'%Q_ci)+")"+skip_string)
                        
                        except IndexError: # shouldn't trip
                            all_scores.append("Average test "+an+"-"+mode+" (CI) = 0.000"+skip_string)

                    else:
                        print("\t\t",preds)
                        num_entries = np.count_nonzero(~np.isnan(np.array(preds)))
                        avg_score = np.nanmean(preds)
                        se_score = np.nanstd(preds) / np.sqrt(num_entries) # standard dev / sqrt(n)
                        ci_score = Z95 * se_score
                        all_scores.append("Average test "+an+"-"+mode+" (CI) = "+str('%.3f'%avg_score)+" ("+str('%.3f'%ci_score)+")"+skip_string)


    print("\nDone!")
    return all_scores, fig


def run_map_analysis(scenario, specs, model_name, example_1, example_0, label_dict, img_path, ppmgt_path, folder, fig_path, num_examples=1, reduce_flag=False):
    ppm_targets = utils.deserialize(folder + model_name + specs + "_PPM.obj")
    ppm_probs = utils.deserialize(folder + model_name + specs + "_PPMprobs.obj")
    sal_dict = utils.deserialize(folder + model_name + specs + "_saliency_dict.obj")
    try:
        att_dict = utils.deserialize(folder + model_name + specs + "_attention_dict.obj")
    except FileNotFoundError:
        att_dict = None
    ppmgt_dict = utils.deserialize(ppmgt_path)
    all_scores, fig = map_accuracy(scenario, specs, example_1, example_0, label_dict, att_dict, sal_dict, ppm_targets, ppm_probs, ppmgt_dict, img_path, model_name, num_examples, reduce_flag)

    print("saving report card at: " + fig_path + '/' + model_name + specs + '.png')
    plt.savefig(fig_path + '/' + model_name + specs + '.png', bbox_inches='tight')

    gc.collect()
    return all_scores

   

def create_prediction_reportcard(scenario, specs, folder, label_path, img_path, ppmgt_path, model_name_list, model_predpath_list, example_1, example_0, fig_path, num_examples=1, reduce_flag=False):
    # gives us the report card for a single model

    # get labels for parition
    label_dict = utils.deserialize(label_path)

    # patch level predictions
    rocs_p, prcs_p, aps_p = plot_roc_prc(model_predpath_list, model_name_list)

    compare_scores = []
    all_accs, all_aurocs, all_auprcs, all_aps = [], [], [], []

    for i, m in enumerate(model_name_list):

        #image level predictions
        ppm_targets = utils.deserialize(folder + m + specs + "_PPM.obj")
        ppm_probs = utils.deserialize(folder + m + specs + "_PPMprobs.obj")
        accs, aurocs, auprcs, aps = image_accuracies(ppm_targets, ppm_probs, img_path, label_dict)
        all_accs.append(accs)
        all_aurocs.append(aurocs)
        all_auprcs.append(auprcs)
        all_aps.append(aps)

        # map analysis
        all_scores = run_map_analysis(scenario, specs, m, example_1, example_0, label_dict, img_path, ppmgt_path, folder, fig_path, num_examples, reduce_flag)
        compare_scores.append(all_scores)
        gc.collect()

    for i, m in enumerate(model_name_list):
        # print
        print()
        print("Prediction Report Card for model: " + m + "\n"+"="*50)

        print("Patch-level predictions" + "\n"+"-"*40)
        print("AUROC =", '%.3f'%rocs_p[i])
        print("AUPRC =", '%.3f'%prcs_p[i])
        print("AP =", '%.3f'%aps_p[i])
        print()

        print("Image-level predictions" + "\n"+"-"*40)
        for j, s in enumerate(all_accs[i]):
            print(s)
            print("AUROC = ", '%.3f'%all_aurocs[i][j])
            print("AUPRC = ", '%.3f'%all_auprcs[i][j])
            print("AP = ", '%.3f'%all_aps[i][j] , "\n")
        print()

        print("Map Evaluations" + "\n"+"-"*40)
        for s in compare_scores[i]:
            print(s)
        print()

    return [rocs_p, prcs_p, aps_p, all_accs, all_aurocs, all_auprcs, all_aps, compare_scores]


def main():

    # ARGPARSE
    #---------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str, help="Where you'd like to save the model outputs.")
    parser.add_argument('--model_class', default=None, type=str, help='Select one of: VGG19/VGG19_bn/VGG_att.')

    parser.add_argument('--batch_size', default=36, type=int, help="Batch size. Default is 36.")
    parser.add_argument('--channel_dim', default=1, type=int, help="Channel dimension. Default is 1.")
    parser.add_argument('--dataset_name', default=None, type=str, help="What you want to name your dataset. For pre-defined label dictionaries, use: u54codex to search utils.")
    parser.add_argument('--dataloader_type', default="stored", type=str, help="Type of data loader: stored vs otf (on-the-fly).")
    parser.add_argument('--patch_size', default=96, type=int, help="Patch/instance size. Default is 96.")
    parser.add_argument('--cache_name', default=None, type=str, help="Cached name for dictionaries.")

    # paths
    parser.add_argument('--data_path', default=None, type=str, help="Dataset path.")
    parser.add_argument('--patchlist_path', default=None, type=str, help="Patch list path. This is a cached result of the preprocess.py script.")
    parser.add_argument('--save_path', default=None, type=str, help="Save path for predictions.")
    parser.add_argument('--labeldict_path', default=None, type=str, help="Label dictionary path. This is a cached result of the preprocess.py script.")
    parser.add_argument('--preddict_path', default=None, type=str, help="prediction dictionary path.")

    parser.add_argument('--saliency_resolution', default="patch", type=str, help="Resolution of saliency. Defaults to patch-level (takes mean of pixels in patch).")

    args = parser.parse_args()
    # set to defaults since it doesn't matter
    setattr(args, "patch_loading", "random")

    label_dict = utils.deserialize(args.labeldict_path)
    setattr(args, "label_dict", label_dict)

    # SET-UP
    #-------
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        print("gpu available!")
    else:
        device = torch.device('cpu')
        print("gpu NOT available!")
    setattr(args, "device", device)

    # make maps
    #-----------
    saliency_loader_stitch = DataLoader(args)
    model_pt = torch.load(args.model_path, map_location=device)

    flavor_text = args.model_path.split("/")[-1].split(".")[0]

    regs_normal = utils.deserialize(args.save_path + "/" + flavor_text + "_regs_normal.obj")
    regs_50 = utils.deserialize(args.save_path + "/" + flavor_text + "_regs_50.obj")
    imgdim_dict = utils.deserialize(args.save_path + "/" + args.cache_name + "-imgdim_dict.obj")
    pred_dict = utils.deserialize(args.save_path + "/" + flavor_text + "_preddict.obj")

    saliency_dict, attention_dict = stitch_expmaps(saliency_loader_stitch, model_pt, args, regs_normal, regs_50, imgdim_dict, pred_dict)
    utils.serialize(saliency_dict, args.save_path + "/" + flavor_text + "_saliency_dict.obj")
    utils.serialize(attention_dict, args.save_path + "/" + flavor_text + "_attention_dict.obj")
    print("FINISHED EXPLANATIONS")


if __name__ == "__main__":
	main()
