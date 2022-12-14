import numpy as np
from os import listdir
import os
import pdb
import argparse
import pickle

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, average_precision_score
from scipy.signal import medfilt

import utils
from utils import serialize, deserialize
from dataloader import DataLoaderCustom
from train import Flatten, check_patch_accuracy, str2bool

from preprocess import get_imgdims, inflate_2by2

# Constants
#-----------
LEARN_RATE = 1e-5
USE_GPU = True
EPOCHS = 10

print_every = 10
val_every = 20
bs = 36
ppb = 5
dtype = torch.float32


#====================
# Notebook functions
#====================
# prediction of patches and saliency 
# ex: 'reg016_patch771_coords0-3_normal.npy', 'reg016_patch621_coords0-13_shift50.npy', 
def plot_ppms(ppm_targets, ppm_probs, labeldict_path, mode="demo"):

    labels_dict = deserialize(labeldict_path)

    for regi in ppm_targets.keys():
        ppm_prob = ppm_probs[regi]

        if labels_dict[regi] == 0:
            cmap = sns.diverging_palette(280, 145, sep=10, center="dark", as_cmap=True) # purple to green
        else:
            cmap = sns.diverging_palette(10, 220, sep=10, center="dark", as_cmap=True) # red to blue

        # plot
        plt.figure(figsize=(7, 5)) 
        ax = plt.imshow(ppm_prob, cmap=cmap, vmin=-1, vmax=1)
        cbar = plt.colorbar(ax)
        cbar.set_label('Probability of prediction', rotation=270, labelpad=15)
        plt.title('Patch prediction map (PPM) for ' + regi + " [label=" + str(labels_dict[regi])+"]")
        plt.axis('off')
        plt.show()

        if mode == "demo":
            break


# statistical performance
#------------------------
def plot_multiclass_roc_prc(model_files, model_names):

    lw = 4
    aucs = []
    all_preds = []
    all_labels = []
    cs = []

    # get model outputs
    for i, modfile in enumerate(model_files):
        pred_dict = deserialize(utils.model_dir + modfile)
        preds = [v[0] for k,v in pred_dict.items()]
        probs = [v[1] for k,v in pred_dict.items()]
        labels = [v[2] for k,v in pred_dict.items()]

        # auc_ovr = roc_auc_score(labels, probs, multi_class="ovr")
        # auc_ovo = roc_auc_score(labels, probs, multi_class="ovo")

        # aucs.append(auc_ovr)
        # aucs.append(auc_ovo)

        # all_preds.append(probs)
        # all_labels.append(labels)
        c = confusion_matrix(labels, preds)
        # pdb.set_trace()
        cs.append(c)

    return cs


def plot_roc_prc(model_files, model_names, dataset_name="controls", plot_flag=False):
	
    lw = 4
    tprs = []
    fprs = []
    rocs = []
    all_preds = []
    all_labels = []
    prcs = []
    aps = []

    # get model outputs
    for i, modfile in enumerate(model_files):
        pred_dict = deserialize(modfile)
        # pdb.set_trace()
        # preds = [v[0] for k,v in pred_dict.items()]
        # probs = [v[1] for k,v in pred_dict.items()]
        # labels = [v[2] for k,v in pred_dict.items()]

        preds, probs, labels = [], [], []
        for i, (k,v) in enumerate(pred_dict.items()):
            pred = v[0]
            preds.append(pred)
            # print(pred, v[1])
            # pdb.set_trace()
            probs.append(v[1][1]) # 2nd slice: 0=0-class, 1=1-class.
            labels.append(v[2])

        fpr, tpr, _ = roc_curve(labels, probs)
        roc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(labels, probs)
        prc = auc(recall, precision)
        average_precision = average_precision_score(labels, probs)
        
        aps.append(average_precision)
        fprs.append(fpr)
        tprs.append(tpr)
        rocs.append(roc)
        prcs.append(prc)
        all_preds.append(probs)
        all_labels.append(labels)

    if plot_flag == True:
        # ROC
        plt.figure(figsize=(6,4))
        for i in range(len(model_names)):
            if i == 0: 
                m = "o"
            else:
                m = "None"
            print("plotting", model_names[i])
            plt.plot(fprs[i], tprs[i], color=sns.color_palette()[i], marker=m, markevery=0.05, markersize=10, lw=lw, label='%s (area = %0.2f)' % (model_names[i], rocs[i]))

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('ROC curves for PatchCNNs')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig('roc_all_models.png')
        plt.show()

        # PRC
        plt.figure(figsize=(6,4)) # used to be 15,10
        for i in range(len(model_names)):
            if i == 0: 
                m = "o"
            else:
                m = "None"
            average_precision = average_precision_score(all_labels[i], all_preds[i])
            precision, recall, thresholds = precision_recall_curve(all_labels[i], all_preds[i])
            plt.step(recall, precision, color=sns.color_palette()[i], where='post', marker=m, markevery=0.05, markersize=10, lw=lw, label='%s (area = %0.2f)' % (model_names[i], average_precision))

        if dataset_name == "u54codex":
            plt.plot([0, 1], [0.6667, 0.6667], color='navy', lw=lw, linestyle='--')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(loc="lower right")
        plt.title('Precision-Recall curves for PatchCNNs'.format(average_precision))
        plt.savefig('prc_all_models.png')
        plt.show()

    return rocs, prcs, aps



# image-level predictions:
#-------------------------
def topk_majority(votes_ppm, probs_ppm, k=20):
    # assuming: vote/prediction, probability, label
    # Using sorted() + itemgetter() + items()

    # pdb.set_trace()
    # res = dict(sorted(votes.items(), key=lambda x: x[1][1], reverse=True)[:k])
    
    # num = 0
    # for i,l in enumerate(res.values()):
    #     num += l[0]
    # return int(num/i > 0.5)

    #(probs_ppm).argsort(axis=-1)[:, :k]  # np.argpartition(probs_ppm, -k)[-k:]

    zero_inds = np.where(probs_ppm < 0)
    probs_ppm[zero_inds] = -probs_ppm[zero_inds] # -P(y=0) ==> +P(y=0)
    topk_ind = np.unravel_index(np.argsort(probs_ppm.ravel())[-k:], probs_ppm.shape)
    
    # topk_votes = votes_ppm[topk_ind]
    # vote = np.sum(topk_votes) / len(topk_votes) >= 0.5

    # topk_probs = probs_ppm[topk_ind]
    # prob = np.sum(topk_probs) / len(topk_probs) # average probability of P(y=1)

    probs_ppm[zero_inds] = 1-probs_ppm[zero_inds] # P(y=0) ==> 1-P(y=0)=P(y=1)
    
    topk_probs = probs_ppm[topk_ind]
    prob = np.sum(topk_probs) / len(topk_probs) # average probability of P(y=1)

    vote = int(prob >= 0.5)

    return vote, prob


def all_max(votes_ppm, probs_ppm): # MIL rule
    # vote = int(np.max(votes_ppm))

    # if vote == 0: # get correct prob
    #     prob = 1 + np.min(probs_ppm) # most negative 0-vote is highest probability vote 
    # else:
    #     prob = np.max(probs_ppm) # gives you max prob for 1-vote 

    # max_0prob = -np.min(probs_ppm) 
    # max_1prob = np.max(probs_ppm) 
    # if max_0prob > max_1prob:
    #     vote = 0
    #     prob = max_0prob
    # else:
    #     vote = 1
    #     prob = max_1prob

    zero_inds = np.where(probs_ppm < 0)
    probs_ppm[zero_inds] = 1 + probs_ppm[zero_inds] # -P(y=0) ==> 1-P(y=0)=P(y=1)
    prob = np.max(probs_ppm) 
    vote = int(prob >= 0.5)

    return vote, prob
    
        
def all_majority(votes_ppm, probs_ppm):
    # vote = int(np.mean(votes_ppm) >= 0.5)

    zero_inds = np.where(probs_ppm < 0)
    probs_ppm[zero_inds] = 1 + probs_ppm[zero_inds] # -P(y=0) ==> P(y=1) 
    
    prob = np.mean(probs_ppm) # makes sense, should be positive if vote is 1
    # if vote == 0 and prob < 0: # double check to make sure prob is > 0
    #     prob = -prob

    vote = int(prob >= 0.5)

    return vote, prob
        
    
def all_weighted_majority(votes_ppm, probs_ppm):
    weight_vote = np.sum(votes_ppm * probs_ppm) #0-votes are al 0 terms
    prob = weight_vote / votes_ppm.size # sum of 1-votes P(y=1) / all votes
    vote = int(prob >= 0.5)

    # prob = np.mean(probs_ppm) # makes sense, should be positive if vote is 1
    # if vote == 0 and prob < 0: # double check to make sure prob is > 0
    #     prob = -prob

    return vote, prob


def all_caucus_max(votes_ppm, probs_ppm): # MIL rule by caucus
    import skimage.measure
    # caucuses = skimage.measure.block_reduce(votes_ppm, (10,10), np.max)
    # vote = int(np.mean(caucuses) >= 0.5) # if did max here, it would be same as all_max

    # caucus for probabilities
    # zero_inds = np.where(probs_ppm < 0)
    # probs_ppm[zero_inds] = 1 + probs_ppm[zero_inds] # get P(y=1) from -P(y=0)
    # probs = skimage.measure.block_reduce(probs_ppm, (10,10), np.max)

    # probs_ppm[zero_inds] = -probs_ppm[zero_inds] # -P(y=0) ==> +P(y=0)
    # probs = skimage.measure.block_reduce(probs_ppm, (10,10), np.max)

    # probs = np.where(caucuses==1, probs_1, 1+probs_0) # want -P(y=0) to be converted to +P(y=1)

    zero_inds = np.where(probs_ppm < 0)
    
    # max_1prob = skimage.measure.block_reduce(probs_ppm, (10,10), np.max)
    # min_0prob = skimage.measure.block_reduce(probs_ppm, (10,10), np.min)
    # probs = np.where(max_1prob > -min_0prob, max_1prob, 1+min_0prob)

    probs_ppm[zero_inds] = 1 + probs_ppm[zero_inds] #  # -P(y=0) ==> 1-P(y=0)=P(y=1)

    max_1prob = skimage.measure.block_reduce(probs_ppm, (10,10), np.max)

    prob = np.mean(max_1prob) # we don't take max here since that would result in MIL
    vote = int(prob >= 0.5)

    return vote, prob
    
    
def all_caucus_majority(votes_ppm, probs_ppm): # MIL rule by caucus
    import skimage.measure
    # caucuses = skimage.measure.block_reduce(votes_ppm, (10,10), np.mean)
    # vote = int(np.mean(caucuses) >= 0.5) 

    # zero_inds = np.where(votes_ppm == 0)
    # probs_ppm[zero_inds] = 1 + probs_ppm[zero_inds] # get P(y=1) from -P(y=0)
    # probs = skimage.measure.block_reduce(probs_ppm, (10,10), np.mean)
    # prob = np.mean(probs)

     # caucus for probabilities
    zero_inds = np.where(probs_ppm < 0)
    probs_ppm[zero_inds] = 1 + probs_ppm[zero_inds] # -P(y=0) ==> P(y=1) 
    mean_1prob = skimage.measure.block_reduce(probs_ppm, (10,10), np.mean)
    prob = np.mean(mean_1prob)

    vote = int(prob >= 0.5)

    return vote, prob


# def bag_of_scores_clf():
#     if split == "train":
#         pass
#         # fit logreg on image-level labels and (#0s, #1s)
#     elif split == "test":
#         pass
#         # predict logreg model 

        
# def maxembeddings_clf(max_embeds_train, labels_train, max_embeds_test):
#     # fit lasso on image-level labels and mean-image embeddings
#     from sklearn.linear_model import LogisticRegression
#     clf = LogisticRegression(random_state=0, penalty="l2").fit(max_embeds_train, labels_train)
#     # predict lasso model 
#     clf.predict(max_embeds_test)
        
def image_classify(votes_ppm, probs_ppm, mode):
    if mode == "topk_majority":
        rule = topk_majority
    elif mode == "all_max":
        rule = all_max
    elif mode == "all_majority":
        rule = all_majority
    elif mode == "all_weighted_majority":
        rule = all_weighted_majority
    elif mode == "all_caucus_max":
        rule = all_caucus_max
    elif mode == "all_caucus_majority":
        rule = all_caucus_majority
    else:
        print("Error: please choose a valid image classification rule!")

    pred, prob = rule(votes_ppm, probs_ppm)
    return pred, prob


def image_accuracies(ppm_targets, ppm_probs, reference_path, label_dict, print_flag=False):
    modes = ["topk_majority", "all_max", "all_majority", "all_weighted_majority", "all_caucus_max", "all_caucus_majority"]
    
    aurocs, auprcs, aps = [], [], []
    accs = []
    labs = []
    probs, preds = [], []

    for mode in modes: # each image-level classifier
        if print_flag == True:
            print("testing accuracy for mode:", mode)
        num_correct = 0
        
        for i, regi in enumerate(ppm_targets.keys()):
            if "subject" in regi: # hard code for pathology controls
                regi_underscore = "_".join(regi.split("-")) 
                print(regi_underscore)
            else:
                regi_underscore = regi

            if reference_path:
                img_list = os.listdir(reference_path)
            else:
                img_list = label_dict.keys()

            img_match = [x for x in img_list if regi_underscore in x][0]

            # old code
            #---------
            # img_match = [x for x in img_list if x.startswith(regi)][0]

            # if "guilty" in reference_path:
            #     if "guilty" in img_match:
            #         lab = 1
            #     else:
            #         lab = 0
            # else:
            #     if "cold" in img_match:
            #         lab = 0
            #     elif "hot" in img_match:
            #         lab = 1 

   
            lab = label_dict[regi]

            pred, prob = image_classify(ppm_targets[regi], ppm_probs[regi], mode)
            num_correct += pred == lab
            probs.append(prob)
            preds.append(pred)
            labs.append(lab)

        acc = num_correct / (i+1)
        if print_flag == True:
            print("accuracy =", '%.3f'%acc , "\n")
        # accs.append(acc)
        accs.append(mode + " accuracy = " + '%.3f'%acc)

        # AUROC and AUPRC stats at image level
        fpr, tpr, _ = roc_curve(labs, probs)
        roc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(labs, probs)
        prc = auc(recall, precision)
        average_precision = average_precision_score(labs, probs)

        aurocs.append(roc)
        auprcs.append(prc)
        aps.append(average_precision)
   
    return accs, aurocs, auprcs, aps


# Methods to call with shell:
#------------------------------
def image_patch_summary(pred_dict, img_path, HW, args):
    # get all files/regs in val set + get their instantiated sizes
    # this is only telling us the max dims of the patches seen
    # pdb.set_trace()
    if img_path is not None:
        imgdim_dict = get_imgdims(img_path, HW)
    else:
        imgdim_dict = None

    regs_normal, regs_50 = {}, {}
    patch_names = list(pred_dict.keys())

    # check on type of patch naming convention we have
    if "patient" in patch_names[0] and args.dataset_name == "cam":
        multi_sample_flag = True
    else:
        multi_sample_flag = False
    
    for k, pn in enumerate(patch_names):
        if multi_sample_flag == False:
            contents = pn.split("_")
            regi = contents[0]
            patchnum = int(contents[1].split("patch")[1])
            coords = contents[2]
            shift = contents[3]
            aug = contents[4].split(".npy")[0]            
        else:
            contents = pn.split("_")
            regi = contents[0] + "_" + contents[1] + "_" + contents[2] + "_" + contents[3]
            patchnum = int(contents[4].split("patch")[1])
            coords = contents[5]
            shift = contents[6]
            aug = contents[7]

        if aug != "noaug":
            continue # only interested in non-augmented patches
        
        ii = int(coords.split("-")[0].split("coords")[1])
        jj = int(coords.split("-")[1])
        
        # if k > 0 and k % 1000 == 0: 
        #     print("finished processing means of", k, "patches!")

        # redefine to use in old code
        xi = ii
        yi = jj 

        if shift == "noshift":
            if regi not in regs_normal:
                regs_normal[regi] = [xi, yi, patchnum]
            else:
                if xi > regs_normal[regi][0]:
                    regs_normal[regi][0] = xi
                if yi > regs_normal[regi][1]:
                    regs_normal[regi][1] = yi
                if patchnum > regs_normal[regi][2]:
                    regs_normal[regi][2] = patchnum
                    
        elif shift == "50shift":
            if regi not in regs_50:
                regs_50[regi] = [xi, yi, patchnum]
            else:
                if xi > regs_50[regi][0]:
                    regs_50[regi][0] = xi
                if yi > regs_50[regi][1]:
                    regs_50[regi][1] = yi
                if patchnum > regs_50[regi][2]:
                    regs_50[regi][2] = patchnum

    return regs_normal, regs_50, imgdim_dict

    
def image_patch_pred_map(regs_normal, regs_50, imgdim_dict, pred_dict, labeldict_path, model_path, save_path, args):
    # check on type of patch naming convention we have
    patch_names = list(pred_dict.keys())
    if "patient" in patch_names[0] and args.dataset_name == "cam":
        multi_sample_flag = True
    else:
        multi_sample_flag = False

    target_arrs, count_arrs, prob_arrs = {}, {}, {}
    
    # instantiate the maps
    # pdb.set_trace()
    if imgdim_dict is not None:
        for regi in imgdim_dict.keys():
            [rows, cols] = imgdim_dict[regi]

            target_arr = [np.ones((rows+1, cols+1))*-1, np.ones((rows+1, cols+1))*-1]
            count_arr = [np.zeros((rows+1, cols+1)), np.zeros((rows+1, cols+1))]
            prob_arr =  [np.zeros((rows+1, cols+1)), np.zeros((rows+1, cols+1))]

            target_arrs[regi] = target_arr
            count_arrs[regi] = count_arr
            prob_arrs[regi] = prob_arr
    else:
        # pdb.set_trace()       
        for regi in regs_normal.keys():
            rows1, cols1 = regs_normal[regi][0], regs_normal[regi][1]
            rows2, cols2 = regs_50[regi][0], regs_50[regi][1]
            rows = int(np.max([rows1, rows2]))
            cols = int(np.max([cols1, cols2]))

            target_arr = [np.ones((rows+1, cols+1))*-1, np.ones((rows+1, cols+1))*-1]
            count_arr = [np.zeros((rows+1, cols+1)), np.zeros((rows+1, cols+1))]
            prob_arr =  [np.zeros((rows+1, cols+1)), np.zeros((rows+1, cols+1))]

            target_arrs[regi] = target_arr
            count_arrs[regi] = count_arr
            prob_arrs[regi] = prob_arr

    # build heatmaps
    for pn, ppl in pred_dict.items():
        if multi_sample_flag == False:
            contents = pn.split("_")
            regi = contents[0]
            patchnum = int(contents[1].split("patch")[1])
            coords = contents[2]
            shift = contents[3]
            aug = contents[4].split(".npy")[0]
        else:
            contents = pn.split("_")
            regi = contents[0] + "_" + contents[1] + "_" + contents[2] + "_" + contents[3]
            patchnum = int(contents[4].split("patch")[1])
            coords = contents[5]
            shift = contents[6]
            aug = contents[7]

        if aug != "noaug":
            continue # only interested in non-augmented patches; used as a sanity check
        
        pred = ppl[0]
        prob = ppl[1] # 0,1 probs as tuple
        label = ppl[2]

        row = int(coords.split("-")[0].split("coords")[1])
        col = int(coords.split("-")[1])
        
        # target
        if shift == "noshift":
            target_arrs[regi][0][row, col] = int(pred)
        elif shift == "50shift":
            target_arrs[regi][1][row, col] = int(pred)

        # probs
        if shift == "noshift":
            if pred == 0:
                prob_arrs[regi][0][row, col] = -np.float(prob[0]) 
            else:
                prob_arrs[regi][0][row, col] = np.float(prob[1]) 
        elif shift == "50shift":
            if pred == 0:
                prob_arrs[regi][1][row, col] = -np.float(prob[0]) 
            else:
                prob_arrs[regi][1][row, col] = np.float(prob[1]) 

        # counts
        if shift == "noshift":
            count_arrs[regi][0][row, col] += 1
        elif shift == "50shift":
            count_arrs[regi][1][row, col] += 1


    ppm_target_dict = {}
    ppm_prob_dict = {}

    for regi, target_arri in target_arrs.items():      

        prob_arri_noshift = prob_arrs[regi][0]
        prob_arri_50shift = prob_arrs[regi][1]

        count_arri_noshift = count_arrs[regi][0]
        count_arri_50shift = count_arrs[regi][1]

        target_arri_noshift = target_arrs[regi][0]
        target_arri_50shift = target_arrs[regi][1]

        prob_arri_noshift_inflate = inflate_2by2(prob_arri_noshift)
        prob_arri_50shift_inflate = inflate_2by2(prob_arri_50shift)
        count_arri_noshift_inflate = inflate_2by2(count_arri_noshift)
        count_arri_50shift_inflate = inflate_2by2(count_arri_50shift)
        target_arri_noshift_inflate = inflate_2by2(target_arri_noshift)
        target_arri_50shift_inflate = inflate_2by2(target_arri_50shift)

        regh = prob_arri_noshift_inflate.shape[0]
        regw = prob_arri_noshift_inflate.shape[1]
        s50h = prob_arri_50shift_inflate.shape[0]
        s50w = prob_arri_50shift_inflate.shape[1]

        maxh = np.max([regh, s50h])
        maxw = np.max([regw, s50w])

        h = maxh + 1 #2 * ((maxh//2)+1)
        w = maxw + 1 #2 * ((maxw//2)+1)

        prob_arri = np.zeros((h,w))
        prob_arri[0:regh,0:regw] += prob_arri_noshift_inflate
        prob_arri[1:s50h+1,1:s50w+1] += prob_arri_50shift_inflate
        prob_arri = prob_arri / 2 # avg

        count_arri = np.zeros((h,w))
        count_arri[0:regh,0:regw] += count_arri_noshift_inflate
        count_arri[1:s50h+1,1:s50w+1] += count_arri_50shift_inflate
        count_arri = count_arri / 2 # avg

        target_arri = np.ones((h,w)) * -1
        target_arri[0:regh,0:regw] += target_arri_noshift_inflate
        target_arri[1:s50h+1,1:s50w+1] += target_arri_50shift_inflate
        target_arri = target_arri / 2 # avg

        final_arri_prob   = prob_arri * count_arri #/ count_arri
        final_arri_target = target_arri * count_arri

        #save
        ppm_prob_dict[regi] = final_arri_prob
        ppm_target_dict[regi] = final_arri_target

    flavor_text = model_path.split("/")[-1].split(".")[0]
    serialize(ppm_prob_dict, save_path + "/" + flavor_text + "_PPMprobs.obj")
    serialize(ppm_target_dict, save_path + "/" + flavor_text + "_PPM.obj")
    return


def eval_1epoch(args):
    
    model = torch.load(args.model_name, args.device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    model.to(args.device)
  
    num_correct, num_samples, cum_loss, losses, probs, preds, patch_names, labels = check_patch_accuracy(model, args)
    # pdb.set_trace()
   
    pred_dict = {}
    for i, pn in enumerate(patch_names):
        pred_dict[pn] = [preds[i], probs[i], labels[i]]

    return num_correct, num_samples, cum_loss, losses, pred_dict


# Main routine
#-------------
def main():

    # ARGPARSE
    #---------
    parser = argparse.ArgumentParser()
    parser.add_argument('--description', default="no-description", type=str, help='Description of your experiement, with no spaces. E.g. VGG19_bn-random_loading-label_inherit-bce_loss-on_MFL-1')
    parser.add_argument('--model_class', default=None, type=str, help='Select one of: VGG19/VGG19_bn/VGG_att.')
    parser.add_argument('--model_choice', default=None, type=str, help='Select one of: final/manual/best_val (not for controls)')
    parser.add_argument('--model_path', default=None, type=str, help="Where you'd like to save the model outputs.")
   
    # may only be needed for getting guide/shallow classifiers in the pipeline
    parser.add_argument('--game_description', default="patchcnn", type=str, help="Descriptor for gamified learning. Default is patchcnn, i.e. no game played.")

    parser.add_argument('--batch_size', default=36, type=int, help="Batch size. dDfault is 36.")
    parser.add_argument('--channel_dim', default=1, type=int, help="Channel dimension. Default is 1.")
    parser.add_argument('--normalize_flag', default=False, type=str2bool, help="T/F if patches need normalization. Default is False.")
    parser.add_argument('--dataset_name', default=None, type=str, help="What you want to name your dataset. For pre-defined label dictionaries, use: u54codex to search utils.")
    parser.add_argument('--dataloader_type', default="stored", type=str, help="Type of data loader: stored vs otf (on-the-fly).")

    # parameters for patches
    parser.add_argument('--patch_size', default=96, type=int, help="Patch/instance size. Default is 96.")
    parser.add_argument('--patch_loading', default="random", type=str, help="Patch loading scheme: random or blocked. Default is random.")
    parser.add_argument('--patch_labeling', default="inherit", type=str, help="Patch labeling function: inherit or proxy. Default is inhert.")
    parser.add_argument('--patch_loss', default="bce", type=str, help="Patch loss function. Default is bce. Future support for uncertainty.")
    
    # paths
    parser.add_argument('--data_path', default=None, type=str, help="Dataset path.")
    parser.add_argument('--reference_path', default=None, type=str, help="If dataset is patches, this refers to image path that patches were created from.")
    parser.add_argument('--patchlist_path', default=None, type=str, help="Patch list path. This is a cached result of the preprocess.py script.")
    parser.add_argument('--labeldict_path', default=None, type=str, help="Label dictionary path. This is a cached result of the preprocess.py script.")
    parser.add_argument('--save_path', default=None, type=str, help="Save path for predictions.")
   
    args = parser.parse_args()
    setattr(args, "string_details", args.model_class + "-" + args.dataset_name + "-" + str(args.patch_size) + "-" + args.patch_loading + "-" + args.patch_labeling)

    # SET-UP
    #-------
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        print("gpu available!")
    else:
        device = torch.device('cpu')
        print("gpu NOT available!")

    setattr(args, "device", device)

    # sanity checks
    if args.model_path == None or args.model_choice == None:
        print("Error: please specify a model path and/or model choice. Exiting...")
        exit()

    if args.model_class == None:
        print("No model entered. Please choose a model using the parser help flag. Exiting...")
        exit()

    if args.data_path == None or args.patchlist_path == None:
        print("No data path or patchlist path entered. Exiting...")
        exit()

    if args.save_path == None:
        print("Please choose save path. Exiting...")
        exit()

    if args.labeldict_path == None or args.labeldict_path == "predefined":
        if args.dataset_name == "u54codex":
            if args.patch_labeling == "inherit":
                label_dict = utils.labels_dict
            elif args.patch_labeling == "proxy":
                print("proxy labeling not yet implemented, exiting...")
                exit()
                # label_dict = None # use the discretizer function
    else:
        label_dict = deserialize(args.labeldict_path)

    setattr(args, "label_dict", label_dict)


    # choosing model
    if os.path.isfile(args.model_path) == True: # if given a specific model
        print("Single model detected! Bypassing model_choice input, since this is inherently a manual choice.")
        model_name = args.model_path

    elif os.path.isdir(args.model_path) == True: # given a directory
        # print("model directory detected!")
        # match_list = [s for s in os.listdir(args.model_path) if args.string_details in s] # list of models that match the descriptor
        # if args.model_choice == "final":
        #     print("Detecting choice of final epoch...")
        #     final1 = [s for s in match_list if "final" in s]
        #     epoch_final = np.max([int(s.split("epoch")[1]) for s in match_list])
        #     final2 = [s for s in match_list if str(epoch_final) in s]
        #     if final1 == []:
        #         model_name = final2[0]
        #     else:
        #         model_name = final1[0]

        # elif args.model_choice == "best_val":
        #     if args.dataset_name == "controls":
        #         print("No validation set for controls. Please choose: final or manual for model_choice. Exiting....")
        #         exit()
        #     else:
        #         print("Detecting choice of best validation score... searching...")
        #         print("Error: Oops, not yet implemented! Exiting early")
        #         exit()

        print("Detecting a path with models to choose from. Not yet implemented. Please specify specific model!")
        exit() 
        # need to edit this


    else:
        print("Error: Unsure of input path.... Exiting")
        exit()

    setattr(args, "model_name", model_name)
    flavor_text = args.model_path.split("/")[-1].split(".")[0]

    # run eval and get some evaluation stats
    if os.path.isfile(args.save_path + "/" + flavor_text + "_stats.obj"):
        print("detected cached stats and predictions dictionaries!")
        stats_tup = deserialize(args.save_path + "/" + flavor_text + "_stats.obj")
        pred_dict = deserialize(args.save_path + "/" + flavor_text + "_preddict.obj")
    else:
        print("beginning evaluation...")
        num_correct, num_samples, cum_loss, losses, pred_dict = eval_1epoch(args)
        stats_tup = (int(num_correct), int(num_samples), cum_loss)
        serialize(stats_tup, args.save_path + "/" + flavor_text + "_stats.obj")
        serialize(pred_dict, args.save_path + "/" + flavor_text + "_preddict.obj")
    
    print("Here are some quick performance stats!")
    print(stats_tup)
 
    # pdb.set_trace()
    # make PPM dictionaries for plotting later
    regs_normal, regs_50, imgdim_dict = image_patch_summary(pred_dict, args.reference_path, args.patch_size, args)
    serialize(regs_normal, args.save_path + "/" + flavor_text + "_regs_normal.obj")
    serialize(regs_50, args.save_path + "/" + flavor_text + "_regs_50.obj")
    # serialize(imgdim_dict, args.save_path + "/" + "imgdim_dict.obj")
    
    image_patch_pred_map(regs_normal, regs_50, imgdim_dict, pred_dict, args.labeldict_path, args.model_path, args.save_path, args)
    print("FINISHED PREDICTIONS")


if __name__ == "__main__":
	main()









