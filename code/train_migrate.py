import glob
import os
import sys
import tifffile as tiff
import numpy as np
import pdb
import shutil
from pprint import pprint # for printing dictionaries
from utils import labels_dict


# for debugging purposes
def augmentations_per_reg(dir):
    all_files = os.listdir(dir)    
    
    aug_dict = {}
    for i, x in enumerate(all_files):

        reg = x.split("_")[0].split("reg")[1]
        patch = x.split("_")[1].split("patch")[1]
        
        # remove extraneous dashes
        if "-" in patch:
            patch = patch[1:]
            
        trans = x.split("_")[2]
        
        if reg not in aug_dict.keys():
            aug_dict[reg] = {}
        
        if patch not in aug_dict[reg].keys():
            aug_dict[reg][patch] = []
            
        aug_dict[reg][patch].append(trans)
        
    return aug_dict


def files_to_move(dir):
    all_files = os.listdir(dir)  
    
    aug_dict = {}
    for i, x in enumerate(all_files):

        reg = x.split("_")[0].split("reg")[1]
        patch = x.split("_")[1].split("patch")[1]
        trans = x.split("_")[2]
        
        # remove extraneous dashes
        if "-" in patch:
            patch = patch[1:]
                    
        if reg not in aug_dict.keys():
            aug_dict[reg] = {}
            
        if patch not in aug_dict[reg].keys():
            aug_dict[reg][patch] =  {"normal":[], "aug":[], "final":[]}
        
        if trans == "normal.npy":
            aug_dict[reg][patch]["normal"].append(x)
        else:
            aug_dict[reg][patch]["aug"].append(x)
                    
    return aug_dict
        

def count_files(l):
    return len([1 for x in l])

def unique_files(l):
    return set([x.split("_")[0].split("reg")[1] for x in l])

def set_splits(l):
    all_files = [x.split("_")[0].split("reg")[1] for x in l]
    labels = [labels_dict[u][1] for u in all_files]
    pos = np.sum(labels)
    neg = len(labels) - pos
    return pos, neg  



def main():
    print("\ndict of augmentations...\n------------------------")
    print("train augmentations:")
    pprint(augmentations_per_reg(utils.data_dir + 'train/'))
    print("val augmentations:")
    pprint(augmentations_per_reg(utils.data_dir + 'val/'))
    print("test augmentations:")
    pprint(augmentations_per_reg(utils.data_dir + 'test/'))

    # create list of files to use for train
    #---------------------------------------

    curr_path = utils.data_dir + 'train_full/'
    train_full_dict = files_to_move(curr_path)

    files_to_move = []

    from utils import labels_dict

    for reg in train_full_dict.keys():
        for patch in train_full_dict[reg].keys():
            normal = train_full_dict[reg][patch]["normal"]
            if labels_dict[reg][1] == 1:     
                aug = np.random.choice(train_full_dict[reg][patch]["aug"], 1)
            if labels_dict[reg][1] == 0:
                aug = train_full_dict[reg][patch]["aug"]
            train_full_dict[reg][patch]["final"].extend(normal)
            train_full_dict[reg][patch]["final"].extend(aug)

            files_to_move.extend(train_full_dict[reg][patch]["final"])


    print(len(files_to_move))
    print(len(set(files_to_move)))

#     files_to_move = set(files_to_move)
    
    # checking summary stats for this migrated set
    #---------------------------------------------

    print("After augmentation/up-sampling, we have...\n------------------------------------------")
    print("train set size:", count_files(files_to_move))

    print("\nSee composition of patients in sets...\n--------------------------------------")
    print("train set unique files:", unique_files(files_to_move))

    print("\n(+/-) splits in sets...\n-----------------------")
    print("train set split:", set_splits(files_to_move))

    # execute migration
    #------------------

    curr_path = utils.data_dir + "train_full/"
    dest_path = utils.data_dir + "train/"

    for f in files_to_move:
        shutil.move(curr_path + f, dest_path + f)

