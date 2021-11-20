import os
import numpy as np
import pandas as pd
import random
import glob
from torch.utils import data
import torch
# import h5py
from pathlib import Path
import pdb
from sklearn.preprocessing import KBinsDiscretizer

import utils
from preprocess import axis_rotate, axis_reflect, normalize


class LabelDiscretizer(object):
    def __init__(self, path, sample_size=1000, normalize=False):
        # note: data already normalized in preprocessing, exept if mode = dspt
        self.normalize = normalize
        self.path = path
        self.sample_size = sample_size
        np.random.seed(231)

        self.files = os.listdir(path)
        random.shuffle(self.files)     

    def fit_bins(self, channel=50):
        # print(self.files[:self.sample_size]) yes, random assortment achieved
        meds = []
        for i in range(self.sample_size):
            f = self.files[i]
            try:
                dat = np.load(self.path + f)
            except:
                continue
            med = np.median(dat[channel, :, :])
            meds.append(med)

        enc = KBinsDiscretizer(n_bins=3, encode='ordinal')
        dat_binner = enc.fit(np.expand_dims(np.array(meds), axis=1))
        return dat_binner



class DataLoader(object):
    # This data loader can work on-the-fly (otf) or using pre-stored patches (stored)

    # future: support random and blocked patch loading

    def __init__(self, args):        
        self.args = args
        self.mode = args.dataloader_type

        if self.mode == "otf":
            self.files = utils.deserialize(args.patchlist_path)
            self.images = os.listdir(args.data_path)
        elif self.mode == "stored":
            # pdb.set_trace()
            print("data path:", args.data_path)
            self.files = os.listdir(args.data_path) 
            # print(self.files) 
            # print(type(args.data_path))
            
            if "]" not in self.files[0]: # only patches have this in their names
                print("Detecting stored data loading. Check to make sure data_path is the patch directory. Exiting...")
                exit()
            self.images = None
        else:
            print("enter valid dataloader type")
            exit()

        np.random.seed(231)
        if args.patch_loading == "random":
            random.shuffle(self.files)
        elif args.patch_loading == "blocked":
            print("Warning: blocked loading not yet implemented!")
            if args.batch_size != 25 or args.batch_size != 36:
                print("Detecting blocked data loading, please select a square number batch size.")
                exit()


    def __iter__(self):
        i = 0
        while i < len(self.files):
            batch = []
            labels = []
            filenames = []

            for f in self.files[i:i+self.args.batch_size]:
                if self.mode == "otf":
                    try:
                        # GET ALL INFO FROM patch name and slice into tiff/npy file
                        pieces = f.split("_")
                        reg_id, patch = pieces[0], pieces[1]
                        coord, shift, aug =  pieces[2], pieces[3], pieces[4]

                        [x1x2,y1y2] = coord.split("-")[2].strip('][').split(',')
                        x1,x2 = [int(el) for el in x1x2.split(":")]
                        y1,y2 = [int(el) for el in y1y2.split(":")]
                        img_match = [i for i in self.images if i.startswith(str(reg_id))][0]
                        img = np.load(self.args.data_path + "/" + img_match)

                        if self.args.normalize == True:
                            im = normalize(im, self.args.dataset_name)

                        _,_,D = img.shape
                        dat = img[x1:x2, y1:y2, :].reshape(D, self.args.patch_size, self.args.patch_size)

                        # REFLECT AND ROTATE
                        if aug.startswith("rot"):
                            rot = int(aug.split("rot")[1])
                            dat = axis_rotate(dat, rot)
                        elif aug.startswith("refl"):
                            refl = int(aug.split("refl")[1])
                            dat = axis_reflect(dat, refl) 
                    except:
                        continue
                
                elif self.mode == "stored":
                    try:
                        dat = np.load(self.args.data_path + "/" + f)
                    except:
                        continue
                else:
                    print("Error: valid data loader type not specified! Exiting!")
                    quit()

                lab = self.get_label(f)
                
                if dat.shape == (self.args.channel_dim, self.args.patch_size, self.args.patch_size):
                    batch.append(dat)
                    labels.append(lab)
                    filenames.append(f)

            batch = np.stack(batch, axis=0)
            labels = np.array(labels)
            yield (filenames, batch, labels)
            i += self.args.batch_size
    

    def get_label(self, fname):
        reg_id = fname.split("_")[0]
        # print(self.args.label_dict)
        label = self.args.label_dict[reg_id]
        return label



