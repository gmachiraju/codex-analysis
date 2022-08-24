import os
import numpy as np
import random
import glob
from torch.utils import data
import torch
import h5py
from pathlib import Path
import pdb
from sklearn.preprocessing import KBinsDiscretizer

import utils
from preprocess import axis_rotate, axis_reflect, normalize


class LabelDiscretizer(object):
    """
    Used for proxy prediction. In multiplexed images, channel can be used as target to train a model. 
    """
    def __init__(self, path, sample_size=1000, normalize=False):
        # note: data likely already normalized in preprocessing
        self.normalize = normalize
        self.path = path
        self.sample_size = sample_size
        np.random.seed(231)

        self.files = os.listdir(path)
        random.shuffle(self.files)     

    def fit_bins(self, channel=50):
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



class DataLoaderCustom(object):
    """
    This dataloader can work on-the-fly (otf) or using pre-stored patches (stored)
    """
    def __init__(self, args):        
        self.args = args
        self.mode = args.dataloader_type

        if self.mode == "otf":
            self.files = utils.deserialize(args.patchlist_path)
            self.images = os.listdir(args.data_path)
        
        elif self.mode == "stored":
            print("data path:", args.data_path)
            self.files = os.listdir(args.data_path) 
            if "]" not in self.files[0]: # only patches have this in their names
                print("Detecting stored data loading. Check to make sure data_path is the patch directory. Exiting...")
                exit()
            self.images = None

        elif self.mode == "hdf5":
            hf = h5py.File(args.data_path, 'r')
            self.files = list(hf.keys())
            self.images = None
            self.retriever = hf

        else:
            print("Error: enter valid dataloader type")
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
        """
        Custom generator to retrieve batches
        """
        i = 0
        while i < len(self.files):
            batch = []
            labels = []
            filenames = []

            for f in self.files[i:i+self.args.batch_size]:
                
                if self.mode == "otf":
                    try:
                        # get all info from patch name and slice into tiff/npy file
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

                elif self.mode == "hdf5":
                    try:
                        dat = self.retriever[f]
                    except:
                        print("unable to retrieve")
                        continue
                    
                else:
                    print("Error: valid data loader type not specified! Exiting!")
                    quit()

                # get label
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
        """
        Grab label for associated file "fname"
            fname: string for filename being queried
        """
        if self.args.dataset_name == "cam":
            if "patient" in fname: #validation
                pieces = fname.split("_")
                reg_id = pieces[0] + "_" + pieces[1] + "_" + pieces[2] + "_" + pieces[3] + ".tif"
            else: # train or test
                pieces = fname.split("_")
                reg_id = pieces[0] + "_" + pieces[1] 
        else:
            reg_id = fname.split("_")[0]


        label = self.args.label_dict[reg_id]
        return label

