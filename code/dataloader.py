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
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
        elif self.mode == "hdf5_triplets":
            hf = h5py.File(args.data_path, 'r')
            self.files = utils.deserialize(args.triplet_list)
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
                        dat = self.retriever[f][()] # added [()] recently
                    except:
                        print("unable to retrieve")
                        continue
                elif self.mode == "hdf5_triplets":
                    try:
                        xa = self.retriever[f[0]][()]
                        xn = self.retriever[f[1]][()]
                        xd = self.retriever[f[2]][()] # added [()] recently
                        dat = np.stack([xa,xn,xd], axis=0)
                    except:
                        print("unable to retrieve")
                        continue
                else:
                    print("Error: valid data loader type not specified! Exiting!")
                    quit()

                # get label
                if self.mode != "hdf5_triplets":
                    lab = self.get_label(f)
                    if dat.shape == (self.args.channel_dim, self.args.patch_size, self.args.patch_size):
                        batch.append(dat)
                        labels.append(lab)
                        filenames.append(f)
                else:
                    laba = self.get_label(f[0])
                    labn = self.get_label(f[1])
                    labd = self.get_label(f[2])
                    lab = [laba, labn, labd]
                    if dat.shape == (3, self.args.channel_dim, self.args.patch_size, self.args.patch_size):
                        batch.append(dat)
                        labels.extend(lab)
                        filenames.append(f)
            
            # pdb.set_trace()
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


class TripletDataset(Dataset):
    """
    Created for easier and faster dataloading in pytorch
    Assumes:
        - hdf5 dataset created to store all valid image or time-series patches
        - serialized list of triplet name tuples to reference hdf5 dataset with
    Adapted from: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html and
    https://github.com/ermongroup/tile2vec/blob/master/src/datasets.py 
    """
    def __init__(self, patch_dir, triplet_names, args, transform=None, target_transform=None):
        self.patch_dir = patch_dir
        hf = h5py.File(self.patch_dir, 'r')
        self.retriever = hf
        if isinstance(triplet_names, str):
            self.triplet_names = utils.deserialize(triplet_names)
        elif isinstance(triplet_names, list):
            if args.selfsup_mode != "sextuplet":
                if len(triplet_names) > 2:
                    print("Error: only allow for max of 2 triplet lists")
                    exit()
                self.triplet_names = utils.deserialize(triplet_names[0]) + utils.deserialize(triplet_names[1])
            else: # carta
                self.triplet_names = []
                trips0 = utils.deserialize(triplet_names[0])
                trips1 = utils.deserialize(triplet_names[1])
                num_iter = int(np.min([len(trips0), len(trips1)]))
                for i in range(num_iter):
                    trip0 = trips0[i]
                    trip1 = trips1[i]
                    self.triplet_names.append([trip0, trip1]) 

        self.transform = transform
        self.target_transform = target_transform
        self.args = args
        if args.toy_flag == True:
            if args.overfit_flag == True:
                factor = 2000
            else:
                factor = 2 # we've previously tried 1, 2, 4, 20
            print("toy_flag raised, thus only training on subset of triplet dataset: 1/"+str(factor))
            n = len(self.triplet_names)
            print("Training on", n//factor, "triplets instead of full dataset of", n)
            # first get random sample (in-place)
            np.random.seed(0)
            np.random.shuffle(self.triplet_names)
            # then get subset
            self.triplet_names = self.triplet_names[0:n//factor]

    def __len__(self):
        # number of triplets
        return len(self.triplet_names)

    def get_label(self, fname):
        if "patient" in fname: #validation
            pieces = fname.split("_")
            reg_id = pieces[0] + "_" + pieces[1] + "_" + pieces[2] + "_" + pieces[3] + ".tif"
        else: # train or test
            pieces = fname.split("_")
            reg_id = pieces[0] + "_" + pieces[1] 
        label = self.args.label_dict[reg_id]
        return label

    def __getitem__(self, idx):
        # this grabs a triplet
        # idx is the index in the triplet list
        if self.args.selfsup_mode != "sextuplet":
            triplet = self.triplet_names[idx]
            xa = self.retriever[triplet[0]][()]
            xn = self.retriever[triplet[1]][()]
            xd = self.retriever[triplet[2]][()]
            sample = {'anchor': xa, 'neighbor': xn, 'distant': xd}

            laba = self.get_label(triplet[0])
            labn = self.get_label(triplet[1])
            labd = self.get_label(triplet[2])
            label = {'anchor': laba, 'neighbor': labn, 'distant': labd}

            if self.transform:
                sample = self.transform(sample)
            if self.target_transform:
                label = self.target_transform(label)   
            return sample, label
        else:
            sextuplet = self.triplet_names[idx]
            triplet0, triplet1 = sextuplet[0], sextuplet[1]
            x0a = self.retriever[triplet0[0]][()]
            x0n = self.retriever[triplet0[1]][()]
            x0d = self.retriever[triplet0[2]][()]
            sample0 = {'anchor': x0a, 'neighbor': x0n, 'distant': x0d}
            x1a = self.retriever[triplet1[0]][()]
            x1n = self.retriever[triplet1[1]][()]
            x1d = self.retriever[triplet1[2]][()]
            sample1 = {'anchor': x1a, 'neighbor': x1n, 'distant': x1d}

            lab0a = self.get_label(triplet0[0])
            lab0n = self.get_label(triplet0[1])
            lab0d = self.get_label(triplet0[2])
            label0 = {'anchor': lab0a, 'neighbor': lab0n, 'distant': lab0d}
            lab1a = self.get_label(triplet1[0])
            lab1n = self.get_label(triplet1[1])
            lab1d = self.get_label(triplet1[2])
            label1 = {'anchor': lab1a, 'neighbor': lab1n, 'distant': lab1d}
            if self.transform:
                sample0 = self.transform(sample0)
                sample1 = self.transform(sample1)
            if self.target_transform:
                label0 = self.target_transform(label0)
                label1 = self.target_transform(label1)
            
            return sample0, sample1, label0, label1


class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """
    def __call__(self, sample):
        a, n, d = (torch.from_numpy(sample['anchor']).float(),
                   torch.from_numpy(sample['neighbor']).float(),
                   torch.from_numpy(sample['distant']).float())
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        return sample


class ToNormalizedTensor(object):
    """
    Channel-wise normalization or standardization in Pytorch. To be applied after ToFloatTensor.
        Below is an implementation for standardization with z-score
    """
    def __call__(self, sample):
        a, n, d = (sample['anchor'], sample['neighbor'], sample['distant'])
        # tensor([173.5706, 172.3651, 173.4979]) tensor([53.0683, 54.0479, 53.1360])
        mu = torch.FloatTensor([173.5706, 172.3651, 173.4979]).unsqueeze(1).unsqueeze(2)
        sig = torch.FloatTensor([53.0683, 54.0479, 53.1360]).unsqueeze(1).unsqueeze(2)
        a = (a - mu) / sig
        n = (n - mu) / sig
        d = (d - mu) / sig
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        return sample


def create_triplet_dataloader(args, patch_dir, triplet_names, shuffle=True, num_workers=4, normalize_flag=False):
    transform_list = [ToFloatTensor()]
    if normalize_flag == True:
        print("Normalizing with mu/sig (per channel)")
        transform_list.append(ToNormalizedTensor())
    transform = transforms.Compose(transform_list)
    dataset = TripletDataset(patch_dir, triplet_names, args, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def reduce_Z(Z, kmeans_model):
    h,w,d = Z.shape
    # print("Z shape:", h,w,d)
    cluster_labs = kmeans_model.labels_
    zero_id = np.max(cluster_labs) + 1
    Z_viz = np.zeros((h, w, 1)) + zero_id
    # plot clusters for image
    for i in range(h):
        for j in range(w):
            Zij = Z[i,j,:]
            if np.sum(Zij) > 0.0:
                Zij = Zij.reshape(1, -1)
                cluster = kmeans_model.predict(Zij)
                Z_viz[i,j,:] = cluster
    return Z_viz   


class EmbedDataset(Dataset):
    def __init__(self, data_dir, label_dict_path, split_list, mode, kmeans_model):
        self.data_dir = data_dir
        self.label_dict = utils.deserialize(label_dict_path)
        all_Zs = os.listdir(data_dir)
        self.Zs = [Z for Z in all_Zs if Z in split_list]
        self.mode = mode
        self.kmeans_model = kmeans_model

    def __len__(self):
        return len(self.Zs)
    
    def get_label(self, fname):
        id_num = fname.split(".npy")[0]
        reg_id = id_num.split("-")[1]
        label = self.label_dict[reg_id]
        return label
    
    def __getitem__(self, idx):
        Z_id = self.Zs[idx]        
        y = self.get_label(Z_id)
        Z = np.load(self.data_dir + "/" + Z_id)
        if self.mode == "fullZ":
            pass
        elif self.mode == "clusterZ":
            Z = reduce_Z(Z, self.kmeans_model)
        elif self.mode == "clusterbag":
            Z = reduce_Z(Z, self.kmeans_model)
            unique, counts = np.unique(Z, return_counts=True)
            Z = np.array(counts)
        elif self.mode == "meanpool":
            Z_vec = np.mean(Z, axis=2)
            one_channel = np.sum(Z, axis=(1,2))
            n_nonzero = np.count_nonzero(one_channel)
            n = Z.shape[0] * Z.shape[1]
            Z = Z_vec * (n/n_nonzero)
        elif self.mode == "maxpool":
            Z_vec = np.max(Z, axis=2)
            one_channel = np.sum(Z, axis=(1,2))
            n_nonzero = np.count_nonzero(one_channel)
            n = Z.shape[0] * Z.shape[1]
            Z = Z_vec * (n/n_nonzero)
        elif self.mode == "meanmaxpool":
            Z_vec = np.mean(Z, axis=2)
            one_channel = np.sum(Z, axis=(1,2))
            n_nonzero = np.count_nonzero(one_channel)
            n = Z.shape[0] * Z.shape[1]
            Z_mean = Z_vec * (n/n_nonzero)
            Z_vec = np.max(Z, axis=2)
            one_channel = np.sum(Z, axis=(1,2))
            n_nonzero = np.count_nonzero(one_channel)
            n = Z.shape[0] * Z.shape[1]
            Z_max = Z_vec * (n/n_nonzero)
            Z = np.stack([Z_mean, Z_max])

        return Z, y