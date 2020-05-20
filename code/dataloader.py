import os
import numpy as np
import random
import utils
import glob
from utils import labels_dict

class DataLoader(object):
    def __init__(self, path, batch_size, transfer=False, mode = 'full', normalize=True):
        self.mode = mode
        self.normalize = normalize
        self.path = path
        self.batch_size = batch_size
        self.transfer = transfer
        np.random.seed(231)
        if self.mode == 'dev':
            reg004_samples = [filename for filename in os.listdir(utils.data_dir + 'train/') if filename.startswith("reg004")]
            reg015_samples = [filename for filename in os.listdir(utils.data_dir + 'train/') if filename.startswith("reg015")]
            reg014_samples = [filename for filename in os.listdir(utils.data_dir + 'train/') if filename.startswith("reg014")]
            reg020_samples = [filename for filename in os.listdir(utils.data_dir + 'train/') if filename.startswith("reg020")]
            neg = list(np.random.choice(reg004_samples, 350)) + list(np.random.choice(reg015_samples, 350))
            pos = list(np.random.choice(reg014_samples, 350)) + list(np.random.choice(reg020_samples, 350))
            self.files = pos + neg
            random.shuffle(self.files)
        else:
            self.files = os.listdir(path)
            random.shuffle(self.files)
        

    def __iter__(self):
        i = 0
        while i < len(self.files):
            batch = []
            labels = []
            filenames =[]
            for f in self.files[i:i+self.batch_size]:
                try:
                    dat = np.load(self.path + f)
                except:
                    continue
                lab = self.get_label(f)
                if self.transfer:
                    dat_split = np.split(dat, dat.shape[0] // 3,  axis=0)
                    for dat in dat_split:
                        dat -= np.mean(dat, axis=(1,2)).reshape(-1,1,1)
                        dat /= (np.std(dat, axis=(1,2)).reshape(-1,1,1) + 1e-5)
                        batch.append(dat)
                        labels.append(lab)
                        filenames.append(f)
                else:
                    if self.normalize:
                        dat -= np.mean(dat, axis=(1,2)).reshape(-1,1,1)
                        dat /= (np.std(dat, axis=(1,2)).reshape(-1,1,1) + 1e-5)
                    batch.append(dat)
                    labels.append(lab)
                    filenames.append(f)

            batch = np.stack(batch, axis=0)
            labels = np.array(labels)
            yield (filenames, batch, labels)
            i += self.batch_size
    
    def get_label(self, fname):
        reg_id = fname.split('reg')[1].split('_')[0]
        label = utils.labels_dict[reg_id][1]
        return label
    
    
class TransferLoader(object):
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)

    def __iter__(self):
        for base in self.get_basenames():
            arrs = []
            for f in glob.glob(self.path + base + '*.npy'):
                arr = np.load(f)
                arrs.append(arr)
            arrs = np.stack(arrs, axis=0)
            yield arrs
            
    def get_basenames(self):
        return list(set([file.split('slice')[0] for file in self.files]))

    
if __name__ == "__main__":
    dataloader = DataLoader(utils.data_dir + 'test/', batch_size=2, transfer=True)
    for f, d, l in dataloader:
        print(f, d.shape, l.shape)

