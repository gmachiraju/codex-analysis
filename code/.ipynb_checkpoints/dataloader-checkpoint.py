import os
import numpy as np
import random
import utils
import glob

class DataLoader(object):
    def __init__(self, path, batch_size, transfer=False):
        self.path = path
        self.batch_size = batch_size
        self.files = os.listdir(path)
        random.shuffle(self.files)
        self.transfer = transfer

    def __iter__(self):
        i = 0
        while i < len(self.files):
            batch = []
            labels = []
            filenames =[]
            for f in self.files[i:i+self.batch_size]:
                dat = np.load(self.path + f)
                lab = self.get_label(f)
                if self.transfer:
                    dat_split = np.split(dat, dat.shape[0] // 3,  axis=0)
                    for dat in dat_split:
                        batch.append(dat)
                        labels.append(lab)
                        filenames.append(f)
                else:
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
