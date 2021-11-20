import os
import glob
import sys
import math
import numpy as np
from numpy import errstate, isneginf
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import tifffile as tiff
import gc # grabage collect for memory issues

# helpers
from utils import labels_dict
from dataloader import DataLoader
from preprocess import reshape_4dto3d, summarize_embed, analyze_embed, patchify


# Constants
#-----------

# dataset directory
data_dir = "/home/groups/plevriti/gautam/codex_analysis/codex-analysis/data_u54/primary"

# Sample options for train set visualization:
train_samples = ["004", "015", "014", "024", "020", "007", "008", "027", "034", "012"]


cmap = LinearSegmentedColormap.from_list(name='test', 
    colors=['darkgreen','green','yellow','red','darkred']
    )
#     colors=['darkred','red','yellow','green','blue','darkblue']
#     colors=['darkblue','blue','green','yellow','red','darkred']
#     colors=['darkblue','blue','green','yellow','red','darkred']


# Helper functions
#------------------

def find_manual_norm_bounds(data_dir, mode, sample_or_channel):
    
    if mode == "sample":
        filename = "reg" + sample_or_channel + "_montage.tif"
        im = tiff.imread(os.path.join(data_dir, filename))
        im_3d = reshape_4dto3d(im) 
        cyclechannel, imgwidth, imgheight = im_3d.shape

        # for non-infinity: add 1 to all values
        im_3d += np.uint16(1)

        print("manual norm requested")
        max_pix = np.uint16(1)
        min_pix = np.uint16(1e10)
        for i in range(cyclechannel):
            min_pix_i = np.min(im_3d[i,:,:])
            max_pix_i = np.max(im_3d[i,:,:])
            if min_pix_i < min_pix:
                min_pix = min_pix_i
            if max_pix_i > max_pix:
                max_pix = max_pix_i

        print("finished assigning norm:")
        print(min_pix, max_pix)
        return (min_pix, max_pix)
    
    elif mode == "channel":
        files = ["reg" + sample + "_montage.tif" for sample in train_samples]
        num_images = len(files)
        
        print("manual norm requested")
        max_pix = np.uint16(1)
        min_pix = np.uint16(1e10)
        for i in range(num_images):
            filename = files[i]
            im = tiff.imread(os.path.join(data_dir, filename))
            im_3d = reshape_4dto3d(im) 

            # for non-infinity: add 1 to all values
            im_3d += np.uint16(1)
            
            min_pix_i = np.min(im_3d[sample_or_channel,:,:])
            max_pix_i = np.max(im_3d[sample_or_channel,:,:])
            if min_pix_i < min_pix:
                min_pix = min_pix_i
            if max_pix_i > max_pix:
                max_pix = max_pix_i

        print("finished assigning norm:")
        print(min_pix, max_pix)
        return (min_pix, max_pix)
    
    

def visualize_1sample_allchannels(data_dir, sample, manual_norm=False):
    """
    visualize all channels for a given sample
    """
    filename = "reg" + sample + "_montage.tif"
    im = tiff.imread(os.path.join(data_dir, filename))
    im_3d = reshape_4dto3d(im) 
    cyclechannel, imgwidth, imgheight = im_3d.shape

    # for non-infinity: add 1 to all values
    im_3d += np.uint16(1)

    grid_cols = 4
    grid_rows = int(cyclechannel // grid_cols)
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(20, 80), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    if manual_norm == True:
        (min_pix, max_pix) = find_manual_norm_bounds(data_dir, "sample", sample)
        log_norm = LogNorm(vmin=min_pix, vmax=max_pix)
        a = np.floor(np.log10(min_pix))
        b = 1 + np.ceil(np.log10(max_pix))
        if a == float("inf") or a == float("-inf"): # shouldn't trip
            a = 0 # bottom of range is now 10^0, or 1
        cbar_ticks = [math.pow(10, i) for i in range(int(a), int(b))]

    for i, ax in zip(range(cyclechannel), axes.flat):
        if manual_norm == True:
            sns.heatmap(im_3d[i,:,:], ax=ax, cmap=cmap, vmin=min_pix, vmax=max_pix,
                       norm=log_norm, cbar_kws={"ticks": cbar_ticks})
        
        elif manual_norm == False:
            with np.errstate(divide='ignore'):
                logged = np.log10(im_3d[i,:,:])
                logged[isneginf(logged)] = np.uint16(1)
                sns.heatmap(logged, ax=ax, cmap=cmap)

        ax.set_title("channel " + str(i))
        ax.tick_params(left=False, bottom=False)
        ax.set_yticks([])
        ax.set_xticks([])
        print("finished plotting channel", i)
        
    plt.savefig('figs/data_heatmap_grids/samples/manualnorm_' + str(manual_norm) + 
                '/1samp_allch_' + str(sample) + '.png', bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    

    
def visualize_allsamples_1channel(data_dir, channel, manual_norm=False):
    """
    visualize all samples for a given channel
    """
    files = ["reg" + sample + "_montage.tif" for sample in train_samples]
    num_images = len(files)
    
    grid_cols = 2
    grid_rows = int(num_images // grid_cols)
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(20, 50), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    
    if manual_norm == True:
        (min_pix, max_pix) = find_manual_norm_bounds(data_dir, "channel", channel)
        log_norm = LogNorm(vmin=min_pix, vmax=max_pix)
        a = np.floor(np.log10(min_pix))
        b = 1 + np.ceil(np.log10(max_pix))
        if a == float("inf") or a == float("-inf"): # shouldn't trip
            a = 0 # bottom of range is now 10^0, or 1
        cbar_ticks = [math.pow(10, i) for i in range(int(a), int(b))]
        
    for i, ax in zip(range(num_images), axes.flat):
        filename = files[i]
        im = tiff.imread(os.path.join(data_dir, filename))
        im_3d = reshape_4dto3d(im) 
        cyclechannel, imgwidth, imgheight = im_3d.shape

        # for non-infinity: add 1 to all values
        im_3d += np.uint16(1)

        if manual_norm == True:
            sns.heatmap(im_3d[channel,:,:], ax=ax, cmap=cmap, vmin=min_pix, vmax=max_pix,
                       norm=log_norm, cbar_kws={"ticks": cbar_ticks})
        elif manual_norm == False:
            with np.errstate(divide='ignore'):
                logged = np.log10(im_3d[channel,:,:])
                logged[isneginf(logged)] = np.uint16(1)
                sns.heatmap(logged, ax=ax, cmap=cmap)

        ax.set_title(filename)
        ax.tick_params(left=False, bottom=False) 
        ax.set_yticks([])
        ax.set_xticks([])
        print("finished plotting sample", filename)
    
    plt.savefig('figs/data_heatmap_grids/channels/manualnorm_' + str(manual_norm) + 
                '/allsamp_1ch_' + str(channel) + '.png', bbox_inches='tight')
    plt.close(fig)
    gc.collect()

    

# Main function
#---------------

def main(flag):

    if flag == "sample":
        for ts in train_samples:
            
            print("-"*80 + "\n starting manual norm for train sample:", ts, "\n" + "-"*80)
            visualize_1sample_allchannels(data_dir, ts, manual_norm=True)
            print("done.")

            print("-"*80 + "\n starting auto norm for train sample:", ts, "\n" + "-"*80)
            visualize_1sample_allchannels(data_dir, ts)
            print("done.")
            
    
    elif flag == "channel":
        for channel in range(84):
                  
            print("-"*80 + "\n starting manual norm for channel:", channel, "\n" + "-"*80)
            visualize_allsamples_1channel(data_dir, channel, manual_norm=True)
            print("done.")

            print("-"*80 + "\n starting auto norm for channel:", channel, "\n" + "-"*80)
            visualize_allsamples_1channel(data_dir, channel)
            print("done.")
      
    else:
        print("Please choose a flag: {'sample', 'channel'}")
    
    
if __name__ == "__main__":
    flag = sys.argv[1]
    main(flag)
    
    
    