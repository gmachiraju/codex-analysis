import glob
import os
import sys
import tifffile as tiff
import numpy as np
import pandas as pd
import pdb

from scipy import ndimage as ndi
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import gc 
from PIL import Image

import argparse
import h5py
# import torch
# from torchvision import transforms
import cv2

# only toggle for python3.6
#==========================
# import openslide

# import stain_norm
# import torchstain
# from skimage.filters import gabor_kernel
# from skimage.feature import ORB, match_descriptors

import utils
from utils import labels_dict, reg_dict, ctrl_dict, str2bool, deserialize



normalize_refpath = "/home/groups/plevriti/gautam/codex_analysis/codex-analysis/code/ref_imgs/target.png"

# constants
random.seed(100)
HW = 96

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def filter_bank(im):
    # prepare filter bank kernels
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    # prepare reference features
    ref_feats = np.zeros((1, len(kernels), 2), dtype=np.double)
    ref_feats[0, :, :] = compute_feats(im, kernels)
    return ref_feats


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    # Returns a byte-scaled image - old source code for scipy 
    """
    Byte scales an array (image).

    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.

    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.

    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.

    Examples
    --------
    >>> img = array([[ 91.06794177,   3.39058326,  84.4221549 ],
                     [ 73.88003259,  80.91433048,   4.88878881],
                     [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)

    """
    if data.dtype == np.uint8:
        return data

    if high < low:
        raise ValueError("`high` should be larger than `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data * 1.0 - cmin) * scale + 0.4999
    bytedata[bytedata > high] = high
    bytedata[bytedata < 0] = 0
    return np.cast[np.uint8](bytedata) + np.cast[np.uint8](low)


def reshape_4dto3d(im):
    """
    taking original 4D tiff im and converting to 3D numpy arrays
    """
    cycle, channel, imgwidth, imgheight = im.shape
    # reshaped image
    im_reshaped = im.reshape(cycle * channel, imgwidth, imgheight)
        
    return im_reshaped

def get_gabors(im, cyclechannel):
    # get gabor embeddings
    gabors = []
    for i in range(cyclechannel):
        gabors.extend(filter_bank(im[i,:,:]))
    return gabors

def get_orbs(im, cyclechannel):
    # get ORB embeddings
    detector_extractor = ORB(n_keypoints=10)
    orbs = []
    for i in range(cyclechannel):
        detector_extractor.detect_and_extract(im[i,:,:])
        orbs.extend(detector_extractor.scales) 
    return orbs

def get_densities(im, cyclechannel, n=1000):
    # get random sample of densities/channel histograms
    dens = []
    for i in range(cyclechannel):
        rs = random.sample(im[i,:,:], n)
        dens.extend(rs) 
    return dens


def summarize_embed(data_dir, feat, mode):
    labels = []
    study_arms = []
    embed_dict = defaultdict(list)      
    shapes = []
    idxs = [] # reg-X
    
    files = os.listdir(data_dir)
    if mode == "image":
        files = [filename for filename in files if filename.endswith(".tif") == True]
    elif mode == "patch":
        files = [filename for filename in files if filename.endswith(".npy") == True]
        files = random.choices(files, k=1000) # random sample of patches
        print("Processing:", len(files), "patches")
    else:
        print("error: type of image in directory not clear: choose patch or image")
        pass
        
    for filename in files:
        if mode == "image":
            print("Processing:", filename)
            im = tiff.imread(os.path.join(data_dir, filename))  # read file in
            im_3d = reshape_4dto3d(im)                          # reshape IMAGES to 3D arrays (instead of 4D)
        elif mode == "patch":
            im_3d = np.load(os.path.join(data_dir, filename))
            
        cyclechannel, imgwidth, imgheight = im_3d.shape
        shapes.append(im_3d.shape)
#         im_3d_8bit = bytescale(im_3d) # gray scale normalization (0,255) --> use for KP detection

        if feat == "median":  # get median values of all channels for 3D images
            e = list(np.median(im_3d, axis=[1,2])) 
            embed_dict[feat].append(e) 
        else:
            print("error: choose a supported summarization: median")
            # note: eventially add ORB, Gabor, densities, LoG - pyradiomics (https://pyradiomics.readthedocs.io/en/latest/radiomics.html#module-radiomics.featureextractor)
            continue
        
        # Isolate case name
        im_folder = filename.split(".")[0]  # "reg{idx}_montage"
        im_id = im_folder.split("_")[0]  # "reg{idx}""
        idx = im_id.split("reg")[1]  # {idx}
        study_arm, label = labels_dict[idx]

        idxs.append(idx)
        study_arms.append(study_arm)
        labels.append(label)
        
    # MAKE DATAFRAME: only supporting median right now....
    meds = embed_dict[feat]
    print("number of images:", len(meds), "| number of channels/image:", len(meds[0]))
    inds = ["im"+str(i) for i in range(len(meds))]
    cols = ["ch"+str(i) for i in range(len(meds[0]))]    
    meds_df = pd.DataFrame(meds, index=inds, columns=cols)
    
    # get the dimensions of the images and get a histogram
    shapes_df = pd.DataFrame(shapes, index=inds, columns=["channels", "Height", "Width"])
    shapes_df["labels"] = labels
    shapes_df["splits"] = study_arms
    shapes_df["IDs"] = idxs
    pixels = []
    for s in shapes:
        pixels.append(int(int(s[1])*int(s[2])))
    shapes_df["pixels"] = pd.Series(pixels, index=shapes_df.index)
    
    # combine
    summary_df = meds_df.merge(shapes_df, left_index=True, right_index=True, how='inner')
    summary_df = summary_df.reset_index()

    return study_arms, summary_df


def analyze_embed(study_arms, summary_im_df, mode, norm=False):
    """
    Plotting the image and patch summary
    """
    
    summary_im = summary_im_df.describe()
    
    val_im_df = summary_im_df[summary_im_df["splits"] == "val"]
    test_im_df = summary_im_df[summary_im_df["splits"] == "test"]
    train_im_df = summary_im_df[summary_im_df["splits"] == "train"] # only want to analyze statistics of training

    n_patients = summary_im_df.shape[0]
    n_val = val_im_df.shape[0]
    n_test = test_im_df.shape[0]
    n_train = train_im_df.shape[0]

    print("total images/patients:", n_patients)
    print("validation images:", n_val)
    print("test images:", n_test)
    print("train images:", n_train)
    
    fig = plt.figure(figsize=(10,8))
    axes = fig.gca()
    summary_im_df[["Height", "Width", "channels", "labels", "pixels"]].hist(ax=axes)
    plt.show()
    
    cols = ["ch"+str(i) for i in range(summary_im_df["channels"][0])]    
    train_im_dfl = pd.melt(train_im_df, id_vars=['index', "IDs", "labels", "splits"], value_vars=cols, var_name='channel', value_name='values')

    if norm == True:
        norm_str = "(normalized)"
    elif norm == False:
        norm_str = "(pre-normalized)"
    
    # plot violin plots of medians
    plt.figure(figsize=(6, 20))
    sns.violinplot(data=train_im_dfl, y="channel", x='values')
    plt.title("Channel-wise medians of training " + mode + "es " + norm_str)
    plt.savefig("figs/" + mode + "_med_violin.png")
    plt.show()

    # tsne projection
    tsne_df = train_im_df[cols].copy()
    tsne_array = tsne_df.values
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(tsne_array)
    
    tsne_df['tsne-1'] = tsne_results[:,0]
    tsne_df['tsne-2'] = tsne_results[:,1]
    tsne_df["labels"] = train_im_df["labels"]
    tsne_df["IDs"] = train_im_df["IDs"].astype(np.int)

    # plotting
    plt.figure(figsize=(5,5))
    sns.scatterplot(
        x="tsne-1", y="tsne-2",
        hue="labels",
        data=tsne_df,
        alpha=1
    )
    plt.title("t-SNE projection of " + mode + "es by label " + norm_str)
    plt.show()
    
    plt.figure(figsize=(5,5))
    sns.scatterplot(
        x="tsne-1", y="tsne-2",
        hue="IDs",
        data=tsne_df,
        palette=sns.color_palette("colorblind", len(set(tsne_df["IDs"]))),
        alpha=0.7
    )
    plt.title("t-SNE projection of " + mode + "es by image ID " + norm_str)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.show()
    
    return summary_im, train_im_df, train_im_dfl



#####################
# PATCHING CODE
#####################

def random_rotate(tile):
    num = np.random.choice([1, 2, 3])
    return np.rot90(tile, k=num, axes=(1, 2)), num


def axis_rotate(tile, num):
    return np.rot90(tile, k=num, axes=(1, 2))


def random_reflect(tile):
    num = np.random.choice([0, 1])
    return np.flip(tile, num), num


def axis_reflect(tile, axis):
    return np.flip(tile, axis)


def fit_normalizer(norm_type="macenko"):
    if norm_type == "macenko":
        # reference image
        target = cv2.imread(normalize_refpath)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # -- please install torchstain and then uncomment the below lines 
        # transform_fn = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Lambda(lambda x: x*255)
        # ])

        # normalize_fn = torchstain.MacenkoNormalizer(backend='torch')
        # normalize_fn.fit(transform_fn(target))
        print("torchstain package not imported. Exiting...")
        exit()

    elif norm_type == "vahadane":
        normalize_fn = stain_norm.Normalizer()
        normalize_fn.fit(normalize_refpath)
        transform_fn = lambda x: x # dummy function

    else:
        print("please choose valid normalization type, or default to vahadane")
        exit()

    return transform_fn, normalize_fn


def normalize(im, dataset_name, transform_fn=None, normalize_fn=None):
    # really a standardization process
    if dataset_name == "u54codex":
        # toss blank channels if they exist
        null_channels = [82, 81, 79, 78, 77, 74, 73, 69, 65]
        im = np.delete(im, null_channels, axis=0)
    
        channel_means1 = np.mean(im, axis=(1,2)).reshape(-1,1,1)
        # channel_means2 = np.mean(im, axis=(1,2)).reshape(im.shape[0], 1, 1)
        # print(channel_means1 == channel_means2)
        
        new_im = im - channel_means1
        new_im /= (np.std(new_im, axis=(1,2)).reshape(-1,1,1) + 1e-5)

    elif dataset_name == "cam":
        channel, imgwidth, imgheight = im.shape
        im = im.reshape(imgwidth, imgheight, channel)
        im = transform_fn(im)
        # pdb.set_trace()

        im, H, E = normalize_fn.normalize(I=im, stains=True)
        del H
        del E
        im = im.cpu().detach().numpy()
        new_im = im.reshape(channel, imgwidth, imgheight)

    else:
        new_im = im

    return new_im



def toss_blank_tile(im, q, dataset_name="u54codex", tol=0.05, thresh_tile=None, bg_remove_flag=False):
    """
    Toss out tiles with >=80% low intensity pixels
    OR toss out middle values..... you set the filtering function!
    """
    if dataset_name == "u54codex" :
        summed = np.sum(im, 0)
        if np.mean(summed < q) > 0.8:
            return True
        return False

    elif dataset_name == "cam":
        summed = np.sum(im, 0)
        if bg_remove_flag == True: # some are background-removed images!
            if np.mean(summed) == 0: # "black"/blank region in val set
                return True
            elif np.std(summed) == 0: # black or white tiles
                return True
            elif np.any(summed == 0):
                return True
        else:
            if np.mean(summed > q) > 0.5 and np.mean(thresh_tile) < 0.6: # 0.75,0.5
                return True
        return False

    elif dataset_name == "controls":
        mean_val = np.mean(im)
        if (mean_val < q+tol and mean_val > q-tol) or (np.isnan(mean_val)):
            # print("mean=", np.mean(im), "--> tossing b/c is NaN or mean was too close to", q)
            return True
        return False


def crop_coords(im, height, width, shift="noshift", doubling_flag=False):
    """
    solely returns coordinates for an image
    notes: https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
    """
    # # translate image to the right and down 50%
    # if shift == "50shift":
    #     im = im[:, args.HW // 2:, args.HW // 2:]

    channel, imgheight, imgwidth = im.shape

    if shift == "noshift" or (shift=="50shift" and doubling_flag==True):
        # top-bottom, left-right
        for i in range(imgheight // height):
            for j in range(imgwidth // width):
                yield [i * height, (i + 1) * height, j * width, (j + 1) * width] #x1,x2,y1,y2
    
    elif shift == "50shift": 
        eps = height // 2
        for i in range(imgheight // height):
            for j in range(imgwidth // width):
                # if ((((i + 1) * height) + eps) < imgheight) and ((((j + 1) * width) + eps) < imgwidth):
                yield [(i * height) + eps, ((i + 1) * height) + eps, (j * width) + eps, ((j + 1) * width) + eps] #x1,x2,y1,y2
    


def process_patch(im, im_id, q_low, patches, shift, args, patch_tossed_i, patch_tossed, patch_count_i, thresh_im=None):

    saved_flag = False
    channel, imgheight, imgwidth = im.shape

    # print("Your image is of shape:", im.shape)
    # print(im_id)

    # # debugging
    # print("your image's shape is:", im.shape)
    # pdb.set_trace()
    # np.save(args.save_dir + '/' + "tester-img.npy", np.asarray(im, dtype=np.uint8))
    # print(im[ 12096:12320,2464:2688, :])
    # pdb.set_trace()

    # make HDF5 dataset
    if args.hdf5_flag == True:
        hdf5_filename = args.study_arm + '.hdf5'
        hdf5_savepath = args.save_dir + '/' + hdf5_filename
        hf = h5py.File(hdf5_savepath, 'a') # open a hdf5 file
        # hf.close()

    # make grid
    num_cols = int(imgwidth // args.HW) # should be floor right? used to be ceil
    num_rows = int(imgheight // args.HW)
    grid = np.zeros((num_rows, num_cols))

    # if args.dataset_name == "cam":
    #     transform_fn, normalize_fn = fit_normalizer(norm_type="macenko")

    if args.dataset_name == "cam":
        reshaped_im = im.reshape(imgheight, imgwidth, channel)

    # repeating data examples - just for controls
    if args.dataset_name == "controls":
        d_flag = True
    else:
        d_flag = False

    # pdb.set_trace()
    summed = np.sum(im, 0)
    if np.any(summed == 0):
        detected_zero_flag = True
        print("detected 0-pixels...")
    else:
        detected_zero_flag = False


    # iterate over patches
    #----------------------
    for k, coords_list in enumerate(crop_coords(im, args.HW, args.HW, shift=shift, doubling_flag=d_flag)):

        # get patch coordinate
        [x1, x2, y1, y2] = coords_list
       
        curr_row = int(x2 / args.HW) - 1 #int((num_rows // (k+1)))
        curr_col = int(y2 / args.HW) - 1 #int(((k+1) % num_cols) - 1)

        # tile/patch slice!
        if args.dataset_name == "cam":
            tile = reshaped_im[x1:x2, y1:y2, :]
            tileh, tilew, tilec = tile.shape
            tile = tile.reshape(tilec, tileh, tilew)
        else:
            tile = im[:, x1:x2, y1:y2]

        # resize if needed - not implemented yet
        # if args.resize_flag == True and args.resize_HW is not None:
        #     # print("resizeing the saved patch")
        #     pass

        if args.dataset_name == "cam":
            thresh_tile = thresh_im[x1:x2, y1:y2]

        #debugging
        # if k == 0:
        #     np.save(args.save_dir + '/' + "tester-pre.npy", np.asarray(tile, dtype=np.uint8))
        #     # pdb.set_trace()

        # checks for valid tile/patch
        #-----------------------------
        # is patch the valid size?
        if tile.shape[1] != args.HW or tile.shape[2] != args.HW: # not square patch
            patch_tossed_i += 1
            patch_tossed += 1
            continue

        # for cam, want to ignore borders
        if args.dataset_name == "cam":
            # print("image is of size:", im.shape)
            # print("curr_row", curr_row, "imgheight", imgheight, "imgwidth", imgwidth, x1, x2)
            if args.study_arm == "train":
                if curr_row < (imgheight * (1/6) / args.HW) or curr_row > (imgheight * (5/6) / args.HW):
                    patch_tossed_i += 1
                    patch_tossed += 1
                    del tile
                    #print("patch in top or bottom 1/6th of slide. skipping...")
                    continue
                if curr_col < (imgwidth * (1/8) / args.HW) or curr_col > (imgwidth * (7/8) / args.HW):
                    patch_tossed_i += 1
                    patch_tossed += 1
                    del tile
                    continue

        if args.filtration_type == "background": # if not, we keep all patches including noisy background
            if toss_blank_tile(tile, q_low, args.dataset_name, thresh_tile=thresh_tile, bg_remove_flag=bool(args.bg_remove_flag*detected_zero_flag)): # blank patch
                patch_tossed_i += 1
                patch_tossed += 1
                del tile
                continue

        # debugging
        # print("your patch's shape is:", tile.shape)
        # np.save(args.save_dir + '/' + "tester.npy", np.asarray(tile, dtype=np.uint8))
        # tile_str = "%s_patch%s_coords%s-%s-[%s:%s,%s:%s]_%s_noaug" % (im_id, k, curr_row, curr_col, x1, x2, y1, y2, shift)
        # print(tile_str)
        # # pdb.set_trace()
        # exit()


        #NORMALIZE?
        #----------
        # if args.dataset_name == "cam":
        #     tile = normalize(tile, args.dataset_name, transform_fn, normalize_fn)

        # non-aug patch
        #---------------
        tile_str = "%s_patch%s_coords%s-%s-[%s:%s,%s:%s]_%s_noaug" % (im_id, k, curr_row, curr_col, x1, x2, y1, y2, shift)
        patches.append(tile_str)
        if args.prepatch_flag == True:
            if args.hdf5_flag == True:
                # hf = h5py.File(hdf5_savepath, 'a') # open again
                dset = hf.create_dataset(tile_str, data=tile)  # write the data to hdf5 file
                # hf.close()  # close the hdf5 file
                del dset
            else:
                np.save(args.save_dir + '/' + tile_str, tile)
            saved_flag = True

        patch_count_i += 1 
        # update grid
        grid[curr_row, curr_col] = 1

        # add any tiles/patches if augmentation enabled
        if args.study_arm == "train": 
            if args.augment_level == "high":
                for rot in [1, 2, 3]:
                    tile_str = "%s_patch%s_coords%s-%s-[%s:%s,%s:%s]_%s_rot%s" % (im_id, k, curr_row, curr_col, x1, x2, y1, y2, shift, rot)
                    patches.append(tile_str)
                    if args.prepatch_flag == True:
                        tile = axis_rotate(tile, rot)
                        if args.hdf5_flag == True:
                            # hf = h5py.File(hdf5_savepath, 'a') # open again
                            dset = hf.create_dataset(tile_str, data=tile)  # write the data to hdf5 file
                            # hf.close()  # close the hdf5 file
                            del dset
                        else:
                            np.save(args.save_dir + '/' + tile_str, tile)
                        saved_flag = True

                for refl in [0, 1]:
                    tile_str = "%s_patch%s_coords%s-%s-[%s:%s,%s:%s]_%s_refl%s" % (im_id, k, curr_row, curr_col, x1, x2, y1, y2, shift, refl)
                    patches.append(tile_str)
                    if args.prepatch_flag == True:
                        tile = axis_reflect(tile, refl)
                        if args.hdf5_flag == True:
                            # hf = h5py.File(hdf5_savepath, 'a') # open again
                            dset = hf.create_dataset(tile_str, data=tile)  # write the data to hdf5 file
                            # hf.close()  # close the hdf5 file
                            del dset
                        else:
                            np.save(args.save_dir + '/' + tile_str, tile)
                        saved_flag = True

            elif args.augment_level == "low":
                # Random rotation
                tile, rot = random_rotate(tile)
                tile_str = "%s_patch%s_coords%s-%s-[%s:%s,%s:%s]_%s_rot%s" % (im_id, k, curr_row, curr_col, x1, x2, y1, y2, shift, rot)
                patches.append(tile_str)
                if args.prepatch_flag == True:
                    if args.hdf5_flag == True:
                        # hf = h5py.File(hdf5_savepath, 'a') # open again
                        dset = hf.create_dataset(tile_str, data=tile)  # write the data to hdf5 file
                        # hf.close()  # close the hdf5 file
                        del dset
                    else:
                        np.save(args.save_dir + '/' + tile_str, tile)
                    saved_flag = True
     
                # Random reflection
                tile, refl = random_reflect(tile)
                tile_str = "%s_patch%s_coords%s-%s-[%s:%s,%s:%s]_%s_refl%s" % (im_id, k, curr_row, curr_col, x1, x2, y1, y2, shift, refl)
                patches.append(tile_str)
                if args.prepatch_flag == True:
                    if args.hdf5_flag == True:
                        # hf = h5py.File(hdf5_savepath, 'a') # open again
                        dset = hf.create_dataset(tile_str, data=tile)  # write the data to hdf5 file
                        # hf.close()  # close the hdf5 file
                        del dset
                    else:
                        np.save(args.save_dir + '/' + tile_str, tile)
                    saved_flag = True

        del tile
        # print(saved_flag)

    # make HDF5 dataset
    if args.hdf5_flag == True:
        hf.close()  # close the hdf5 file

    if shift == "noshift" and args.verbosity_flag == True:
        print_grid(grid)

    print("-->", int(np.sum(grid)), "patches counted from grid analysis of", shift, "sampling!")
    print("--> In an image with", imgheight, "x", imgwidth, "pixels, we subdivide into", num_rows, "x", num_cols, "patches (of size "+str(args.HW)+")")
    print()

    return patch_tossed_i, patch_tossed, patch_count_i, patches


def print_grid(grid):
    # grid is a numpy array of 1s and 0s
    grid_str = []
    for row in grid:
        row_str = []
        for el in row:
            if el == 1:
                row_str.append("X")
            elif el == 0:
                row_str.append(" ")
        grid_str.append(row_str)

    num_cols = len(row_str)
    print("+" + "-"*num_cols + "+")
    for row_str in grid_str:
        row_str = "|" + ''.join(row_str) + "|" 
        print(row_str)
    print("+" + "-"*num_cols + "+")


def otsu_thresh(im):
    #rgb -> gray
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Otsu's thresholding
    ret,th = cv2.threshold(g,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # pdb.set_trace()
    # thresh_im = np.multiply(np.array(im), np.repeat(np.expand_dims(1-th,2),3,2))
    return 1-th


def patchify(args, level=3):
    # creates a list of all patches per image if prepatch_flag=True
    # else: creates and saves patches 

    # future: implement class balancing options given ratio of imbalance between positives/negatives 
    # -- or do whatever level of augmentation and then truncate/delete patches to make both class counts equal

    # future: add blocked version to both prepatch flags. The patch_list should be saved to account for this blocking.

    if (args.save_dir == None or args.save_dir.lower() == "none") and args.prepatch_flag == True:
        print("Detected preppatch_flag=True... But save_dir is missing. Please specify directory")
        exit()

    if args.save_dir != None and args.prepatch_flag == True:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
            print("creating a directory at", args.save_dir)
        else: # exists
            if args.overwrite_flag == False:
                print("Chosen to not overwrite data! Continuing...")
                return
            elif args.overwrite_flag == True:
                print("overwriting old patches!")

                # pdb.set_trace()
                for pn in os.listdir(args.save_dir):
                    if os.path.isfile(args.save_dir + "/" + pn):
                        os.remove(args.save_dir + "/" + pn)
                
                # pdb.set_trace()
                if len(os.listdir(args.save_dir)) == 0:
                    print("successfully cleared out old patches!")
                else:
                    print("Issue with deleting all patches... proceeding anyway")
                    # exit()

        print("Now, creating and storing patches there... let's begin!")
    
    print("Using dataset at:", args.data_dir)
    files = os.listdir(args.data_dir)
    if len(files) == 0:
        print("Error! Directory empty! No patching can be done. Exiting...")
        exit()

    image_count = 0
    patch_tossed = 0
    patches = []

    if args.dataset_name == "controls" or args.dataset_name == "cam":
        splitlab_dict = {}

    # if args.dataset_name == "cam":
    #     transform_fn, normalize_fn = fit_normalizer(norm_type="macenko")


    #################
    # FILE ITERATION
    #################
    for filename in files:
        print("="*60)
        print('\nCurrently on file {}'.format(filename))

        # run only on valid files in dir
        if args.dataset_name == "u54codex":
            if not filename.endswith(".tif"):
                continue

        elif (args.dataset_name == "controls") or (args.dataset_name == "cam"):
            if (not filename.endswith(".npy")) and (not filename.endswith(".tif")):
                continue

        # Read File
        if args.dataset_name == "u54codex":
            im = tiff.imread(os.path.join(args.data_dir, filename)) 
            splitlab_dict = utils.reg_dict
            # reshape
            cycle, channel, imgwidth, imgheight = im.shape
            im = im.reshape(cycle * channel, imgwidth, imgheight)

        elif args.dataset_name == "controls" or args.dataset_name == "cam":
            if filename.endswith(".npy"):
                im = np.load(os.path.join(args.data_dir, filename))

            elif filename.endswith(".tif"):
                im_pyramid = openslide.OpenSlide(os.path.join(args.data_dir, filename))
                desired_downsample = im_pyramid.level_downsamples[level]
                desired_dims = im_pyramid.level_dimensions[level]
                print(desired_downsample, desired_dims)
                rgba_img = im_pyramid.read_region((0,0), level, desired_dims)
                del im_pyramid
                rgba_arr = np.asarray(rgba_img)
                rgb_arr = cv2.cvtColor(rgba_arr, cv2.COLOR_RGBA2RGB)
                im = np.asarray(rgb_arr)

            # debugging
            # print("your image's shape is:", im.shape)
            # np.save(args.save_dir + '/' + "tester-img-loaded.npy", np.asarray(im, dtype=np.uint8))
            # print(im[12096:12320, 2464:2688, :])
            # pdb.set_trace()
            try:
                imgwidth, imgheight, channel = im.shape
            except ValueError: # dimension mismatch
                if len(im.shape) == 2:
                    im = np.expand_dims(im, axis=2) # reflect above
                    imgwidth, imgheight, channel = im.shape

            if args.dataset_name == "cam":
                 # perform otsu
                im_copy = np.copy(im)
                thresh_im = otsu_thresh(im)
                im = np.copy(im_copy)
                del im_copy
                print("Finished Otsu thresholding!")
            
            im = im.reshape(channel, imgwidth, imgheight)

        # debugging
        # print("your image's shape is:", im.shape)
        # np.save(args.save_dir + '/' + "tester-img-prepatch.npy", np.asarray(im, dtype=np.uint8))
        # im = im.reshape(imgwidth, imgheight, channel)
        # print(im[12096:12320, 2464:2688, :])
        # pdb.set_trace()
   
        # parse names
        if args.dataset_name == "cam":
            im_id = filename.split(".")[0]  # "label_{idx}", e.g. normal_001
            idx = im_id.split("_")[1]  # {idx}
            if "patient" in im_id: # validation from cam17
                splitlab_dict = deserialize(args.label_dict)

            else: # training/testing
                if "normal" in im_id:
                    splitlab_dict[im_id] = 0
                else: # tumor
                    splitlab_dict[im_id] = 1


        elif args.dataset_name == "u54codex":
            im_folder = filename.split(".")[0]  # "reg{idx}_montage"
            im_id = im_folder.split("_")[0]  # "reg{idx}""
            idx = im_id.split("reg")[1]  # {idx}
                    
            if args.study_arm != splitlab_dict[idx][0]:
                continue

        elif args.dataset_name == "controls":
            # pathology-like
            if ("morphological" in args.cache_name) and ("superpixels" in args.cache_name):
                histopath_flag = True
                # S11100003_P495_subject310_stitched_labelS.npy
                im_folder = filename.split(".")[0]  # "S11100003_P495_subject310_stitched_labelS"
                im_id = "-".join(im_folder.split("_")[0:3])  # "S11100003_P495_subject310""
                # idx = im_id.split("subject")[1]  # {idx}
                if "labelA" in im_folder:
                    splitlab_dict[im_id] = 0
                else:
                    splitlab_dict[im_id] = 1
                print("histopath?",histopath_flag,". Image:", im_id, ". label=", splitlab_dict[im_id])

            else: # non-pathology data
                histopath_flag = False
                im_folder = filename.split(".")[0]  # "reg{idx}_XXX"
                im_id = im_folder.split("-")[0]  # "reg{idx}""
                idx = im_id.split("reg")[1]  # {idx}

                # dictionary creation
                if "morphological" in args.cache_name: 
                    if "canary" in im_folder:
                        if any(word in im_folder for word in ["sin", "cos", "affine"]):
                            splitlab_dict[im_id] = 1
                        else:
                            splitlab_dict[im_id] = 0
                    else:
                        splitlab_dict[im_id] = 1

                elif "guilty" in args.cache_name: # just for guilty superpixels
                    if "guilty" in im_folder:
                        splitlab_dict[im_id] = 1 
                    else:
                        splitlab_dict[im_id] = 0

                elif "fractal" in args.cache_name:
                    if "fractal" in im_folder:
                        splitlab_dict[im_id] = 1 
                    else:
                        splitlab_dict[im_id] = 0

                else:
                    temp = im_folder.split("-")[-2]
                    if temp == "hot":
                        splitlab_dict[im_id] = 1                    
                    elif temp == "cold":
                        splitlab_dict[im_id] = 0

            print("current label dict length:", len(splitlab_dict))
      

        print('Original image dimensions (in pixels): {}'.format(im.shape) + "\n" + "="*60)
        image_count += 1
        # channel, imgheight, imgwidth = im.shape

        # Normalize Image + toss blank channels
        if args.dataset_name == "u54codex":
            im = normalize(im, args.dataset_name)

        # skipping stain normalization for now
        #---------------------------------------
        # elif args.dataset_name == "cam":
        #     im = normalize(im, args.dataset_name, transform_fn, normalize_fn)
        #     print("finished normalizing the path image")

        # find low intensity threshold
        if args.filtration_type == "background":
            # Identify low-intensity threshold
            if args.dataset_name == "u54codex":
                im_sum = np.sum(im, axis=0) # Does this make sense?
                q_low = np.quantile(im_sum, q=0.1)
            elif args.dataset_name == "cam":
                im_sum = np.sum(im, axis=0) # calculating summed intensities over all channels
                q_low = np.quantile(im_sum, q=0.08) # bump up the threshold a bit from codex

            if args.dataset_name == "controls":
                if histopath_flag == False:
                    q_low = 0.5
                else:
                    q_low = 0 # black/zero background in pathology like images
        else:
            q_low = None # dummy
            
        patch_tossed_i, patch_count_i = 0, 0

        #==========
        # PATCHING
        #==========
        # printable grid

        shifts = ["noshift", "50shift"]
        for shift in shifts:
            
            if shift == "noshift":
                print('Left-edge cropping...')
            else:
                print('50-percent shifted cropping...')
            patch_tossed_i, patch_tossed, patch_count_i, patches = process_patch(im, im_id, q_low, patches, shift, args, patch_tossed_i, patch_tossed, patch_count_i, thresh_im)

        print('Images completed: {}, Previous image patches (non-augmented): {}, total patches: {}, tossed: {}, total tossed: {}\n'.format(image_count, patch_count_i, len(patches), patch_tossed_i, patch_tossed))
        # pdb.set_trace()
        gc.collect()

        # # TOGGLE OFF IF NOT TESTING
        # break
    
    # always serialize if want to keep order
    print("Serializing patch list!")
    utils.serialize(patches, args.cache_dir + "/outputs/" + args.study_arm + "-" + args.cache_full_name + "-patchlist.obj")    
    
    if args.dataset_name == "controls" or args.dataset_name == "cam":
        print("Serializing label dictionary for controls")
        utils.serialize(splitlab_dict, args.cache_dir + "/outputs/" + args.study_arm + "-" + args.cache_full_name + "-labeldict.obj")    


#===============
# Ground truths
#===============
def generate_ground_truths(path):
    gt_dict = {}
    print("Processing a total of", len(os.listdir(path)), "patches")

    for k, pn in enumerate(os.listdir(path)):
                    
        contents = pn.split("_")
        # print(contents)
        # pdb.set_trace()
        
        regi = contents[0]
        patchnum = int(contents[1].split("patch")[1])
        coords = contents[2]
        shift = contents[3]
        aug = contents[4].split(".npy")[0]
        if aug != "noaug":
            continue # only interested in non-augmented patches
        

        ii = int(coords.split("-")[0].split("coords")[1])
        jj = int(coords.split("-")[1])
        
        if k > 0 and k % 1000 == 0: 
            print("finished processing means of", k, "patches!")

        # adding to dict
        if regi not in gt_dict: 
            # reg means list, shift means list, maxH, maxW
            gt_dict[regi] = [[], [], ii, jj]
        else:
            # update max dims
            if ii > gt_dict[regi][2]:
                gt_dict[regi][2] = ii
            if jj > gt_dict[regi][3]:
                gt_dict[regi][3] = jj
            
            # assign mean
            try:
                patch = np.load(path + "/" + pn)
                patch_mean = np.mean(patch.squeeze())
            except ValueError or TypeError:
                print("detected corrupted patch, skipping...")
                continue
            if shift == "noshift":
                gt_dict[regi][0].append((ii,jj,patch_mean))
            elif shift == "50shift":
                gt_dict[regi][1].append((ii,jj,patch_mean))
            del patch, patch_mean

            # # for debugging 
            # if k == 1000:
            #     return gt_dict
        
    return gt_dict


def get_imgdims(img_path, HW):
    imgdim_dict = {}

    for k,im_str in enumerate(os.listdir(img_path)):
        print(im_str)
        # pdb.set_trace()

        if "subject" in im_str: # hard code for pathology controls
            im_folder = im_str.split(".")[0]  # "S11100003_P495_subject310_stitched_labelS"
            regi = "-".join(im_folder.split("_")[0:3])  # "S11100003_P495_subject310""

        elif "reg" in im_str: # hard code for non-pathology controls
            regi = im_str.split("-")[0]

        im = np.load(img_path + "/" + im_str)
        maxH, maxW = im.shape[0:2]
        
        num_cols = int(np.floor(maxW / HW))
        num_rows = int(np.floor(maxH / HW))
        
        print("expecting patch grid of", num_rows, "x", num_cols, "for a", maxH, "x", maxW, "image")
        imgdim_dict[regi] = [num_rows, num_cols]
        del im

    return imgdim_dict
        

def generate_ppm_ground_truths(gt_dict, imgdim_dict, buff=0):
    ppmgt_dict = {}
    print(gt_dict)
    print(imgdim_dict)

    for regi in gt_dict.keys():
        ppmgt_dict[regi] = [np.ones((imgdim_dict[regi][0]+1, imgdim_dict[regi][1]+1)) *-1, np.ones((imgdim_dict[regi][0]+1, imgdim_dict[regi][1]+1)) *-1]
        
        print("Generating Ground Truth PPM for image:", regi)
        print("dims:", imgdim_dict[regi][0], imgdim_dict[regi][1])
        print("num patches:",len(gt_dict[regi][0]), len(gt_dict[regi][1]))
        print("maxes:", gt_dict[regi][2],gt_dict[regi][3])

        for p in gt_dict[regi][0]: # no shift patches
            ii,jj = p[0], p[1]
            pmean = p[2]
            ppmgt_dict[regi][0][ii,jj] = pmean + buff
        for p in gt_dict[regi][1]: # 50 shift patches
            ii,jj = p[0], p[1]
            pmean = p[2]
            ppmgt_dict[regi][1][ii,jj] = pmean + buff

    return ppmgt_dict


def generate_ppm_ground_truths_from_labels(seg_lab_dict, gt_dict, imgdim_dict, buff=0):
    ppmgt_dict = {}
    print(gt_dict)
    print(imgdim_dict)

    for regi in gt_dict.keys():
        ppmgt_dict[regi] = [np.ones((imgdim_dict[regi][0]+1, imgdim_dict[regi][1]+1)) *-1, np.ones((imgdim_dict[regi][0]+1, imgdim_dict[regi][1]+1)) *-1]
        
        print("Generating Ground Truth PPM for image:", regi)
        print("dims:", imgdim_dict[regi][0], imgdim_dict[regi][1])
        print("num patches:",len(gt_dict[regi][0]), len(gt_dict[regi][1]))
        print("maxes:", gt_dict[regi][2],gt_dict[regi][3])

        for p in gt_dict[regi][0]: # no shift patches
            ii,jj = p[0], p[1]
            pmean = p[2]
            lab = seg_lab_dict[(regi, "noshift", ii, jj)] # plus 1 for ii,jj?
            ppmgt_dict[regi][0][ii,jj] = lab + buff
        for p in gt_dict[regi][1]: # 50 shift patches
            ii,jj = p[0], p[1]
            pmean = p[2]
            lab = seg_lab_dict[(regi, "50shift", ii, jj)] # plus 1 for ii,jj?
            ppmgt_dict[regi][1][ii,jj] = lab + buff

    return ppmgt_dict


def inflate_2by2(arr):
    arr_p = np.repeat(arr, 2, axis=0)
    arr_pp = np.repeat(arr_p, 2, axis=1)
    return arr_pp


def create_overlay_ppmgts(ppmgt_dict):
    ppmgts = {}
    for key in ppmgt_dict.keys():
        reg = ppmgt_dict[key][0]
        s50 = ppmgt_dict[key][1]

        # inflate
        reg = inflate_2by2(reg)
        s50 = inflate_2by2(s50)

        regh = reg.shape[0]
        regw = reg.shape[1]
        s50h = s50.shape[0]
        s50w = s50.shape[1]
        
        maxh = np.max([regh, s50h])
        maxw = np.max([regw, s50w])
        
        h = maxh + 1 #2 * ((maxh//2)+1)
        w = maxw + 1 #2 * ((maxw//2)+1)
        
        out = np.zeros((h,w))
        out[0:regh,0:regw] += reg
        out[1:s50h+1,1:s50w+1] += s50
        ppmgts[key] = out / 2 # avg
    
    return ppmgts


# main function
#--------------
def main():
    # prepatch_flag=False: storage-saving; recommended for multiplexed images
    # prepatch_flag=True:  runtime-saving; recommended for 1-channel controls

    # ARGPARSE
    #---------
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default=None, type=str, help='Dataset name: u54codex, controls')
    parser.add_argument('--data_dir', default=None, type=str, help='Dataset directory.')
    parser.add_argument('--study_arm', default="train", type=str, help='Study arm: train/val/test. Deafult is train.')
    parser.add_argument('--cache_name', default=None, type=str, help="Name associated with the cached objects. For example, for controls partitions/scenarios, please name the partition.")
    parser.add_argument('--HW', default=96, type=int, help="Patch size. Default is 96.")
    parser.add_argument('--resize_flag', default=False, type=str2bool, help="T/F if patches are to be resized to another dimension.")
    parser.add_argument('--resize_HW', default=96, type=int, help="Dimension to resize to. Requires resize_flag=True.")

    parser.add_argument('--augment_level', default="none", type=str, help="Level of augmentation: none/low/high. Default is none.")
    parser.add_argument('--filtration_type', default="background", type=str, help="Filtration type: {background, none}. Defaults to background filtration.")
    parser.add_argument('--prepatch_flag', default=False, type=str2bool, help="T/F if patches are stored/saved at the directory, save_dir.")
    parser.add_argument('--overwrite_flag', default=False, type=str2bool, help="Warning! This can take up to 3 hours if overwrite=True! Do you want to overwrite patch dataset? Refers to save_dir.")
    parser.add_argument('--save_dir', default=None, type=str, help="Where to save patches if prepatch_flag=True. Otherwise, not used.")
    parser.add_argument('--cache_dir', default=None, type=str, help="Where to save cached/serialized information, like the patch list.")
    parser.add_argument('--verbosity_flag', default=False, type=str2bool, help="Do you want CLI printouts of patches? Default is True.")
    parser.add_argument('--control_groundtruth_flag', default=False, type=str2bool, help="Do you want to create ground truths? Only works if dataset name = controls.")
    parser.add_argument('--overwrite_gt_flag', default=False, type=str2bool, help="Do you want to create NEW ground truths?")

    parser.add_argument('--bg_remove_flag', default=False, type=str2bool, help="Is the background removed and replaced with 0-values? Defaults to False.")
    parser.add_argument('--label_dict', default=None, type=str, help="Optional: if you have a label dictionary, give the path here.")
    parser.add_argument('--hdf5_flag', default=False, type=str2bool, help="Do you want to store the patches as HDF5?")

    args = parser.parse_args()

    if args.dataset_name == None:
        print("Please enter a dataset name! Check out valid names / add one to your pipeline.")
        exit()
    if args.data_dir == None:
        print("please enter a dataset directory!")
        exit()
    if args.cache_name == None or args.cache_dir == None:
        print("please enter a cache name and directory!")
        exit()
    if args.save_dir == None and prepatch_flag == True:
        print("Detecting prepatching option... Please enter a save directory!")
        exit()

    cache_full_name = args.dataset_name + "-" + args.cache_name + "-" + str(args.HW) + "-" + args.filtration_type
    setattr(args, "cache_full_name", cache_full_name)

    # MAIN CALL
    #==========
    patchify(args)
    # pdb.set_trace()

    mil_runs = ["cam", "guilty_superpixels"]

    if args.control_groundtruth_flag == False and args.study_arm == "test" and args.dataset_name == "cam":
        print("Ground truths are provided already. Exiting..")


    if args.control_groundtruth_flag == True and args.study_arm == "test" and args.dataset_name == "controls":
        out_dir = args.cache_dir + "/outputs/"
        print("generating ground truths!")

        # saving time here if dictionary already exists
        if not os.path.exists(out_dir + cache_full_name + "-gt_dict.obj") or args.overwrite_gt_flag == True:
            print("overwriting or creating new ground truth dictionary...")
            gt_dict = generate_ground_truths(args.save_dir)
            utils.serialize(gt_dict, out_dir + cache_full_name + "-gt_dict.obj")
        else: 
            print("using old ground truth dictionary...")
            gt_dict = utils.deserialize(out_dir + cache_full_name + "-gt_dict.obj")

        # now we do the quick things
        imgdim_dict = get_imgdims(args.data_dir, args.HW)
        ppmgt_dict = generate_ppm_ground_truths(gt_dict, imgdim_dict)
        ppmgts = create_overlay_ppmgts(ppmgt_dict)
        utils.serialize(ppmgts, out_dir + cache_full_name + "-ppmgts.obj")
        utils.serialize(imgdim_dict, out_dir + cache_full_name + "-imgdim_dict.obj")

        # seg labels for test-gsp:
        if args.cache_name in mil_runs:
            print("creating seg-like label dictionary...")
            seg_lab_dict = {}
            for reg in gt_dict.keys():
                for i,shift in enumerate(["noshift","50shift"]):
                    for patch in gt_dict[reg][i]:
                        patch_mean = np.array([patch[2]])
                        thresholded = np.where((patch_mean < 0.5) & (patch_mean >= 0), 0, 1)
                        seg_lab_dict[(reg, shift, patch[0], patch[1])] = thresholded[0]
            
            utils.serialize(seg_lab_dict, out_dir + cache_full_name + "-seg_labels_TEST.obj")
            # ppmgt_dict_seg = generate_ppm_ground_truths_from_labels(seg_lab_dict, gt_dict, imgdim_dict)
            # ppmgts_seg = create_overlay_ppmgts(ppmgt_dict_seg)
            # utils.serialize(ppmgts_seg, out_dir + cache_full_name + "-ppmgts-SEG_TEST.obj")

            # gt_save_path2 = args.cache_dir + "/bitmaps/ground_truths/seg_labels/" + args.cache_full_name + "/"
            # if not os.path.exists(gt_save_path2):
            #     os.makedirs(gt_save_path2)
            #     print("creating a directory at", gt_save_path2)
            # for key in ppmgts_seg.keys():
            #     print("saving ground truth for", key)
            #     np.save(gt_save_path2 + key, ppmgts_seg[key]) 

        # save gt PPMs
        gt_save_path = args.cache_dir + "/bitmaps/ground_truths/" + args.cache_full_name + "/"
        if not os.path.exists(gt_save_path):
            os.makedirs(gt_save_path)
            print("creating a directory at", gt_save_path)
        for key in ppmgts.keys():
            print("saving ground truth for", key)
            np.save(gt_save_path + key, ppmgts[key]) 
    else:
        print("Finishing up!")


    # train on gsp
    if args.control_groundtruth_flag == True and args.study_arm == "train" and args.dataset_name == "controls" and args.cache_name in mil_runs:
        out_dir = args.cache_dir + "/outputs/"
        print("generating ground truths for MIL-like training run to potentially be used as labels!")

        # saving time here if dictionary already exists
        if not os.path.exists(out_dir + cache_full_name + "-gt_dict_TRAIN.obj") or args.overwrite_gt_flag == True:
            print("overwriting or creating new ground truth dictionary...")
            gt_dict = generate_ground_truths(args.save_dir)
            utils.serialize(gt_dict, out_dir + cache_full_name + "-gt_dict_TRAIN.obj")
        else: 
            print("using old ground truth dictionary...")
            gt_dict = utils.deserialize(out_dir + cache_full_name + "-gt_dict_TRAIN.obj")

        # if not os.path.exists(out_dir + cache_full_name + "-seg_labels_TRAIN.obj") or args.overwrite_gt_flag == True:
        print("creating seg-like label dictionary...")
        seg_lab_dict = {}
        for reg in gt_dict.keys():
            for i,shift in enumerate(["noshift","50shift"]):
                for patch in gt_dict[reg][i]:
                    patch_mean = np.array([patch[2]])
                    thresholded = np.where((patch_mean < 0.5) & (patch_mean >= 0), 0, 1)
                    seg_lab_dict[(reg, shift, patch[0], patch[1])] = thresholded[0]
        
        utils.serialize(seg_lab_dict, out_dir + cache_full_name + "-seg_labels_TRAIN.obj")
        # ppmgt_dict_seg = generate_ppm_ground_truths_from_labels(seg_lab_dict, gt_dict, imgdim_dict)
        # ppmgts_seg = create_overlay_ppmgts(ppmgt_dict_seg)
        # utils.serialize(ppmgts_seg, out_dir + cache_full_name + "-ppmgts-SEG_TRAIN.obj")
        # else:
        #     print("Seeing old seg-like label dictionary! Skipping!")
    else:
        print("Finishing up!")

if __name__ == "__main__":
    main()

    