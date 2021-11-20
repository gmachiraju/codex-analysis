import glob
import os
import sys
import tifffile as tiff
import numpy as np
import pandas as pd
import pdb
# import cv2
# from skimage.filters import gabor_kernel
# from skimage.feature import ORB, match_descriptors
from scipy import ndimage as ndi
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import gc # grabage collect for memory issues

import utils
from utils import labels_dict, reg_dict, ctrl_dict, str2bool
import argparse

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


def normalize(im, dataset_name):
    # toss blank channels if they exist
    if dataset_name == "u54codex":
        null_channels = [82, 81, 79, 78, 77, 74, 73, 69, 65]
        im = np.delete(im, null_channels, axis=0)
    
    channel_means1 = np.mean(im, axis=(1,2)).reshape(-1,1,1)
    # channel_means2 = np.mean(im, axis=(1,2)).reshape(im.shape[0], 1, 1)
    # print(channel_means1 == channel_means2)
    
    new_im = im - channel_means1
    new_im /= (np.std(new_im, axis=(1,2)).reshape(-1,1,1) + 1e-5)

    return new_im


def toss_blank_tile(im, q, dataset_name="u54codex", tol=0.05):
    """
    Toss out tiles with >=80% low intensity pixels
    OR toss out middle values..... you set the filtering function!
    """
    if dataset_name == "u54codex":
        summed = np.sum(im, 0)
        if np.mean(summed < q) > 0.8:
            return True
        return False

    elif dataset_name == "controls":
        mean_val = np.mean(im)
        if (mean_val < q+tol and mean_val > q-tol) or (np.isnan(mean_val)):
            # print("mean=", np.mean(im), "--> tossing b/c is NaN or mean was too close to", q)
            return True
        # print("mean=", np.mean(im))
        return False


def crop_coords(im, height, width, shift="noshift"):
    """
    solely returns coordinates for an image
    notes: https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
    """
    # translate image to the right and down 50%
    if shift == "50shift":
        im = im[:, height // 2:, width // 2:]

    channel, imgheight, imgwidth = im.shape

    # top-bottom, left-right
    for i in range(imgheight // height):
        for j in range(imgwidth // width):
            yield [i * height, (i + 1) * height, j * width, (j + 1) * width] #x1,x2,y1,y2


def process_patch(im, im_id, q_low, patches, shift, args, patch_tossed_i, patch_tossed, patch_count_i):
    
    channel, imgheight, imgwidth = im.shape

    # make grid
    num_cols = int(imgwidth // args.HW) # should be floor right? used to be ceil
    num_rows = int(imgheight // args.HW)
    grid = np.zeros((num_rows, num_cols))

    for k, coords_list in enumerate(crop_coords(im, args.HW, args.HW, shift=shift)):

        # get patch coordinate
        [x1, x2, y1, y2] = coords_list
       
        curr_row = int(x2 / args.HW) - 1 #int((num_rows // (k+1)))
        curr_col = int(y2 / args.HW) - 1 #int(((k+1) % num_cols) - 1)

        # tile/patch slice!
        tile = im[:, x1:x2, y1:y2]

        # checks for valid tile/patch
        if tile.shape[1] != args.HW or tile.shape[2] != args.HW: # not square patch
            patch_tossed_i += 1
            patch_tossed += 1
            continue
        if args.filtration_type == "background": # if not, we keep all patches including noisy background
            if toss_blank_tile(tile, q_low, args.dataset_name): # blank patch
                patch_tossed_i += 1
                patch_tossed += 1
                del tile
                continue
        
        # non-aug patch
        tile_str = "%s_patch%s_coords%s-%s-[%s:%s,%s:%s]_%s_noaug" % (im_id, k, curr_row, curr_col, x1, x2, y1, y2, shift)
        patches.append(tile_str)
        if args.prepatch_flag == True:
            np.save(args.save_dir + '/' + tile_str, tile)

        patch_count_i += 1 
        # update grid
        grid[curr_row, curr_col] = 1

        if args.study_arm == "train": 
            
            if args.augment_level == "high":
                for rot in [1, 2, 3]:
                    tile_str = "%s_patch%s_coords%s-%s-[%s:%s,%s:%s]_%s_rot%s" % (im_id, k, curr_row, curr_col, x1, x2, y1, y2, shift, rot)
                    patches.append(tile_str)
                    if args.prepatch_flag == True:
                        tile = axis_rotate(tile, rot)
                        np.save(args.save_dir + '/' + tile_str, tile)
                
                for refl in [0, 1]:
                    tile_str = "%s_patch%s_coords%s-%s-[%s:%s,%s:%s]_%s_refl%s" % (im_id, k, curr_row, curr_col, x1, x2, y1, y2, shift, refl)
                    patches.append(tile_str)
                    if args.prepatch_flag == True:
                        tile = axis_reflect(tile, refl)
                        np.save(args.save_dir + '/' + tile_str, tile)

            elif args.augment_level == "low":
                # Random rotation
                tile, rot = random_rotate(tile)
                tile_str = "%s_patch%s_coords%s-%s-[%s:%s,%s:%s]_%s_rot%s" % (im_id, k, curr_row, curr_col, x1, x2, y1, y2, shift, rot)
                patches.append(tile_str)
                if args.prepatch_flag == True:
                    np.save(args.save_dir + '/' + tile_str, tile)
                
                # Random reflection
                tile, refl = random_reflect(tile)
                tile_str = "%s_patch%s_coords%s-%s-[%s:%s,%s:%s]_%s_refl%s" % (im_id, k, curr_row, curr_col, x1, x2, y1, y2, shift, refl)
                patches.append(tile_str)
                if args.prepatch_flag == True:
                    np.save(args.save_dir + '/' + tile_str, tile)
        del tile

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


def patchify(args):
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

    if args.dataset_name == "controls":
        splitlab_dict = {}

    #################
    # FILE ITERATION
    #################
    for filename in files:

        # run only on valid files in dir
        if args.dataset_name == "u54codex":
            if not filename.endswith(".tif"):
                continue

        elif args.dataset_name == "controls":
            if not filename.endswith(".npy"):
                continue

        # Read File
        if args.dataset_name == "u54codex":
            im = tiff.imread(os.path.join(args.data_dir, filename)) 
            splitlab_dict = utils.reg_dict
            # reshape
            cycle, channel, imgwidth, imgheight = im.shape
            im = im.reshape(cycle * channel, imgwidth, imgheight)

        elif args.dataset_name == "controls":
            im = np.load(os.path.join(args.data_dir, filename))
            imgwidth, imgheight, channel = im.shape
            im = im.reshape(channel, imgwidth, imgheight)

        # parse names
        if args.dataset_name == "u54codex":
            im_folder = filename.split(".")[0]  # "reg{idx}_montage"
            im_id = im_folder.split("_")[0]  # "reg{idx}""
            idx = im_id.split("reg")[1]  # {idx}
                    
            if args.study_arm != splitlab_dict[idx][0]:
                continue

        elif args.dataset_name == "controls":
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
            # print("current label dict:", splitlab_dict)
      
        print("="*60)
        print('\nCurrently on file {}'.format(filename))
        print('Original image dimensions (in pixels): {}'.format(im.shape) + "\n" + "="*60)
        image_count += 1
        # channel, imgheight, imgwidth = im.shape

        # Normalize Image + toss blank channels
        if args.dataset_name == "u54codex":
            im = normalize(im, dataset_name)

        if args.filtration_type == "background":
            # Identify low-intensity threshold
            if args.dataset_name == "u54codex":
                im_sum = np.sum(im, axis=0) # Does this make sense?
                q_low = np.quantile(im_sum, q=0.1)

            if args.dataset_name == "controls":
                q_low = 0.5
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
            patch_tossed_i, patch_tossed, patch_count_i, patches = process_patch(im, im_id, q_low, patches, shift, args, patch_tossed_i, patch_tossed, patch_count_i)

        print('Images completed: {}, Previous image patches (non-augmented): {}, total patches: {}, tossed: {}, total tossed: {}\n'.format(image_count, patch_count_i, len(patches), patch_tossed_i, patch_tossed))
        # pdb.set_trace()
        gc.collect()
    
    # always serialize if want to keep order
    print("Serializing patch list!")
    utils.serialize(patches, args.cache_dir + "/outputs/" + args.study_arm + "-" + args.cache_full_name + "-patchlist.obj")    
    
    if args.dataset_name == "controls":
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
        
        regi = im_str.split("-")[0]
        im = np.load(img_path + "/" + im_str)
        maxH, maxW, _ = im.shape
        
        num_cols = int(np.floor(maxW / HW))
        num_rows = int(np.floor(maxH / HW))
        
        print("expecting patch grid of", num_rows, "x", num_cols, "for a", maxH, "x", maxW, "image")
        imgdim_dict[regi] = [num_rows, num_cols]
        del im

    return imgdim_dict
        

def generate_ppm_ground_truths(gt_dict, imgdim_dict, buff=0):
    ppmgt_dict = {}
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
    parser.add_argument('--augment_level', default="none", type=str, help="Level of augmentation: none/low/high. Default is none.")
    parser.add_argument('--filtration_type', default="background", type=str, help="Filtration type. Defaults to background filtration.")
    parser.add_argument('--prepatch_flag', default=False, type=str2bool, help="T/F if patches are stored/saved at the directory, save_dir.")
    parser.add_argument('--overwrite_flag', default=False, type=str2bool, help="Warning! This can take up to 3 hours if overwrite=True! Do you want to overwrite patch dataset? Refers to save_dir.")
    parser.add_argument('--save_dir', default=None, type=str, help="Where to save patches if prepatch_flag=True. Otherwise, not used.")
    parser.add_argument('--cache_dir', default=None, type=str, help="Where to save cached/serialized information, like the patch list.")
    parser.add_argument('--verbosity_flag', default=False, type=str2bool, help="Do you want CLI printouts of patches? Default is True.")
    parser.add_argument('--control_groundtruth_flag', default=False, type=str2bool, help="Do you want to create ground truths? Only works if dataset name = controls.")
    parser.add_argument('--overwrite_gt_flag', default=False, type=str2bool, help="Do you want to create NEW ground truths?")

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

    patchify(args)
    # pdb.set_trace()

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

if __name__ == "__main__":
    main()

    