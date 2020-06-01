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

from utils import labels_dict

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
    plt.title("Channel-wise medians of training samples " + norm_str)
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
    plt.title("t-SNE projection by label " + norm_str)
    plt.show()
    
    plt.figure(figsize=(5,5))
    sns.scatterplot(
        x="tsne-1", y="tsne-2",
        hue="IDs",
        data=tsne_df,
        palette=sns.color_palette("Set2", len(set(tsne_df["IDs"]))),
        alpha=0.7
    )
    plt.title("t-SNE projection by image ID " + norm_str)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.show()
    
    return summary_im, train_im_df, train_im_dfl





def crop(im, height, width, shift="edge"):
    """
    cropping into HWxHWx3 => ~400 images/sample * 40 samples = ~16K images
    notes: https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
    """
    # translate image to the right and down 50%
    if shift == "50shift":
        im = im[:, width // 2:, height // 2:]

    channel, imgwidth, imgheight = im.shape

    for i in range(imgheight // height):
        for j in range(imgwidth // width):
            box = [j * width, i * height,
                   (j + 1) * width, (i + 1) * height]
            # print(box)
            yield im[:, i * height: (i + 1) *
                     height,  j * width: (j + 1) * width]


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


def normalize(im):
    # toss blank channels
    null_channels = [82, 81, 79, 78, 77, 74, 73, 69, 65]
    im = np.delete(im, null_channels, axis=0)

    channel_means = np.mean(im, axis=(1, 2))
    return im - channel_means.reshape(im.shape[0], 1, 1)


def toss_blank_tile(im, q):
    """
    Toss out tiles with >=80% low intensity pixels
    """
    summed = np.sum(im, 0)
    if np.mean(summed < q) > 0.8:
        return True
    return False


def patchify(data_dir, save_dir):
    # "main function"
    
    files = os.listdir(data_dir)
    image_count = 0
    for filename in files:  # assuming gif

        # Select Tif File
        if not filename.endswith(".tif"):
            continue
        
        image_count += 1

        # Read File
        im = tiff.imread(os.path.join(data_dir, filename))
        print('on file {}'.format(filename))
        print('original size: {}'.format(im.shape))
        print("--------------")

        # Isolate case name
        im_folder = filename.split(".")[0]  # "reg{idx}_montage"
        im_id = im_folder.split("_")[0]  # "reg{idx}""
        idx = im_id.split("reg")[1]  # {idx}

        # Skip non-case files
        if idx not in reg_dict:
            continue

#         save_dir = "/scratch/users/gmachi/codex/data"
        study_arm, label = reg_dict[idx]
        save_path = os.path.join(save_dir, study_arm)

        # Make save folder
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Reshape image
        cycle, channel, imgwidth, imgheight = im.shape
        im = im.reshape(cycle * channel, imgwidth, imgheight)

        # Normalize Image
        im = normalize(im)

        # Identify low-intensity threshold
        im_sum = np.sum(im, axis=0)
        q10 = np.quantile(im_sum, q=0.1)

        patch_count = 0
        patch_tossed = 0

        # Patch images
        print('edge cropping...')
        for k, tile in enumerate(crop(im, HW, HW)):
            
            patch_count +=1
            
            # Toss Black/blank tiles
            if toss_blank_tile(tile, q10):
                patch_tossed += 1 
                # future: maybe still do k+=1 to know which patches were removed? 
                # or just save those in another dir called "/rejects"? 
                # need some way of saving info of where on grid these images lie. H/HW x W/HW = # patches.
                continue

            # Save non-augmented patch
            tile_str = "%s_patch%s_normal" % (im_id, k)
            tif_path = os.path.join(save_path, tile_str + ".tif")
            np.save(save_path + '/' + tile_str, tile)

            # Patch augmentations for Train set images
            if study_arm == "train": 

                if label == "positive":
                    # Rotate 90, 180, or 270 degrees
                    for rot in [1, 2, 3]:
                        tile = axis_rotate(tile, rot)
                        tile_str = "%s_patch%s_rot%s" % (im_id, k, rot)
                        tif_path = os.path.join(save_path, tile_str + ".tif")
                        np.save(save_path + '/' + tile_str, tile)
                    # Reflect vertically and horizontally
                    for refl in [0, 1]:
                        tile = axis_reflect(tile, refl)
                        tile_str = "%s_patch%s_refl%s" % (im_id, k, refl)
                        tif_path = os.path.join(save_path, tile_str + ".tif")
                        np.save(save_path + '/' + tile_str, tile)

                if label == "negative":
                    # Random rotation
                    tile, choice = random_rotate(tile)
                    tile_str = "%s_patch%s_rot%s_rand" % (im_id, k, choice)
                    tif_path = os.path.join(save_path, tile_str + ".tif")
                    np.save(save_path + '/' + tile_str, tile)
                    # Random reflection
                    tile, choice = random_reflect(tile)
                    tile_str = "%s_patch%s_refl%s_rand" % (im_id, k, choice)
                    tif_path = os.path.join(save_path, tile_str + ".tif")
                    np.save(save_path + '/' + tile_str, tile)

        # 50% Shift Crops
        print('shifted cropping...')
        for k, tile in enumerate(crop(im, HW, HW, shift="50shift")):
            if toss_blank_tile(tile, q10): # earlier: forgot q10 as 2nd arg
#                 patch_tossed += 1
                continue
    
            tile_str = "%s_patch-%s_shift50" % (im_id, k) # future: remove "-" between "patch" and "%s"
            tif_path = os.path.join(save_path, tile_str + ".tif")
            np.save(save_path + '/' + tile_str, tile)

        print('Images completed: {}, Last image patches: {}, tossed: {}'.format(image_count, patch_count, patch_tossed))

# if __name__ == "__main__":
#     data_dir = sys.argv[1]
#     main(data_dir)