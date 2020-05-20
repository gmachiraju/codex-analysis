import glob
import os
import sys
import tifffile as tiff
import numpy as np
import pdb

# constants
HW = 96

reg_dict = {"005": ("test", "negative"),
            "006": ("test", "positive"),
            "017": ("test", "positive"),
            "019": ("test", "positive"),
            "011": ("val", "negative"),
            "016": ("val", "positive"),
            "030": ("val", "positive"),
            "023": ("val", "positive"),
            "004": ("train", "negative"),
            "015": ("train", "negative"),
            "014": ("train", "positive"),
            "024": ("train", "positive"),
            "020": ("train", "positive"),
            "007": ("train", "positive"),
            "008": ("train", "positive"),
            "027": ("train", "positive"),
            "034": ("train", "positive"),
            "012": ("train", "positive")} 


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


def main(data_dir):

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

        save_dir = "/scratch/users/gmachi/codex/data"
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
            
            # Toss Black tiles
            if toss_blank_tile(tile, q10):
                patch_tossed += 1
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
            if toss_blank_tile(tile):
                continue
            tile_str = "%s_patch-%s_shift50" % (im_id, k)
            tif_path = os.path.join(save_path, tile_str + ".tif")
            np.save(save_path + '/' + tile_str, tile)

        print('Images completed: {}, Last image patches: {}, tossed: {}'.format(image_count, patch_count, patch_tossed))

if __name__ == "__main__":
    data_dir = sys.argv[1]
    main(data_dir)