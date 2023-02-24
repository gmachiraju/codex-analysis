import argparse
import ftplib
import os
import pdb
# import openslide
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import draw
import skimage
from skimage import measure

from utils import serialize, deserialize, str2bool
LV=3 # was 5 earlier


# Converting TIFs to NPYs
#--------------------------
def get_imgdims(sample_id, data_path, level=LV):
    img_file = data_path + "/" + sample_id + ".tif"

    img = openslide.OpenSlide(img_file)
    dim = img.level_dimensions[0]

    desired_downsample = img.level_downsamples[level]
    desired_dims = img.level_dimensions[level]
 
    # width = int(dim[1] / desired_downsample) #* scale_percent / 100)
    # height = int(dim[0] / desired_downsample) #* scale_percent / 100)
    # new_dim = (width, height)
    del img
    print("desired downsample is:", desired_downsample)

    return dim, desired_dims, desired_downsample


# def get_foreground_maps(sample_id, data_path, level=LV):
#     pass


def process_image(sample_id, data_path, save_path, level=LV):
    img_file = data_path + "/" + sample_id
    print(img_file)
    id_num = sample_id.split(".tif")[0]
    print(id_num)

    img = openslide.OpenSlide(img_file)
    desired_downsample = img.level_downsamples[level]
    desired_dims = img.level_dimensions[level]
    print(desired_downsample, desired_dims)

    rgba_img = img.read_region((0,0), level, desired_dims)
    del img
    rgba_arr = np.asarray(rgba_img)
    rgb_arr = cv2.cvtColor(rgba_arr, cv2.COLOR_RGBA2RGB)
    out = np.asarray(rgb_arr)
    np.save(save_path + "/" + id_num + ".npy", out)
    del rgba_arr
    del rgb_arr

    print("Finished saving image:", sample_id)
    print("of shape:", out.shape)
    print("")
    
    
def parse_xml(xml_file, reduction=None, verbosity="low"):
    save_list, ys, xs = [], [], []
    annot_count = 0
    open_flag = False
    with open(xml_file, 'r') as infile:
        for i, line in enumerate(infile):
            if verbosity == "high":
                print(line)
            if ('<Annotation Name=' in line) and ((("_1" in line) or ("_0" in line) or ("Tumor" in line))):
                annot_count += 1
                annot = []
                x, y = [], []
                open_flag = True
            elif ("<Coordinate Order=" in line) and (open_flag == True):
                xy = line.split(" ")[2:4]
                xy = [float(xy[0].split("=")[1].strip('\"')), float(xy[1].split("=")[1].strip('\"'))]
                if reduction:
                    xy = [xy[0] // reduction, xy[1] // reduction]
                x.append(xy[0]) 
                y.append(xy[1])
                annot.append(xy)
            elif ("</Annotation>" in line) and (open_flag == True): 
                if annot_count > 0:
                    save_list.append(annot)
                    xs.append(x)
                    ys.append(y)
                open_flag = False # close annotation
    return save_list, ys, xs


def view_mask_elements(sample_id, xml_path, data_path, save_path):
    xml_file = xml_path + "/" + sample_id + ".xml"
    img_file = data_path + "/" + sample_id + ".tif"
    dim, new_dim, desired_downsample = get_imgdims(sample_id, data_path)
    save_list, ys, xs = parse_xml(xml_file)
    
    for i in range(len(save_list)):
        image = draw.polygon2mask(dim, np.stack((xs[i], ys[i]), axis=1))
        y_min, y_max = int(np.round(np.min(ys[i])) - 100), int(np.round(np.max(ys[i])) + 100)
        x_min, x_max = int(np.round(np.min(xs[i])) - 100), int(np.round(np.max(xs[i])) + 100)
        zoom = image[x_min:x_max, y_min:y_max]
        plt.figure()
        plt.imshow(zoom, cmap="gray")


def process_downsampled_mask(sample_id, xml_path, data_path, save_path):
    """
    For high-memory settings
    """
    img_file = data_path + "/" + sample_id + ".tif"
    xml_file = xml_path + "/" + sample_id + ".xml"
    dim, new_dim, desired_downsample = get_imgdims(sample_id, data_path)
    save_list, ys, xs = parse_xml(xml_file)
    
    mask = np.zeros(new_dim)
    print("original mask shape is:", dim)
    print("mask shape is:", new_dim)

    print("we've detected", len(save_list), "ROIs in this sample")
    for i in range(len(save_list)):
        image = draw.polygon2mask(dim, np.stack((xs[i], ys[i]), axis=1))
        image = cv2.resize(np.float32(image), new_dim, interpolation=cv2.INTER_AREA)
        mask += image
        del image
        print("finished ROI:", i)

    plt.figure()
    plt.imshow(mask, cmap="gray")    
    np.save(save_path + "/" + sample_id + ".npy", mask)
    del mask
    print("Finished saving mask:", sample_id)
    print("of shape:", mask.shape)
    print("")


def process_patch_mask(sample_id, xml_path, data_path, save_path, patch_size=224):
    """
    For low-memory settings with patches as the units of interest
    """
    print("starting on sample:", sample_id)
    img_file = data_path + "/" + sample_id + ".tif"
    xml_file = xml_path + "/" + sample_id + ".xml"
    dim, new_dim, desired_downsample = get_imgdims(sample_id, data_path)
    save_list, ys, xs = parse_xml(xml_file, reduction=desired_downsample)

    patch_array_dim = [new_dim[0] // patch_size, new_dim[1] // patch_size]
    mask = np.zeros(patch_array_dim)
    print("original mask shape is:", dim)
    print("mask shape is:", new_dim, "--> downsampled reduction of:", desired_downsample)
    print("patch array shape is:", patch_array_dim, "--> block-/patched reduction of:", patch_size)
    print("we've detected", len(save_list), "ROIs in this sample")

    for i in range(len(save_list)):
        image = draw.polygon2mask(new_dim, np.stack((xs[i], ys[i]), axis=1))
        image = skimage.measure.block_reduce(image, block_size=(patch_size, patch_size), func=np.mean)
        try:
            mask += image
        except ValueError: # still a missmatch
            print("Detecting a mismatch in annotation and overall mask dimensions... adjusting")
            print("image")
            # assuming smaller iamge than mask by 1 row/col
            if image.shape[0] > mask.shape[0]: # height
                image = image[:-1, :]
            if image.shape[1] > mask.shape[1]: # width
                image = image[:, :-1]
            mask += image
            
        del image
        print("finished ROI:", i)

    mask = mask > 0
    plt.figure()
    plt.imshow(mask, cmap="gray")    
    np.save(save_path + "/" + sample_id + "_MASK.npy", mask)
    print("Finished saving mask:", sample_id)
    print("of shape:", mask.shape)
    print("")


def computeEvaluationMaskXML_lowres(xmlDIR, og_dims, resolution, level):
    scale_factor = 2**level
    # w,h for cv2
    print("og dims:", og_dims)
    desired_dims = (og_dims[0] // scale_factor, og_dims[1] // scale_factor)
    print("desired dims", desired_dims)
    save_list, xs, ys = parse_xml(xmlDIR, reduction=scale_factor)
    if len(save_list) == 0:
        save_list, xs, ys = parse_xml(xmlDIR, reduction=None, verbosity="high")
        print(save_list)
        print("breaking...")
        return None
    # print(save_list)
    mask = np.zeros(desired_dims)
    print("processing", len(save_list), "ROIs...")
    for i in range(len(save_list)):
        image = draw.polygon2mask(desired_dims, np.stack((xs[i], ys[i]), axis=1))
        mask += image.astype(float)
        del image
    print("number of mask pixels:", np.sum(mask))
    return mask


def computeEvaluationMaskXML(xmlDIR, og_dims, resolution, level):
    scale_factor = 2**level
    desired_dims = (og_dims[0] // scale_factor, og_dims[1] // scale_factor)
    save_list, xs, ys = parse_xml(xmlDIR, reduction=None)
    print("done parsing XML...")
    mask = np.zeros(desired_dims)
    for i in range(len(save_list)):
        print("starting on ROI:", i)
        xmin, xmax = np.min(xs[i]), np.max(xs[i])
        ymin, ymax = np.min(ys[i]), np.max(ys[i])
        roi_crop = np.zeros((int(xmax-xmin), int(ymax-ymin)))
        image = draw.polygon2mask(roi_crop.shape, np.stack((xs[i] - xmin, ys[i] - ymin), axis=1))
        # width, height for cv2
        desired_crop_dims = roi_crop.shape[1] // scale_factor, roi_crop.shape[0] // scale_factor
        image = cv2.resize(np.float32(image), desired_crop_dims, interpolation=cv2.INTER_AREA)
        i0,i1 = int(xmin//scale_factor), int(xmax//scale_factor)
        j0,j1 = int(ymin//scale_factor), int(ymax//scale_factor)
        h,w = image.shape
        
        mask[i0:i0+h, j0:j0+w] = image
        del image
        print("finished ROI:", i)
    return mask


def computeEvaluationMaskXML_mosaic(xmlDIR, og_dims, resolution, level, ps=224):
    scale_factor = 2**level
    desired_dims = (og_dims[0] // (scale_factor * 224), og_dims[1] // (scale_factor * 224))
    save_list, xs, ys = parse_xml(xmlDIR, reduction=scale_factor)
    if len(save_list) == 0:
        save_list, xs, ys = parse_xml(xmlDIR, reduction=None, verbosity="high")
        print(save_list)
        print("breaking...")
        return None
    print("done parsing XML...have #ROIs:", len(save_list))
    mask = np.zeros(desired_dims)
    for i in range(len(save_list)):
        image = draw.polygon2mask(desired_dims, np.stack((xs[i], ys[i]), axis=1))
        image = skimage.measure.block_reduce(image, block_size=(ps, ps), func=np.mean)
        try:
            mask += image
        except ValueError: # still a missmatch
            print("Detecting a mismatch in annotation and overall mask dimensions... adjusting")
            # assuming smaller iamge than mask by 1 row/col
            if image.shape[0] > mask.shape[0]: # height
                image = image[:-1, :]
            if image.shape[1] > mask.shape[1]: # width
                image = image[:, :-1]
            mask += image
        del image

    mask = mask > 0
    return mask


if __name__ == "__main__":
    
    # DOWNLOAD TIFs
    #----------------
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument("--arm", type=str, help="train/val/test")
    arg_parser.add_argument("--tifdir", type=str, help="Where to find TIF files to convert to numpy images.")
    arg_parser.add_argument("--xmldir", type=str, help="Where to find XML files to convert to binary masks.")
    arg_parser.add_argument("--savedir", type=str, help="Absolute path for saving images.")
    arg_parser.add_argument("--savedirmasks", type=str, help="Absolute path for saving masks.")
    arg_parser.add_argument("--debugflag", type=str2bool, default=True, help="Is this run testing functionality? Default: True.")
    args = arg_parser.parse_args()
    print(args)
    
    if args.debugflag == True:
        print("testing testing 1 2 3...")
        # just try firsts of the train cohort
        normal_id = "normal_001"
        normal_img = args.tifdir + "/" + normal_id + ".tif"
        process_image(normal_id, args.tifdir, args.savedir)

        tumor_id = "tumor_001"
        tumor_img = args.tifdir + "/" + tumor_id + ".tif"
        process_image(tumor_id, args.tifdir, args.savedir)

        tumor_xml = args.xmldir + "/" + tumor_id + ".xml"
        process_mask(tumor_id, args.xmldir, args.tifdir, args.savedirmasks)

        print("Done! Go check out the outputs why don't ya")
        exit()

    print("Not a debugging run! Processing...")

    foreground_dict = {}
    for i, tifstr in enumerate(os.listdir(args.tifdir)):
        print("Starting on:", tifstr)
        if ".tif" in tifstr:
            # process_image(tifstr, args.tifdir, args.savedir)
            
            print("processed .tif image:", tifstr, "as .npy data. Done with file #:", i)

            if args.arm == "test":
                process_mask(tifstr, args.xmldir, args.tifdir, args.savedirmasks)








