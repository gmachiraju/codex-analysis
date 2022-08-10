import argparse
import ftplib
import os
import pdb
import openslide
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import draw
import skimage
from skimage import measure

from utils import serialize, deserialize, str2bool
LV=3 # was 5

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

    return dim, desired_dims


def get_foreground_maps(sample_id, data_path, level=LV):
    


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
    
    
def process_mask(sample_id, xml_path, data_path, save_path):
    img_file = data_path + "/" + sample_id + ".tif"
    xml_file = xml_path + "/" + sample_id + ".xml"
    dim, new_dim = get_imgdims(sample_id, data_path)
    
    save_list, ys, xs = [], [], []
    annot_count = 0
    with open(xml_file, 'r') as infile:
        for i, line in enumerate(infile):
            if "<Annotation Name=" in line:
                annot_count += 1
                annot = []
                x, y = [], []
            elif "<Coordinate Order=" in line:
                xy = line.split(" ")[2:4]
                xy = [float(xy[0].split("=")[1].strip('\"')), float(xy[1].split("=")[1].strip('\"'))]
                x.append(xy[0]) 
                y.append(xy[1])
                annot.append(xy)
            elif "</Annotation>" in line:
                if annot_count > 0:
                    save_list.append(annot)
                    xs.append(x)
                    ys.append(y)
    
    mask = np.zeros(new_dim)
    print("mask shape is:", mask.shape)

    for i in range(len(save_list)):
        image = draw.polygon2mask(dim, np.stack((xs[i], ys[i]), axis=1))
        image = cv2.resize(np.float32(image), new_dim, interpolation=cv2.INTER_AREA)
        mask += image
        del image
            
    np.save(save_path + "/" + sample_id + ".npy", mask)
    del mask

    print("Finished saving mask:", sample_id)
    print("of shape:", mask.shape)
    print("")



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








