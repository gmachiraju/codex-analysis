# adapted from: https://gist.github.com/drbeh/a54c9da7826fd98558b61ba96014375e
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


def download_camelyon16_image(filename, savepath):
    filename = filename.lower()
    if os.path.exists(filename):
        print(f"The image [{filename}] already exist locally.")
    else:
        print(f"Downloading '{filename}'...")
        prefix = filename.split("_")[0].lower()
        if prefix == "test":
            arm = "test"
            folder_name = "testing/images"
        elif prefix in ["normal", "tumor"]:
            arm = "train"
            folder_name = f"training/{prefix}"
        else:
            raise ValueError(
                f"'{filename}' not found on the server."
                " File name should be like 'test_001.tif', 'tumor_001.tif', or 'normal_001.tif'"
            )
        path = f"gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/{folder_name}/"
        ftp = ftplib.FTP("parrot.genomics.cn")
        ftp.login("anonymous", "")
        ftp.cwd(path)
        ftp.retrbinary("RETR " + filename, open(savepath+"/"+arm+"/"+filename, "wb").write)
        # savepath+"/"+arm+"/"+filename | filename
        print("Saved:", filename)
        ftp.quit()


def download_camelyon17_image(filename, savepath):
    filename = filename.lower()
    if os.path.exists(filename):
        print(f"The image [{filename}] already exist locally.")
    else:
        print(f"Downloading '{filename}'...")
        prefix = filename.split("_")[0].lower()
        suffix = filename.split(".")[1].lower()
        id_num = int(filename.split(".")[0].split("_")[1])
        if prefix == "patient" and suffix == "zip":
            arm = "val"
            if id_num < 20:
                center = "center_0"
            elif id_num >= 20 and id_num <= 39:
                center = "center_1" 
            elif id_num >= 40 and id_num <= 59:
                center = "center_2"
            elif id_num >= 60 and id_num <= 79:
                center = "center_3"
            elif id_num >= 80 and id_num <= 99:
                center = "center_4"
            folder_name = "training/" + center
        else:
            raise ValueError(
                f"'{filename}' not found on the server."
                " File name should be like 'patient_001.zip'"
            )
        path = f"gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/{folder_name}/"
        ftp = ftplib.FTP("parrot.genomics.cn")
        ftp.login("anonymous", "")
        ftp.cwd(path)
        ftp.retrbinary("RETR " + filename, open(savepath+"/"+arm+"/"+filename, "wb").write)
        # savepath+"/"+arm+"/"+filename | filename
        print("Saved:", filename)
        ftp.quit()



def download_camelyon16_all(savepath):
    for img_type in ["normal", "tumor", "test"]:
        if img_type == "normal": #train
            for i in range(1,161):
                filename = img_type + "_" + str(i).zfill(3) + ".tif"
                download_camelyon16_image(filename, savepath)
            print("--------FINISHING NORMAL--------")
            
        elif img_type == "tumor": #train
            for i in range(1,112):
                filename = img_type + "_" + str(i).zfill(3) + ".tif"
                download_camelyon16_image(filename, savepath)
            print("--------FINISHING TUMOR--------")
            
        elif img_type == "test": 
            for i in range(1,131):
                filename = img_type + "_" + str(i).zfill(3) + ".tif"
                download_camelyon16_image(filename, savepath)
            print("--------FINISHING TEST--------")



if __name__ == "__main__":
    
    # DOWNLOAD TIFs
    #----------------
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument("--filename", type=str, help="The image name in Camelyon16 dataset to be downloaded.")
    arg_parser.add_argument("--savedir", type=str, help="Absolute path for saving images.")
    arg_parser.add_argument("--whichcam", type=str, help="Cam17 or Cam16?")

    args = arg_parser.parse_args()

    if args.whichcam == "cam16":
        if args.filename == "all":
            download_camelyon16_all(args.savedir)
        else:
            download_camelyon16_image(args.filename, args.savedir)
    elif args.whichcam == "cam17":
        download_camelyon17_image(args.filename, args.savedir)


