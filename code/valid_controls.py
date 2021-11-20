from PIL import Image, ImageOps  
from skimage import color
from skimage import io
from skimage.transform import resize
from skimage import transform
from skimage.filters import threshold_otsu
import skimage

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os
import tifffile as tiff
from argparse import ArgumentParser
import argparse
import random
from matplotlib.patches import Circle
import imageio
import cv2

import pdb
import json

import utils
from utils import str2bool

np.random.seed(999)

# 16 bit:
#--------
# min_bit = 0
# max_bit = 65535

# swapped to an already normalized version
min_bit = 0 
max_bit = 1

shift_bit = 0.6 # used to be 0.75


#----------
# HELPERS
#----------
def read_binary_mask(data_path, numpy_flag=False):
	"""
	Draws a binarized version of the input mask to use for downstream image creation.
	Uses otsu binarization/thresholding to accomplish this.
	"""
	if numpy_flag == False:
		mask = mpimg.imread(data_path)
	else:
		mask = np.load(data_path)

	h,w,_ = mask.shape

	target_h, target_w = 3000, 3000
	scale_h, scale_w = int(target_h // h), int(target_w // w)
	scale = int(np.min([scale_h, scale_w]))

	# color image
	mask_resized = resize(mask, (h*scale, w*scale))

	return mask_resized

def draw_binary_mask_otsu(data_path, numpy_flag=False):
	"""
	Draws a binarized version of the input mask to use for downstream image creation.
	Uses otsu binarization/thresholding to accomplish this.
	"""

	if numpy_flag == False:
		bitmap = mpimg.imread(data_path)
	else:
		bitmap = np.load(data_path)

	h,w,_ = bitmap.shape

	target_h, target_w = 3000, 3000
	scale_h, scale_w = int(target_h // h), int(target_w // w)
	scale = int(np.min([scale_h, scale_w]))

	# color image
	bitmap_resized = resize(bitmap, (h*scale, w*scale))

	# grayscale - 1D in channel space
	bitmap_resized_gray = np.expand_dims(color.rgb2gray(bitmap_resized), 2)

	# Binarized mask - manual thresholding 
	thresh = threshold_otsu(bitmap_resized_gray)

	mask = bitmap_resized_gray > thresh

	return mask


def draw_binary_mask(data_path, min_thresh=0.1, max_thresh=0.9, scale=10, numpy_flag=False):
	"""
	Draws a binarized version of the input mask to use for downstream image creation.
	"""
	if numpy_flag == False:
		bitmap = mpimg.imread(data_path)
	else:
		bitmap = np.load(data_path)

	h,w,_ = bitmap.shape

	if scale == None:
		target_h, target_w = 3000, 3000
		scale_h, scale_w = int(target_h // h), int(target_w // w)
		scale = int(np.min([scale_h, scale_w]))

	# color image
	bitmap_resized = resize(bitmap, (h*scale, w*scale))

	# grayscale - 1D in channel space
	bitmap_resized_gray = np.expand_dims(color.rgb2gray(bitmap_resized), 2)

	# Binarized mask - manual thresholding 
	mask = ((bitmap_resized_gray > min_thresh) & (bitmap_resized_gray < max_thresh))

	return mask


def check_partition_status(args):
	if not os.path.exists(args.split_path):
		os.makedirs(args.split_path)
	else:
		if args.partition_overwrite_flag == False:
			print("\nOld partition directory detected and overwrite=False. Please either (1) turn overwrite flag to True or (2) delete old directory+rerun this script if you wish to overwrite it.")
			quit()
		elif args.partition_overwrite_flag == True:
			print("\nOld partition directory detected and overwrite=True. Clearing and overwriting...")
			for cn in os.listdir(args.split_path):
				os.remove(args.split_path + "/" + cn)
	return


# helper methods for partitions
def draw_affine_transform(Dp, temp_flag, mask_path, split_path, idx, mask_str, mask, seed_multiplier, current_split, rotation=1.5, shear=0.5, translation=(3000,800), scale=(1,1)):

	img = mask

	# Create Afine transform
	affine_tf = transform.AffineTransform(rotation=rotation, shear=shear, translation=translation, scale=scale)

	# Apply transform to image data
	modified = transform.warp(img, inverse_map=affine_tf)
	# img_str = str(idx) + "-" + mask_str + "-" + temp_flag + "-affine-rot" + str(rotation) + "-shear" + str(shear) + "-scale" + str(scale[0]) + "-" + current_split
	img_str = mask_str + "-" + temp_flag + "-affine-rot" + str(rotation) + "-shear" + str(shear) + "-scale" + str(scale[0]) + "-" + current_split

	# np.save(split_path + "/reg" + img_str, modified)
	print("NOTE: SAVING DEFORMED MASK!")
	if not os.path.exists(mask_path + "/deformed"):
		os.makedirs(mask_path + "/deformed")
	np.save(mask_path + "/deformed/reg" + img_str, modified)

	############################
	# save mask to mask path
	draw_1_hotcold(Dp, temp_flag, split_path, idx, img_str, modified, seed_multiplier, current_split, fuzzy_flag=True)
	############################

	# print("COMPLETED:", img_str)
	del img, mask, modified

	return


def draw_sinusoid_transform(Dp, temp_flag, mask_path, split_path, idx, mask_str, mask, seed_multiplier, current_split, k=3, axis="x", sinusoid="cos"):

	img = mask
	A = img.shape[0] / 3.0
	w = 2.0 / img.shape[1]

	# img_str = str(idx) + "-" + mask_str + "-" + temp_flag + "-" + sinusoid + "-" + axis + "-k=" + str(k) + "-" + current_split
	img_str = mask_str + "-" + temp_flag + "-" + sinusoid + "-" + axis + "-k=" + str(k) + "-" + current_split

	if sinusoid == "cos":
		shift = lambda x: A * np.cos(k/2*np.pi*x * w)
	elif sinusoid == "sin":
		shift = lambda x: A * np.sin(k/2*np.pi*x * w)

	if axis == "x":
		for j in range(img.shape[0]):
			img[:,j] = np.roll(img[:,j], int(shift(j)))
	elif axis == "y":
		for i in range(img.shape[0]):
			img[i,:] = np.roll(img[i,:], int(shift(i)))

	# np.save(split_path + "/reg" + img_str, img)
	print("NOTE: SAVING DEFORMED MASK!")
	if not os.path.exists(mask_path + "/deformed"):
		os.makedirs(mask_path + "/deformed")
	np.save(mask_path + "/deformed/reg" + img_str, img)

	############################	
	# save mask to mask path
	draw_1_hotcold(Dp, temp_flag, split_path, idx, img_str, img, seed_multiplier, current_split, fuzzy_flag=True)
	############################

	# print("COMPLETED:", img_str)
	del img, mask

	return


def draw_1_hotcold(Dp, temp_flag, split_path, idx, mask_str, mask, seed_multiplier, current_split, fuzzy_flag=False, distrib_shifted=False, superpixel_flag=False, fractal_flag=False, guilty_flag=False, guilty_coords=(None, None)):

	if fuzzy_flag == True:
		mask_str += "-fuzzy"

	if distrib_shifted == True:
		mask_str += "_distrib"

	H,W,D = mask.shape
	np.random.seed(idx*seed_multiplier)
	noisy = np.random.randint(low=min_bit, high=max_bit+1, size=(H,W,Dp)) # background noise to be added

	# do shift first
	if distrib_shifted == True:
		if temp_flag == "hot":
			mask = np.random.uniform(low=min_bit+(shift_bit*max_bit), high=max_bit, size=(H,W,D)) * mask
		elif temp_flag == "cold":
			mask = 1 - (np.random.uniform(low=min_bit+(shift_bit*max_bit), high=max_bit, size=(H,W,D)) * (1-mask))

	if temp_flag == "hot":
		if fuzzy_flag == False: 
			background = np.where(mask == min_bit, max_bit, min_bit) * np.expand_dims(noisy[:,:,0], axis=2)
			foreground = np.copy(mask) # np.where(mask != 0, mask, 0)
		elif fuzzy_flag == True:
			fuzzy_filter = np.random.choice(np.array([min_bit,max_bit]), size=noisy.shape, p=[0.1,0.9])
			background = np.where(mask == min_bit, max_bit, min_bit) * np.expand_dims(noisy[:,:,0], axis=2)
			foreground = np.copy(mask) * fuzzy_filter
			del fuzzy_filter

	elif temp_flag == "cold":
		if fuzzy_flag == False: 
			background = np.where(mask == max_bit, max_bit, min_bit) * np.expand_dims(noisy[:,:,0], axis=2)
			foreground = np.where(mask != max_bit, mask, min_bit)
		elif fuzzy_flag == True:
			if distrib_shifted == True:
				fuzzy_filter = np.random.choice(np.array([min_bit,max_bit]), size=noisy.shape, p=[0.9,0.1]) * 2
				background = np.where(mask == max_bit, max_bit, min_bit) * np.expand_dims(noisy[:,:,0], axis=2)
				foreground = np.where(mask != max_bit, mask, min_bit)
				binary = np.where(foreground != min_bit, max_bit, min_bit)
				foreground = (foreground + fuzzy_filter) * binary
				foreground = np.where(foreground >= max_bit, 1, foreground)
				del fuzzy_filter, binary
			elif distrib_shifted == False:
				fuzzy_filter = np.random.choice(np.array([min_bit,max_bit]), size=noisy.shape, p=[0.9,0.1])
				background = np.where(mask == max_bit, max_bit, min_bit) * np.expand_dims(noisy[:,:,0], axis=2)
				foreground = np.where(mask != max_bit, mask, max_bit) + fuzzy_filter
				binary = 1-mask
				foreground = foreground * binary
	else:
		print("Error: enter valid temp...")
		exit()

	onehotcold_channel = background + foreground
	del foreground, background
	
	if Dp > 1:
		# im = np.concatenate([onehotcold_channel, noisy[:,:,1:]], axis=2)
		print("Unsupported channel dimensionlity of >1... Exiting.")
	else:
		im = onehotcold_channel

	# add guilty if needed
	if guilty_flag == True:
		(r,c) = guilty_coords
		im[r,c,:] = 0

	img_str = str(idx) + "-" + mask_str + "-" + temp_flag + "-" + current_split
	if superpixel_flag == True:
		img_str += "_superpixels"
	if fractal_flag == True:
		img_str += "_fractal"

	np.save(split_path + "/reg" + img_str, im)
	print("COMPLETED:", img_str)
	del im, noisy, onehotcold_channel

	return


#----------------------------
# PARTITION-SPECIFIC ROUTINES
#----------------------------
def draw_extremepixel_partition(args, seed_multiplier, current_split):
	"""
	Creates the extreme value pixels OR distribution-shifted pixel partitions.
	"""	

	mask_path = args.mask_path
	data_path = args.data_path
	Dp = args.channel_dim
	partition = args.partition
	save_path = args.save_path
	split_path = args.split_path

	# checking if split folder exits, create one if not.
	# If so, abort and ask user to clear up folder before re-running
	check_partition_status(args)

	if partition == "extreme_value_pixels":
		print("INITIALIZING: Extreme Value Pixels")
	elif partition == "distribution_shifted_pixels":
		print("INITIALIZING: Distribution Shifted Pixels")


	#########################
	# SIMPLE SUB-PARTITON 
	#########################
	#note: (X) = visualized for accuracy in a notebook
	idx = 0
	print("\nSTARTING: SIMPLE SUB-PARTITION\n" + "-"*40)
	masks_str = os.listdir(mask_path)
	
	for i,f in enumerate(masks_str):
		if f.endswith(".npy"):
			mask_str = f.split(".")[0]

			for temp in ["hot", "cold"]:
				idx += 1
				mask = np.load(mask_path + "/" + f)
				if temp == "cold": #and partition == "extreme_value_pixels":
					mask = 1 - mask
				H,W,D = mask.shape

				# # distrib shifted masks
				# if partition == "distribution_shifted_pixels":
				# 	if temp == "hot":
				# 		mask = np.random.uniform(low=min_bit+(shift_bit*max_bit), high=max_bit, size=(H,W,Dp)) * mask
				# 	elif temp == "cold":
				# 		mask = 1 - (np.random.uniform(low=min_bit+(shift_bit*max_bit), high=max_bit, size=(H,W,Dp)) * (1-mask))

				if partition == "extreme_value_pixels":
					draw_1_hotcold(Dp, temp, split_path, idx, mask_str, mask, seed_multiplier, current_split)
				elif partition == "distribution_shifted_pixels":
					draw_1_hotcold(Dp, temp, split_path, idx, mask_str, mask, seed_multiplier, current_split, distrib_shifted=True)


	########################
	# FUZZY SUB-PARTITON
	########################
	print("\nSTARTING: FUZZY SUB-PARTITION\n" + "-"*40)
	
	for i,f in enumerate(masks_str):
		if f.endswith(".npy"):
			mask_str = f.split(".")[0]

			for temp in ["hot", "cold"]:
				idx += 1
				mask = np.load(mask_path + "/" + f)
				if temp == "cold": # and partition == "extreme_value_pixels":
					mask = 1 - mask
				H,W,D = mask.shape

				# # distrib shifted masks
				# if partition == "distribution_shifted_pixels":
				# 	if temp == "hot":
				# 		mask = np.random.uniform(low=min_bit+(shift_bit*max_bit), high=max_bit, size=(H,W,Dp)) * mask
				# 	elif temp == "cold":
				# 		mask = 1 - (np.random.uniform(low=min_bit+(shift_bit*max_bit), high=max_bit, size=(H,W,Dp)) * (1-mask))

				if partition == "extreme_value_pixels":
					draw_1_hotcold(Dp, temp, split_path, idx, mask_str, mask, seed_multiplier, current_split, fuzzy_flag=True)
				elif partition == "distribution_shifted_pixels":
					draw_1_hotcold(Dp, temp, split_path, idx, mask_str, mask, seed_multiplier, current_split, fuzzy_flag=True, distrib_shifted=True)

	return 


def draw_morphology_partition(args, seed_multiplier, current_split):

	mask_path = args.mask_path
	data_path = args.data_path
	Dp = args.channel_dim
	partition = args.partition
	save_path = args.save_path
	split_path = args.split_path

	# checking if split folder exits, create one if not.
	# If so, abort and ask user to clear up folder before re-running
	check_partition_status(args)

	print("INITIALIZING: Morphological Differences")

	########################
	# CANARIES
	########################
	idx = 0
	print("\nSTARTING: CANARY CREATION\n" + "-"*40)
	masks_str = os.listdir(mask_path)
	masks_str = [el for el in masks_str if el.endswith(".npy")] # make sure we don't include deformed masks
	masks_str.remove("canary.npy")
	temp = "hot"

	# pdb.set_trace()
	for i in range(len(masks_str) + 5): # 5*2=10 is the number of arbitrarily chosen deformations
		# 20 extreme-value fuzzy
		idx += 1
		mask = np.load(mask_path + "/" + "canary.npy")
		H,W,D = mask.shape
		draw_1_hotcold(Dp, temp, split_path, idx, "canary", mask, seed_multiplier, current_split)

		# 20 distrib-shifted fuzzy
		idx += 1
		# mask = np.random.uniform(low=min_bit+(shift_bit*max_bit), high=max_bit, size=(H,W,Dp)) * mask 
		# ^ if want distrib shift, legacy when transform not in 1_hotcold code
		draw_1_hotcold(Dp, temp, split_path, idx, "canary", mask, seed_multiplier, current_split, fuzzy_flag=True)
		del mask

	########################
	# OTHER MORPHOS & DEFORMS
	########################
	print("\nSTARTING: OTHER MORPHOLOGY CREATION\n" + "-"*40)
	# img_str = random.choice(os.listdir(split_path))
	img_str = mask_path + "/" + "canary.npy"
	axes = ["x", "x", "x", "x", "y", "y", "y", "y"]
	sinusoids = ["cos", "cos", "sin", "sin", "cos", "cos", "sin", "sin"]
	ks = [1, 3, 1, 3, 1, 3, 1, 3]
	rotations = [1.3, 1.5]
	shears = [0.7, 0.5]

	for i in range(10):
		idx += 1
		if i < 8:
			# rand_img = np.load(split_path + "/" + img_str)
			rand_img = np.load(img_str) # not random
			draw_sinusoid_transform(Dp, temp, mask_path, split_path, idx, "canary", rand_img, seed_multiplier, current_split, k=ks[i], axis=axes[i], sinusoid=sinusoids[i])
		if i >= 8:
			# rand_img = np.load(split_path + "/" + img_str)
			rand_img = np.load(img_str)	# not random
			draw_affine_transform(Dp, temp, mask_path, split_path, idx, "canary", rand_img, seed_multiplier, current_split, rotation=rotations[i-8], shear=shears[i-8])
		del rand_img
		
	# new masks
	for i,f in enumerate(masks_str):
		if f.endswith(".npy"):
			mask_str = f.split(".")[0]

			idx += 1
			mask = np.load(mask_path + "/" + f)
			H,W,D = mask.shape
			draw_1_hotcold(Dp, temp, split_path, idx, mask_str, mask, seed_multiplier, current_split)

			idx += 1
			# mask = np.random.uniform(low=min_bit+(shift_bit*max_bit), high=max_bit, size=(H,W,Dp)) * mask
		    # ^ if want distrib shift, legacy when transform not in 1_hotcold code
			draw_1_hotcold(Dp, temp, split_path, idx, mask_str, mask, seed_multiplier, current_split, fuzzy_flag=True)
			del mask
	return


def guilty_coords(img):

	H,W,D = img.shape

	# random sample function
	draw_rand = lambda lo, hi: np.random.randint(low=round(lo), high=round(hi))

	# sampling for "hot zone"
	cont_search = True
	while cont_search == True:
		y,x = draw_rand(H*0.25, H*0.75), draw_rand(W*0.25, W*0.75)
		if np.mean(img[y-10:y+10, x-10:x+10, :]) > 0.8: # probably in a "hot zone"
			cont_search = False

	rad = draw_rand(100, 300)
	r,c = skimage.draw.circle(y,x, rad, shape=(H,W,D))
	
	return r,c


def process_guilty_masks(guilty_mask_str, mask, mask_path):

	r,c = guilty_coords(mask)
	guilty_mask = np.copy(mask)
	guilty_mask[r,c,:] = 0 # reassign to guilty pixels

	print("NOTE: SAVING GUILTY MASK!")
	if not os.path.exists(mask_path + "/guilty"):
		os.makedirs(mask_path + "/guilty")
	np.save(mask_path + "/guilty/reg" + guilty_mask_str, guilty_mask)

	return r, c


def draw_guilty_partition(args, seed_multiplier, current_split):
	
	mask_path = args.mask_path
	data_path = args.data_path
	Dp = args.channel_dim
	partition = args.partition
	save_path = args.save_path
	split_path = args.split_path

	# checking if split folder exits, create one if not.
	# If so, abort and ask user to clear up folder before re-running
	check_partition_status(args)

	idx = 0
	print("INITIALIZING: Guilty Superpixels")
	masks_str = os.listdir(mask_path)
	temp = "hot"

	# std size hot images 
	for i,f in enumerate(masks_str):
		if f.endswith(".npy"):
			mask_str = f.split(".")[0]

			# regular
			idx += 1
			mask = np.load(mask_path + "/" + f)
			H,W,D = mask.shape
			draw_1_hotcold(Dp, temp, split_path, idx, mask_str, mask, seed_multiplier, current_split, fuzzy_flag=False)

			# regular guilty
			idx += 1
			guilty_mask_str = mask_str + "_guilty"
			r, c = process_guilty_masks(guilty_mask_str, mask, mask_path)
			draw_1_hotcold(Dp, temp, split_path, idx, guilty_mask_str, mask, seed_multiplier, current_split, fuzzy_flag=False, guilty_flag=True, guilty_coords=(r,c))

			# fuzzy
			idx += 1
			# mask = np.random.uniform(low=min_bit+(shift_bit*max_bit), high=max_bit, size=(H,W,Dp)) * mask
		    # ^ if want distrib shift, legacy when transform not in 1_hotcold code
			draw_1_hotcold(Dp, temp, split_path, idx, mask_str, mask, seed_multiplier, current_split, fuzzy_flag=True)
			
			# fuzzy guilty
			idx += 1
			r, c = process_guilty_masks(guilty_mask_str, mask, mask_path)
			draw_1_hotcold(Dp, temp, split_path, idx, guilty_mask_str, mask, seed_multiplier, current_split, fuzzy_flag=True, guilty_flag=True, guilty_coords=(r,c))

			del mask
	return


def add_margin(pil_img, top, right, bottom, left, color):

    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def draw_superpixel_hadamard_mask(idx, seed_multiplier, save_path, mask):
	
	H,W,D = mask.shape
	np.random.seed(idx*seed_multiplier)

	cell_mask_str = random.choice(os.listdir(save_path + "/cell_masks")) 
	cell_mask = np.load(save_path + "/cell_masks/" + cell_mask_str, allow_pickle=True)
	Ht,Wt,Dt = cell_mask.shape

	topbot = np.abs(H-Ht)//2
	lr = np.abs(W-Wt)//2
	cell_mask = Image.fromarray(np.squeeze(cell_mask, axis=2)) # PIL image

	# center cell mask crop
	if (W >= Wt) and (H >= Ht):
		cell_mask = add_margin(cell_mask, topbot, lr, topbot, lr, (0,0,0)) # padding
	elif (H < Ht) or (W < Wt): # takes care of case where only one dim is larger. Almost always the case.
		cell_mask = cell_mask.crop((lr, topbot, lr+W, topbot+H))
	
	cell_mask = np.array(cell_mask)
	mask = mask * np.expand_dims(cell_mask, axis=2)

	return mask


def draw_extremesuperpixel_partition(args, seed_multiplier, current_split):
	
	mask_path = args.mask_path
	data_path = args.data_path
	Dp = args.channel_dim
	partition = args.partition
	save_path = args.save_path
	split_path = args.split_path

	# checking if split folder exits, create one if not.
	# If so, abort and ask user to clear up folder before re-running
	check_partition_status(args)

	idx = 0
	print("INITIALIZING: Extreme Value Superpixels")
	masks_str = os.listdir(mask_path)

	# std size hot images 
	for i,f in enumerate(masks_str):
		if f.endswith(".npy"):
			mask_str = f.split(".")[0]
			character_mask = np.load(mask_path + "/" + f)

			idx += 1
			H,W,D = character_mask.shape
			mask = draw_superpixel_hadamard_mask(idx, seed_multiplier, save_path, character_mask)
			draw_1_hotcold(Dp, "hot", split_path, idx, mask_str, mask, seed_multiplier, current_split, fuzzy_flag=False, superpixel_flag=True)

			idx += 1
			mask = 1-mask
			draw_1_hotcold(Dp, "cold", split_path, idx, mask_str, mask, seed_multiplier, current_split, fuzzy_flag=False, superpixel_flag=True)

			idx += 1
			# mask = np.random.uniform(low=min_bit+(shift_bit*max_bit), high=max_bit, size=(H,W,Dp)) * character_mask
			# ^ if want distrib shift, legacy when transform not in 1_hotcold code
			mask = character_mask
			mask = draw_superpixel_hadamard_mask(idx, seed_multiplier, save_path, mask)
			draw_1_hotcold(Dp, "hot", split_path, idx, mask_str, mask, seed_multiplier, current_split, fuzzy_flag=True, superpixel_flag=True)
			
			idx += 1
			mask = 1-mask
			draw_1_hotcold(Dp, "cold", split_path, idx, mask_str, mask, seed_multiplier, current_split, fuzzy_flag=True, superpixel_flag=True)

			del mask, character_mask
	return


def draw_fractal_hadamard_mask(mask):
	
	H,W,D = mask.shape
	# pdb.set_trace()
	mask = mask.squeeze(2).astype(int)

	# check contents of mask
	total_pix = int(H * W)
	top_80p = mask > 0.8 # just in case we have distrib-shifted
	num_80p = np.sum(top_80p)
	percent_white = num_80p / total_pix

	# very filled edges/corners can go ahead and get fractalized! e.g. jerry west
	toprow1s, botrow1s = np.sum(mask[0,:]), np.sum(mask[-1,:])
	lcol1s, rcol1s = np.sum(mask[:,0]), np.sum(mask[:,-1])
	buff = 50
	tl1s, tr1s = np.sum(mask[0:buff,0:buff]), np.sum(mask[0:buff,-buff:-1])
	bl1s, br1s = np.sum(mask[0:buff,-buff:-1]), np.sum(mask[-buff:-1,-buff:-1])

	busy_edge_flag = False
	busy_corner_flag = False
	edge_thresh = 0.5
	if (toprow1s/W > edge_thresh) or (botrow1s/W > edge_thresh) or (lcol1s/H > edge_thresh) or (rcol1s/H > edge_thresh):
		busy_edge_flag = True
	if tl1s > 1 or tr1s > 1 or bl1s > 1 or br1s > 1:
		busy_corner_flag = True

	# pdb.set_trace()

	if percent_white > 0.5 or busy_edge_flag == True or busy_corner_flag == True:
		fractal_count = 10
		img = cv2.resize(np.float32(mask), dsize=(W//fractal_count, H//fractal_count), interpolation=cv2.INTER_AREA)
		img = np.tile(img, (fractal_count, fractal_count)) 
		img = np.where(img > 1, 1, img)

		img = Image.fromarray(np.uint8(img * 255), 'L') #Image.fromarray(img) 

	else: # small image
		# pdb.set_trace()

		print("Large image borders detected! Cropping mask before resizing+tiling.")
		fractal_count = 10
		# check "edges" of image
		# print(top_80p.shape)

		# pdb.set_trace()
		# max_H = np.max(np.where(top_80p)[0])
		# min_H = np.min(np.where(top_80p)[0])
		# max_W = np.max(np.where(top_80p)[1])
		# min_W = np.min(np.where(top_80p)[1])
		# print(min_H, max_H, min_W, max_W)
		# max_H, max_W = np.argwhere(top_98p == True).max(0)
		# min_H, min_W = np.argwhere(top_98p == True).min(0)

		top_80p_nums = np.where(top_80p == True, 1, 0)
		y = top_80p_nums.sum(1)
		x = top_80p_nums.sum(0)
		tol = 200

		min_W = np.argwhere(np.where(x > tol, 1, 0) == 1).min()
		max_W = np.argwhere(np.where(x > tol, 1, 0) == 1).max()
		min_H = np.argwhere(np.where(y > tol, 1, 0) == 1).min()
		max_H = np.argwhere(np.where(y > tol, 1, 0) == 1).max()

		img = Image.fromarray(np.uint8(mask * 255), 'L')  #Image.fromarray(mask)
		# pdb.set_trace()
		img = img.crop((min_W-buff, min_H-buff, max_W+buff, max_H+buff)) # L,T,R,B
		img = np.array(img) 
		img = cv2.resize(np.float32(img), dsize=(W//fractal_count, H//fractal_count), interpolation=cv2.INTER_AREA)
		img = np.tile(img, (6 * fractal_count, 6 * fractal_count)) # for safety, we scale count, crop will take care of later
		img = img / img.max()  # / 255.0 # renormalize to 0,1

		img = Image.fromarray(np.uint8(img * 255), 'L')  


	Ht,Wt = np.array(img).shape

	# PIL images
	mask = Image.fromarray(np.uint8(mask * 255), 'L') #Image.fromarray(mask)

	topbot = np.abs(H-Ht)//2
	lr = np.abs(W-Wt)//2

	if (Ht < H) or (Wt < W): # need to crop mask
		mask = mask.crop((lr, topbot, lr+Wt, topbot+Ht))
	elif (H < Ht) or (W < Wt): # need to crop img
		img = img.crop((lr, topbot, lr+W, topbot+H))
	
	# back to numpy arrays, renormalized
	img = np.array(img)
	mask = np.array(mask)
	img_norm = img / img.max()  # / 255.0 # renormalize to 0,1
	mask_norm = mask / mask.max()

	# mask = np.array(mask) 
	mask = np.expand_dims(mask_norm, 2) * np.expand_dims(img_norm, 2)
	# pdb.set_trace()

	del img, mask_norm, img_norm
	return mask


def draw_fractal_partition(args, seed_multiplier, current_split):

	mask_path = args.mask_path
	data_path = args.data_path
	Dp = args.channel_dim
	partition = args.partition
	save_path = args.save_path
	split_path = args.split_path

	# checking if split folder exits, create one if not.
	# If so, abort and ask user to clear up folder before re-running
	check_partition_status(args)

	idx = 0
	print("INITIALIZING: Fractals")
	masks_str = os.listdir(mask_path)
	temp = "hot"

	# std size masks - large!
	for i,f in enumerate(masks_str):
		if f.endswith(".npy"):
			mask_str = f.split(".")[0]
			character_mask = np.load(mask_path + "/" + f)
			character_mask2 = np.copy(character_mask)
			H,W,D = character_mask.shape

			# regular 
			idx += 1
			draw_1_hotcold(Dp, temp, split_path, idx, mask_str, character_mask, seed_multiplier, current_split, fuzzy_flag=False, fractal_flag=False)

			# fractal 
			idx += 1
			fractal_mask = draw_fractal_hadamard_mask(character_mask2)
			draw_1_hotcold(Dp, temp, split_path, idx, mask_str, fractal_mask, seed_multiplier, current_split, fuzzy_flag=False, fractal_flag=True)

			# regular fuzzy
			idx += 1
			mask = character_mask #only use for distrib-shift: np.random.uniform(low=min_bit+(shift_bit*max_bit), high=max_bit, size=(H,W,Dp)) * character_mask
			# ^ if want distrib shift, legacy when transform not in 1_hotcold code
			draw_1_hotcold(Dp, temp, split_path, idx, mask_str, mask, seed_multiplier, current_split, fuzzy_flag=True, fractal_flag=False)

			# fractal fuzzy
			idx += 1
			fractal_mask = draw_fractal_hadamard_mask(mask)
			draw_1_hotcold(Dp, temp, split_path, idx, mask_str, fractal_mask, seed_multiplier, current_split, fuzzy_flag=True, fractal_flag=True)

			del mask, fractal_mask, character_mask
	return


#----------------------------
# MAIN ROUTINES
#---------------------------
def draw_partition(args, draw_func, seed_multiplier, split):
	
	data_path = args.data_path 
	channel_dim = args.channel_dim
	split = args.split
	partition = args.partition
	save_path = args.save_path

	if split == "both":
		for i,s in enumerate(seed_multiplier):
			if i == 0:
				current_split = "train"
			else:
				current_split = "test"

			print("Currently on set split:", current_split, '\n'+"~"*40)
			
			split_path = save_path + "/" + str(channel_dim) + "-channel"  + "/" + partition + "/" + current_split 
			setattr(args, "split_path", split_path)

			draw_func(args, s, current_split)
			print("\nFINISHED SPLIT:", current_split, "\n" + "="*60)

		thumbify = True
	
	else:
		current_split = split
		split_path = save_path + "/" + str(channel_dim) + "-channel"  + "/" + partition + "/" + current_split 
		setattr(args, "split_path", split_path)

		if current_split == "train":
			thumbify = True
		else:
			thumbify = False # don't make thumnails of test set

		draw_func(args, seed_multiplier, current_split)
		print("\nFINISHED SPLIT:", current_split, "\n" + "="*60)

	# Generate Thumbnails
	if thumbify == True: 
		print("\nCreating PNG thumnails for quick viewing....")
		thumb_path = data_path + "/PIC-1_thumbnails/" + str(channel_dim) + "-channel"  + "/" + partition + "/" + "train" # current_split 
	else:
		print("Forgoing PNG thumbnails since only test set was created... Exiting")
		return

	if channel_dim == 1:
		if args.partition_overwrite_flag == True:
			print("Overwriting old thumbnails....")
			if os.path.exists(thumb_path):
				for tn in os.listdir(thumb_path):
					if tn.endswith(".jpg"):
						os.remove(thumb_path + "/" + tn)

		if not os.path.exists(thumb_path):
			os.makedirs(thumb_path)

		# only save train as thumbnails
		split_path_tosave = save_path + "/" + str(channel_dim) + "-channel"  + "/" + partition + "/" + "train" #current_split 

		for f in os.listdir(split_path_tosave): 
			print("making JPG for:", f)
			if f.endswith(".npy"):
				# pdb.set_trace()
				img = np.load(split_path_tosave + "/" + f)
				img = Image.fromarray(np.uint8(img[:,:,0] * 255), 'L') # old: Image.fromarray(img[:,:,0])
				img = img.resize([int(0.25 * s) for s in img.size]) # compress

				plt.figure()
				plt.imshow(img, cmap="gray")
				# plt.clim(0, 1)
				# plt.colorbar()
				plt.axis('off')
				numpy_img_str = f.split("/")[-1]
				plt.savefig(thumb_path + "/" + ''.join(numpy_img_str.split(".")[:-1]) + ".jpg", bbox_inches='tight')
				plt.close()



def draw_fresh_controls(args):

	data_path = args.data_path # directory with all bitmaps
	channel_dim = args.channel_dim
	split = args.split
	partition = args.partition
	save_path = args.save_path

	mask_path = save_path + "/masks" 
	setattr(args, "mask_path", mask_path) # adding mask_path to the dict

	# make the mask directory
	if not os.path.exists(mask_path):
		os.makedirs(mask_path)

	#============
	# OVERWRITE
	#============
	if args.mask_overwrite_flag == True:
		print("Generating masks / overwriting old masks....")
		
		if os.path.exists(mask_path + "/deformed/"):
			print("removing deformed masks")
			for mn in os.listdir(mask_path + "/deformed/"):
				if os.path.isfile(mn):
					os.remove(mask_path + "/deformed/" + mn) # deformed masks

		if os.path.exists(mask_path + "/guilty/"):
			print("removing guilty masks")
			for mn in os.listdir(mask_path + "/guilty/"):
				if os.path.isfile(mn):
					os.remove(mask_path + "/guilty/" + mn) # guilty masks

		for mn in os.listdir(mask_path):
			if os.path.isfile(mn):
				os.remove(mask_path + "/" + mn) # regular masks
	
		if args.bin_flag == False:
			print("User specified non-binarized inputs... Continuing with binarization")
		elif args.bin_flag == True:
			print("User specified already binarized inputs... Reading in, rescaling, and saving as numpy arrays")
		else: 
			print("Error: Please select a valid set of parameters. See help docstrings.")
			exit()
			
		print("")

		# CALL MASK CREATION HERE
		for i,f in enumerate(os.listdir(data_path)):
			
			if args.bin_flag == False:
				if args.manualbin_flag == True and args.manualbin_settings_path != None: # if want manual						
					if f.endswith(".npy"):
						print("Generating a MANUAL binary mask for input NUMPY ARRAY:", f)
						bin_args = args.manualbin_settings[f] #utils.thresh_dict[f] -- old usage
						mask_fill = draw_binary_mask(data_path + "/" + f, bin_args[0], bin_args[1], bin_args[2], numpy_flag=True)
					elif f.endswith(".jpeg") or f.endswith(".jpg") or f.endswith(".png"):
						print("Generating a MANUAL binary mask for input IMAGE:", f)
						bin_args = args.manualbin_settings[f] #utils.thresh_dict[f] -- old usage
						mask_fill = draw_binary_mask(data_path + "/" + f, bin_args[0], bin_args[1], bin_args[2])
					else:
						continue

				elif args.manualbin_flag == False: # automatic	
					if f.endswith(".npy"):
						print("Generating a AUTOMATED/OTSU binary mask for input NUMPY ARRAY:", f)
						mask_fill = draw_binary_mask_otsu(data_path + "/" + f, numpy_flag=True)
					elif f.endswith(".jpeg") or f.endswith(".jpg") or f.endswith(".png"):
						print("Generating a AUTOMATED/OTSU binary mask for input IMAGE:", f)
						mask_fill = draw_binary_mask_otsu(data_path + "/" + f)
					else:
						continue

				else: # manual_bin not T/F
					print("Error: Please select a valid set of parameters. See help docstrings.")
					exit()

			elif args.bin_flag == True:
				if f.endswith(".npy"):
					print("Only rescaling for megapixel size for input NUMPY ARRAY:", f)
					mask_fill = read_binary_mask(data_path + "/" + f, numpy_flag=True)
				elif f.endswith(".jpeg") or f.endswith(".jpg") or f.endswith(".png"):
					print("Rescaling and converting to numpy array for iunput IMAGE:", f)
					mask_fill = read_binary_mask(data_path + "/" + f)
				else:
					continue

			obj_name = f.split(".")[0]
			np.save(mask_path + "/" + obj_name, mask_fill)
			del mask_fill

	#=================
	# DON'T OVERWRITE
	#=================
	elif len(os.listdir(mask_path)) > 0 and args.mask_overwrite_flag == False:
		print("Old masks directory detected and overwrite=false... Continuing to generating synthetic images!")

		if args.partition != "morphological_differences":
			for mn in os.listdir(mask_path): # safety check
				if ("-sin-" in mn) or ("-cos-" in mn) or ("-affine-" in mn):
					print("Detecting old morphological deformation masks... Deleting those for now to avoid partition mixing")
					os.remove(mask_path + "/" + mn)


	if args.mask_debug_flag == True:
		print("Early exit for mask debugging!")
		exit()

	# set random seeds for train and test
	if split == "both":
		seed_multiplier = [1,100]
	elif split == "train":
		seed_multiplier = 1
	elif split =="test":
		seed_multiplier = 100		

	if partition == "extreme_value_pixels" or partition == "distribution_shifted_pixels":
		draw_func = draw_extremepixel_partition
	elif partition == "morphological_differences":
		draw_func = draw_morphology_partition
	elif partition == "guilty_superpixels":
		draw_func = draw_guilty_partition
	elif partition == "extreme_value_superpixels":
		draw_func = draw_extremesuperpixel_partition
	elif partition == "fractal_morphologies":
		draw_func = draw_fractal_partition
	else:
		print("Error: Please choose a supported partition.")
		quit()

	print("=" * 50)
	print("=" * 50)
	print("BEGINNING CREATION OF", partition, "PARTITION!")
	print("=" * 50)
	print("=" * 50)

	# run
	draw_partition(args, draw_func, seed_multiplier, split)


def main():
	# ARGPARSE
	#---------
	parser = ArgumentParser()
	parser.add_argument('--data_path', default=None, type=str, help='Path for either your input (1) 2D color bitmaps or (2) binary masks, both of which are PNG or JPG format.')
	parser.add_argument('--bin_flag', default=True, type=str2bool, help='T/F of whether or not data_path contains binary masks or not. If starting with color images, choose False. Defaults to True.')
	parser.add_argument('--manualbin_flag', default=False, type=str2bool, help='T/F for manual binarization if bin_flag=False. Defaults to False, which is an automatic (Otsu) method.')
	parser.add_argument('--manualbin_settings_path', default=None, type=str, help='Path for manual binarization settings. In that file, please build python dictionary with {filename: [min_threshold, max_threshold, scale]}.')
	parser.add_argument('--mask_overwrite_flag', default=True, type=str2bool, help='T/F for wanting to overwrite binary mask creation.')
	parser.add_argument('--partition_overwrite_flag', default=True, type=str2bool, help='T/F for wanting to overwrite synthetic partition creation.')
	parser.add_argument('--channel_dim', default=1, type=int, help='Number of desired output channels. Defaults to 35.')
	parser.add_argument('--split', default="train", type=str, help='Choose one of: {train, test, both}. Defaults to both.')
	parser.add_argument('--partition', default="extreme_value_pixels", type=str, help='Choose one of: {extreme_value_pixels, distribution_shifted_pixels, morphological_deformation, instance_size, number_of_instances}. Defaults to extreme_value_pixels.')
	parser.add_argument('--save_path', default=None, type=str, help='Save path for newly created megapixel controls. No default, will error out with no provided path.')
	parser.add_argument('--mask_debug_flag', default=False, type=str2bool, help='Do you want to abort early to check masks? Input True for early exit before syntehtic control generation.')

	args = parser.parse_args()
	print("\n\n\n")

	print("=" * 50)
	print("=" * 50)
	print("BEGINNING CREATION OF MASKS")
	print("=" * 50)
	print("=" * 50)

	if args.bin_flag == False:
		print("User specified non-binarized input images...")
		if args.manualbin_flag == True:
			print("User specified desire to manually binarize input images...")
			if args.manualbin_settings_path == None:
				print("Error: ...However, no manual settings path has been specified. Exiting...")
				exit()
			else:
				print("Commencing manual binarization of input images w.r.t settings specified in:", args.manualbin_settings_path)
				# pdb.set_trace()
				with open(args.manualbin_settings_path, encoding='utf-8', errors='ignore') as json_data:
					manualbin_settings = json.load(json_data, strict=False)
				setattr(args, "manualbin_settings", manualbin_settings)
		else:
			print("Commencing automatic binarization of input images")
	else:
		print("User specified pre-binarized images! Skipping (automated/manual) binarization procedure for input images...")
		
	if args.channel_dim > 1:
		print("Error: we only support channel dimensionality of 1 at this time. Feel free to leave out the argument and we will default to D=1.")
		quit()

	if args.data_path == None or args.save_path == None:
		print("Error: please indicate a bitmap path and save path. Use help docstring if needed.")
		quit()

	if args.split not in ["train", "test", "both"]:
		print("Error: please choose valid split(s): train/test/both. See help.")
		exit()
	
	draw_fresh_controls(args)


if __name__ == "__main__":
	main()

















