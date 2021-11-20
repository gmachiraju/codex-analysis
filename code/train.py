import numpy as np
import os 
import pdb 
import argparse
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import gc

import skimage.io
import skimage
from sklearn.metrics import roc_auc_score as skroc
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.models as models
from torchvision import transforms as trn
from torchsummary import summary

from torchviz import make_dot
from sklearn.decomposition import PCA

# personal imports
from dataloader import DataLoader
import utils
from utils import labels_dict, count_files, unique_files, set_splits
from utils import train_dir, val_dir, test_dir
from utils import serialize, deserialize, str2bool

from models import VGG19, VGGEmbedder, AttnVGG_before

# Constants/defaults
#-----------
LEARN_RATE = 5e-4 #1e-4 #5e-5 #1e-5
LEARN_RATE_IMG = 8e-2 #5e-2 #1e-2

USE_GPU = True
EPOCHS = 10
ALPHA = 0.01 # used to be 1, 0.1
NUM_SUBEPOCH = 100 #50 #100

print_every = 10
val_every = 20
# bs = 36
ppb = 5
dtype = torch.float32

LAMBDA1 = 0.5
LAMBDA2 = 0.01
L1FLAG = False

# #-------------------------
# taskcombo_flag = "uncertainty"
# graph_flag = True
# #-------------------------


def flatten(x):
    N = x.shape[0] # read in N, C, H, W, where N is batch size
    return x.view(N, -1)


class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)
    

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    def forward(self, x):
        print(x.shape)
        return x


def pool(data, mode='max'):
    """
    pool over 25 sub-patch slices
    
    Inputs
    - data: numpy array of shape (N, H, W, C)
    
    Returns
    - pooled: numpy array of shape (H, W, C)
    """
    pooled = None
    N, C, H, W = data.shape
    if mode == 'max':
        pooled = torch.max(data, dim=0)[0]
    elif mode == 'mean':
        pooled = torch.mean(data, dim=0)     
    return pooled


def pool_batch(data, batch_size=ppb, mode='max'):
    arr = []
    splits = torch.split(data, 25)
    for split in splits:
        pooled = pool(split, mode)
        arr.append(pooled)
    arr = torch.stack(arr, dim=0)
    return arr
        

def pool_labels(labels):
    seq = list(range(0, labels.shape[0], 25))
    return labels[seq]


def maxpool_embeds(embeds):
	# pdb.set_trace()
	cats = torch.stack(embeds, dim=0)
	# print(cats.shape)
	pooled = torch.max(cats, dim=0)[0]
	# print(pooled.shape)
	return pooled


def check_mb_accuracy(scores, ys):
	_, preds = scores.max(1)
	probs = F.softmax(scores, dim=1) # used to slice like this: [:,1] # but let's instead take both

	num_correct = (preds == ys).sum()
	num_samples = preds.size(0)
	return preds, probs, num_correct, num_samples


def forward_pass_VGGs_eval(x, y, model, model_class, device):
	x = torch.from_numpy(x)
	y = torch.from_numpy(y)

	x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
	y = y.to(device=device, dtype=torch.long)

	if model_class == "VGG_att":
		[scores, c1, c2, c3] = model(x)
	else: #VGG19 and VGG19_bn
		scores = model(x)
		
	val_loss = F.cross_entropy(scores, y)
	preds, probs, num_correct, num_samples = check_mb_accuracy(scores, y)

	return val_loss.item(), num_correct, num_samples, probs.tolist(), preds.tolist(), y.tolist()


def check_patch_accuracy(model, args, batch_flag=False, model_other=None):
	
	num_correct, num_samples, cum_loss = 0, 0, 0
	losses, probs, preds, patch_names, labels = [], [], [], [], []
	model.eval()  # set model to evaluation mode

	if args.model_class.startswith("VGG") == True:
		loader = DataLoader(args)
	else:
		print("choose valid model type! See help docstring!")
		exit()

	if "VGG" in args.model_class:
		with torch.no_grad():
			for i, (fxy, x, y) in enumerate(loader):
				# run batches
				loss, correct, samples, probs_batch, preds_batch, labels_batch = forward_pass_VGGs_eval(x, y, model, args.model_class, args.device)
				losses.append(loss)
				probs.extend(probs_batch)
				preds.extend(preds_batch)
				patch_names.extend(fxy)
				labels.extend(labels_batch)

				num_correct += correct
				num_samples += samples
				cum_loss += loss

				if batch_flag == True:
					break
		 
			acc = float(num_correct) / num_samples
			print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
			print('Cumulative Loss scaled by iter: {0:0.4f}'.format(cum_loss / i))

	return num_correct, num_samples, cum_loss, losses, probs, preds, patch_names, labels


# Training routines
#------------------
def store_embeds(embed_dict, fxy, x, trained_model, att_flag=False):
	
	x = x.detach().clone() 
	embedder = VGGEmbedder(trained_model, att_flag)
	z = embedder(x)
	z = z.detach().clone()
 	# ^ from: https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second-time-but

	for i, f in enumerate(fxy):
		im_folder = f.split(".")[0]  # "reg{idx}_montage"
		im_id = im_folder.split("_")[0]  # "reg{idx}""
		idx = im_id.split("reg")[1]  # {idx}
		
		embed_dict[idx].append(z[0, :])
		z = z[1:, :] # delete 0th row to match with f

		embed_dict[idx] = [maxpool_embeds(embed_dict[idx])]

	gc.collect()
	torch.cuda.empty_cache()

	return embed_dict



def train_VGGs(model, device, optimizer, args, save_embeds_flag=True):
	"""
	Train a model on image data using the PyTorch Module API.

	Inputs:
	- model: A PyTorch Module giving the model to train.
	- optimizer: An Optimizer object we will use to train the model
	- epochs: (Optional) A Python integer giving the number of epochs to train for

	Returns: Nothing, but prints model accuracies during training.
	"""
	train_losses = []
	model = model.to(device=device) # move the model parameters to CPU/GPU

	att_flag = False
	if args.model_class == "VGG_att":
		att_flag = True
	print("training", args.model_class)

	if args.model_class.startswith("VGG") == True:
		if args.model_class != "VGG_att": # this errors for some reason
			summary(model, input_size=(args.channel_dim, args.patch_size, args.patch_size)) # print model
		train_loader = DataLoader(args)
	else:
		print("choose valid model type! See help docstring!")
		exit()

	model.train()  # put model to training mode

	# files loaded differently per model class
	for e in range(args.num_epochs):
		if save_embeds_flag == True:
			embed_dict = defaultdict(list)

		for t, (fxy, x, y) in enumerate(train_loader):
			x = torch.from_numpy(x)
			y = torch.from_numpy(y)

			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.long)

			if args.model_class == "VGG_att":
				[scores, c1, c2, c3] = model(x)
			else: #VGG19 and VGG19_bn
				scores = model(x)

			train_loss = F.cross_entropy(scores, y)

			# Zero out all of the gradients for the variables which the optimizer
			# will update.
			optimizer.zero_grad()

			# This is the backwards pass: compute the gradient of the loss with
			# respect to each  parameter of the model.
			train_loss.backward()

			# Actually update the parameters of the model using the gradients
			# computed by the backwards pass.
			optimizer.step()

			if t % print_every == 0:
				print('Iteration %d, train loss = %.4f' % (t + print_every, train_loss.item()))
				preds, probs, num_correct, num_samples = check_mb_accuracy(scores, y)
				acc = float(num_correct) / num_samples
				print('minibatch training accuracy: %.4f' % (acc * 100))

			train_losses.append(train_loss.item())

			# if save_embeds_flag == True:
			# 	embed_dict = store_embeds(embed_dict, fxy, x, model, att_flag)

			# 	# save embeddings per epoch // overwrite each epoch's embeddings for now
			# 	serialize(embed_dict, utils.code_dir + args.model_class + "-epochs" + str(args.num_epochs) + "-max_embeddings_train.obj")
			# move away from utils.code_dir and instead ask for a cache_dir in args

		# save model per epoch --> skipping for now
		#      torch.save(model, args.model_path + "/" + args.string_details + "_epoch%s.pt" % e)
		# could also check val acc every epoch

	# full model save
	torch.save(model, args.model_path + "/" + args.string_details + "_full.pt")

	return train_losses



# Main routine
#-------------
def main():

	# ARGPARSE
	#---------
	parser = argparse.ArgumentParser()
	parser.add_argument('--description', default="no-description", type=str, help='Description of your experiement, with no spaces. E.g. VGG19_bn-random_loading-label_inherit-bce_loss-on_MFL-1')
	parser.add_argument('--model_class', default=None, type=str, help='Select one of: VGG19/VGG19_bn/VGG_att.')
	parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs to train for. Default is 10.')
	parser.add_argument('--hyperparameters', default=0.01, type=float, help="Denotes hyperparameters for custom losses. Only used for Uncertainty Loss. Default value is alpha=0.1, where a higer value indicates more focus on img-level predictions")
	parser.add_argument('--batch_size', default=36, type=int, help="Batch size. dDfault is 36.")
	parser.add_argument('--channel_dim', default=1, type=int, help="Channel dimension. Default is 1.")
	parser.add_argument('--normalize_flag', default=False, type=str2bool, help="T/F if patches need normalization. Default is False.")
	parser.add_argument('--dataset_name', default=None, type=str, help="What you want to name your dataset. For pre-defined label dictionaries, use: u54codex to search utils.")
	parser.add_argument('--dataloader_type', default="stored", type=str, help="Type of data loader: stored vs otf (on-the-fly).")

	# parameters for patches
	parser.add_argument('--patch_size', default=96, type=int, help="Patch/instance size. Default is 96.")
	parser.add_argument('--patch_loading', default="random", type=str, help="Patch loading scheme: random or blocked. Default is random.")
	parser.add_argument('--patch_labeling', default="inherit", type=str, help="Patch labeling function: inherit or proxy. Default is inhert.")
	parser.add_argument('--patch_loss', default="bce", type=str, help="Patch loss function. Default is bce. Future support for uncertainty.")
	
	# paths
	parser.add_argument('--data_path', default=None, type=str, help="Dataset path. If patches, will use stored data loader, if images, will use OTF data loader.")
	parser.add_argument('--patchlist_path', default=None, type=str, help="Patch list path. This is a cached result of the preprocess.py script.")
	parser.add_argument('--labeldict_path', default=None, type=str, help="Label dictionary path. This is a cached result of the preprocess.py script.")
	parser.add_argument('--model_path', default=None, type=str, help="Where you'd like to save the model outputs.")

	# parse for sanity checks
	args = parser.parse_args()
	# print(args)

	if args.model_class == None:
		print("No model entered. Please choose a model using the parser help flag. Exiting...")
		exit()

	if args.data_path == None or args.patchlist_path == None:
		print("No data path or patchlist path entered. Exiting...")
		exit()

	if args.model_path == None:
		print("No model save path entered. Exiting...")
		exit()

	if args.labeldict_path == None or args.labeldict_path == "predefined":
		if args.dataset_name == "u54codex":
			if args.patch_labeling == "inherit":
				label_dict = utils.labels_dict
			elif args.patch_labeling == "proxy":
				print("proxy labeling not yet implemented, exiting...")
				exit()
				# label_dict = None # use the discretizer function
	else:
		label_dict = deserialize(args.labeldict_path)
		# print(label_dict)

	setattr(args, "label_dict", label_dict)

	if isinstance(args.hyperparameters, float):
		alpha = args.hyperparameters
	elif type(args.hyperparameters) == list:
		eta1 = 0.05
		eta2 = 0.05 # not used for now

	patch_list = deserialize(args.patchlist_path)

	if args.patch_loss == "bce":
		num_classes = 2

	# get training set size
	print("\nBEGINNING TRAINING OF MODEL:", args.model_class + "\n" + "="*60)

	print("We get to train on...\n" + "-"*60) # could be augmented, could not be
	print("train set size (#patches):", len(patch_list))
	print("of patch size:", args.patch_size)
	# print("\nComposition of patients in sets...\n" + "-"*45)
	print("train set unique images:", len(label_dict))

	# should implement later
	# print("\n(+/-) splits in sets...\n" + "-"*30)
	# print("train set split:", set_splits(utils.train_dir))

	del patch_list # can deserialize in other functions

	# SET-UP
	#-------
	if USE_GPU and torch.cuda.is_available():
		device = torch.device('cuda')
		print("\nNote: gpu available!")
	else:
		device = torch.device('cpu')
		print("\nNote: gpu NOT available!")

	# MODEL CHOICE + TRAINING
	#------------------------
	if args.model_class.startswith("VGG"):
		if args.model_class == "VGG19":
			model = VGG19(bn_flag=False).arch
		elif args.model_class == "VGG19_bn":
			model = VGG19(bn_flag=True).arch
		elif args.model_class == "VGG_att":
			model = AttnVGG_before(args.channel_dim, args.patch_size, num_classes)
		else:
			print("Enter a correctly specified ModVGG model: VGG19/VGG19_bn/VGG_att")
			exit()
		
		setattr(args, "string_details", args.description) # more complete
		# setattr(args, "string_details", args.model_class + "-" + args.dataset_name + "-" + str(args.patch_size) + "-" + args.patch_loading + "-" + args.patch_labeling)
		
		if args.patch_labeling == "inherit":
			optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
		elif args.patch_labeling == "proxy":
			optimizer = optim.RMSprop(model.parameters(), lr=LEARN_RATE) #or GD -- only for future models

		loss_history = train_VGGs(model, device, optimizer, args, save_embeds_flag=False) # flag used to be true, but skipping for now

	else:
		print("Enter a correctly specified ModVGG model: VGG19/VGG19_bn")
		exit()

	# cache the losses
	serialize(loss_history, args.model_path + "/" + args.string_details + "_trainloss.obj")
	fig = plt.plot(loss_history, c="blue", label="train")
	plt.savefig(args.model_path + "/"  + args.string_details + "_trainloss.png", bbox_inches="tight")


if __name__ == "__main__":
	main()