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
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

import torchvision
import torchvision.models
from torchvision import transforms as trn

import wandb

# personal imports
from dataloader import DataLoaderCustom
import utils
from utils import labels_dict, count_files, unique_files, set_splits
from utils import train_dir, val_dir, test_dir
from utils import serialize, deserialize, str2bool
from models import VGG19, VGGEmbedder, AttnVGG_before, vgg19, vgg19_bn, MultiTaskLoss, ElasticLinear


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
taskcombo_flag = "uncertainty"
graph_flag = True
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


def pool_embeds(embeds, pool_flag="max"):
	cats = torch.stack(embeds, dim=0)
	if pool_flag == "max":
		pooled = torch.max(cats, dim=0)[0]
	elif pool_flag == "mean":
		pooled = torch.mean(cats, dim=0)[0]
	return pooled


def check_mb_accuracy(scores, ys):
	_, preds = scores.max(1)
	probs = F.softmax(scores, dim=1) 
	# used to slice like this: [:,1] 
	# but let's instead take both
	
	# print(probs)
	# print(ys)
	# pdb.set_trace()

	num_correct = (preds == ys).sum()
	num_samples = preds.size(0)
	return preds, probs, num_correct, num_samples


def forward_pass_clf_eval(x, y, model, model_class, device):
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


def forward_pass_tandem_eval(x, y, model, model_class, device, mode="patch"):
	
	if mode == "patch":
		return forward_pass_clf_eval(x, y, model, model_class, device)

	if mode == "image":
		x = torch.stack(x, dim=0)
		y = torch.from_numpy(np.array(y))
		x = x.to(device=device, dtype=dtype)  
		y = y.to(device=device, dtype=torch.long)
		x = x_im.to(device=device, dtype=dtype)  

		scores = model(x)
		val_loss = F.cross_entropy(scores, y.long())
		preds, probs, num_correct, num_samples = check_mb_accuracy(scores, y)
		return val_loss.item(), num_correct, num_samples, probs.tolist(), preds.tolist(), y.tolist()



def check_patch_accuracy(model, args, batch_flag=False):
	
	num_correct, num_samples, cum_loss = 0, 0, 0
	losses, probs, preds, patch_names, labels = [], [], [], [], []
	model.eval()  # set model to evaluation mode

	if args.model_class.startswith("VGG") == True:
		loader = DataLoaderCustom(args)
	else:
		print("choose valid model type! See help docstring!")
		exit()

	if "VGG" in args.model_class:
		with torch.no_grad():
			for i, (fxy, x, y) in enumerate(loader):
				# run batches
				loss, correct, samples, probs_batch, preds_batch, labels_batch = forward_pass_clf_eval(x, y, model, args.model_class, args.device)
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

	elif args.gamified_flag == True:
		num_correct_img, num_samples_img, cum_loss_img = 0, 0, 0
		losses_img, probs_img, preds_img, img_names, labels_img = [], [], [], [], []
		model_t.eval() 

		with torch.no_grad():
			embed_dict = defaultdict(list)

			for i, (fxy, x, y) in enumerate(loader):
				loss, correct, samples, probs_batch, preds_batch, labels_batch = forward_pass_tandem_eval(x, y, model, device, mode="patch")
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

				x = torch.from_numpy(x)
				x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
				embed_dict = store_embeds(embed_dict, fxy, x, model)

			acc = float(num_correct) / num_samples
			print('Got %d / %d correct (%.2f) for patch' % (num_correct, num_samples, 100 * acc))
			print('Cumulative Loss scaled by iter: {0:0.4f}'.format(cum_loss / i))

	return num_correct, num_samples, cum_loss, losses, probs, preds, patch_names, labels


# Training routines
#------------------

def store_embeds(embed_dict, fxy, x, trained_model, args, att_flag=False):
	"""
	Stores embeddings from a backbone model
	Inputs:
	- embed_dict: already created dictionary to write to
	- fxy: filename
	- x: hidden vector input into shallow learner
	- trained_model: pytorch model of weak embedder
	"""
	if args.backprop_level != "full":
		x = x.detach().clone() 
	
	if args.model_class.startswith("VGG"):
		embedder = VGGEmbedder(trained_model, args, att_flag)
	elif args.model_class == "ViT":
		embedder = ViTEmbedder()

	# register hooks
	if args.backprop_level != "none":
		activation = {}
		def getActivation(name):
		  # the hook signature
		  def hook(model, input, output):
		    activation[name] = output.detach()
		  return hook

		h_linear = embedder.embedder.classifier[0].register_forward_hook(getActivation('linear'))

	# pdb.set_trace()

	z = embedder(x)
	if args.backprop_level != "none":
		h_linear.remove()
		z = activation["linear"]
	if args.backprop_level != "full":
		z = z.detach().clone()
 	# ^ from: https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second-time-but

	if args.pool_type == "max":
		for i, f in enumerate(fxy):
			if args.dataset_name == "u54codex":
				im_folder = f.split(".")[0]  # "reg{idx}_montage"
				im_id = im_folder.split("_")[0]  # "reg{idx}""
				idx = im_id.split("reg")[1]  # {idx}
			elif args.dataset_name == "cam":
				pieces = f.split("_")
				idx = pieces[0] + "_" + pieces[1]
			
			embed_dict[idx].append(z[0, :])
			z = z[1:, :] # delete 0th row to match with f
			embed_dict[idx] = [pool_embeds(embed_dict[idx], args.pool_type)]
	else:
		print("Pooling type specified is not yet implemented!")
		exit()

	gc.collect()
	if args.backprop_level != "full":
		torch.cuda.empty_cache()

	return embed_dict


def train_tandem(model_patch, device, optimizer, args, save_embeds_flag=True):
	"""
	Trains the gamified learning runs
	Inputs:
	- model_patch: A PyTorch Module of the model to train.
	- optimizer: An Optimizer object we will use to train the model (image-level)
	- epochs: (Optional) A Python integer giving the number of epochs to train for
	Returns: Nothing, but prints model accuracies during training.
	"""

	# Logging with Weights & Biases
	#-------------------------------
	if args.backprop_level == "none":
		game_type = "r" # regularizer
	elif args.backprop_level == "blindfolded":
		game_type = "b"
	elif args.backprop_level == "full":
		game_type = "f"
	else:
		print("backprop_level not supported")

	experiment = game_type + "SGN-" + args.model_class + "-" + args.dataset_name
	wandb.init(project=experiment, entity="gamified-learning")
	wandb.config = {
	  "learning_rate": LEARN_RATE,
	  "epochs": args.num_epochs,
	  "batch_size": args.batch_size
	}

	# set-up
	#--------
	train_losses_patch, train_losses_img = [], []
	train_losses = []
	graph_flag = args.backprop_level
	game_descriptors = "gamify-" + taskcombo_flag + "-backprop" + str(args.backprop_level) + "-" + args.pool_type + "_pooling-"
	
	# override optimizer to make sure MTL loss is loaded in
	mtl = MultiTaskLoss(model=model_patch, eta=[2.0, 1.0], combo_flag=taskcombo_flag)
	optimizer = optim.RMSprop(mtl.parameters(), lr=LEARN_RATE)

	# gather model info
	#-------------------
	# sample size for meta-scale
	sample_size = len(args.label_dict)
	
	# hidden size
	train_loader = DataLoaderCustom(args)
	for t, (fxy, x, y) in enumerate(train_loader):
		print("Gathering hidden dimensions:")
		x = torch.from_numpy(x)
		break

	if args.model_class.startswith("VGG"):
		if args.model_class in ["VGG19", "VGG19_bn"]:
			att_flag = False
		elif args.model_class == "VGG_att":
			att_flag = True
		embedder = VGGEmbedder(model_patch, args, att_flag=att_flag)
		# if backprop is none, we get an actually embedder
		# else, we get the model itself back and need to apply hooks

	elif args.model_class == "ViT":
		 # embedder = ViTEmbedder()
		 print("ViT embedder not yet fully implemented")
		 exit()

	# gathering embeddings
	if args.backprop_level == "none":
		z = embedder(x.float())
	else: # register hooks instead
		activation = {}
		def getActivation(name):
		  # the hook signature
		  def hook(model, input, output):
		    activation[name] = output.detach()
		  return hook
		h_linear = embedder.embedder.classifier[0].register_forward_hook(getActivation('linear'))
		z = embedder(x.float()) # forward pass
		h_linear.remove()
		z = activation["linear"]

	hidden_size = z.shape[1]

	# define model_img: LASSO classifier
	#------------------------------------
	model_img = ElasticLinear(loss_fn=torch.nn.CrossEntropyLoss(), n_inputs=hidden_size, l1_lambda=0.05, l2_lambda=0.0, learning_rate=0.05)
	model_patch = model_patch.to(device=device) 
	model_img = model_img.to(device=device) 
	model_patch.train()  
	model_img.train()  
	print("Finishing model configuration! Onto training...")

	# Main training loop
	#--------------------
	for e in range(args.num_epochs):
		print("="*30 + "\n", "Beginning epoch", e, "\n" + "="*30)

		# IMAGE-LEVEL
		#=============
		print("img-level prediction!\n" + "-"*60)
		
		# random initialization of embeddings
		if e == 0: 
			max_epochs = 1			
			torch.random.manual_seed(444)
			x_im = torch.rand([sample_size, hidden_size])

			if args.dataset_name == "u54codex":
				ys_items = list(args.label_dict.items())
				ys = [int(yi[1][1]) for yi in ys_items if yi[1][0] == "train"]
			elif args.dataset_name == "cam":
				ys = list(args.label_dict.values())
			else:
				print("Error: configure dataset for gamified learning")
				exit() 
			y_im = torch.from_numpy(np.array(ys)).unsqueeze(1)

		# collecting embeddings in dictionary into arrays
		else:
			max_epochs = NUM_SUBEPOCH
			xs, ys = [], []
			for sample in embed_dict.keys():
				x_i = embed_dict[sample][0]
				y_i = args.label_dict[sample][1] #--> num
				xs.append(x_i)
				ys.append(y_i)

			x_im = torch.stack(xs, dim=0)
			y_im = torch.from_numpy(np.array(ys))
				
		# train with pytorch lightning
		trainer = pl.Trainer(max_epochs=max_epochs)	
		x_im = x_im.to(device=device, dtype=dtype) 
		y_im = y_im.to(device=device, dtype=torch.float32)
		dataset_train = TensorDataset(x_im, y_im)
		dataloader_train = DataLoader(dataset_train, batch_size=sample_size//10, shuffle=True)
		trainer.fit(model_img, dataloader_train)
		train_loss_img = trainer.logged_metrics['loss']
		print("Image-level training loss:", train_loss_img)
		train_losses_img.append(train_loss_img.item())

		# PATCH-LEVEL
		#=============
		print("-"*60 + "\n" + "entering patch predictions!\n" + "-"*60)
		embed_dict = defaultdict(list) 	# optional, create new embedding dict per epoch

		# added to initiate new seed/shuffle every epoch
		train_loader = DataLoaderCustom(args)

		for t, (fxy, x, y) in enumerate(train_loader):
			print("Patch minibatch #:", t)
			x = torch.from_numpy(x)
			y = torch.from_numpy(y)
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.long)

			if graph_flag != "none":
				scores, losses, train_loss = mtl(x, y, train_loss_img)
			else: 
				# backprop_level=none ==> just regularization term after transfer learning
				scores, losses, train_loss = mtl(x, y, train_loss_img.detach().clone())

			# should always retain graph at this point b/c all necessary detachment has happened by now
			optimizer.zero_grad()
			train_loss.backward(retain_graph=True)
			# if graph_flag != "none":
			# 	train_loss.backward(retain_graph=True)
			# else:
			#  	train_loss.backward(retain_graph=False)
			optimizer.step()

			if t % print_every == 0:
				print('Iteration %d, loss = %.4f' % (t, train_loss.item()))
				preds, probs, num_correct, num_samples = check_mb_accuracy(scores, y)
				acc = float(num_correct) / num_samples
				print('minibatch training accuracy: %.4f' % (acc * 100))
				print("Uncertainty loss parameters (weights for losses):", mtl.eta)

			train_losses_patch.append(train_loss.item())
			train_losses.append(train_loss.item())

			# Store embeds
			embed_dict = store_embeds(embed_dict, fxy, x, model_patch, args)

			# save embeddings every 4 epochs so we can visualize 
			if save_embeds_flag == True and ((e+1) % 4 == 0):
				serialize(embed_dict, args.cache_path + "/" + game_descriptors + args.string_details + "-epoch" + str(e) + "-embeddings_train.obj")
		
		# save model per epoch
		torch.save(model_patch, args.model_path + "/" + game_descriptors + args.string_details + "_EMBEDDER_epoch%s.pt" % e)
		torch.save(model_img, args.model_path + "/" + game_descriptors + args.string_details + "_SHALLOW_epoch%s.pt" % e)
		# Future: could check val acc every epoch

		# cache the losses every epoch
		serialize(train_losses_patch, args.model_path + "/" + args.string_details + "_trainlossPATCH.obj")
		fig = plt.plot(train_losses_patch, c="blue", label="Train loss for weak patch-level model")
		plt.savefig(args.model_path + "/"  + args.string_details + "_trainlossPATCH.png", bbox_inches="tight")

		serialize(train_losses_img, args.model_path + "/" + args.string_details + "_trainlossIMG.obj")
		fig = plt.plot(train_losses_img, c="blue", label="Train loss for shallow image-level model")
		plt.savefig(args.model_path + "/"  + args.string_details + "_trainlossIMG.png", bbox_inches="tight")

		# more logging
		wandb.log({"loss-image": train_loss_img,
				   "loss-patch": train_loss})
		# wandb.watch((model_img,model_patch,mtl))

	# full model save
	torch.save(model_patch, args.model_path + "/" + game_descriptors + args.string_details + "_EMBEDDER_full.pt")
	torch.save(model_img, args.model_path + "/" + game_descriptors + args.string_details + "_SHALLOW_full.pt")

	return train_losses, train_losses_img




def train_classifier(model, device, optimizer, args, save_embeds_flag=True):
	"""
	Train a model on image data using the PyTorch Module API.

	Inputs:
	- model: A PyTorch Module giving the model to train.
	- optimizer: An Optimizer object we will use to train the model
	- epochs: (Optional) A Python integer giving the number of epochs to train for

	Returns: Nothing, but prints model accuracies during training.
	"""
	if save_embeds_flag == True:
		if args.cache_path == None:
			print("Error: Missing cache path for embeddings. Please enter a path to save your cache.")

	train_losses = []
	model = model.to(device=device)

	att_flag = False
	if args.model_class == "VGG_att":
		att_flag = True
	print("Now training:", args.model_class)

	if args.model_class.startswith("VGG") == True:
		if args.model_class != "VGG_att": # this errors 
			# summary(model, input_size=(args.channel_dim, args.patch_size, args.patch_size)) # print model
			pass
	
	train_loader = DataLoaderCustom(args)
	model.train()

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
			elif args.model_class in ["VGG19", "VGG19_bn"]:
				scores = model(x)

			train_loss = F.cross_entropy(scores, y)
			optimizer.zero_grad()
			train_loss.backward()
			optimizer.step()

			if t % print_every == 0:
				print('Iteration %d, train loss = %.4f' % (t + print_every, train_loss.item()))
				preds, probs, num_correct, num_samples = check_mb_accuracy(scores, y)
				acc = float(num_correct) / num_samples
				print('minibatch training accuracy: %.4f' % (acc * 100))

			train_losses.append(train_loss.item())
			gc.collect()

			# save embeddings every 4 epochs for standard classifiers
			if save_embeds_flag == True and ((e+1) % 4 == 0):
				embed_dict = store_embeds(embed_dict, fxy, x, model, att_flag)
				serialize(embed_dict, args.cache_path + "/" + args.string_details + "-epoch" + str(e) + "-embeddings_train.obj")
				
		# save model per epoch
		torch.save(model, args.model_path + "/" + args.string_details + "_epoch%s.pt" % e)
		# Future: check val acc every epoch

		# cache the losses every epoch
		serialize(train_losses, args.model_path + "/" + args.string_details + "_trainloss.obj")
		fig = plt.plot(train_losses, c="blue", label="train loss")
		plt.savefig(args.model_path + "/"  + args.string_details + "_trainloss.png", bbox_inches="tight")

	# full model save
	torch.save(model, args.model_path + "/" + args.string_details + "_full.pt")

	return train_losses



def main():

	# ARGPARSE
	#==========
	parser = argparse.ArgumentParser()
	parser.add_argument('--description', default="no-description", type=str, help='Description of your experiement, with no spaces. E.g. VGG19_bn-random_loading-label_inherit-bce_loss-on_MFL-1')
	parser.add_argument('--model_class', default=None, type=str, help='Select one of: VGG19/VGG19_bn/VGG_att.')
	parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs to train for. Default is 10.')
	parser.add_argument('--hyperparameters', default=0.01, type=float, help="Denotes hyperparameters for custom multi-task losses. Default value is alpha=0.01, where a higer value indicates more focus on img-level predictions")
	parser.add_argument('--batch_size', default=36, type=int, help="Batch size. dDfault is 36.")
	parser.add_argument('--channel_dim', default=1, type=int, help="Channel dimension. Default is 1.")
	parser.add_argument('--normalize_flag', default=False, type=str2bool, help="T/F if patches need normalization. Default is False.")
	parser.add_argument('--dataset_name', default=None, type=str, help="What you want to name your dataset. For pre-defined label dictionaries, use: u54codex to search utils.")
	parser.add_argument('--dataloader_type', default="stored", type=str, help="Type of data loader: stored vs otf (on-the-fly).")
	
	# gamified learning specific args
	parser.add_argument('--save_embeds_flag', default=False, type=str2bool, help="T/F if you want to save embeddings every 4 epochs. Defaults to F.")
	parser.add_argument('--gamified_flag', default=False, type=str2bool, help="T/F if you are running gamified learning with the model_class specified and a shallow learner. Defaults to F.")
	parser.add_argument('--backprop_level', default="blindfold", type=str, help="Level of cross-model learning in gamified learning setup. Options are none, blindfolded, full. Defaults to blindfold. Only relevant if gamified_flag = True.")
	parser.add_argument('--pool_type', default="max", type=str, help="Type of pooling for gamified learning. Defaults to max. Only relevant if gamified_flag = True.")

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
	parser.add_argument('--cache_path', default=None, type=str, help="Where you'd like to save the model outputs.")

	args = parser.parse_args()

	# ERROR CHECKING
	#================
	# check viable model class
	if args.model_class == None:
		print("No model entered. Please choose a model using the parser help flag. Exiting...")
		exit()

	supported_models = ["VGG19", "VGG19_bn", "VGG_att", "ResNet50", "ViT", "SwinT", "FlashViT", "FlashSwinT"]
	supported_str = ', '.join(supported_models)
	if args.model_class not in supported_models:
		print("Error: Unsupported model class for backbone classifier. Please choose one of:", supported_str)
		exit()

	# check on data, model, and label paths
	if args.data_path == None:
		print("No data path path entered. Exiting...")
		exit()

	if args.model_path == None:
		print("No model save path entered. Exiting...")
		exit()

	if args.labeldict_path == None or args.labeldict_path == "predefined":
		if args.dataset_name == "u54codex":
			if args.patch_labeling == "inherit":
				label_dict = utils.labels_dict
			elif args.patch_labeling == "proxy":
				label_dict = None # use the discretizer function
	else:
		print(args.labeldict_path)
		label_dict = deserialize(args.labeldict_path)

	setattr(args, "label_dict", label_dict)

	# PRINTS
	#========
	print("\nBEGINNING TRAINING MODEL:", args.model_class + "\n" + "="*60)
	print("We get to train on...\n" + "-"*60)
	if args.patchlist_path:
		patch_list = deserialize(args.patchlist_path)
		print("train set size (#patches):", len(patch_list))
		del patch_list # we can deserialize in other functions
	
	print("of patch size:", args.patch_size)
	print("train set unique images:", len(label_dict))

	# SET-UP
	#========
	if USE_GPU and torch.cuda.is_available():
		device = torch.device('cuda')
		print("\nNote: gpu available!")
	else:
		device = torch.device('cpu')
		print("\nNote: gpu NOT available!")

	# define hyperparameters, etc.
	if isinstance(args.hyperparameters, float):
		alpha = args.hyperparameters
	elif type(args.hyperparameters) == list:
		eta1 = 0.05
		eta2 = 0.05 # not used for now

	if args.patch_loss == "bce":
		num_classes = 2

	setattr(args, "string_details", args.description)
	
	# MODEL INSTANTIATION 
	#=====================
	if args.model_class == "ViT":
		model = torchvision.models.vit_b_16()

	elif args.model_class.startswith("VGG"):
		if args.model_class == "VGG19":
			if args.patch_size == 96:
				model = VGG19(bn_flag=False).arch
			elif args.patch_size == 224:
				model = vgg19(pretrained=False)
		elif args.model_class == "VGG19_bn":
			if args.patch_size == 96:
				model = VGG19(bn_flag=True).arch
			elif args.patch_size == 224:
				model = vgg19_bn(pretrained=False, in_channels=args.channel_dim)
				# model = torchvision.models.vgg19_bn(weights=None) 


		elif args.model_class == "VGG_att":
			model = AttnVGG_before(args.channel_dim, args.patch_size, num_classes)


	# OPTIMIZER INSTANTIATION
	#=========================
	if args.patch_labeling == "inherit" or args.patch_labeling == "seg":
		optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
	
	elif args.patch_labeling == "proxy":
		optimizer = optim.RMSprop(model.parameters(), lr=LEARN_RATE) 
		#or GD -- only for future models

	# TRAINING ROUTINE
	#==================
	if args.model_class in supported_models and args.gamified_flag == False:
		loss_history = train_classifier(model, device, optimizer, args) 
	
	elif args.model_class in supported_models and args.gamified_flag == True: 
		print("="*60 + "\nInitiating backbone architecture for Gamified Learning!" + "\n" + "="*60)
		if args.pool_type != "max":
			print("Error: mean-pool and other pooling not efficiently implemented yet. Exiting...")
			exit()	
		loss_history, loss_history_img = train_tandem(model, device, optimizer, args, save_embeds_flag=True)
	
	else:
		print("Error: Please choose a valid model to train")


if __name__ == "__main__":
	main()

