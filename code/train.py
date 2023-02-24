import numpy as np
import os 
import pdb 
import argparse
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import gc
# from time import time

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
from torch.autograd import Variable

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torchvision
import torchvision.models
from torchvision import transforms as trn

import wandb

# personal imports
from dataloader import DataLoaderCustom, create_triplet_dataloader
import utils
from utils import labels_dict, count_files, unique_files, set_splits
from utils import train_dir, val_dir, test_dir
from utils import serialize, deserialize, str2bool
from models import ResNet18, VGG19, VGGEmbedder, AttnVGG_before, vgg19, vgg19_bn, MultiTaskLoss, ElasticLinear


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

				if (i+1) % 50 == 0:
					print("On minibatch", i)
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
			print('In an image-level labeling sense: got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
			print('Cumulative Loss scaled by iteration: {0:0.4f}'.format(cum_loss / i))

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


def get_label(dataset_name, fname, label_dict):
	"""
	Grab label for associated file "fname"
		fname: string for filename being queried
	Note: See similar function to custom dataloader method
	"""
	if dataset_name == "cam":
		if "patient" in fname: #validation
			pieces = fname.split("_")
			reg_id = pieces[0] + "_" + pieces[1] + "_" + pieces[2] + "_" + pieces[3] + ".tif"
		else: # train or test
			pieces = fname.split("_")
			reg_id = pieces[0] + "_" + pieces[1] 
	else: # u54codex
		reg_id = fname.split("_")[0]

	label = label_dict[reg_id]
	return label



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
	if args.model_to_load is None:
		train_losses_patch, train_losses_img = [], []
		train_losses = []
	else:
		train_losses_patch = utils.deserialize(args.model_path + "/" + args.string_details + "_trainlossPATCH.obj")
		train_losses_img = utils.deserialize(args.model_path + "/" + args.string_details + "_trainlossIMG.obj")
		train_losses = utils.deserialize(args.model_path + "/" + args.string_details + "_trainlossPATCH.obj") # same as train loss

	graph_flag = args.backprop_level
	game_descriptors = "gamify-" + taskcombo_flag + "-backprop" + str(args.backprop_level) + "-" + args.pool_type + "_pooling-"
	
	# override optimizer to make sure MTL loss is loaded in
	if args.mtl_to_load is None:
		mtl = MultiTaskLoss(model=model_patch, eta=[2.0, 1.0], combo_flag=taskcombo_flag)
	else:
		mtl = torch.load(args.mtl_to_load, map_location=device)
	
	model_patch = model_patch.to(device=device) 
	mtl = mtl.to(device=device) 
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
		x = x.to(device=device, dtype=dtype) 
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
	
	embedder = embedder.to(device=device) 

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
	if args.model2_to_load is None:
		model_img = ElasticLinear(loss_fn=torch.nn.CrossEntropyLoss(), n_inputs=hidden_size, l1_lambda=0.05, l2_lambda=0.0, learning_rate=0.05)
	else:
		print("loading previous shallow model")
		model_img = torch.load(args.model2_to_load, map_location=device)
	model_img = model_img.to(device=device) 

	if args.embeds_to_load is not None:
		print("loading previous embeddings... assuming GPU access")
		# embed_dict = utils.deserialize(args.embeds_to_load)
		embed_dict = torch.load(args.embeds_to_load, map_location=device)

	print("Finishing model configuration! Onto training...")	
	model_patch.train()  
	model_img.train()  
	mtl.train()  

	# Main training loop
	#--------------------
	# for e in range(args.num_epochs):
	for e in range(0 + args.prev_epoch, args.num_epochs + args.prev_epoch):
		print("="*30 + "\n", "Beginning epoch", e, "\n" + "="*30)

		# IMAGE-LEVEL
		#=============
		print("img-level prediction!\n" + "-"*60)
		
		# random initialization of embeddings
		if e == 0 or args.embeds_to_load is None: 
			print("using random embeddings since previous embeddings not found")
			max_epochs = 3 # something small			
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
				if args.dataset_name == "cam":
					y_i = get_label(args.dataset_name, sample, args.label_dict)
				elif args.dataset_name == "u54codex":
					y_i = args.label_dict[sample][1] 
				xs.append(x_i)
				ys.append(y_i)

			x_im = torch.stack(xs, dim=0)
			y_im = torch.from_numpy(np.array(ys))
				
		# train with pytorch lightning
		early_stop_callback = EarlyStopping(monitor="loss", mode="min")
		trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[early_stop_callback])	
		x_im = x_im.to(device=device, dtype=dtype) 
		y_im = y_im.to(device=device, dtype=torch.float32)
		dataset_train = TensorDataset(x_im, y_im)
		dataloader_train = DataLoader(dataset_train, batch_size=sample_size//10, shuffle=True)
		trainer.fit(model_img, dataloader_train)
		train_loss_img = trainer.logged_metrics['loss']
		if e > 0 and args.embeds_to_load is None:
			train_loss_img = train_loss_img.to(device=device) 

		print("Image-level training loss:", train_loss_img)
		train_losses_img.append(train_loss_img.item())

		# PATCH-LEVEL
		#=============
		print("-"*60 + "\n" + "entering patch predictions!\n" + "-"*60)
		embed_dict = defaultdict(list) 	# optional, create new embedding dict per epoch

		# added to initiate new seed/shuffle every epoch
		train_loader = DataLoaderCustom(args)

		for t, (fxy, x, y) in enumerate(train_loader):
			# print("Patch minibatch #:", t)
			x = torch.from_numpy(x)
			y = torch.from_numpy(y)
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.long)

			if graph_flag != "none":
				# pdb.set_trace()
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
				# more logging
				wandb.log({"loss-patch": train_loss})

			train_losses_patch.append(train_loss.item())
			train_losses.append(train_loss.item())

			# Store embeds
			embed_dict = store_embeds(embed_dict, fxy, x, model_patch, args, att_flag)

			# save embeddings every 4 epochs so we can visualize 
			# serialize(embed_dict, args.cache_path + "/" + game_descriptors + args.string_details + "-curr_embeddings_train.obj")
			torch.save(embed_dict, args.cache_path + "/" + game_descriptors + args.string_details + "-curr_embeddings_train.obj")
			if save_embeds_flag == True and ((e+1) % 4 == 0):
				# serialize(embed_dict, args.cache_path + "/" + game_descriptors + args.string_details + "-epoch" + str(e) + "-embeddings_train.obj")
				torch.save(embed_dict, args.cache_path + "/" + game_descriptors + args.string_details + "-epoch" + str(e) + "-embeddings_train.obj")

		# save models per epoch
		torch.save(model_patch, args.model_path + "/" + game_descriptors + args.string_details + "_EMBEDDER_epoch%s.pt" % e)
		torch.save(model_img, args.model_path + "/" + game_descriptors + args.string_details + "_SHALLOW_epoch%s.pt" % e)
		torch.save(mtl, args.model_path + "/" + game_descriptors + args.string_details + "_MTL_epoch%s.pt" % e)

		# Future: could check val acc every epoch

		# cache the losses every epoch
		serialize(train_losses_patch, args.model_path + "/" + args.string_details + "_trainlossPATCH.obj")
		fig = plt.plot(train_losses_patch, c="blue", label="Train loss for weak patch-level model")
		plt.savefig(args.model_path + "/"  + args.string_details + "_trainlossPATCH.png", bbox_inches="tight")

		serialize(train_losses_img, args.model_path + "/" + args.string_details + "_trainlossIMG.obj")
		fig = plt.plot(train_losses_img, c="blue", label="Train loss for shallow image-level model")
		plt.savefig(args.model_path + "/"  + args.string_details + "_trainlossIMG.png", bbox_inches="tight")

		# more logging
		wandb.log({"end-of-epoch loss-image": train_loss_img,
				   "end-of-epoch loss-patch": train_loss})
		# wandb.watch((model_img,model_patch,mtl))		

	# full model save
	torch.save(model_patch, args.model_path + "/" + game_descriptors + args.string_details + "_EMBEDDER_full.pt")
	torch.save(model_img, args.model_path + "/" + game_descriptors + args.string_details + "_SHALLOW_full.pt")
	torch.save(mtl, args.model_path + "/" + game_descriptors + args.string_details + "_MTL_full.pt")

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

	# Logging with Weights & Biases
	#-------------------------------
	experiment = "PatchCNN-" + args.model_class + "-" + args.dataset_name
	wandb.init(project=experiment, entity="gamified-learning")
	wandb.config = {
	  "learning_rate": LEARN_RATE,
	  "epochs": args.num_epochs,
	  "batch_size": args.batch_size
	}

	if args.model_to_load is None:
		train_losses = []
	else:
		train_losses = utils.deserialize(args.model_path + "/" + args.string_details + "_trainloss.obj") # same as train loss

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
	for e in range(0 + args.prev_epoch, args.num_epochs + args.prev_epoch):
		print("="*30 + "\n", "Beginning epoch", e, "\n" + "="*30)
	
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
				# more logging
				wandb.log({"loss": train_loss})

			train_losses.append(train_loss.item())
			gc.collect()

			embed_dict = store_embeds(embed_dict, fxy, x, model, args, att_flag)

			# save embeddings every 4 epochs for standard classifiers
			serialize(embed_dict, args.cache_path + "/" + args.string_details + "-curr_embeddings_train.obj")
			if save_embeds_flag == True and ((e+1) % 4 == 0):
				serialize(embed_dict, args.cache_path + "/" + args.string_details + "-epoch" + str(e) + "-embeddings_train.obj")
				
		# save model per epoch
		torch.save(model, args.model_path + "/" + args.string_details + "_epoch%s.pt" % e)
		# Future: check val acc every epoch

		# cache the losses every epoch
		serialize(train_losses, args.model_path + "/" + args.string_details + "_trainloss.obj")
		fig = plt.plot(train_losses, c="blue", label="train loss")
		plt.savefig(args.model_path + "/"  + args.string_details + "_trainloss.png", bbox_inches="tight")

		# more logging
		wandb.log({"end-of-epoch loss": train_loss})

	# full model save
	torch.save(model, args.model_path + "/" + args.string_details + "_full.pt")

	return train_losses

# adapted from tile2vec
def prep_triplets(triplets, cuda, triplet_type="dict", dtype=torch.float, rescale=True):
	"""
	Takes a batch of triplets and converts them into Pytorch variables 
	and puts them on GPU if available.
	"""
	if triplet_type == "tuple":
		a, n, d = torch.from_numpy(triplets[:,0,:,:,:]), torch.from_numpy(triplets[:,1,:,:,:]), torch.from_numpy(triplets[:,2,:,:,:])
		a, n, d = (Variable(a), Variable(n), Variable(d))
		if cuda == torch.device('cuda'):
			a, n, d = (a.cuda(), n.cuda(), d.cuda())
		a, n, d = (a.to(device=cuda, dtype=dtype), n.to(device=cuda, dtype=dtype), d.to(device=cuda, dtype=dtype))  # move to device, e.g. GPU
	elif triplet_type == "dict":
		a, n, d = (Variable(triplets['anchor']), Variable(triplets['neighbor']), Variable(triplets['distant']))
		if cuda == torch.device('cuda'):
			a, n, d = (a.cuda(), n.cuda(), d.cuda())
		a, n, d = (a.to(device=cuda, dtype=dtype), n.to(device=cuda, dtype=dtype), d.to(device=cuda, dtype=dtype))  # move to device, e.g. GPU
	maxi = 255.0
	if rescale == True:
		(a, n, d) = (a / maxi, n / maxi, d / maxi)
	return (a, n, d)


def train_selfsup(model, device, optimizer, args, margin=10, l2=0.01, print_every=1000, t0=None, summary_stats_flag=False):
	"""
	Trains a model for E epochs using the provided dataloader.
	"""
	# if t0 is None:
	# 	t0 = time.time()

	# Logging with Weights & Biases
	os.environ["WANDB_MODE"] = "online"
	experiment = "SelfSup-" + args.model_class + "-" + args.dataset_name
	if args.overfit_flag == True:
		experiment = "OVERFIT-" + experiment
	wandb.init(project=experiment, entity="selfsup-longrange")
	wandb.config = {
	  "learning_rate": args.learn_rate,
	  "epochs": args.num_epochs,
	  "batch_size": args.batch_size
	}
	print("hyperparams:\n" + "="*30)
	print("Adam optimizer learn rate:", args.learn_rate)
	print("margin:", margin)
	print("l2 regularization (lambda):", l2)
	print("Starting training procedure shortly...\n")

	# if args.model_to_load is None:
	# 	train_losses = []
	# else:
	# 	train_losses = utils.deserialize(args.model_path + "/" + args.string_details + "_trainloss.obj") # same as train loss
	
	# scheduler:
	# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

	# step size used to be 5, gamma was 0.1

	# load dataloader
	# train_loader = DataLoaderCustom(args)
	# n_train = len(train_loader.files) # number patch triplets
	# n_batches = n_train // args.batch_size 
	# triplet_type = "tuple"

	# data class to load
	if args.selfsup_mode == "0":
		print("Detecting 0-class training only...")
		train_loader = create_triplet_dataloader(args, args.data_path, args.trip0_path, shuffle=True)
		train_loaders = [train_loader]
	elif args.selfsup_mode == "01":
		print("Detecting sequential 0- and 1-class training...")
		train_loader = create_triplet_dataloader(args, args.data_path, args.trip0_path, shuffle=True)
		train_loader_alt = create_triplet_dataloader(args, args.data_path, args.trip1_path, shuffle=True)
		n_train, n_batches = len(train_loader_alt.dataset), len(train_loader_alt)
		print("class-1 stats: # triplets, # batches:", n_train, n_batches)
		train_loaders = [train_loader, train_loader_alt]
	elif args.selfsup_mode == "mix":
		print("Detecting mixed 0/1 training...")
		train_loader = create_triplet_dataloader(args, args.data_path, [args.trip0_path, args.trip1_path], shuffle=True)
		train_loaders = [train_loader]
		print("below is all mixed patches...")
	
	n_train, n_batches = len(train_loader.dataset), len(train_loader)
	print("class-0 stats: # triplets, # batches:", n_train, n_batches)

	# run only once to get mean and std
	# adapted from ptrblck: https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/2
	if summary_stats_flag == True:
		print("\nBeginning Mean/Std calc")
		sample_size = 1000 # number of triplet batches
		mean = torch.zeros((3))
		std = torch.zeros((3))
		for idx, (sample, _) in enumerate(train_loader):
			a, n, d = (sample['anchor'], sample['neighbor'], sample['distant'])
			batch_samples = a.size(0) # batch size (the last batch can have smaller size!)
			for token in [a,n,d]:
				token = token.view(batch_samples, a.size(1), -1)
				mean += token.mean(2).sum(0)
				std += token.std(2).sum(0)
			if idx >= sample_size:
				print(a.shape)
				print(token.shape)
				print(mean.shape)
				print(idx)
				break
		mean /= (idx * batch_samples * 3)
		std /= (idx * batch_samples * 3)
		print("Approximated dataset mean and std (per channel):", mean, std)

	model = model.to(device=device)
	model.train()

	# get a previous run backup
	if args.prev_epoch > 0:
		print("Cacheing previous state dict with prefix: PREVRUN")
		torch.save({
            'epoch': args.prev_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': None,
            }, args.model_path + "/PREVRUN-" + args.string_details + ".sd")

	print("\nBeginning training!")
	for e in range(0 + args.prev_epoch+1, args.num_epochs + args.prev_epoch+1):
		sum_loss, sum_l_n, sum_l_d, sum_l_nd = (0, 0, 0, 0)
		print_sum_loss = 0
		mid_save_flag = False
		# train over all train_loaders: e.g. 0 and 1 class
		for loader_idx, tl in enumerate(train_loaders):
			print("On training dataloader #:", loader_idx)
			for idx, (triplets, labels) in enumerate(tl):
			# for idx, (fxy, triplets, labels) in enumerate(train_loader):
				# can try overfitting first
				# if (idx + 1) * args.batch_size > 100:
				# 	print("Breaking...attempting to overfit on toy subset")
				# 	break
				# idx = triplet index
				verbosity = False
				if idx == 0:
					verbosity = True

				p, n, d = prep_triplets(triplets, device, triplet_type="dict")
				if idx == 0:
					print("tensor input:", p.shape)
					# print("preview of input values:", p)

				optimizer.zero_grad()
				loss, l_n, l_d, l_nd = model.loss(p, n, d, margin=margin, l2=l2, verbose=verbosity)
				loss.backward()
				optimizer.step()

				loss_to_store = loss.item()
				sum_loss += loss_to_store
				# train_losses.append(loss_to_store)
				sum_l_n += l_n.item()
				sum_l_d += l_d.item()
				sum_l_nd += l_nd.item()
				
				if ((idx + 1) * args.batch_size) % print_every == 0:
					# print("minibatch loss:", loss_to_store)
					print_avg_loss = (sum_loss - print_sum_loss) / (print_every / args.batch_size)
					print('Epoch {}: [{}/{} ({:0.0f}%)], Avg loss: {:0.4f}'.format(e, (idx + 1) * args.batch_size, n_train, 100 * (idx + 1) / n_batches, print_avg_loss))
					print_sum_loss = sum_loss

				# save state every few iterations
				if (mid_save_flag == False) and (((idx + 1) * args.batch_size / n_train) > 0.5):
					print("----saving model state mid-epoch----")
					torch.save({
						'epoch': e,
						'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),
						'loss': loss,
						}, args.model_path + "/" + args.string_details + ".sd")
					# always keep a backup
					torch.save({
						'epoch': e,
						'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),
						'loss': loss,
						}, args.model_path + "/BACKUP-" + args.string_details + ".sd")
					mid_save_flag = True

				scheduler.step()
				
		avg_loss = sum_loss / n_batches #(idx + 1) #n_batches
		avg_l_n = sum_l_n / n_batches #(idx + 1) #n_batches
		avg_l_d = sum_l_d / n_batches #(idx + 1) #n_batches
		avg_l_nd = sum_l_nd / n_batches #(idx + 1) #n_batches

		print('Finished epoch', e)
		print('  Average loss: {:0.4f}'.format(avg_loss))
		print('  Average l_n: {:0.4f}'.format(avg_l_n))
		print('  Average l_d: {:0.4f}'.format(avg_l_d))
		print('  Average l_nd: {:0.4f}'.format(avg_l_nd))

		# save model per epoch
		print("saving model for epoch", e, "\n")
		torch.save(model, args.model_path + "/" + args.string_details + "_epoch%s.pt" % e)
		# serialize(train_losses, args.model_path + "/" + args.string_details + "_trainloss.obj")
		# fig = plt.plot(train_losses, c="blue", label="train loss")
		# plt.savefig(args.model_path + "/"  + args.string_details + "_trainloss.png", bbox_inches="tight")
		wandb.log({"end-of-epoch avg loss": avg_loss})
		torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, args.model_path + "/" + args.string_details + "_epoch%s.sd" % e)
		torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, args.model_path + "/" + args.string_details + ".sd")
		# always keep a backup
		torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, args.model_path + "/BACKUP-" + args.string_details + ".sd")

	# full model save
	torch.save(model, args.model_path + "/" + args.string_details + "_full.pt")
	return (avg_loss, avg_l_n, avg_l_d, avg_l_nd)


def train_carta(model, device, optimizer, args, margin=10, l2=0.01, print_every=1000, t0=None, summary_stats_flag=False):
	"""
	Trains a model for E epochs using the provided dataloader.
	"""
	# Logging with Weights & Biases
	os.environ["WANDB_MODE"] = "online"
	experiment = "CARTA-" + args.model_class + "-" + args.dataset_name
	if args.overfit_flag == True:
		experiment = "OVERFIT-" + experiment
	wandb.init(project=experiment, entity="selfsup-longrange")
	wandb.config = {
	  "learning_rate": args.learn_rate,
	  "epochs": args.num_epochs,
	  "batch_size": args.batch_size
	}
	print("hyperparams:\n" + "="*30)
	print("Adam optimizer learn rate:", args.learn_rate)
	print("margin:", margin)
	print("l2 regularization (lambda):", l2)
	print("Starting training procedure shortly...\n")

	# if args.model_to_load is None:
	# 	train_losses = []
	# else:
	# 	train_losses = utils.deserialize(args.model_path + "/" + args.string_details + "_trainloss.obj") # same as train loss
	
	# scheduler:
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

	# data class to load
	if args.selfsup_mode == "sextuplet":
		print("Detecting contrastive training over sextuplets...")
		train_loader = create_triplet_dataloader(args, args.data_path, [args.trip0_path, args.trip1_path], shuffle=True)
		train_loaders = [train_loader]
	else:
		print("please choose self supervision mode as 'sextuplet'")
		exit()
	
	n_train, n_batches = len(train_loader.dataset), len(train_loader)
	print("sextuplet stats: # triplets, # batches:", n_train, n_batches)

	#------------------------------
	print("\nBeginning training!")
	model = model.to(device=device)
	model.train()
	for e in range(0 + args.prev_epoch, args.num_epochs + args.prev_epoch):
		sum_loss, sum_l_n, sum_l_d, sum_l_nd, sum_l_cc = (0, 0, 0, 0, 0)
		print_sum_loss = 0

		# train over all train_loaders: e.g. 0 and 1 class
		for loader_idx, tl in enumerate(train_loaders):
			print("On training dataloader #:", loader_idx)
			for idx, (triplets0, triplets1, labels0, labels1) in enumerate(tl):
				verbosity = False
				if idx == 0:
					verbosity = True
				p0, n0, d0 = prep_triplets(triplets0, device, triplet_type="dict")
				p1, n1, d1 = prep_triplets(triplets1, device, triplet_type="dict")
				if idx == 0:
					print("tensor input:", p0.shape)

				optimizer.zero_grad()
				loss0, l_n0, l_d0, l_nd0 = model.loss(p0, n0, d0, margin=margin,   l2=l2,   verbose=verbosity)
				loss1, l_n1, l_d1, l_nd1 = model.loss(p1, n1, d1, margin=margin*2, l2=l2/2, verbose=verbosity)
				loss_cc = model.sextuplet_loss(p0, n0, d0, p1, n1, d1, margin=margin) # centroid contrast
				loss = loss0 + loss1 + loss_cc # full carta loss

				loss.backward()
				optimizer.step()
				loss_to_store = loss.item()
				sum_loss += loss_to_store
				# train_losses.append(loss_to_store)
				sum_l_n += l_n0.item() + l_n1.item()
				sum_l_d += l_d0.item() + l_d1.item()
				sum_l_nd += l_nd0.item() + l_nd1.item()
				sum_l_cc += loss_cc.item()

				if ((idx + 1) * args.batch_size) % print_every == 0:
					# print("minibatch loss:", loss_to_store)
					print_avg_loss = (sum_loss - print_sum_loss) / (print_every / args.batch_size)
					print('Epoch {}: [{}/{} ({:0.0f}%)], Avg loss: {:0.4f}'.format(e, (idx + 1) * args.batch_size, n_train, 100 * (idx + 1) / n_batches, print_avg_loss))
					print_sum_loss = sum_loss

				scheduler.step()
					
		avg_loss = sum_loss / n_batches #(idx + 1) #n_batches
		avg_l_n = sum_l_n / n_batches #(idx + 1) #n_batches
		avg_l_d = sum_l_d / n_batches #(idx + 1) #n_batches
		avg_l_nd = sum_l_nd / n_batches #(idx + 1) #n_batches

		print('Finished epoch', e)
		print('  Average loss: {:0.4f}'.format(avg_loss))
		print('  Average l_n: {:0.4f}'.format(avg_l_n))
		print('  Average l_d: {:0.4f}'.format(avg_l_d))
		print('  Average l_nd: {:0.4f}\n'.format(avg_l_nd))

		# save model per epoch
		print("saving model for epoch", e)
		# torch.save(model, args.model_path + "/CARTA-" + args.string_details + "_epoch%s.pt" % e)
		
		# serialize(train_losses, args.model_path + "/CARTA-" + args.string_details + "_trainloss.obj")
		# fig = plt.plot(train_losses, c="blue", label="train loss")
		# plt.savefig(args.model_path + "/CARTA-" + args.string_details + "_trainloss.png", bbox_inches="tight")
		wandb.log({"end-of-epoch avg loss": avg_loss})
		
		torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, args.model_path + "/CARTA-" + args.string_details + ".sd")

	# full model save
	torch.save(model, args.model_path + "/CARTA-" + args.string_details + "_full.pt")
	return (avg_loss, avg_l_n, avg_l_d, avg_l_nd)


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
	parser.add_argument('--toy_flag', default=True, type=str2bool, help="T/F for a smaller training dataset for rapid experimentation. Default is True.")
	parser.add_argument('--overfit_flag', default=False, type=str2bool, help="T/F for intentional overfitting. Run name is modified to reflect this. Default is False.")

	# warm start
	parser.add_argument('--model_to_load', default=None, type=str, required=False, help="If you would like to start re-training an already trained Torch model, add the path here. This corresponds to the deep model.")
	parser.add_argument('--model2_to_load', default=None, type=str, required=False, help="If you would like to start re-training an already trained Torch model, add the path here. This corresponds to the shallow model.")
	parser.add_argument('--mtl_to_load', default=None, type=str, required=False, help="If you would like to start re-training an already trained Torch model, add the path here. This corresponds to the learned Uncertainty loss function.")
	parser.add_argument('--embeds_to_load', default=None, type=str, help="If you would like to start re-training an already trained Torch model, add the path for embeddings here. This corresponds to the stored embeddings to use.")

	# gamified learning specific args
	parser.add_argument('--save_embeds_flag', default=False, type=str2bool, help="T/F if you want to save embeddings every 4 epochs. Defaults to F.")
	parser.add_argument('--gamified_flag', default=False, type=str2bool, help="T/F if you are running gamified learning with the model_class specified and a shallow learner. Defaults to F.")
	parser.add_argument('--backprop_level', default="blindfold", type=str, help="Level of cross-model learning in gamified learning setup. Options are none, blindfolded, full. Defaults to blindfold. Only relevant if gamified_flag = True.")
	parser.add_argument('--pool_type', default="max", type=str, help="Type of pooling for gamified learning. Defaults to max. Only relevant if gamified_flag = True.")

	# self-supervised training
	parser.add_argument('--selfsup_flag', default=False, type=str2bool, help="T/F if training is self-supervised. Defaults to F.")
	parser.add_argument('--trip0_path', default=None, type=str, help="Path to 0-class triplets.")
	parser.add_argument('--trip1_path', default=None, type=str, help="Path to 1-class triplets.")
	parser.add_argument('--selfsup_mode', default=None, type=str, help="options are: {0, 1, 01, mix, sextuplet}")
	parser.add_argument('--coaxial_flag', default=False, type=str2bool, help="T/F if training uses coaxial subspaces. Defaults to F.")

	# parameters for patches
	parser.add_argument('--patch_size', default=96, type=int, help="Patch/instance size. Default is 96.")
	parser.add_argument('--patch_loading', default="random", type=str, help="Patch loading scheme: random or blocked. Default is random.")
	parser.add_argument('--patch_labeling', default="inherit", type=str, help="Patch labeling function: inherit or proxy. Default is inhert.")
	parser.add_argument('--patch_loss', default="bce", type=str, help="Patch loss function. Default is bce. Future support for uncertainty.")
	
	# paths
	parser.add_argument('--data_path', default=None, type=str, help="Dataset path. If patches, will use stored data loader, if images, will use OTF data loader.")
	parser.add_argument('--patchlist_path', default=None, type=str, help="Patch list path. This is a cached result of the preprocess.py script.")
	parser.add_argument('--labeldict_path', default=None, type=str, help="Label dictionary path. This is a cached result of the preprocess.py script.")
	parser.add_argument('--model_path', default=None, type=str, help="Where you'd like to save the models.")
	parser.add_argument('--cache_path', default=None, type=str, help="Where you'd like to save the model outputs.")

	args = parser.parse_args()

	# ERROR CHECKING
	#================
	# check viable model class
	if args.model_class == None:
		print("No model entered. Please choose a model using the parser help flag. Exiting...")
		exit()

	supported_models = ["VGG19", "VGG19_bn", "VGG_att", "ResNet50", "ResNet18", "ViT", "SwinT", "FlashViT", "FlashSwinT"]
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
		print("train set size (#unique patches):", len(patch_list))
		del patch_list # we can deserialize in other functions
	
	print("of patch size:", args.patch_size)
	print("train set unique images:", len(label_dict))

	# SET-UP
	#========
	print("GPU detected?", torch.cuda.is_available())
	if USE_GPU and torch.cuda.is_available():
		device = torch.device('cuda')
		print("\nNote: gpu available & selected!")
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
	if args.model_to_load == None or args.model_to_load.endswith(".sd"):
		prev_epoch = 0 # no previous training
		print("Starting a fresh model to train!")
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

		elif args.model_class.startswith("ResNet18") and args.selfsup_flag == True:
			# model = ResSelfEmbedder(num_blocks=[2,2,2,2,2], in_channels=3, z_dim=512)
			model = ResNet18(n_classes=2, in_channels=3, z_dim=128, supervised=False, no_relu=False, loss_type='triplet', tile_size=224, activation='relu')
			# z_dim used to be 512
	else:
		print("Detected a previously trained model to continue training on! Initiating warm start from torch load...")
		model = torch.load(args.model_to_load, map_location=device)
		prev_epoch_temp = args.model_to_load.split("epoch")[1]
		prev_epoch = int(prev_epoch_temp.split(".pt")[0]) + 1
		print("previously trained on", prev_epoch, "epochs")

	# OPTIMIZER INSTANTIATION
	#=========================
	if args.patch_labeling == "inherit" or args.patch_labeling == "seg":
		optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
	elif args.patch_labeling == "proxy":
		optimizer = optim.RMSprop(model.parameters(), lr=LEARN_RATE) #or GD -- only for future models
	elif args.patch_labeling == "selfsup":
		if args.overfit_flag == True:
			lr = 1e-4 # small data
			setattr(args, "learn_rate", lr)
			optimizer = optim.Adam(model.parameters(), lr=args.learn_rate, betas=(0.5, 0.999)) # used to have manual betas: betas=(0.5, 0.999)
		else:
			lr = 1e-3 # bigger dataset ; we did 1e-3 for 3.5 epochs (0,1,2,half of 3), 5e-4 for (3,4), 1e-3 for (5,6) / 5e-4 for (5,6)
			setattr(args, "learn_rate", lr) 
			optimizer = optim.Adam(model.parameters(), lr=args.learn_rate, betas=(0.5, 0.999)) # used to have manual betas: betas=(0.5, 0.999)

	# load state dicts
	# https://pytorch.org/tutorials/beginner/saving_loading_models.html
	if isinstance(args.model_to_load, str) and args.model_to_load.endswith(".sd"):
		print("Detected a previously trained model to continue training on! Initiating warm start from state dict...")
		checkpoint = torch.load(args.model_to_load)
		model.load_state_dict(checkpoint['model_state_dict'])
		model.to(device)
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		prev_epoch = checkpoint['epoch']
		loss = checkpoint['loss']
		print("previous training loss:", loss)
		print("loading state dict for optimizer as well")

	setattr(args, "prev_epoch", prev_epoch)

	# TRAINING ROUTINE
	#==================
	if args.model_class not in supported_models:
		print("Error: Please choose a valid model to train")
		exit()

	if args.selfsup_flag == True:
		if args.selfsup_mode == "0":
			setattr(args, "triplet_list", args.trip0_path)
		elif args.selfsup_mode == "1":
			setattr(args, "triplet_list", args.trip1_path)
		elif args.selfsup_mode == "01":
			setattr(args, "triplet_list", [args.trip0_path, args.trip1_path])
		elif args.selfsup_mode == "mix" or args.selfsup_mode == "sextuplet":
			setattr(args, "triplet_list", [args.trip0_path, args.trip1_path])
		else:
			print("Error: please enter valid triplet selection.")
		if args.coaxial_flag == True:
			loss_history = train_carta(model, device, optimizer, args)
		else:
			loss_history = train_selfsup(model, device, optimizer, args)
		exit()

	if args.gamified_flag == False:
		loss_history = train_classifier(model, device, optimizer, args) 
	elif args.gamified_flag == True: 
		print("="*60 + "\nInitiating backbone architecture for Gamified Learning!" + "\n" + "="*60)
		if args.pool_type != "max":
			print("Error: mean-pool and other pooling not efficiently implemented yet. Exiting...")
			exit()	
		loss_history, loss_history_img = train_tandem(model, device, optimizer, args, save_embeds_flag=True)
	else:
		print("Error: Please choose a valid model to train")
		exit()

if __name__ == "__main__":
	main()

