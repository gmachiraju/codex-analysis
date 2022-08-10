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
import torchvision.models
from torchvision import transforms as trn
from torchsummary import summary

# from torchviz import make_dot
from sklearn.decomposition import PCA
# import vit_pytorch


# personal imports
from dataloader import DataLoader
import utils
from utils import labels_dict, count_files, unique_files, set_splits
from utils import train_dir, val_dir, test_dir
from utils import serialize, deserialize, str2bool

from models import VGG19, VGGEmbedder, AttnVGG_before, vgg19, MultiTaskLoss


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


def forward_pass_coop_eval(x, y, model, device, mode="patch"):
	if mode == "patch":
		x = torch.from_numpy(x)
		y = torch.from_numpy(y)

		x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
		y = y.to(device=device, dtype=torch.long)

		# do we need to set model.eval()?
		scores = model(x)
		val_loss = F.cross_entropy(scores, y)

	if mode == "image":
		x = torch.stack(x, dim=0)
		y = torch.from_numpy(np.array(y))

		x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
		y = y.to(device=device, dtype=torch.long)

		# -------------------need to add PCA here------------
		print(x.shape)

		pca = PCA(n_components=10) # have to make 3 instead of 10 here
		x_im = pca.fit_transform(x.cpu())
		x_im = torch.from_numpy(x_im)
		x = x_im.to(device=device, dtype=dtype)  # move to device, e.g. GPU
		#-------------------------------------------------------

		scores = model(x)
		val_loss = F.cross_entropy(scores, y.long())
	
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

	elif "coop" in model_flag:
		num_correct_img, num_samples_img, cum_loss_img = 0, 0, 0
		losses_img, probs_img, preds_img, img_names, labels_img = [], [], [], [], []
		model_t.eval() 

		with torch.no_grad():
			embed_dict = defaultdict(list)

			for i, (fxy, x, y) in enumerate(loader):
				loss, correct, samples, probs_batch, preds_batch, labels_batch = forward_pass_proxypred_eval(x, y, model, device, mode="patch")
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
def store_embeds(embed_dict, fxy, x, trained_model, att_flag=False):
	#-----------
	# x = x.detach().clone() # used to be enabled
	#-----------
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
	#-----------
	# torch.cuda.empty_cache() # used to be enabled
	#------------
	return embed_dict




def train_tandem(model_patch, device, optimizer, model_flag, data_flag, bs, alpha=0.05, epochs=EPOCHS, pool_flag=True):
	"""
	Inputs:
	- model: A PyTorch Module of the model to train.
	- optimizer: An Optimizer object we will use to train the model
	- epochs: (Optional) A Python integer giving the number of epochs to train for

	Returns: Nothing, but prints model accuracies during training.
	"""
	# embed_dict = defaultdict(list) # moved to per epoch
	train_losses_patch, train_losses_img = [], []
	train_losses = []
	
	# override optimizer
	if taskcombo_flag == "uncertainty":
		mtl = MultiTaskLoss(model=model_patch, eta=[2.0, 1.0], combo_flag=taskcombo_flag)
		print(list(mtl.parameters()))
		optimizer = optim.RMSprop(mtl.parameters(), lr=LEARN_RATE)
	elif taskcombo_flag == "learnAlpha":
		mtl = MultiTaskLoss(model=model_patch, eta=[0.01], combo_flag=taskcombo_flag) # eta is just alpha
		print(list(mtl.parameters()))
		optimizer = optim.RMSprop(mtl.parameters(), lr=LEARN_RATE)
	elif taskcombo_flag == "simple":
		pass
	else:
		print("specified type of multi-task loss is unsupported")
		return


	# define model_img-------------- #old input size was 4096 --> 10
	model_img = LogisticRegression(input_size=10, num_classes=2) 
	optimizer_img = optim.SGD(model_img.parameters(), lr=LEARN_RATE_IMG, weight_decay=LAMBDA2) # L2 also applied!
	#-------------------------------

	model_patch = model_patch.to(device=device) # move the model parameters to CPU/GPU
	model_img = model_img.to(device=device) # move the model parameters to CPU/GPU

	if model_flag.startswith("ModVGG19") == True:
		# print model 
		summary(model_patch, input_size=(74, 96, 96)) # remove one channel
		reshuffle_seed = 1
		train_loader = DataLoader(utils.train_dir, batch_size=bs, transfer=False, proxy=True, rand_seed=reshuffle_seed)
	else:
		print("choose valid model type! see help docstring!")
		exit()

	model_patch.train()  # put model to training mode
	model_img.train()  # put model to training mode

	# files loaded differently per model class
	for e in range(epochs):

		print("="*30 + "\n", "beginning epoch", e, "\n" + "="*30)
		print("img-level prediction!\n" + "-"*60)
		
		# ********** EXTRACT Embeddings & predict patients *******************
		accumulated_loss_img = 0.0
		train_loss_img = 0.0

		if e == 0: # random initialization of embeddings
			scale_factor = 1

			torch.random.manual_seed(444)
			x_im = torch.rand([10, 4096]) #.requires_grad_(True)
			ys_items = list(utils.labels_dict.items())
			ys = [int(yi[1][1]) for yi in ys_items if yi[1][0] == "train"]
			y_im = torch.from_numpy(np.array(ys))
			# pdb.set_trace()

			x_im = x_im.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y_im = y_im.to(device=device, dtype=torch.long)

			#----------adding PCA-----
			pca = PCA(n_components=10)
			x_im = pca.fit_transform(x_im.cpu())
			x_im = torch.from_numpy(x_im)
			x_im = x_im.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			#--------------------------

			scores_im = model_img(x_im)
			# make_dot(scores, params=dict(list(model_img.named_parameters()))).render("rnn_torchviz", format="png")
			# pdb.set_trace()

			if L1FLAG == True:
				all_linear_params = torch.cat([x.view(-1) for x in model_img.linear.parameters()])
				l1_regularization = LAMBDA1 * torch.norm(all_linear_params, 1)
			else:
				l1_regularization = 0.0
			# above 2 lines are new
			train_loss_img = F.cross_entropy(scores_im, y_im.long(), reduction="mean") #.requires_grad_(True) # UPDATE of image loss!
			train_loss_img += l1_regularization # added
			# accumulated_loss_img += train_loss_img
			# print(train_loss_img)
			# print(accumulated_loss_img)

		    # adding new code:
			optimizer_img.zero_grad()
			# model_img.zero_grad()

			train_loss_img.backward(retain_graph=graph_flag) # used to be True #used to be everything falseeee
			optimizer_img.step()
			train_losses_img.append(train_loss_img.item())

			print("kicking off random image loss!")
			print('[Sub-epoch] Iteration %d, loss = %.4f' % (0, train_loss_img.item()))
			preds, probs, num_correct, num_samples = check_mb_accuracy(scores_im, y_im)
			acc = float(num_correct) / num_samples
			print('[Sub-epoch] minibatch training accuracy: %.4f' % (acc * 100))

		else:
			scale_factor = NUM_SUBEPOCH

			# collecting embeddings in dictionary into arrays
			xs, ys = [], []
			for sample in embed_dict.keys():
				x_i = embed_dict[sample][0]
				y_i = utils.labels_dict[sample][1] #--> num
				xs.append(x_i)
				ys.append(y_i)

			# begin "sub-sepochs" and take multiple steps
			if len(ys) > bs: # greater than standard minibatch size, need to partition and make minibatches
				pass #TO-DO
				#for i in range(100) sub-epochs: 
				#	for batch of 10: 
				#		load part of list and do similar to below
			else:
				for t in range(NUM_SUBEPOCH): # sub-epochs!!
					x_im = torch.stack(xs, dim=0)
					y_im = torch.from_numpy(np.array(ys))
					x_im = x_im.to(device=device, dtype=dtype)  # move to device, e.g. GPU
					y_im = y_im.to(device=device, dtype=torch.long)

					#----------adding PCA-----
					pca = PCA(n_components=10)
					x_im = pca.fit_transform(x_im.cpu())
					x_im = torch.from_numpy(x_im)
					x_im = x_im.to(device=device, dtype=dtype)  # move to device, e.g. GPU
					#--------------------------

					scores_im = model_img(x_im)
					# make_dot(scores_im)

					if L1FLAG == True:
						all_linear_params = torch.cat([x.view(-1) for x in model_img.linear.parameters()])
						l1_regularization = LAMBDA1 * torch.norm(all_linear_params, 1)
					else:
						l1_regularization = 0.0
					# above 2 lines are new
					train_loss_img = F.cross_entropy(scores_im, y_im.long(), reduction="mean") #.requires_grad_(True) # UPDATE of image loss!
					# train_loss_img += l1_regularization # added
					# accumulated_loss_img += train_loss_img
			        
			        # adding new code:
					optimizer_img.zero_grad()
					# model_img.zero_grad()

					if t == NUM_SUBEPOCH-1:
						# adding so we take step in patch level...used to be below 2 lines:
						train_loss_img.backward(retain_graph=graph_flag) # used to be true # should be True so can influence patch level preds?
						optimizer_img.step()
					else:
						train_loss_img.backward(retain_graph=graph_flag)
						optimizer_img.step()

					train_losses_img.append(train_loss_img.item())
					# print(train_loss_img)
					# print(accumulated_loss_img)

					print('[Sub-epoch] Iteration %d, loss = %.4f' % (t, train_loss_img.item()))
					preds, probs, num_correct, num_samples = check_mb_accuracy(scores_im, y_im)
					acc = float(num_correct) / num_samples
					print('[Sub-epoch] minibatch training accuracy: %.4f' % (acc * 100))
		#*********************************************************************

		print("-"*60 + "\n" + "entering patch predictions!\n" + "-"*60)

		# create new embedding dict per epoch; not necessary, but oh well
		embed_dict = defaultdict(list)

		# added to initiate new seed/shuffle every epoch
		train_loader = DataLoader(utils.train_dir, batch_size=bs, transfer=False, proxy=True, rand_seed=reshuffle_seed)

		for t, (fxy, x, y) in enumerate(train_loader):
			print("we're on patch minibatch #:", t)
			if t == 0:
				full_batch_shape = x.shape

			x = torch.from_numpy(x)
			y = torch.from_numpy(y)
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.long)

			# make_dot(scores, params=dict(list(model_patch.named_parameters()))).render("rnn_torchviz", format="png")
			# pdb.set_trace()

			if (taskcombo_flag == "uncertainty") or (taskcombo_flag == "learnAlpha"):
				if graph_flag == True:
					scores, losses, train_loss = mtl(x, y, train_loss_img)
				else:
					scores, losses, train_loss = mtl(x, y, train_loss_img.detach().clone())

			elif taskcombo_flag == "simple":
				scores = model_patch(x)
				train_loss = F.cross_entropy(scores, y, reduction="mean") #.requires_grad_(True) 
				print("train loss before alpha", train_loss)
				if graph_flag == True:
					train_loss += alpha * train_loss_img
				else:
					train_loss += alpha * train_loss_img.detach().clone() 
				print("train loss AFTER alpha", train_loss)

			#torch.clone(train_loss_img) #(accumulated_loss_img / scale_factor) # new!! scrap since already added per backward?
			# Note: Can eventually make alpha a learnable parameter

			# Zero out all of the gradients for the variables which the optimizer
			# will update.
			optimizer.zero_grad()
			# model_patch.zero_grad()

			# This is the backwards pass: compute the gradient of the loss with
			# respect to each parameter of the model.
			train_loss.backward(retain_graph=graph_flag) #used to be True

			# ^ retain_graph makes sure this is carried over when we update with loss_img
			# train_loss.backward(retain_graph=True) # used to be False -> True
			
			# if x.shape == full_batch_shape: # not last batch
			# 	train_loss.backward(retain_graph=False) # used to be true
			# else: # last batch
			# 	print("final batch! retaining graph")
			# 	print("usual batch size:", full_batch_shape)
			# 	print("last batch size:", x.shape)
			# 	train_loss.backward(retain_graph=False)

			# if t == 0:
			# 	train_loss.backward(retain_graph=True)
			# else:
			# 	train_loss.backward(retain_graph=False)


			# Actually update the parameters of the model using the gradients
			# computed by the backwards pass.
			optimizer.step()
			# print(type(train_loss))

			if t % print_every == 0:
				if t == 0:
					print('Iteration %d, loss = %.4f' % (t, train_loss.item()))
				else:
					print('Iteration %d, loss = %.4f' % (t + print_every, train_loss.item()))
				preds, probs, num_correct, num_samples = check_mb_accuracy(scores, y)
				acc = float(num_correct) / num_samples
				print('minibatch training accuracy: %.4f' % (acc * 100))

			#if t != 0 and t % val_every == 0:
			#	print('Checking validation accuracy:')
			#	check_val_accuracy(model, utils.val_dir, "val", device, dtype, batch_flag=True)
			#	print()
			# OR SEE BELOW IF WANT MORE INFREQ

			train_losses_patch.append(train_loss.item())
			train_losses.append(train_loss.item())

			# Store embeds
			embed_dict = store_embeds(embed_dict, fxy, x, model_patch)
		
		# save model per epoch
		torch.save(model_patch, utils.model_dir + model_flag + "_" + data_flag + "_alpha"+ str(alpha) + "_epoch%s_vgg.pt" % e)
		torch.save(model_img, utils.model_dir + model_flag + "_" + data_flag + "_alpha"+ str(alpha) + "_epoch%s_logreg.pt" % e)

		# could also check val acc every epoch
		reshuffle_seed += 1


	# full model save
	torch.save(model_patch, utils.model_dir + model_flag + "_" + data_flag + "_alpha"+ str(alpha) + "_full_vgg.pt")
	torch.save(model_img, utils.model_dir + model_flag + "_" + data_flag + "_alpha"+ str(alpha) + "_full_logreg.pt")

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
			gc.collect()

			if save_embeds_flag == True:
				embed_dict = store_embeds(embed_dict, fxy, x, model, att_flag)

				# save embeddings per epoch // overwrite each epoch's embeddings for now
				serialize(embed_dict, utils.code_dir + args.model_class + "-epochs" + str(args.num_epochs) + "-max_embeddings_train.obj")
			#move away from utils.code_dir and instead ask for a cache_dir in args

		# save model per epoch --> skipping for now
		torch.save(model, args.model_path + "/" + args.string_details + "_epoch%s.pt" % e)
		# could also check val acc every epoch

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
	parser.add_argument('--blindfolded_flag', default=False, type=str2bool, help="T/F if you are running simultaneous gamified learning. Defaults to F. Only relevant if gamified_flag = True.")
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
	print("\nBEGINNING TRAINING OF MODEL:", args.model_class + "\n" + "="*60)
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
				model = torchvision.models.vgg19_bn(pretrained=False) 

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
	if args.model_class in supported_models:
		loss_history = train_classifier(model, device, optimizer, args, save_embeds_flag=False) # flag used to be true, but skipping for now
	
	elif args.gamified_flag == True:
		loss_history, loss_history_img = train_tandem(model, device, optimizer, args, ALPHA, save_embeds_flag=True)
		print("Missing training call for non-patchCNN VGG run")

	# cache the losses
	serialize(loss_history, args.model_path + "/" + args.string_details + "_trainloss.obj")
	fig = plt.plot(loss_history, c="blue", label="train")
	plt.savefig(args.model_path + "/"  + args.string_details + "_trainloss.png", bbox_inches="tight")


if __name__ == "__main__":
	main()