import os
import numpy as np
import glob
import torch
import h5py
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import utils
from utils import serialize, deserialize
from dataloader import EmbedDataset, reduce_Z
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


"""
X: raw data, images or time-series
x: tokens from X
z: token embedding (after inference on x)
Z: embedded data using concatenation of embedded tokens (z)
"""

def print_X_names(label_dict_path):
    keys = []
    label_dict = utils.deserialize(label_dict_path)
    for k in label_dict.keys():
        keys.append(k)
    return keys

def parse_x_coords(x_name):
    pieces = x_name.split("_")
    im_id = pieces[0] + "_" + pieces[1]

    pos = pieces[3]
    ij = pos.split("-")
    i = ij[0].split("coords")[1]
    j = ij[1]
    return im_id, int(i), int(j)

def gather_Z_dims(patch_dir, X_names):
    hf = h5py.File(patch_dir, 'r')
    files = list(hf.keys())
    print("Gathering dimensions...")
    dim_dict = {}
    for im in X_names:
        dim_dict[im] = [0,0]
    for idx,f in enumerate(files):
        im_id, i, j = parse_x_coords(f)
        if i > dim_dict[im_id][0]:
            dim_dict[im_id][0] = i
        if j > dim_dict[im_id][1]:
            dim_dict[im_id][1] = j
    print("done!")
    return dim_dict

def inference_z(model_path, patch_dir, scope="all", cpu=True):
    # enable device
    print("GPU detected?", torch.cuda.is_available())
    if (cpu == False) and torch.cuda.is_available():
        device = torch.device('cuda')
        print("\nNote: gpu available & selected!")
    else:
        device = torch.device('cpu')
        print("\nNote: gpu NOT available!")

    # load data
    hf = h5py.File(patch_dir, 'r')
    files = list(hf.keys())
    print("We have", len(files), "unique patches to embed")
    print("loading model for inference...")
    model = torch.load(model_path, map_location=device)
    model.eval()
    embed_dict = {}

    scope_flag = False
    if isinstance(scope, list):
        print("Specifically embedding for data:", scope)
        scope_flag = True
    with torch.no_grad():
        x_batch, x_names = [], []
        for idx,f in enumerate(files):
            if scope_flag:
                im_id, _, _ = parse_x_coords(f)
                if im_id not in scope:
                    continue
                else:
                    # print("Found patch for", im_id)
                    x = hf[f][()]
                    x_names.append(f)
                    x_batch.append(x)
                    # print("current batch size:", len(x_batch))
            if len(x_batch) == 3:
                # print("we now have a batch!")
                x_batch = np.stack(x_batch, axis=0)
                # print(x_batch.shape)
                x_batch = torch.from_numpy(x_batch)
                if cpu == False:
                    x_batch = x_batch.cuda()
                x_batch = x_batch.to(device=device, dtype=torch.float)
                z_batch = model.encode(x_batch)
                for b,name in enumerate(x_names):
                    embed_dict[name] = z_batch[b,:].cpu().detach().numpy()
                x_batch, x_names = [], [] # reset
            if (idx+1) % 10000 == 0:
                print("finished inference for", (idx+1), "patches")
    
    utils.serialize(embed_dict, "inference_z_embeds.obj")
    print("serialized numpy embeds at: inference_z_embeds.obj")
    return 

def visualize_z(embed_dict_path, dim_dict, scope="all", K=8):
    if scope == "all":
        print("not yet implemented random sampling of all embeds~")
        exit()
    if isinstance(scope, list) and len(scope) > 1:
        print("not yet implemented scope of more than one X") 
        exit()

    embed_dict = utils.deserialize(embed_dict_path)
    embeds_list = []
    sources = []
    x_locs = []
    for k in embed_dict.keys():
        v = embed_dict[k]
        im_id, i, j = parse_x_coords(k)
        embeds_list.append(v)
        sources.append(im_id)
        x_locs.append([i,j])

    array = np.vstack(embeds_list)
    print("total embeds:", array.shape)

    # tsne - color by source
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(array)
    print(X_embedded.shape)
    plt.figure()
    plt.xlabel("tSNE-0")
    plt.ylabel("tSNE-1")
    plt.scatter(X_embedded[:,0], X_embedded[:,1], alpha=0.5, cmap="Dark2")

    kmeans = KMeans(n_clusters=K, random_state=0).fit(array)
    cluster_labs = kmeans.labels_
    plt.figure()
    plt.xlabel("tSNE-0")
    plt.ylabel("tSNE-1")
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=cluster_labs, alpha=0.5, cmap="Dark2")

    plt.figure()
    plt.hist(cluster_labs)

    # plot clusters for image
    zero_id = np.max(cluster_labs) + 1
    our_id = sources[0]
    Z_dim = dim_dict[our_id]
    print("Z is of size:", Z_dim)
    Z = np.zeros((Z_dim[0]+1, Z_dim[1]+1)) + zero_id
    for coord_id,coord in enumerate(x_locs):
        i,j = coord[0], coord[1]
        Z[i,j] = cluster_labs[coord_id]
    
    plt.figure(figsize=(12, 6), dpi=80)
    plt.imshow(Z, vmin=0, vmax=zero_id, cmap="Dark2")
    plt.show()
    print(Z)

def construct_Z(embed_dict_path, X_id, Z_dim):
    embed_dict = utils.deserialize(embed_dict_path)
    embeds_id = {}
    x_locs = []
    for k in embed_dict.keys():
        im_id, i, j = parse_x_coords(k)
        if im_id == X_id:
            v = embed_dict[k]
            embeds_id[(i,j)] = v
            x_locs.append([i,j])

    Z = np.zeros((Z_dim[0]+1, Z_dim[1]+1))
    for coord_id,coord in enumerate(x_locs):
        i,j = coord[0], coord[1]
        Z[i,j] = embeds_id[(i,j)]
    return Z

def pad_Z(Z, desired_dim=110):
    """
    Z: single embedded datum
    """
    h,w,d = Z.shape
    canvas = np.zeros((desired_dim, desired_dim, d))
    i_start = (desired_dim - h) // 2
    j_start = (desired_dim - w) // 2
    canvas[i_start:i_start+h, j_start:j_start+w, :] = Z
    return canvas

def construct_Zs(embed_dict_path, dim_dict, save_dir, scope="all"):
    """
    Reads embeds dict, gathers by image ID, and then creates tensor; saves in a save_dir 
    """
    print("Constructing Z tensors for", scope)
    for X_id in dim_dict.keys():
        print("On ID:", X_id)
        Z_dim = dim_dict[X_id]
        Z = construct_Z(embed_dict_path, X_id, Z_dim)
        Z = pad_Z(Z)
        np.save(Z, save_dir + "/Z-" + im_id + ".npy")
    print("Done!")
    print("saved Z tensors at:", save_dir)


#-------------------------------------------

def construct_Z_efficient(X_id, Z_dim, files, hf, model, device, sample_size=20, d=128):
    # grab triplets and fo inference to get embeddings
    with torch.no_grad():
        x_locs_all = []
        x_batch, x_locs = [], []
        files_remaining = []
        # initialize Z array
        Z = np.zeros((Z_dim[0]+1, Z_dim[1]+1, d))

        # iterate through
        for idx, f in enumerate(files):
            im_id, i, j = parse_x_coords(f)
            if im_id != X_id:
                if "no_shift" in f:
                    files_remaining.append(f)
                continue
            else: # match with X_id
                # skipping no shift for now to keep dimensionality low
                if "noshift" in f: 
                    x = hf[f][()]
                    x_locs.append((i,j))
                    x_locs_all.append((i,j))
                    x_batch.append(x)

            if len(x_batch) == 3:
                x_batch = np.stack(x_batch, axis=0)
                x_batch = torch.from_numpy(x_batch)
                x_batch = x_batch.to(device=device, dtype=torch.float)
                z_batch = model.encode(x_batch)
                for b,loc in enumerate(x_locs):
                    i, j = loc[0], loc[1]
                    Z[i,j,:] = z_batch[b,:].cpu().detach().numpy()
                x_batch, x_locs = [], [] # reset
            # TO-DO: 
            # Need to check for any straggler patches at end that may not be a full batch
            # Then we grab the first couple batch entries

        # grab sample size emebds from id list
        sample_z_dict = {}
        sample_locs = np.random.choice(len(x_locs_all), size=sample_size, replace=False)
        sample_coords = [x_locs_all[sl] for sl in sample_locs]
        for sc in sample_coords:
            sample_z_dict[X_id+"_"+str(i)+"_"+str(j)] = Z[sc[0],sc[1],:]

    return Z, files_remaining, sample_z_dict
       

def construct_Zs_efficient(model, patch_dir, dim_dict, save_dir, device, scope="all"):
    """
    Reads embeds dict, gathers by image ID, and then creates tensor; saves in a save_dir 
    """
    # print("Constructing Z tensors for", scope)
    # print("GPU detected?", torch.cuda.is_available())
    # if (cpu == False) and torch.cuda.is_available():
    #     device = torch.device('cuda')
    #     print("\nNote: gpu available & selected!")
    # else:
    #     device = torch.device('cpu')
    #     print("\nNote: gpu NOT available!")

    # iterate with whatever matches in scope
    if scope == "all":
        scope = dim_dict.keys()
    elif isinstance(scope, list):
        pass

    # load data
    hf = h5py.File(patch_dir, 'r')
    files = list(hf.keys())
    print("We have", len(files), "unique patches to embed")
    # print("loading model for inference...")
    # if model_dir.endswith(".sd"):
    #     print("loading from state dict")
    #     print("OOps not implemented state dict")
    #     # model = ...
    #     model.to(device=device)
    #     exit()
    # elif model_dir.endswith(".pt"):
    #     print("loading from cached model")
    #     model = torch.load(model_dir, map_location=device)
    model.eval()
    sampled_embed_dict = {} # for sampling of embeds 

    # iterate and construct
    for X_id in scope:
        print("On ID:", X_id)
        Z_dim = dim_dict[X_id]
        Z, files, sample_z_dict = construct_Z_efficient(X_id, Z_dim, files, hf, model, device)
        Z = pad_Z(Z)
        np.save(save_dir + "/Z-" + X_id + ".npy", Z)
        for z in sample_z_dict.keys():
            sampled_embed_dict[z] = sample_z_dict[z] # load

    utils.serialize(sampled_embed_dict, "sampled_inference_z_embeds.obj")
    print("serialized sampled numpy embeds at: sampled_inference_z_embeds.obj")
    print("saved Z tensors at:", save_dir)
    print("Done!")


def fit_clustering(embed_dict_path, K=20):
    # uses sampled embeddings to get K clusters
    embed_dict = utils.deserialize(embed_dict_path)
    embeds_list = []
    for k in embed_dict.keys():
        v = embed_dict[k]
        # [im_id, i, j] = k.split("_")
        embeds_list.append(v)

    array = np.vstack(embeds_list)
    print("total embeds:", array.shape)

    # tsne - color by source
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(array)
    print(X_embedded.shape)
    plt.figure()
    plt.xlabel("tSNE-0")
    plt.ylabel("tSNE-1")
    plt.scatter(X_embedded[:,0], X_embedded[:,1], alpha=0.5, cmap="Dark2")

    kmeans = KMeans(n_clusters=K, random_state=0).fit(array)
    cluster_labs = kmeans.labels_
    plt.figure()
    plt.xlabel("tSNE-0")
    plt.ylabel("tSNE-1")
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=cluster_labs, alpha=0.5, cmap="Dark2")

    plt.figure()
    plt.hist(cluster_labs)
    return kmeans


def visualize_Z(Z_path, kmeans_model):
    Z = np.load(Z_path)
    Z_viz = reduce_Z(Z, kmeans_model)
    
    plt.figure(figsize=(12, 6), dpi=80)
    plt.yticks([])
    plt.xticks([])
    plt.imshow(Z_viz, vmin=0, vmax=zero_id, cmap="Dark2")
    plt.show()


def validation_performance(val_loader, model, device):
    val_losses = []
    model.eval()
    for idx, (Z,y) in enumerate(val_loader):
        B,H,W,D = Z.shape
        Z = Z.to(torch.float)
        y = y.to(torch.long)
        Z = torch.reshape(Z, (B,D,H,W))
        Z.to(device=device)
        y.to(device=device)
        scores = model(Z.cuda())
        loss = F.cross_entropy(scores, y.cuda())
        val_losses.append(loss.item())
    model.train()
    return np.mean(val_losses)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_on_Z(model, device, optimizer, Z_path, label_dict_path, train_set, val_set, kmeans_model, epochs=30, batch_size=3, mode="fullZ"):
    # instantiate dataloader
    train_dataset = EmbedDataset(Z_path, label_dict_path, split_list=train_set, mode=mode, kmeans_model=kmeans_model)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataset = EmbedDataset(Z_path, label_dict_path, split_list=val_set, mode=mode, kmeans_model=kmeans_model)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    early_stopper = EarlyStopper(patience=2, min_delta=0.05)

    model.to(device=device)
    model.train()
    train_losses, train_losses_epoch = [], []
    val_losses = []
    for e in range(epochs):
        for idx, (Z,y) in enumerate(train_loader):
            # print("On minibatch:", idx)
            B,H,W,D = Z.shape
            Z = Z.to(torch.float)
            y = y.to(torch.long)
            Z = torch.reshape(Z, (B,D,H,W))
            Z.to(device=device)
            y.to(device=device)
            
            optimizer.zero_grad()
            scores = model(Z.cuda())
            loss = F.cross_entropy(scores, y.cuda())
            loss.backward()
            optimizer.step()
            train_losses_epoch.append(loss.item())
    
        print("end of epoch", e, "train loss:", loss.item())
        train_loss_epoch = np.mean(train_losses_epoch)
        print("end of epoch", e, "AVG train loss:", train_loss_epoch)
        train_losses.append(train_loss_epoch)

        # validation 
        with torch.no_grad(): 
            val_loss = validation_performance(val_loader, model, device)                  
            print("end of epoch", e, "AVG val loss:", val_loss)
            val_losses.append(val_loss)
        print()
        
        if early_stopper.early_stop(val_loss): 
            print("Early stopping triggered!")            
            break
        
    return model, train_losses, val_losses

