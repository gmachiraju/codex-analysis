import os
import numpy as np
import pandas as pd
import glob
import torch
import h5py
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
# import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.filters import threshold_otsu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.metrics import silhouette_score

# import shap
from sklearn.ensemble import GradientBoostingClassifier
import scipy

import utils
from utils import serialize, deserialize
from dataloader import EmbedDataset, reduce_Z
from sod_utils import MeanAveragePrecision

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from dataloader import baggify, colocalization
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter


"""
X: raw data, images or time-series
x: tokens from X
z: token embedding (after inference on x)
Z: embedded data using concatenation of embedded tokens (z)
"""

def print_X_names(label_dict_path, arm="train"):
    keys = []
    label_dict = utils.deserialize(label_dict_path)
    for k in label_dict.keys():
        if arm == "train":
            keys.append(k)
        elif arm == "val" and "node" in k:
            keys.append(k.split(".")[0])
    return keys


def parse_x_coords(x_name, arm="train"):
    pieces = x_name.split("_")
    if arm == "train" or arm == "test":
        im_id = pieces[0] + "_" + pieces[1]
        pos = pieces[3]
    elif arm == "val":
        im_id = pieces[0] + "_" + pieces[1] + "_" + pieces[2] + "_" + pieces[3]
        pos = pieces[5]

    ij = pos.split("-")
    i = ij[0].split("coords")[1]
    j = ij[1]
    return im_id, int(i), int(j)


def gather_Z_dims(patch_dir, X_names, arm="train"):
    hf = h5py.File(patch_dir, 'r')
    files = list(hf.keys())
    print("Gathering dimensions...")
    dim_dict = {}
    for im in X_names:
        dim_dict[im] = [0,0]

    for idx,f in enumerate(files):
        im_id, i, j = parse_x_coords(f, arm)
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
    plt.scatter(X_embedded[:,0], X_embedded[:,1], s=2, alpha=0.1, cmap="Dark2")

    kmeans = KMeans(n_clusters=K, random_state=0).fit(array)
    cluster_labs = kmeans.labels_
    plt.figure()
    plt.xlabel("tSNE-0")
    plt.ylabel("tSNE-1")
    plt.scatter(X_embedded[:,0], X_embedded[:,1], s=2, c=cluster_labs, alpha=0.1, cmap="Dark2")

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


def pad_Z(Z, desired_dim=124):
    """
    Z: single embedded datum
    """
    h,w,d = Z.shape
    canvas = np.zeros((desired_dim, desired_dim, d))
    i_start = (desired_dim - h) // 2
    j_start = (desired_dim - w) // 2
    canvas[i_start:i_start+h, j_start:j_start+w, :] = Z
    coords = [(i_start, i_start+h), (j_start, j_start+w)]
    print("padding C (" + str(h) + "x" + str(w) + ") with H,W crops at:", coords)
    return canvas, coords


def construct_Zs(embed_dict_path, dim_dict, save_dir, scope="all"):
    """
    Reads embeds dict, gathers by image ID, and then creates tensor; saves in a save_dir 
    """
    crop_dict = {}
    print("Constructing Z tensors for", scope)
    for X_id in dim_dict.keys():
        print("On ID:", X_id)
        Z_dim = dim_dict[X_id]
        Z = construct_Z(embed_dict_path, X_id, Z_dim)
        Z, crop_coords = pad_Z(Z)
        np.save(Z, save_dir + "/Z-" + im_id + ".npy")
        crop_dict[X_id] = crop_coords
    print("Done!")
    print("saved Z tensors at:", save_dir)


#-------------------------------------------

def construct_Z_efficient(X_id, Z_dim, files, hf, model, device, sample_size=20, d=128, arm="train"):
    num_files = len(files)

    # grab triplets and do inference to get embeddings
    with torch.no_grad():
        x_locs_all = []
        x_batch, x_locs = [], []
        files_remaining = []
        # initialize Z array
        Z = np.zeros((Z_dim[0]+1, Z_dim[1]+1, d))
        save_backup = False

        # iterate through
        for idx, f in enumerate(files):
            im_id, i, j = parse_x_coords(f, arm)
            if im_id != X_id:
                if "noshift" in f:
                    files_remaining.append(f)
                continue
            else: # match with X_id
                # skipping 50shift for now to keep dimensionality low
                if "noshift" in f: 
                    # print("hit:", f)
                    x = hf[f][()]
                    # black background
                    summed = np.sum(x, axis=0)
                    if np.abs(np.mean(summed) - 0.0) < 100 and np.std(summed) < 10:
                        # print("black bg tile!")
                        continue 
                    if np.abs(np.mean(summed) - 765) < 100 and np.std(summed) < 10:
                        # print("white bg tile!")
                        continue 
                    x_locs.append((i,j))
                    x_locs_all.append((i,j))
                    x_batch.append(x)

            # Need to check for any straggler patches at end that may not be a full batch
            # Then we grab the first couple batch entries
            if idx == num_files and len(x_batch) < 3: 
                num_needed = 3 - len(x_batch)
                x_batch.extend([x_backup[i] for i in range(num_needed)])

            if len(x_batch) == 3:
                if save_backup == False:
                    x_backup = x_batch.copy() # for any stragglers
                    save_backup = True
                x_batch = np.stack(x_batch, axis=0)
                x_batch = torch.from_numpy(x_batch)
                x_batch = x_batch.to(device=device, dtype=torch.float)
                z_batch = model.encode(x_batch)
                for b,loc in enumerate(x_locs):
                    i, j = loc[0], loc[1]
                    # print(b,i,j)
                    try:
                        Z[i,j,:] = z_batch[b,:].cpu().detach().numpy()
                    except IndexError:
                        print(i, "or", j, "not in index range for Z:", Z.shape)
                        continue
                x_batch, x_locs = [], [] # reset

        # grab sample size emebds from id list
        sample_z_dict = {}
        sample_locs = np.random.choice(len(x_locs_all), size=sample_size, replace=False)
        sample_coords = [x_locs_all[sl] for sl in sample_locs]
        for sc in sample_coords:
            try:
                sample_z_dict[X_id+"_"+str(sc[0])+"_"+str(sc[1])] = Z[sc[0],sc[1],:]
            except IndexError:
                continue

    return Z, files_remaining, sample_z_dict
       

def construct_Zs_efficient(model, patch_dir, dim_dict, save_dir, device, scope="all", arm="train"):
    """
    Reads embeds dict, gathers by image ID, and then creates tensor; saves in a save_dir 
    """
    # iterate with whatever matches in scope
    if scope == "all":
        scope = dim_dict.keys()
        print("we have this # of test set images:", len(scope))
    elif isinstance(scope, list):
        pass

    # load data
    hf = h5py.File(patch_dir, 'r')
    files = list(hf.keys())
    print("We have", len(files), "unique patches to embed")
    model.eval()
    sampled_embed_dict = {} # for sampling of embeds 
    crop_dict = {}
    # iterate and construct
    for idx, X_id in enumerate(scope):
        Z_dim = dim_dict[X_id]
        print(Z_dim)
        if Z_dim[0] == 0 or Z_dim[1] == 0:
            continue
        print("On ID:", X_id)
        Z, files, sample_z_dict = construct_Z_efficient(X_id, Z_dim, files, hf, model, device, arm=arm)
        # if idx < 3:
        #     plt.figure()
        #     plt.imshow(np.sum(Z, axis=2), cmap="Dark2")
        Z, crop_coords = pad_Z(Z)
        # if idx < 3:
        # plt.figure()
        # plt.imshow(np.sum(Z, axis=2), cmap="jet")
        # plt.show()

        crop_dict[X_id] = crop_coords
        np.save(save_dir + "/Z-" + X_id + ".npy", Z)
        for z in sample_z_dict.keys():
            sampled_embed_dict[z] = sample_z_dict[z] # load
        print("We now have", len(sampled_embed_dict.keys()), "embeddings stored as a sample")
        print()

    embed_path = arm + "_sampled_inference_z_embeds.obj"
    utils.serialize(sampled_embed_dict, embed_path)
    utils.serialize(crop_dict, arm + "_crop_coords.obj")
    print("serialized sampled numpy embeds at:", embed_path)
    print("saved Z tensors at:", save_dir)
    print("Done!")


# function returns WSS score for k values
def calculate_ideal_k(embed_dict_path, ks):
    embed_dict = utils.deserialize(embed_dict_path)
    embeds_list = []
    for k in embed_dict.keys():
        v = embed_dict[k]
        embeds_list.append(v)
    points = np.vstack(embeds_list)
    # scaler = StandardScaler()
    # points = scaler.fit_transform(points)

    sse, sil = [], []
    for k in ks:
        # print("fitting k=" + str(k))
        kmeans = KMeans(n_clusters=k, random_state=0).fit(points)
        
        # WSS/elbow method
        #------------------
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0
        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
        sse.append(curr_sse)

        # Silhouette method
        #-------------------
        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        labels = kmeans.labels_
        sil.append(silhouette_score(points, labels, metric = 'euclidean'))

    return sse, sil


def fit_clustering(embed_dict_path, K=20, alg="kmeans_euc", verbosity="full"):
    # uses sampled embeddings to get K clusters
    embed_dict = utils.deserialize(embed_dict_path)
    embeds_list = []
    for k in embed_dict.keys():
        v = embed_dict[k]
        embeds_list.append(v)
    array = np.vstack(embeds_list)
    # scaler = StandardScaler()
    # array = scaler.fit_transform(array)
    if verbosity == "full":
        print("total embeds:", array.shape[0])
        print("collapsing from dim", array.shape[1], "--> 2")

    # tsne - color by source
    for perplexity in [5,10,20]:
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perplexity).fit_transform(array)
        if verbosity == "full":
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Sampled embeddings for cluster assignment')
            ax1.set_xlabel("tSNE-0")
            ax1.set_ylabel("tSNE-1")
            ax1.set_title("t-SNE (perplexity="+str(perplexity)+")")
            ax1.scatter(X_embedded[:,0], X_embedded[:,1], alpha=0.3, s=1, cmap="Dark2")
        if alg == "kmeans_euc":
            cluster_algo = KMeans(n_clusters=K, random_state=0).fit(array)
        elif alg == "hierarchical_euc":
            # linkage default = ward
            cluster_algo = AgglomerativeClustering(n_clusters=K).fit(array)
        else:
            print("Error: only supporting 'kmeans_euc' and 'hierarchical' for now. No other algos/distance metrics supported as of now")
            exit()
        cluster_labs = cluster_algo.labels_
        if verbosity == "full":
            ax2.set_xlabel("tSNE-0")
            ax2.set_title("t-SNE with K="+str(K)+" clusters")
            ax2.scatter(X_embedded[:,0], X_embedded[:,1], c=cluster_labs, alpha=0.3, s=1, cmap="Dark2")

    unique, counts = np.unique(cluster_labs, return_counts=True)
    if verbosity == "full":
        plt.figure()
        plt.title("Cluster bar chart")
        plt.bar(unique, height=counts)

    return cluster_algo


def visualize_Z(Z_path, kmeans_model):
    Z = np.load(Z_path)
    Z_viz, zero_id = reduce_Z(Z, kmeans_model)
    
    plt.figure(figsize=(12, 6), dpi=80)
    plt.yticks([])
    plt.xticks([])
    plt.imshow(Z_viz, vmin=0, vmax=zero_id, cmap="Dark2")
    plt.show()


def get_contigs(row):
    contig_lengths = []
    for idx,el in enumerate(row):
        if idx == 0:
            prev_token = 0
        if prev_token == 0: # start new
            if el == 1:
                contig = 1
                prev_token = 1
            elif el == 0:
                continue
        elif prev_token == 1: # continue
            if el == 1:
                contig += 1
                prev_token = 1
            elif el == 0: # store and start over
                contig_lengths.append(contig)
                prev_token = 0
    return contig_lengths


def clean_Z(Z, Z_id):
    d = Z.shape[2]
    flatZ = np.sum(Z, axis=2)
    H, W = flatZ.shape[0], flatZ.shape[1]

    # clip crop edges
    crop_dict = utils.deserialize("/home/codex_analysis/codex-analysis/code/test_crop_coords.obj")
    Z_id_trim = Z_id.split("Z-")[1]
    if Z_id_trim in crop_dict.keys():
        crop_coords = crop_dict[Z_id_trim]
        i0, i1 = crop_coords[0][0], crop_coords[0][1]
        j0, j1 = crop_coords[1][0], crop_coords[1][1]
        H_crop, W_crop = i1-i0, j1-j0 
        if (H_crop / 1.75 > W_crop) or (W_crop / 1.75 > H_crop):
            # print("detecting extra long/wide WSI, performing edge trim")
            lr_trim = W_crop // 8
            tb_trim = H_crop // 8
            Z[i0:i0+tb_trim,:,:] = np.zeros((tb_trim,W,d)) # top rows
            Z[i1-1-tb_trim:i1-1,:,:] = np.zeros((tb_trim,W,d)) # bot row
            Z[:,j0:j0+lr_trim,:] = np.zeros((H,lr_trim,d)) # left col
            Z[:,j1-1-lr_trim:j1-1,:] = np.zeros((H,lr_trim,d)) # right col
        else:
            # print("detecting square WSI, performing edge trim")
            lr_trim = W_crop // 15
            tb_trim = H_crop // 12
            Z[i0:i0+tb_trim,:,:] = np.zeros((tb_trim,W,d)) # top rows
            Z[i1-1-tb_trim:i1-1,:,:] = np.zeros((tb_trim,W,d)) # bot row
            Z[:,j0:j0+lr_trim,:] = np.zeros((H,lr_trim,d)) # left col
            Z[:,j1-1-lr_trim:j1-1,:] = np.zeros((H,lr_trim,d)) # right col
    return Z

    # #border lines
    # for i in range(H):
    #     if i < (1/8)*H or i > (7/8)*H:
    #         row = flatZ[i,:] > 0
    #         contigs = get_contigs(row)
    #         # if len(contigs) > 0 and np.max(contigs) > 10:
    #         # super long or many medium ones
    #         if len(contigs) > 0 and ((len(contigs) < 4 and np.max(contigs) > 20) or (len(contigs) > 4 and np.max(contigs) > 10)):
    #             if i < (1/8)*H and np.mean(flatZ[i+1:i+3,:] > 0) < 0.1: # look down
    #                 Z[i,:,:] = np.zeros((H,d))
    #             elif i > (7/8)*H and np.mean(flatZ[i-2:i,:] > 0) < 0.2: # look up
    #                 Z[i,:,:] = np.zeros((H,d))

    # for j in range(W):  
    #     if j < (1/3)*W or j > (2/3)*W:
    #         col = flatZ[:,j] > 0
    #         contigs = get_contigs(col)
    #         if len(contigs) > 0 and ((len(contigs) < 4 and np.max(contigs) > 20) or (len(contigs) > 4 and np.max(contigs) > 10)):
    #             if j < (1/3)*W and np.mean(flatZ[:,j+1:j+3] > 0) < 0.2: # look right
    #                 Z[:,j,:] = np.zeros((W,d))
    #             elif j > (2/3)*W and np.mean(flatZ[:,j-2:j] > 0) < 0.2: # look left
    #                 Z[:,j,:] = np.zeros((W,d))

    # # individual horiz lines
    # for i in range(H):
    #     if i >= 2 or i > H-2:
    #         row = flatZ[i,:] > 0
    #         row_window = flatZ[i-2:i+3,:] > 0
    #         contigs = get_contigs(row)
    #         if len(contigs) > 0 and np.sum(row_window) - np.sum(np.array(contigs)) < 5:
    #             Z[i,:,:] = np.zeros((H,d))

    # # individual vert lines
    # for j in range(W):
    #     if j >= 2 and j < W-2: 
    #         col = flatZ[:,j] > 0
    #         col_window = flatZ[:,j-2:j+3] > 0
    #         contigs = get_contigs(col)
    #         if len(contigs) > 0 and np.sum(col_window) - np.sum(np.array(contigs)) < 5:
    #             Z[:,j,:] = np.zeros((W,d))

    # # clip top and bot edges
    # Z[0,:,:] = np.zeros((W,d))
    # Z[1,:,:] = np.zeros((W,d))
    # Z[W-2,:,:] = np.zeros((W,d))
    # Z[W-1,:,:] = np.zeros((W,d))

    # # clip crop edges
    # crop_dict = utils.deserialize("/home/codex_analysis/codex-analysis/code/test_crop_coords.obj")
    # Z_id_trim = Z_id.split("Z-")[1]
    # if Z_id_trim in crop_dict.keys():
    #     crop_coords = crop_dict[Z_id_trim]
    #     i0, i1 = crop_coords[0][0], crop_coords[0][1]
    #     j0, j1 = crop_coords[1][0], crop_coords[1][1]
    #     Z[i0,:,:] = np.zeros((W,d)) # top rows
    #     Z[i1-1,:,:] = np.zeros((W,d)) # bot row
    #     Z[:,j0,:] = np.zeros((H,d)) # left col
    #     Z[:,j1-1,:] = np.zeros((H,d)) # right col

    # # detect some long contigs near crop borders
    # for i in range(15):
    #     idx_top, idx_bot = i0+i, i1-i-1
    #     row_top, row_bot = flatZ[idx_top,:] > 0, flatZ[idx_bot,:] > 0
    #     row_window_t = flatZ[idx_top-2:idx_top+3,:] > 0
    #     row_window_b = flatZ[idx_bot-2:idx_bot+3,:] > 0
    #     contigs_top, contigs_bot = get_contigs(row_top), get_contigs(row_bot)
    #     if np.sum(row_window_t) - np.sum(np.array(contigs_top)) < 5:
    #         if len(contigs_top) > 0 and len(contigs_top) < 3 and np.max(contigs_top) > 10:
    #             Z[idx_top,:,:] = np.zeros((W,d)) # top rows
    #     if np.sum(row_window_b) - np.sum(np.array(contigs_bot)) < 5:
    #         if len(contigs_bot) > 0 and len(contigs_bot) < 3 and np.max(contigs_bot) > 10:
    #             Z[idx_bot,:,:] = np.zeros((W,d)) # bot rows
    #     if len(contigs_top) > 0 and ((j1-j0) - np.max(contigs_top) < 6):
    #         Z[idx_top,:,:] = np.zeros((W,d))
    #     if len(contigs_bot) > 0 and ((j1-j0) - np.max(contigs_bot) < 6):
    #         Z[idx_bot,:,:] = np.zeros((W,d))

    # for j in range(15):
    #     idx_left, idx_right = j0+j, j1-j-1
    #     col_left, col_right = flatZ[:,idx_left] > 0, flatZ[:,idx_right] > 0
    #     col_window_l = flatZ[:,idx_left-2:idx_left+3] > 0
    #     col_window_r = flatZ[:,idx_right-2:idx_right+3] > 0
    #     contigs_left, contigs_right = get_contigs(col_left), get_contigs(col_right)
    #     if np.sum(col_window_l) - np.sum(np.array(contigs_left)) < 5:
    #         if len(contigs_left) > 0 and len(contigs_left) < 3 and np.max(contigs_left) > 10:
    #             Z[:,idx_left,:] = np.zeros((H,d)) # left cols
    #     if np.sum(col_window_r) - np.sum(np.array(contigs_right)) < 5: 
    #         if len(contigs_right) > 0 and len(contigs_right) < 3 and np.max(contigs_right) > 10:
    #             Z[:,idx_right,:] = np.zeros((H,d)) # right cols
    #     if len(contigs_left) > 0 and ((i1-i0) - np.max(contigs_left) < 6):
    #         Z[:,idx_left,:] = np.zeros((H,d))
    #     if len(contigs_right) > 0 and ((i1-i0) - np.max(contigs_right) < 6):
    #         Z[:,idx_right,:] = np.zeros((H,d))

    # # small flecks on borders
    # for i in range(H):
    #     for j in range(W):
    #         if (j > 2 or j < W-2) and (i > 2 or i < H-2):
    #             if (i < (1/8)*H or i > (7/8)*H) or (j < (1/4)*W or j > (3/4)*W):
    #                 nbhd = flatZ[i-3:i+3,j-3:j+3] > 0
    #                 if np.sum(nbhd) < 4:
    #                     Z[i,j,:] = np.zeros((1,d))
    # return Z


def clean_Zs(Z_test_path, save_dir):
    Z_list = os.listdir(Z_test_path)
    plt.figure(figsize=(12,9))
    for idx, file in enumerate(Z_list):
        Z = np.load(Z_test_path + "/" + file)
        flatZ = np.sum(Z, axis=2) > 0
        plt.subplot(13,10,idx+1)
        plt.imshow(flatZ, cmap = "bone")
        plt.axis('off')
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()
    
    plt.figure(figsize=(12,9))
    for idx, file in enumerate(Z_list):
        Z_id = file.split(".npy")[0]
        Z = np.load(Z_test_path + "/" + file)
        Z = clean_Z(Z, Z_id)
        np.save(save_dir + "/" + Z_id, Z)
        
        flatZ = np.sum(Z, axis=2) > 0
        plt.subplot(13,10,idx+1)
        plt.imshow(flatZ, cmap = "bone")
        plt.axis('off')
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()

    # plt.figure(figsize=(12,9))
    # for idx, file in enumerate(Z_list):
    #     Z = np.load(Z_test_path + "/" + file)
    #     Zn = clean_Z(Z)
    #     np.save(save_dir + "/" + file.split(".npy")[0], Zn)

    #     flatZ = np.sum(Z, axis=2) > 0
    #     flatZn = np.sum(Zn, axis=2) > 0
    #     diff = flatZ.astype(int) + flatZn.astype(int)
    #     plt.subplot(13,10,idx+1)
    #     plt.imshow(diff, cmap = "jet")
    #     plt.axis('off')
    #     plt.subplots_adjust(wspace=0.01, hspace=0.01)
    # plt.show()
    return


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


def class_tfidf(Z_path, label_dict, kmeans_model, K, r, mode="mean"):
    # modes include: mean, concat
    # compute class-based stats for training set --> apply to test set
    class0, class1 = [], []
    class0_norm_tuples, class1_norm_tuples = [], []   
    for Z_file in os.listdir(Z_path):
        Z_id = Z_file.split("Z-")[1].split(".npy")[0]
        lab = label_dict[Z_id]
        Z = np.load(Z_path + "/" + Z_file)
        C, _ = reduce_Z(Z, kmeans_model)
        all_cluster_labs, _ = np.unique(kmeans_model.labels_, return_counts=True)
        bag_dict, num_valid_bag = baggify(C, all_cluster_labs)
        num_valid = (num_valid_bag)
        feat_names = [key for key in sorted(bag_dict.keys())]
        counts = np.array([bag_dict[key] for key in sorted(bag_dict.keys())]) # no normalization -- do it later
        if r > 0:
            coloc_dict, num_valid_coloc = colocalization(C, all_cluster_labs, nbhd_size=r)
            num_valid = (num_valid_bag, num_valid_coloc)
            feat_names_add = [key for key in sorted(coloc_dict.keys())]
            feat_names = feat_names + feat_names_add
            counts_annex = np.array([coloc_dict[key] for key in sorted(coloc_dict.keys())]) # no normalization -- do it later
            counts = np.concatenate([counts, counts_annex])       
        if lab == 0:
            class0.append(counts)
            class0_norm_tuples.append(num_valid)
        else:
            class1.append(counts)
            class1_norm_tuples.append(num_valid)
    
    if mode == "concat":  
        class0 = np.array(class0)
        class1 = np.array(class1)
        class0_tf = np.sum(class0, axis=0) / np.sum(class0) # coloc features are amplified
        class1_tf = np.sum(class1, axis=0) / np.sum(class1) 
        doc_freq0 = np.sum(class0, axis=0) > 0.0
        doc_freq1 = np.sum(class1, axis=0) > 0.0
        # print(class1_tf.shape, doc_freq1.shape)
        class0_idf = np.log(1 / 1 + (doc_freq0))
        class1_idf = np.log(1 / 1 + (doc_freq1))
        # this sholuld be over both classes^

        class0_tfidf = class0_tf * class0_idf
        class1_tfidf = class1_tf * class1_idf
        p_vals = None

    elif mode == "mean":
        class0_tf, class1_tf = [], []
        for idx, doc in enumerate(class0):
            if r == 0:
                norm_idx = doc / class0_norm_tuples[idx]
                class0_tf.append(norm_idx)
            else:
                norm_idx = np.concatenate([doc[:K] / class0_norm_tuples[idx][0], doc[K:] / class0_norm_tuples[idx][1]])
                class0_tf.append(norm_idx)
        for idx, doc in enumerate(class1):
            if r == 0:
                norm_idx = doc / class1_norm_tuples[idx]
                class1_tf.append(norm_idx)
            else:
                norm_idx = np.concatenate([doc[:K] / class1_norm_tuples[idx][0], doc[K:] / class1_norm_tuples[idx][1]])
                class1_tf.append(norm_idx)
       
        class0 = np.array(class0_tf)
        class1 = np.array(class1_tf)
        corpus_tf = np.concatenate([class0, class1], axis=0)
        idf = idf_scale(corpus_tf)
        # print(idf)
        class0 = np.array([doc * idf for doc in class0_tf]) 
        class1 = np.array([doc * idf for doc in class1_tf])
        
        # create outputs to return
        class0_tfidf = np.mean(class0, axis=0)
        class1_tfidf = np.mean(class1, axis=0)
        pvals = []
        p = class0.shape[1]
        for idx in range(p):
            # _, pval = scipy.stats.ttest_ind(class1[:,idx], class0[:,idx], equal_var=False)
            _, pval = scipy.stats.mannwhitneyu(class1[:,idx], class0[:,idx])
            pvals.append(pval / p)
        
    return class0_tfidf, class1_tfidf, pvals


def idf_scale(Z):
    try:
        Z = Z.numpy() # for inputs that are torch
    except:
        pass
    n = Z.shape[0]
    doc_freq = np.count_nonzero(Z, axis=0)
    idf = np.log(n / (1+doc_freq))
    return idf


def train_on_Z(model, device, optimizer, Z_path_train, Z_path_val, label_dict_path_train, label_dict_path_val, train_set, val_set, kmeans_model, epochs=30, batch_size=10, mode="fullZ", nbhd_size=2, verbosity="low"):
    if mode != "fullZ" or mode != "clusterZ":
        if train_set == None or val_set == None:
            batch_size_train = len(os.listdir(Z_path_train)) # overwrite to make sure we keep all in matrix
            batch_size_val = len(os.listdir(Z_path_val))
        elif train_set == [] or val_set == []:
            batch_size_train = len(os.listdir(Z_path_train))
            batch_size_val = len(os.listdir(Z_path_val))
        else:
            batch_size_train = len(train_set) 
            batch_size_val = len(val_set)
    else:
        batch_size_train = batch_size
        batch_size_val = batch_size

    # instantiate dataloader
    train_dataset = EmbedDataset(Z_path_train, label_dict_path_train, split_list=train_set, mode=mode, kmeans_model=kmeans_model, arm="train", nbhd_size=nbhd_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)
    val_dataset = EmbedDataset(Z_path_val, label_dict_path_val, split_list=val_set, mode=mode, kmeans_model=kmeans_model, arm="test", nbhd_size=nbhd_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, num_workers=1)

    if mode != "fullZ" or mode != "ClusterZ":
        for idx, (Z_train, y_train) in enumerate(train_loader):
            break
        for idx, (Z_val, y_val) in enumerate(val_loader):
            break
        if verbosity == "high":
            print("Z train is shape:", Z_train.shape)
            print("Z val is shape:", Z_val.shape)
            print("Fitting sci-kit learn models: returning y_probs, ys")
        
        # idf scaling
        if mode == "clusterbag" or mode == "coclusterbag":
            idf = idf_scale(Z_train)
            Z_train = Z_train * idf
            Z_val = Z_val * idf

        scaler = StandardScaler()
        Z_train = scaler.fit_transform(Z_train.numpy())
        Z_val = scaler.transform(Z_val.numpy())
        y_train = y_train.numpy().astype(int)
        y_val = y_val.numpy().astype(int)
        # print(Z_train)
        # pdb.set_trace()

        model = model.fit(Z_train, y_train)
        score = model.score(Z_val, y_val)
        print("\toverall mean accuracy", score)
        val_losses = y_val
        train_losses = model.predict_proba(Z_val)
        # placeholder names, really ys, y_preds

        FI = model.coef_[0]
        # explainer = shap.KernelExplainer(model.predict_proba, Z_train)
        # FI = explainer.shap_values(Z_val)
        # print(FI)

    else:
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
        print("Fitting torch models: returning train losss and val losses")

    return model, FI, train_losses, val_losses


def eval_classifier(ys, y_probs):
    preds = y_probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(ys, preds)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(ys, preds)
    prc_auc = metrics.auc(recall, precision)
    return roc_auc, prc_auc


def diff_exp(tfidfs, K, r, tau=1.0, alpha=0.05, plot_flag=False):
    class0, class1, pvals = tfidfs[(K,r)]
    if plot_flag == True:
        print("any nans for class0?", np.isnan(class0).any())
        print("any nans for class1?", np.isnan(class1).any())
        print("any nans for pvals?", np.isnan(pvals).any())
    
    x = [str(f) for f in range(len(class0))]
    eps = 1e-10
    log2fc = np.log2((class1+eps) / (class0+eps))

    if plot_flag == True:
        print("mean:", np.mean(log2fc))
        print("min:", np.min(log2fc))
        print("max:", np.max(log2fc))
        print("number of + peaks > 1: ", np.sum(log2fc > tau))
        print("number of - peaks < -1:", np.sum(log2fc < -tau))
   
        plt.figure()
        plt.bar(x, height=log2fc)
        plt.title("Class TF-IDFs features (K=" + str(K) + ",r=" + str(r) + ")")
        plt.xlabel("feature")
        plt.xticks([])
        plt.ylabel("Log2FC of class tfidfs")
        plt.show()

    colors = []
    for idx,l in enumerate(log2fc):
        pval = pvals[idx]
        if (pval > alpha) or (np.abs(l) < tau):
            colors.append("gray")
        elif (pval <= alpha) and (l >= tau):
            colors.append("r")
        elif (pval <= alpha) and (l <= -tau):
            colors.append("b")
        else: # catch all
            colors.append("gray")
        
    neglog10p = -np.log10(pvals)
    min_fc = np.min(log2fc)
    max_fc = np.max(log2fc)
    min_p = np.min(neglog10p)
    max_p = np.max(neglog10p)

    if plot_flag == True:
        plt.figure()
        plt.scatter(log2fc, neglog10p, c=colors, alpha=0.3)
        plt.plot([-1, -1],[min_p,max_p], "k--")
        plt.plot([1, 1],[min_p,max_p], "k--")
        plt.plot([min_fc, max_fc],[min_p,min_p], "k--")
        plt.title("Volcano plot for TF-IDFs features (K=" + str(K) + ",r=" + str(r) + ")")
        plt.xlabel("log2(Fold change)")
        plt.ylabel("-log10(p-value)")
        plt.show()

    # print(len(log2fc), len(neglog10p), len(colors))
    # pdb.set_trace()
    return log2fc, neglog10p, colors


def lofi_map(Z_path, kmeans_model, FI, mode="clusterbag", nbhd_size=4):
    Z = np.load(Z_path)
    # print("pre reduce")
    C, zero_id = reduce_Z(Z, kmeans_model)
    # print("post reduce")
    all_cluster_labs, _ = np.unique(kmeans_model.labels_, return_counts=True)
    # print("pre bag")
    bag_dict, _ = baggify(C, all_cluster_labs)
    # print("post bag") 
    feat_names = [key for key in sorted(bag_dict.keys())]
    if mode == "coclusterbag":
        # print("pre coloc") 
        bag_dict, _ = colocalization(C, all_cluster_labs)
        # print("post coloc") 
        feat_names_add = [key for key in sorted(bag_dict.keys())]
        feat_names = feat_names + feat_names_add
    # print(feat_names)

    # build dict of FIs
    score_dict = {}
    for idx, fn in enumerate(feat_names):
        score_dict[fn] = FI[idx]

    # pad and search all 3x3 nbhds
    # combo_dict = {}
    # combos = itertools.combinations_with_replacement(all_cluster_labs, 2)
    bg = np.max(all_cluster_labs) + 1.0

    H,W,_ = C.shape
    C_pad = np.ones((H+nbhd_size+1, W+nbhd_size+1)) * -1
    C_pad[nbhd_size:H+nbhd_size, nbhd_size:W+nbhd_size] = C[:,:,0]
    M_pad = np.zeros((H+nbhd_size+1, W+nbhd_size+1)) # output

    for i in range(1,H+1):
        for j in range(1,W+1):
            cij = C_pad[i,j]
            if (cij == -1.0) or (cij == bg): # padding or bg labels
                continue
            M_pad[i,j] += score_dict[cij]
            # co clusters keep adding attributions
            if mode == "coclusterbag":
                nbhd = C_pad[i-nbhd_size:i+nbhd_size+1, j-nbhd_size:j+nbhd_size+1]
                unique, counts = np.unique(nbhd, return_counts=True)
                for idx,u in enumerate(unique):
                    if (u == -1.0) or (u == bg): # padding or bg labels
                        continue
                    if u > cij:
                        M_pad[i,j] += score_dict[(cij,u)] * counts[idx]
                    elif u < cij:
                        M_pad[i,j] += score_dict[(u,cij)] * counts[idx]
                    elif u == cij: 
                        M_pad[i,j] += score_dict[(cij,u)] * (counts[idx] - 1)           
    return M_pad


def sod_map_generator(Z_path, M, kmeans_model, crop_dict, mask_path=None, nbhd_size=2, highres_flag=False, viz_flag=True):
    Z_id = Z_path.split("/")[-1].split(".npy")[0].split("Z-")[1]
    Z = np.load(Z_path)
    Z_viz, zero_id = reduce_Z(Z, kmeans_model)

    minM = np.min(M)
    maxM = np.max(M)
    maxmag = np.max([np.abs(minM), np.abs(maxM)])
    
    mask = None
    if viz_flag == True:
        if mask_path is not None:
            try:
                mask = np.load(mask_path)
            except FileNotFoundError:
                print("mask not found, thus also decreasing to low-res")
                highres_flag = False

        fig = plt.figure(figsize=(16, 8))
        plt.suptitle("Slide ID: "+Z_id, fontsize=24)
        if mask_path is None or mask is None:
            grid = ImageGrid(fig, 111,
                            nrows_ncols = (1,3),
                            axes_pad = 0.1,
                            cbar_location = "right",
                            cbar_mode="single",
                            cbar_size="5%",
                            cbar_pad=0.1
                            )
        else:
            grid = ImageGrid(fig, 111,
                            nrows_ncols = (1,4),
                            axes_pad = 0.1,
                            cbar_location = "right",
                            cbar_mode="single",
                            cbar_size="5%",
                            cbar_pad=0.1
                            )

    filename = Z_path.split("/")[-1]
    dropnpy = filename.split(".npy")[0] 
    X_id = dropnpy.split("Z-")[1] 
    i0, i1 = crop_dict[X_id][0]
    j0, j1 = crop_dict[X_id][1]
    C_crop = Z_viz[i0:i1, j0:j1]
    # fg = C_crop < zero_id 
    fg = np.ma.masked_where(C_crop == zero_id, C_crop)
    custom_cmap = plt.get_cmap("Dark2")
    custom_cmap.set_bad(color='white')
    if highres_flag == True:
        fg = cv2.resize(fg, (mask.shape[1],mask.shape[0]), interpolation=cv2.INTER_AREA)
    if viz_flag == True:
        grid[0].imshow(fg, vmin=0, vmax=zero_id, cmap=custom_cmap)
        #used to be C_crop
        grid[0].set_yticks([])
        grid[0].set_xticks([])
        # grid[0].axis("off")
        grid[0].set_title('Mosaic', fontsize=20)

    if nbhd_size > 0:
        M_noborder = M[nbhd_size:-nbhd_size, nbhd_size:-nbhd_size]
    else:
        M_noborder = M
    M_crop = M_noborder[i0:i1, j0:j1]
    if highres_flag == True:
        M_crop = cv2.resize(M_crop, (mask.shape[1],mask.shape[0]), interpolation=cv2.INTER_AREA)
    if viz_flag == True:
        im2 = grid[1].imshow(M_crop, vmin=-maxmag, vmax=maxmag, cmap="bwr")
        grid[1].set_yticks([])
        grid[1].set_xticks([])
        grid[1].set_title('LOFI Map', fontsize=20)
        plt.colorbar(im2, cax=grid.cbar_axes[0])

    # bg_val = 0 #1e10
    # M_pos = np.where(M_crop > 0, M_crop, bg_val)
    # vals_to_search = M_crop[M_crop > 0] 
    # thresh = threshold_otsu(vals_to_search) # M_pos)
    # M_thresh = M_pos > thresh
    # fgbg = M_thresh.astype(float)*2 + fg.squeeze().astype(float)
    # print("thresholding at", thresh) 
    # grid[2].imshow(fgbg, cmap="gray")
    
    vals_to_search = M_crop[M_crop > 0.0] # zero is bg, != 0?
    if len(vals_to_search) > 0:
        M_crop = np.where(M_crop == 0.0, minM, M_crop)
        if len(set(vals_to_search)) == 1: # all the same value, rare
            normalized = np.where(M_crop > 0.0, M_crop, 0)
        else:
            normalized = (M_crop-np.min(vals_to_search)) / (np.max(vals_to_search)-np.min(vals_to_search))
            normalized = np.where(normalized > 0.0, normalized, 0)
    else:
        normalized = np.zeros(M_crop.shape)
    if highres_flag == True:
        normalized = cv2.resize(normalized, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    if viz_flag == True:
        grid[2].imshow(normalized, cmap="gray")
        grid[2].set_yticks([])
        grid[2].set_xticks([])
        grid[2].axis("off")
        grid[2].set_title('P(z=1)', fontsize=20)

    if mask is not None and viz_flag == True:
        print("mask shape:", mask.shape)
        print("map shape: ", M_crop.shape)
        # mask = cv2.resize(mask, (fgbg.shape[1], fgbg.shape[0]), interpolation=cv2.INTER_AREA)
        if highres_flag == False:
            mask = cv2.resize(mask, (M_crop.shape[1], M_crop.shape[0]), interpolation=cv2.INTER_AREA)
            print("mask shape:", mask.shape)
        grid[3].imshow(mask, cmap="bone")
        grid[3].set_yticks([])
        grid[3].set_xticks([])
        grid[3].set_title('Ground Truth', fontsize=20)
        # plt.subplots_adjust(top=0.925)
    
    return C_crop, zero_id, M_crop, normalized, mask


def prob_map_to_coords(Mp):
    idxs_i, idxs_j = np.nonzero(Mp > 0)
    values = Mp[Mp > 0]
    d = {"Confidence":values, "X coordinate":idxs_i, "Y coordinate":idxs_j}
    df = pd.DataFrame(data=d)
    return df


def grid_search_elastic(device, embed_path, Z_train_path, lab_train_path, Z_val_path, lab_val_path):
    Ks = [5,7,10] # used to do 5,10,15,20
    clusterers = ["kmeans_euc"] # ["hierarchical_euc"] # 
    featurizers = ["coclusterbag_4"] #[ "clusterbag", "coclusterbag_1", "coclusterbag_2", "meanpool", "maxpool", "meanmaxpool"]
    l1_mixes = [0.5] #elastic net mixing param: 0-lasso, 1-ridge, 0.5EN
    
    y_probs_all = []
    ys_all = []
    model_strs = []
    adapters_trained = []
    FIs = []
    total_models = len(Ks) * len(clusterers) * len(featurizers) * len(l1_mixes)
    kmeans_models = {}

    print("Beginning shallow training on suite of", total_models, "models")
    model_num = 1
    for K in Ks:
        for clusterer in clusterers:
            print("Fitting clusterer:", clusterer + "...")
            kmeans_model = fit_clustering(embed_path, K=K, alg=clusterer, verbosity="none")
            kmeans_models[str(K)+"-"+clusterer] = kmeans_model
            p = (K*K + 3*K)/2
            for featurizer in featurizers:
                if "_" in featurizer:
                    (m, ns) = featurizer.split("_")
                else:
                    m,ns = featurizer, 0
                for l1_mix in l1_mixes:
                    model_str = m + "-K"+str(K)+"-"+clusterer+"-N"+str(ns)+"-L"+str(l1_mix)
                    print("Training [", model_str, "] --> model #", model_num, "/", total_models)
                    model_num += 1
                    adapter = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=l1_mix, random_state=0, max_iter=3000)
                    # adapter = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(2*p, 2), random_state=0, max_iter=3000)
                    # adapter = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
                    adapter_trained, FI, y_probs, ys = train_on_Z(adapter, device, None, Z_train_path, Z_val_path, lab_train_path, lab_val_path, None, None, epochs=20, mode=m, kmeans_model=kmeans_model, nbhd_size=int(ns))

                    y_probs_all.append(y_probs)
                    ys_all.append(ys)
                    adapters_trained.append(adapter_trained)
                    FIs.append(FI)
                    model_strs.append(model_str)

    return model_strs, y_probs_all, ys_all, FIs, kmeans_models
    

def calc_sod_accs(threshs, probs, y_true):    
    accs = []
    for t in threshs:
        y_pred = probs > t
        y_true = y_true > 0 # extra check on bool enforcement
        cm = confusion_matrix(y_true, y_pred, labels=[True, False])
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / (tp+tn+fn+fp)
        accs.append(acc)
    return accs


def sod_acc_corpus(threshs, accs_0_dict, accs_1_dict, class1_only=False):
    maxs_1 = []
    means_1 = np.zeros(len(threshs))
    for j,key in enumerate(accs_1_dict.keys()):
        accs = accs_1_dict[key]
        means_1 += accs
        maxs_1.append(np.max(accs))
    
    if class1_only == True:
        norm_means_1 = means_1 / (j+1)
        mean_maxs_1 = np.mean(maxs_1)
        return None, mean_maxs_1, None, None, norm_means_1, None

    maxs_0 = []
    means_0 = np.zeros(len(threshs))
    for i,key in enumerate(accs_0_dict.keys()):
        accs = accs_0_dict[key]
        means_0 += accs
        maxs_0.append(np.max(accs))

    maxs = maxs_0 + maxs_1
    means = means_0 + means_1
    mean_maxs_0, mean_maxs_1, mean_maxs = np.mean(maxs_0), np.mean(maxs_1), np.mean(maxs)

    norm_means_0 = means_0 / (i+1)
    norm_means_1 = means_1 / (j+1)
    norm_means = means / (i+j+2)

    return mean_maxs_0, mean_maxs_1, mean_maxs, norm_means_0, norm_means_1, norm_means


def calc_clf_stats(map_ys):
    preds_fh, preds_gs, ys = [], [], []
    for key in map_ys.keys():
        (y_prob_fh, y_prob_gs, lab) = map_ys[key]
        preds_fh.append(y_prob_fh)
        preds_gs.append(y_prob_gs)
        ys.append(lab)

    fpr, tpr, threshold = metrics.roc_curve(ys, preds_fh)
    roc_auc_fh = metrics.auc(fpr, tpr)
    fpr, tpr, threshold = metrics.roc_curve(ys, preds_gs)
    roc_auc_gs = metrics.auc(fpr, tpr)
    
    precision, recall, threshold = precision_recall_curve(ys, preds_fh)
    prc_auc_fh = metrics.auc(recall, precision)
    precision, recall, threshold = precision_recall_curve(ys, preds_gs)
    prc_auc_gs = metrics.auc(recall, precision)
    return roc_auc_fh, roc_auc_gs, prc_auc_fh, prc_auc_gs


def few_hot_classification(M_probs, few=5):
    # vals_to_search = M_crop[M_crop != 0.0]
    #normed_lofi_fg = (vals_to_search - np.min(vals_to_search)) / (np.max(vals_to_search) - np.min(vals_to_search))
    # y_prob = np.mean(normed_lofi_fg) 
    # relu_lofi = M_crop[M_crop > 0.0]
    relu_prob = M_probs[M_probs > 0.0]
    if len(relu_prob) == 0:
        return 0.0
    sorted_index_array = np.argsort(relu_prob)
    sorted_array = relu_prob[sorted_index_array]
    top_few = sorted_array[-few:]
    return np.mean(top_few)
        

def gauss_smooth_classification(M_probs, s=1):
    smoothed = gaussian_filter(M_probs, sigma=s)
    return np.max(smoothed)


def generate_model_outputs(Zs_path, kmeans_model, FI, mode, nbhd_size, crop_dict, save_path, gt_path, label_dict):
    threshs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    lofi_clf = {}
    accs_0_valid, accs_1_valid = {}, {}
    accs_0_so, accs_1_so = {}, {}

    metric_0 = MeanAveragePrecision() # kick off COCO mAP
    metric_1 = MeanAveragePrecision() # kick off COCO mAP
    metric_2 = MeanAveragePrecision() # kick off COCO mAP
    metric_3 = MeanAveragePrecision() # kick off COCO mAP
    metric_4 = MeanAveragePrecision() # kick off COCO mAP
    metric_5 = MeanAveragePrecision() # kick off COCO mAP
    metric_6 = MeanAveragePrecision() # kick off COCO mAP
    metric_7 = MeanAveragePrecision() # kick off COCO mAP
    metric_8 = MeanAveragePrecision() # kick off COCO mAP
    metric_9 = MeanAveragePrecision() # kick off COCO mAP
    metric_10 = MeanAveragePrecision() # kick off COCO mAP
    metrics = [metric_0, metric_1, metric_2, metric_3, metric_4, metric_5, metric_6, metric_7, metric_8, metric_9, metric_10]
    update_hit = False

    gt_exists = False
    for Z_file in os.listdir(Zs_path):
        Z_id = Z_file.split(".npy")[0].split("Z-")[1]
        # print("Processing:", Z_id)
        lab = label_dict[Z_id]
        Z_path = Zs_path + "/" + Z_file
        # C, zero_id = reduce_Z(np.load(Z_path), kmeans_model)
        # print("hi") 
        M = lofi_map(Z_path, kmeans_model, FI=FI, mode=mode, nbhd_size=nbhd_size)
        # print("heyo") 
        C_crop, zero_id, M_crop, M_probs, mask = sod_map_generator(Z_path, M, kmeans_model, crop_dict, nbhd_size=nbhd_size, mask_path=None, viz_flag=False)
        
        # Preprocess Mask
        #-----------------
        if lab == 1:
            try:
                gt = np.load(gt_path + "/" + Z_id + "_gt.npy")
                gt_exists = True
            except FileNotFoundError:
                gt_exists = False
                pass
            gt_patch = cv2.resize(np.float32(gt), (M_probs.shape[1], M_probs.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            gt_exists = False
            gt_patch = np.zeros((M_probs.shape))

        # Preprocess Mosaic
        #--------------------
        # foreground (legit tissue patches):
        idxs_i_C, idxs_j_C = np.nonzero(C_crop[:,:,0] != zero_id)
        idxs_i_gt, idxs_j_gt = np.nonzero(gt_patch > 0)
        idxs_i_total, idxs_j_total = np.concatenate([idxs_i_C, idxs_i_gt]), np.concatenate([idxs_j_C, idxs_j_gt])
        # print("# tissue patches from our two sources:", len(idxs_i_C), len(idxs_i_gt))

        # check gt for duplicate indices
        unique = set()
        for idx in range(len(idxs_i_total)):
            i,j = idxs_i_total[idx],idxs_j_total[idx]
            if (i,j) not in unique:
                unique.add((i,j))
        idxs_i = np.array([el[0] for el in unique])
        idxs_j = np.array([el[1] for el in unique])

        # valid tissue subset
        tissue_preds_valid = M_probs[idxs_i, idxs_j]
        tissue_ys_valid = gt_patch[idxs_i, idxs_j]

        # SO tissue subset
        if gt_exists == True:
            tissue_preds_so = M_probs[idxs_i_gt, idxs_j_gt]
            tissue_ys_so = gt_patch[idxs_i_gt, idxs_j_gt]
            # print(tissue_preds_so)
            # print(tissue_ys_so)

        # Img-level clf - based on map
        #-----------------------------
        # print("Label:", lab)
        y_prob_fh = few_hot_classification(M_probs)
        y_prob_gs = gauss_smooth_classification(M_probs)
        lofi_clf[Z_id] = (y_prob_fh, y_prob_gs, lab)

        # SOD Accuracy
        #--------------
        accs_valid = calc_sod_accs(threshs, tissue_preds_valid, tissue_ys_valid)
        if lab == 1:
            accs_1_valid[Z_id] = accs_valid
        else:
            accs_0_valid[Z_id] = accs_valid

        if gt_exists == True:
            accs_so = calc_sod_accs(threshs, tissue_preds_so, tissue_ys_so)
            if lab == 1:
                accs_1_so[Z_id] = accs_so
            else: # useless
                accs_0_so[Z_id] = accs_so

        # SOD mAP
        #---------
        if gt_exists == True:
            M_bin = M_probs > 0.0  # enforce binarization
            gt_bin = gt_patch > 0.0  # enforce binarization
            if (np.sum(M_bin) == 0.0) or (np.sum(gt_patch) == 0.0):
                pass
            else:
                # batch size of 1
                for idx, metric in enumerate(metrics):
                    M_thresh = M_probs > threshs[idx]
                    if np.sum(M_thresh) == 0.0:
                        pass
                    else:
                        update_hit = True
                        metric.update(preds=M_thresh, target=gt_bin)
                        # pdb.set_trace()

        # SOD FROC - analysis from Camelyon
        #-----------------------------------
        # if gt_exists == True:
        #     print("FROC analysis: processing mask->csv for:", Z_id)
        #     # resize to level 5
        #     M_probs = cv2.resize(np.float32(M_probs), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
        #     df = prob_map_to_coords(M_probs)
        #     df.to_csv(save_path + "/" + Z_id + ".csv", index=False)
        # print()

    result_dict = {}
    # Acc - all valid patches
    mean_max_accs_0, mean_max_accs_1, mean_max_accs, mean_accs_thresh_0, mean_accs_thresh_1, mean_accs_thresh = sod_acc_corpus(threshs, accs_0_valid, accs_1_valid)
    result_dict["mean_max_accs_0_valid"], result_dict["mean_max_accs_1_valid"], result_dict["mean_max_accs_valid"] = mean_max_accs_0, mean_max_accs_1, mean_max_accs
    result_dict["mean_accs_thresh_0_valid"], result_dict["mean_accs_thresh_1_valid"], result_dict["mean_accs_thresh_valid"] = mean_accs_thresh_0, mean_accs_thresh_1, mean_accs_thresh 
    # Acc - so patches
    mean_max_accs_0, mean_max_accs_1, mean_max_accs, mean_accs_thresh_0, mean_accs_thresh_1, mean_accs_thresh = sod_acc_corpus(threshs, accs_0_so, accs_1_so, class1_only=True)
    result_dict["mean_max_accs_0_so"], result_dict["mean_max_accs_1_so"], result_dict["mean_max_accs_so"] = mean_max_accs_0, mean_max_accs_1, mean_max_accs
    result_dict["mean_accs_thresh_0_so"], result_dict["mean_accs_thresh_1_so"], result_dict["mean_accs_thresh_so"] = mean_accs_thresh_0, mean_accs_thresh_1, mean_accs_thresh 
    # image clf
    roc_auc_fh, roc_auc_gs, prc_auc_fh, prc_auc_gs = calc_clf_stats(lofi_clf)
    result_dict["roc_auc_fh"], result_dict["roc_auc_gs"] = roc_auc_fh, roc_auc_gs
    result_dict["prc_auc_fh"], result_dict["prc_auc_gs"] = prc_auc_fh, prc_auc_gs
    # mAP
    if update_hit == False:
        print("warning: some threhsolds of mAP were not generating evaluable SOD prediction maps")
    for idx in range(len(metrics)):
        map_dict_idx = metrics[idx].compute()
        result_dict["map_stats_" + str(idx)] = map_dict_idx
    # run_test(gt_path, save_path, label_dict)

    return result_dict
