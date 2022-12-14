import h5py
import argparse
import utils
import pdb 
import random
import matplotlib.pyplot as plt

def get_id(fname, dataset_name):
    """
    Grab label for associated file "fname"
        fname: string for filename being queried
    """
    if dataset_name == "cam":
        if "patient" in fname: #validation
            pieces = fname.split("_")
            reg_id = pieces[0] + "_" + pieces[1] + "_" + pieces[2] + "_" + pieces[3] + ".tif"
        else: # train or test
            pieces = fname.split("_")
            reg_id = pieces[0] + "_" + pieces[1] 
    else:
        reg_id = fname.split("_")[0]

    # label = label_dict[reg_id]
    # return reg_id, label
    return reg_id


def get_patch_coords(fname, dataset_name):
    """
    Grab coodinates for associated file "fname"
        fname: string for filename being queried
    """
    if dataset_name == "cam":
        if "patient" in fname: #validation
            pieces = fname.split("_")
            pdb.set_trace()
            coords = pieces[3]
        else: # train or test
            pieces = fname.split("_")
            coords = pieces[3].split("coords")[1].split("-")[0:2] 
            shift_flag = pieces[4]
    else:
        print("error: unsupported dataset -- please configure string search")
        exit()
        # pdb.set_trace()
    
    coords = (int(coords[0]), int(coords[1]))
    return coords, shift_flag


def get_sorted_patch_ids(data_path, dataset_name):
    """
    open hdf5 file
    get keys for all files --> list
    sort list by image ID
    split list by image --> list of lists
    return sorted_keys
    """
    if not data_path.endswith(".hdf5"):
        print("error: only supporting hdf5 for now!")
        exit()

    hf = h5py.File(data_path, 'r')
    files = list(hf.keys())
    hf.close()
    sorted_files = sorted(files)
    partitioned_files = {}

    for file in files:
        file_id = get_id(file, dataset_name)
        if file_id in partitioned_files.keys():
            partitioned_files[file_id].append(file)
        else:
            partitioned_files[file_id] = [file] 
    
    return partitioned_files


def triplet_sampling(args, partitioned_files, arm="train", neighborhood=1):
    """
    create two lists of string triplets to save/cache -- one for 0-class, one for 1-class
        distant_mode: {same, across}; does the distant patch need to come from the *same* image 
                    or can it be sampled from *across* the corpus of patches?
    """
    trip0, trip1 = [], []
    label_dict = utils.deserialize(args.labeldict_path)

    for img_i, key in enumerate(partitioned_files.keys()):
        print("beginning triplet sampling for image:", key) 
        if img_i == 0:
            # print("preparing plot for image 0")
            xas, xns, xds = [], [], []
            trip_coords = []

        lab = label_dict[key]
        patch_list = partitioned_files[key] # for the id we are on
        prev_trip0 = len(trip0)
        prev_trip1 = len(trip1)

        # store all valid patches in some dict for quick access
        patchid_dict = {} 
        for p in patch_list:
            coords, shift_flag = get_patch_coords(p, args.dataset_name)
            if coords in patchid_dict.keys():
                patchid_dict[coords].append(p)
            else:
                patchid_dict[coords] = [p]
        
        # get triplets
        for k, p in enumerate(patch_list):
            if ((k+1) % 30) == 0:
                print("completed sampling for", k+1, "patches")
            
            x_a = p # anchor
            coords, shift_flag = get_patch_coords(p, args.dataset_name)
            
            # get neighbor
            nbhd_samples = []
            for di in range(-neighborhood, neighborhood+1):
                for dj in range(-neighborhood, neighborhood+1):
                    if (di == 0) and (dj == 0):
                        continue # don't want to add anchor as candidate neighbor
                    try:
                        nbr_patch = patchid_dict[coords[0]+di, coords[1]+dj]
                        nbhd_samples.extend(nbr_patch)
                    except KeyError:
                        pass
            if len(nbhd_samples) > 0:
                # randomly select a patch as neighbor
                x_n = random.choice(nbhd_samples)
                coords_n, _ = get_patch_coords(x_n, args.dataset_name)
            else: 
                continue # this patch is unusable if no neighbors

            # get distant
            switch_modes = False
            if args.distant_mode == "across": # needs to be same class though
                same_class = False
                while same_class == False:
                    img_id_to_sample = random.choice(list(partitioned_files.keys()))
                    lab_other = label_dict[img_id_to_sample]
                    if lab_other == lab:
                        same_class = True
                if img_id_to_sample != img_i:
                    # grab any
                    x_d = random.choice(partitioned_files[img_id_to_sample])
                    coords_d, _ = get_patch_coords(x_d, args.dataset_name)
                else:
                    switch_modes = True # now try to sample within image
            elif args.distant_mode == "same" or switch_modes == True:
                dist_selected = False
                all_coords = list(patchid_dict.keys())
                while dist_selected == False:
                    candidate = random.choice(all_coords) # randomly select from dict keys 
                    if (candidate[0] > coords[0]+neighborhood) or (candidate[0] < coords[0]-neighborhood) or (candidate[1] > coords[1]+neighborhood) or (candidate[1] < coords[1]-neighborhood):  # if beyond neighborhood:
                        x_d = random.choice(patchid_dict[candidate]) # then use and get id
                        dist_selected = True
                        coords_d, _ = get_patch_coords(x_d, args.dataset_name)
                    # print("still searching for distant patch...")
           
            # store triplet
            if img_i == 0:
                xas.append(coords)
                xns.append(coords_n)
                xds.append(coords_d)
                trip_coords.append([coords, coords_n, coords_d])

            if lab == 0:
                trip0.append((x_a, x_n, x_d))
            elif lab == 1:
                trip1.append((x_a, x_n, x_d))
            # print("coordinates (a,n,d):", coords, coords_n, coords_d)
            
        # plot coordinates
        if img_i == 0 and args.distant_mode == "same":
            plt.figure()
            x_vala, y_vala = [x[0] for x in xas], [x[1] for x in xas]
            plt.scatter(x_vala, y_vala)
            x_valn, y_valn = [x[0] for x in xns], [x[1] for x in xns]
            plt.scatter(x_valn, y_valn)
            x_vald, y_vald = [x[0] for x in xds], [x[1] for x in xds]
            plt.scatter(x_vald, y_vald)
            plt.savefig("triplet_sampling_coordinates.png")

            plt.figure()
            for coords_list in trip_coords:
                x_vald, y_vald = [x[0] for x in coords_list], [x[1] for x in coords_list]
                plt.plot(x_valn, y_valn)
            plt.savefig("triplet_sampling_connectivity.png")

        curr_trip0 = len(trip0)
        curr_trip1 = len(trip1)
        print("finished generating triplets for:", key)
        print("current triplet count (0,1):", curr_trip0, curr_trip1)
        print("added triplet count (0,1):", curr_trip0-prev_trip0, curr_trip1-prev_trip1)
        print("serializing triplets")
        utils.serialize(trip0, args.save_path + "/triplets0_list.obj")
        utils.serialize(trip1, args.save_path + "/triplets1_list.obj")
    
    print("completed sampling process!")
    return

    
def quadruplet_sampling(args, partitioned_files, neighborhood=2):
    pass   



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=None, type=str, help='path to hdf5 file containing valid patches extracted from data')
    parser.add_argument('--sampling_number', default=3, type=int, help='style of sampling: triplet (3) or quadruplet (4)')
    parser.add_argument('--labeldict_path', default=None, type=str, help='path for label dictionary')
    parser.add_argument('--dataset_name', default=None, type=str, help='e.g. cam')
    parser.add_argument('--arm', default="train", type=str, help='study arm: train,val,test')
    parser.add_argument('--distant_mode', default="same", type=str, help='Where should distant token come from? *same* data source? or *across* the corpus?')

    args = parser.parse_args()

    # get the parent dir of the hdf5 path
    folders = args.data_path.split("/")
    save_path = '/'.join(folders[:-1])
    setattr(args, "save_path", save_path)
    
    if args.sampling_number in [3,4]:
        ids = get_sorted_patch_ids(args.data_path, args.dataset_name)
        if args.sampling_number == 3:
            triplet_sampling(args, ids)
        elif args.sampling_number == 4:
            quadruplet_sampling(args, ids)
    else:
        print("Error: unsupported sampling_number input~")
        exit()

if __name__ == "__main__":
	main()

