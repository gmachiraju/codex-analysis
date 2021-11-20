import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

def serialize(obj, path):
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)

def deserialize(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)
    
    
# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/36031646
def str2bool(v):
	if isinstance(v, bool):
	   return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
        
thresh_dict = {"beaker-wikipedia.jpeg":[0.01,0.98,10],
              "bernie-etsy.jpg":[0.01,0.99,2],
              "bunsen-wikipedia.jpeg":[0.05,0.975,10],
              "canary.png":[0.1,0.9,10],
              "chef-wikipedia.jpeg":[0.02,0.97,10],
              "fozzie-wikipedia.jpeg":[0.02,0.9,10],
              "gonzo-wikipedia.jpeg":[0.02,0.9,10],
              "kermit-wikipedia.jpeg":[0.01,0.95,10],
              "oski-mountainproject.jpeg":[0.02,0.9,5],
              "pepe-wikipedia.jpeg":[0.02,0.9,10],
              "piggy-wikipedia.jpeg":[0.001,0.9,10],
              "rizzo-muppetmindset.wordpress.jpeg":[0.02,0.9,10],
              "rowlf-wikipedia.jpeg":[0.02,0.95,10],
              "sam-wikipedia.jpeg":[0.02,0.9,10],
              "scooter-wikipedia.jpeg":[0.01,0.97,10],
              "walter-wikipedia.jpeg":[0.02,0.9,10],
               "fine-knowyourmeme.jpeg":[0.02,0.9,10],
               "ponyo-polygon.jpeg":[0.02,0.9,10],
               "butterfly-insider.jpeg":[0.02,0.9,10],
               "jerrywest-logodesignlove.jpeg":[0.02,0.9,10],
               "cat-bbc.jpeg":[0.02,0.9,10],
               "confused-memearsenal.jpeg":[0.02,0.9,10],
               "noface-fandom.jpeg":[0.02,0.9,10],
               "calcifer-syfywire.jpeg":[0.02,0.9,10]
              }

blank_chs = [82, 81, 79, 78, 77, 74, 73, 69, 65]

ch_names = "/home/groups/plevriti/gautam/codex_analysis/codex-analysis/data_u54/channelNames.txt" 
chs = pd.read_csv(ch_names, header=None)
chs.columns = ["name"]
chs = chs.drop(chs.index[blank_chs])
chs = chs.reset_index(drop=True)
ch_dict = chs.T.to_dict("records")[0]

codex_dir = "/home/groups/plevriti/gautam/codex_analysis/codex-analysis/"
code_dir = codex_dir + "code/"
data_dir =  codex_dir + "data_u54/primary"
patch_dir = codex_dir + "patches/"
model_dir = codex_dir + "models/"

train_dir = patch_dir + "balanced/train/"
val_dir =   patch_dir + "balanced/val/"
test_dir =  patch_dir + "balanced/test/"

# control images -- old
# ctrl_data_dir = "/home/groups/plevriti/gautam/codex_analysis/codex-analysis/data_control/"
# ctrl_patch_dir = "/home/groups/plevriti/gautam/codex_analysis/codex-analysis/patches_control/null_set/"

	# old dataset locations
	# save_dir_rand_images = "/home/groups/plevriti/gautam/codex_analysis/codex-analysis/data_control/"
	# save_dir_rand_patches = "/home/groups/plevriti/gautam/codex_analysis/codex-analysis/patches_control/"

# control images -- new
# ctrl_data_dir = "/home/groups/plevriti/gautam/codex_analysis/codex-analysis/data_control/train_test_regime/"
# ctrl_patch_dir = "/home/groups/plevriti/gautam/codex_analysis/codex-analysis/patches_control/train_test_regime/"

ctrl_data_dir = "/oak/stanford/groups/paragm/gautam/syncontrols/canary-1/extreme_value_pixels/train/" # custom for now 
# ctrl_train_dir = ctrl_data_dir + "train/"
# ctrl_test_dir = ctrl_data_dir + "test/"


# pretraining - now defunct
pretrain_dir = codex_dir + "pcam"
pretrain_x_db = pretrain_dir + "/camelyonpatch_level_2_split_valid_x.h5"
pretrain_y_db = pretrain_dir + "/camelyonpatch_level_2_split_valid_meta.csv"

labels_dict = {"005": ("test", 0),
            "006": ("test", 1),
            "017": ("test", 1),
            "019": ("test", 1),
            "011": ("val", 0),
            "016": ("val", 1),
            "030": ("val", 1),
            "023": ("val", 1),
            "004": ("train", 0),
            "015": ("train", 0),
            "014": ("train", 1),
            "024": ("train", 1),
            "020": ("train", 1),
            "007": ("train", 1),
            "008": ("train", 1),
            "027": ("train", 1),
            "034": ("train", 1),
            "012": ("train", 1),
            "canaryfill": ("val_ctrl", 0), 
            "canaryoutline": ("val_ctrl", 0),
            "canary": ("val_ctrl", 0), # used to be None, but doesn't matter
            "noise": ("val_ctrl", 0)} 

reg_dict = {"005": ("test", "negative"),
            "006": ("test", "positive"),
            "017": ("test", "positive"),
            "019": ("test", "positive"),
            "011": ("val", "negative"),
            "016": ("val", "positive"),
            "030": ("val", "positive"),
            "023": ("val", "positive"),
            "004": ("train", "negative"),
            "015": ("train", "negative"),
            "014": ("train", "positive"),
            "024": ("train", "positive"),
            "020": ("train", "positive"),
            "007": ("train", "positive"),
            "008": ("train", "positive"),
            "027": ("train", "positive"),
            "034": ("train", "positive"),
            "012": ("train", "positive"),
            "canaryfill": ("val_ctrl", 0), 
            "canaryoutline": ("val_ctrl", 0),
            "canary": ("val_ctrl", 0), # used to be None, but doesn't matter
            "noise": ("val_ctrl", 0)} 

ctrl_dict = {"onehot-random-test": ("test", "postive"),
            "onehot-outline-test": ("test", "positive"),
            "onehot-fill-test": ("test", "positive"),
            "onevec-outline-test": ("test", "positive"),
            "onevec-fill-test": ("test", "positive"),
            "complete-noise-test": ("test", "negative"),
            "zerocold-outline-test": ("test", "negative"),
            "zerocold-fill-test": ("test", "negative"),
            "zerovec-outline-test": ("test", "negative"),
            "zerovec-fill-test": ("test", "negative"),
            "onehot-random-train": ("train", "postive"), #train below
            "onehot-outline-train": ("train", "positive"),
            "onehot-fill-train": ("train", "positive"),
            "onevec-outline-train": ("train", "positive"),
            "onevec-fill-train": ("train", "positive"),
            "complete-noise-train": ("train", "negative"),
            "zerocold-outline-train": ("train", "negative"),
            "zerocold-fill-train": ("train", "negative"),
            "zerovec-outline-train": ("train", "negative"),
            "zerovec-fill-train": ("train", "negative")}

ctrl_labels_dict = {"000": ("val", 0),
                    "001": ("val", 1)}


def degree_supervision_guilty(dir):
    # dir: the patch directory
    for patchname in os.listdir(dir):
        patch = np.load(dir + "/" + patchname)
        pass
    return


def count_files(dir):
    return len([1 for x in list(os.scandir(dir)) if x.is_file()])

def unique_files(dir):
    return set([x.split("_")[0].split("reg")[1] for x in os.listdir(dir)])

def set_splits(dir, labels_dict):
    all_files = [x.split("_")[0].split("reg")[1] for x in os.listdir(dir)]
    labels = [labels_dict[u][1] for u in all_files]
    pos = np.sum(labels)
    neg = len(labels) - pos
    return pos, neg
        
def show(image, now=True, fig_size=(5, 5)):
    image = image.astype(np.float32)
    m, M = image.min(), image.max()
    if fig_size != None:
        plt.rcParams['figure.figsize'] = (fig_size[0], fig_size[1])
    plt.imshow((image - m) / (M - m), cmap='gray')
    plt.axis('off')
    if now == True:
        plt.show()

# might not need
# def calculate_auc(y_true, y_pred):
#     print("sklearn auc: {}".format(skroc(y_true, y_pred)))
#     auc, update_op = tf.compat.v1.metrics.auc(y_true, y_pred)
#     with tf.Session() as sess:
#         sess.run(tf.local_variables_initializer())
#         print("tf auc: {}".format(sess.run([auc, update_op])))

# def auc(y_true, y_pred):
#     auc = metrics.auc(y_true, y_pred)[1]
#     get_session().run(local_variables_initializer())
#     return auc