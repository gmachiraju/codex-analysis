# from tensorflow import metrics, local_variables_initializer
# from keras.backend import get_session
# import pandas as pd
from sklearn.metrics import roc_auc_score as skroc
# import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

train_dir = '/home/disk2/train/'
val_dir = '/home/data/val_balance/'
test_dir = '/home/data/test_balance/'
transfer_dir = '/scratch/users/gmachi/codex/transfer/'
model_dir = '/home/data/model/'


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
            "012": ("train", 1)} 

def auc(y_true, y_pred):
    auc = metrics.auc(y_true, y_pred)[1]
    get_session().run(local_variables_initializer())
    return auc

def calculate_auc(y_true, y_pred):
    print("sklearn auc: {}".format(skroc(y_true, y_pred)))
    auc, update_op = tf.compat.v1.metrics.auc(y_true, y_pred)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        print("tf auc: {}".format(sess.run([auc, update_op])))
        
def show(image, now=True, fig_size=(5, 5)):
    image = image.astype(np.float32)
    m, M = image.min(), image.max()
    if fig_size != None:
        plt.rcParams['figure.figsize'] = (fig_size[0], fig_size[1])
    plt.imshow((image - m) / (M - m), cmap='gray')
    plt.axis('off')
    if now == True:
        plt.show()
