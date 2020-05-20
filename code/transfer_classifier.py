import numpy as np
import torch
# from keras.models import Model
# from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten, Concatenate, Dropout, Dense, Input

def pool_batch(data, batch_size=5, mode='max'):
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

# def classify_from_pooled(dim):
#     x = Input(shape=dim)
#     out1 = GlobalMaxPooling2D()(x)
#     out2 = GlobalAveragePooling2D()(x)
#     out3 = Flatten()(x)
#     out = Concatenate(axis=-1)([out1, out2, out3])
#     out = Dropout(0.5)(out)
#     y = Dense(1, activation='sigmoid')(out)

#     model = Model(inputs=x, outputs=y)
#     return model
