# -*- coding: utf-8 -*-

#!pip install voxelmorph
import os, sys
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'Need Tensorflow 2.0+'
import voxelmorph as vxm
import neurite as ne
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def load_data
# Load data
(x_train_load, y_train_load), (x_test_load, y_test_load) = mnist.load_data()

digit_sel = np.random.random_integers(0,9)
# extract only instances of the digit selected
x_train = x_train_load[y_train_load==digit_sel, ...]
y_train = y_train_load[y_train_load==digit_sel]
x_test = x_test_load[y_test_load==digit_sel, ...]
y_test = y_test_load[y_test_load==digit_sel]

nb_val = 1000  # keep 1,000 subjects for validation
x_val = x_train[-nb_val:, ...]  # this indexing means "the last nb_val entries" of the zeroth axis
y_val = y_train[-nb_val:]
x_train = x_train[:-nb_val, ...]
y_train = y_train[:-nb_val]

# fix data: normalize
x_train = x_train.astype('float')/255
x_val = x_val.astype('float')/255
x_test = x_test.astype('float')/255

# fix data: we force our images to be size 32x
pad_amount = ((0, 0), (2,2), (2,2))
x_train = np.pad(x_train, pad_amount, 'constant')
x_val = np.pad(x_val, pad_amount, 'constant')
x_test = np.pad(x_test, pad_amount, 'constant')
print('shape of x_train: {}, y_train: {}'.format(x_train.shape, y_train.shape))

# visualize the data: neurite to plot slices!
nb_vis = 5
idx = np.random.choice(x_train.shape[0], nb_vis, replace=False)
example_digits = list(x_train[idx, ...])
if 0: # plot slices
    ne.plot.slices(example_digits, cmaps=['gray'], do_colorbars=True);


# build model using VxmDense
inshape = x_train.shape[1:]

# configure unet features 
nb_features = [
    [32, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]

vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)


print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))

# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

def vxm_data_generator(x_data, batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """
    # preliminary sizing
    vol_shape = x_data.shape[1:] # extract data shape
    ndims = len(vol_shape)
    
    # prepare a zero array the size of the deformation
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        
        inputs = [moving_images, fixed_images]
        #import pdb;pdb.set_trace()
        
        #outputs: fixed image itself and the deformation field. 
        outputs = [fixed_images, zero_phi]
        
        yield (inputs, outputs)

# let's test it
train_generator = vxm_data_generator(x_train)
in_sample, out_sample = next(train_generator)

# visualize X and Y
images = [img[0, :, :, 0] for img in in_sample + out_sample] 
titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
if 0: #plot 
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True);


hist = vxm_model.fit(train_generator, epochs=10, steps_per_epoch=100, verbose=2);


def plot_history(hist, loss_name='loss'):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
plot_history(hist)

# let's get some data
val_generator = vxm_data_generator(x_val, batch_size = 1)
val_input, _ = next(val_generator)


val_pred = vxm_model.predict(val_input)


images = [img[0, :, :, 0] for img in val_input + val_pred] 
titles = ['moving', 'fixed', 'moved', 'flow']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True);
ne.plot.flow([val_pred[1].squeeze()], width=5);

# Test on other digits
for i in range(0,10):
    # extract only instances of the digit 7
    x_sevens = x_train_load[y_train_load==i, ...].astype('float') / 255
    x_sevens = np.pad(x_sevens, pad_amount, 'constant')

    # predict
    seven_generator = vxm_data_generator(x_sevens, batch_size=1)
    seven_sample, _ = next(seven_generator)
    seven_pred = vxm_model.predict(seven_sample)

    # visualize
    images = [img[0, :, :, 0] for img in seven_sample + seven_pred] 
    titles = ['moving', 'fixed', 'moved', 'flow']
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True);





