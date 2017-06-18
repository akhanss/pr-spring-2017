"""
Trains a CNN model using tflearn wrapper for tensorflow
"""


import tflearn
import h5py
import numpy as np
from cnn_model import CNNModel 


# Load HDF5 dataset
#h5f = h5py.File('../data/train.h5', 'r')
h5fs = []
h5fs.append(h5py.File('../data/traindataset1.h5'))
h5fs.append(h5py.File('../data/traindataset2.h5'))
h5fs.append(h5py.File('../data/traindataset3.h5'))
h5fs.append(h5py.File('../data/traindataset4.h5'))
h5fs.append(h5py.File('../data/traindataset5.h5'))

X_train_images = []
Y_train_labels = []
X_val_images = None
Y_val_labels = None
pos = 1
for i in range(0, 5):
  if i == pos:
    X_val_images = h5fs[i]['X']
    Y_val_labels = h5fs[i]['Y']
  else:
    X_train_images += list(h5fs[i]['X'])
    Y_train_labels += list(h5fs[i]['Y'])

X_train_images = np.array(X_train_images)
Y_train_labels = np.array(Y_train_labels)
## Model definition
convnet  = CNNModel()
network = convnet.define_network(X_train_images)
model = tflearn.DNN(network, tensorboard_verbose=0,\
		 checkpoint_path='nodule3-classifier.tfl.ckpt')
model.fit(X_train_images, Y_train_labels, n_epoch = 70, shuffle=True,\
			validation_set = (X_val_images, Y_val_labels), show_metric = True,\
			batch_size = 96, snapshot_epoch = True, run_id = 'nodule3-classifier')
model.save("nodule3-classifier_2.tfl")
print("Network trained and saved as nodule3-classifier_2.tfl!")
