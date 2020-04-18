import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import time
import numpy as np
import keras
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import backend as K

IMG_TYPES = ['sagittal', 'coronal', 'axial']

def load_resnet():
  p = 'data/models/resnet50.h5'
  model = keras.models.load_model(p)
  model.compile()
  return model

def load_softmax(model_type):
  #return keras.models.load_model('models/{}.h5'.format(model_type))
  p = 'data/models/{}.h5'.format(model_type)
  return keras.models.load_model(p)

def mk_softmax(model_type):
  p = 'data/models/{}_wts.h5'.format(model_type)
  model = Sequential()
  model.add(Dense(1, activation='sigmoid', input_dim=2048*3))
  model.load_weights(p)
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
  return model

def get_resnet_output(resnet, paths):
  samples = int(len(paths)/3)
  output = np.zeros((samples, 2048*3))
  for i in range(samples):
    this_output = []
    for j in range(3):
      # each path contains a stack (20-40) of greyscale MRI images 
      batch = np.load(paths[ i * 3 + j ])
      # replicate the greyscale images to 3 channels
      batch = np.repeat(batch[:, :, :, np.newaxis], 3, axis=3)
      batch = batch.astype('float64')
      batch = preprocess_input(batch)
      # take the max output across the stack of images
      pre_output = np.amax(resnet.predict_on_batch(batch), axis=0)
      this_output.append(pre_output)
    # original file order is sagittal, coronal, axial.
    # the model was trained with the output concatenated in reverse order.
    this_output.reverse()
    output[i] = np.concatenate(this_output)
  return output

def read_input_paths(fname):
  prefix = 'data/'
  with open(fname, 'r') as f:
    return [prefix + line.strip() for line in f.readlines()]

def mk_predictions(models, inputs):
  probs = np.hstack([model.predict(inputs) for model in models])
  return probs

def write_train_paths(fname):
  directions = ['sagittal', 'coronal', 'axial']
  dfiles = []
  for d in directions:
    dfiles.append(sorted(glob.glob('data/MRNet-v1.0/train/{}/*.npy'.format(d))))
  combined = []
  for files in zip(dfiles[0], dfiles[1], dfiles[2]):
    combined.append(files[0] + '\n')
    combined.append(files[1] + '\n')
    combined.append(files[2] + '\n')
  with open(fname, 'w') as f:
    f.writelines(combined)
  return

def main(args):
  train_path = 'data/MRNet-v1.0/train_paths.txt'
  #write_train_paths(train_path)
  with open(train_path, 'r') as f:
    path_list =  [line.strip() for line in f.readlines()]
  train_output = get_resnet_output(load_resnet(), path_list)
  print(train_output.shape)
  return
  output_csv = args[2]
  input_paths = read_input_paths(args[1])
  model_types = ['abnormal', 'acl', 'meniscus']
  models = [mk_softmax(x) for x in model_types]
  print(models)
  output = get_resnet_output(load_resnet(), input_paths)
  preds = mk_predictions(models, output)
  print(preds)
  np.savetxt(args[2], preds, delimiter=',')


if __name__ == '__main__':
  main(sys.argv)
