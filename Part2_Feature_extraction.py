
# coding: utf-8

import os
import numpy as np
import h5py
import gc

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential,Model
from keras import applications
from keras import backend as K

#config the GPU. If you don't use GPU, comment them(However, CPU will need too much time to extract features)
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
#limit GPU to 80% memory. Change it if you need
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))
print('GPU configed')

# parameters and data path, save path
img_width, img_height = 224, 224
train_data_dir = '\\final_dogs&cats\\data\\train'
validation_data_dir = '\\final_dogs&cats\\data\\validation'
predict_data_dir = '\\final_dogs&cats\\data\\test'
path_folder='\\final_dogs&cats\\feature\\'
batch_size = 128

# Set the image Generator: change reshape the image to img_width, img_height, and create class for training and validation dataset
datagen = ImageDataGenerator(rescale=1. / 255)
features_generator_train = datagen.flow_from_directory(train_data_dir,
                                                target_size=(img_width, img_height),
                                                batch_size=batch_size,
                                                class_mode=None,
                                                shuffle=False)
features_generator_validation = datagen.flow_from_directory(validation_data_dir,
                                                target_size=(img_width, img_height),
                                                batch_size=batch_size,
                                                class_mode=None,
                                                shuffle=False)

features_generator_test = datagen.flow_from_directory(predict_data_dir,
                                                target_size=(img_width, img_height),
                                                batch_size=batch_size,
                                                class_mode=None,
                                                shuffle=False)



#extraction the features from different model ans save features in h5py
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16

def ExtractFeatures(MODEL,features_generator_train,features_generator_validation,features_generator_test):
    print('Extract features from' MODEL.__name__ )
	#channel-last is for tensorflow backend.
	input_tensor = Input(shape=(224, 224, 3))  
    feature_model = MODEL(input_tensor=input_tensor, weights='imagenet', include_top=False)
    with h5py.File(path_folder+"Features_%s.h5"%MODEL.__name__) as h:
        features_train = feature_model.predict_generator(features_generator_train)
        h.create_dataset("train", data=features_train)
        del features_train 
        gc.collect()
        
        features_validation = feature_model.predict_generator(features_generator_validation)
        h.create_dataset("validation", data=features_validation)
        del features_validation
        gc.collect()
        
        features_test = feature_model.predict_generator(features_generator_test)
        h.create_dataset("test", data=features_test) 
        del features_test
        gc.collect()
        
        h.create_dataset("label_train", data=features_generator_train.classes)
        h.create_dataset("label_validation", data=features_generator_validation.classes)
    print(MODEL.__name__,'feature saved')


#extract features
for MODEL in [ResNet50,Xception,InceptionV3,VGG19,VGG16]:
    ExtractFeatures(MODEL,features_generator_train,features_generator_validation,features_generator_test)

