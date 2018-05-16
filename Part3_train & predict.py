
# coding: utf-8


#import packages
import os
import numpy as np
import pandas as pd
import h5py
import gc

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential,Model
from keras.layers import Activation, Dropout, Flatten, Dense,Input

from keras import applications
from keras import optimizers
from keras.optimizers import SGD

#config the GPU. If you don't use GPU comment this part
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))
print('GPU configed')

path_features = '\\dogs&cats\\feature\\'
path_folder='\\dogs&cats\\Combined_4_prediction\\'
epochs = 20
batch_size = 128
predict_data_dir = '\\dogs&cats\\data\\test'
#combe 4 features and flatten to train the model(Only use 4 features because of memory limitation. You can try 3 or 5 features)

features = os.listdir(path_features )

for i,feature in enumerate(features):
    if i != 3:
        print(i,feature)
        with h5py.File(path_features+feature, 'r') as h:
            y_train = np.array(h['label_train'])
            y_validation = np.array(h['label_validation'])

            X_train = np.empty([len(y_train),0])
            X_validation = np.empty([len(y_validation),0])


            train=np.array(h['train'])
            shape=train.shape
            train=train.reshape(shape[0],shape[1]*shape[2]*shape[3])

            validation=np.array(h['validation'])
            shape=validation.shape
            validation=validation.reshape(shape[0],shape[1]*shape[2]*shape[3])

        X_train=np.concatenate((X_train,train),axis=1)
        del train
        gc.collect()

        X_validation=np.concatenate((X_validation,validation),axis=1)
        del validation
        gc.collect()

print("X_train.shape",X_train.shape,
		"X_validation.shape",X_validation.shape,
		"y_train.shape",y_train.shape,
		"y_validation.shape",y_validation.shape)




#training the top model


model = Sequential()
model.add(Dense(256, activation='relu',input_shape=X_train.shape[1:]))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', metrics=['accuracy'])

hist_combined=model.fit(X_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(X_validation, y_validation)
          )



#same the weights
model.summary()
model.save_weights(path_folder+'3_combined_top_weight.h5')
print('Weight saved')


# In[6]:


#Creating training plot and save the pictures
import matplotlib.pyplot as plt
plt.figure(figsize=(16,10)) 
epochs = range(1, len(hist_combined.history['acc']) + 1)
plt.plot(epochs, hist_combined.history['loss'], label='Training Loss') 
plt.plot(epochs, hist_combined.history['val_loss'], label='Validation Loss') 

plt.title('Loss of Top combined 4 Features')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(path_folder+'3_loss_combined_4.jpg')
plt.show()
print('Plot saved')

plt.figure(figsize=(16,10)) 
epochs = range(1, len(hist_combined.history['acc']) + 1)
plt.plot(epochs, hist_combined.history['acc'], label='Training Accuracy') 
plt.plot(epochs, hist_combined.history['val_acc'], label='Validation Accuracy') 
plt.title('Accuracy of Top combined 4 Features')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(path_folder+'3_acc_combined_4.jpg')
plt.show()
print('Plot saved')


# clean the memory
del X_train
del X_validation
gc.collect()
del y_train
del y_validation
gc.collect()

# combine the feature of test dataset and flatten 
for i,feature in enumerate(features):
    if i != 3:
        print(i,feature)
        with h5py.File(path_features+feature, 'r') as h:
            test=np.array(h['test'])
            X_test = np.empty([len(test),0])
            shape=test.shape
            test=test.reshape(shape[0],shape[1]*shape[2]*shape[3])
        X_test=np.concatenate((X_test,test),axis=1)
        del test
        gc.collect()
print(X_test.shape)

#predict the class
pred_class = model.predict(X_test)
print(pred_class[:5])

#save the prediction
ID_n = np.arange(1,len(X_test)+1)
ID={'id':ID_n}


test_img = os.listdir(predict_data_dir+os.sep+'test')
print(test_img[0:5])

prediction_label_df=pd.DataFrame(ID)
prediction_label_df['label']=pred_class
prediction_label_df['Image']=test_img

print('Prediction save to ',path_folder)

# changing the order to the picture order
picture_number = []
for i in range(len(prediction_label_df)):
    picture_number.append(int(prediction_label_df['Image'][i][:-4]))
picture_number
prediction_label_df['picture_number']=picture_number
prediction_pic_order=prediction_label_df.sort_index(axis = 0,ascending = True,by = 'picture_number')
prediction_pic_order['id']=np.arange(1,len(X_test)+1)

#Save the result
#"gbk" is easy to open in excel, utf is easy to reload to python.
prediction_label_df.to_csv(path_folder+'FileOrder_WithImg_gbk.csv',header=True,index=False,encoding="gbk", sep=',')
prediction_label_df.to_csv(path_folder+'FileOrder_label_gbk.csv',columns=['id','label'],header=True,index=False,encoding="gbk", sep=',')
prediction_label_df.to_csv(path_folder+'FileOrder_WithImg_utf8.csv',header=True,index=False)
prediction_label_df.to_csv(path_folder+'FileOrder_label_utf8.csv',columns=['id','label'],header=True,index=False)
print('Prediction save to ',path_folder)

#"gbk" is easy to open in excel, utf is easy to reload to python.
prediction_pic_order.to_csv(path_folder+'PictureOrder_WithImg_gbk.csv',header=True,index=False,encoding="gbk", sep=',')
prediction_pic_order.to_csv(path_folder+'PictureOrder_label_gbk.csv',columns=['id','label'],header=True,index=False,encoding="gbk", sep=',')
prediction_pic_order.to_csv(path_folder+'PictureOrder_WithImg_utf8.csv',header=True,index=False)
prediction_pic_order.to_csv(path_folder+'PictureOrder_label_utf8.csv',columns=['id','label'],header=True,index=False)
print('Prediction save to ',path_folder)

