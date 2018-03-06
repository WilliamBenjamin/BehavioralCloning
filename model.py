
import numpy as np
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
df = pd.read_csv("/home/william/Desktop/driving_log.csv")
df.columns = ['center', 'left', 'right', 'steering_angle', 'throttle', 'brake', 'speed']

#filtering based on steering




def read_image(path,dir):
    image = plt.imread(os.path.join(dir,os.path.basename(path)))
    return cv2.cvtColor(image,cv2.COLOR_RGB2YUV) # convert to YUV space

def flip_image(path,dir):
    image = read_image(path,dir)
    return cv2.flip(image,1)
# process the center images

df_large_str = df[ abs(df.steering_angle) > 0.14]
fraction_to_cull = 0.4
df_small_str = df[ abs(df.steering_angle) < 0.14]
df_frac_small = df_small_str.sample(frac=fraction_to_cull) # cull fraction_to_cull  data
df_filtered = df_large_str.append(df_frac_small)

X_db = []
y_db = np.array([])
for name in ['center']:
    images_paths = df_filtered[name].tolist()
    X_db = X_db + images_paths
    y_db = np.concatenate([y_db,df_filtered['steering_angle'].values])

#process the left images
left_correction = 0.2
for name in ['left']:
    images_paths = df_filtered[name].tolist()
    X_db = X_db + images_paths
    y_db = np.concatenate([y_db, df_filtered['steering_angle'].values+left_correction])

# process the  left images
right_correction = -0.2
for name in ['right']:
    images_paths = df_filtered[name].tolist()
    X_db = X_db + images_paths
    y_db = np.concatenate([y_db, df_filtered['steering_angle'].values+right_correction])


X_db, y_db = shuffle(X_db,y_db)
X_train, X_test, y_train, y_test = train_test_split(X_db, y_db, test_size=0.2, random_state=42)

def generator(X_paths,y,batch_size = 32):
    num_samples = len(X_paths)
    batch_offset = int(batch_size/2)
    while 1:
        for offset in range(0, num_samples, batch_offset):
            X_paths_sample = X_paths[offset:offset + batch_offset]
            y_sample = y[offset:offset + batch_offset]
            X_sample = [read_image(file,"/home/william/Desktop/IMG") for file in X_paths_sample]
            # add image flips
            X_sample_flipped = [flip_image(file, "/home/william/Desktop/IMG") for file in X_paths_sample]
            y_sample_flipped = y_sample*(-1)
            # concatenate
            X_concat = X_sample+X_sample_flipped
            y_concat = np.concatenate((y_sample,y_sample_flipped))
            X_train = np.array(X_concat)
            y_train = np.array(y_concat)
            yield shuffle(X_train, y_train)

train_generator = generator(X_train,y_train, batch_size=64)
validation_generator = generator(X_test,y_test, batch_size=32)


model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(16, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001),init='glorot_uniform'))
model.add(ELU())

model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='valid',W_regularizer=l2(0.001), init='glorot_uniform'))
model.add(ELU())

model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001),init='glorot_uniform'))
model.add(ELU())

model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid',W_regularizer=l2(0.001),init='glorot_uniform'))
model.add(ELU())

model.add(Convolution2D(128, 3, 3, subsample=(1, 1), border_mode='valid',W_regularizer=l2(0.001), init='glorot_uniform'))
model.add(ELU())

model.add(Flatten())

model.add(Dense(100,W_regularizer=l2(0.001)))
model.add(ELU())

model.add(Dense(50,W_regularizer=l2(0.001)))
model.add(ELU())

model.add(Dense(10,W_regularizer=l2(0.001)))
model.add(ELU())



model.add(Dense(1))

model.compile(loss = 'mse',optimizer= 'adam')
#model.fit(X_train,y_train, validation_split=0.2,shuffle=True,nb_epoch=5,batch_size= 10)
model.fit_generator(train_generator, steps_per_epoch= len(X_train), validation_data=validation_generator,validation_steps=len(X_test), epochs=5, verbose =1)
model.save('../models/model.h5')