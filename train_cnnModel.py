import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, MaxPool2D, Conv2D, Input
import keras

import numpy as np
import os

import pandas as pd

from tensorflow.keras.utils import to_categorical

class UtkFaceDataGenerator():
    """
    Data generator for the UTKFace dataset. This class should be used when training our Keras multi-output model.
    """
    def __init__(self, df):
        self.df = df
        
    def generate_split_indexes(self):
        p = np.random.permutation(len(self.df))
        train_up_to = int(len(self.df) * 0.7)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]
        
        return train_idx, test_idx
    
        
    def generate_images(self, image_idx, is_training, batch_size=16):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """
        
        # arrays to store our batched data
        images, classes, lengths = [], [], []
        while True:
            for idx in image_idx:
                person = self.df.iloc[idx]
                classe = person['class']
                length = person['length']
                #print('multi_output_data/' + str(person["class"]) + '/chrom_' + str(person["chr"]) + "_" + str(person["pos"]) + "_" + str(person["length"]) + ".csv")

                im = np.array(pd.read_csv('multi_output_data/' +
                                          str(person["class"]) +
                                          '/chrom_' +
                                          str(person["chr"]) +
                                          "_" +
                                          str(person["pos"]) +
                                          "_" +
                                          str(person["length"]) +
                                          ".csv",
                                         header=None,
                                         sep=" "))
                classes.append(to_categorical(classe, 3))
                lengths.append(length)
                images.append(im.reshape((1,33,33)))
                
                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(images), [np.array(classes), np.array(lengths)]
                    images, classes, lengths = [], [], []
                    
            if not is_training:
                break
                

##########  MODEL ARCHITECTURE
visible = Input(shape=(1,33,33))
conv1 = Conv2D(32,(7,7), activation = 'relu',input_shape=(1,33,33), data_format='channels_first')(visible)
bn1 = BatchNormalization()(conv1)
pool1 = MaxPool2D((2,2),input_shape=(1,33,33), data_format='channels_first')(bn1)
dropout1 = Dropout(0.1)(pool1)

conv2 = Conv2D(64,(3,3), activation = 'relu')(dropout1)
bn2 = BatchNormalization()(conv2)
pool2 = MaxPool2D((2,2))(bn2)
dropout2 = Dropout(0.1)(pool2)

conv3 = Conv2D(128,(3,3), activation = 'relu')(dropout2)
bn3 = BatchNormalization()(conv3)
pool3 = MaxPool2D((2,2))(bn3)
dropout3 = Dropout(0.1)(pool3)

flat = Flatten()(dropout3)
hidden1 = Dense(64,activation = 'relu')(flat)
bn4 = BatchNormalization()(hidden1)
dropout4 = Dropout(0.2)(bn4)

out_clas = Dense(3, activation = 'softmax', name='output1')(dropout4)
out_reg = Dense(1, activation = 'elu', name='output2')(dropout4)

model = Model(inputs = visible, outputs=[out_clas, out_reg] )

model.compile(loss = {'output1':"categorical_crossentropy", 'output2':"mse"}, optimizer = 'adam', metrics={'output1':["categorical_accuracy"],"output2":["mean_squared_error"]})
##########  MODEL ARCHITECTURE




#callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)


print()
print("Fit model:")

params = {'dim': (33,33),
          'batch_size': 64,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}

file = r"data\gentrain_data.txt" # dataframe generated during preprocessing
df = pd.read_csv(file, sep=",")
data_generator = UtkFaceDataGenerator(df)
train_idx, valid_idx = data_generator.generate_split_indexes() 

batch_size = 64
valid_batch_size = 64
train_gen = data_generator.generate_images(train_idx, is_training=True, batch_size=batch_size)

valid_gen = data_generator.generate_images(valid_idx, is_training=True, batch_size=valid_batch_size)
# Train model on dataset


history = model.fit(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=100,
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//valid_batch_size)

hist_df = pd.DataFrame(history.history) 
hist_csv_file = 'history120122.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

model.save("model122321.h5", save_format = "h5")
