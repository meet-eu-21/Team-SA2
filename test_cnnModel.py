import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

class UtkFaceDataGenerator():
    """
    Data generator for the UTKFace dataset. This class should be used when training our Keras multi-output model.
    """
    def __init__(self, df):
        self.df = df
        
    def generate_split_indexes(self, TRAIN_TEST_SPLIT):
        p = np.random.permutation(len(self.df))
        train_up_to = int(len(self.df) * TRAIN_TEST_SPLIT)
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
                images.append(im)
                
                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(images), [np.array(classes), np.array(lengths)]
                    images, classes, lengths = [], [], []
                    
            if not is_training:
                break

a = pd.read_csv("gentest_data.txt") #path of dataframe generated during preprocessing

test_generator = UtkFaceDataGenerator(a)
test_idx, val_none = test_generator.generate_split_indexes(1)


model = tf.keras.models.load_model("model122321.h5") #trained model

clas, class_T, reg, reg_T = [],[],[],[]
for e in test_generator.generate_images(test_idx, True, 1):
	if len(clas) == len(a):
		print("stop")
		break
	else:
		res = model.predict(e[0].reshape((1,1,33,33)))
		clas.append(res[0])
		class_T.append(e[1][0])
		reg.append(res[1])
		reg_T.append(e[1][1])

pd.DataFrame({"clas":clas,"clas_t":class_T,"reg":reg,"reg_T":reg_T}).to_csv("results_2.txt")
    
    
