
import random as rn
import numpy as np
import tensorflow as tf

np.random.seed(42)

rn.seed(12345)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                        inter_op_parallelism_threads=1)

tf.compat.v1.set_random_seed(1234)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, concatenate, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.keras.applications.efficientnet import EfficientNetB5
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

from tensorflow.keras.applications.resnet50 import ResNet50

import pathlib
import cv2
from tqdm import tqdm
import os
from zipfile import ZipFile
from PIL import Image
from os import listdir
from os.path import join
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()

# hyper-parameters
batch_size = 32

# categories of images
num_classes = 2

# number of training epochs
epochs = 500

IMG_SIZE = 224



def load_data2():
    """This function loads dataset, normalized, and labels one-hot encoded"""
    train_data_dir = pathlib.Path('/content/drive/My Drive/MeatFreshnessIdentification/Original_Dataset/Train')
    test_data_dir = pathlib.Path('/content/drive/My Drive/MeatFreshnessIdentification/Original_Dataset/Test')
    train_folders = os.listdir(train_data_dir)
    test_folders = os.listdir(test_data_dir)

    train_image_names = []
    test_image_names = []
    train_labels = []
    train_images = []
    test_labels = []
    test_images = []

    size = 224, 224

    for folder in train_folders:
        for file in os.listdir(os.path.join(train_data_dir,folder)):
            if file.endswith("jpg"):
                train_image_names.append(file)
                train_labels.append(folder)
                img = cv2.imread(os.path.join(train_data_dir,folder,file))
                im = cv2.resize(img,size)
                train_images.append(im)
            else:
                continue

    for folder in test_folders:
        for file in os.listdir(os.path.join(test_data_dir,folder)):
            if file.endswith("jpg"):
                test_image_names.append(file)
                test_labels.append(folder)
                img = cv2.imread(os.path.join(test_data_dir,folder,file))
                im = cv2.resize(img,size)
                test_images.append(im)
            else:
                continue

    X_train = np.array(train_images)
    X_test = np.array(test_images)

    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255

    train_label_dummies = pandas.get_dummies(train_labels)
    test_label_dummies = pandas.get_dummies(test_labels)

    trainlabels = train_label_dummies.values.argmax(1)
    testlabels = test_label_dummies.values.argmax(1)

    X_train = np.array(X_train)
    y_train = np.array(trainlabels)


    X_test = np.array(X_test)
    y_test = np.array(testlabels)

    print('OK')
    return(X_train, y_train),(X_test, y_test), (train_image_names, test_image_names)

def create_model():

    MyModel = EfficientNetB5(weights = "imagenet", include_top = False, input_shape = (224,224,3))
    x = MyModel.output

    print(x.shape)
    x = Flatten()(x)

    x1 = Dense(100, activation = 'relu')(x)
    dropout_1 = Dropout(0.02)(x1)
    x2 = Dense(100, activation = 'relu')(dropout_1)
    dropout_2 = Dropout(0.02)(x2)
    x3 = Dense(100, activation = 'relu')(x2)


    predictions = Dense(100, activation = 'softmax')(x)
    model_1 = Model(inputs = MyModel.input, outputs = predictions)
    model_1.compile(optimizer = tf.compat.v1.train.AdamOptimizer(), loss = tf.keras.losses.sparse_categorical_crossentropy, metrics = ["accuracy"])
    return model_1

if __name__ == "__main__":

    # load the data
    (X_train, y_train), (X_test, y_test), (train_image_names, test_image_names) = load_data2()

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.3, random_state=1)
    #print("Validation samples:", X_val.shape[0])

    # construct the model
    model = create_model()

    # train
    model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=2,
            validation_steps=2,
            validation_data=(X_val, y_val))
    validation_steps = 20

    # evaluate
    loss0, accuracy0 = model.evaluate(X_val, y_val, steps = 20)

    print("Validation Loss: {:.2f}".format(loss0))
    print("Validation Accuracy: {:.2f}".format(accuracy0))

# test item prediction
testLabelPredicted = model.predict(X_test)
testLabelPredicted = testLabelPredicted.argmax(axis=-1)
#print(test_image_names)
testLabelGold = y_test
#print(testLabelGold)

# Prediction Write to a File
predictedFileDir = "/content/drive/My Drive/MeatFreshnessIdentification/Output"
with open(predictedFileDir+"EfficientNetB5_23.txt",'w') as f:
  for item in testLabelPredicted:
    #print(item)
    f.write('%d\n' % (item))
f.close


# Evaluation
results = confusion_matrix(testLabelGold, testLabelPredicted)

print ('Confusion Matrix :')
print (results)

print ('Recall Score :',recall_score(testLabelGold, testLabelPredicted, labels=[0,1], pos_label=0))
print ('Precision Score :',precision_score(testLabelGold, testLabelPredicted, labels=[0,1], pos_label=0))
print ('F1 Score :',f1_score(testLabelGold, testLabelPredicted, labels=[0,1], pos_label=0))
print ('Accuracy :',accuracy_score(testLabelGold, testLabelPredicted))


# 0 = Fresh and 1= NotFresh
