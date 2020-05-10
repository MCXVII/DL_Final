import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers import Lambda, MaxPool2D, BatchNormalization, MaxPooling2D, Input, Softmax
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Input
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import xml.etree.ElementTree as ET
import sklearn
import itertools
import cv2
import scipy
import os
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image



dict_characters = {1:'NEUTROPHIL',2:'EOSINOPHIL',3:'MONOCYTE',4:'LYMPHOCYTE'}
dict_characters2 = {0:'Mononuclear',1:'Polynuclear'}

def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    z = []
    for wbc_type in os.listdir(folder):
        if not wbc_type.startswith('.'):
            if wbc_type in ['NEUTROPHIL']:
                label = 1
                label2 = 1
            elif wbc_type in ['EOSINOPHIL']:
                label = 2
                label2 = 1
            elif wbc_type in ['MONOCYTE']:
                label = 3  
                label2 = 0
            elif wbc_type in ['LYMPHOCYTE']:
                label = 4 
                label2 = 0
            else:
                label = 5
                label2 = 0
            for image_filename in tqdm(os.listdir(folder + wbc_type)):
                img_file = cv2.imread(folder + wbc_type + '/' + image_filename)
                if img_file is not None:
                    img_file = np.array(Image.fromarray(img_file).resize([80, 60]))
                    #scipy.misc.imresize(arr=img_file, size=(60, 80, 3))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
                    z.append(label2)
    X = np.asarray(X)
    y = np.asarray(y)
    z = np.asarray(z)
    return X,y,z
X_train, y_train, z_train = get_data('/scratch/asz241/Blood/dataset2-master/images/TRAIN/')
X_test, y_test, z_test = get_data('/scratch/asz241/Blood/dataset2-master/images/TEST/')
X_test_simple, y_test_simple, z_test_simple = get_data('/scratch/asz241/Blood/dataset2-master/images/TEST_SIMPLE/')
# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical
y_trainHot = to_categorical(y_train, num_classes = 5)
y_testHot = to_categorical(y_test, num_classes = 5)
y_stestHot = to_categorical(y_test_simple, num_classes = 5)
z_trainHot = to_categorical(z_train, num_classes = 2)
z_testHot = to_categorical(z_test, num_classes = 2)
z_stestHot = to_categorical(z_test_simple, num_classes = 5)
print(dict_characters)
print(dict_characters2)


X_train=np.array(X_train)
X_train=X_train/255.0

X_test=np.array(X_test)
X_test=X_test/255.0


X_test_simple=np.array(X_test_simple)
X_test_simple=X_test_simple/255.0

x_train1, x_valid, y_train1, y_valid = train_test_split(X_train, y_trainHot, test_size=0.20, shuffle= True)


from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

class MetricsCheckpoint(Callback):
    """Callback that saves metrics after each epoch"""
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)

def plotKerasLearningCurve():
    plt.figure(figsize=(10,5))
    metrics = np.load('logs.npy')[()]
    filt = ['acc'] # try to add 'loss' to see the loss learning curve
    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):
        l = np.array(metrics[k])
        plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
        x = np.argmin(l) if 'loss' in k else np.argmax(l)
        y = l[x]
        plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
        plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   
    plt.legend(loc=4)
    plt.axis([0, None, None, None]);
    plt.grid()
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_learning_curve(history):
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./accuracy_curve.png')
    #plt.clf()
    # summarize history for loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./loss_curve.png')
    

input_img = Input(shape = (60, 80, 3))

tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)

tower_2 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_2)



tower_3 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
tower_3 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_3)
tower_3 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_3)


tower_4 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
tower_4 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_4)
tower_4 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_4)
tower_4 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_4)



tower_5 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
tower_5 = Conv2D(64, (1,1), padding='same', activation='relu')(tower_5)

output = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4, tower_5], axis = -1)
output = MaxPooling2D((3,3), strides = (1,1), padding='same')(output)
output = Flatten()(output)
out = Dense(5, activation='softmax')(output)



mod8 = Model(inputs = input_img, outputs = out)
plot_model(mod8, to_file='mod8.png', show_shapes=True, show_layer_names=True)


checkpoint = ModelCheckpoint(filepath='mod8.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
mod8.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

his8 = mod8.fit_generator(datagen.flow(x_train1, y_train1, batch_size=32),
                        steps_per_epoch=len(x_train1)/32, epochs=50, validation_data = (x_valid, y_valid), callbacks = [MetricsCheckpoint('logs'), checkpoint])

score = mod8.evaluate(X_test,y_testHot, verbose=0)

print(score)
