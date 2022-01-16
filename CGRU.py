'''
import os
import itertools
import re
import datetime
import numpy as np
from keras.utils import plot_model
from scipy import ndimage
import pylab
os.environ["PATH"] += os.pathsep + 'D:/软件/Graphviz/bin'
from keras import backend as K, models, layers, regularizers
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Input, Dense, Activation, Reshape, Lambda, Dropout
from keras.layers.merge import add, concatenate
from keras.layers.recurrent import GRU
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, Callback

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc

model = models.Sequential()
model.add(layers.Conv1D(256, 5, activation='relu', input_shape=(126, 1)))
model.add(layers.MaxPooling1D(pool_size=(8)))
model.add(layers.Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling1D(pool_size=(3)))

model.add(GRU(256,input_shape=(None,5,128),init='glorot_uniform', inner_init='orthogonal',
               activation='tanh',
               inner_activation='hard_sigmoid', W_regularizer=regularizers.l2(0.0005),
               U_regularizer=regularizers.l2(0.0005), b_regularizer=regularizers.l2(0.005),
               dropout_W=0., dropout_U=0., return_sequences=False))
model.add(Dense(output_dim=128, init='glorot_uniform',
                W_regularizer=regularizers.l2(0.0005), b_regularizer=regularizers.l2(0.005)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
print(model.output_shape)

model.add(Dense(6))
model.add(Activation('softmax'))
plot_model(model, to_file='mfcc_model.png', show_shapes=True)
model.summary()
'''
from keras import Sequential
from keras.layers import Dense, GRU, Activation, Dropout, Bidirectional, Masking
from keras import regularizers
from random import seed
import librosa
import keras
from keras import models
from keras.layers import GRU, Dropout, Dense, BatchNormalization, Bidirectional,Flatten
from keras.utils import plot_model
import numpy as np
import sys
import os
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.metrics import classification_report


sys.path.append("D:/code/SER/convGRU/GRU")
from confusion_matrix import cm_plot
from attention_LSTM import Attention_layer
os.environ["PATH"] += os.pathsep + 'D:/软件/Graphviz/bin'
def normalizeVoiceLen(y, normalizedLen):  #normalizeVoiceLen标准化
    nframes = len(y)
    y = np.reshape(y, [nframes, 1]).T
    # 归一化音频长度为2s,32000数据点
    if (nframes < normalizedLen):
        res = normalizedLen - nframes
        res_data = np.zeros([1, res], dtype=np.float32)
        y = np.reshape(y, [nframes, 1]).T
        y = np.c_[y, res_data]
    else:
        y = y[:, 0:normalizedLen]
    return y[0]

#getNearestLen()函数根据声音的采样率确定一个合适的语音帧长用于傅立叶变换
def getNearestLen(framelength, sr):
    framesize = framelength * sr
    # 找到与当前framesize最接近的2的正整数次方
    nfftdict = {}
    lists = [32, 64, 128, 256, 512, 1024]
    for i in lists:
        nfftdict[i] = abs(framesize - i)
    sortlist = sorted(nfftdict.items(), key=lambda x: x[1])  # 按与当前framesize差值升序排列
    framesize = int(sortlist[0][0])  # 取最接近当前framesize的那个2的正整数次方值为新的framesize
    return framesize

import os
import pickle

counter = 0
fileDirCASIA = r'D:\code\casia'

mfccs = {}
mfccs['angry'] = []
mfccs['fear'] = []
mfccs['happy'] = []
mfccs['neutral'] = []
mfccs['sad'] = []
mfccs['surprise'] = []
#mfccs['disgust'] = []

listdir = os.listdir(fileDirCASIA)
for persondir in listdir:
    if (not r'.' in persondir):
        emotionDirName = os.path.join(fileDirCASIA, persondir)
        emotiondir = os.listdir(emotionDirName)
        for ed in emotiondir:
            if (not r'.' in ed):
                filesDirName = os.path.join(emotionDirName, ed)
                files = os.listdir(filesDirName)
                for fileName in files:
                    if (fileName[-3:] == 'wav'):
                        counter += 1
                        fn = os.path.join(filesDirName, fileName)
                        #print(str(counter) + fn)
                        y, sr = librosa.load(fn, sr=None)
                        VOICE_LEN = 32000
                        y = normalizeVoiceLen(y, VOICE_LEN)  # 归一化长度
                        N_FFT = getNearestLen(0.25, sr)
                        mfcc_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=int(N_FFT / 4))
                        stft_coff = abs(librosa.stft(y, 1024, 256, 1024))  # 分帧然后求短时傅里叶变换，分帧参数与对数能量梅尔滤波器组参数设置要相同
                        energy = np.sum(np.square(stft_coff), 0)  # 求每一帧的平均能量
                        MFCC_Energy = np.vstack((mfcc_data, energy))  # 将每一帧的MFCC与短时能量拼接在一起
                        feature = np.mean(MFCC_Energy, axis=0)
                        mfccs[ed].append(feature.tolist())

with open('MFCC_Energy_feature_dict.pkl', 'wb') as f:
    pickle.dump(mfccs, f)

import pickle
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import to_categorical

# 读取特征
mfccs = {}
with open('MFCC_Energy_feature_dict.pkl', 'rb') as f:
    mfccs = pickle.load(f)

# 设置标签
emotionDict = {}
emotionDict['angry'] = 0
emotionDict['fear'] = 1
emotionDict['happy'] = 2
emotionDict['neutral'] = 3
emotionDict['sad'] = 4
emotionDict['surprise'] = 5

data = []
labels = []
data = data + mfccs['angry']
print(len(mfccs['angry']))
for i in range(len(mfccs['angry'])):
    labels.append(0)

data = data + mfccs['fear']
print(len(mfccs['fear']))
for i in range(len(mfccs['fear'])):
    labels.append(1)

print(len(mfccs['happy']))
data = data + mfccs['happy']
for i in range(len(mfccs['happy'])):
    labels.append(2)

print(len(mfccs['neutral']))
data = data + mfccs['neutral']
for i in range(len(mfccs['neutral'])):
    labels.append(3)

print(len(mfccs['sad']))
data = data + mfccs['sad']
for i in range(len(mfccs['sad'])):
    labels.append(4)

print(len(mfccs['surprise']))
data = data + mfccs['surprise']
for i in range(len(mfccs['surprise'])):
    labels.append(5)

print(len(data))
print(len(labels))

# 设置数据维度
data = np.array(data)
data = data.reshape((data.shape[0], data.shape[1], 1))

labels = np.array(labels)
labels = to_categorical(labels)

# 数据标准化
DATA_MEAN = np.mean(data, axis=0)
DATA_STD = np.std(data, axis=0)

data -= DATA_MEAN
data /= DATA_STD
paraDict={}
paraDict['mean']=DATA_MEAN
paraDict['std']=DATA_STD
paraDict['emotion']=emotionDict
with open('MFCC_Energy_model_para_dict.pkl', 'wb') as f:
    pickle.dump(paraDict, f)

ratioTrain = 0.8
numTrain = int(data.shape[0] * ratioTrain)
print(numTrain)
permutation = np.random.permutation(data.shape[0])
data = data[permutation, :]
labels = labels[permutation, :]

x_train = data[:numTrain]
x_val = data[numTrain:]
y_train = labels[:numTrain]
y_val = labels[numTrain:]

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

model = models.Sequential()
model.add(layers.Conv1D(256, 5, activation='relu', input_shape=(126, 1)))
model.add(layers.MaxPooling1D(pool_size=(8)))
model.add(layers.Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling1D(pool_size=(3)))

model.add(GRU(256,input_shape=(None,5,128),init='glorot_uniform', inner_init='orthogonal',
               activation='tanh',
               inner_activation='hard_sigmoid', W_regularizer=regularizers.l2(0.0005),
               U_regularizer=regularizers.l2(0.0005), b_regularizer=regularizers.l2(0.005),
               dropout_W=0., dropout_U=0., return_sequences=False))
model.add(Dense(output_dim=128, init='glorot_uniform',
                W_regularizer=regularizers.l2(0.0005), b_regularizer=regularizers.l2(0.005)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
print(model.output_shape)

model.add(Dense(6))
model.add(Activation('softmax'))
plot_model(model, to_file='mfcc_model.png', show_shapes=True)
model.summary()
plot_model(model, to_file='GRUmodel.png', show_shapes=True)
opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
callbacks_list=[
    keras.callbacks.EarlyStopping(
        monitor='acc',
        patience=50,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='speechMFCC_Energy_model_checkpoint.h5',
        monitor='val_loss',
        save_best_only=True
    ),
    keras.callbacks.TensorBoard(
        log_dir='speechMFCC_Energy_train_log'
    )
]
history=model.fit(x_train, y_train,
                  batch_size=32,
                  epochs=200,
                  validation_data=(x_val, y_val),
                 callbacks=callbacks_list)
#打印分类每类的准确率
Y_test = np.argmax(y_val, axis=1) # Convert one-hot to index这里把onehot转成了整数[1,2,10,1,2,1]
y_pred = model.predict_classes(x_val)#这里假设你的GT标注也是整数 [1,2,10,1,2,1]
print(classification_report(Y_test, y_pred,digits=4))
cm_plot(Y_test, y_pred)
model.save('speech_MFCC_Energy_model.h5')
model.save_weights('speech_MFCC_Energy_model_weight.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

