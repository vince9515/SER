import os
import pickle
import librosa
import sys
import numpy as np
fileDirCASIA = r'D:\code\test\CASIA_noise'
import sys

from keras.optimizers import RMSprop

sys.path.append("D:/code/SER/")

os.environ["PATH"] += os.pathsep + 'D:/软件/Graphviz/bin'

import matplotlib.pyplot as plt


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
'''
y, sr = librosa.load(fileDirCASIA, sr=None) # y是数字音频序列
VOICE_LEN = 32000
# 获得N_FFT的长度
N_FFT = getNearestLen(0.25, sr)
# 统一声音范围为前两秒
y = normalizeVoiceLen(y, VOICE_LEN)
'''
counter = 0


mfccs = {}
mfccs['angry'] = []
mfccs['fear'] = []
mfccs['happy'] = []
mfccs['neutral'] = []
mfccs['sad'] = []
mfccs['surprise'] = []
#mfccs['disgust'] = []

listdir = os.listdir(fileDirCASIA)
VOICE_LEN = 32000

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
                        N_FFT = getNearestLen(0.25, sr)
                        y = normalizeVoiceLen(y, VOICE_LEN)  # 归一化长度
                        mfcc_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=int(N_FFT / 4))
                        stft_coff = abs(librosa.stft(y, 1024, 256, 1024))    # 分帧然后求短时傅里叶变换，分帧参数与对数能量梅尔滤波器组参数设置要相同
                        energy = np.sum(np.square(stft_coff), 0)            # 求每一帧的平均能量
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

import keras
from keras.utils import plot_model
from keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix

from keras_self_attention import SeqSelfAttention
''''
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
class Metrics(Callback):
  def on_train_begin(self, logs={}):
    self.val_f1s = []
    self.val_recalls = []
    self.val_precisions = []

  def on_epoch_end(self, epoch, logs={}):
    val_predict=(np.asarray(self.model.predict(self.model.validation_data[0]))).round()
    val_targ = self.model.validation_data[1]
    _val_f1 = f1_score(val_targ, val_predict)
    _val_recall = recall_score(val_targ, val_predict)
    _val_precision = precision_score(val_targ, val_predict)
    self.val_f1s.append(_val_f1)
    self.val_recalls.append(_val_recall)
    self.val_precisions.append(_val_precision)
    print('-val_f1: %.4f --val_precision: %.4f --val_recall: %.4f'%(_val_f1, _val_precision, _val_recall))
    return

metrics = Metrics()
'''

model = models.Sequential()
#model.add(SeqSelfAttention(attention_activation='sigmoid'))
model.add(layers.Conv1D(256, 5, activation='relu', input_shape=(126, 1)))
model.add(layers.Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling1D(pool_size=(8)))
model.add(layers.Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling1D(pool_size=(3)))
model.add(layers.Conv1D(256, 5, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.1))
model.add(layers.Flatten())
#尝试添加GRU
#model.add(layers.GRU(32,input_shape=(None,1280)))
#model.add(layers.Dense(1))
#model.compile(optimizer=RMSprop(), loss='mae')

model.add(layers.Dense(6, activation='softmax'))

plot_model(model, to_file='mfcc_model.png', show_shapes=True)
model.summary()
opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
#model.compile(loss='categorical_squared_hinge', optimizer=opt,metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['_val_f1', '_val_precision', '_val_recall'])
import keras

callbacks_list=[
    keras.callbacks.EarlyStopping(
        monitor='acc',
        patience=50,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='MFCC_Energy_model_checkpoint.h5',
        monitor='val_loss',
        save_best_only=True
    ),
    keras.callbacks.TensorBoard(
        log_dir='speechMFCC_Energy_train_log'
    ),

    keras.callbacks.EarlyStopping(monitor='acc',min_delta=0.002,
                              patience=0, verbose=0, mode='auto',
                              baseline=None, restore_best_weights=False)
]
history=model.fit(x_train, y_train,
                  batch_size=16,
                  epochs=200,
                  validation_data=(x_val, y_val),
                 callbacks=callbacks_list)
#打印分类每类的准确率
Y_test = np.argmax(y_val, axis=1) # Convert one-hot to index这里把onehot转成了整数[1,2,10,1,2,1]
y_pred = model.predict_classes(x_val)#这里假设你的GT标注也是整数 [1,2,10,1,2,1]
print(classification_report(Y_test, y_pred))


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


'''
迭代次数为130，Dropout=0.3时，测试集表现最好为57.50%
迭代次数为130,Dropout=0.4时，测试集表现最好为61.25%
迭代次数为130，全连接层使用dropout=0.4时，使用批次规范化最好效果为58.75%
迭代次数为130，全连接层亦使用批次规范化，最好效果为49.17%
迭代次数为130，，Dropout=0.5时，测试集最好效果为59.58%
'''

from keras.models import load_model
import pickle

model = load_model('speech_MFCC_Energy_model.h5')
paradict = {}
with open('MFCC_Energy_model_para_dict.pkl', 'rb') as f:
    paradict = pickle.load(f)
DATA_MEAN = paradict['mean']
DATA_STD = paradict['std']
emotionDict = paradict['emotion']
edr = dict([(i, t) for t, i in emotionDict.items()])
import librosa

filePath = 'D:/code/SER/record1.wav'
y, sr = librosa.load(filePath, sr=None)
y = normalizeVoiceLen(y, VOICE_LEN)  # 归一化长度
mfcc_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=int(N_FFT / 4))
feature = np.mean(mfcc_data, axis=0)
feature = feature.reshape((126, 1))
feature -= DATA_MEAN
feature /= DATA_STD
feature = feature.reshape((1, 126, 1))
result = model.predict(feature)
index = np.argmax(result, axis=1)[0]
print(edr[index])
