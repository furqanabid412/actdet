import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import matplotlib.pyplot as plt
import wandb
from keras.utils import to_categorical

from keras.applications.mobilenet_v2 import MobileNetV2
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import load_model
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import tensorflow as tf
from collections import deque
import random
import pickle
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import cv2
from sklearn.metrics import accuracy_score

################################################################

class dataset(Dataset):
    def __init__(self,dataset='RLVD',trainset=True):
        self.trainset=trainset
        if trainset:
            self.data,self.labels = np.load(os.path.join('datasets',dataset,'train','data.npy')),\
                                    np.load(os.path.join('datasets',dataset,'train','labels.npy'))
        else:
            self.data,self.labels = np.load(os.path.join('datasets',dataset,'test','data.npy')),\
                                    np.load(os.path.join('datasets',dataset,'test','labels.npy'))

    def __len__(self):
        return len(self.data[:,0,0,0])

    def get_data(self):
        return self.data,self.labels

    def __getitem__(self, index):
        return self.data[index],self.labels[index]


################################################################

def mobileNet_BiLSTM(sequence_length, img_h,img_w,classes):
    mobilenet = MobileNetV2(include_top=False, weights="imagenet")
    # Fine-Tuning to make the last 40 layer trainable
    mobilenet.trainable = True

    print(len(mobilenet.layers))
    for layer in mobilenet.layers[:-40]:
        # print('name',layer.name)
        layer.trainable = False

    model = Sequential()
    # Specifying Input to match features shape
    model.add(Input(shape=(sequence_length, img_h, img_w, 3)))
    # Passing mobilenet in the TimeDistributed layer to handle the sequence
    model.add(TimeDistributed(mobilenet))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Flatten()))
    lstm_fw = LSTM(units=32)
    lstm_bw = LSTM(units=32, go_backwards=True)
    model.add(Bidirectional(lstm_fw, backward_layer=lstm_bw))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(len(classes), activation='softmax'))
    model.summary()
    return model

################################################################

def plot_history(history):

    # plotting the loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # plotting the accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

################################################################

def train_and_save(train_data,train_labels,test_data,test_labels,dataset_name,classes,epochs):
    _, sequence_length, img_h, img_w, _ = train_data.shape
    model = mobileNet_BiLSTM(sequence_length, img_h, img_w, classes)
    early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=5, min_lr=0.00005,
                                                     verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])
    MobBiLSTM_model_history = model.fit(x=train_data, y=train_labels, epochs=epochs, batch_size=8, shuffle=True,
                                        validation_split=0.2, callbacks=[early_stopping_callback, reduce_lr])

    save_path = 'saved/{}/epoch{}.keras'.format(dataset_name,len(MobBiLSTM_model_history.history['loss']))
    model.save(save_path)
    print('model saved successfully in {}'.format(save_path))

    plot_history(MobBiLSTM_model_history)

    save_path = 'saved/{}/history/train_history'.format(dataset_name)
    with open(save_path, 'wb') as file_pi:
        pickle.dump(MobBiLSTM_model_history.history, file_pi)

    model_evaluation_history = model.evaluate(test_data, test_labels)
    print('model history successfully in {}'.format(save_path))
    return MobBiLSTM_model_history,model_evaluation_history

def load_saved_model(path):
    return load_model(path)

def load_and_calc_accu(path,test_data,test_labels):
    model = load_saved_model(path)
    labels_predict = model.predict(test_data)
    labels_predict = np.argmax(labels_predict, axis=1)
    labels_test_normal = np.argmax(test_labels, axis=1)
    AccScore = accuracy_score(labels_predict, labels_test_normal)
    print('Accuracy Score is : ', AccScore)
    return model

################################################################

def predict_frames(video_file_path, output_file_path, SEQUENCE_LENGTH,model,IMAGE_HEIGHT,IMAGE_WIDTH,CLASSES_LIST):
    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)
        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]
        if predicted_class_name == "Violence":
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 12)
        else:
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 12)
        video_writer.write(frame)
    video_reader.release()
    video_writer.release()


def show_pred_frames(pred_video_path,SEQUENCE_LENGTH):
    plt.figure(figsize=(20, 12))
    video_reader = cv2.VideoCapture(pred_video_path)
    frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    random_range = sorted(random.sample(range(SEQUENCE_LENGTH, frames_count), 12))
    for counter, random_index in enumerate(random_range, 1):
        plt.subplot(5, 4, counter)
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, random_index)
        ok, frame = video_reader.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax = plt.imshow(frame)
        # ax.figure.set_size_inches(20, 20)
        plt.tight_layout()

    plt.show()
    video_reader.release()

def reverse_labels(labels):
    reversed_labels = np.zeros_like(labels)
    reversed_labels[:, 0] = labels[:, 1]
    reversed_labels[:, 1] = labels[:, 0]
    return reversed_labels

################################################################

def datasets_combine():
    dataset_name = 'RLVD'  # 1.'RLVD' 2.'RWF' 3.HockeyFight
    # classes = ['NonViolence', 'Violence']  # 1. ['NonViolence','Violence'] 2.['Fight','NonFight'] 3. ['fight','nofight']

    train_data1, train_labels1 = dataset(dataset=dataset_name, trainset=True).get_data()
    test_data1, test_labels1 = dataset(dataset=dataset_name, trainset=False).get_data()

    dataset_name = 'RWF'
    train_data2, train_labels2 = dataset(dataset=dataset_name, trainset=True).get_data()
    test_data2, test_labels2 = dataset(dataset=dataset_name, trainset=False).get_data()
    train_labels2, test_labels2 = reverse_labels(train_labels2), reverse_labels(test_labels2)

    dataset_name = 'HockeyFight'
    train_data3, train_labels3 = dataset(dataset=dataset_name, trainset=True).get_data()
    test_data3, test_labels3 = dataset(dataset=dataset_name, trainset=False).get_data()
    train_labels3, test_labels3 = reverse_labels(train_labels3), reverse_labels(test_labels3)

    return np.vstack((train_data1,train_data2,train_data3)),\
           np.vstack((train_labels1,train_labels2,train_labels3)),\
           np.vstack((test_data1,test_data2,test_data3)),\
           np.vstack((test_labels1,test_labels2,test_labels3)),\
           'combined',['NonViolence','Voilence']


################################################################
if __name__ == '__main__':
    epochs = 100

    train_data,train_labels,test_data,test_labels,dataset_name,classes= datasets_combine()

    # dataset_name = 'RLVD' # 1.'RLVD' 2.'RWF' 3.HockeyFight
    # classes = ['NonViolence','Violence'] # 1. ['NonViolence','Violence'] 2.['Fight','NonFight'] 3. ['fight','nofight']
    # train_data, train_labels = dataset(dataset=dataset_name,trainset=True).get_data()
    # test_data, test_labels = dataset(dataset=dataset_name,trainset=False).get_data()
    # train_labels,test_labels = reverse_labels(train_labels),reverse_labels(test_labels)
    train_and_save(train_data, train_labels, test_data, test_labels, dataset_name, classes, epochs)

    model=load_and_calc_accu('saved/HockeyFight/epoch25.keras', test_data, test_labels)
    model = load_saved_model('saved/RLVD/epoch37.keras')
    plt.style.use("default")
    # Construct the output video path.
    test_videos_directory = 'test_videos'
    os.makedirs(test_videos_directory, exist_ok=True)
    output_video_file_path = f'{test_videos_directory}/Output-Test-Video.mp4'


    # input_video_file_path = "E:\Research\ActionDetection\dataset\RLVSdataset\Real Life Violence Dataset\Violence/V_378.mp4"
    # input_video_file_path = "E:\Research\ActionDetection\dataset\RLVSdataset\Real Life Violence Dataset/NonViolence/NV_95.mp4"
    # input_video_file_path = "E:\Research\ActionDetection\dataset\RWF/val/NonFight/1W8hsVvyKt4_1.avi"
    input_video_file_path = "E:\Research\ActionDetection\dataset\RWF/val/Fight/RTHTRS_720.avi"

    _, sequence_length, img_h, img_w, _ = train_data.shape
    predict_frames(input_video_file_path, output_video_file_path, sequence_length,model,img_h, img_w,classes)
    show_pred_frames(output_video_file_path,sequence_length)


    print('s')

