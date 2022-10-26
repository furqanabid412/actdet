import numpy as np
import os
from keras.utils import to_categorical
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class create_dataset():
    def __init__(self,dataset_dir= "E:\Research\ActionDetection\dataset\RLVSdataset\Real Life Violence Dataset/",
                 frames=40,img_height=64,img_width=64,
                 class_list = ['NonViolence', 'Violence'],split=0.2):

        self.dataset_dir = dataset_dir
        self.frames = frames
        self.img_height = img_height
        self.img_width = img_width
        self.class_list = class_list
        self.split = split
        self.create_dataset()

    def frames_extraction(self,video_path):
        frames_list = []
        video_reader = cv2.VideoCapture(video_path)
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames_window = max(int(video_frames_count / self.frames), 1)
        for frame_counter in range(self.frames):
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
            success, frame = video_reader.read()
            if not success:
                break
            resized_frame = cv2.resize(frame, (self.img_height, self.img_width))
            normalized_frame = resized_frame / 255
            frames_list.append(normalized_frame)
        video_reader.release()
        return frames_list

    def create_dataset(self):
        data = []
        labels = []
        video_files_paths = []
        for class_index, class_name in enumerate(self.class_list):

            print(f'Extracting Data of Class: {class_name}')
            files_list = os.listdir(os.path.join(self.dataset_dir, class_name))
            for file_name in tqdm(files_list):
                video_file_path = os.path.join(self.dataset_dir, class_name, file_name)
                frames = self.frames_extraction(video_file_path)
                if len(frames) == self.frames:
                    data.append(frames)
                    labels.append(class_index)
                    video_files_paths.append(video_file_path)

        data = np.asarray(data)
        labels = np.array(labels)

        one_hot_encoded_labels = to_categorical(labels)

        if self.split > 0.0 :
            train_set, test_set, train_labels, test_labels = train_test_split(data, one_hot_encoded_labels,
                                                                          test_size=self.split, shuffle=True,
                                                                          random_state=42)
        # Saving the extracted data
        np.save("datasets/HockeyFight/train/data.npy", train_set)
        np.save("datasets/HockeyFight/train/labels.npy", train_labels)
        np.save("datasets/HockeyFight/test/data.npy", test_set)
        np.save("datasets/HockeyFight/test/labels.npy", test_labels)
        print('saved train-test set')


if __name__ == '__main__':
    create_dataset(dataset_dir='E:\Research\ActionDetection\dataset/hockeyfight/data/', frames=40
                   , img_height=64, img_width=64, class_list=['fight', 'nofight'], split=0.2)