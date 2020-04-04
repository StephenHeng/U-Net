import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import json

class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset_path, split='test1', clip_len=16, preprocess=True):
        self.root_dir = dataset_path
        self.output_dir = r"E:\dataset\test1"
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 320
        self.resize_width = 240
        self.crop_size = 112

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        if preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset_path))
            self.preprocess()

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    # def check_preprocess(self):
    #     # TODO: Check image size in output_dir
    #     if not os.path.exists(self.output_dir):
    #         return False
    #     elif not os.path.exists(os.path.join(self.output_dir, 'train')):
    #         return False
    #
    #     for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
    #         for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
    #             video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
    #                                 sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
    #             image = cv2.imread(video_name)
    #             if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
    #                 return False
    #             else:
    #                 break
    #
    #         if ii == 10:
    #             break

    #     return True

    def preprocess(self):
        output_dir = os.path.join(self.output_dir, 'test1')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        json_dir = r"E:\github\video\pytorch-template-master-UCF50-part2/test1.json"
        with open(json_dir, 'r') as load_f:
            load_dict = json.load(load_f)
           # print(load_dict)
        # Split train/val/test sets
        # for file in os.listdir(self.root_dir):
        #     file_path = os.path.join(self.root_dir, file)
        #     video_files = [name for name in os.listdir(file_path)]
        for k in load_dict:
            video_files = os.path.join(self.root_dir, load_dict[k], k)
            # print(k, load_dict[k])

            #
            # train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            # train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'test1', load_dict[k])
            # val_dir = os.path.join(self.output_dir, 'val', file)
            # test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            self.process_video(video_files, k, load_dict[k], train_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, img_label, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = action_name.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        # cv2.VideoCapture打开视频文件
        capture = cv2.VideoCapture(os.path.join(self.root_dir, img_label, action_name))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        # EXTRACT_FREQUENCY = 4
        # if frame_count // EXTRACT_FREQUENCY <= 16:  # //运算符表示向下取整
        #     EXTRACT_FREQUENCY -= 1
        #     if frame_count // EXTRACT_FREQUENCY <= 16:
        #         EXTRACT_FREQUENCY -= 1
        #         if frame_count // EXTRACT_FREQUENCY <= 16:
        #             EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            # if count % EXTRACT_FREQUENCY == 0:
            if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
            i += 1
        count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

if __name__ == "__main__":
    data_dir = r"E:\dataset\UCF50"
    train_data = VideoDataset(data_dir, split='test1', clip_len=8, preprocess=True)
