import os
import shutil
from shutil import copy
data_dir = r"E:\dataset\test1\test1"
labels = os.listdir(data_dir)
m=0
for label in labels:
    video_names = os.listdir(os.path.join(data_dir, label))
    # print(video_names)
    # for video_name in video_names:
    #     images = os.listdir(os.path.join(data_dir, label, video_name))
    #     m = len(images)
    #     file_path = os.path.join(data_dir, label, video_name,'n_frames')
    #     f = open(file_path, 'w')
    #     f.writelines(str(m))
    #     f.close()
    m = m+1
    for i in range(len(video_names)):
        f = open("test1.txt", 'a')
        # strcontent = label + video_names[i] + str(m)
        # f.writelines(str(id + 1) + ' ' + label + '\n')
        f.writelines(label + '/' + video_names[i] + ' ' + str(m) + '\n')
