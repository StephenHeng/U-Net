import json
import os
import math
train_json_text={}
test_json_text={}
# train_data = r"E:\dataset\UCF50"
# train_dir_1 = "E:\MyNewProject\dataset\mask_01_data/train/1"
test_data = r"E:\dataset\UCF50"
# test_dir_1 = "E:\MyNewProject\dataset\mask_01_data/test/1"
# def train_json_append(dir):
#     files = os.listdir(train_data)
#     label=files
#     # print(label)
# def train_label_match(train_data):
#     files = os.listdir(train_data)
#     for file in files:
#         videos = os.listdir(os.path.join(train_data, file))
#         for video in videos:
#             train_json_text[video] =file
def train_data_list(train_data):
    files = os.listdir(train_data)
    for file in files:
        videos = os.listdir(os.path.join(train_data, file))
        video_len = len(videos)
        train_video = math.ceil(video_len*0.7)
        for i in range(train_video):
            train_json_text[videos[i]] = file
        for i in range(train_video, video_len):
            test_json_text[videos[i]] = file

# def test_json_append(dir,label):
#     for root, dirs, files in os.walk(dir):
#         for file in files:
#             test_json_text[file] = label

# train_json_append(train_dir_0, 0)
# train_json_append(train_dir_1, 1)
# test_json_append(test_dir_0, 0)
# test_json_append(test_dir_1, 1)
# jsondata_train = json.dumps(train_json_text)
# jsondata_test = json.dumps(test_jsontext)
# # f = open("test.json", 'w')
# # f.write(jsondata_test)
# f.close()

if __name__=='__main__':
    # train_json_append(train_data)
    # train_label_match(train_data)
    # print(train_json_text)
    # train_data_list(train_data)
    # jsondata_train = json.dumps(train_json_text)
    # f = open("tran.json", 'w')
    # f.write(jsondata_train)
    jsondata_test = json.dumps(test_json_text)
    f = open("test1.json", 'w')
    f.write(jsondata_test)
    # print(len(train_json_text))
    print(len(test_json_text))

