from math import log
import operator
import os
import json
#获取良恶性文件名
def get_filename(file_dir):
    temp=os.listdir(file_dir)
    list1=[]
    list2=[]
    for item in temp:
        #print(item)
        list1=list(item.split('.'))
        list2.append(list1[0])
    return list2
def get_label(filename):
    if filename in benign:
        return 0
    if filename in malignant:
        return 1
#将json文件转化为list
def read_dataset(filename):
    with open(filename, 'r') as f:
        temp = json.loads(f.read())
        for item in temp.items():
            temp1 = list(item)
            if len(temp1[1]) != 0:
                label = get_label(temp1[0])
                if label is not None:
                    del temp1[0]
                    temp1[0].append(label)
                    datasets.append(temp1[0])

filename = 'E:\decision_tree_project\dataset\shapes.json'
benign_dir = 'E:\decision_tree_project\dataset\mask_01_data/train\pic0'
malignant_dir = 'E:\decision_tree_project\dataset\mask_01_data/train\pic1'
labels = ['凹凸性', '纵横比', '致密性', '圆差异性', '椭圆差异性']
benign = []
malignant = []
datasets = []
benign=get_filename(benign_dir)
malignant = get_filename(malignant_dir)
read_dataset(filename)





