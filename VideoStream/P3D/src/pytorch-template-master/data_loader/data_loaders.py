from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import torch
import numpy as np
import json
import os
import random
from PIL import Image
import json

class BreastDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, input_shape, num_seg_class, temporal_channel, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.CenterCrop((400, 800)),
            transforms.Resize((input_shape[0], input_shape[1])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.data_dir = data_dir
        self.dataset = BreastDataset(
            self.data_dir, input_shape, num_seg_class, temporal_channel, train=training, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers, collate_fn)


def collate_fn(batch):
    imgs = []
    label = []
    order = []
    # order_swap = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        imgs.append(sample[2])
        order.append(torch.tensor(sample[1], dtype=torch.long))
        order.append(torch.tensor(sample[3], dtype=torch.long))
        label.append(sample[4])
        label.append(sample[4])
        # if sample[3] == -1:
        #     label_swap.append(1)
        #     label_swap.append(0)
        # else:
        #     label_swap.append(sample[2])
        #     label_swap.append(sample[3])
        # law_swap.append(sample[4])
        # law_swap.append(sample[5])
        img_name.append(sample[5])
    return torch.stack(imgs, 0), torch.stack(label, 0), torch.stack(order, 0), img_name


class BreastDataset(Dataset):
    def __init__(self, root_dir, input_shape, num_seg_class, temporal_channel, train, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.input_shape = input_shape
        self.temporal_chanel = temporal_channel
        self.select_all = {}
        self.num_seg_class = num_seg_class
        if train:
            self.data = self.read_data(os.path.join(
                self.root_dir, "ann", "train.json"))
        else:
            self.data = self.read_data(os.path.join(
                self.root_dir, "ann", "val.json"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data_item_info = self.data[idx]
        label = data_item_info['class_name']
        # labels = np.zeros(self.num_seg_class)
        #         # labels[int(label)] = 1
        # file_list = data_item_info['data']['all']
        file_list = data_item_info['data']['all']
        # file_list.sort(key=lambda x: int(x[:-4]))

        if idx not in self.select_all.keys():
            select_item = np.random.beta(
                3, 6, self.temporal_chanel)*len(file_list)
            self.select_all[idx] = sorted(select_item)
        select_file = self.select_all[idx]
        label = torch.tensor(int(label), dtype=torch.long)


        image_all = []
        image_order = []
        image_all_swap = []
        image_order_swap = []
        for i, item in enumerate(select_file):

            img_path = os.path.join(self.root_dir, "data",
                                    file_list[int(item)])
            if self.input_shape[2] == 3:
                # image = cv2.imread(img_path)
                image = Image.open(img_path).convert('RGB')
                # image_all = self.img_proced(image)
            elif self.input_shape[2] == 1:
                image = cv2.imread(img_path, 2)
                # image = self.img_proced(image)
                image = np.expand_dims(image, 2)

            if self.transform:
                image = self.transform(image)
            image_all.append(image)
            image_order.append(i)
            image_order_swap.append(i)
        image_order_swap = self.shuffle_order(image_order_swap)
        image_all_swap = [image_all[i].clone() for i in image_order_swap]

        image_all = torch.stack(image_all, 1)
        # image_order = self.cata_to_one_hot(image_order, 16)

        image_all_swap = torch.stack(image_all_swap, 1)
        # image_order_swap = self.cata_to_one_hot(image_order_swap, 16)
        return image_all, image_order, image_all_swap, image_order_swap, label, file_list[int(item)]

        # return images, labels
    def cata_to_one_hot(self, label, class_num):

        batch_size = len(label)
        label = [[x] for x in label]
        label = torch.LongTensor(label)
        one_hot = torch.zeros(batch_size, class_num).scatter_(1, label, 1)
        return one_hot

    def read_data(self, file_path):
        return json.load(open(file_path))

    def shuffle_order(self, data):
        tmpx = []
        count_x = 0
        k = 1
        RAN = 2
        for i in data:
            tmpx.append(i)
            count_x += 1
            if len(tmpx) >= k:
                tmp = tmpx[count_x - RAN:count_x]
                random.shuffle(tmp)
                tmpx[count_x - RAN:count_x] = tmp
        return tmpx
