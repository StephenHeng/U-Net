from torchvision import datasets, transforms
from base import BaseDataLoader
from Data.VodeoDataset import VideoDataset
import os
from torch.utils.data import DataLoader
from skimage import io, transform
import torch
import numpy as np
import torch.nn.functional as F
# class MnistDataLoader(BaseDataLoader):
# #     """
# #     MNIST data loading demo using BaseDataLoader
# #     """
# #     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
# #         trsfm = transforms.Compose([
# #             transforms.ToTensor(),
# #             transforms.Normalize((0.1307,), (0.3081,))
# #         ])
# #         self.data_dir = data_dir
# #         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
# #         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
class ClipSubstractMean(object):
    def __init__(self, b=104, g=117, r=123):
        self.means = np.array((r, g, b))

    def __call__(self, sample):
        video_x, video_label = sample['video_x'], sample['video_label']
        new_video_x = video_x - self.means
        return {'video_x': new_video_x, 'video_label': video_label}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size=(182, 242)):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        video_x, video_label = sample['video_x'], sample['video_label']

        h, w = video_x.shape[1], video_x.shape[2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        new_video_x = np.zeros((16, new_h, new_w, 3))
        for i in range(16):
            image = video_x[i, :, :, :]
            img = transform.resize(image, (new_h, new_w))
            new_video_x[i, :, :, :] = img
        return {'video_x': new_video_x, 'video_label': video_label}



class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(160, 160)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        video_x, video_label = sample['video_x'], sample['video_label']
        # video_x = sample['video_x']
        h, w = video_x.shape[1], video_x.shape[2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        new_video_x = np.zeros((16, new_h, new_w, 3))
        for i in range(16):
            image = video_x[i, :, :, :]
            image = image[top: top + new_h, left: left + new_w]
            new_video_x[i, :, :, :] = image

        return {'video_x': new_video_x, 'video_label': video_label}
        # return {'video_x': new_video_x}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        video_x, video_label = sample['video_x'], sample['video_label']
        # video_x = sample['video_x']

        # swap color axis because
        # numpy image: batch_size x H x W x C
        # torch image: batch_size x C X H X W
        video_x = video_x.transpose((3, 0, 1, 2))
        video_x = np.array(video_x)
        # video_label = [video_label]
        video_label = torch.tensor(video_label-1)
        return {'video_x': torch.from_numpy(video_x), 'video_label': video_label}

class MyDataLoader(BaseDataLoader):
    def __init__(self, data_dir, num_class, batch_size, shuffle=True, validation_split=0.0, num_workers=0):
        self.data_dir = data_dir
        info_list = r"E:\github\video\pytorch-template-master-UCF50-part2/test.txt"
        self.dataset = VideoDataset(info_list, num_class, self.data_dir,
                         transform=transforms.Compose([ClipSubstractMean(), Rescale(), RandomCrop(), ToTensor()]))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

