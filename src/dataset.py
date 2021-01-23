from os.path import join

import torch.utils.data as data
from PIL import Image

from src.utils import get_image_paths
import albumentations as A
import numpy as np
import cv2
import pandas as pd
import torchvision.transforms as T

from src.utils import *


class DatasetFromFolder(data.Dataset):
    """
    Datasetで画像のロード・argumentation, データセット受け渡しの全てが行われている。
    """

    def __init__(self, image_folder):

        self.transform = A.Compose(
            [
                A.Flip(p=0.5),
                A.RandomRotate90(),
                A.ShiftScaleRotate(),
                A.RandomCrop(height=256, width=256, p=1),
            ]
        )

        self.transform_color = A.Compose([A.ToGray(p=1.0)])
        self.transform_to_tensor = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        self.files = get_image_paths(image_folder)

    def __getitem__(self, index):
        image = Image.open(self.files[index])

        image_np = np.array(image)

        image = self.transform(image=image_np)

        gray_image = self.transform_color(image=image["image"])

        image_pil = Image.fromarray(image["image"])
        gray_image_pil = Image.fromarray(gray_image["image"])

        image_tensor = self.transform_to_tensor(image_pil)
        gray_image_tensor = self.transform_to_tensor(gray_image_pil)

        return image_tensor, gray_image_tensor

    def __len__(self):
        return len(self.files)
