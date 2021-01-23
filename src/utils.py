from os.path import join

import numpy as np
from PIL import Image, ImageEnhance

import cv2
from matplotlib import pyplot as plt

Image.MAX_IMAGE_PIXELS = None

import os
from os import listdir
from os.path import isfile, join, isdir


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath, input_image_size=512):
    img = Image.open(filepath).convert("RGB")
    img = img.resize((input_image_size, input_image_size), Image.LANCZOS)
    return img


def tensor2image(image_tensor):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    return image_numpy


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


# get full image paths
def get_image_paths(folder):
    image_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    if join(folder, ".DS_Store") in image_paths:
        image_paths.remove(join(folder, ".DS_Store"))
    for path in reversed(image_paths):
        if basename(path)[-4:] != ".jpg" and basename(path)[-4:] != ".png":
            image_paths.remove(path)

    image_paths = sorted(image_paths)
    return image_paths


# get '17asdfasdf2d_0_0.jpg' from 'train_folder/train/o/17asdfasdf2d_0_0.jpg'
def basename(path):
    return path.split("/")[-1]


# get 'train_folder/train/o' from 'train_folder/train/o/17asdfasdf2d_0_0.jpg'
def basefolder(path):
    return "/".join(path.split("/")[:-1])


# just get the name of images in a folder
def get_image_names(folder):
    image_names = [f for f in listdir(folder) if isfile(join(folder, f))]
    if ".DS_Store" in image_names:
        image_names.remove(".DS_Store")

    for name in image_names:
        if name[-4:] != ".jpg" or name[-4:] != ".png":
            image_names.remove(name)
    image_names = sorted(image_names)
    return image_names


# get full image paths
def get_image_paths(folder):
    image_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    if join(folder, ".DS_Store") in image_paths:
        image_paths.remove(join(folder, ".DS_Store"))
    for path in reversed(image_paths):
        if basename(path)[-4:] != ".jpg" and basename(path)[-4:] != ".png":
            image_paths.remove(path)

    image_paths = sorted(image_paths)
    return image_paths


# get subfolders
def get_subfolder_paths(folder):
    subfolder_paths = [
        join(folder, f)
        for f in listdir(folder)
        if (isdir(join(folder, f)) and f[0] != ".")
    ]
    if join(folder, ".DS_Store") in subfolder_paths:
        subfolder_paths.remove(join(folder, ".DS_Store"))
    subfolder_paths = sorted(subfolder_paths)
    return subfolder_paths


# create an output folder if it does not already exist
def confirm_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


def cv2pil(image):
    """ OpenCV型 -> PIL型 """
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def pil2cv(image):
    """ PIL型 -> OpenCV型 """
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def load_image(image_path):
    if image_path[-3:] == "npy":
        image = np.load(image_path)
    else:
        image = Image.open(image_path)
        image = np.array(image, dtype=np.uint8)

    if image.dtype != "uint8":
        image = image.astype("uint8")
    if image.shape[2] == 4:
        image = image[:, :, 0:3]
    return image


#         image = cv2.imread(image_path)
#         return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_image_size(image_path):
    return Image.open(image_path, mode="r").size


def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]


# getting the classes for classification
def get_classes(folder):
    subfolder_paths = sorted(
        [
            f
            for f in listdir(folder)
            if (isdir(join(folder, f)) and ".DS_Store" not in f)
        ]
    )
    return subfolder_paths


def get_Wighted_for_sampler_from_train_path(path, num_sampler=None):
    label = []
    count = []

    for disease_name in config.classes:
        image_pahts = get_image_paths(os.path.join(path, disease_name))
        print(f"class 【{disease_name}】has {len(image_pahts)} images")

        label.append(disease_name)
        count.append(len(image_pahts))

    class_weight = []
    for c in count:
        if c:
            class_weight.append(sum(count) / c)
        else:
            class_weight.append(int(0))

    weight_for_sampler = []
    for i, disease_name in enumerate(config.classes):
        image_pahts = get_image_paths(os.path.join(path, disease_name))
        weight_for_sampler.extend([class_weight[i]] * len(image_pahts))

    return sorted(count, reverse=True), weight_for_sampler


# padding for small crops
def padding(image, patch_size):
    """
    いただいた画像の端の領域は正方形じゃない。
    画像を正方形に戻す関数。
    """

    x = image.shape[0]  # get current x and y of image
    y = image.shape[1]
    if x >= patch_size and y >= patch_size:
        return image  # if its already big enough, then do nothing

    x_new = max(x, patch_size)
    y_new = max(y, patch_size)
    new_image = np.ones((x_new, y_new, 3)) * 255
    new_image = new_image.astype(np.int16)
    #     x_start = int(x_new/2 - x/2)
    #     y_start = int(y_new/2 - y/2) #find where to place the old image
    x_start = 0
    y_start = 0
    new_image[
        x_start : x_start + x, y_start : y_start + y, :
    ] = image  # place the old image

    return new_image  # return the padded image
