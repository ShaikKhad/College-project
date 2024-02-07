# download the food dataset (and untar)
# !wget -P data http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
# !tar -xzf data/food-101.tar.gz -C data

# Name of the directory
data_dir = "data/food-101"

import json

from os import mkdir
from os import listdir

from shutil import copyfile
from random import shuffle
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
# %matplotlib inline

def raed_txt(file_name):
    with open(file_name, 'r') as f:
        return [l.strip() for l in f]

def read_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

# read meta files
train_json = read_json(data_dir+"/meta/train.json")
test_json = read_json(data_dir+"/meta/test.json")
label_imgs = raed_txt(data_dir+"/meta/classes.txt")


def plot_counts(json_file, set_name):
    labels, counts = [], []

    for k, v in train_json.items():
        labels.append(k)
        counts.append(len(v))

    plt.figure(figsize=(20, 10))
    plt.bar(labels, counts)

    plt.title("Image counts for the {} set".format(set_name))
    plt.ylabel("Image count")
    plt.xlabel("Category")
    plt.xticks(rotation='vertical')

    plt.show()
plot_counts(train_json, "training")

plot_counts(test_json, "test")
# add numbers to each folder
label_imgs = {l:"{:03d}.{}".format(i, l) for i, l in enumerate(label_imgs)}
mkdir("data/train")
mkdir("data/valid")
mkdir("data/test")
# split the original training set into train and valid dir
for k in tqdm(train_json.keys()):
    # generate random indexs to decide train or valid
    rand_idx = [i for i in range(len(train_json[k]))]
    shuffle(rand_idx)

    # split 80-20, train valid
    n_train = int(len(train_json[k]) * 0.8)

    # copy to training
    mkdir("data/train/" + label_imgs[k])
    for i in range(n_train):
        src = "data/food-101/images/{}.jpg".format(train_json[k][i])
        dst = "data/train/{}/{}.jpg".format(label_imgs[k], train_json[k][i].replace("/", "_"))
        copyfile(src, dst)

    # copy to validation
    mkdir("data/valid/" + label_imgs[k])
    for i in range(n_train, len(rand_idx)):
        src = "data/food-101/images/{}.jpg".format(train_json[k][i])
        dst = "data/valid/{}/{}.jpg".format(label_imgs[k], train_json[k][i].replace("/", "_"))
        copyfile(src, dst)

# copy test images into test dir
for k in tqdm(test_json.keys()):
    mkdir("data/test/"+label_imgs[k])
    for i in range(len(test_json[k])):
        src = "data/food-101/images/{}.jpg".format(test_json[k][i])
        dst = "data/test/{}/{}.jpg".format(label_imgs[k], test_json[k][i].replace("/", "_"))

# Delete original img data (to save space)
# !rm -r data/food-101/images

# make a list of all training images
all_train_imgs = []
all_test_imgs = []
for v in label_imgs.values():
    train_dir = "data/train/{}".format(v)
    all_train_imgs += [train_dir + "/" + d for d in listdir(train_dir)]

    test_dir = "data/test/{}".format(v)
    all_test_imgs += [test_dir + "/" + d for d in listdir(test_dir)]

shuffle(all_train_imgs)
shuffle(all_test_imgs)

def show_imgs(img_list, nrows=5, ncols=7):
    """
    show multiple images in a tile format.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(20,17))
    for idx, ax in enumerate(axes.flat):
        img = cv2.imread(img_list[idx])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)

        ax.set_title(img_list[idx].split("/")[-2])
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.grid(False)

show_imgs(all_train_imgs)
show_imgs(all_test_imgs)
