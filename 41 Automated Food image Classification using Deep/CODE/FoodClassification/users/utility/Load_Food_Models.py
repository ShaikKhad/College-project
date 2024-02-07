import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from django.conf import settings
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(299, 299))
    # convert PIL.Image.Image type to 3D tensor with shape (299, 299, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 299, 299, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


label_path = os.path.join(settings.MEDIA_ROOT, 'models', 'labels.txt')
with open(label_path, 'r') as f:
    food101 = [l.strip().lower() for l in f]

# weights_path = "food101_final_model.h5" # orginal weights converted from caffe
weights_path = os.path.join(settings.MEDIA_ROOT, "models", "food101_final_model.h5")

from keras.applications.inception_v3 import InceptionV3
from keras.layers import AveragePooling2D, Dropout, Dense, Flatten
from keras.models import Model

n_classes = 101

# base model is inception_v3 weights pre-trained on ImageNet
base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(299, 299, 3)
)

x = base_model.output

# added layers to the base model
x = AveragePooling2D(pool_size=(8, 8))(x)
x = Dropout(.4)(x)
x = Flatten()(x)

# add softmax activation
predictions = Dense(n_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights(weights_path)


def start_process(img_path, plot=False):
    img_path = os.path.join(settings.MEDIA_ROOT, img_path)
    # model_path = os.path.join(settings.MEDIA_ROOT, "models", "food101_final_model.h5")
    img = path_to_tensor(img_path)
    img = preprocess_input(img)

    # make prediction
    predicted_vec = model.predict(img)
    predicted_label = food101[np.argmax(predicted_vec)]

    # show predicted image
    img = cv2.imread(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.axis('off')
    plt.title("yummy! It looks like {}".format(predicted_label))
    plt.show()

    # show top 5 predictions with probability
    if plot:
        # take top 5 probable pics
        top5_probs = np.sort(predicted_vec)[0][-5:]
        top5_labels = np.argsort(predicted_vec)[0][-5:]

        # plot bar graph
        x_pos = np.arange(len(top5_labels))
        plt.bar(x_pos, top5_probs)
        plt.title("top 5 predictions")
        plt.xticks(x_pos, [food101[int(idx)] for idx in top5_labels], rotation=20)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.show()
    return predicted_label
