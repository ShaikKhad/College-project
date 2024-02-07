from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=[.8, 1],
    channel_shift_range=30,
    fill_mode='reflect',
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    vertical_flip=True
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

# set path to the data sets
train_path = 'data/train'
valid_path = 'data/valid'
test_path = 'data/test'

# set input dimention to 299x299 (default for inception_v3)
dimentions = (299, 299)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=dimentions,
    batch_size=batch_size
)

validation_generator = test_datagen.flow_from_directory(
    valid_path,
    target_size=dimentions,
    batch_size=batch_size
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=dimentions,
    batch_size=batch_size
)

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


def step_decay(epoch):
    initial_lr = 1e-2

    if epoch < 9:
        return initial_lr
    else:
        return 0.00008


import matplotlib.pyplot as plt

x = [i for i in range(20)]
y = [step_decay(i) for i in range(20)]
plt.plot(x, y)

from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.metrics import top_k_categorical_accuracy

# optimizer
opt = SGD(lr=.01, momentum=.9)

# add LearningRateScheduler to update it
lr_scheduler = LearningRateScheduler(step_decay)


# calculate top_5_accuracy to evalute the model
def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


# putting them together and compile the model
model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy', top_5_accuracy]
)

from keras.callbacks import ModelCheckpoint

model_file = 'saved_models/weights.best.InceptionV3_final_model.hdf5'
checkpointer = ModelCheckpoint(filepath=model_file,
                               verbose=1, save_best_only=True)

# %%time

epochs = 16

history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              epochs=epochs,
                              callbacks=[checkpointer, lr_scheduler],
                              steps_per_epoch=train_generator.samples // batch_size,
                              validation_steps=validation_generator.samples // batch_size
                              )

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])

plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])

plt.show()

model.load_weights(model_file)
model.evaluate_generator(test_generator, steps=10, verbose=1)

### saving the trained model
# serialize model to JSON
model_json = model.to_json()
with open("saved_models/food101_final_model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("saved_models/food101_final_model.h5")

## Load the class labels (which are indexes are the same as the ones from generator)
with open('data/labels.txt', 'r') as f:
    food101 = [l.strip().lower() for l in f]

from keras.preprocessing import image
import numpy as np


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(299, 299))
    # convert PIL.Image.Image type to 3D tensor with shape (299, 299, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 299, 299, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


import cv2


def img_analysis(img_path, plot=False):
    # process image
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


img_analysis('test_imgs/sushi.jpg')
