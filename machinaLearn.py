from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

from tensorflow.keras import datasets, layers, models
import os
from thresh import *
from trainer import Trainer
from numpy import random


def makeImgArr(path, label):
    img_size = 30
    folder, dirs, files = next(os.walk(path))
    file_count = len(files)
    train_len = int(file_count * .8)
    test_len = int(file_count * .2)
    train_images = np.ndarray(shape=(train_len, img_size, img_size))
    train_labels = np.ndarray(shape=(train_len, 1))
    test_images = np.ndarray(shape=(test_len, img_size, img_size))
    test_labels = np.ndarray(shape=(test_len, 1))
    for i, file in enumerate(os.listdir(path)):
        if (i < train_len):
            train_images[i] = cv2.imread(path + file, 0)
            train_labels[i] = label
        else:
            test_images[i - train_len] = cv2.imread(path + file, 0)
            test_labels[i - train_len] = label

    return train_images,train_labels,test_images,test_labels



#print(img)
'''path = "test_1/"
train_images = np.ndarray(shape=(2400,30,30))
train_labels = np.ndarray(shape=(2400,1))
test_images = np.ndarray(shape=(600,30,30))
test_labels = np.ndarray(shape=(600,1))
for i,file in enumerate(os.listdir(path)):
    if (i < 1200):
        train_images[i] = cv2.imread(path + file, cv2.IMREAD_UNCHANGED)
        train_labels[i] = 0
    else:
        test_images[i-1200]= cv2.imread(path + file, cv2.IMREAD_UNCHANGED)
        test_labels[i-1200]= 0




path = 'test_2/'
for i,file in enumerate(os.listdir(path)):
    if (i < 1200):
        train_images[i+1200] = cv2.imread(path + file, cv2.IMREAD_UNCHANGED)
        train_labels[i+1200] = 1
    else:
        test_images[i-900]= cv2.imread(path + file, cv2.IMREAD_UNCHANGED)
        test_labels[i-900]= 1

def shuffleImgs(imgs,labels):
    rng_state = random.get_state()
    random.shuffle(imgs)
    random.set_state(rng_state)
    random.shuffle(labels)'''



'''shuffleImgs(train_images,train_labels)
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images.reshape([2400,30,30,1])
test_images = test_images.reshape([600,30,30,1])'''

'''cv2.namedWindow('yer')
for img in train_images:
    cv2.imshow('yer',img)
    cv2.waitKey(50)'''

path1 = 'test_1/'
path2 = 'test_2/'
train = Trainer(30)
train.addImages(path1,0)
train.addImages(path2,1)
train.shuffleArrs()
train.prepare()


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train.train_images, train.train_labels, epochs=10, batch_size = 20,
                    validation_data=(train.test_images, train.test_labels))

model.save('machLearnModv2.h5')
#david is faggot


