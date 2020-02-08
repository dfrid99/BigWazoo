from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

from tensorflow.keras import datasets, layers, models
from trainer import Trainer



path1 = 'test_1/'
path2 = 'test_2/'
path3 = 'test_3/'
train = Trainer(30)
train.addImages(path1,0)
train.addImages(path2,1)
train.addImages(path3,2)
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
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train.train_images, train.train_labels, epochs=10, batch_size = 20,
                    validation_data=(train.test_images, train.test_labels))

model.save('machLearnModv3.h5')
#david is faggot


