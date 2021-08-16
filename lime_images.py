import keras
import numpy as np
import tensorflow.keras.datasets.cifar10 as cifar10
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, InputLayer, Dropout
from keras.utils.np_utils import to_categorical


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = np.expand_dims(x_train.astype("float32") / 255, -1)
# x_test = np.expand_dims(x_test.astype("float32") / 255, -1)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model = keras.Sequential()
model.add(InputLayer(input_shape=(100, 100, 3,)))
model.add(Conv2D(8, 5, padding='same', activation="relu"))
model.add(MaxPooling2D(2))
model.add(Conv2D(16, 3, padding='same', activation="relu"))
model.add(MaxPooling2D(2))
model.add(Conv2D(32, 3, padding='same', activation="relu"))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()






model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=16, epochs=2)

from lime import lime_image

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(x_test[0], model.predict, top_labels=3, num_samples=1000)

from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt

temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=False)
temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                hide_rest=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
ax1.imshow(mark_boundaries(temp_1, mask_1))
ax2.imshow(mark_boundaries(temp_2, mask_2))
ax1.axis('off')
ax2.axis('off')
