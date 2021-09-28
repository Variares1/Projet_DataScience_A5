import pathlib
import zipfile
import os

import numpy
import tensorflow as tf
from PIL import Image as Image
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Résumé du modèle
def create_model(num_classes = 5,
                 loss_fn_to_use = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)):
    model = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip(
            mode="horizontal_and_vertical",seed=42,input_shape=(image_h, image_w, 3)),
        layers.experimental.preprocessing.RandomRotation(
            factor = (-0.2, 0.3),
            fill_mode="reflect",
            interpolation="bilinear",
            seed=42,
            fill_value=0.0,
        ),
        layers.experimental.preprocessing.RandomZoom(
            height_factor= (0.2, 0.3),
            width_factor=None,
            fill_mode="reflect",
            interpolation="bilinear",
            seed=42,
            fill_value=0.0,
        ),
        tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255, offset=0),
        tf.keras.layers.Conv2D(64, [3,3], activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, [3,3],activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, [3,3],activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(8, [3,3],activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(
            rate=0.2, noise_shape=None, seed=42,
        ),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes),
    ])

    #Set Loss Function
    loss_fn=loss_fn_to_use
    #On compile le modèle.
    model.compile(optimizer = 'adam',
                  loss = loss_fn,
                  metrics=['accuracy'])
    return model

clean_dataset = "Dataset/Project_Dataset_Clean"
light_dataset = "Dataset/Project_Dataset_Test"

data_dir = clean_dataset

data_dir = pathlib.Path(data_dir)

image_h = 180
image_w = 180
batch_s = 32

# Le train_set
train_set = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split= 0.20,
  subset = 'training',
  seed=42,
  image_size=(image_h, image_w),
  batch_size=batch_s,
  color_mode='rgb',
  label_mode='int',
  labels="inferred"
)

# Le test_set
test_set = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split= 0.20,
  subset = 'validation',
  seed=42,
  image_size=(image_h, image_w),
  batch_size=batch_s,
  color_mode='rgb',
  label_mode='int',
  labels="inferred"
)

class_names =  train_set.class_names #A COMPLETER
print(class_names)



X, Y = tuple(zip(*test_set))



plt.figure(figsize=(8, 8))
for images, labels in train_set.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

images, labels = next(iter(train_set))
print(images.shape)
print(labels.shape)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_set = train_set.cache().shuffle(5).prefetch(buffer_size=AUTOTUNE)
test_set = test_set.cache().prefetch(buffer_size=AUTOTUNE)

epochs=25
name = 'my_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
checkpoint_filepath = '../model/' + name
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    #save_freq="epoch",
    save_best_only=True),

tf.debugging.set_log_device_placement(True)

model = create_model()
#with tf.device('/gpu:0'):
#    history = model.fit(train_set,validation_data = test_set,epochs=epochs, callbacks=[model_checkpoint_callback])

#cross val
seed = 7
numpy.random.seed(seed)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
model.summary()
#model.load_weights(checkpoint_filepath)
#model.save_weights('./test_model')

#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']

#loss = history.history['loss']
#val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
#plt.plot(epochs_range, acc, label='Training Accuracy')
#plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
#plt.plot(epochs_range, loss, label='Training Loss')
#plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)