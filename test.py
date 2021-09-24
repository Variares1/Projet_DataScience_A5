import pathlib
import zipfile
import os
import tensorflow as tf
from PIL import Image as Image
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

path_to_zip_file = "Dataset projet" #"C:\\Users\\nicos\\PycharmProjects\\Projet_DataScience_A5\\Dataset projet"
directory_to_extract_to = "C:\\Users\\nicos\\PycharmProjects\\Projet_DataScience_A5\\Dataset_Test"
data_dir = directory_to_extract_to

data_dir = pathlib.Path(data_dir)

image_h = 256
image_w = 256
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


# Résumé du modèle
def create_model(summary = False, num_classes = 5, input_shape=(28,28,3),
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
        tf.keras.layers.Dropout(
            rate=0.2, noise_shape=None, seed=42,
        ),
        tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255, offset=0),
        tf.keras.layers.Conv2D(16, [3,3], strides = (1,1), input_shape = input_shape, padding= "valid",activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (1,1),padding = "same"),
        tf.keras.layers.Conv2D(8, [5,5], strides = (1,1), padding= "valid",activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (1,1),padding = "same"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='relu'),
        tf.keras.layers.Softmax(),
    ])
    if summary:
        model.summary()
    #Set Loss Function
    loss_fn=loss_fn_to_use
    #On compile le modèle.
    model.compile(optimizer = 'adam',
                  loss = loss_fn,
                  metrics=['accuracy'])
    return model


epochs=10
#physical_device_desc: "device: 0, name: NVIDIA GeForce RTX 2070 Super, pci bus id: 0000:01:00.0, compute capability: 7.5"
tf.debugging.set_log_device_placement(True)

#with tf.device('/gpu:0'):
#    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#    c = tf.matmul(a, b)
#print (tf.print(c))
print("test GPU")
model = create_model(summary = True)
with tf.device('/gpu:0'):
    history = model.fit(train_set,validation_data = test_set,epochs=epochs)

model.summary()
#model.save_weights('./test_model')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
