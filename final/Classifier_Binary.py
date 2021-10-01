import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

def dataset (data_dir,image_h = 180,image_w = 180,batch_s = 32):
    # Le train_set
    train_set = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.20,
        subset='training',
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
        validation_split=0.20,
        subset='validation',
        seed=42,
        image_size=(image_h, image_w),
        batch_size=batch_s,
        color_mode='rgb',
        label_mode='int',
        labels="inferred"
    )
    print(test_set.class_names)
    return train_set, test_set

def autotune_Dataset(train_set,test_set):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_set = train_set.cache().shuffle(5).prefetch(buffer_size=AUTOTUNE)
    test_set = test_set.cache().prefetch(buffer_size=AUTOTUNE)

# Résumé du modèle
def create_model(num_classes = 5,
                 loss_fn_to_use=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                 optimizer_to_use='adam',
                 metrics_to_use=['accuracy'],
                 image_h = 180,image_w = 180,):

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
        tf.keras.layers.Dense(num_classes),
    ])

    #Set Loss Function
    loss_fn=loss_fn_to_use
    #On compile le modèle.
    model.compile(optimizer = optimizer_to_use,
                  loss = loss_fn,
                  metrics= metrics_to_use)
    return model

def save_model(model_name):
    name = "my_best_model_" + model_name + ".epoch{epoch:02d}-loss{val_loss:.2f}.hdf5"
    checkpoint_filepath = '../model/' + name
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        #save_freq="epoch",
        save_best_only=True
    )
    return model_checkpoint_callback

def execute_training_model(train_set,test_set,model_checkpoint_callback, model_name, model,epochs):

    tf.debugging.set_log_device_placement(True)

    if tf.test.is_gpu_available():
        with tf.device('/gpu:0'):
            history = model.fit(train_set, validation_data=test_set, epochs=epochs,
                                callbacks=[model_checkpoint_callback])
    else:
        history = model.fit(train_set, validation_data=test_set, epochs=epochs, callbacks=[model_checkpoint_callback])

    model.summary()

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
    plt.title('Training and Validation Loss ' + model_name)
    plt.savefig('images/Training_and_Validation_Loss_' + model_name)
    plt.show()


def process_model(model_name,epochs,data_dir):
    train_set,test_set = dataset(data_dir)
    autotune_Dataset(train_set,test_set)
    model = create_model(num_classes=2)
    model_checkpoint_callback = save_model(model_name)
    execute_training_model(train_set,test_set,model_checkpoint_callback,model_name, model,epochs=epochs)


