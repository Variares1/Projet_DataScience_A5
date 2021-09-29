import tensorflow as tf
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.layers import Dense, Input

Photo_Painting_model = tf.keras.models.load_model('../model/my_best_model_Photo_Painting.epoch11-loss0.62.hdf5')
Photo_Schementic_model = tf.keras.models.load_model('../model/my_best_model_Photo_Schementic.epoch11-loss0.62.hdf5')
Photo_Sketch_model = tf.keras.models.load_model('../model/my_best_model_Photo_Sketch.epoch11-loss0.62.hdf5')
Photo_Text_model = tf.keras.models.load_model('../model/my_best_model_Photo_Text.epoch11-loss0.62.hdf5')
