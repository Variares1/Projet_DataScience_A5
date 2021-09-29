import tensorflow as tf
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from tensorflow.python.ops.gen_control_flow_ops import Merge
import Classifier_Binary

Photo_Painting_model = tf.keras.models.load_model('../model/my_best_model_Photo_Painting.epoch16-loss0.37.hdf5')
Photo_Schementic_model = tf.keras.models.load_model('../model/my_best_model_Photo_Schementic.epoch10-loss0.12.hdf5')
Photo_Sketch_model = tf.keras.models.load_model('../model/my_best_model_Photo_Sketch.epoch18-loss0.01.hdf5')
Photo_Text_model = tf.keras.models.load_model('../model/my_best_model_Photo_Text.epoch10-loss0.01.hdf5')

merge_model = Merge([Photo_Painting_model, Photo_Schementic_model,Photo_Sketch_model,Photo_Text_model])

model_name = "All_Dataset"
epochs = 25

train_set,test_set = Classifier_Binary.dataset("../Dataset/Project_Dataset_Clean",image_h = 180,image_w = 180,batch_s = 32)
Classifier_Binary.autotune_Dataset(train_set,test_set)
Classifier_Binary.create_model(num_classes=2)
model_checkpoint_callback = Classifier_Binary.save_model(value=model_name)
Classifier_Binary.execute_training_model(train_set,test_set,model_checkpoint_callback,model_name, merge_model,epochs=epochs)