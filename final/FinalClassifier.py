import tensorflow as tf
import Classifier_Binary

Photo_Painting_model = tf.keras.models.load_model('../model/my_best_model_Photo_Painting.epoch16-loss0.37.hdf5')
Photo_Schementic_model = tf.keras.models.load_model('../model/my_best_model_Photo_Schementic.epoch10-loss0.12.hdf5')
Photo_Sketch_model = tf.keras.models.load_model('../model/my_best_model_Photo_Sketch.epoch18-loss0.01.hdf5')
Photo_Text_model = tf.keras.models.load_model('../model/my_best_model_Photo_Text.epoch10-loss0.01.hdf5')
datasets = ['../Dataset_Binary_Project/Project_Dataset_Ph_Pa','../Dataset_Binary_Project/Project_Dataset_Ph_Sh','../Dataset_Binary_Project/Project_Dataset_Ph_Sk','../Dataset_Binary_Project/Project_Dataset_Ph_Te']
all_model = [Photo_Painting_model,Photo_Schementic_model,Photo_Sketch_model,Photo_Text_model]

def all_binary_classifier(dataset,batch_s,model):
    train_set, test_set = Classifier_Binary.dataset(dataset, image_h=180, image_w=180,
                                                    batch_s=batch_s)
    Classifier_Binary.autotune_Dataset(train_set, test_set)
    score, accuracy = model.evaluate(test_set, batch_size=batch_s)
    print('Loss : ', score)
    print('Global Accuracy : ', accuracy)

def all_classifier(batch_s,all_model, datasets):
    #for dataset in datasets :
        for model in all_model:
            all_binary_classifier("../Dataset/Project_Dataset_Test",batch_s, model)

all_classifier(32,all_model,datasets)

