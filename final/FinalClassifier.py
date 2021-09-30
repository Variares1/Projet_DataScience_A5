import tensorflow as tf
import Classifier_Binary
import shutil
import os


Photo_Painting_model = tf.keras.models.load_model('../model/my_best_model_Photo_Painting.epoch16-loss0.37.hdf5')
Photo_Schementic_model = tf.keras.models.load_model('../model/my_best_model_Photo_Schementic.epoch10-loss0.12.hdf5')
Photo_Sketch_model = tf.keras.models.load_model('../model/my_best_model_Photo_Sketch.epoch18-loss0.01.hdf5')
Photo_Text_model = tf.keras.models.load_model('../model/my_best_model_Photo_Text.epoch10-loss0.01.hdf5')
all_model = [Photo_Painting_model,Photo_Schementic_model,Photo_Sketch_model,Photo_Text_model]

binary_dataset_dir_path = "../Dataset_Binary_Project"
dataset_to_extract = "../Dataset/Project_Dataset_Clean"
dataset_to_compare = "Photo"

def create_binary_dataset(dataset_to_extract,binary_dataset_dir_path, dataset_to_compare):
    if not os.path.exists(binary_dataset_dir_path):
        os.mkdir(binary_dataset_dir_path)

    for directory in os.listdir(dataset_to_extract):
        print(dataset_to_extract + "/" + directory)
        if directory == dataset_to_compare:
            continue

        print("Check directory " + binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare)
        if not os.path.exists(binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare):
            os.mkdir(binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare)
            print("Create directory " + binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare)

        print("Check directory " + binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare + "/" + directory)
        if not os.path.exists(binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare + "/" + directory):
            os.mkdir(binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare + "/" + directory)
            print("Create directory " + binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare + "/" + directory)

        print("Check directory " + binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare + "/" + dataset_to_compare)
        if not os.path.exists(binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare + "/" + dataset_to_compare):
            os.mkdir(binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare + "/" + dataset_to_compare)
            print("Create directory " + binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare + "/" + dataset_to_compare)

        print("Copy file in " + binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare + "/" + directory)
        for file in os.listdir(dataset_to_extract + "/" + directory):
            if not os.path.exists(binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare + "/" + directory + "/" + file):
                shutil.copy2(dataset_to_extract + "/" + directory + "/" + file, binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare + "/" + directory)
        print("Finished copy")

        print("Copy file in " + binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare + "/" + dataset_to_compare)
        for file in os.listdir(dataset_to_extract + "/" + dataset_to_compare):
            if not os.path.exists(binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare + "/" + dataset_to_compare + "/" + file):
                shutil.copy2(dataset_to_extract + "/" + dataset_to_compare + "/" + file, binary_dataset_dir_path + "/" + directory + "_" + dataset_to_compare + "/" + dataset_to_compare)
        print("Finished copy")

def all_binary_classifier(dataset,batch_s,model):
    train_set, test_set = Classifier_Binary.dataset(dataset, image_h=180, image_w=180,
                                                    batch_s=batch_s)
    Classifier_Binary.autotune_Dataset(train_set, test_set)
    score, accuracy = model.evaluate(test_set, batch_size=batch_s)
    print('Loss : ', score)
    print('Global Accuracy : ', accuracy)

def all_classifier(batch_s,all_model):
        for model in all_model:
            all_binary_classifier("../Dataset/Project_Dataset_Test",batch_s, model)

all_classifier(32,all_model)
create_binary_dataset(dataset_to_extract,binary_dataset_dir_path,dataset_to_compare)

