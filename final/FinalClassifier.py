import sklearn as sk
import tensorflow as tf
from sklearn.svm._libsvm import predict
from sklearn.metrics import confusion_matrix
import os
import shutil
import Classifier_Binary
import matplotlib as plt

Photo_Painting_model = tf.keras.models.load_model('../model/my_best_model_Photo_Painting.epoch16-loss0.37.hdf5')
Photo_Schementic_model = tf.keras.models.load_model('../model/my_best_model_Photo_Schementic.epoch10-loss0.12.hdf5')
Photo_Sketch_model = tf.keras.models.load_model('../model/my_best_model_Photo_Sketch.epoch18-loss0.01.hdf5')
Photo_Text_model = tf.keras.models.load_model('../model/my_best_model_Photo_Text.epoch10-loss0.01.hdf5')
all_model = [Photo_Painting_model, Photo_Schementic_model, Photo_Sketch_model, Photo_Text_model]


binary_dataset_dir_path = "../Dataset_Binary_Project"
dataset_to_extract_path = "../Dataset/Project_Dataset_Clean"
binary_dataset_test_path = "../Dataset_Binary_Project_test"
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

def create_binary_dataset_test(dataset_to_extract,binary_dataset_test,dataset_to_compare):
    if not os.path.exists(binary_dataset_test):
        os.mkdir(binary_dataset_test)

    if not os.path.exists(binary_dataset_test + "/" + dataset_to_compare):
        os.mkdir(binary_dataset_test + "/" + dataset_to_compare)

    if not os.path.exists(binary_dataset_test + "/all_pictures"):
        os.mkdir(binary_dataset_test + "/all_pictures")

        for directory in os.listdir(dataset_to_extract):
            print(dataset_to_extract + "/" + directory)
            if directory == dataset_to_compare:
                continue

            print("Copy file in " + binary_dataset_test + "/all_pictures")
            for file in os.listdir(dataset_to_extract + "/" + directory):
                if not os.path.exists(binary_dataset_test + "/all_pictures/" + file):
                    shutil.copy2(dataset_to_extract + "/" + directory + "/" + file, binary_dataset_test + "/all_pictures")
            print("Finished copy")

        print("Copy file in " + binary_dataset_test + "/" + dataset_to_compare)
        for file in os.listdir(dataset_to_extract + "/" + dataset_to_compare):
            if not os.path.exists(binary_dataset_test + "/" + dataset_to_compare + "/" + file):
                shutil.copy2(dataset_to_extract + "/" + dataset_to_compare + "/" + file, binary_dataset_test + "/" + dataset_to_compare)
        print("Finished copy")


def all_binary_classifier(train_set, test_set, batch_s, model):
    Classifier_Binary.autotune_Dataset(train_set, test_set)
    score, accuracy = model.evaluate(test_set, batch_size=batch_s)
    print('Loss : ', score)
    print('Global Accuracy : ', accuracy)
    return model.predict(test_set, verbose=1).argmax(axis=1)


def all_classifier(train_set, test_set, batch_s, all_model):
    predictions_list = []
    for model in all_model:
        predictions_list.append(all_binary_classifier(train_set, test_set, batch_s, model))
    return predictions_list


batch_size = 32
train_set, test_set = Classifier_Binary.dataset("../Dataset_Binary_Project_test/", image_h=180, image_w=180,
                                                batch_s=batch_size)
pred = all_classifier(train_set, test_set, batch_size, all_model)
print(pred)
labels_pred = []
for p1, p2, p3, p4 in zip(pred[0], pred[1], pred[2], pred[3]):
    res = p1 + p2 + p3 + p4
    proba = res / 4 if res != 0 else res
    labels_pred.append(1 if proba > 0.5 else 0)

labels = []
for image_batch, label_batch in test_set:
    [labels.append(y.numpy()) for y in label_batch]

print(confusion_matrix(labels, labels_pred))

new_confus_mtx = sk.metrics.confusion_matrix(labels,labels_pred)

disp = sk.metrics.ConfusionMatrixDisplay(new_confus_mtx,display_labels=test_set.class_names).plot
plt.show()

#create_binary_dataset_test(dataset_to_extract_path,binary_dataset_test_path,dataset_to_compare)
#create_binary_dataset(dataset_to_extract_path,binary_dataset_dir_path,dataset_to_compare)
