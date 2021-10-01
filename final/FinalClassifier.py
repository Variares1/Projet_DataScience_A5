import sklearn as sk
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import Classifier_Binary
import matplotlib as plt

Photo_Painting_model = tf.keras.models.load_model('../model/my_best_model_Photo_Painting.epoch16-loss0.37.hdf5')
Photo_Schementic_model = tf.keras.models.load_model('../model/my_best_model_Photo_Schementic.epoch10-loss0.12.hdf5')
Photo_Sketch_model = tf.keras.models.load_model('../model/my_best_model_Photo_Sketch.epoch18-loss0.01.hdf5')
Photo_Text_model = tf.keras.models.load_model('../model/my_best_model_Photo_Text.epoch10-loss0.01.hdf5')
all_model = [Photo_Painting_model, Photo_Schementic_model, Photo_Sketch_model, Photo_Text_model]

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

def pred_multi_binary_model(batch_size, dataset_binary):
    train_set, test_set = Classifier_Binary.dataset(dataset_binary, image_h=180, image_w=180,
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

    disp = sk.metrics.ConfusionMatrixDisplay(new_confus_mtx,display_labels=test_set.class_names)
    disp.plot()
    disp.show()

pred_multi_binary_model(32,"../Dataset_Binary_Project_test/")