from keras.models import model_from_json
import numpy
import os
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def Test_Combine(X,Y):

    # MyModel.CNN_SIFT()


    json_model = open("ConvSIFTNET_1_model.json", 'r')
    loaded_json_model = json_model.read()
    json_model.close()
    model_SIFTNET = model_from_json(loaded_json_model)
    model_SIFTNET.load_weights("ConvSIFTNET_1_best_weights.hdf5")
    model_SIFTNET.summary()

    Split = np.load('dataset\Fer_Usage.npy')
    z_index, = np.where(Split == 'PrivateTest')

    X_SIFT = np.load("feature_extraction\Fer2013_SIFTDetector_Histogram.npy")
    X_SIFT = X_SIFT.astype('float64')
    print(X_SIFT.shape)

    X_SIFT_Test = X_SIFT[z_index[0]:z_index[-1]+1]

    #CNN1
    # MyModel.CNN1()
    json_model1 = open("ConvNetV1_1_model.json", 'r')
    loaded_json_model1 = json_model1.read()
    json_model1.close()
    model_CNN1 = model_from_json(loaded_json_model1)
    model_CNN1.load_weights("ConvNetV1_1_best_weights.hdf5")
    model_CNN1.summary()

    #CNN2
    # MyModel.CNN2()
    json_model2 = open("ConvNetV2_1_model.json",'r')
    loaded_json_model2 = json_model2.read()
    json_model2.close()
    model_CNN2 = model_from_json(loaded_json_model2)
    model_CNN2.load_weights("ConvNetV2_1_best_weights.hdf5")
    model_CNN2.summary()

    #CNN3
    json_model3 = open("ConvNetV3_1_model.json",'r')
    loaded_json_model3 = json_model3.read()
    json_model3.close()
    model_CNN3 = model_from_json(loaded_json_model3)
    model_CNN3.load_weights("ConvNetV3_1_best_weights.hdf5")
    model_CNN3.summary()

    predicted_SIFT = model_SIFTNET.predict([X,X_SIFT_Test])
    predicted_CNN3 = model_CNN3.predict(X)
    predicted_CNN2 = model_CNN2.predict(X)
    predicted_CNN1 = model_CNN1.predict(X)

    predicted_combine = (predicted_SIFT + predicted_CNN1 + predicted_CNN2 + predicted_CNN3) / 4.0


    True_Y = []
    Predicted_Y = []
    predicted_list = predicted_combine.tolist()
    true_Y_list = Y.tolist()

    for i in range(len(Y)):
        Proba_max = max(predicted_combine[i])
        current_class = max(true_Y_list[i])
        class_of_Predict_Y = predicted_list[i].index(Proba_max)
        class_of_True_Y = true_Y_list[i].index(current_class)

        True_Y.append(class_of_True_Y)
        Predicted_Y.append(class_of_Predict_Y)

    print("Accuracy on test set :" + str(accuracy_score(True_Y,Predicted_Y)*100) + "%")

    np.save("MODEL_CNN_True_y.npy", True_Y)
    np.save("MODEL_CNN_Predict_y.npy",Predicted_Y)

Y = np.load("dataset\Fer2013_Y_test.npy")
X = np.load("dataset\Fer2013_X_test.npy")

Test_Combine(X,Y)



