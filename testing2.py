from keras.models import model_from_json
from sklearn.metrics import accuracy_score
import numpy as np
import cv2

def Test_Combine(X):

    json_model =  open("ConvNetV2_1_model.json","r")
    loaded_model_json = json_model.read()
    json_model.close()
    model_CNN = model_from_json(loaded_model_json)
    model_CNN.load_weights("ConvNetV2_1_best_weights.hdf5")
    model_CNN._make_predict_function()
    predicted_CNN = model_CNN.predict(X)
    predicted_list = predicted_CNN.tolist()
    print(predicted_list[0])
    proba_max = max(predicted_CNN[0])
    class_of_Predict_Y = predicted_list[0].index(proba_max)
    print(class_of_Predict_Y)
    # trueY_list = Y.tolist()
    # True_Y = []
    # Predicted_Y = []
    # for i in range(len(Y)):
    #     proba_max = max(predicted_list[i])
    #     current_class = max(trueY_list[i])
    #     class_of_Predict_Y = predicted_list[i].index(proba_max)
    #     class_of_True_Y = trueY_list[i].index(current_class)
    #
    #     True_Y.append(class_of_True_Y)
    #     Predicted_Y.append(class_of_Predict_Y)
    #
    # print("Accuracy on test set :" + str(accuracy_score(True_Y, Predicted_Y) * 100) + "%")
    #
    # np.save("MODEL_CNN_True_y.npy_True_y", True_Y)
    # np.save("MODEL_CNN_Predict_y.npy1_Predict_y", Predicted_Y)

# Y = np.load("Fer2013_Y_test.npy")

# X = np.load("Fer2013_X_test.npy")
# X=cv2.imread("C:/Users/Rajshree/Pictures/Screenshots/expTest.png",0)

# X=cv2.imread("C:/Users/Rajshree/Pictures/Screenshots/Untitled6.png",0)
X=cv2.imread("C:/Users/Rajshree/Pictures/Camera Roll/IMG-20190907-WA0025.jpg",0)

# X2 = np.expand_dims(X1,axis=2)
# X3 = np.expand_dims(X2,axis=0)

roi = cv2.resize(X, (48, 48))
# pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
cv2.imshow("image",X)
cv2.waitKey(0)
Test_Combine(roi[np.newaxis, :, :, np.newaxis])