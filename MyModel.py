import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization , Input ,concatenate
from keras.losses import categorical_crossentropy,categorical_hinge,hinge,squared_hinge
from keras.optimizers import Adam
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from sklearn.metrics import accuracy_score


# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
# from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.utils import plot_model


def fer_dataset():
    x=np.load('dataset\Fer_X.npy')
    y=np.load('dataset\Fer_Y.npy')
    x=np.expand_dims(x,-1)
    x = x / 255.0
    y = np.eye(7, dtype='uint8')[y]

    Split = np.load('dataset\Fer_Usage.npy')
    x_index = np.where(Split == 'Training')
    y_index = np.where(Split == 'PublicTest')
    z_index = np.where(Split == 'PrivateTest')
    # print("X_INDEX = ",x_index)
    # print("X_INDEX[0] = ", x_index[0][0])
    # print("X_INDEX[-1] = ",x_index[0][-1])
    X_Train = x[x_index[0][0]:x_index[0][-1]+1]
    X_Valid = x[y_index[0][0]:y_index[0][-1]+1]
    X_Test = x[z_index[0][0]:z_index[0][-1]+1]


    # print("Y_INDEX = ", y_index)
    # print("Y_INDEX[0] = ", y_index[0][0])
    # print("Y_INDEX[-1] = ",y_index[0][-1])
    Y_Train = y[x_index[0][0]:x_index[0][-1]+1]
    Y_Valid = y[y_index[0][0]:y_index[0][-1] + 1]
    Y_Test = y[z_index[0][0]:z_index[0][-1] + 1]

    return X_Train,X_Valid,X_Test,Y_Train,Y_Valid,Y_Test

def CNN3():
    num_labels = 7
    batch_size = 128
    epochs = 300
    width, height , depth = 48, 48 ,1
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)


    print("Loading Data !")

    X_Train, X_Valid, X_Test, Y_Train, Y_Valid, Y_Test = fer_dataset()

    # Initialising the CNN
    model = Sequential()

    # 1 - Convolution
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2nd Convolution layer
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 3rd Convolution layer
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 4th Convolution layer
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flattening
    model.add(Flatten())

    # Fully connected layer 1st layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # Fully connected layer 2nd layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(7, activation='softmax'))

    opt = Adam(lr=0.0005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # model.fit(X_Train, Y_Train, validation_data=(X_Valid, Y_Valid), epochs=300, verbose=1)

    filepath = "ConvNetV3_1_best_weights.hdf5"
    model_json = model.to_json()
    with open("ConvNetV3_1_model.json", "w") as jsonfile:
        jsonfile.write(model_json)

    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=50)
    mc = ModelCheckpoint(filepath, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    model.fit_generator(data_generator.flow(X_Train, Y_Train, batch_size=batch_size),
                        validation_data=(X_Valid, Y_Valid), steps_per_epoch=len(Y_Train) / batch_size,
                        epochs=epochs, verbose=1, callbacks=[es, mc])
    print("Model has been saved to disk ! Training time done !")


def CNN2():
    #
    # num_labels = 7
    # batch_size = 128
    # epochs = 300
    # width, height , depth = 48, 48 ,1
    # data_generator = ImageDataGenerator(
    #     featurewise_center=False,
    #     featurewise_std_normalization=False,
    #     rotation_range=10,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     zoom_range=.1,
    #     horizontal_flip=True)

    img_size = 48
    batch_size = 64
    epochs = 300

    data_generator = ImageDataGenerator(horizontal_flip=True)


    print("Loading Data !")

    X_Train, X_Valid, X_Test, Y_Train, Y_Valid, Y_Test = fer_dataset()

    # Initialising the CNN
    model = Sequential()

    # 1 - Convolution
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2nd Convolution layer
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 3rd Convolution layer
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 4th Convolution layer
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flattening
    model.add(Flatten())

    # Fully connected layer 1st layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # Fully connected layer 2nd layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(7, activation='softmax'))

    opt = Adam(lr=0.0005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # model.fit(X_Train, Y_Train, validation_data=(X_Valid, Y_Valid), epochs=300, verbose=1)

    filepath = "ConvNetV2_1_best_weights.hdf5"
    model_json = model.to_json()
    with open("ConvNetV2_1_model.json","w") as jsonfile:
        jsonfile.write(model_json)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=2, min_lr=0.00001, mode='auto')
    mc = ModelCheckpoint(filepath, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    model.fit_generator(data_generator.flow(X_Train, Y_Train, batch_size=batch_size),validation_data=(X_Valid, Y_Valid),steps_per_epoch= len(Y_Train) / batch_size,
                        epochs=epochs, verbose=1, callbacks=[reduce_lr,mc])


    print("Model has been saved to disk ! Training time done !")

    # json_model = open("ConvNetV2_1_model.json", 'r')
    # loaded_json_model = json_model.read()
    # json_model.close()
    # model_NET2 = model_from_json(loaded_json_model)
    # model_NET2.load_weights("ConvNetV2_1_best_weights.hdf5")
    #
    # predicted_NET = model_NET2.predict(X_Test)
    # True_Y = []
    # Predicted_Y = []
    # predicted_list = predicted_NET.tolist()
    # true_Y_list = Y_Test.tolist()
    #
    # for i in range(len(Y_Test)):
    #     Proba_max = max(predicted_NET[i])
    #     current_class = max(true_Y_list[i])
    #     class_of_Predict_Y = predicted_list[i].index(Proba_max)
    #     class_of_True_Y = true_Y_list[i].index(current_class)
    #
    #     True_Y.append(class_of_True_Y)
    #     Predicted_Y.append(class_of_Predict_Y)
    #
    # print("Accuracy on test set :" + str(accuracy_score(True_Y,Predicted_Y)*100) + "%")
    #
    # np.save("MODEL_CNN_True_y.npy", True_Y)
    # np.save("MODEL_CNN_Predict_y.npy",Predicted_Y)

def ExtractFeatures_Layer(dim):



    model = Sequential()
    model.add(Dense(4096,input_dim=dim,kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))

    return model

def CNN1():
    num_features = 64
    num_labels = 7
    batch_size = 128
    epochs = 300
    w, h , d = 48, 48 ,1
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)

    print("Loading Data !")

    X_Train, X_Valid, X_Test, Y_Train, Y_Valid, Y_Test = fer_dataset()
    model = Sequential()
    model.add(
        Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(w, h, d), data_format='channels_last',
               kernel_regularizer=l2(0.01)))
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(4 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(4 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])
    # model.fit(X_Train, Y_Train, validation_data=(X_Valid, Y_Valid), epochs=300, verbose=1)

    filepath = "ConvNetV1_1_best_weights.hdf5"
    model_json = model.to_json()
    with open("ConvNetV1_1_model.json","w") as jsonfile:
        jsonfile.write(model_json)
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=50)
    mc = ModelCheckpoint(filepath, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    model.fit_generator(data_generator.flow(X_Train, Y_Train, batch_size=batch_size),validation_data=(X_Valid, Y_Valid),steps_per_epoch= len(Y_Train) / batch_size,
                        epochs=epochs, verbose=1, callbacks=[es,mc])
    print("Model has been saved to disk ! Training time done !")

def CNN_Layer(w,h,d):

    num_features = 64
    model = Sequential()
    model.add(Conv2D(num_features,kernel_size=(3,3),activation = 'relu',input_shape = (w,h,d),data_format='channels_last', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))


    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(4 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(4 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    return model

    # model.add(Dense(2048, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_labels, activation='softmax'))
    #
    # model.compile(loss=categorical_crossentropy,
    #               optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
    #               metrics=['accuracy'])
    # # model.fit(X_Train, Y_Train, validation_data=(X_Valid, Y_Valid), epochs=300, verbose=1)
    #
    # filepath = "ConvNetV1_1_best_weights.hdf5"
    # model_json = model.to_json()
    # with open("ConvNetV1_1_model.json","w") as jsonfile:
    #     jsonfile.write(model_json)
    # es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=50)
    # mc = ModelCheckpoint(filepath, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    # model.fit_generator(data_generator.flow(X_Train, Y_Train, batch_size=batch_size),validation_data=(X_Valid, Y_Valid),steps_per_epoch= len(Y_Train) / batch_size,
    #                     epochs=epochs, verbose=1, callbacks=[es,mc])
    # print("Model has been saved to disk ! Training time done !")

# def CNN_DSIFT():
#     num_labels = 7
#     batch_size = 128
#     epochs = 300
#     width, height , depth = 48, 48 ,1
#     data_generator = ImageDataGenerator(
#         featurewise_center=False,
#         featurewise_std_normalization=False,
#         rotation_range=10,

#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         zoom_range=.1,
#         horizontal_flip=True)
#
#     print("Loading Data !")
#
#     X_Train, X_Valid, X_Test, Y_Train, Y_Valid, Y_Test = fer_dataset()
#
#     Split = np.load('dataset\Fer_Usage.npy')
#     x_index, = np.where(Split == 'Training')
#     y_index, = np.where(Split == 'PublicTest')
#
#     X_SIFT = np.load("feature_extraction\Fer2013_DSIFT_ector_Histogram.npy")
#     X_SIFT = X_SIFT.astype('float64')
#     X_SIFT_Train = X_SIFT[x_index[0]:x_index[-1] + 1]
#     X_SIFT_Valid = X_SIFT[y_index[0]:y_index[-1] + 1]
#
#     print("Data has been gernerated !")
#     print(X_SIFT_Train.shape[1])
#     SIFT = ExtractFeatures_Layer(X_SIFT_Train.shape[1])
#     CNN = CNN_Layer(width, height, depth)
#
#     MergeModel = concatenate([CNN.output, SIFT.output])
#
#     m = Dense(2048, activation='relu')(MergeModel)
#     m = Dropout(0.5)(m)
#     m = Dense(num_labels, activation='softmax')(m)
#
#     model = Model(inputs=[CNN.input, SIFT.input], outputs=m)
#
#     model.compile(loss=categorical_crossentropy,
#                   optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
#                   metrics=['accuracy'])
#
#     filepath = "ConvSIFTNET_1_best_weights.hdf5"
#     early_stop = EarlyStopping(monitor='val_accuracy', patience=50, mode='max')
#     checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
#     callbacks_list = [checkpoint, early_stop]
#
#     model_json = model.to_json()
#     with open("ConvSIFTNET_1_model.json", "w") as json_file:
#         json_file.write(model_json)
#
#     model.fit_generator(data_generator.flow([X_Train, X_SIFT_Train], Y_Train,
#                                             batch_size=batch_size),
#                         steps_per_epoch=len(Y_Train) / batch_size,
#                         epochs=epochs,
#                         verbose=1,
#                         callbacks=callbacks_list,
#                         validation_data=([X_Valid, X_SIFT_Valid], Y_Valid),
#                         shuffle=True
#                         )
#
#     print("Model has been saved to disk ! Training time done !")

def CNN_SIFT():
    num_labels = 7
    batch_size = 128
    epochs = 300
    width, height , depth = 48, 48 ,1
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)

    print("Loading Data !")

    X_Train, X_Valid, X_Test, Y_Train, Y_Valid, Y_Test = fer_dataset()

    Split = np.load('dataset\Fer_Usage.npy')
    x_index, = np.where(Split == 'Training')
    y_index, = np.where(Split == 'PublicTest')

    X_SIFT = np.load("feature_extraction\Fer2013_SIFTDetector_Histogram.npy")
    X_SIFT = X_SIFT.astype('float64')
    X_SIFT_Train = X_SIFT[x_index[0]:x_index[-1] + 1]
    X_SIFT_Valid = X_SIFT[y_index[0]:y_index[-1] + 1]

    print("Data has been gernerated !")
    print(X_SIFT_Train.shape[1])
    SIFT = ExtractFeatures_Layer(X_SIFT_Train.shape[1])
    CNN = CNN_Layer(width, height, depth)

    MergeModel = concatenate([CNN.output, SIFT.output])

    m = Dense(2048, activation='relu')(MergeModel)
    m = Dropout(0.5)(m)
    m = Dense(num_labels, activation='softmax')(m)

    model = Model(inputs=[CNN.input, SIFT.input], outputs=m)

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])

    filepath = "ConvSIFTNET_1_best_weights.hdf5"
    early_stop = EarlyStopping(monitor='val_accuracy', patience=50, mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, early_stop]


    model_json = model.to_json()
    with open("ConvSIFTNET_1_model.json", "w") as json_file:
        json_file.write(model_json)

    model.fit_generator(data_generator.flow([X_Train, X_SIFT_Train], Y_Train,
                                            batch_size=batch_size),
                        steps_per_epoch=len(Y_Train) / batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks_list,
                        validation_data=([X_Valid, X_SIFT_Valid], Y_Valid),
                        shuffle=True
                        )

    print("Model has been saved to disk ! Training time done !")


CNN3()