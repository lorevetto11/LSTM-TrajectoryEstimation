import keras
from keras import layers
from keras.models import load_model
import numpy as np
import random 
import sys
import glob
import matplotlib.pyplot as plt
from Interpolation import interpolation
from Dataset_Extract.Dataset import Dataset

Method = 5

ideal_line = False

sentences = [] 

x1_test, x2_test, y1_test, y2_test, xx, yy = [], [], [], [], [], []

test_track, test_track2 = [], []

X, Y = [], []

if __name__ == '__main__':

    dataset = Dataset()

    if(ideal_line):
        dataset.extract_F1_dataset_idealline()
        dataset.reshape_multi_array()
        X, Y = dataset.get_dataset()

    else:
        #dataset.extract_AC_dataset()
        dataset.extract_F1_dataset()
        dataset.extract_center_trajectory()
        #dataset.reshape_multi_array()
        dataset.reshape_one_array()
        X, Y = dataset.get_dataset()


    if(Method == 1):
        model = keras.models.Sequential()
        model.add(layers.LSTM(128, input_shape=(10, 4)))
        model.add(layers.Dense(2, activation='softmax'))
        optimizer = keras.optimizers.RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        print(model.summary(90))

    elif(Method == 2):
        model = keras.models.Sequential()
        model.add(layers.LSTM(128, activation='relu', input_shape=(1, 4)))
        model.add(layers.Dense(2))
        model.compile(optimizer='adam', loss='mse')

        print(model.summary(90))

    elif(Method == 3):
        model = keras.models.Sequential()
        model.add(layers.LSTM(64, activation='relu', input_shape=(10, 4), return_sequences=True))
        model.add(layers.LSTM(64, activation='relu', input_shape=(10, 4)))
        model.add(layers.Dense(2))
        model.compile(optimizer='adam', loss='mse')

        print(model.summary(90))

    else:
        model = load_model('demo.h5')

    #for f, g in zip(X, Y):
        #model.fit(f, g, epochs=100, validation_split=0.1, verbose=1)
        
    
    #model.save('demo.h5')      
    '''
    for a, b in zip(X, Y):
        for f, g in zip(a, b):
            model.fit(f, g, epochs=300, validation_split=0.1, verbose=1)
            model.save('my_model.h5')  #creates a HDF5 file 'my_model.h5'    
            model = load_model('my_model.h5')
    '''

    csv_data_temp = np.loadtxt('Test/Berlin_track.csv', comments='#', delimiter=',')

    
    mod = int(len(csv_data_temp)/10)

    interpolation = interpolation.Interpolation()

    track_interp_left = interpolation.interpolate_polyline(csv_data_temp[:, 0], csv_data_temp[:, 1], mod*10)
    track_interp_right = interpolation.interpolate_polyline(csv_data_temp[:, 2], csv_data_temp[:, 3], mod*10)

    x1_test.append(track_interp_left[:, 0])
    y1_test.append(track_interp_left[:, 1])
    x2_test.append(track_interp_right[:, 0])
    y2_test.append(track_interp_right[:, 1])

    #plt.plot(x1_test, y1_test, "bs", linewidth=0.01)
    #plt.plot(x2_test, y2_test, "gs", linewidth=0.01)
    #plt.show()

    for xi, yi, xf, yf in zip(x1_test[0], y1_test[0], x2_test[0], y2_test[0]):
        test_input = np.array([xi, yi, xf, yf])
        test_input = test_input.reshape(1, 1, 4)
        test_output = model.predict(test_input, verbose=1)
        test_output = test_output.reshape(1,2)
        xx.append(test_output[0][0])
        yy.append(test_output[0][1])
        print(test_output)
    '''

    for i in range(0, len(x1_test)):
        test_track.append(np.column_stack((x1_test[i], y1_test[i], x2_test[i], y2_test[i])))
 
    test_track2 = np.array(test_track)

    test_track2 = test_track2.reshape(int(len(test_track[0])/10),10, 4)

    test_output = model.predict(test_track2, verbose=1)
    output = test_output.reshape(int(len(test_output)), 10,2)
    
    plt.plot(x1_test, y1_test, "bs")
    plt.plot(x2_test, y2_test, "gs")

    for i in range(0, len(output)):
        plt.plot(output[i][:, 0], output[i][:, 1], "rs")

    plt.show()

    for test_input in test_track2:
        test_output = model.predict(test_input, verbose=1)
        output = test_output.reshape(int(len(test_output)), 10,2)
        plt.plot(x1_test, y1_test, "bs")
        plt.plot(x2_test, y2_test, "gs")
        plt.plot(output[:, 0], output[:, 1], "rs")
        plt.show()
        
    '''
    plt.plot(x1_test, y1_test, "bs", linewidth=0.01)
    plt.plot(x2_test, y2_test, "gs", linewidth=0.01)

 
    plt.plot(xx, yy, "rs", linewidth=0.01)
    plt.show()
