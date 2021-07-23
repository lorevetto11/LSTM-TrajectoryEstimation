import numpy as np
import keras
import glob
import json
from keras.models import load_model
import time

from Augmentation.augmentation import Augmentation

from Plotting import plotting
from Interpolation import interpolation

class Predict:
    def __init__(self):
        self.x1_test = []
        self.y1_test = []
        self.x2_test = []
        self.y2_test = []
        self.x = []
        self.y = []
        self.track_files = []

        self.noise = False
        self.mse = True
        self.track = 0

        self.plotter = plotting.Plotting()

        self.augmentation = Augmentation()

    def Average(self, lst):
        sum = 0
        for i in lst:
            sum += i
        return sum / len(lst)
    
    def Sum(self, lst):
        sum = 0
        for i in lst:
            sum += i
        return sum 

    def extract_center_trajectory(self, x1, y1, x2, y2):
            y_true = []
            y_true.append(float((x1 + x2) / 2))
            y_true.append(float((y1 + y2) / 2))
            return y_true

    def test_predict_trajectory_OneArray(self):

        model = load_model('Model/LSTM/Saved/Bidirectional/rotate.h5')
        #model = load_model('Model/GRU/Saved/OneLayer.h5')

        for document in glob.glob('Predict/Track/*.csv'):
            self.track_files.append(document)

        for file in self.track_files:

            self.x1_test = []
            self.y1_test = []
            self.x2_test = []
            self.y2_test = []

            self.x = []
            self.y = []
            
            csv_data_temp = np.loadtxt(file, comments='#', delimiter=',')
        
            #mod = int(len(csv_data_temp)/10)

            interp = interpolation.Interpolation()

            self.x1_test.append(csv_data_temp[:, 0])
            self.y1_test.append(csv_data_temp[:, 1])
            self.x2_test.append(csv_data_temp[:, 2])
            self.y2_test.append(csv_data_temp[:, 3])
            tottime = []
            self.track += 1           
            for xl, yl, xr, yr in zip(self.x1_test[0], self.y1_test[0], self.x2_test[0], self.y2_test[0]):
                
                if(self.noise):
                    xl = self.augmentation.gaussian_noise(xl)
                    xr = self.augmentation.gaussian_noise(xr)
                
                test_input = np.array([xl, yl, xr, yr])
                test_input = test_input.reshape(1, 1, 4)
                test_input += 1000
                starttime = time.time()
                test_output = model.predict(test_input, verbose=1)
                endtime = time.time()
                tottime.append(endtime - starttime)
                test_output -= 1000
                test_output = test_output.reshape(1,2)
                self.x.append(test_output[0][0])
                self.y.append(test_output[0][1])

                if(self.mse):
                    y_true = np.array(self.extract_center_trajectory(xl, yl, xr, yr))
                    y_true = y_true.reshape(1,2)
                    mse = keras.losses.MeanSquaredError()
                    value = mse(y_true, test_output[0]).numpy()
                    print(value)
                    outF = open("predict/MSE_Result/rotate_track_"+str(self.track)+".txt", "a")
                    outF.write(str(value) + "\n")

            print('Tot Sec = ' + str(self.Sum(tottime)))
            print('AVG step = ' + str(self.Average(tottime)))

            #interp_path = interp.interpolate_polyline(self.x, self.y, len(self.x))

            self.plotter.plot(self.x1_test, self.y1_test, self.x2_test, self.y2_test, self.x, self.y)

            
    def test_predict_trajectory_StepArray(self):

        model = load_model('Model/GRU/Saved/OneLayer.h5')

        csv_data_temp = np.loadtxt('Predict/Track/Berlin_track.csv', comments='#', delimiter=',')
    
        mod = int(len(csv_data_temp)/10)

        interp = interpolation.Interpolation()

        track_interp_left = interp.interpolate_polyline(csv_data_temp[:, 0], csv_data_temp[:, 1], mod*10)
        track_interp_right = interp.interpolate_polyline(csv_data_temp[:, 2], csv_data_temp[:, 3], mod*10)

        self.x1_test.append(track_interp_left[:, 0])
        self.y1_test.append(track_interp_left[:, 1])
        self.x2_test.append(track_interp_right[:, 0])
        self.y2_test.append(track_interp_right[:, 1])
    
        for xi, yi, xf, yf in zip(self.x1_test[0], self.y1_test[0], self.x2_test[0], self.y2_test[0]):
            test_input = np.array([xi, yi, xf, yf])
            test_input = test_input.reshape(1, 1, 4)
            test_output = model.predict(test_input, verbose=1)
            test_output = test_output.reshape(1,2)
            self.x.append(test_output[0][0])
            self.y.append(test_output[0][1])

        self.plotter.plot(self.x1_test, self.y1_test, self.x2_test, self.y2_test, self.x, self.y)