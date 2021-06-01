import numpy as np
import keras
import glob
from keras.models import load_model

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

        self.plotter = plotting.Plotting()

    def test_predict_trajectory_OneArray(self):

        model = load_model('Model/LSTM/Saved/TEST4.h5')

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

            #interp = interpolation.Interpolation()

            #track_interp_left = interp.interpolate_polyline(csv_data_temp[:, 0], csv_data_temp[:, 1], mod*10)
            #track_interp_right = interp.interpolate_polyline(csv_data_temp[:, 2], csv_data_temp[:, 3], mod*10)

            self.x1_test.append(csv_data_temp[:, 0])
            self.y1_test.append(csv_data_temp[:, 1])
            self.x2_test.append(csv_data_temp[:, 2])
            self.y2_test.append(csv_data_temp[:, 3])
        
            for xi, yi, xf, yf in zip(self.x1_test[0], self.y1_test[0], self.x2_test[0], self.y2_test[0]):
                test_input = np.array([xi, yi, xf, yf])
                test_input = test_input.reshape(1, 1, 4)
                test_output = model.predict(test_input, verbose=1)
                test_output = test_output.reshape(1,2)
                self.x.append(test_output[0][0])
                self.y.append(test_output[0][1])

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