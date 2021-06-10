import keras
from keras import layers
from keras.models import load_model
import numpy as np
import random 
import sys
import glob
import time

from Interpolation import interpolation
from Dataset_Extract.Dataset import Dataset
from Predict import predict
from Model.LSTM.TrainModel.LSTM_Model import LSTM_Model
from Model.GRU.TrainModel.GRU_Model import GRU_Model

Method = 7

ideal_line = False

sentences = [] 

x1_test, x2_test, y1_test, y2_test, xx, yy = [], [], [], [], [], []

test_track, test_track2 = [], []

X, Y = [], []

if __name__ == '__main__':

    dataset = Dataset()
    
    predict = predict.Predict()

    lstm_mod = LSTM_Model()
    gru_mod = GRU_Model()

    if(ideal_line):
        dataset.extract_F1_dataset_IdealLine()
        dataset.reshape_multi_array()
        X, Y = dataset.get_dataset()

    else:
        
        #dataset.extract_AC_dataset()
        #dataset.extract_F1_dataset()
        #dataset.extract_MOD_dataset()
        #dataset.extract_center_trajectory()

        dataset.extract_F1_dataset_IdealLine()
        
        dataset.reshape_one_array()
        X, Y = dataset.get_dataset()


    if(Method == 1):
        lstm_mod.train_model_basic(X, Y)
        predict.test_predict_trajectory_OneArray()

    elif(Method == 2):
        #lstm_mod.train_model_one_array(X, Y)
        predict.test_predict_trajectory_OneArray()

    elif(Method == 3):
        #lstm_mod.train_model_multi_layer(X, Y)
        predict.test_predict_trajectory_OneArray()

    elif(Method == 4):
        gru_mod.train_model_one_array(X, Y)
        predict.test_predict_trajectory_OneArray()

    elif(Method == 5):
        gru_mod.train_model_multi_layer(X, Y)
        predict.test_predict_trajectory_OneArray()

    elif(Method == 6):
        lstm_mod.train_model_multi_layer(X, Y)
        predict.test_predict_trajectory_StepArray()

    elif(Method == 7):
        #lstm_mod.train_model_bidirectional_lstm_one_layer(X, Y)
        predict.test_predict_trajectory_OneArray()

    elif(Method == 8):
        lstm_mod.train_model_bidirectional_lstm_multi_layer(X, Y)
        predict.test_predict_trajectory_OneArray()
    else:
        predict.test_predict_trajectory_OneArray()