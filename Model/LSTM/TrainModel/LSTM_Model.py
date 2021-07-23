import keras
from keras import layers
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import matplotlib.pyplot as plt
import glob
import numpy as np
from Interpolation import interpolation

class LSTM_Model:

    def __init__(self):
        self.model = keras.models.Sequential()

        self.x1, self.x2, self.y1, self.y2 = [], [], [], []
        self.x_goal, self.y_goal = [], []
        self.X, self.Y = [], []
        self.myNewX, self.myNewY = [], []

        self.dataset_files, self.dataset_files_tra, self.dataset_files2, self.dataset_files3 = [], [], [], []

        self.interpolation = interpolation.Interpolation()
        self.acc, self.loss, self.val_acc, self.val_loss = [], [], [], []

    def train_model_bidirectional_lstm_one_layer(self, X, Y):
        self.model.add(layers.Bidirectional(layers.LSTM(128, activation='relu'), input_shape=(1, 4)))
        self.model.add(layers.Dense(2))
        self.model.compile(optimizer='adam', loss='mse', metrics=['acc'])

        print(self.model.summary(90))

        x_test, y_test = self.extract_validation_data_AC()

        for f, g in zip(X, Y):

            history = self.model.fit(f, g, epochs=100, validation_split=0.1, verbose=1, validation_data=(x_test, y_test))

            self.acc.append(history.history['acc'])
            self.val_acc.append(history.history['val_acc'])
            self.loss.append(history.history['loss'])
            self.val_loss.append(history.history['val_loss'])
        
        x = np.array(self.acc)
        x = np.matrix(x.reshape(len(x), 100))
        acc = x.mean(0)
        acc = np.array(acc[0])

        x = np.array(self.val_acc)
        x = np.matrix(x.reshape(len(x), 100))
        val_acc = x.mean(0)
        val_acc = np.array(val_acc[0])

        x = np.array(self.loss)
        x = np.matrix(x.reshape(len(x), 100))
        loss = x.mean(0).reshape(1, 100)
        loss = np.array(loss[0])

        x = np.array(self.val_loss)
        x = np.matrix(x.reshape(len(x), 100))
        val_loss = x.mean(0).reshape(1, 100)
        val_loss = np.array(val_loss[0])

        epochs = range(1, 101)

        epochs = np.array(epochs)
        epochs = epochs.reshape(1, 100)
        
        plt.plot(epochs[0], acc[0], 'bo', label='Training acc')
        plt.plot(epochs[0], val_acc[0], 'b', label='Validation acc')
        plt.title('training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs[0], loss[0], 'bo', label='Training loss')
        plt.plot(epochs[0], val_loss[0], 'b', label='Validaion loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        
        self.model.save('Model/LSTM/Saved/Bidirectional/rotate.h5')

    def train_model_bidirectional_lstm_multi_layer(self, X, Y):
        self.model.add(layers.Bidirectional(layers.LSTM(64, activation='relu'), input_shape=(1, 4)))
        self.model.add(layers.Bidirectional(layers.LSTM(64, activation='relu'), input_shape=(1, 4)))
        self.model.add(layers.Dense(2))
        self.model.compile(optimizer='adam', loss='mse')

        print(self.model.summary(90))

        for f, g in zip(X, Y):
            history = self.model.fit(f, g, epochs=100, validation_split=0.1, verbose=1)
            
            acc = history.history['acc']
            val_acc = history.history['val_acc']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs = range(1, len(acc) + 1)

            plt.plot(epochs, acc, 'bo', label='Training acc')
            plt.plot(epochs, val_acc, 'b', label='Validation acc')
            plt.title('training and validation accuracy')
            plt.legend()

            plt.figure()

            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'b', label='Validaion loss')
            plt.title('Training and validation loss')
            plt.legend()

            plt.show()

        self.model.save('Model/LSTM/Saved/Bidirectional/MultiLayer.h5')
    
    def train_model_cnn_lstm_one_layer(self, X, Y):
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=( 1, 1, 4)))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(layers.LSTM(128, activation='relu', input_shape=(1, 4)))
        self.model.add(layers.Dense(2))
        self.model.compile(optimizer='adam', loss='mse')

        print(self.model.summary(90))

        for f, g in zip(X, Y):
            self.model.fit(f, g, epochs=100, validation_split=0.1, verbose=1)

        #self.model.save('Model/LSTM/Saved/MultiLayer_FullDataset.h5')
        self.model.save('Model/LSTM/Saved/CNN_LSTM.h5')

    def train_model_one_array(self, X, Y):
        self.model.add(layers.LSTM(128, activation='relu', input_shape=(1, 4)))
        self.model.add(layers.Dense(2))
        self.model.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])
        
        x_test, y_test = self.extract_validation_data_AC()

        print(self.model.summary(90))

        for f, g in zip(X, Y):
            history = self.model.fit(f, g, epochs=100, validation_split=0.1, verbose=1, validation_data=(x_test, y_test))

            #self.acc.append(history.history['acc'])
            #self.val_acc.append(history.history['val_acc'])
            self.loss.append(history.history['loss'])
            self.val_loss.append(history.history['val_loss'])
        '''
        x = np.array(self.acc)
        x = np.matrix(x.reshape(len(x), 100))
        acc = x.mean(0)
        acc = np.array(acc[0])

        x = np.array(self.val_acc)
        x = np.matrix(x.reshape(len(x), 100))
        val_acc = x.mean(0)
        val_acc = np.array(val_acc[0])
        '''
        x = np.array(self.loss)
        x = np.matrix(x.reshape(len(x), 100))
        loss = x.mean(0).reshape(1, 100)
        loss = np.array(loss[0])

        x = np.array(self.val_loss)
        x = np.matrix(x.reshape(len(x), 100))
        val_loss = x.mean(0).reshape(1, 100)
        val_loss = np.array(val_loss[0])

        epochs = range(1, 101)

        epochs = np.array(epochs)
        epochs = epochs.reshape(1, 100)
        '''
        plt.plot(epochs[0], acc[0], 'bo', label='Training acc')
        plt.plot(epochs[0], val_acc[0], 'b', label='Validation acc')
        plt.title('training and validation accuracy')
        plt.legend()

        plt.figure()
        '''
        plt.plot(epochs[0], loss[0], 'bo', label='Training loss')
        plt.figure()
        plt.legend()

        plt.plot(epochs[0], val_loss[0], 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()
        #self.model.save('Model/LSTM/Saved/MultiLayer_FullDataset.h5')
        self.model.save('Model/LSTM/Saved/TEST10.h5')


    def train_model_basic(self, X, Y):
        self.model.add(layers.LSTM(128, input_shape=(10, 4)))
        self.model.add(layers.Dense(2, activation='softmax'))
        optimizer = keras.optimizers.RMSprop(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        print(self.model.summary(90))

        for f, g in zip(X, Y):
            self.model.fit(f, g, epochs=100, validation_split=0.1, verbose=1)

        self.model.save('Model/LSTM/Saved/OneLayerBasic.h5')

    def train_model_multi_layer(self, X, Y):
        self.model.add(layers.LSTM(64, activation='relu', input_shape=(1, 4), return_sequences=True))
        self.model.add(layers.LSTM(64, activation='relu', input_shape=(1, 4)))
        self.model.add(layers.Dense(2))
        self.model.compile(optimizer='adam', loss='mse')

        print(self.model.summary(90))

        for f, g in zip(X, Y):
            self.model.fit(f, g, epochs=100, validation_split=0.1, verbose=1)


        self.model.save('Model/LSTM/Saved/TEST1.h5')

    def train_model_step_array(self, X, Y):
        self.model.add(layers.LSTM(128, activation='relu', input_shape=(10, 4)))
        self.model.add(layers.Dense(20))
        self.model.compile(optimizer='adam', loss='mse')

        print(self.model.summary(90))

        for f, g in zip(X, Y):
            self.model.fit(f, g, epochs=100, validation_split=0.1, verbose=1)

        self.model.save('Model/LSTM/Saved/MultiStep_OneLayer.h5')

    def export_model(self):
        return self.model

    def extract_validation_data(self):
        for document in glob.glob('Predict/ValidationData/*_track.csv'):
            self.dataset_files.append(document)

        for document in glob.glob('Predict/ValidationData/*_trajectory.csv'):
            self.dataset_files_tra.append(document)

        for file, file1 in zip(self.dataset_files, self.dataset_files_tra):
            csv_data_temp = np.loadtxt(file, comments='#', delimiter=',')
            csv_data_temp2 = np.loadtxt(file1, comments='#', delimiter=',')

            mod = int(len(csv_data_temp) / 10)

            interp_track_left = self.interpolation.interpolate_polyline(csv_data_temp[:, 0], csv_data_temp[:, 1], int(len(csv_data_temp2)))
            interp_track_right = self.interpolation.interpolate_polyline(csv_data_temp[:, 2], csv_data_temp[:, 3], int(len(csv_data_temp2)))


            self.x1.append(interp_track_left[:, 0])
            self.y1.append(interp_track_left[:, 1])
            self.x2.append(interp_track_right[:, 0])
            self.y2.append(interp_track_right[:, 1])

            self.x_goal.append(csv_data_temp2[:, 0])
            self.y_goal.append(csv_data_temp2[:, 1])

            for i in range(0, len(self.x1)):
                self.X.append(np.column_stack((self.x1[i], self.y1[i], self.x2[i], self.y2[i])))
                self.myNewX.append(self.X[i].reshape(len(self.X[i]),1,4))

                self.Y.append(np.column_stack((self.x_goal[i], self.y_goal[i])))
                self.myNewY.append(self.Y[i].reshape(len(self.Y[i]), 2))

            return self.myNewX, self.myNewY

    def extract_validation_data_AC(self):
        for document2 in glob.glob('AC_Dataset/*_side_l.csv'):
            self.dataset_files2.append(document2)

        for document2 in glob.glob('AC_Dataset/*_side_r.csv'):
            self.dataset_files3.append(document2)

        for i in range(0, len(self.dataset_files2)):

                csv_data_temp1 = np.loadtxt(self.dataset_files2[i], comments='#', delimiter=',')

                csv_data_temp2 = np.loadtxt(self.dataset_files3[i], comments='#', delimiter=',')
                
                if(len(csv_data_temp1[:, 0]) >= len(csv_data_temp2[:, 0])):

                    mod = int(len(csv_data_temp1[:, 0])/10)

                    interp_path_left = self.interpolation.interpolate_polyline(csv_data_temp1[:, 0], csv_data_temp1[:, 2], mod*10)
                    interp_path_right = self.interpolation.interpolate_polyline(csv_data_temp2[:, 0], csv_data_temp2[:, 2], mod*10)

                    self.x1.append(interp_path_left[:, 0])
                    self.y1.append(interp_path_left[:, 1])

                    self.x2.append(interp_path_right[:, 0])
                    self.y2.append(interp_path_right[:, 1])

                else:
                    mod = int(len(csv_data_temp2[:, 0])/10)

                    interp_path_left = self.interpolation.interpolate_polyline(csv_data_temp1[:, 0], csv_data_temp1[:, 2], mod*10)
                    interp_path_right = self.interpolation.interpolate_polyline(csv_data_temp2[:, 0], csv_data_temp2[:, 2], mod*10)

                    self.x1.append(interp_path_left[:, 0])
                    self.y1.append(interp_path_left[:, 1])

                    self.x2.append(interp_path_right[:, 0])
                    self.y2.append(interp_path_right[:, 1])

        for i in range(0, len(self.x1)):
            self.x_goal.append((self.x1[i] + self.x2[i]) / 2)
            self.y_goal.append((self.y1[i] + self.y2[i]) / 2)

        for i in range(0, len(self.x1)):
            self.X.append(np.column_stack((self.x1[i], self.y1[i], self.x2[i], self.y2[i])))
            self.myNewX.append(self.X[i].reshape(len(self.X[i]),1,4))

            self.Y.append(np.column_stack((self.x_goal[i], self.y_goal[i])))
            self.myNewY.append(self.Y[i].reshape(len(self.Y[i]), 2))

        return self.myNewX, self.myNewY