import keras
from keras import layers
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

class GRU_Model:

    def __init__(self):
        self.model = keras.models.Sequential()

    def train_model_bidirectional_gru_one_layer(self, X, Y):
        self.model.add(layers.Bidirectional(layers.GRU(128, activation='relu'), input_shape=(1, 4)))
        self.model.add(layers.Dense(2))
        self.model.compile(optimizer='adam', loss='mse', metrics=['acc'])

        print(self.model.summary(90))

        for f, g in zip(X, Y):
            
            x_test, y_test = self.extract_validation_data()

            history = self.model.fit(f, g, epochs=100, validation_split=0.1, verbose=1, validation_data=(x_test, y_test))

            #self.acc.append(history.history['acc'])
            #self.val_acc.append(history.history['val_acc'])
            #self.loss.append(history.history['loss'])
            #self.val_loss.append(history.history['val_loss'])
        '''
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

        #plt.figure()

        plt.plot(epochs[0], loss[0], 'bo', label='Training loss')
        plt.plot(epochs[0], val_loss[0], 'b', label='Validaion loss')
        plt.title('Training and validation loss')
        #plt.legend()
        plt.show()
        '''
        self.model.save('Model/GRU/Saved/Bidirectional/OneLayer.h5')

    def train_model_bidirectional_gru_multi_layer(self, X, Y):
        self.model.add(layers.Bidirectional(layers.GRU(64, activation='relu'), input_shape=(1, 4), return_sequences=True))
        self.model.add(layers.Bidirectional(layers.GRU(64, activation='relu'), input_shape=(1, 4)))
        self.model.add(layers.Dense(2))
        self.model.compile(optimizer='adam', loss='mse')

        print(self.model.summary(90))

        for f, g in zip(X, Y):
            history = self.model.fit(f, g, epochs=100, validation_split=0.1, verbose=1)

        self.model.save('Model/GRU/Saved/Bidirectional/MultiLayer.h5')
    
    def train_model_cnn_gru_one_layer(self, X, Y):
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(1, 4)))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(layers.GRU(50, activation='relu', input_shape=(1, 4)))
        self.model.add(layers.Dense(2))
        self.model.compile(optimizer='adam', loss='mse')

        print(self.model.summary(90))

        for f, g in zip(X, Y):
            self.model.fit(f, g, epochs=100, validation_split=0.1, verbose=1)

        self.model.save('Model/GRU/Saved/CNN_GRU.h5')

    def train_model_one_array(self, X, Y):
        self.model.add(layers.GRU(128, activation='relu', input_shape=(1, 4)))
        self.model.add(layers.Dense(2))
        self.model.compile(optimizer='adam', loss='mse')

        print(self.model.summary(90))

        for f, g in zip(X, Y):
            self.model.fit(f, g, epochs=100, validation_split=0.1, verbose=1)

        self.model.save('Model/GRU/Saved/OneLayer.h5')


    def train_model_basic(self, X, Y):
        self.model.add(layers.LSTM(128, input_shape=(10, 4)))
        self.model.add(layers.Dense(2, activation='softmax'))
        optimizer = keras.optimizers.RMSprop(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        print(self.model.summary(90))

        for f, g in zip(X, Y):
            self.model.fit(f, g, epochs=100, validation_split=0.1, verbose=1)

        self.model.save('Model/GRU/Saved/OneLayerBasic.h5')

    def train_model_multi_layer(self, X, Y):
        self.model.add(layers.GRU(64, activation='relu', input_shape=(10, 4), return_sequences=True))
        self.model.add(layers.GRU(64, activation='relu', input_shape=(10, 4)))
        self.model.add(layers.Dense(2))
        self.model.compile(optimizer='adam', loss='mse')

        print(self.model.summary(90))

        for f, g in zip(X, Y):
            self.model.fit(f, g, epochs=100, validation_split=0.1, verbose=1)

        self.model.save('Model/GRU/Saved/MultiLayer.h5')

    def export_model(self):
        return self.model