import keras
from keras import layers

class LSTM_Model:

    def __init__(self):
        self.model = keras.models.Sequential()

    def train_model_one_array(self, X, Y):
        self.model.add(layers.LSTM(128, activation='relu', input_shape=(1, 4)))
        self.model.add(layers.Dense(2))
        self.model.compile(optimizer='adam', loss='mse')

        print(self.model.summary(90))

        for f, g in zip(X, Y):
            self.model.fit(f, g, epochs=100, validation_split=0.1, verbose=1)

        #self.model.save('Model/LSTM/Saved/MultiLayer_FullDataset.h5')
        self.model.save('Model/LSTM/Saved/TEST6.h5')


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
        self.model.add(layers.LSTM(64, activation='relu', input_shape=(10, 4), return_sequences=True))
        self.model.add(layers.LSTM(64, activation='relu', input_shape=(10, 4)))
        self.model.add(layers.Dense(2))
        self.model.compile(optimizer='adam', loss='mse')

        print(self.model.summary(90))

        i = 0

        for f, g in zip(X, Y):
            self.model.fit(f, g, epochs=100, validation_split=0.1, verbose=1)
            print(i)
            i = i+1

        self.model.save('Model/LSTM/Saved/TEST8.h5')

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