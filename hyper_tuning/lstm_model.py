import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner import HyperModel,RandomSearch

class LSTM_HyperModel(HyperModel):
    """
    Build an LSTM-based model for sequence classification with hyperparameter search.

    Parameters:
    hp (HyperParameters): The hyperparameters object provided by the Kerastuner.

    Returns:
    model (keras.Sequential): The built LSTM-based model.

    """
    def __init__(input_shape,num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
    def build_model(self, hp):


        model = keras.Sequential()
        model.add(layers.LSTM(units=hp.Int('units', min_value=64, max_value=256, step=32),
                              input_shape=self.input_shape,
                              return_sequences=True))
        model.add(layers.Dropout(hp.Float('dropout_1', min_value=0.3, max_value=0.7, step=0.1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=hp.Int('dense_units', min_value=64, max_value=512, step=64), activation='elu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4,1e-5])),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
