import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner import HyperModel, RandomSearch

class Dense_HyperModel(HyperModel):
    """
    HyperModel subclass for building a dense neural network architecture with hyperparameter search.

    Parameters:
    input_shape (tuple): The shape of the input data.
    num_classes (int): The number of output classes.

    Methods:
    build(hp): Build a dense neural network architecture based on hyperparameter search.

    """

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        """
        Build a dense neural network architecture based on hyperparameter search.

        Parameters:
        hp (HyperParameters): The hyperparameters object provided by the Kerastuner.

        Returns:
        model (keras.Sequential): The built dense neural network model.

        """




        model = keras.Sequential()
        model.add(layers.Input(shape=self.input_shape))
        model.add(layers.Flatten())
        num_dense_layers = hp.Int('num_dense_layers', 2, 6)

        for i in range(num_dense_layers):
            units = hp.Choice(f'units_{i}', [128, 256, 512,1024], default=128)
            model.add(layers.Dense(units, activation='elu'))

        dropout_rate = hp.Float('dropout_rate', 0.25, 0.75, step=0.25)
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(self.num_classes, activation='softmax'))

        learning_rate = hp.Choice('learning_rate', [0.01, 0.001, 0.0001, 0.00001])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model
