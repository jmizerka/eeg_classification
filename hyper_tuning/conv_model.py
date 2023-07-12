import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner import HyperModel, RandomSearch

class Conv_HyperModel(HyperModel):
    """
    HyperModel subclass for building a convolutional  neural network architecture with hyperparameter search.

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
        num_conv_layers = hp.Int('num_conv_layers', 3, 4)

        for i in range(num_conv_layers):
            units = hp.Choice(f'units_{i}', [128, 256], default=128) #500
            filters = hp.Choice(f'conv_filters{i}',values=[32,64,96])
            model.add(layers.Conv2D(filters,kernel_size=(1,units),padding='same',activation='elu'))
            model.add(layers.MaxPooling2D(padding='valid', pool_size=(1, hp.Choice(f'pool_size{i}', values=[2, 4]))))
            model.add(layers.BatchNormalization())
        dropout_rate = hp.Float(f'dropout_rate', 0.25, 0.75, step=0.25)
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=self.num_classes))
        model.add(layers.Activation('softmax'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


