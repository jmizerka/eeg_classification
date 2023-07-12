import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

class Conv_ModelBuilder():
    """
    A class for creating a convolutional neural network model.

    Parameters:
    lr (float): The learning rate for the optimizer.
    patience (int): The number of epochs with no improvement after which training will be stopped for EarlyStopping.
    units (int): The number of units (classes) in the dense layer.
    chans (int): The number of channels in the input data.

    Methods:
    create_model(): Create and compile the convolutional neural network model.

    """

    def __init__(self, lr=0.0001, patience=4, units=40, chans=11):
        self.lr = lr
        self.patience = patience
        self.units = units
        self.chans = chans

    def create_model(self):
        """
        Create and compile the convolutional neural network model.

        Returns:
        model (Sequential): The built convolutional neural network model.
        earlystop (tf.keras.callbacks.EarlyStopping): The EarlyStopping callback.
        checkpoint (tf.keras.callbacks.ModelCheckpoint): The ModelCheckpoint callback.

        """

        model = Sequential()
        model.add(layers.Conv2D(96, (1, 256), padding='same', activation='elu'))
        model.add(layers.AveragePooling2D((1, 4)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (1, 256), padding='same', activation='elu'))
        model.add(layers.AveragePooling2D((1, 2)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, (1, 128), padding='same', activation='elu'))
        model.add(layers.AveragePooling2D((1, 2)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, (1, 128), padding='same', activation='elu'))
        model.add(layers.AveragePooling2D((1, 2)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=self.units))
        model.add(layers.Activation('softmax'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        earlystop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=self.patience,
            verbose=0
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='./model_conv',
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=False
        )

        return model, earlystop, checkpoint
class Dense_ModelBuilder():
    """
    A class for creating a basic neural network model.

    Parameters:
    lr (float): starting learning rate for the optimizer.
    patience (int): The number of epochs with no improvement after which training will be stopped for EarlyStopping.
    units (int): The number of units/neurons in the last dense layer (classes).
    chans (int): The number of channels in the input data.

    Methods:
    create_model(): Create and compile the neural network model.

    """

    def __init__(self, lr=0.00001, patience=4, units=40, chans=11):
        self.lr = lr
        self.patience = patience
        self.units = units
        self.chans = chans

    def create_model(self):
        """
        Create and compile the neural network model.

        Returns:
        model (Sequential): The built neural network model.
        earlystop (tf.keras.callbacks.EarlyStopping): The EarlyStopping callback.
        checkpoint (tf.keras.callbacks.ModelCheckpoint): The ModelCheckpoint callback.

        """

        model = Sequential()
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(units=512, activation='elu'))
        model.add(layers.Dense(units=512, activation='elu'))
        model.add(layers.Dense(units=self.units, activation='softmax'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        earlystop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=self.patience,
            verbose=0
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='./model_dense',
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=False
        )

        return model, earlystop, checkpoint
class EEGNet_ModelBuilder():
    """
    A class for creating a convolutional neural network model.

    Parameters:
    lr (float): The learning rate for the optimizer.
    patience (int): The number of epochs with no improvement after which training will be stopped for EarlyStopping.
    units (int): The number of units/neurons in the dense layer.
    chan (int): The number of channels in the input data.

    Methods:
    create_model(): Create and compile the convolutional neural network model.

    based on:
    1]. Waytowich, N. et. al. (2018). Compact Convolutional Neural Networks
    for Classification of Asynchronous Steady-State Visual Evoked Potentials.
    Journal of Neural Engineering vol. 15(6).
    http://iopscience.iop.org/article/10.1088/1741-2552/aae5d8

    """

    def __init__(self, lr=0.00009, patience=4, units=40, chans=11):
        self.lr = lr
        self.patience = patience
        self.units = units
        self.chans = chans

    def create_model(self):
        """
        Create and compile the convolutional neural network model.

        Returns:
        model (Sequential): The built convolutional neural network model.
        earlystop (tf.keras.callbacks.EarlyStopping): The EarlyStopping callback.
        checkpoint (tf.keras.callbacks.ModelCheckpoint): The ModelCheckpoint callback.

        """

        model = Sequential()
        model.add(layers.Conv2D(96, (1, 256), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.DepthwiseConv2D((self.chans, 1), depth_multiplier=1, depthwise_constraint=tf.keras.constraints.max_norm(1.)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('elu'))
        model.add(layers.AveragePooling2D((1, 4)))
        model.add(layers.Dropout(0.5))
        model.add(layers.SeparableConv2D(96, (1, 16), use_bias=False, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('elu'))
        model.add(layers.AveragePooling2D((1, 8)))
        model.add(layers.Dropout(0.5))
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(layers.Dense(units=self.units))
        model.add(layers.Activation('softmax'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        earlystop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=self.patience,
            verbose=0
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='./model_eegnet',
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=False
        )

        return model, earlystop, checkpoint
class LSTM_ModelBuilder:
    """
    A class for creating an LSTM model.

    Parameters:
    lr (float): The learning rate for the optimizer.
    patience (int): The number of epochs with no improvement after which training will be stopped for EarlyStopping.
    units (int): The number of units (classes) in the last dense layer.
    chan (int): The number of channels in the input data.

    Methods:
    create_model(): Create and compile the LSTM model.

    """

    def __init__(self, lr=0.00001, patience=4, units=40, chans=11):
        self.lr = lr
        self.patience = patience
        self.units = units
        self.chans = chans

    def create_model(self):
        """
        Create and compile the LSTM model.

        Returns:
        model (Sequential): The built LSTM model.
        earlystop (tf.keras.callbacks.EarlyStopping): The EarlyStopping callback.

        """

        model = keras.Sequential()
        model.add(layers.LSTM(224, input_shape=(self.chans, 500), return_sequences=True))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=self.units, activation='elu'))
        model.add(layers.Dense(40, activation='softmax'))

        optimizer = keras.optimizers.Adam(self.lr)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        earlystop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=self.patience,
            verbose=0
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='./model_lstm',
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=False
        )
        return model, earlystop, checkpoint
class SVM_ModelBuilder():
    """
    A class for building and evaluating an SVM model.

    Parameters:
    X_train: The training data.
    y_train: The training labels.
    X_val: The validation data.
    y_val: The validation labels.

    Methods:
    fit_svm(): Fit the SVM model on the training data.
    predict_score(): Predict the labels for the validation data and calculate the accuracy score.

    """
    def __init__(self):
        self.svm = SVC()
    def fit_svm(self,X_train,y_train):
        X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2])
        self.svm.fit(X_train,y_train)
    def predict_score(self,X_val,y_val):
        X_val = X_val.reshape(-1, X_val.shape[1], X_val.shape[2])
        y_pred = self.svm.predict(X_val)
        score = accuracy_score(y_val,y_pred)
        print(f'Accuracy: {score}')
        return score
