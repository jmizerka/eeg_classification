from kerastuner import RandomSearch
class KerasTuner():
    """
    A class for hyperparameter tuning using KerasTuner.

    Parameters:
    max_trials (int): The maximum number of trials for the hyperparameter search.
    patience (int): The number of epochs with no improvement after which training will be stopped for Ea>
    min_delta (float): The minimum change in the monitored quantity to qualify as an improvement for Ear>

    Methods:
    build(hypermodel, project_name): Builds the KerasTuner RandomSearch object with the given hypermodel>
    add_earlystop(): Adds EarlyStopping callback with the specified patience and min_delta values.
    search_params(X_train, y_train, X_val, y_val, epochs, callbacks='None'): Performs hyperparameter sea>
    return_best(): Retrieves the best hyperparameters found during the search and returns their values.
    """

    def __init__(self,max_trials,patience,min_delta):
        self.max_trials = max_trials
        self.patience = patience
        self.min_delta = min_delta

    def build(self,hypermodel,project_name):
        """
        Builds the KerasTuner RandomSearch object with the given hypermodel and project name.

        Parameters:
        hypermodel: The hypermodel object for building the tuner.
        project_name (str): The name of the project for the tuner.

        """
        self.tuner = RandomSearch(hypermodel, objective='val_loss', max_trials=self.max_trials, directory='tuners',project_name=project_name)

    def add_earlystop(self):
        """
        Adds EarlyStopping callback with the specified patience and min_delta values.

        """


        self.earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=self.min_delta,patience=self.patience,verbose=0)
    def search_params(self,X_train,y_train,X_val,y_val,epochs,callbacks='None'):
        """
        Performs hyperparameter search using the tuner on the provided training and validation data.

        Parameters:
        X_train: The training data.
        y_train: The training labels.
        X_val: The validation data.
        y_val: The validation labels.
        epochs (int): The number of epochs for training.
        callbacks: Optional callbacks to be used during training.

        """

        self.tuner.search(X_train,y_train,validation_data=(X_val,y_val),epochs=epochs,callbacks=self.earlystop)
    def return_best(self):
        """
        Retrieves the best hyperparameters found during the search and returns their values.

        Returns:
        best_hyperparameters.values: The values of the best hyperparameters.

        """

        best_hyperparameters = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        print(best_hyperparameters.values)
        return best_hyperparameters.values




