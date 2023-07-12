import numpy as np

from preproces.dataset_prep import create_dataset, split_data
from preproces.filters import apply_filter
from preproces.standardize_data import standardize

from hyper_tuning.svm_model import tune_svm
from hyper_tuning.conv_model import Conv_HyperModel
from hyper_tuning.lstm_model import LSTM_HyperModel
from hyper_tuning.dense_model import Dense_HyperModel
from hyper_tuning.keras_tuner import KerasTuner

from models.models import Dense_ModelBuilder,Conv_ModelBuilder,LSTM_ModelBuilder,SVM_ModelBuilder,EEGNet_ModelBuilder

def run_model(sub_num,model_type,length,channels):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2)
    logo = LeaveOneGroupOut()
    groups = np.repeat(np.arange(0, sub_num), X.shape[0] / sub_num) # Creates an array of group labels for groupshufflesplit
    accuracies = []

    for train_val, test in logo.split(X, y, groups):
        X_test, y_test = X[test], y[test]
        X_train_val, y_train_val = X[train_val], y[train_val]

        X_test = standardize(apply_filter(X_test, 6, 40), scaler_type = 'standard') # (6,40) - filterband # scaler_type = ['minmax','stanard']

        for train, val in gss.split(X_train_val, y_train_val, groups[int(groups.shape[0] / 35):]):
            X_train, y_train = X_train_val[train], y_train_val[train]
            X_val, y_val = X_train_val[val], y_train_val[val]
            X_train = standardize(apply_filter(X_train, 6, 36),scaler_type = 'standard')
            X_val = standardize(apply_filter(X_val, 6, 36),scaler_type = 'standard')

            if model_type == 'conv' or model_type == 'eegnet':
                X_train,X_val,X_test = X_train.reshape(-1,channels,length,1),X_val.reshape(-1,channels,length,1),X_test.reshape(-1,channels,length,1)
                if model_type == 'conv':
                    model,earlystop,checkpoint = Conv_ModelBuilder().create_model()
                else:
                    model, earlystop, checkpoint = EEGNet_ModelBuilder().create_model()
            if model_type == 'lstm':
                model, earlystop, checkpoint = LSTM_ModelBuilder().create_model()
            elif model_type == 'dense':
                model, earlystop, checkpoint = Dense_ModelBuilder().create_model()
            if model_type == 'svm':
                svm = SVM_ModelBuilder()
                svm.fit_svm(X_train,y_train)
                X_val = np.concatenate((X_val,X_test),axis=0)
                y_val = np.concatenate((y_val, y_test), axis=0)
                scores = svm.predict_score(X_val,y_val)
            else:
                 model.fit(X_train, y_train, batch_size=64, epochs=500,
                           validation_data=(X_val, y_val), shuffle=True,
                           callbacks=[earlystop,checkpoint])
                 scores = model.evaluate(X_test, y_test, verbose=0)
            accuracies.append(scores[1])
        print(f'Score : {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    print(f'Mean accuracy: {np.mean(accuracies)}')
    return accuracies




LENGTH = 500
SPLIT = 0.8
CLASSES_NUM = 40
SUB_NUM = 35
TRIALS_NUM = 6
ELECTRODE_IDX = [46,51,52,53,54,55,56,57,59,60,61]
MODEL_TYPE = 'lstm' # [lstm,eegnet,conv,dense]


X,y = create_dataset(ELECTRODE_IDX,LENGTH,'data/') # [list of electrodes, length of data, folder_path]
# X_train,X_val,X_test,y_train,y_val,y_test = split_data(X,y, LENGTH,SPLIT,CLASSES_NUM,SUB_NUM,TRIALS_NUM) uncomment for subject-dependent classification


#### HYPERPARAMETER TUNING

#param_grid = {'C': [0.1, 0.5, 1, 5, 10],'kernel':['linear','rbf','poly']}
#tune_svm(X_train,y_train, X_test, y_test, param_grid)
#lstm = LSTM_HyperModel((11,500),40)
#conv = Conv_HyperModel((11,500),40)
#dense = Dense_HyperModel((11,500),40)
#tuner = KerasTuner(15,4,0.0001)
#tuner.build(conv,'dense_tuner')
#tuner.add_earlystop()
#tuner.search_params(X_train,y_train,X_val,y_val,10,callbacks=tuner.earlystop)



#### RUNING MODELS

accuracies = run_model(SUB_NUM, MODEL_TYPE, LENGTH, len(ELECTRODE_IDX))
print(accuracies)






