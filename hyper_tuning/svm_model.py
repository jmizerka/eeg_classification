from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def tune_svm(X_train,y_train,X_test,y_test,param_grid):
    '''
    performs grid search to optimize svm parameters
    possible parameters can be checked in sklearn docs e.g.:

    'C': regularization parameter, the degree of regularization is inverse to C
    'kernel':  refers to data transformation function

    returns: dict with the best values for each parameter 

    '''
    svm = SVC()
    grid_search = GridSearchCV(svm,param_grid,cv=5)
    grid_search.fit(X_train,y_train)
    best_params = grid_search.best_params_
    print('The best params: ',best_params)
    print('The best accuracy: ', grid_search.score(X_test,y_test)) 
    
    return grid_search.best_params_


