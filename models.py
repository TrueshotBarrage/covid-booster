import numpy as np
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso

# Constants to identify ML algorithms.
LINREG = 0
ADABOOST = 1
FORESTS = 2
KNN = 3
SVM = 4
RIDGEREG = 5
TREES = 6
LASSOREG = 7
GRADBOOST = 8
XGBOOST = 9


def train(xTr, yTr, algorithm, fit=True, bagging=True):
    """Train the model with labeled data. 
    Assumption: data is not invalid (e.g. no NaN values) and labels are correct.

    Args:
        xTr (np.ndarray): (n, d) input data array
        yTr (np.ndarray): (n,) input label array
        algorithm (enumerated int): The type of algorithm to train the model
        fit (bool): If true, fit xTr and yTr to the algorithm model
        bagging (bool): If true, use bagging
    
    Returns:
        Regressor: The trained regression model
    """
    if algorithm == LINREG:
        regr = LinearRegression()
    elif algorithm == ADABOOST:
        regr = AdaBoostRegressor(n_estimators=100, random_state=0)
    elif algorithm == FORESTS:
        regr = RandomForestRegressor(max_features=5, max_depth=4)
        #  random_state=0)
    elif algorithm == KNN:
        regr = KNeighborsRegressor(n_neighbors=5)
    elif algorithm == SVM:
        regr = SVR(kernel="rbf", C=8.7, epsilon=0.99)
    elif algorithm == RIDGEREG:
        regr = BayesianRidge()
    elif algorithm == TREES:
        regr = DecisionTreeRegressor(max_depth=10)
    elif algorithm == LASSOREG:
        regr = Lasso(alpha=0.1)
    elif algorithm == GRADBOOST:
        regr = GradientBoostingRegressor(n_estimators=200,
                                         max_features=8,
                                         max_depth=4,
                                         learning_rate=0.0686648845)
    elif algorithm == XGBOOST:
        regr = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.1)

    if bagging:
        regr = BaggingRegressor(base_estimator=regr,
                                n_estimators=15,
                                random_state=0)
    # Train the model with the data
    if fit:
        regr.fit(xTr, yTr)

    return regr


def retrain(xTr, yTr, model):
    """Retrain the model with labeled data. 
    Assumption: data is not invalid (e.g. no NaN values) and labels are correct.

    Args:
        xTr (np.ndarray): (n, d) input data array
        yTr (np.ndarray): (n,) input label array
        model (Regressor): Trained regression model to be updated
    
    Returns:
        Regressor: The retrained regression model
    """
    # Train the model with the data
    model.fit(xTr, yTr)

    return model


def validate(xVal, yVal, model):
    """Validate a trained model with validation data and show the performance.
    
    Args:
        xVal (np.ndarray): (n, d) input data array
        yVal (np.ndarray): (n,) input label array
        model (Regressor): Regression model, trained with an ML algorithm
    
    Returns:
        float: Validation score R^2
        float: Mean squared log error
    """

    # val_score = model.score(xVal, yVal)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    val_score = np.absolute(
        cross_val_score(model,
                        xVal,
                        yVal,
                        cv=cv,
                        scoring="neg_mean_squared_error",
                        n_jobs=-1)).mean()
    rmsle = mean_squared_error(yVal, model.predict(xVal))
    return val_score, rmsle


def grid_search(x, y, model=XGBRegressor()):
    # regr = GridSearchCV(estimator=model,
    #              param_grid={
    #                  "C": [1, 10],
    #                  "epsilon": [0.1, 0.5],
    #                  "kernel": ["rbf", "linear"]
    #              })
    # regr = GridSearchCV(
    #     estimator=model,
    #     param_grid={
    #         # "C": np.linspace(1, 50, num=20),
    #         # "epsilon": np.logspace(0.00001, 0.001, num=3),
    #         # "kernel": ["rbf"]
    #         "max_features": [4],
    #         "max_depth": [4, 5],
    #         "learning_rate": np.logspace(-1, -5, num=50)
    #     })
    regr = GridSearchCV(estimator=model,
                        param_grid={
                            "n_estimators": [100, 150, 200],
                            "max_depth": [3, 4],
                            "learning_rate": [0.1, 0.3, 0.5, 0.7]
                        })

    regr.fit(x, y)

    return regr


def predict(xTe, model):
    """Predict regression label values for a dataset using a trained model.

    Args:
        xTe (np.ndarray): (n, d) input data array
        model (Regressor): Regression model, trained with an ML algorithm

    Returns:
        np.ndarray: (n,) predicted label array for xTe
    """
    return model.predict(xTe)
