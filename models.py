from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Constants to identify ML algorithms.
LINREG = 0
ADABOOST = 1
FORESTS = 2


def train(xTr, yTr, algorithm):
    """Train the model with labeled data. 
    Assumption: data is not invalid (e.g. no NaN values) and labels are correct.

    Args:
        xTr (np.ndarray): (n, d) input data array
        yTr (np.ndarray): (n,) input label array
        algorithm (enumerated int): The type of algorithm to train the model
    
    Returns:
        Regressor: The trained regression model
    """
    if algorithm == LINREG:
        regr = LinearRegression()
    elif algorithm == ADABOOST:
        regr = AdaBoostRegressor(n_estimators=100, random_state=0)
    elif algorithm == FORESTS:
        regr = RandomForestRegressor(max_depth=2, random_state=0)

    # Train the model with the data
    regr.fit(xTr, yTr)

    return regr


def validate(xVal, yVal, model):
    """Validate a trained model with validation data and show the performance.
    
    Args:
        xVal (np.ndarray): (n, d) input data array
        yVal (np.ndarray): (n,) input label array
        model (Regressor): Regression model, trained with an ML algorithm
    
    Returns:
        float: How well the model performs on the input data
    """
    return model.score(xVal, yVal)


def predict(xTe, model):
    """Predict regression label values for a dataset using a trained model.

    Args:
        xTe (np.ndarray): (n, d) input data array
        model (Regressor): Regression model, trained with an ML algorithm

    Returns:
        np.ndarray: (n,) predicted label array for xTe
    """
    return model.predict(xTe)
