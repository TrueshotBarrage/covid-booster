import pickle
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def load_data(filename="covid_dataset.pkl"):
    # Load the COVID dataset into a dict
    file = open(filename, "rb")
    checkpoint = pickle.load(file)
    file.close()

    # Unpack the numpy arrays from the COVID dataset dict
    xTr, yTr = checkpoint["X_train"], checkpoint["y_train_log_pos_cases"]
    xVal, yVal = checkpoint["X_val"], checkpoint["y_val_log_pos_cases"]
    xTe = checkpoint["X_test"]

    # Cast the numpy arrays into appropriate types
    xTr, yTr, xVal, yVal, xTe = map(lambda arr: arr.astype("float"),
                                    (xTr, yTr, xVal, yVal, xTe))

    return xTr, yTr, xVal, yVal, xTe


def prune(x, y=None, remove_rows=True):
    """Prune input array(s) x (and y) to remove NaN values.

    Args:
        x (np.ndarray): n x d data array
        y (np.ndarray): n x 1 label array, corresponding to x
        remove_rows (bool): If true, remove rows with NaN values, instead of 
        imputing them
    
    Returns:
        x (np.ndarray): The original input data array x with treatment
        [optional] y (np.ndarray): The original input label array y with 
        treatment, if it was provided
    """
    if y is None:
        xy = x
    else:
        xy = np.vstack((x.T, y.T)).T

    nan_rows = np.isnan(xy).any(axis=1)

    if remove_rows:
        xy = xy[~nan_rows]
        if y is None:
            return xy
        return xy.T[:-1].T, xy.T[-1].flatten()
    else:
        imp = IterativeImputer(max_iter=10, sample_posterior=True)
        imp.fit(x)
        x = imp.transform(x)
        if y is None:
            return x
        return x, y

    
