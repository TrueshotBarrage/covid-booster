import pickle
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.preprocessing import normalize as nm
from sklearn.preprocessing import RobustScaler, StandardScaler


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


def drop_categorical(x):
    """Drop categorical features from the data. In this case, the first col

    Args:
        x (np.ndarray): (n, d) data array
    
    Returns:
        np.ndarray: The treated input data array
    """
    return x[:, 1:]


def drop_feature(x, feature):
    if not isinstance(feature, list):
        assert type(feature) == int, "Feature to drop must be an index (int)"
        feature = [feature]

    return np.delete(x, feature, axis=1)


def prune(x, y=None, remove_rows=True):
    """Prune input array(s) x (and y) to remove NaN values.

    Args:
        x (np.ndarray): (n, d) data array
        y (np.ndarray): (n,) label array, corresponding to x
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
        # imp = IterativeImputer(max_iter=10, sample_posterior=True)
        imp = SimpleImputer(strategy="mean")
        # imp = KNNImputer(n_neighbors=5, weights="distance")
        imp.fit(x)
        x = imp.transform(x)
        if y is None:
            return x
        return x, y


def normalize(x):
    """Normalize the input array by dividing by the mean for each column.
    
    Args:
        x (np.ndarray): (n, d) input data array, where each of the n rows are 
        normalized by every one of its d columns
    
    Returns:
        np.ndarray: (n, d) output data array
    """
    # ss = StandardScaler()
    # x = ss.fit_transform(x)  # Scaling makes it way worse, so for now # out

    x = nm(x, axis=1, norm="max")  # L2 or max seem to work best
    return x