import time
import argparse

import load_data as ld
import models

# ML algorithms
LINREG = models.LINREG
ADABOOST = models.ADABOOST
FORESTS = models.FORESTS
KNN = models.KNN
SVM = models.SVM
RIDGEREG = models.RIDGEREG
TREES = models.TREES
LASSOREG = models.LASSOREG
GRADBOOST = models.GRADBOOST
XGBOOST = models.XGBOOST
algorithms = [
    LINREG, ADABOOST, FORESTS, KNN, SVM, RIDGEREG, TREES, LASSOREG, GRADBOOST,
    XGBOOST
]


# Maps the enumerated type of algorithm to its string representation
def _name_of_algorithm(algorithm):
    if algorithm == LINREG:
        return "Linear Regression"
    if algorithm == FORESTS:
        return "Random Forests"
    if algorithm == ADABOOST:
        return "AdaBoost"
    if algorithm == KNN:
        return "K-Nearest Neighbors"
    if algorithm == SVM:
        return "SVM"
    if algorithm == RIDGEREG:
        return "Bayesian Ridge Regression"
    if algorithm == TREES:
        return "Decision Trees"
    if algorithm == LASSOREG:
        return "Lasso Regression"
    if algorithm == GRADBOOST:
        return "Gradient Boosting Regression"
    if algorithm == XGBOOST:
        return "XGBoost"


def train_and_predict(xTr, yTr, xVal, yVal, xTe, algorithm):
    # Train and build a model
    start_time = time.time()
    model = models.train(xTr, yTr, algorithm)
    elapsed_time = time.time() - start_time
    print(f"Model trained with {_name_of_algorithm(algorithm)}\n"
          f"Time elapsed: {elapsed_time} seconds")

    # Validate the trained model
    val_score, rmsle = models.validate(xVal, yVal, model)
    print(f"Validation score: {val_score}, MSLE: {rmsle}")

    # Predict the test data labels
    test_pred = models.predict(xTe, model)

    return model, val_score, rmsle, test_pred


# Save the prediction array pred into a valid submission file
def submit(pred, suffix=""):
    import pandas as pd
    pd.DataFrame(pred).to_csv(f"predictions{suffix}.csv",
                              header=["cases"],
                              index_label="id")


# Initialize the command line argument parser
def _init_argparse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    help_str = ""
    for a in algorithms:
        help_str += f"({a}) {_name_of_algorithm(a)}\n"
    parser.add_argument("-a",
                        "--algorithm",
                        default=LINREG,
                        type=int,
                        help=help_str)
    parser.add_argument("-n",
                        "--num_iterations",
                        default=100,
                        type=int,
                        help="The number of times to train the model")
    return parser.parse_args()


def main():
    # Debug flag
    debug = False

    # Validate the arguments from the command line
    args = _init_argparse()
    assert args.algorithm in algorithms, "No such algorithm"

    # Load and preprocess the data
    xTr, yTr, xVal, yVal, xTe = ld.load_data()

    # Encode region data with one-hot encoder
    # xTr = ld.drop_categorical(xTr, encode=True)
    # xVal = ld.drop_categorical(xVal, encode=True)
    # xTe = ld.drop_categorical(xTe, encode=True)
    # print("Encoded features using a one-hot encoder")
    # print(xTr.shape, xVal.shape, xTe.shape)

    # Impute missing data with the mean of the feature
    xTr, yTr = ld.prune(xTr, yTr, remove_rows=False)
    xVal, yVal = ld.prune(xVal, yVal, remove_rows=False)
    xTe = ld.prune(xTe, remove_rows=False)
    print("Missing data from xTr, xVal, xTe imputed")

    # Remove specific features (e.g. coastline, land utility)
    feature_list = [0, 11, 18, 19]
    xTr = ld.drop_feature(xTr, feature_list)
    xVal = ld.drop_feature(xVal, feature_list)
    xTe = ld.drop_feature(xTe, feature_list)
    print(f"Removed the following features: {feature_list}")

    # Drop the first column (continent index) -- this needs to be tested more
    # xTr = ld.drop_categorical(xTr, encode=True)
    # xVal = ld.drop_categorical(xVal, encode=True)
    # xTe = ld.drop_categorical(xTe, encode=True)
    # print("Encoded features using a one-hot encoder")
    # print("Removed categorical features (first column)")

    # Normalize across each row (NOT column, i.e. each data point normalized)
    xTr = ld.normalize(xTr)
    xVal = ld.normalize(xVal)
    xTe = ld.normalize(xTe)
    print("xTr, xVal, xTe normalized w.r.t. data point")

    # Do grid search if debugging
    if debug:
        print("Starting grid search")
        searched_model = models.grid_search(
            xTr,
            yTr,
            # model=models.train(_, _, SVM,
            # fit=False)
        )
        print(searched_model.best_params_)
        print(vars(searched_model))
        return

    # Now for the workout...
    highest_val_score = -1
    mean_val_score = 0
    lowest_rmsle = 100000
    mean_rmsle = 0
    for _ in range(args.num_iterations):
        model, val_score, rmsle, test_pred = train_and_predict(
            xTr, yTr, xVal, yVal, xTe, args.algorithm)
        mean_val_score += val_score
        mean_rmsle += rmsle

        # Keep iteratively replacing the model with a "better" one
        if highest_val_score < val_score:
            highest_val_score = val_score
        if lowest_rmsle > rmsle:
            lowest_rmsle = rmsle
            best_model = model
            best_test_pred = test_pred
            # submit(best_test_pred)

    mean_val_score /= args.num_iterations
    mean_rmsle /= args.num_iterations
    print(f"\nBest model score: {highest_val_score}")
    print(f"Mean model score: {mean_val_score}")
    print(f"Best MSLE: {lowest_rmsle}")
    print(f"Mean MSLE: {mean_rmsle}")

    tr_score, tr_rmsle = models.validate(xTr, yTr, model)
    print(f"tr_score: {tr_score}, tr_rmsle: {tr_rmsle}")

    import numpy as np
    # print(xTr.shape)
    # print(xVal.shape)
    xTr = np.concatenate((xTr, xVal), axis=0)
    yTr = np.concatenate((yTr, yVal), axis=0)

    model, val_score, rmsle, test_pred = train_and_predict(
        xTr, yTr, xVal, yVal, xTe, args.algorithm)
    submit(test_pred, suffix="_gradboost")

    # import numpy as np
    # np.savetxt("xTr.csv", xTr, delimiter=",")

    # Retrain the model with the validation data (moar data!!!)
    # retrained_model = models.retrain(xVal, yVal, best_model)
    # retrained_test_pred = models.predict(xTe, retrained_model)
    # print("Retrained model with xVal")  # It seems this is a bad idea
    # print(f"1st iteration: {test_pred}\n2nd iteration: {retrained_test_pred}")
    # submit(retrained_test_pred, suffix="_retrained")


if __name__ == "__main__":
    main()