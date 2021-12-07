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
algorithms = [LINREG, ADABOOST, FORESTS, KNN, SVM, RIDGEREG, TREES, LASSOREG]


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


def train_and_predict(xTr, yTr, xVal, yVal, xTe, algorithm):
    # Train and build a model
    start_time = time.time()
    model = models.train(xTr, yTr, algorithm)
    elapsed_time = time.time() - start_time
    print(f"Model trained with {_name_of_algorithm(algorithm)}\n"
          f"Time elapsed: {elapsed_time} seconds")

    # Validate the trained model
    val_score = models.validate(xVal, yVal, model)
    print(f"Validation score: {val_score}")

    # Predict the test data labels
    test_pred = models.predict(xTe, model)

    return model, val_score, test_pred


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
    # Validate the arguments from the command line
    args = _init_argparse()
    assert args.algorithm in algorithms, "No such algorithm"

    # Load and preprocess the data
    xTr, yTr, xVal, yVal, xTe = ld.load_data()

    # Impute missing data with the mean of the feature
    xTr, yTr = ld.prune(xTr, yTr, remove_rows=False)
    xVal, yVal = ld.prune(xVal, yVal, remove_rows=False)
    xTe = ld.prune(xTe, remove_rows=False)
    print("Missing data from xTr, xVal, xTe imputed")

    # Remove specific features (e.g. coastline, land utility)
    feature_list = [4, 12]
    xTr = ld.drop_feature(xTr, feature_list)
    xVal = ld.drop_feature(xVal, feature_list)
    xTe = ld.drop_feature(xTe, feature_list)
    print(f"Removed the following features: {feature_list}")

    # Drop the first column (continent index) -- this needs to be tested more
    xTr = ld.drop_categorical(xTr)
    xVal = ld.drop_categorical(xVal)
    xTe = ld.drop_categorical(xTe)
    print("Removed categorical features (first column)")

    # Normalize across each row (NOT column, i.e. each data point normalized)
    xTr = ld.normalize(xTr)
    xVal = ld.normalize(xVal)
    xTe = ld.normalize(xTe)
    print("xTr, xVal, xTe normalized w.r.t. data point")

    # Now for the workout...
    highest_val_score = -1
    mean_val_score = 0
    for _ in range(args.num_iterations):
        model, val_score, test_pred = train_and_predict(xTr, yTr, xVal, yVal,
                                                        xTe, args.algorithm)
        mean_val_score += val_score

        # Keep iteratively replacing the model with a "better" one
        if highest_val_score < val_score:
            highest_val_score = val_score
            best_model = model
            best_test_pred = test_pred
            submit(best_test_pred)

    mean_val_score /= args.num_iterations
    print(f"Best model score: {highest_val_score}")
    print(f"Mean model score: {mean_val_score}")

    # Retrain the model with the validation data (moar data!!!)
    # retrained_model = models.retrain(xVal, yVal, best_model)
    # retrained_test_pred = models.predict(xTe, retrained_model)
    # print("Retrained model with xVal")  # It seems this is a bad idea
    # print(f"1st iteration: {test_pred}\n2nd iteration: {retrained_test_pred}")
    # submit(retrained_test_pred, suffix="_retrained")


if __name__ == "__main__":
    main()