import time
import argparse

import load_data as ld
import models

# ML algorithms
LINREG = models.LINREG
ADABOOST = models.ADABOOST
FORESTS = models.FORESTS
algorithms = [LINREG, ADABOOST, FORESTS]


def _name_of_algorithm(algorithm):
    if algorithm == LINREG:
        return "Linear Regression"
    if algorithm == FORESTS:
        return "Random Forests"
    if algorithm == ADABOOST:
        return "AdaBoost"


def train_and_predict(xTr, yTr, xVal, yVal, xTe, algorithm):
    # Impute the missing (NaN) data from the dataset
    xTr, yTr = ld.prune(xTr, yTr, remove_rows=False)
    xVal, yVal = ld.prune(xVal, yVal, remove_rows=False)
    xTe = ld.prune(xTe, remove_rows=False)
    print("xTr, xVal, xTe imputed")

    # Train and build a model
    print("Training model...")
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


def submit(pred):
    import pandas as pd
    # Save the prediction array pred into a valid submission file
    pd.DataFrame(pred).to_csv("predictions.csv",
                              header=["cases"],
                              index_label="id")


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
    return parser.parse_args()


def main():
    args = _init_argparse()
    assert args.algorithm in algorithms, "No such algorithm"
    xTr, yTr, xVal, yVal, xTe = ld.load_data()
    model, val_score, test_pred = \
        train_and_predict(xTr, yTr, xVal, yVal, xTe, algorithm=args.algorithm)
    submit(test_pred)


if __name__ == "__main__":
    main()