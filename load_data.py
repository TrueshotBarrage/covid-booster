import pickle
import numpy as np

import naive_bayes as nb

# Load the COVID dataset into a dict
file = open("covid_dataset.pkl", "rb")
checkpoint = pickle.load(file)
file.close()

# Unpack the numpy arrays from the COVID dataset dict
X_train, y_train = checkpoint["X_train"], checkpoint["y_train_log_pos_cases"]
X_val, y_val = checkpoint["X_val"], checkpoint["y_val_log_pos_cases"]
X_test = checkpoint["X_test"]

# print(f"X_train: {X_train.shape}")
# print(f"y_train: {y_train.shape}")
# print(f"X_val: {X_val.shape}")
# print(f"y_val: {y_val.shape}")
# print(f"X_test: {X_test.shape}")

# print(f"X_train[0]: {X_train[0]}")
# print(f"y_train[0]: {y_train[0]}")


def main():
    w, b = nb.naivebayesCL(X_train, y_train, nb.naivebayesPXY_smoothing)
    train_error = np.mean(nb.classifyLinear(X_train, w, b) != y_train)
    test_error = np.mean(nb.classifyLinear(X_val, w, b) != y_val)

    print("Training error (Smoothing with Laplace estimate): %.2f%%" %
          (100 * train_error))
    print("Test error (Smoothing with Laplace estimate): %.2f%%" %
          (100 * test_error))


if __name__ == "__main__":
    main()