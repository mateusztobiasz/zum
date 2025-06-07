from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def split_data(X, y, initial_size, test_size):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y
    )
    X_initial, X_pool, y_initial, y_pool = train_test_split(
        X_train_val, y_train_val, train_size=initial_size, stratify=y_train_val
    )
    return X_initial, y_initial, X_pool, y_pool, X_test, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
    }
