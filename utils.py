from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


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


def plot_metrics(metrics, baseline_metrics=None):
    train_sizes = [m["train_dataset_size"] for m in metrics]
    metric_names = ["accuracy", "f1_macro", "precision", "recall"]

    plt.figure(figsize=(10, 6))

    for name in metric_names:
        values = [m[name] for m in metrics]
        # Plot metric line and get the line color
        line, = plt.plot(train_sizes, values, marker='o', label=name)
        color = line.get_color()

        # Draw baseline line with the same color
        if baseline_metrics and name in baseline_metrics:
            plt.axhline(
                y=baseline_metrics[name],
                color=color,
                linestyle='--',
                linewidth=1,
                label=f"{name} baseline"
            )

    plt.xlabel("Training Set Size")
    plt.ylabel("Score")
    plt.title("Model Performance vs Training Set Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()