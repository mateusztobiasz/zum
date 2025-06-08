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
    return X_initial, y_initial, X_pool, y_pool, X_test, y_test, X_train_val, y_train_val


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
    }


def plot_metrics(model, strategy, metrics, baseline_metrics):
    train_sizes = [m["train_dataset_size"] for m in metrics]
    metric_names = ["accuracy", "f1_macro", "precision", "recall"]
    metric_values = {metric: [r[metric] for r in metrics] for metric in metric_names}

    plt.figure(figsize=(14, 10))
    for i, metric in enumerate(metric_names, 1):
        plt.subplot(2, 2, i)
        plt.plot(train_sizes, metric_values[metric], marker="o", label=metric.capitalize())
        plt.axhline(
            y=baseline_metrics[metric], color="r", linestyle="--", label="Baseline"
        )
        plt.title(f"{metric.capitalize()} vs Training Set Size")
        plt.xlabel("Training Set Size")
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        plt.xlim(min(train_sizes), max(train_sizes))
        plt.legend()
        plt.grid(True)

    plt.suptitle(f"Model: {model}, Strategy: {strategy}", fontsize=16)
    plt.tight_layout()
    plt.show()
