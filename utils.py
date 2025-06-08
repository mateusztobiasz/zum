from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from collections import defaultdict


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


def plot_strategy_comparison(results_all, metric_name):
    model_results = defaultdict(list)
    for result in results_all:
        model_results[result["model"]].append(result)

    for model_name, model_group in model_results.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{model_name} - {metric_name} over AL iterations")
        ax.set_xlabel("Train dataset size")
        ax.set_ylabel(metric_name)

        baseline_value = None
        styles = ['-', '--', '-.', ':']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        for idx, result in enumerate(model_group):
            strategy_name = result["strategy"]
            steps = result["results"]

            if not steps:
                continue

            x_values = [step["train_dataset_size"] for step in steps]
            y_values = [step[metric_name] for step in steps]

            ax.plot(
                x_values,
                y_values,
                label=strategy_name,
                linestyle=styles[idx % len(styles)],
                color=colors[idx % len(colors)]
            )

            if baseline_value is None:
                baseline_value = result["baseline"][metric_name]

        if baseline_value is not None:
            ax.axhline(
                y=baseline_value,
                color="black",
                linestyle="--",
                label="baseline"
            )

        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        plt.show()