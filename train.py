import numpy as np
from tqdm import tqdm
from utils import evaluate_model


class ActiveLearner:
    def __init__(self, model, query_strategy):
        self.model = model
        self.query_strategy = query_strategy

    def train(
        self,
        X_train,
        y_train,
        X_pool,
        y_pool,
        X_test,
        y_test,
        n_queries=10,
        query_size=10,
    ):
        metrics = []

        for _ in tqdm(range(n_queries)):
            self.model.fit(X_train, y_train)
            metrics.append(evaluate_model(self.model, X_test, y_test))

            if len(X_pool) < query_size:
                break

            probs = self.model.predict_proba(X_pool)
            indices = self.query_strategy.select(probs, query_size)

            X_train = np.concatenate([X_train, X_pool[indices]])
            y_train = np.concatenate([y_train, y_pool[indices]])

            X_pool = np.delete(X_pool, indices, axis=0)
            y_pool = np.delete(y_pool, indices, axis=0)

        return metrics
