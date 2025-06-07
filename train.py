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
        # Convert to arrays immediately for consistent indexing
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_pool = np.asarray(X_pool)
        y_pool = np.asarray(y_pool)
        
        metrics = []

        for _ in tqdm(range(n_queries)):
            self.model.fit(X_train, y_train)
            eval_result = evaluate_model(self.model, X_test, y_test)
            eval_result["train_dataset_size"] = len(X_train)
            metrics.append(eval_result)
        
            if len(X_pool) < query_size:
                break
        
            probs = self.model.predict_proba(X_pool)
            indices = self.query_strategy.select(probs, query_size)
            
            # Ensure indices is a 1D array of integers
            indices = np.array(indices).flatten().astype(int)
            
            # Add queried samples to training set
            X_train = np.vstack([X_train, X_pool[indices]])
            y_train = np.concatenate([y_train, y_pool[indices]])
            
            # Remove queried samples from pool using mask
            mask = np.ones(len(X_pool), dtype=bool)
            mask[indices] = False
            X_pool = X_pool[mask]
            y_pool = y_pool[mask]

        return metrics