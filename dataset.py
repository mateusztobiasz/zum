from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

DATASET_MAP = {
    "iris": load_iris,  # 150 samples, 3 flower species, 4 numeric features
    "breast_cancer": load_breast_cancer,  # 569 samples, binary, 30 numeric features
    "wine": load_wine,  # 178 samples, 3 wine cultivars, 13 numeric features
    "adult": lambda: fetch_openml("adult", version=2, as_frame=True),  # ~48k, binary, mixed types
    "covertype": lambda: fetch_openml("covertype", version=1, as_frame=True),  # ~580k, 7 classes
    "credit-g": lambda: fetch_openml("credit-g", version=1, as_frame=True),  # ~1k, binary
    "yeast": lambda: fetch_openml(data_id=181, as_frame=True),  # ~1.5k, 10 classes
    "optdigits": lambda: fetch_openml("optdigits", version=1, as_frame=True),  # 5620 samples, 10 digits (0–9), 64 numeric features
}

class Dataset:
    def __init__(self, name: str):
        self.name = name.lower()
        self.loader = DATASET_MAP.get(self.name)
        if self.loader is None:
            raise ValueError(f"Dataset '{self.name}' not found.")
        self.dataset = self.loader()

    def get(self):
        X = self.dataset["data"]
        y = self.dataset["target"]

        if isinstance(y, pd.Series) and y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        if isinstance(X, pd.DataFrame):
            cat_cols = X.select_dtypes(include=["object", "category"]).columns
            num_cols = X.select_dtypes(include=["number"]).columns
            preprocessor = ColumnTransformer([
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            ])
            X = preprocessor.fit_transform(X)
        else:
            X = StandardScaler().fit_transform(X)

        return X, y