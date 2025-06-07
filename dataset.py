from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

DATASET_MAP = {
    # Iris flower classification dataset: 150 samples, 3 classes (setosa, versicolor, virginica), 4 features (sepal/petal length/width)
    "iris": load_iris,
    
    # Breast Cancer Wisconsin dataset: 569 samples, binary classification (malignant/benign), 30 features from digitized images
    "breast_cancer": load_breast_cancer,
    
    # Wine classification dataset: 178 samples, 3 classes (wine cultivars), 13 chemical analysis features
    "wine": load_wine,
    
    # Adult Census Income dataset: ~48,000 samples, binary classification (>50K or <=50K income), mix of categorical and numeric census data
    "adult": lambda: fetch_openml("adult", version=2, as_frame=True),
    
    # Forest Covertype dataset: ~580,000 samples, multiclass classification of forest cover types based on cartographic variables
    "covertype": lambda: fetch_openml("covertype", version=1, as_frame=True),
    
    # German Credit dataset: ~1,000 samples, binary credit risk classification (good/bad), mix of categorical and numerical features
    "credit-g": lambda: fetch_openml("credit-g", version=1, as_frame=True),
}


class Dataset:
    def __init__(self, name: str):
        self.name = name.lower()
        self.loader = DATASET_MAP.get(self.name)
        if self.loader is None:
            raise ValueError(f"Dataset '{self.name}' not found in DATASET_MAP.")
        self.dataset = self.loader()

    def get(self):
        X = self.dataset["data"]
        y = self.dataset["target"]

        # Encode target if it's string
        if isinstance(y, pd.Series) and y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        # If X is a DataFrame (i.e., from OpenML), apply preprocessing
        if isinstance(X, pd.DataFrame):
            cat_cols = X.select_dtypes(include=["object", "category"]).columns
            num_cols = X.select_dtypes(include=["number"]).columns

            preprocessor = ColumnTransformer([
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            ])

            X = preprocessor.fit_transform(X)
        else:
            # Standardize numeric numpy arrays (sklearn built-ins)
            X = StandardScaler().fit_transform(X)

        return X, y