from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits

class Dataset:
    
    DATASET_MAP = {
        "iris": load_iris,
        "digits": load_digits,
        "breast_cancer": load_breast_cancer,
        "wine": load_wine
    }

    def __init__(self, name: str):
        self.dataset = self.DATASET_MAP.get(name.lower())()
    
    def get(self):
        if self.dataset is None:
            return None, None
        else:
            return self.dataset.data, self.dataset.target