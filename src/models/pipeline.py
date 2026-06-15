"""
Reusable sklearn pipeline factory: StandardScaler -> PCA -> Classifier.
"""

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_pca_pipeline(
    n_components: int = 2,
    classifier: str = "logistic",
    random_state: int = 42
) -> Pipeline:
    """
    Build a sklearn Pipeline: StandardScaler -> PCA -> Classifier.

    Parameters
    ----------
    n_components : int
    classifier : str — 'logistic' or 'random_forest'
    random_state : int
    """
    classifiers = {
        "logistic": LogisticRegression(random_state=random_state, max_iter=1000),
        "random_forest": RandomForestClassifier(random_state=random_state)
    }
    if classifier not in classifiers:
        raise ValueError(f"classifier must be one of {list(classifiers.keys())}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("clf", classifiers[classifier])
    ])
