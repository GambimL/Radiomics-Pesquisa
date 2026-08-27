"""
Classifier registry mirroring the 23 models from Faita et al. (2022),
grouped by the 6 categories from the MATLAB Classification Learner
(Amancio et al., 2014 — reference [18] in the paper).

Each entry is a (name, estimator) tuple compatible with scikit-learn API.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Decision Trees (4)
# ---------------------------------------------------------------------------
DECISION_TREES = [
    (
        "Simple Tree",
        DecisionTreeClassifier(max_depth=4, random_state=42),
    ),
    (
        "Medium Tree",
        DecisionTreeClassifier(max_depth=20, random_state=42),
    ),
    (
        "Complex Tree",
        DecisionTreeClassifier(max_depth=None, random_state=42),
    ),
]

# ---------------------------------------------------------------------------
# Discriminant Analysis (2)
# ---------------------------------------------------------------------------
DISCRIMINANT_ANALYSIS = [
    (
        "Linear Discriminant Analysis",
        LinearDiscriminantAnalysis(),
    ),
    (
        "Quadratic Discriminant Analysis",
        QuadraticDiscriminantAnalysis(reg_param=0.1),
    ),
]

# ---------------------------------------------------------------------------
# Logistic Regression (1)
# ---------------------------------------------------------------------------
LOGISTIC_REGRESSION = [
    (
        "Logistic Regression",
        LogisticRegression(max_iter=1000, random_state=42),
    ),
]

# ---------------------------------------------------------------------------
# Support Vector Machines (6)
# ---------------------------------------------------------------------------
SUPPORT_VECTOR_MACHINES = [
    (
        "Linear SVM",
        SVC(kernel="linear", probability=True, random_state=42),
    ),
    (
        "Quadratic SVM",
        SVC(kernel="poly", degree=2, probability=True, random_state=42),
    ),
    (
        "Cubic SVM",
        SVC(kernel="poly", degree=3, probability=True, random_state=42),
    ),
    (
        "Fine Gaussian SVM",
        SVC(kernel="rbf", gamma=10.0, probability=True, random_state=42),
    ),
    (
        "Medium Gaussian SVM",
        SVC(kernel="rbf", gamma="scale", probability=True, random_state=42),
    ),
    (
        "Coarse Gaussian SVM",
        SVC(kernel="rbf", gamma=0.01, probability=True, random_state=42),
    ),
]

# ---------------------------------------------------------------------------
# K-Nearest Neighbours (5)
# ---------------------------------------------------------------------------
K_NEAREST_NEIGHBOURS = [
    (
        "Fine KNN",
        KNeighborsClassifier(n_neighbors=1),
    ),
    (
        "Medium KNN",
        KNeighborsClassifier(n_neighbors=10),
    ),
    (
        "Coarse KNN",
        KNeighborsClassifier(n_neighbors=100),
    ),
    (
        "Cosine KNN",
        KNeighborsClassifier(n_neighbors=10, metric="cosine"),
    ),
    (
        "Weighted KNN",  # best classifier in Faita et al.
        KNeighborsClassifier(n_neighbors=10, weights="distance"),
    ),
]

# ---------------------------------------------------------------------------
# Ensemble Classifiers (5)
# ---------------------------------------------------------------------------
ENSEMBLE = [
    (
        "Boosted Trees",
        AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=30,
            random_state=42,
        ),
    ),
    (
        "Bagged Trees",
        BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=30,
            random_state=42,
        ),
    ),
    (
        "Subspace Discriminant",
        BaggingClassifier(
            estimator=LinearDiscriminantAnalysis(),
            n_estimators=30,
            max_features=0.5,
            bootstrap=False,
            random_state=42,
        ),
    ),
    (
        "Subspace KNN",
        BaggingClassifier(
            estimator=KNeighborsClassifier(n_neighbors=10),
            n_estimators=30,
            max_features=0.5,
            bootstrap=False,
            random_state=42,
        ),
    ),
    (
        "RUSBoosted Trees",
        AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=30,
            random_state=42,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Full registry — 23 classifiers total
# ---------------------------------------------------------------------------
ALL_CLASSIFIERS: list[tuple[str, object]] = (
    DECISION_TREES
    + DISCRIMINANT_ANALYSIS
    + LOGISTIC_REGRESSION
    + SUPPORT_VECTOR_MACHINES
    + K_NEAREST_NEIGHBOURS
    + ENSEMBLE
)


def get_classifier(name: str):
    """Return a fresh (cloned) estimator by name."""
    from sklearn.base import clone

    for n, clf in ALL_CLASSIFIERS:
        if n == name:
            return clone(clf)
    raise ValueError(f"Classifier '{name}' not found. Available: {list_classifier_names()}")


def list_classifier_names() -> list[str]:
    return [name for name, _ in ALL_CLASSIFIERS]


def build_selectkbest_pipeline(clf, k) -> Pipeline:
    """Pipeline da Trilha A: StandardScaler → SelectKBest (F-Score ANOVA) → classificador.

    k='all' pula o SelectKBest (usado no k-sweep para comparar contra "todas as features").
    """
    if k == 'all':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    clf),
        ])
    return Pipeline([
        ('scaler',   StandardScaler()),
        ('selector', SelectKBest(f_classif, k=k)),
        ('clf',      clf),
    ])


def build_plain_pipeline(clf) -> Pipeline:
    """Pipeline da Trilha B: StandardScaler → classificador (todas as features)."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    clf),
    ])
