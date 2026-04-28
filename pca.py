"""
PCA analysis on the Iris dataset.
Reduces 4 features to 2-3 principal components and evaluates classification before and after.
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class IrisPCAAnalysis:
    """
    Full PCA pipeline for the Iris dataset.
    Compares classification accuracy on raw features vs PCA-reduced features.
    Provides scree plot data, loadings, and biplot coordinates.
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.scaler = None
        self.pca = None
        self.feature_names: List[str] = []
        self.class_names: List[str] = []
        self._fitted = False

    def load_iris_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        iris = load_iris()
        self.feature_names = list(iris.feature_names)
        self.class_names = list(iris.target_names)
        return iris.data, iris.target

    def fit_pca(self, X: np.ndarray) -> np.ndarray:
        """Scale features and fit PCA. Returns transformed data."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.pca = PCA(n_components=self.n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        self._fitted = True
        return X_pca

    def explained_variance_table(self) -> pd.DataFrame:
        """Scree table with per-component and cumulative variance explained."""
        if not self._fitted:
            raise RuntimeError("Call fit_pca() first.")
        evr = self.pca.explained_variance_ratio_
        return pd.DataFrame({
            "component": [f"PC{i+1}" for i in range(len(evr))],
            "eigenvalue": self.pca.explained_variance_.round(4),
            "variance_pct": (evr * 100).round(2),
            "cumulative_pct": (np.cumsum(evr) * 100).round(2),
        })

    def loadings_dataframe(self) -> pd.DataFrame:
        """Feature loading matrix for all components."""
        if not self._fitted:
            raise RuntimeError("Call fit_pca() first.")
        return pd.DataFrame(
            self.pca.components_.T,
            index=self.feature_names,
            columns=[f"PC{i+1}" for i in range(self.n_components)],
        ).round(4)

    def biplot_data(self, X_pca: np.ndarray, y: np.ndarray) -> Dict:
        """
        Return data needed to render a biplot:
        scores (projection of samples), loadings, and class labels.
        """
        loadings = self.pca.components_.T
        scale = np.max(np.abs(X_pca[:, :2]))
        scale_l = np.max(np.abs(loadings[:, :2]))
        scale_factor = scale / scale_l if scale_l > 0 else 1.0
        return {
            "scores_pc1": X_pca[:, 0].tolist(),
            "scores_pc2": X_pca[:, 1].tolist() if X_pca.shape[1] > 1 else [],
            "labels": y.tolist(),
            "class_names": self.class_names,
            "loading_x": (loadings[:, 0] * scale_factor).round(4).tolist(),
            "loading_y": (loadings[:, 1] * scale_factor).round(4).tolist() if loadings.shape[1] > 1 else [],
            "feature_names": self.feature_names,
        }

    def compare_classifiers(self, X_raw: np.ndarray, X_pca: np.ndarray,
                             y: np.ndarray, cv_folds: int = 5) -> pd.DataFrame:
        """
        Compare classifier accuracy on raw features vs PCA components using cross-validation.
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        classifiers = {
            "LogisticRegression": LogisticRegression(max_iter=300),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
        }
        scaler_raw = StandardScaler()
        X_raw_scaled = scaler_raw.fit_transform(X_raw)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        results = []
        for name, clf in classifiers.items():
            raw_cv = cross_val_score(clf, X_raw_scaled, y, cv=skf, scoring="accuracy")
            pca_cv = cross_val_score(clf, X_pca, y, cv=skf, scoring="accuracy")
            results.append({
                "classifier": name,
                "raw_accuracy": round(float(raw_cv.mean()), 4),
                "pca_accuracy": round(float(pca_cv.mean()), 4),
                "accuracy_delta": round(float(pca_cv.mean() - raw_cv.mean()), 4),
                "n_pca_components": X_pca.shape[1],
            })
        return pd.DataFrame(results)

    def lda_comparison(self, X_raw: np.ndarray, y: np.ndarray) -> Dict:
        """Compare LDA vs PCA projections."""
        if not SKLEARN_AVAILABLE:
            return {}
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_raw, y)
        lda_scores = lda.transform(X_raw)
        return {
            "lda_explained_variance_ratio": lda.explained_variance_ratio_.round(4).tolist(),
            "lda_first_component_variance_pct": round(
                float(lda.explained_variance_ratio_[0]) * 100, 2
            ) if len(lda.explained_variance_ratio_) > 0 else 0.0,
        }


if __name__ == "__main__":
    analysis = IrisPCAAnalysis(n_components=4)
    if SKLEARN_AVAILABLE:
        X, y = analysis.load_iris_data()
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} classes")
        print(f"Features: {analysis.feature_names}")
        X_pca_full = analysis.fit_pca(X)

        print("\nExplained variance (all 4 components):")
        print(analysis.explained_variance_table().to_string(index=False))

        print("\nFeature loadings:")
        print(analysis.loadings_dataframe().to_string())

        analysis2 = IrisPCAAnalysis(n_components=2)
        X_pca_2d = analysis2.fit_pca(X)

        print("\nClassifier comparison (raw vs 2D PCA):")
        comparison = analysis2.compare_classifiers(X, X_pca_2d, y)
        print(comparison.to_string(index=False))

        lda_info = analysis2.lda_comparison(X, y)
        print("\nLDA analysis:", lda_info)

        biplot = analysis2.biplot_data(X_pca_2d, y)
        print(f"\nBiplot scores (first 3 samples): PC1={biplot['scores_pc1'][:3]}")
    else:
        print("scikit-learn not available. Install with: pip install scikit-learn")
