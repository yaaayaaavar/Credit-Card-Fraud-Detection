import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def log_transform(X, columns):
    """Applies log transformation to the specified columns."""
    for col in columns:
        X[:, col] = np.log1p(X[:, col])
    return X

def add_statistical_features(X, top_features_indices):
    """Adds mean and standard deviation features for the selected features."""
    X_mean = np.mean(X[:, top_features_indices], axis=1).reshape(-1, 1)
    X_std = np.std(X[:, top_features_indices], axis=1).reshape(-1, 1)
    return np.hstack((X, X_mean, X_std))

def add_interaction_terms(X, top_features_indices, degree=2):
    """Adds interaction terms for the selected features."""
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    interaction_features = poly.fit_transform(X[:, top_features_indices])
    return np.hstack((X, interaction_features))
