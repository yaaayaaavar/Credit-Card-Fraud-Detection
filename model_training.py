from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib

def train_model(X, y, params):
    """Trains an XGBoost model with given parameters."""
    model = XGBClassifier(**params)
    model.fit(X, y)
    return model

def tune_hyperparameters(model, param_grid, X, y):
    """Performs hyperparameter tuning using GridSearchCV."""
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def save_model(model, file_path):
    """Saves the trained model to the given file path."""
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")
