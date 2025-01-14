import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Loads the dataset from the given file path."""
    return pd.read_csv(file_path)

def clean_data(data):
    """Performs basic data cleaning like handling duplicates."""
    return data.drop_duplicates()

def scale_features(X, scaler=None):
    """Scales numerical features using StandardScaler."""
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
