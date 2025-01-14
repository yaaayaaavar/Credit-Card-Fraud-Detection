import numpy as np

def calculate_scale_pos_weight(y):
    """Calculates the scale_pos_weight for imbalanced datasets."""
    class_counts = np.bincount(y)
    return class_counts[0] / class_counts[1]
