from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """Evaluates the model's performance."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auprc = auc(recall, precision)
    print(f"\nArea Under Precision-Recall Curve (AUPRC): {auprc:.4f}")
    return precision, recall, auprc

def plot_precision_recall_curve(precision, recall, auprc):
    """Plots the Precision-Recall Curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AUPRC: {auprc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()
    plt.show()
