import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_roc_curve(model, X_test, y_test, save_path=None):
    """Plot ROC curve."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, X, save_path=None):
    """Plot feature importance."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        features = np.arange(len(importance))
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importance, y=features)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def generate_pdf_report(results, task, y_pred, y_test, output_path):
    """Generate PDF report with results."""
    c = canvas.Canvas(output_path, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "AutoMLPipe Report")
    
    y = 700
    for name, result in results.items():
        c.drawString(100, y, f"{name}: {'Accuracy' if task != 'regression' else 'R2 Score'}: {result['score']:.4f}")
        y -= 20
    
    c.showPage()
    c.save()