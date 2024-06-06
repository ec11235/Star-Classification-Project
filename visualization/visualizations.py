import sys
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from model.preprosessing import load_and_preprocess_data
from model.model_creation import StarClassifier

# Add the project directory to the sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load and preprocess data
x_train, x_test, y_train, y_test, feature_names = load_and_preprocess_data('dataset/star_classification_data.csv')

# Load the best parameters from a JSON file
with open('model/best_params.json', 'r') as f:
    best_params = json.load(f)

# Initialize and train the model with the best parameters
classifier = StarClassifier(**best_params)
classifier.train(x_train, y_train)

# Generate predictions
y_pred = classifier.model.predict(x_test)
y_pred_prob = classifier.model.predict_proba(x_test)

# Define output directory for saving visualizations
output_dir = '/app/output'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Confusion Matrix Heatmap
def plot_confusion_matrix(y_test, y_pred):
    """
    Plots the confusion matrix as a heatmap and saves it to a file.

    Parameters:
    y_test (array-like): True labels.
    y_pred (array-like): Predicted labels.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    file_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(file_path)
    print(f"Saved confusion matrix to {file_path}")
    plt.close()

# ROC Curve
def plot_roc_curve(y_test, y_pred_prob):
    """
    Plots the ROC curve for each class and saves it to a file.

    Parameters:
    y_test (array-like): True labels.
    y_pred_prob (array-like): Predicted probabilities for each class.
    """
    plt.figure(figsize=(10, 7))
    for i in range(len(classifier.model.classes_)):
        fpr, tpr, _ = roc_curve(y_test == i, y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='Class %d (area = %0.2f)' % (i, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    file_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(file_path)
    print(f"Saved ROC curve to {file_path}")
    plt.close()

# Feature Importance Plot
def plot_feature_importance(model, feature_names):
    """
    Plots the feature importances of the model and saves it to a file.

    Parameters:
    model (RandomForestClassifier): Trained RandomForestClassifier model.
    feature_names (list): List of feature names.
    """
    importances = model.feature_importances_
    indices = importances.argsort()
    
    plt.figure(figsize=(10, 7))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    file_path = os.path.join(output_dir, 'feature_importances.png')
    plt.savefig(file_path)
    print(f"Saved feature importances to {file_path}")
    plt.close()

# Main visualization function
def main():
    """
    Main function to evaluate the classifier, perform cross-validation, 
    and generate visualizations.
    """
    classifier.evaluate(x_test, y_test)
    classifier.cross_validate(x_train, y_train)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_prob)
    plot_feature_importance(classifier.model, feature_names)

if __name__ == '__main__':
    main()






