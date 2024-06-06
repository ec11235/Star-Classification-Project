import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

class StarClassifier:
    """
    A classifier for predicting star types using RandomForestClassifier.
    """
    def __init__(self, n_estimators=100, random_state=42, **kwargs):
        """
        Initializes the classifier with the specified parameters.

        Parameters:
        n_estimators (int): Number of trees in the forest.
        random_state (int): Random seed for reproducibility.
        **kwargs: Additional keyword arguments for RandomForestClassifier.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, **kwargs)
    
    def train(self, x_train, y_train):
        """
        Trains the classifier on the training data.

        Parameters:
        x_train (array-like): Training feature set.
        y_train (array-like): Training labels.
        """
        self.model.fit(x_train, y_train)
    
    def evaluate(self, x_test, y_test):
        """
        Evaluates the classifier on the test data and prints the results.

        Parameters:
        x_test (array-like): Test feature set.
        y_test (array-like): Test labels.
        """
        y_pred = self.model.predict(x_test)
        # print(confusion_matrix(y_test, y_pred))
        # print(classification_report(y_test, y_pred))
    
    def cross_validate(self, x, y, cv=5):
        """
        Performs cross-validation and prints the scores.

        Parameters:
        x (array-like): Feature set.
        y (array-like): Labels.
        cv (int): Number of cross-validation folds.
        """
        cv_scores = cross_val_score(self.model, x, y, cv=cv)
        # print(f"Cross-validation scores: {cv_scores}")
        # print(f"Average cross-validation score: {cv_scores.mean()}")
    
    def feature_importances(self, feature_names):
        """
        Plots the feature importances of the classifier.

        Parameters:
        feature_names (list): List of feature names.
        """
        importances = self.model.feature_importances_
        indices = importances.argsort()
        
        plt.figure(figsize=(10, 7))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()




