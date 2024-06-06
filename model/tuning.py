import json
from sklearn.model_selection import GridSearchCV
from model.preprosessing import load_and_preprocess_data
from model.model_creation import StarClassifier

# Load and preprocess data
x_train, x_test, y_train, y_test, feature_names = load_and_preprocess_data('dataset/star_classification_data.csv')

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],             # Number of trees in the forest
    'max_features': ['auto', 'sqrt', 'log2'],    # Number of features to consider at every split
    'max_depth': [None, 10, 20, 30],             # Maximum number of levels in tree
    'min_samples_split': [2, 5, 10],             # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4]                # Minimum number of samples required at each leaf node
}

# Initialize GridSearchCV with a new RandomForestClassifier instance
grid_search = GridSearchCV(estimator=StarClassifier().model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data to find the best hyperparameters
grid_search.fit(x_train, y_train)

# Get the best parameters found by GridSearchCV
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Save the best parameters to a JSON file for future use
with open('model/best_params.json', 'w') as f:
    json.dump(best_params, f)


