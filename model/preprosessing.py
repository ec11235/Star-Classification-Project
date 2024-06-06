import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(filepath):
    """
    Loads and preprocesses the star data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file containing star data.

    Returns:
    x_train (array-like): Standardized training feature set.
    x_test (array-like): Standardized test feature set.
    y_train (array-like): Training labels.
    y_test (array-like): Test labels.
    feature_names (Index): Names of the features.
    """
    # Load the dataset
    star_data = pd.read_csv(filepath)

    # Encoding categorical features
    label_encoder = LabelEncoder()
    star_data['Star color'] = label_encoder.fit_transform(star_data['Star color'])
    star_data['Spectral Class'] = label_encoder.fit_transform(star_data['Spectral Class'])

    # Features and target
    x = star_data.drop('Star type', axis=1)
    y = star_data['Star type']

    # Save feature names for later use
    feature_names = x.columns

    # Splitting the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Standardizing the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, feature_names

