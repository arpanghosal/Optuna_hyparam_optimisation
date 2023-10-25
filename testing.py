import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load your data from a CSV file (replace 'your_data.csv' with the actual file path)
data = pd.read_csv('merged.csv')

# Drop the specified columns
columns_to_drop = ['Sample_Name', 'Sample_Type']
data = data.drop(columns=columns_to_drop)

# Separate features (X) and labels (y)
X = data.drop('Label', axis=1)  # Replace 'target_column_name' with your actual target column name
y = data['Label']  # Replace 'target_column_name' with your actual target column name


# Define an objective function to optimize
def objective(trial):
    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 8)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    n_estimators = trial.suggest_int("n_estimators", 100, 500)

    # Create and train the model with the suggested hyperparameters
    model = GradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_features=max_features,
        n_estimators=n_estimators
    )
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_valid)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_valid, y_pred)

    return accuracy

# Create a study object for optimization
study = optuna.create_study(direction="maximize")  # maximize accuracy

# Optimize the objective function
study.optimize(objective, n_trials=500)  # You can adjust the number of trials

# Get the best hyperparameters
best_params = study.best_params
best_accuracy = study.best_value

print("Best Hyperparameters:", best_params)
print("Best Accuracy:", best_accuracy)

