import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from processing import DataProcessor  # Import the DataProcessor class
import matplotlib.pyplot as plt

# Load the datasets using DataProcessor
def load_data(dataset_names, df_type):
    dataframes = []

    for name in dataset_names:
        processor = DataProcessor(name)
        # Get the specified DataFrame from the current dataset
        df_raw = processor.get_df(df_type)
        
        # Add a target column to distinguish datasets
        df_raw['dataset'] = name
        
        # Append the DataFrame to the list
        dataframes.append(df_raw)

    # Combine the datasets
    combined_data = pd.concat(dataframes, ignore_index=True)
    return combined_data

# Preprocess the data
def preprocess_data(data, processor, df_type):
    identifiers = processor.get_identifiers(df_type)  # Get identifiers for the specific DataFrame type
    X = data.drop(columns=identifiers + ['dataset'])  # Drop identifiers and the target column
    y = data['dataset']  # Target variable


    y = pd.factorize(y)[0]  # Automatically encodes classes as integers


    return X, y

# Train and evaluate the Random Forest classifier
def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99)

    # Initialize and train the classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    # Check the number of classes
    num_classes = y_pred_proba.shape[1]
    print(f"Number of classes: {num_classes}")

    # Handle binary and multiclass cases
    if num_classes == 2:
        # For binary classification, get probabilities for the positive class
        y_pred_proba_positive = y_pred_proba[:, 1]
    elif num_classes > 2:
        # For multiclass classification, you can handle it differently
        y_pred_proba_positive = y_pred_proba  # Keep all probabilities
    else:
        raise ValueError("Unexpected number of classes in the target variable.")

    # Evaluate the classifier
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Get feature importances
    feature_importances = clf.feature_importances_

    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    })

    # Sort the DataFrame by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Print the feature importances
    print("\nFeature Importances:")
    print(feature_importance_df)

    # ROC Curve
    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_positive)
        roc_auc = auc(fpr, tpr)

        # Plotting the ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()
    else:
        print("ROC curve is not applicable for multiclass classification.")

if __name__ == "__main__":

    combined_data = load_data(['HUP', 'MNI'], 'region')  # Change 'region' to any other DataFrame type 

    label_counts = combined_data['dataset'].value_counts()
    print("Instances of each label in the combined dataset:")
    print(label_counts)

    processor = DataProcessor('HUP')  # choose any dataset for the processor, it is to get data types
    
    X, y = preprocess_data(combined_data, processor, 'region')


    train_classifier(X, y)


