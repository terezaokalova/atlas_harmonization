import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from processing import DataProcessor 
import matplotlib.pyplot as plt


"""
Note on Performance: in its current implementation despite train_test split or k-fold cv the classifier
achieves perfect performance when trained on all catch 22 features. This raises a huge red flag however I cannot find error.
As the most important features are dropped, dropping just 2 features achieves much more reasonable performance.
This may suggest leakage. I am still stumped at why with 0.01 train data it still achieves perfect
performance. 
"""

# Load the datasets using DataProcessor
def load_data(dataset_names, df_type):
    " Labels and lists dfs and then returns one large combined df"
    dataframes = []

    for name in dataset_names:
        processor = DataProcessor(name)
        # Get specified DataFrame from the current dataset
        df_raw = processor.get_df(df_type)
        
        # adding data set identifier
        df_raw['dataset'] = name
        

        dataframes.append(df_raw)


    combined_data = pd.concat(dataframes, ignore_index=True)
    return combined_data


def preprocess_data(data, processor, df_type):
    identifiers = processor.get_identifiers(df_type)  # Get identifiers for the specific DataFrame type
    X = data.drop(columns=identifiers + ['dataset', 'SB_BinaryStats_diff_longstretch0', 'SB_BinaryStats_mean_longstretch1'])  # Drop identifiers, target column, and any other specified features
    y = data['dataset']  # Target variable


    print(f"Number of columns after dropping identifiers: {X.shape[1]}")

    # Convert categorical target variable to numerical
    y = pd.factorize(y)[0]  # Automatically encodes classes as integers

    return X, y


def train_classifier(X, y):
    # Initialize classifier with L2 regularization
    clf = LogisticRegression(multi_class='ovr', penalty='l2', C=1.0)

    # k-fold cross-validation
    k = 10
    cv_scores = cross_val_score(clf, X, y, cv=k)

    # Print cross-validation results
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean Cross-Validation Score: {cv_scores.mean()}")

    # Fit the model on the entire dataset
    clf.fit(X, y)

    # Make predictions
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    # Evaluate classifier
    print("Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))

    # Feature Importance
    feature_importance = clf.coef_[0]  # Change for class of interest
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Print feature importances
    print("\nFeature Importances:")
    print(feature_importance_df)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='lightgreen')
    plt.xlabel('Importance')
    plt.title('Feature Importance for Logistic Regression Classifier')
    plt.gca().invert_yaxis()  # Invert y-axis to have most important feature on top
    plt.show()

    # ROC Curve
    for i in range(len(clf.classes_)):
        fpr, tpr, _ = roc_curve(y, y_pred_proba[:, i], pos_label=i)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC curve for class {0} (area = {1:0.2f})'.format(i, roc_auc))

    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multiclass')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    # Load the combined data from both datasets for a specified DataFrame type
    combined_data = load_data(['HUP', 'MNI'], 'region_avg')  # Change 'region_avg' to use other df types

    label_counts = combined_data['dataset'].value_counts()
    print("Instances of each label in the combined dataset:")
    print(label_counts)

    # Initialize DataProcessor for preprocessing
    processor = DataProcessor('HUP')  # You can choose any dataset for the processor this is just to get types

    X, y = preprocess_data(combined_data, processor, 'region_avg')

    train_classifier(X, y)
