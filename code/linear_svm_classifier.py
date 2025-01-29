import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from processing import DataProcessor
import matplotlib.pyplot as plt

# Load the datasets using DataProcessor
def load_data(dataset_names, df_type):
    dataframes = []
    
    for name in dataset_names:
        processor = DataProcessor(name)
        # Get specified DataFrame from the current dataset
        df_raw = processor.get_df(df_type)
        
        # Adding dataset identifier
        df_raw['dataset'] = name
        
        dataframes.append(df_raw)

    combined_data = pd.concat(dataframes, ignore_index=True)
    return combined_data

# Preprocess the data
def preprocess_data(data, processor, df_type):
    identifiers = processor.get_identifiers(df_type)  # Get identifiers 
    X = data.drop(columns=identifiers + ['dataset', 'SB_BinaryStats_diff_longstretch0_mean'])  # Drop identifiers and target column
    y = data['dataset'] 

    print(f"Number of columns after dropping identifiers: {X.shape[1]}")

    # categorical target variable to numerical
    y = pd.factorize(y)[0]  

    return X, y

def train_classifier(X, y):

    clf = SVC(kernel='linear', C=1.0)

    k = 100
    cv_scores = cross_val_score(clf, X, y, cv=k)

    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean Cross-Validation Score: {cv_scores.mean()}")

    clf.fit(X, y)

    y_pred = clf.predict(X)
    y_pred_proba = clf.decision_function(X)  

    print("Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))


    feature_importance = clf.coef_[0]  # Change for class of interest
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importances:")
    print(feature_importance_df)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='lightgreen')
    plt.xlabel('Importance')
    plt.title('Feature Importance for Linear SVM Classifier')
    plt.gca().invert_yaxis()  # Invert y-axis to have most important feature on top
    plt.show()

    # ROC Curve
    if len(set(y)) == 2:  # Only applicable for binary classification
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)


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
    # Load the combined data from both datasets for a specified DataFrame type
    combined_data = load_data(['HUP', 'MNI'], 'region')  # Change 'region_avg' to use other df types

    label_counts = combined_data['dataset'].value_counts()
    print("Instances of each label in the combined dataset:")
    print(label_counts)

    processor = DataProcessor('HUP')  # You can choose any dataset for the processor this is just to get types

    X, y = preprocess_data(combined_data, processor, 'region')

    train_classifier(X, y)
