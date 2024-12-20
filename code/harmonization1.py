import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import neuroHarmonize as nh
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, ConfusionMatrixDisplay

class DataLoader:
    def __init__(self, results_path):
        self.results_path = results_path
        self.feature_columns = [
            'deltaRel_mean', 'thetaRel_mean', 'alphaRel_mean', 
            'betaRel_mean', 'gammaRel_mean', 'entropy_1min_mean', 
            'entropy_fullts_mean'
        ]
    
    def load_data(self):
        # Load HUP and MNI data
        hup_region_features = pd.read_csv(os.path.join(self.results_path, 'ge_go_hup_region_features.csv'))
        mni_region_features = pd.read_csv(os.path.join(self.results_path, 'mni_region_features.csv'))

        # Add site labels
        hup_region_features['site'] = 'HUP'
        mni_region_features['site'] = 'MNI'

        # Combine datasets
        region_features = pd.concat([hup_region_features, mni_region_features], ignore_index=True)

        # Aggregate features per patient
        patient_features = region_features.groupby(['patient_id', 'site'])[self.feature_columns].mean().reset_index()
        patient_features = patient_features.set_index('patient_id')

        return patient_features, self.feature_columns


class MLTrainer:
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns
        self.model_defs = {
            'logistic': (
                LogisticRegression(max_iter=1000),
                {
                    'C': [0.1, 1.0, 10.0],
                    'class_weight': ['balanced', None]
                }
            ),
            'rf': (
                RandomForestClassifier(),
                {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, None],
                    'class_weight': ['balanced', None]
                }
            ),
            'svm': (
                SVC(probability=True),
                {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf'],
                    'class_weight': ['balanced', None]
                }
            )
        }

    def train_and_evaluate_feature(self, X_train, y_train, X_test, y_test, model_key, feature_name):
        # Use only one feature
        X_train_feat = X_train[[feature_name]].copy()
        X_test_feat = X_test[[feature_name]].copy()

        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_feat)
        X_test_scaled = scaler.transform(X_test_feat)

        model, param_grid = self.model_defs[model_key]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=cv, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_

        # Evaluate on test set
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        y_pred = best_model.predict(X_test_scaled)

        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, target_names=['MNI', 'HUP'], output_dict=True)

        return {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'cm': cm,
            'cr': cr,
            'best_params': grid_search.best_params_
        }


def main_post_harmonization():
    # Set your paths
    code_directory = '/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/code'
    os.chdir(code_directory)
    results_path = '../results'

    # Load data
    loader = DataLoader(results_path)
    patient_features, feature_columns = loader.load_data()

    # Create labels
    y = patient_features['site'].map({'HUP': 1, 'MNI': 0})

    # Prepare covars for NeuroCombat
    # neuroHarmonize expects a column named SITE
    covars = pd.DataFrame({'SITE': patient_features['site'].values}, index=patient_features.index)

    # Extract data matrix
    X_data = patient_features[feature_columns].values

    # HarmonizationLearn returns (harmonized_data, model)
    X_harmonized, model = nh.harmonizationLearn(X_data, covars)

    # Replace with harmonized data
    harmonized_features = patient_features.copy()
    harmonized_features[feature_columns] = X_harmonized

    y_harm = harmonized_features['site'].map({'HUP': 1, 'MNI': 0})
    X_harm = harmonized_features[feature_columns]

    # Train/test split on harmonized data
    X_train_harm, X_test_harm, y_train_harm, y_test_harm = train_test_split(
        X_harm, y_harm, test_size=0.2, stratify=y_harm, random_state=42
    )

    # Train models after harmonization
    trainer = MLTrainer(feature_columns)

    # Run all models and features
    all_results = {}
    for model_key in ['logistic', 'rf', 'svm']:
        for feat in feature_columns:
            results = trainer.train_and_evaluate_feature(X_train_harm, y_train_harm, X_test_harm, y_test_harm, model_key, feat)
            all_results[(model_key, feat)] = results

            # Plot ROC curve
            plt.figure(figsize=(6, 5))
            plt.plot(results['fpr'], results['tpr'], label=f'AUC={results["auc"]:.2f}')
            plt.plot([0,1],[0,1],'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve After Harmonization - {model_key.upper()} - Feature: {feat}')
            plt.legend()
            plt.tight_layout()
            plt.show()

            print(f"Model: {model_key}, Feature: {feat}")
            print("Best Params:", results['best_params'])
            print("Confusion Matrix:\n", results['cm'])
            print("Classification Report:\n", pd.DataFrame(results['cr']))
            print("-"*50)

    # Plot confusion matrices in a separate step
    for (model_key, feat), res in all_results.items():
        cm = res['cm']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['MNI', 'HUP'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix After Harmonization - {model_key.upper()} - Feature: {feat}')
        plt.show()

if __name__ == "__main__":
    main_post_harmonization()
