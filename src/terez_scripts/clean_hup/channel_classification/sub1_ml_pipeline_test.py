# %% [code]
import os
import numpy as np
import pandas as pd
import pycatch22
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def get_feature_list():
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_features = [f"{band}_{metric}" for band in band_names for metric in ['power', 'rel', 'log']]
    fooof_features = [
        'fooof_aperiodic_offset',
        'fooof_aperiodic_exponent',
        'fooof_r_squared',
        'fooof_error',
        'fooof_num_peaks'
    ]
    entropy_features = ['entropy_5secwin']
    dummy = np.random.randn(100).tolist()
    res = pycatch22.catch22_all(dummy, catch24=False)
    catch22_features = [f"catch22_{nm}" for nm in res['names']]
    return band_features + fooof_features + entropy_features + catch22_features

def convert_spared_to_label(val):
    if isinstance(val, bool):
        return 0 if val else 1
    elif isinstance(val, str):
        return 0 if val.strip().upper() == 'TRUE' else 1
    else:
        return 0 if bool(val) else 1

# Load aggregated features for subject sub-RID0031.
data_path = "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data/hup/derivatives/clean/sub-RID0031/sub-RID0031_features_averaged.pkl"
df = pd.read_pickle(data_path)
print(f"Data loaded from {data_path} with shape: {df.shape}")

feature_list = get_feature_list()
present_features = [feat for feat in feature_list if feat in df.columns]
missing_features = set(feature_list) - set(present_features)
if missing_features:
    print("Warning: The following expected features are missing:")
    print(missing_features)

if 'spared' not in df.columns:
    raise ValueError("Column 'spared' not found in the data.")
df['label'] = df['spared'].apply(convert_spared_to_label)

# Drop rows with missing values for the selected features.
df_clean = df.dropna(subset=present_features)
X = df_clean[present_features].copy()
y = df_clean['label'].values
print(f"After dropping missing values, data shape is: {X.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_choice = "logistic"  # change to "random_forest" if desired
if model_choice == "logistic":
    clf = LogisticRegression(max_iter=1000, random_state=42)
elif model_choice == "random_forest":
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    raise ValueError("Model choice must be 'logistic' or 'random_forest'.")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies, aucs = [], []
for train_idx, test_idx in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, y_proba))
    accuracies.append(accuracy_score(y_test, y_pred))

print("Cross-validation results:")
print(f"  Average Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
if aucs:
    print(f"  Average ROC AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

clf.fit(X_scaled, y)
y_pred_all = clf.predict(X_scaled)
print("\nClassification Report on Full Data:")
print(classification_report(y, y_pred_all))

# %%
