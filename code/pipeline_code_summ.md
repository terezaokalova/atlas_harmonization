Feature extraction part:

# Run with default settings
python main.py

# Force recomputation of all features
python main.py --force

# Process specific cohorts
python main.py --cohorts hup mni

# Use a different config file
python main.py --config custom_config.yaml

GLOBAL ANALYSIS:
Mann-Whitney U uses two-tailed by default (which is what we want)

ML on these features:
Model Parameters:

Logistic Regression:
C: Inverse regularization strength (smaller = stronger regularization)
class_weight: Handles class imbalance by adjusting weights

Random Forest:
n_estimators: Number of trees
max_depth: Maximum tree depth
class_weight: Class balancing weights

SVM:
C: Regularization parameter
kernel: Transformation function (linear vs rbf/Gaussian)

Cross-validation: 
Splits data into 5 folds while preserving class ratios
n_jobs=-1 uses all CPU cores

Feature Importance (not PCA):
For logistic regression: Coefficient magnitudes show feature influence
Positive values = positive correlation with HUP class
shows gamma and delta bands are most predictive

Confusion Matrix:
The solid rectangle suggests perfect prediction for one class
This could indicate overfitting or class imbalance
Verify predictions aren't stuck on majority class

Metrics:
F1 score: Harmonic mean of precision and recall (0.87 weighted avg is good)
Support: Number of samples per class (106 MNI vs 28 HUP shows imbalance)
ROC curves: Random Forest performs best (AUC=0.95) followed by SVM (0.84) and Logistic (0.81)

Bias-Variance Analysis:
Random Forest shows best balance between bias and variance
High precision but lower recall for HUP class suggests some bias toward majority class