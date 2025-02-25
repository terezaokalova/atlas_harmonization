import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold, 
                                   GridSearchCV, cross_val_score)
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import RocCurveDisplay
# from sklearn.metrics import (roc_curve, auc, confusion_matrix, 
                        #    classification_report, plot_roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from regional_analysis import RegionalAnalysis_unpaired  # Import from your stats script

# diagnostics
from sklearn.model_selection import (
    learning_curve, 
    cross_validate, 
    cross_val_predict, 
    StratifiedKFold,
    cross_val_score
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve, 
    auc, 
    precision_recall_curve,
    average_precision_score
)

# Global configurations
CODE_DIR = '/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/code'
RESULTS_DIR = '../results'
DATA_DIR = '../Data'
FIGURES_DIR = '../figures'

class RegionalMLAnalysis:
    def __init__(self, region_analysis: RegionalAnalysis_unpaired):
        """
        Initialize with previously created RegionalAnalysis instance
        """
        self.logger = logging.getLogger(__name__)
        self.region_analysis = region_analysis
        self.feature_columns = region_analysis.feature_columns
        self.results = {}
        
    def prepare_region_data(self, region: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for a specific region for ML analysis
        """
        try:
            # Get data for specific region
            hup_region = self.region_analysis.hup_features[
                self.region_analysis.hup_features['roi'] == region
            ]
            mni_region = self.region_analysis.mni_features[
                self.region_analysis.mni_features['roi'] == region
            ]
            
            # Group by patient to get mean values
            hup_data = hup_region.groupby('patient_id')[self.feature_columns].mean()
            mni_data = mni_region.groupby('patient_id')[self.feature_columns].mean()
            
            # Check if enough samples
            if len(hup_data) < 5 or len(mni_data) < 5:
                raise ValueError(f"Insufficient samples for region {region} "
                            f"(HUP: {len(hup_data)}, MNI: {len(mni_data)})")
            
            # Create feature matrix and labels
            X = np.vstack([hup_data.values, mni_data.values])
            y = np.hstack([np.ones(len(hup_data)), np.zeros(len(mni_data))])
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing data for region {region}: {str(e)}")
            raise
    
    def train_evaluate_region(self, region: str, save_plots: bool = True) -> Dict:
        """
        Train and evaluate models for a specific region
        """
        try:
            # Prepare data
            X, y = self.prepare_region_data(region)
            
            # Initialize results dictionary
            region_results = {
                'n_hup': sum(y == 1),
                'n_mni': sum(y == 0),
                'models': {}
            }
            
            # Define models to try
            models = {
                'logistic': (LogisticRegression(), {
                    'C': [0.1, 1.0, 10.0],
                    'class_weight': ['balanced', None]
                }),
                'rf': (RandomForestClassifier(), {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, None],
                    'class_weight': ['balanced', None]
                }),
                'svm': (SVC(probability=True), {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf'],
                    'class_weight': ['balanced', None]
                })
            }
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Cross-validation setup
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Train and evaluate each model type
            for model_name, (model, param_grid) in models.items():
                # Grid search
                grid_search = GridSearchCV(
                    model, param_grid,
                    cv=cv, scoring='roc_auc',
                    n_jobs=-1
                )
                
                # Fit model
                grid_search.fit(X_scaled, y)
                
                # Get cross-validation scores
                cv_scores = cross_val_score(
                    grid_search.best_estimator_,
                    X_scaled, y,
                    cv=cv,
                    scoring='roc_auc'
                )
                
                # Store results
                model_results = {
                    'best_params': grid_search.best_params_,
                    'best_cv_score': grid_search.best_score_,
                    'cv_scores_mean': cv_scores.mean(),
                    'cv_scores_std': cv_scores.std()
                }
                
                # Get feature importances for interpretable models
                if model_name == 'logistic':
                    importances = pd.Series(
                        grid_search.best_estimator_.coef_[0],
                        index=self.feature_columns
                    ).sort_values(ascending=False)
                    model_results['feature_importances'] = importances.to_dict()
                elif model_name == 'rf':
                    importances = pd.Series(
                        grid_search.best_estimator_.feature_importances_,
                        index=self.feature_columns
                    ).sort_values(ascending=False)
                    model_results['feature_importances'] = importances.to_dict()
                
                region_results['models'][model_name] = model_results
                
                # Save ROC curves if requested
                if save_plots:
                    plt.figure(figsize=(10, 6))
                    for train_idx, test_idx in cv.split(X_scaled, y):
                        probas = grid_search.best_estimator_.predict_proba(
                            X_scaled[test_idx]
                        )
                        fpr, tpr, _ = roc_curve(y[test_idx], probas[:, 1])
                        plt.plot(
                            fpr, tpr, 
                            alpha=0.3, 
                            label=f'ROC fold (AUC = {auc(fpr, tpr):.2f})'
                        )
                    
                    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'{region} - {model_name} ROC Curves')
                    plt.legend()
                    
                    # Create figures directory if it doesn't exist
                    os.makedirs(os.path.join(FIGURES_DIR, 'ml_results'), exist_ok=True)
                    plt.savefig(
                        os.path.join(
                            FIGURES_DIR, 
                            'ml_results', 
                            f'{region}_{model_name}_roc.png'
                        )
                    )
                    plt.close()
            
            return region_results
            
        except Exception as e:
            self.logger.warning(f"Error processing region {region}: {str(e)}")
            return None

    def analyze_all_regions(self, save_plots: bool = True):
        """
        Perform ML analysis for all regions with sufficient data
        """
        for region in self.region_analysis.common_regions:
            self.logger.info(f"Analyzing region: {region}")
            results = self.train_evaluate_region(region, save_plots)
            if results is not None:
                self.results[region] = results
    
    def summarize_results(self):
        """
        Print summary of ML analysis results and save to file
        """
        summary_data = []
        
        for region, results in self.results.items():
            best_score = max(
                model['cv_scores_mean'] 
                for model in results['models'].values()
            )
            best_model = max(
                results['models'].items(),
                key=lambda x: x[1]['cv_scores_mean']
            )[0]
            
            summary_data.append({
                'region': region,
                'best_model': best_model,
                'cv_score': best_score,
                'n_hup': results['n_hup'],
                'n_mni': results['n_mni']
            })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('cv_score', ascending=False)
        
        # Save summary to file
        summary_df.to_csv(os.path.join(RESULTS_DIR, 'ml_results_summary.csv'), index=False)
        
        # Print summary
        print("\nML Analysis Summary")
        print("=" * 50)
        print(f"\nTotal regions analyzed: {len(self.results)}")
        print(f"Regions with ROC AUC > 0.7: {sum(summary_df['cv_score'] > 0.7)}")
        
        print("\nTop 10 most distinguishable regions:")
        print(summary_df.head(10))
        
        # Calculate average performance by model type
        model_scores = {model: [] for model in ['logistic', 'rf', 'svm']}
        for results in self.results.values():
            for model_name, model_results in results['models'].items():
                model_scores[model_name].append(model_results['cv_scores_mean'])
        
        print("\nAverage performance by model type:")
        for model_name, scores in model_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"{model_name}: {mean_score:.3f} ± {std_score:.3f}")

# def main():
#     # Set up logging
#     logging.basicConfig(level=logging.INFO)
    
#     try:
#         # Initialize and run statistical analysis
#         stat_analysis = RegionalAnalysis_unpaired()
#         stat_analysis.load_data()
        
#         # Initialize and run ML analysis
#         ml_analysis = RegionalMLAnalysis(stat_analysis)
#         ml_analysis.analyze_all_regions()
#         ml_analysis.summarize_results()
        
#     except Exception as e:
#         print(f"Error in main execution: {str(e)}")
#         raise

# if __name__ == "__main__":
#     # Set working directory
#     os.chdir(CODE_DIR)
#     main()

### DIAGNOSTICS OF MODEL BEHAVIOR

class EnhancedRegionalMLAnalysis:
    def __init__(self, region_analysis):
        self.logger = logging.getLogger(__name__)
        self.region_analysis = region_analysis
        self.feature_columns = region_analysis.feature_columns
        self.results = {}
        
    def enhanced_model_evaluation(self, X, y, model, region_name):
        """
        Comprehensive model evaluation with multiple metrics
        """
        results = {}
        
        # 1. Learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, 
            cv=5, 
            n_jobs=-1,
            train_sizes=np.linspace(0.3, 1.0, 5),  # Modified for small samples
            scoring='roc_auc'
        )
        
        results['learning_curves'] = {
            'train_sizes': train_sizes,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1)
        }
        
        # 2. Detailed cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=5,
            scoring={
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1',
                'roc_auc': 'roc_auc'
            },
            return_train_score=True
        )
        results['cv_results'] = cv_results
        
        # 3. Probability calibration
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_prob = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
        
        # Handle case where model doesn't support predict_proba
        if y_prob is not None:
            prob_true, prob_pred = calibration_curve(
                y, y_prob[:, 1], 
                n_bins=min(5, len(np.unique(y)))  # Adjust bins for small samples
            )
            results['calibration'] = {
                'prob_true': prob_true,
                'prob_pred': prob_pred
            }
        
        # 4. Sample size analysis
        results['sample_size'] = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'class_balance': np.bincount(y)
        }
        
        return results
    
    def prepare_region_data(self, region: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for a specific region with explicit type handling
        """
        try:
            # Get data for specific region
            hup_region = self.region_analysis.hup_features[
                self.region_analysis.hup_features['roi'] == region
            ]
            mni_region = self.region_analysis.mni_features[
                self.region_analysis.mni_features['roi'] == region
            ]
            
            # Group by patient and get mean values
            hup_data = hup_region.groupby('patient_id')[self.feature_columns].mean()
            mni_data = mni_region.groupby('patient_id')[self.feature_columns].mean()
            
            # Check if enough samples
            if len(hup_data) < 5 or len(mni_data) < 5:
                raise ValueError(f"Insufficient samples for region {region} "
                            f"(HUP: {len(hup_data)}, MNI: {len(mni_data)})")
            
            # Create feature matrix and labels with explicit types
            X = np.vstack([hup_data.values, mni_data.values]).astype(np.float64)
            y = np.concatenate([
                np.ones(len(hup_data), dtype=np.int32),
                np.zeros(len(mni_data), dtype=np.int32)
            ])
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing data for region {region}: {str(e)}")
            raise

    def robust_model_analysis(self, X, y, model, region_name):
        """
        Bootstrap-based model evaluation with regularization analysis
        """
        results = {}
        
        # 1. Bootstrap validation
        n_iterations = min(1000, int(2**len(y)))  # Adjust for small samples
        bootstrap_scores = []
        feature_importances = []
        
        for i in range(n_iterations):
            # Bootstrap sample with stratification
            indices = self._stratified_bootstrap_indices(y)
            X_boot, y_boot = X[indices], y[indices]
            
            # Out-of-bag sample
            oob_indices = np.setdiff1d(np.arange(len(X)), np.unique(indices))
            
            if len(oob_indices) > 0:  # Only if we have OOB samples
                X_oob, y_oob = X[oob_indices], y[oob_indices]
                
                # Fit and evaluate
                model.fit(X_boot, y_boot)
                score = model.score(X_oob, y_oob)
                bootstrap_scores.append(score)
                
                # Store feature importances if available
                if hasattr(model, 'coef_'):
                    feature_importances.append(model.coef_[0])
                elif hasattr(model, 'feature_importances_'):
                    feature_importances.append(model.feature_importances_)
        
        # Compute confidence intervals
        if bootstrap_scores:
            results['bootstrap'] = {
                'mean_score': np.mean(bootstrap_scores),
                'ci': (np.percentile(bootstrap_scores, 2.5),
                      np.percentile(bootstrap_scores, 97.5)),
                'std': np.std(bootstrap_scores)
            }
        
        # 2. Regularization analysis for models that support it
        if hasattr(model, 'C'):
            C_range = np.logspace(-4, 4, 20)
            reg_scores = []
            reg_score_stds = []
            
            for C in C_range:
                model.set_params(C=C)
                scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
                reg_scores.append(np.mean(scores))
                reg_score_stds.append(np.std(scores))
                
            results['regularization'] = {
                'C_values': C_range,
                'scores': reg_scores,
                'score_stds': reg_score_stds
            }
        
        # 3. Feature importance stability
        if feature_importances:
            feature_imp_array = np.array(feature_importances)
            results['feature_stability'] = {
                'mean_importance': np.mean(feature_imp_array, axis=0),
                'std_importance': np.std(feature_imp_array, axis=0),
                'feature_names': self.feature_columns
            }
        
        return results
    
    def _stratified_bootstrap_indices(self, y):
        """Helper function for stratified bootstrap sampling"""
        classes = np.unique(y)
        indices = []
        for c in classes:
            class_indices = np.where(y == c)[0]
            indices.extend(
                np.random.choice(
                    class_indices, 
                    size=len(class_indices), 
                    replace=True
                )
            )
        return np.array(indices)
    
    def plot_diagnostics(self, model_results, region_name):
        """
        Create diagnostic plots for a region's model results
        """
        plt.figure(figsize=(15, 10))
        
        # 1. Learning curves
        plt.subplot(2, 2, 1)
        lc = model_results['learning_curves']
        plt.plot(lc['train_sizes'], lc['train_scores_mean'], 'o-', label='Training')
        plt.plot(lc['train_sizes'], lc['val_scores_mean'], 'o-', label='Validation')
        plt.fill_between(lc['train_sizes'], 
                        lc['train_scores_mean'] - lc['train_scores_std'],
                        lc['train_scores_mean'] + lc['train_scores_std'], 
                        alpha=0.1)
        plt.fill_between(lc['train_sizes'], 
                        lc['val_scores_mean'] - lc['val_scores_std'],
                        lc['val_scores_mean'] + lc['val_scores_std'], 
                        alpha=0.1)
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend()
        
        # 2. Calibration plot
        if 'calibration' in model_results:
            plt.subplot(2, 2, 2)
            cal = model_results['calibration']
            plt.plot(cal['prob_pred'], cal['prob_true'], 'o-')
            plt.plot([0, 1], [0, 1], '--', color='gray')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('True Probability')
            plt.title('Calibration Plot')
        
        # 3. Cross-validation score distribution
        plt.subplot(2, 2, 3)
        cv = model_results['cv_results']
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_roc_auc']
        plt.boxplot([cv[m] for m in metrics])
        plt.xticks(range(1, len(metrics) + 1), 
                  [m.replace('test_', '') for m in metrics], 
                  rotation=45)
        plt.title('Cross-validation Metrics')
        
        # 4. Feature importances if available
        if 'feature_stability' in model_results:
            plt.subplot(2, 2, 4)
            feat = model_results['feature_stability']
            plt.errorbar(range(len(feat['feature_names'])), 
                        feat['mean_importance'],
                        yerr=feat['std_importance'],
                        fmt='o')
            plt.xticks(range(len(feat['feature_names'])), 
                      feat['feature_names'], 
                      rotation=45)
            plt.title('Feature Importance Stability')
        
        plt.tight_layout()
        plt.savefig(f'../figures/ml_diagnostics_{region_name}.png')
        plt.close()

    def analyze_region(self, region: str):
        """
        Analyze a single region with robust error handling
        """
        try:
            # Prepare data
            X, y = self.prepare_region_data(region)
            
            # Check class balance
            class_counts = np.bincount(y)
            min_class_count = min(class_counts)
            
            if min_class_count < 3:  # Minimum samples needed for CV
                raise ValueError(f"Insufficient samples per class: {class_counts}")
            
            # Determine number of CV folds based on sample size
            n_splits = min(5, min_class_count)
            
            # Initialize models with balanced class weights
            models = {
                'logistic': LogisticRegression(
                    C=0.1, 
                    class_weight='balanced',
                    max_iter=1000,  # Increase max iterations
                    random_state=42
                ),
                'rf': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=3,
                    class_weight='balanced',
                    random_state=42,
                    min_samples_leaf=2  # Ensure minimum samples in leaves
                ),
                'svm': SVC(
                    C=0.1,
                    kernel='rbf',
                    probability=True,
                    class_weight='balanced',
                    random_state=42
                )
            }
            
            region_results = {}
            
            # Scale features once for all models
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            for model_name, model in models.items():
                try:
                    # Basic evaluation
                    basic_results = self.enhanced_model_evaluation(
                        X_scaled, y, model, region
                    )
                    
                    # Robust analysis
                    robust_results = self.robust_model_analysis(
                        X_scaled, y, model, region
                    )
                    
                    # Store results
                    region_results[model_name] = {
                        'basic_evaluation': basic_results,
                        'robust_analysis': robust_results,
                        'data_info': {
                            'n_samples': len(y),
                            'class_balance': class_counts.tolist(),
                            'n_features': X.shape[1]
                        }
                    }
                    
                    # Create diagnostic plots
                    self.plot_diagnostics(
                        {**basic_results, **robust_results},
                        f"{region}_{model_name}"
                    )
                    
                except Exception as e:
                    self.logger.warning(
                        f"Error in {model_name} analysis for {region}: {str(e)}"
                    )
                    continue
            
            return region_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing region {region}: {str(e)}")
            return None
        
def main_diagnostics():
    # Initialize analyses
    stat_analysis = RegionalAnalysis_unpaired()
    stat_analysis.load_data()
    
    ml_analysis = EnhancedRegionalMLAnalysis(stat_analysis)
    
    # Analyze all regions
    for region in stat_analysis.common_regions:
        results = ml_analysis.analyze_region(region)
        
        if results is not None:
            print(f"\nResults for {region}:")
            for model_name, model_results in results.items():
                print(f"\n{model_name} model:")
                print("Bootstrap CI:", 
                      model_results['robust_analysis']['bootstrap']['ci'])
                print("Mean CV ROC-AUC:", 
                      model_results['basic_evaluation']['cv_results']['test_roc_auc'].mean())

if __name__ == "__main__":
    main_diagnostics()