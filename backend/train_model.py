"""
Train Model Script for Campus Placement Predictor
Synthesizes historical placement records (including GitHub & LinkedIn analysis features),
trains a calibrated Logistic Regression model, audits fairness across subgroups,
computes metrics, and serializes model.joblib.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, recall_score, brier_score_loss, precision_score, f1_score

# Set seed for reproducibility
NP_SEED = 42
np.random.seed(NP_SEED)

# Define Feature Groups
NUMERIC_FEATURES = [
    'tenth_pct',
    'twelfth_or_diploma_pct',
    'cgpa',
    'active_backlogs',
    'backlog_history',
    'coding_problems_solved',
    'aptitude_score_pct',
    'communication_score',
    'projects_count',
    'project_quality_score',
    'internships_count',
    'internship_months',
    'certifications_count',
    'hackathons_count',
    'club_roles_count',
    # New GitHub & LinkedIn Profile Analysis Features
    'github_public_repos',
    'github_contributions_last_year',
    'github_stars_count',
    'linkedin_connections_count',
    'linkedin_profile_score'
]

CATEGORICAL_FEATURES = ['branch']
SENSITIVE_AUDIT_FEATURES = ['gender'] # Used ONLY for fairness audit, NOT model input

ACTIONABLE_FEATURES = [
    'active_backlogs',
    'coding_problems_solved',
    'aptitude_score_pct',
    'communication_score',
    'projects_count',
    'project_quality_score',
    'internships_count',
    'internship_months',
    'certifications_count',
    'hackathons_count',
    'club_roles_count',
    'github_public_repos',
    'github_contributions_last_year',
    'github_stars_count',
    'linkedin_connections_count',
    'linkedin_profile_score'
]


def generate_synthetic_data(n_samples=600):
    """Generates realistic historical campus placement dataset based on institutional patterns."""
    np.random.seed(NP_SEED)
    
    batches = np.random.choice(['Batch 2023', 'Batch 2024', 'Batch 2025'], size=n_samples, p=[0.3, 0.35, 0.35])
    branches = np.random.choice(['Computer Science', 'Information Tech', 'Electronics', 'Mechanical', 'Civil'], size=n_samples, p=[0.35, 0.25, 0.2, 0.1, 0.1])
    genders = np.random.choice(['Male', 'Female', 'Other'], size=n_samples, p=[0.55, 0.43, 0.02])
    
    tenth_pct = np.clip(np.random.normal(78, 10, n_samples), 50, 98)
    twelfth_pct = np.clip(np.random.normal(75, 11, n_samples), 48, 97)
    cgpa = np.clip(np.random.normal(7.4, 1.1, n_samples), 5.0, 9.9)
    
    # Backlogs (correlated negatively with CGPA)
    active_backlogs = np.where(cgpa < 6.5, np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.3, 0.4, 0.2, 0.1]),
                               np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]))
    backlog_history = active_backlogs + np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    
    coding_problems = np.clip(np.round(np.random.exponential(110, n_samples)), 0, 600)
    aptitude_score = np.clip(np.random.normal(68, 14, n_samples), 30, 98)
    comm_score = np.clip(np.random.normal(3.4, 0.8, n_samples), 1.0, 5.0)
    
    projects_count = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05])
    project_quality = np.clip(np.random.normal(3.2, 0.9, n_samples), 1.0, 5.0)
    
    internships_count = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.45, 0.38, 0.14, 0.03])
    internship_months = internships_count * np.random.choice([2, 3, 6], size=n_samples, p=[0.5, 0.3, 0.2])
    
    certifications = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.4, 0.35, 0.2, 0.05])
    hackathons = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.6, 0.25, 0.1, 0.05])
    club_roles = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])

    # GitHub & LinkedIn Features
    github_repos = np.random.choice([1, 3, 6, 12, 20], size=n_samples, p=[0.2, 0.35, 0.25, 0.15, 0.05])
    github_contribs = np.clip(np.round(np.random.exponential(180, n_samples)), 10, 1200)
    github_stars = np.random.choice([0, 2, 5, 15, 50], size=n_samples, p=[0.5, 0.3, 0.12, 0.06, 0.02])

    linkedin_connections = np.clip(np.round(np.random.normal(320, 140, n_samples)), 20, 500)
    linkedin_score = np.clip(np.random.normal(3.6, 0.8, n_samples), 1.0, 5.0)
    
    # Realistic Placement Probability Logit Formulation
    logit = (
        -4.8 +
        0.02 * tenth_pct +
        0.02 * twelfth_pct +
        0.52 * (cgpa - 7.0) -
        0.85 * active_backlogs +
        0.005 * coding_problems +
        0.032 * (aptitude_score - 60) +
        0.30 * (comm_score - 3.0) +
        0.28 * projects_count +
        0.42 * internships_count +
        0.18 * certifications +
        # GitHub & LinkedIn Boost Factors
        0.0012 * (github_contribs - 100) +
        0.30 * (linkedin_score - 3.0) +
        0.0008 * (linkedin_connections - 200)
    )
    
    prob = 1 / (1 + np.exp(-logit))
    placed = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        'student_id': [f"STU-{1000+i}" for i in range(n_samples)],
        'batch': batches,
        'gender': genders,
        'branch': branches,
        'tenth_pct': np.round(tenth_pct, 1),
        'twelfth_or_diploma_pct': np.round(twelfth_pct, 1),
        'cgpa': np.round(cgpa, 2),
        'active_backlogs': active_backlogs,
        'backlog_history': backlog_history,
        'coding_problems_solved': coding_problems.astype(int),
        'aptitude_score_pct': np.round(aptitude_score, 1),
        'communication_score': np.round(comm_score, 1),
        'projects_count': projects_count,
        'project_quality_score': np.round(project_quality, 1),
        'internships_count': internships_count,
        'internship_months': np.round(internship_months, 1),
        'certifications_count': certifications,
        'hackathons_count': hackathons,
        'club_roles_count': club_roles,
        'github_public_repos': github_repos,
        'github_contributions_last_year': github_contribs.astype(int),
        'github_stars_count': github_stars,
        'linkedin_connections_count': linkedin_connections.astype(int),
        'linkedin_profile_score': np.round(linkedin_score, 1),
        'placed': placed
    })
    
    return df


def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculates Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return float(ece)


def main():
    print("Generating synthetic placement dataset with GitHub & LinkedIn profile analysis...")
    df = generate_synthetic_data(n_samples=600)
    
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(backend_dir, 'synthetic_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to: {csv_path} ({len(df)} records)")
    
    # Separate Inputs (X) and Target (y)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['placed']
    
    # Preprocessing Pipeline
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, NUMERIC_FEATURES),
            ('cat', cat_transformer, CATEGORICAL_FEATURES)
        ]
    )
    
    # Primary Model Candidate: Regularized Logistic Regression with Platt Calibration
    base_clf = LogisticRegression(max_iter=1000, C=1.0, random_state=NP_SEED)
    calibrated_clf = CalibratedClassifierCV(estimator=base_clf, cv=5, method='sigmoid')
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', calibrated_clf)
    ])
    
    # Perform Stratified 5-Fold Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=NP_SEED)
    
    roc_aucs, recalls, precisions, brier_scores, eces = [], [], [], [], []
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        full_pipeline.fit(X_tr, y_tr)
        probs = full_pipeline.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        roc_aucs.append(roc_auc_score(y_val, probs))
        recalls.append(recall_score(y_val, preds))
        precisions.append(precision_score(y_val, preds))
        brier_scores.append(brier_score_loss(y_val, probs))
        eces.append(calculate_ece(y_val.values, probs))
        
    print("\n--- Cross-Validation Performance Metrics ---")
    print(f"Mean ROC-AUC:      {np.mean(roc_aucs):.4f} (Target >= 0.75)")
    print(f"Mean Recall:       {np.mean(recalls):.4f} (Target >= 0.70)")
    print(f"Mean Precision:    {np.mean(precisions):.4f}")
    print(f"Mean Brier Score:  {np.mean(brier_scores):.4f}")
    print(f"Mean ECE:          {np.mean(eces):.4f} (Target <= 0.08)")
    
    # Fit on Full Dataset
    full_pipeline.fit(X, y)
    
    # Extract Feature Weights from underlying fitted base model
    fitted_preprocessor = full_pipeline.named_steps['preprocessor']
    fitted_calibrated_clf = full_pipeline.named_steps['classifier']
    
    cat_encoder = fitted_preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = NUMERIC_FEATURES + cat_feature_names
    
    base_coeffs = np.mean([estimator.estimator.coef_[0] for estimator in fitted_calibrated_clf.calibrated_classifiers_], axis=0)
    base_intercept = float(np.mean([estimator.estimator.intercept_[0] for estimator in fitted_calibrated_clf.calibrated_classifiers_]))
    
    feature_importance_map = dict(zip(all_feature_names, base_coeffs.tolist()))
    
    # Audit Demographic Fairness across Gender Subgroups
    print("\n--- Post-Hoc Demographic Fairness Audit ---")
    gender_metrics = {}
    for gender in df['gender'].unique():
        sub_df = df[df['gender'] == gender]
        if len(sub_df) >= 10:
            X_sub = sub_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
            y_sub = sub_df['placed']
            sub_probs = full_pipeline.predict_proba(X_sub)[:, 1]
            sub_preds = (sub_probs >= 0.5).astype(int)
            
            tpr = recall_score(y_sub, sub_preds) if sum(y_sub) > 0 else 0.0
            fpr = np.mean(sub_preds[y_sub == 0]) if sum(y_sub == 0) > 0 else 0.0
            auc = roc_auc_score(y_sub, sub_probs) if len(np.unique(y_sub)) > 1 else 0.5
            
            gender_metrics[gender] = {
                'sample_size': len(sub_df),
                'roc_auc': round(auc, 4),
                'tpr': round(tpr, 4),
                'fpr': round(fpr, 4)
            }
            print(f"Gender [{gender}]: n={len(sub_df)}, ROC-AUC={auc:.4f}, TPR={tpr:.4f}, FPR={fpr:.4f}")
            
    tprs = [v['tpr'] for v in gender_metrics.values()]
    tpr_gap = max(tprs) - min(tprs) if tprs else 0.0
    print(f"Max Gender TPR Gap: {tpr_gap:.4f} (Target <= 0.10)")
    
    # Save Model & Metadata Artifacts
    model_path = os.path.join(backend_dir, 'model.joblib')
    joblib.dump(full_pipeline, model_path)
    
    meta_path = os.path.join(backend_dir, 'model_metadata.json')
    metadata = {
        'model_version': '2026.2.0-social',
        'training_date': '2026-09-23',
        'data_range': 'Batches 2023-2025',
        'total_samples': len(df),
        'metrics': {
            'roc_auc': round(float(np.mean(roc_aucs)), 4),
            'recall': round(float(np.mean(recalls)), 4),
            'precision': round(float(np.mean(precisions)), 4),
            'brier_score': round(float(np.mean(brier_scores)), 4),
            'ece': round(float(np.mean(eces)), 4),
            'max_tpr_fairness_gap': round(float(tpr_gap), 4)
        },
        'feature_coefficients': feature_importance_map,
        'intercept': base_intercept,
        'all_feature_names': all_feature_names,
        'numeric_features': NUMERIC_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'actionable_features': ACTIONABLE_FEATURES,
        'fairness_audit': gender_metrics
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nModel and metadata with GitHub & LinkedIn analysis successfully saved to:")
    print(f" - {model_path}")
    print(f" - {meta_path}")

if __name__ == '__main__':
    main()
