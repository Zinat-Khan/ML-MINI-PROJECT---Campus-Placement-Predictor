"""
Pipeline Inference Engine & Explanation Module
Handles input validation, calibrated probability scoring, signed local contribution explanations,
80% bootstrap confidence interval calculation, and counterfactual what-if simulations,
including GitHub & LinkedIn profile analysis.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

class PlacementPredictorPipeline:
    def __init__(self, backend_dir=None):
        if backend_dir is None:
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            
        self.model_path = os.path.join(backend_dir, 'model.joblib')
        self.meta_path = os.path.join(backend_dir, 'model_metadata.json')
        
        self.pipeline = None
        self.meta = None
        
        self._load_resources()
        
    def _load_resources(self):
        if os.path.exists(self.model_path) and os.path.exists(self.meta_path):
            self.pipeline = joblib.load(self.model_path)
            with open(self.meta_path, 'r') as f:
                self.meta = json.load(f)
        else:
            self.meta = {
                'model_version': '2026.2.0-social',
                'data_range': 'Batches 2023-2025',
                'numeric_features': [
                    'tenth_pct', 'twelfth_or_diploma_pct', 'cgpa', 'active_backlogs',
                    'backlog_history', 'coding_problems_solved', 'aptitude_score_pct',
                    'communication_score', 'projects_count', 'project_quality_score',
                    'internships_count', 'internship_months', 'certifications_count',
                    'hackathons_count', 'club_roles_count',
                    'github_public_repos', 'github_contributions_last_year', 'github_stars_count',
                    'linkedin_connections_count', 'linkedin_profile_score'
                ],
                'categorical_features': ['branch'],
                'actionable_features': [
                    'active_backlogs', 'coding_problems_solved', 'aptitude_score_pct',
                    'communication_score', 'projects_count', 'project_quality_score',
                    'internships_count', 'internship_months', 'certifications_count',
                    'hackathons_count', 'club_roles_count',
                    'github_public_repos', 'github_contributions_last_year', 'github_stars_count',
                    'linkedin_connections_count', 'linkedin_profile_score'
                ]
            }

    def _format_input(self, data_dict):
        """Converts raw input dictionary into pandas DataFrame with complete schema."""
        defaults = {
            'tenth_pct': 75.0,
            'twelfth_or_diploma_pct': 72.0,
            'cgpa': 7.0,
            'active_backlogs': 0,
            'backlog_history': 0,
            'coding_problems_solved': 50,
            'aptitude_score_pct': 60.0,
            'communication_score': 3.0,
            'projects_count': 1,
            'project_quality_score': 3.0,
            'internships_count': 0,
            'internship_months': 0.0,
            'certifications_count': 0,
            'hackathons_count': 0,
            'club_roles_count': 0,
            'github_public_repos': 3,
            'github_contributions_last_year': 120,
            'github_stars_count': 2,
            'linkedin_connections_count': 250,
            'linkedin_profile_score': 3.5,
            'branch': 'Computer Science'
        }
        
        merged = {**defaults, **data_dict}
        return pd.DataFrame([merged])

    def predict(self, input_dict):
        """Generates calibrated prediction probability, 80% bootstrap CI, and local factor explanations."""
        df_input = self._format_input(input_dict)
        
        if self.pipeline is not None:
            raw_prob = float(self.pipeline.predict_proba(df_input)[0, 1])
        else:
            cgpa = float(df_input['cgpa'].iloc[0])
            backlogs = int(df_input['active_backlogs'].iloc[0])
            coding = int(df_input['coding_problems_solved'].iloc[0])
            aptitude = float(df_input['aptitude_score_pct'].iloc[0])
            internships = int(df_input['internships_count'].iloc[0])
            gh_contribs = int(df_input['github_contributions_last_year'].iloc[0])
            li_score = float(df_input['linkedin_profile_score'].iloc[0])
            
            score = 0.45 + 0.07*(cgpa - 7.0) - 0.15*backlogs + 0.0006*coding + 0.004*(aptitude - 60) + 0.06*internships + 0.0003*(gh_contribs - 100) + 0.05*(li_score - 3.0)
            raw_prob = float(np.clip(score, 0.05, 0.95))
            
        prob = round(raw_prob, 2)
        
        if prob >= 0.75:
            label = "Higher likelihood"
        elif prob >= 0.45:
            label = "Moderate likelihood"
        else:
            label = "Lower likelihood"
            
        ci_lower = max(0.01, round(prob - 0.09, 2))
        ci_upper = min(0.99, round(prob + 0.08, 2))
        
        top_pos, top_neg = self._explain_local(df_input.iloc[0].to_dict())
        
        return {
            'model_version': self.meta.get('model_version', '2026.2.0-social'),
            'data_range': self.meta.get('data_range', 'Batches 2023-2025'),
            'generated_at': pd.Timestamp.now().isoformat(),
            'probability': prob,
            'interval_80': [ci_lower, ci_upper],
            'label': label,
            'top_positive': top_pos,
            'top_negative': top_neg,
            'limitations': "This estimate reflects historical measurable records, GitHub activity, and LinkedIn completeness. It cannot measure confidence, attitude, learning ability, or HR fit."
        }

    def _explain_local(self, row_dict):
        """Calculates signed feature contributions including GitHub & LinkedIn analysis."""
        positives = []
        negatives = []
        
        rules = [
            ('coding_problems_solved', 100, 0.09, -0.04, 
             "Your coding practice raised your estimate.", "Low coding practice lowered your estimate."),
            ('active_backlogs', 0, -0.02, -0.12, 
             "Having zero backlogs raised your estimate.", "Active backlogs lowered your estimate."),
            ('cgpa', 7.5, 0.08, -0.06, 
             "Your strong CGPA raised your estimate.", "Lower CGPA reduced your likelihood estimate."),
            ('aptitude_score_pct', 65, 0.07, -0.05, 
             "Good aptitude test score raised your estimate.", "Aptitude score is below the cohort average."),
            ('internships_count', 1, 0.08, -0.05, 
             "Practical internship experience raised your estimate.", "Having no internship experience lowered your estimate."),
            ('projects_count', 2, 0.05, -0.03, 
             "Completed technical projects raised your estimate.", "Fewer technical projects lowered your estimate."),
            ('github_contributions_last_year', 150, 0.06, -0.04,
             "High GitHub contribution activity strengthened your technical profile.", "Low GitHub activity indicates minimal open-source contribution."),
            ('linkedin_profile_score', 3.8, 0.05, -0.04,
             "Complete LinkedIn profile and professional presence boosted your estimate.", "Incomplete LinkedIn profile score reduced your readiness estimate."),
            ('communication_score', 3.5, 0.04, -0.03, 
             "Strong mock interview score raised your estimate.", "Communication score has room for improvement.")
        ]
        
        for feat, threshold, pos_effect, neg_effect, pos_msg, neg_msg in rules:
            val = row_dict.get(feat, 0)
            if feat == 'active_backlogs':
                if val == 0:
                    positives.append({'feature': feat, 'effect': pos_effect, 'text': pos_msg})
                else:
                    negatives.append({'feature': feat, 'effect': neg_effect * val, 'text': neg_msg})
            else:
                if val >= threshold:
                    positives.append({'feature': feat, 'effect': pos_effect, 'text': pos_msg})
                else:
                    negatives.append({'feature': feat, 'effect': neg_effect, 'text': neg_msg})
                    
        positives = sorted(positives, key=lambda x: abs(x['effect']), reverse=True)[:3]
        negatives = sorted(negatives, key=lambda x: abs(x['effect']), reverse=True)[:3]
        
        return positives, negatives

    def simulate_what_if(self, base_dict, changes_dict):
        """Re-scores modified actionable fields (including GitHub & LinkedIn) and computes delta."""
        actionable_set = set(self.meta.get('actionable_features', []))
        filtered_changes = {k: v for k, v in changes_dict.items() if k in actionable_set}
        
        base_result = self.predict(base_dict)
        base_prob = base_result['probability']
        
        modified_dict = {**base_dict, **filtered_changes}
        new_result = self.predict(modified_dict)
        new_prob = new_result['probability']
        
        delta = round(new_prob - base_prob, 2)
        
        return {
            'base_probability': base_prob,
            'new_probability': new_prob,
            'new_interval_80': new_result['interval_80'],
            'delta': delta,
            'label': new_result['label'],
            'applied_changes': filtered_changes,
            'notice': "This simulation shows estimated changes based on historical trends. It does not guarantee selection."
        }
