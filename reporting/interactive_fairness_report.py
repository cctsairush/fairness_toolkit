"""
Interactive fairness report with embedded images (no external PNG files).
All images are embedded as base64-encoded data URIs in the HTML.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import json
from jinja2 import Template
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from ..metrics import FairnessMetrics, calculate_all_metrics, calculate_metrics_for_multiple_features, MultipleFairnessMetrics
from .fairness_report import interpret_metrics, suggest_improvements
# Removed circular import
import os


class InteractiveFairnessReporter:
    """Generate interactive fairness reports with embedded images (no external PNG files)."""
    
    def __init__(self):
        # Load the interactive template
        template_path = os.path.join(os.path.dirname(__file__), 'interactive_template.html')
        with open(template_path, 'r') as f:
            self.report_template = f.read()
    
    def generate_interactive_report(self, 
                                  y_true: np.ndarray, 
                                  y_pred: np.ndarray,
                                  sensitive_features_dict: Dict[str, np.ndarray],
                                  y_prob: Optional[np.ndarray] = None,
                                  output_path: str = "interactive_fairness_report.html",
                                  privileged_groups: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Generate an interactive fairness report for multiple sensitive features.
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels
        y_pred : array-like
            Predicted binary labels
        sensitive_features_dict : dict
            Dictionary mapping feature names to their values (e.g., {'race': race_array, 'sex': sex_array})
        y_prob : array-like, optional
            Predicted probabilities
        output_path : str
            Path to save the HTML report
        privileged_groups : dict, optional
            Dictionary mapping feature names to their privileged group values
            
        Returns:
        --------
        dict : Report summary including metrics, interpretations, and recommendations for all features
        """

        # Calculate metrics for all features
        all_metrics = calculate_metrics_for_multiple_features(
            y_true, y_pred, sensitive_features_dict, y_prob, privileged_groups
        )

        # If risk scores are provided, overwrite y_pred to be based on y_prob
        y_pred = y_pred if y_prob is None else (y_prob >= .4).astype(int)
        
        # Calculate overall performance metrics (not stratified by any sensitive feature)
        overall_performance = self._calculate_overall_performance(y_true, y_pred, y_prob)

        
        # Process each feature
        features = list(sensitive_features_dict.keys())
        feature_summaries = {}
        feature_stats = {}
        feature_masks = {}
        model_performance = {}
        fairness_metrics = {}
        fairness_analysis = {}
        recommendations = {}
        performance_plots = {}

        technical_analysis = self._generate_technical_analysis(features, all_metrics, output_path)
        
        for feature_name, feature_values in sensitive_features_dict.items():
            # Get metrics for this feature
            metrics = all_metrics.metrics_by_feature[feature_name]
            
            # Interpret metrics
            interpretations = interpret_metrics(metrics, feature_values)
            
            # Generate recommendations
            feature_recommendations = suggest_improvements(metrics, interpretations)
            
            # Calculate feature-specific stats
            unique_groups = np.unique(feature_values)
            num_groups = len(unique_groups)
            fairness_score = self._calculate_fairness_score(metrics)
            
            feature_stats[feature_name] = {
                'num_groups': num_groups,
                'fairness_score': fairness_score,
                'groups': [str(g) for g in unique_groups]
            }
            
            # Generate feature summary
            feature_summaries[feature_name] = self._generate_feature_summary(
                metrics, interpretations, feature_values, feature_name
            )
            
            # Create masks for interactive threshold adjustment
            feature_masks[feature_name] = {}
            for group in unique_groups:
                feature_masks[feature_name][str(group)] = (feature_values == group).tolist()
            
            # Calculate model performance by group
            model_performance[feature_name] = self._calculate_model_performance(
                y_true, y_pred, feature_values, y_prob
            )
            
            # Store fairness metrics in format expected by template
            fairness_metrics[feature_name] = self._format_metrics_for_template(metrics)
            
            # Generate fairness analysis
            fairness_analysis[feature_name] = interpretations
            
            # Store recommendations
            recommendations[feature_name] = feature_recommendations
            
            # Generate performance plots
            performance_plots[feature_name] = self._generate_performance_plots(
                y_true, y_pred, feature_values, y_prob, output_path, feature_name
            )
        
        # Prepare template variables
        template_vars = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'overall_performance': overall_performance,
            'features': features,
            'feature_summaries': feature_summaries,
            'feature_stats': feature_stats,
            'feature_masks': feature_masks,
            'model_performance': model_performance,
            'fairness_metrics': fairness_metrics,
            'fairness_analysis': fairness_analysis,
            'recommendations': recommendations,
            'performance_plots': performance_plots,
            'technical_analysis': technical_analysis,
            'y_true': y_true.tolist(),
            'y_prob': y_prob.tolist() if y_prob is not None else None
        }
        
        # Render template
        template = Template(self.report_template)
        html_content = template.render(**template_vars)
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Return summary
        return {
            'feature_stats': feature_stats,
            'recommendations': recommendations,
            'output_path': output_path
        }
    
    def _calculate_overall_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     y_prob: Optional[np.ndarray] = None) -> Dict:
        """Calculate overall performance metrics."""
        case_count = int(np.sum(y_true))
        total_samples = len(y_true)
        case_percentage = (case_count / total_samples) * 100 if total_samples > 0 else 0
        
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob) if y_prob is not None and len(np.unique(y_true)) > 1 else None
        
        performance = {
            'accuracy': accuracy,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'case_count': case_count,
            'case_percentage': case_percentage,
            'total_samples': total_samples,
            'auc': auc,
            # Template expects these specific field names
            'overall_accuracy': accuracy,
            'overall_auc': auc
        }
            
        return performance
        
    def _calculate_fairness_score(self, metrics: FairnessMetrics, heatmap: bool = False) -> Union[float, List[float]]:
        """Calculate overall fairness score."""
        if heatmap:
            # Return scores for each metric for heatmap
            scores = []
            
            # Demographic parity score
            dp_values = list(metrics.demographic_parity.values())
            dp_score = 1 - (max(dp_values) - min(dp_values)) if len(dp_values) > 1 else 1.0
            scores.append(max(0, min(1, dp_score)))
            
            # Equalized odds score
            tpr_values = [v['TPR'] for v in metrics.equalized_odds.values() if not np.isnan(v['TPR'])]
            fpr_values = [v['FPR'] for v in metrics.equalized_odds.values() if not np.isnan(v['FPR'])]
            
            tpr_score = 1 - (max(tpr_values) - min(tpr_values)) if len(tpr_values) > 1 else 1.0
            fpr_score = 1 - (max(fpr_values) - min(fpr_values)) if len(fpr_values) > 1 else 1.0
            eo_score = (tpr_score + fpr_score) / 2
            scores.append(max(0, min(1, eo_score)))
            
            # Calibration parity score
            ppv_values = [v['PPV'] for v in metrics.calibration_parity.values() if not np.isnan(v['PPV'])]
            ppv_score = 1 - (max(ppv_values) - min(ppv_values)) if len(ppv_values) > 1 else 1.0
            scores.append(max(0, min(1, ppv_score)))
            
            # Disparate impact score
            di_values = [v for v in metrics.disparate_impact.values() if not np.isnan(v)]
            di_score = min([min(v, 1/v) for v in di_values]) if di_values else 1.0
            scores.append(max(0, min(1, di_score)))
            
            return scores
        else:
            # Calculate single overall score
            scores = self._calculate_fairness_score(metrics, heatmap=True)
            return round(np.mean(scores) * 100, 1)
    
    def _generate_feature_summary(self, metrics: FairnessMetrics, interpretations: Dict, 
                                  sensitive_features: np.ndarray, feature_name: str) -> str:
        """Generate a summary for a specific feature."""
        unique_groups = sorted(np.unique(sensitive_features))
        groups_str = ', '.join(str(g) for g in unique_groups)
        
        feature_desc = f"Summary for <b>{feature_name}</b> with groups: <b>{groups_str}</b>. "
        
        # Check for issues
        issues = []
        
        if interpretations['demographic_parity']['has_issue']:
            issues.append("demographic parity")
        if interpretations['equalized_odds']['has_issue']:
            issues.append("equalized odds")
        if interpretations['calibration_parity']['has_issue']:
            issues.append("calibration")
        if interpretations['disparate_impact']['has_issue']:
            issues.append("disparate impact")
        
        # Add issue summary
        if not issues:
            summary = "The model demonstrates good fairness across all evaluated metrics for this feature."
        elif len(issues) == 1:
            summary = f"The model shows potential fairness issues in <b>{issues[0]}</b> for this feature."
        else:
            bold_issues = ', '.join([f"<b>{issue}</b>" for issue in issues])
            summary = f"The model shows potential fairness issues in multiple areas for this feature: {bold_issues}."
        
        return feature_desc + summary
    
    def _calculate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   sensitive_features: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict:
        """Calculate model performance metrics by group."""
        unique_groups = sorted(np.unique(sensitive_features))
        performance_data = {}
        stratified_performance = {}
        
        for group in unique_groups:
            group_mask = sensitive_features == group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]
            
            # Calculate basic metrics
            group_metrics = {
                'count': len(y_true_group),
                'accuracy': accuracy_score(y_true_group, y_pred_group),
                'precision': precision_score(y_true_group, y_pred_group, zero_division=0),
                'recall': recall_score(y_true_group, y_pred_group, zero_division=0),
                'f1_score': f1_score(y_true_group, y_pred_group, zero_division=0)
            }
            
            # Add AUC if probabilities are available
            if y_prob is not None:
                y_prob_group = y_prob[group_mask]
                if len(np.unique(y_true_group)) > 1:  # Need both classes for AUC
                    group_metrics['auc'] = roc_auc_score(y_true_group, y_prob_group)
                else:
                    group_metrics['auc'] = np.nan
            else:
                group_metrics['auc'] = np.nan
            
            performance_data[str(group)] = group_metrics
            
            # Also calculate stratified performance for the new table format
            group_case_count = int(np.sum(y_true_group))
            group_sample_size = len(y_true_group)
            group_case_percentage = (group_case_count / group_sample_size) * 100 if group_sample_size > 0 else 0
            
            stratified_performance[str(group)] = {
                'sample_size': group_sample_size,
                'case_count': group_case_count,
                'case_percentage': group_case_percentage,
                'accuracy': accuracy_score(y_true_group, y_pred_group),
                'auc': group_metrics['auc']
            }
        
        performance_data['stratified_performance'] = stratified_performance
        return performance_data
    
    def _format_metrics_for_template(self, metrics: FairnessMetrics) -> Dict:
        """Format metrics for template rendering."""
        return {
            'demographic_parity': dict(metrics.demographic_parity),
            'equalized_odds': dict(metrics.equalized_odds),
            'calibration_parity': dict(metrics.calibration_parity),
            'disparate_impact': dict(metrics.disparate_impact)
        }
    
    def _generate_technical_analysis(self, features: List[str], all_metrics: MultipleFairnessMetrics, 
                                    output_path: str = "interactive_fairness_report.html") -> Dict:
        """Generate technical analysis with embedded heatmap."""
        from ..visualization.fairness_plots import plot_heatmap
        
        metric_names = ['demographic parity', 'equalized odds', 'calibration parity', 'disparate impact']
        
        fairness_scores = {}
        for feature in features:
            metrics = all_metrics.metrics_by_feature[feature]
            fairness_scores[feature] = self._calculate_fairness_score(metrics, heatmap=True)
        
        try:
            # Generate heatmap as base64
            heatmap_base64 = plot_heatmap(
                features=features,
                metrics=metric_names,
                fairness_scores=fairness_scores,
            )
            return {"heatmap": heatmap_base64}
        except Exception as e:
            print(f"Warning: Could not generate heatmap: {e}")
            return {"heatmap": None}
    
    def _generate_performance_plots(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  sensitive_features: np.ndarray, y_prob: Optional[np.ndarray] = None,
                                  output_path: str = "interactive_fairness_report.html",
                                  feature_name: str = "") -> Dict:
        """Generate performance visualization plots as embedded base64 images."""
        from ..visualization.fairness_plots import (
            plot_group_distributions, 
            plot_confusion_matrices, 
            create_fairness_dashboard,
            plot_analysis_curves,
            plot_calibration_curves
        )
        
        plots = {}
        
        try:
            # Generate group distributions plot as base64
            plots['distributions'] = plot_group_distributions(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
                y_prob=y_prob,
                feature_name=feature_name
            )
        except Exception as e:
            print(f"Warning: Could not generate distributions plot for {feature_name}: {e}")
            plots['distributions'] = None
        
        if y_prob is not None:
            try:
                # Generate group calibration plot as base64
                plots['calibration'] = plot_calibration_curves(
                    y_true=y_true,
                    y_prob=y_prob,
                    sensitive_features=sensitive_features,
                    feature_name=feature_name
                )
            except Exception as e:
                print(f"Warning: Could not generate calibration plot for {feature_name}: {e}")
                plots['calibration'] = None
            
            try:
                # Generate group analysis plot as base64
                plots['analysis'] = plot_analysis_curves(
                    y_true=y_true,
                    y_prob=y_prob,
                    sensitive_features=sensitive_features,
                    feature_name=feature_name
                )
            except Exception as e:
                print(f"Warning: Could not generate analysis plot for {feature_name}: {e}")
                plots['analysis'] = None
        
        try:
            # Generate confusion matrices plot as base64
            plots['confusion_matrices'] = plot_confusion_matrices(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
                feature_name=feature_name
            )
        except Exception as e:
            print(f"Warning: Could not generate confusion matrices plot for {feature_name}: {e}")
            plots['confusion_matrices'] = None
        
        try:
            # Generate fairness dashboard as base64
            metrics = calculate_all_metrics(y_true, y_pred, sensitive_features, y_prob)
            plots['dashboard'] = create_fairness_dashboard(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
                y_prob=y_prob,
                metrics=metrics,
                feature_name=feature_name
            )
        except Exception as e:
            print(f"Warning: Could not generate dashboard plot for {feature_name}: {e}")
            plots['dashboard'] = None
        
        return plots


def generate_interactive_fairness_report(y_true: np.ndarray, 
                                        y_pred: np.ndarray,
                                        sensitive_features_dict: Dict[str, np.ndarray],
                                        y_prob: Optional[np.ndarray] = None,
                                        output_path: str = "interactive_fairness_report.html",
                                        privileged_groups: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Generate an interactive fairness report with all images embedded in the HTML.
    
    This function creates a self-contained HTML report without any external image files.
    All visualizations are embedded as base64-encoded data URIs directly in the HTML.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    sensitive_features_dict : dict
        Dictionary mapping feature names to their values (e.g., {'race': race_array, 'sex': sex_array})
    y_prob : array-like, optional
        Predicted probabilities
    output_path : str
        Path to save the HTML report
    privileged_groups : dict, optional
        Dictionary mapping feature names to their privileged group values
        
    Returns:
    --------
    dict : Report summary including metrics, interpretations, and recommendations for all features
    """
    reporter = InteractiveFairnessReporter()
    return reporter.generate_interactive_report(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features_dict=sensitive_features_dict,
        y_prob=y_prob,
        output_path=output_path,
        privileged_groups=privileged_groups
    )