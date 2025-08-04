import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import json
from jinja2 import Template
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from ..metrics import FairnessMetrics, calculate_all_metrics, calculate_metrics_for_multiple_features, MultipleFairnessMetrics
from .fairness_report import interpret_metrics, suggest_improvements
import os


class InteractiveFairnessReporter:
    """Generate interactive fairness reports with support for multiple sensitive features."""
    
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
        
        # Calculate overall performance metrics (not stratified by any sensitive feature)
        overall_performance = self._calculate_overall_performance(y_true, y_pred, y_prob)
        
        # Process each feature
        features = list(sensitive_features_dict.keys())
        feature_summaries = {}
        feature_stats = {}
        model_performance = {}
        fairness_metrics = {}
        fairness_analysis = {}
        recommendations = {}
        performance_plots = {}
        
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
            
            # Calculate model performance
            model_performance[feature_name] = self._calculate_model_performance(
                y_true, y_pred, feature_values, y_prob
            )
            
            # Store metrics and analysis
            fairness_metrics[feature_name] = metrics
            fairness_analysis[feature_name] = self._prepare_metrics_for_template(
                metrics, interpretations
            )
            recommendations[feature_name] = feature_recommendations
            
            # Generate plots for this feature
            performance_plots[feature_name] = self._generate_performance_plots(
                y_true, y_pred, feature_values, y_prob, output_path, feature_name
            )
        
        # Prepare template data
        template_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_samples': len(y_true),
            'num_features': len(features),
            'features': features,
            'overall_performance': overall_performance,
            'feature_summaries': feature_summaries,
            'feature_stats': feature_stats,
            'model_performance': model_performance,
            'fairness_metrics': fairness_metrics,
            'fairness_analysis': fairness_analysis,
            'recommendations': recommendations,
            'performance_plots': performance_plots
        }
        
        # Generate HTML report
        template = Template(self.report_template)
        html_content = template.render(**template_data)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return {
            'all_metrics': all_metrics,
            'feature_stats': feature_stats,
            'recommendations': recommendations,
            'report_path': output_path
        }
    
    def _calculate_fairness_score(self, metrics: FairnessMetrics) -> float:
        """Calculate an overall fairness score (0-100) based on all 4 metrics."""
        scores = []
        
        # 1. Demographic parity score
        dp_values = list(metrics.demographic_parity.values())
        if dp_values:
            dp_variance = np.var(dp_values)
            dp_score = max(0, 100 - dp_variance * 1000)
            scores.append(dp_score)
        
        # 2. Equal opportunity score
        eo_values = [v for v in metrics.equal_opportunity.values() if not np.isnan(v)]
        if eo_values:
            eo_variance = np.var(eo_values)
            eo_score = max(0, 100 - eo_variance * 1000)
            scores.append(eo_score)
        
        # 3. Calibration parity score (using both PPV and NPV)
        ppv_values = []
        npv_values = []
        for group_metrics in metrics.calibration_parity.values():
            if isinstance(group_metrics, dict):
                if 'PPV' in group_metrics and not np.isnan(group_metrics['PPV']):
                    ppv_values.append(group_metrics['PPV'])
                if 'NPV' in group_metrics and not np.isnan(group_metrics['NPV']):
                    npv_values.append(group_metrics['NPV'])
        
        calibration_scores = []
        if ppv_values:
            ppv_variance = np.var(ppv_values)
            ppv_score = max(0, 100 - ppv_variance * 1000)
            calibration_scores.append(ppv_score)
        
        if npv_values:
            npv_variance = np.var(npv_values)
            npv_score = max(0, 100 - npv_variance * 1000)
            calibration_scores.append(npv_score)
        
        if calibration_scores:
            cp_score = np.mean(calibration_scores)
            scores.append(cp_score)
        
        # 4. Disparate impact score
        di_values = [v for v in metrics.disparate_impact.values() if not np.isnan(v)]
        if di_values:
            di_deviations = [abs(1 - v) for v in di_values]
            di_score = max(0, 100 - np.mean(di_deviations) * 200)
            scores.append(di_score)
        
        # Return mean of all 4 metric scores
        return round(np.mean(scores) if scores else 0, 1)
    
    def _generate_feature_summary(self, metrics: FairnessMetrics, interpretations: Dict, 
                                  sensitive_features: np.ndarray, feature_name: str) -> str:
        """Generate a summary for a specific feature."""
        unique_groups = sorted(np.unique(sensitive_features))
        groups_str = ', '.join(str(g) for g in unique_groups)
        
        feature_desc = f"Analysis for <b>{feature_name}</b> with groups: <b>{groups_str}</b>. "
        
        # Check for issues
        issues = []
        
        if interpretations['demographic_parity']['has_issue']:
            issues.append("demographic parity")
        if interpretations['equal_opportunity']['has_issue']:
            issues.append("equal opportunity")
        if interpretations['calibration_parity']['has_issue']:
            issues.append("calibration")
        if interpretations['disparate_impact']['has_issue']:
            issues.append("disparate impact")
        
        # Add issue summary
        if not issues:
            summary = "The model demonstrates good fairness across all evaluated metrics for this feature."
        elif len(issues) == 1:
            summary = f"The model shows potential fairness issues in {issues[0]} for this feature."
        else:
            summary = f"The model shows potential fairness issues in multiple areas for this feature: {', '.join(issues)}."
        
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
    
    def _generate_performance_plots(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  sensitive_features: np.ndarray, y_prob: Optional[np.ndarray] = None,
                                  output_path: str = "interactive_fairness_report.html",
                                  feature_name: str = "") -> Dict:
        """Generate performance visualization plots for a specific feature."""
        import os
        from ..visualization.fairness_plots import plot_group_distributions, plot_confusion_matrices, create_fairness_dashboard
        
        # Get base filename without extension
        base_name = os.path.splitext(output_path)[0]
        plots = {}
        
        try:
            # Generate group distributions plot
            distributions_path = f"{base_name}_{feature_name}_distributions.png"
            plot_group_distributions(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
                y_prob=y_prob,
                save_path=distributions_path,
                feature_name=feature_name
            )
            plots['distributions'] = os.path.basename(distributions_path)
        except Exception as e:
            print(f"Warning: Could not generate distributions plot for {feature_name}: {e}")
            plots['distributions'] = None
        
        try:
            # Generate confusion matrices plot
            confusion_path = f"{base_name}_{feature_name}_confusion_matrices.png"
            plot_confusion_matrices(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
                save_path=confusion_path,
                feature_name=feature_name
            )
            plots['confusion_matrices'] = os.path.basename(confusion_path)
        except Exception as e:
            print(f"Warning: Could not generate confusion matrices plot for {feature_name}: {e}")
            plots['confusion_matrices'] = None
        
        try:
            # Generate fairness dashboard
            dashboard_path = f"{base_name}_{feature_name}_dashboard.png"
            metrics = calculate_all_metrics(y_true, y_pred, sensitive_features, y_prob)
            create_fairness_dashboard(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
                y_prob=y_prob,
                metrics=metrics,
                save_path=dashboard_path,
                feature_name=feature_name
            )
            plots['dashboard'] = os.path.basename(dashboard_path)
        except Exception as e:
            print(f"Warning: Could not generate dashboard plot for {feature_name}: {e}")
            plots['dashboard'] = None
        
        return plots
    
    def _prepare_metrics_for_template(self, metrics: FairnessMetrics, interpretations: Dict) -> Dict:
        """Prepare metrics data for template rendering."""
        
        # Calculate deviations from average
        dp_avg = np.mean(list(metrics.demographic_parity.values()))
        dp_deviations = {k: v - dp_avg for k, v in metrics.demographic_parity.items()}
        
        eo_values = [v for v in metrics.equal_opportunity.values() if not np.isnan(v)]
        eo_avg = np.mean(eo_values) if eo_values else 0
        eo_deviations = {k: (v - eo_avg if not np.isnan(v) else np.nan) 
                        for k, v in metrics.equal_opportunity.items()}
        
        return {
            'demographic_parity': metrics.demographic_parity,
            'demographic_parity_deviations': dp_deviations,
            'demographic_parity_status': self._get_status_class(interpretations['demographic_parity']['severity']),
            'demographic_parity_interpretation': interpretations['demographic_parity']['interpretation'],
            
            'equal_opportunity': metrics.equal_opportunity,
            'equal_opportunity_deviations': eo_deviations,
            'equal_opportunity_status': self._get_status_class(interpretations['equal_opportunity']['severity']),
            'equal_opportunity_interpretation': interpretations['equal_opportunity']['interpretation'],
            
            'calibration_parity': metrics.calibration_parity,
            'calibration_parity_status': self._get_status_class(interpretations['calibration_parity']['severity']),
            'calibration_parity_interpretation': interpretations['calibration_parity']['interpretation'],
            
            'disparate_impact': metrics.disparate_impact,
            'disparate_impact_status': self._get_status_class(interpretations['disparate_impact']['severity']),
            'disparate_impact_interpretation': interpretations['disparate_impact']['interpretation'],
        }
    
    def _get_status_class(self, severity: str) -> str:
        """Convert severity to CSS class."""
        mapping = {
            'none': 'success',
            'low': 'success',
            'medium': 'warning',
            'high': 'danger'
        }
        return mapping.get(severity, 'warning')
    
    def _calculate_overall_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     y_prob: Optional[np.ndarray] = None) -> Dict:
        """Calculate overall performance metrics without stratification."""
        total_samples = len(y_true)
        case_count = np.sum(y_true)  # Count of true label = 1
        case_percentage = (case_count / total_samples) * 100 if total_samples > 0 else 0
        overall_accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate AUC if probabilities are available
        if y_prob is not None and len(np.unique(y_true)) > 1:
            overall_auc = roc_auc_score(y_true, y_prob)
        else:
            overall_auc = np.nan
        
        return {
            'total_samples': total_samples,
            'case_count': int(case_count),
            'case_percentage': case_percentage,
            'overall_accuracy': overall_accuracy,
            'overall_auc': overall_auc
        }


def generate_interactive_fairness_report(y_true: np.ndarray, 
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
    dict : Report summary including metrics, interpretations, and recommendations
    """
    reporter = InteractiveFairnessReporter()
    return reporter.generate_interactive_report(
        y_true, y_pred, sensitive_features_dict, y_prob, output_path, privileged_groups
    )