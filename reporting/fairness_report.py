import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
from jinja2 import Template
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from ..metrics import FairnessMetrics, calculate_all_metrics, calculate_metrics_for_multiple_features


class FairnessReporter:
    """Generate comprehensive fairness reports with interpretations and recommendations."""
    
    def __init__(self):
        self.report_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Fairness Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        h3 { color: #666; }
        .metric-card { background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50; }
        .warning { background-color: #fff3cd; border-left-color: #ffc107; }
        .danger { background-color: #f8d7da; border-left-color: #dc3545; }
        .success { background-color: #d4edda; border-left-color: #28a745; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #4CAF50; color: white; }
        /* Dynamic header colors based on parent metric-card status */
        .success th { background-color: #28a745; }
        .warning th { background-color: #856404; }
        .danger th { background-color: #721c24; }
        tr:hover { background-color: #f5f5f5; }
        .recommendation { background-color: #e7f3ff; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #2196F3; }
        .summary-box { background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .metric-value { font-weight: bold; color: #4CAF50; }
        .timestamp { color: #666; font-size: 0.9em; }
        .feature-selector { background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .feature-selector label { font-weight: bold; margin-right: 10px; }
        .feature-selector select { padding: 8px 12px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }
        .feature-content { display: none; }
        .feature-content.active { display: block; }
        .hidden { display: none; }
    </style>
    <script>
        function changeFeature() {
            var selector = document.getElementById('featureSelect');
            var selectedFeature = selector.value;
            
            // Hide all feature content
            var allContent = document.getElementsByClassName('feature-content');
            for (var i = 0; i < allContent.length; i++) {
                allContent[i].classList.remove('active');
            }
            
            // Show selected feature content
            var selectedContent = document.getElementsByClassName('feature-' + selectedFeature);
            for (var i = 0; i < selectedContent.length; i++) {
                selectedContent[i].classList.add('active');
            }
        }
        
        window.onload = function() {
            changeFeature();
        };
    </script>
</head>
<body>
    <div class="container">
        <h1>Fairness Analysis Report</h1>
        <p class="timestamp">Generated on: {{ timestamp }}</p>
        
        <div class="summary-box">
            <h2>Executive Summary</h2>
            <p>{{ executive_summary }}</p>
            <ul>
                <li>Total Samples: <span class="metric-value">{{ total_samples }}</span></li>
                <li>Number of Groups: <span class="metric-value">{{ num_groups }}</span></li>
                <li>Overall Fairness Score: <span class="metric-value">{{ fairness_score }}%</span></li>
            </ul>
        </div>
        
        <h2>Model Performance by Group</h2>
        <div class="summary-box">
            <p>This section shows how the model performs across different groups for the sensitive feature being analyzed.</p>
            
            <table style="margin-top: 20px;">
                <tr>
                    <th>Group</th>
                    <th>Count</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>AUC-ROC</th>
                </tr>
                {% for group, metrics in model_performance.items() %}
                <tr>
                    <td><b>{{ group }}</b></td>
                    <td>{{ metrics.count }}</td>
                    <td>{{ "%.3f"|format(metrics.accuracy) }}</td>
                    <td>{{ "%.3f"|format(metrics.precision) }}</td>
                    <td>{{ "%.3f"|format(metrics.recall) }}</td>
                    <td>{{ "%.3f"|format(metrics.f1_score) }}</td>
                    <td>{{ "%.3f"|format(metrics.auc) if metrics.auc == metrics.auc else "N/A" }}</td>
                </tr>
                {% endfor %}
            </table>
            
            {% if performance_plots %}
            <div style="margin-top: 30px;">
                <h3>Performance Visualizations</h3>
                <div style="text-align: center;">
                    {% if performance_plots.distributions %}
                    <div style="margin: 20px 0;">
                        <h4>Group Distributions</h4>
                        <img src="{{ performance_plots.distributions }}" alt="Group Distributions" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px;">
                    </div>
                    {% endif %}
                    
                    {% if performance_plots.confusion_matrices %}
                    <div style="margin: 20px 0;">
                        <h4>Confusion Matrices by Group</h4>
                        <img src="{{ performance_plots.confusion_matrices }}" alt="Confusion Matrices" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px;">
                    </div>
                    {% endif %}
                </div>
            </div>
            {% endif %}
        </div>
        
        <h2>Fairness Metrics Analysis</h2>
        
        <h3>1. Demographic Parity</h3>
        <div class="metric-card {{ demographic_parity_status }}">
            <p>{{ demographic_parity_interpretation }}</p>
            <table>
                <tr>
                    <th>Group</th>
                    <th>Positive Prediction Rate</th>
                    <th>Deviation from Average</th>
                </tr>
                {% for group, value in demographic_parity.items() %}
                <tr>
                    <td>{{ group }}</td>
                    <td>{{ "%.3f"|format(value) }}</td>
                    <td>{{ "%.3f"|format(demographic_parity_deviations[group]) }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <h3>2. Equalized Odds</h3>
        <div class="metric-card {{ equalized_odds_status }}">
            <p>{{ equalized_odds_interpretation }}</p>
            <table>
                <tr>
                    <th>Group</th>
                    <th>True Positive Rate</th>
                    <th>False Positive Rate</th>
                    <th>TPR Deviation</th>
                    <th>FPR Deviation</th>
                </tr>
                {% for group, values in equalized_odds.items() %}
                <tr>
                    <td>{{ group }}</td>
                    <td>{{ "%.3f"|format(values.TPR) if values.TPR == values.TPR else "N/A" }}</td>
                    <td>{{ "%.3f"|format(values.FPR) if values.FPR == values.FPR else "N/A" }}</td>
                    <td>{{ "%.3f"|format(equalized_odds_deviations[group].TPR) if equalized_odds_deviations[group].TPR == equalized_odds_deviations[group].TPR else "N/A" }}</td>
                    <td>{{ "%.3f"|format(equalized_odds_deviations[group].FPR) if equalized_odds_deviations[group].FPR == equalized_odds_deviations[group].FPR else "N/A" }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <h3>3. Calibration Parity</h3>
        <div class="metric-card {{ calibration_parity_status }}">
            <p>{{ calibration_parity_interpretation }}</p>
            <table>
                <tr>
                    <th>Group</th>
                    <th>PPV</th>
                    <th>NPV</th>
                </tr>
                {% for group, values in calibration_parity.items() %}
                <tr>
                    <td>{{ group }}</td>
                    <td>{{ "%.3f"|format(values.PPV) if values.PPV == values.PPV else "N/A" }}</td>
                    <td>{{ "%.3f"|format(values.NPV) if values.NPV == values.NPV else "N/A" }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <h3>4. Disparate Impact</h3>
        <div class="metric-card {{ disparate_impact_status }}">
            <p>{{ disparate_impact_interpretation }}</p>
            <table>
                <tr>
                    <th>Group</th>
                    <th>Disparate Impact Ratio</th>
                    <th>Status</th>
                </tr>
                {% for group, value in disparate_impact.items() %}
                <tr>
                    <td>{{ group }}</td>
                    <td>{{ "%.3f"|format(value) if value == value else "N/A" }}</td>
                    <td>{{ "Fair" if 0.8 <= value <= 1.25 else "Potential Bias" }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <h2>Recommendations for Improvement</h2>
        {% for recommendation in recommendations %}
        <div class="recommendation">
            <h4>{{ recommendation.title }}</h4>
            <p>{{ recommendation.description }}</p>
            <ul>
                {% for action in recommendation.actions %}
                <li>{{ action }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""

    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                       sensitive_features: np.ndarray,
                       y_prob: Optional[np.ndarray] = None,
                       output_path: str = "fairness_report.html",
                       feature_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive fairness report."""
        
        # Calculate all metrics
        metrics = calculate_all_metrics(y_true, y_pred, sensitive_features, y_prob)
        
        # Interpret metrics and generate recommendations
        interpretations = interpret_metrics(metrics, sensitive_features)
        recommendations = suggest_improvements(metrics, interpretations)
        
        # Calculate summary statistics
        total_samples = len(y_true)
        num_groups = len(np.unique(sensitive_features))
        fairness_score = self._calculate_fairness_score(metrics)
        
        # Calculate model performance by group
        model_performance = self._calculate_model_performance(y_true, y_pred, sensitive_features, y_prob)
        
        # Generate performance plots
        performance_plots = self._generate_performance_plots(y_true, y_pred, sensitive_features, y_prob, output_path)
        
        # Prepare template data
        template_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_samples': total_samples,
            'num_groups': num_groups,
            'fairness_score': fairness_score,
            'executive_summary': self._generate_executive_summary(metrics, interpretations, sensitive_features, feature_name),
            'model_performance': model_performance,
            'performance_plots': performance_plots,
            **self._prepare_metrics_for_template(metrics, interpretations),
            'recommendations': recommendations
        }
        
        # Generate HTML report
        template = Template(self.report_template)
        html_content = template.render(**template_data)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return {
            'metrics': metrics,
            'interpretations': interpretations,
            'recommendations': recommendations,
            'fairness_score': fairness_score,
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
        
        # 2. Equalized odds score (using TPR and FPR)
        tpr_values = []
        fpr_values = []
        for group_metrics in metrics.equalized_odds.values():
            if isinstance(group_metrics, dict):
                if 'TPR' in group_metrics and not np.isnan(group_metrics['TPR']):
                    tpr_values.append(group_metrics['TPR'])
                if 'FPR' in group_metrics and not np.isnan(group_metrics['FPR']):
                    fpr_values.append(group_metrics['FPR'])
        
        equalized_odds_scores = []
        if tpr_values:
            tpr_variance = np.var(tpr_values)
            tpr_score = max(0, 100 - tpr_variance * 1000)
            equalized_odds_scores.append(tpr_score)
        
        if fpr_values:
            fpr_variance = np.var(fpr_values)
            fpr_score = max(0, 100 - fpr_variance * 1000)
            equalized_odds_scores.append(fpr_score)
        
        if equalized_odds_scores:
            eo_score = np.mean(equalized_odds_scores)
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
        
        # 4. Disparate impact score (fixed to account for 0.8-1.25 acceptable range)
        di_values = [v for v in metrics.disparate_impact.values() if not np.isnan(v)]
        if di_values:
            # Calculate violations: how far outside the acceptable 0.8-1.25 range
            violations = []
            for v in di_values:
                if v < 0.8:
                    violations.append(0.8 - v)  # Distance below 0.8
                elif v > 1.25:
                    violations.append(v - 1.25)  # Distance above 1.25
                else:
                    violations.append(0.0)  # Within acceptable range
            
            # Score based on average violation severity
            avg_violation = np.mean(violations)
            # More aggressive penalty: even small violations should result in lower scores
            di_score = max(0, 100 - avg_violation * 400)  # Increased penalty multiplier
            scores.append(di_score)
        
        # Return mean of all 4 metric scores
        return round(np.mean(scores) if scores else 0, 1)
    
    def _generate_executive_summary(self, metrics: FairnessMetrics, interpretations: Dict, 
                                    sensitive_features: np.ndarray, feature_name: Optional[str] = None) -> str:
        """Generate an executive summary of the fairness analysis."""
        # Get unique groups in the sensitive feature
        unique_groups = sorted(np.unique(sensitive_features))
        groups_str = ', '.join(str(g) for g in unique_groups)
        
        # Create feature description
        if feature_name:
            feature_desc = f"This report assesses fairness issues for sensitive feature <b>{feature_name}</b>. "
        else:
            feature_desc = "This report assesses fairness issues for a sensitive feature. "
        
        feature_desc += f"All possible categories include: <b>{groups_str}</b>."
        
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
            summary = " The model demonstrates good fairness across all evaluated metrics. No significant biases were detected between groups."
        elif len(issues) == 1:
            summary = f" The model shows potential fairness issues in {issues[0]}. Review the detailed analysis and recommendations below."
        else:
            summary = f" The model shows potential fairness issues in multiple areas: {', '.join(issues)}. Immediate attention is recommended."
        
        return feature_desc + summary
    
    def _calculate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   sensitive_features: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict:
        """Calculate model performance metrics by group."""
        unique_groups = sorted(np.unique(sensitive_features))
        performance_data = {}
        
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
        
        return performance_data
    
    def _generate_performance_plots(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  sensitive_features: np.ndarray, y_prob: Optional[np.ndarray] = None,
                                  output_path: str = "fairness_report.html") -> Dict:
        """Generate performance visualization plots."""
        import os
        from ..visualization.fairness_plots import plot_group_distributions, plot_confusion_matrices
        
        # Get base filename without extension
        base_name = os.path.splitext(output_path)[0]
        plots = {}
        
        try:
            # Generate group distributions plot
            distributions_path = f"{base_name}_distributions.png"
            plot_group_distributions(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
                y_prob=y_prob,
                save_path=distributions_path
            )
            plots['distributions'] = os.path.basename(distributions_path)
        except Exception as e:
            print(f"Warning: Could not generate distributions plot: {e}")
            plots['distributions'] = None
        
        try:
            # Generate confusion matrices plot
            confusion_path = f"{base_name}_confusion_matrices.png"
            plot_confusion_matrices(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
                save_path=confusion_path
            )
            plots['confusion_matrices'] = os.path.basename(confusion_path)
        except Exception as e:
            print(f"Warning: Could not generate confusion matrices plot: {e}")
            plots['confusion_matrices'] = None
        
        return plots
    
    def _prepare_metrics_for_template(self, metrics: FairnessMetrics, interpretations: Dict) -> Dict:
        """Prepare metrics data for template rendering."""
        
        # Calculate deviations from average
        dp_avg = np.mean(list(metrics.demographic_parity.values()))
        dp_deviations = {k: v - dp_avg for k, v in metrics.demographic_parity.items()}
        
        # For equalized odds, calculate deviations for both TPR and FPR
        tpr_values = [v['TPR'] for v in metrics.equalized_odds.values() if isinstance(v, dict) and 'TPR' in v and not np.isnan(v['TPR'])]
        fpr_values = [v['FPR'] for v in metrics.equalized_odds.values() if isinstance(v, dict) and 'FPR' in v and not np.isnan(v['FPR'])]
        
        tpr_avg = np.mean(tpr_values) if tpr_values else 0
        fpr_avg = np.mean(fpr_values) if fpr_values else 0
        
        eo_deviations = {}
        for k, v in metrics.equalized_odds.items():
            if isinstance(v, dict):
                tpr_dev = v['TPR'] - tpr_avg if 'TPR' in v and not np.isnan(v['TPR']) else np.nan
                fpr_dev = v['FPR'] - fpr_avg if 'FPR' in v and not np.isnan(v['FPR']) else np.nan
                eo_deviations[k] = {'TPR': tpr_dev, 'FPR': fpr_dev}
            else:
                eo_deviations[k] = {'TPR': np.nan, 'FPR': np.nan}
        
        return {
            'demographic_parity': metrics.demographic_parity,
            'demographic_parity_deviations': dp_deviations,
            'demographic_parity_status': self._get_status_class(interpretations['demographic_parity']['severity']),
            'demographic_parity_interpretation': interpretations['demographic_parity']['interpretation'],
            
            'equalized_odds': metrics.equalized_odds,
            'equalized_odds_deviations': eo_deviations,
            'equalized_odds_status': self._get_status_class(interpretations['equalized_odds']['severity']),
            'equalized_odds_interpretation': interpretations['equalized_odds']['interpretation'],
            
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


def interpret_metrics(metrics: FairnessMetrics, sensitive_features: np.ndarray) -> Dict[str, Dict]:
    """Interpret fairness metrics and identify issues."""
    interpretations = {}
    
    # Interpret Demographic Parity
    dp_values = list(metrics.demographic_parity.values())
    dp_range = max(dp_values) - min(dp_values)
    
    if dp_range < 0.1:
        dp_interpretation = "Excellent demographic parity. All groups have similar positive prediction rates."
        dp_severity = "none"
        dp_has_issue = False
    elif dp_range < 0.2:
        dp_interpretation = "Good demographic parity with minor differences between groups."
        dp_severity = "low"
        dp_has_issue = False
    elif dp_range < 0.3:
        dp_interpretation = "Moderate demographic parity issues. Some groups have notably different prediction rates."
        dp_severity = "medium"
        dp_has_issue = True
    else:
        dp_interpretation = "Significant demographic parity issues. Large disparities in prediction rates across groups."
        dp_severity = "high"
        dp_has_issue = True
    
    interpretations['demographic_parity'] = {
        'interpretation': dp_interpretation,
        'severity': dp_severity,
        'has_issue': dp_has_issue,
        'range': dp_range
    }
    
    # Interpret Equalized Odds
    tpr_values = [v['TPR'] for v in metrics.equalized_odds.values() if isinstance(v, dict) and 'TPR' in v and not np.isnan(v['TPR'])]
    fpr_values = [v['FPR'] for v in metrics.equalized_odds.values() if isinstance(v, dict) and 'FPR' in v and not np.isnan(v['FPR'])]
    
    if tpr_values and fpr_values:
        tpr_range = max(tpr_values) - min(tpr_values)
        fpr_range = max(fpr_values) - min(fpr_values)
        max_range = max(tpr_range, fpr_range)
        
        if max_range < 0.1:
            eo_interpretation = "Excellent equalized odds. All groups have similar true positive and false positive rates."
            eo_severity = "none"
            eo_has_issue = False
        elif max_range < 0.2:
            eo_interpretation = "Good equalized odds with minor differences in error rates."
            eo_severity = "low"
            eo_has_issue = False
        elif max_range < 0.3:
            eo_interpretation = "Moderate equalized odds issues. Some groups have different error rates."
            eo_severity = "medium"
            eo_has_issue = True
        else:
            eo_interpretation = "Significant equalized odds issues. Large disparities in true positive and/or false positive rates."
            eo_severity = "high"
            eo_has_issue = True
    else:
        eo_interpretation = "Unable to calculate equalized odds metrics due to insufficient samples."
        eo_severity = "none"
        eo_has_issue = False
        max_range = 0
    
    interpretations['equalized_odds'] = {
        'interpretation': eo_interpretation,
        'severity': eo_severity,
        'has_issue': eo_has_issue,
        'range': max_range
    }
    
    # Interpret Calibration Parity
    ppv_values = [v['PPV'] for v in metrics.calibration_parity.values() if not np.isnan(v['PPV'])]
    npv_values = [v['NPV'] for v in metrics.calibration_parity.values() if not np.isnan(v['NPV'])]
    
    if ppv_values and npv_values:
        ppv_range = max(ppv_values) - min(ppv_values)
        npv_range = max(npv_values) - min(npv_values)
        max_range = max(ppv_range, npv_range)
        
        if max_range < 0.1:
            calib_interpretation = "Excellent calibration parity. Predictions are well-calibrated across groups."
            calib_severity = "none"
            calib_has_issue = False
        elif max_range < 0.2:
            calib_interpretation = "Good calibration with minor differences."
            calib_severity = "low"
            calib_has_issue = False
        elif max_range < 0.3:
            calib_interpretation = "Moderate calibration issues. Prediction reliability varies between groups."
            calib_severity = "medium"
            calib_has_issue = True
        else:
            calib_interpretation = "Significant calibration issues. Large differences in prediction reliability."
            calib_severity = "high"
            calib_has_issue = True
    else:
        calib_interpretation = "Unable to calculate calibration metrics for some groups."
        calib_severity = "none"
        calib_has_issue = False
        max_range = 0
    
    interpretations['calibration_parity'] = {
        'interpretation': calib_interpretation,
        'severity': calib_severity,
        'has_issue': calib_has_issue,
        'range': max_range
    }
    
    # Interpret Disparate Impact
    di_values = [v for v in metrics.disparate_impact.values() if not np.isnan(v)]
    if di_values:
        violations = sum(1 for v in di_values if v < 0.8 or v > 1.25)
        
        if violations == 0:
            di_interpretation = "No disparate impact detected. All groups within acceptable range (0.8-1.25)."
            di_severity = "none"
            di_has_issue = False
        elif violations == 1:
            di_interpretation = "Minor disparate impact detected in one group."
            di_severity = "low"
            di_has_issue = True
        elif violations < len(di_values) / 2:
            di_interpretation = "Moderate disparate impact affecting some groups."
            di_severity = "medium"
            di_has_issue = True
        else:
            di_interpretation = "Significant disparate impact affecting multiple groups."
            di_severity = "high"
            di_has_issue = True
    else:
        di_interpretation = "Unable to calculate disparate impact."
        di_severity = "none"
        di_has_issue = False
        violations = 0
    
    interpretations['disparate_impact'] = {
        'interpretation': di_interpretation,
        'severity': di_severity,
        'has_issue': di_has_issue,
        'violations': violations
    }
    
    return interpretations


def suggest_improvements(metrics: FairnessMetrics, interpretations: Dict) -> List[Dict]:
    """Generate specific recommendations for improving fairness."""
    recommendations = []
    
    # Demographic Parity recommendations
    if interpretations['demographic_parity']['has_issue']:
        rec = {
            'title': 'Addressing Demographic Parity Issues',
            'description': 'The model shows different prediction rates across groups. Consider these approaches:',
            'actions': [
                'Implement demographic parity post-processing to adjust prediction thresholds per group',
                'Use adversarial debiasing during training to reduce dependence on sensitive attributes',
                'Apply reweighting or resampling techniques to balance the training data',
                'Consider using fairness-aware algorithms like Fair SVM or Fair Random Forest'
            ]
        }
        recommendations.append(rec)
    
    # Equalized Odds recommendations
    if interpretations['equalized_odds']['has_issue']:
        rec = {
            'title': 'Improving Equalized Odds',
            'description': 'Different groups have unequal true positive rates and/or false positive rates. Try these methods:',
            'actions': [
                'Apply equalized odds post-processing to adjust decision thresholds per group',
                'Use cost-sensitive learning with group-specific costs for both TPR and FPR',
                'Implement fairness constraints during model training to minimize TPR and FPR disparities',
                'Consider separate models for different groups with careful validation',
                'Use adversarial debiasing techniques to enforce equalized odds'
            ]
        }
        recommendations.append(rec)
    
    # Calibration recommendations
    if interpretations['calibration_parity']['has_issue']:
        rec = {
            'title': 'Enhancing Calibration Parity',
            'description': 'Prediction reliability varies across groups. Consider:',
            'actions': [
                'Apply group-specific calibration using techniques like Platt scaling or isotonic regression',
                'Use temperature scaling with group-specific parameters',
                'Implement histogram binning calibration per group',
                'Verify sufficient sample sizes for all groups in training data'
            ]
        }
        recommendations.append(rec)
    
    # Disparate Impact recommendations
    if interpretations['disparate_impact']['has_issue']:
        rec = {
            'title': 'Mitigating Disparate Impact',
            'description': 'Some groups experience disparate impact. Address with:',
            'actions': [
                'Adjust decision thresholds to achieve the 80% rule (4/5 rule) compliance',
                'Use preprocessing techniques like disparate impact remover',
                'Implement fairness-aware feature selection',
                'Consider legal and ethical implications of the current disparities'
            ]
        }
        recommendations.append(rec)
    
    # General recommendations if any issues exist
    if any(interp['has_issue'] for interp in interpretations.values()):
        general_rec = {
            'title': 'General Fairness Best Practices',
            'description': 'Additional steps to improve overall model fairness:',
            'actions': [
                'Conduct regular fairness audits and monitoring in production',
                'Ensure diverse representation in training data collection',
                'Implement A/B testing with fairness metrics tracking',
                'Establish clear fairness goals and acceptable thresholds',
                'Document all fairness-related decisions and trade-offs',
                'Consider using multiple fairness metrics as no single metric captures all aspects'
            ]
        }
        recommendations.append(general_rec)
    
    return recommendations


def generate_fairness_report(y_true: np.ndarray, y_pred: np.ndarray,
                           sensitive_features: np.ndarray,
                           y_prob: Optional[np.ndarray] = None,
                           output_path: str = "fairness_report.html",
                           include_plots: bool = True,
                           feature_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive fairness report.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    sensitive_features : array-like
        Sensitive attribute values
    y_prob : array-like, optional
        Predicted probabilities
    output_path : str
        Path to save the HTML report
    include_plots : bool
        Whether to generate and include visualization plots
        
    Returns:
    --------
    dict : Report summary including metrics, interpretations, and recommendations
    """
    reporter = FairnessReporter()
    report_data = reporter.generate_report(y_true, y_pred, sensitive_features, y_prob, output_path, feature_name)
    
    if include_plots:
        from ..visualization import create_fairness_dashboard
        
        # Generate visualizations
        plot_path = output_path.replace('.html', '_dashboard.png')
        create_fairness_dashboard(
            y_true, y_pred, sensitive_features, y_prob,
            metrics=report_data['metrics'],
            save_path=plot_path
        )
        report_data['plot_path'] = plot_path
    
    return report_data