# Fairness Toolkit

A comprehensive Python toolkit for evaluating machine learning model fairness across different demographic groups with interactive visualizations.

## Features

- **Interactive HTML Reports**: Generate comprehensive, self-contained HTML reports with embedded visualizations
- **Multiple Fairness Metrics**: Evaluate demographic parity, equalized odds, calibration parity, and disparate impact
- **Easy Sharing**: Single HTML file can be shared via email or messaging

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from fairness_toolkit.reporting.interactive_fairness_report import generate_interactive_fairness_report

# Your model predictions
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])
y_prob = np.array([0.1, 0.8, 0.3, 0.2, 0.9, 0.6, 0.7, 0.4])

# Sensitive features
sensitive_features = {
    'race': np.array(['White', 'Black', 'White', 'Black', 'White', 'Black', 'White', 'Black']),
    'sex': np.array(['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'])
}

# Generate interactive report
report = generate_interactive_fairness_report(
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features_dict=sensitive_features,
    y_prob=y_prob,
    output_path="fairness_report.html"
)

print(f"Report saved to: {report['output_path']}")
```

## Advanced Usage

### Custom Privileged Groups

```python
# Define which groups to use as baseline for comparisons
privileged_groups = {
    'race': 'White',
    'sex': 'Male',
    'age': 'Middle'
}

report = generate_interactive_fairness_report(
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features_dict=sensitive_features,
    y_prob=y_prob,
    privileged_groups=privileged_groups,
    output_path="custom_fairness_report.html"
)
```

### Individual Metrics

```python
from fairness_toolkit.metrics import calculate_all_metrics

# Calculate metrics for a single feature
metrics = calculate_all_metrics(y_true, y_pred, race_groups, y_prob)

print(f"Demographic parity: {metrics.demographic_parity}")
print(f"Equalized odds: {metrics.equalized_odds}")
print(f"Disparate impact: {metrics.disparate_impact}")
```

### Custom Visualizations

```python
from fairness_toolkit.visualization.fairness_plots import (
    plot_group_distributions,
    create_fairness_dashboard,
    plot_heatmap
)

# Create individual plots (returns base64 encoded images)
distributions_plot = plot_group_distributions(
    y_true, y_pred, race_groups, y_prob, feature_name='Race'
)

dashboard_plot = create_fairness_dashboard(
    y_true, y_pred, race_groups, y_prob, feature_name='Race'
)
```

## Report Features

The interactive HTML reports include:

### Overall Performance
- Model accuracy, precision, recall, F1-score, and AUC
- Prominently displayed for immediate assessment

### Feature Analysis
- Dropdown selector for different sensitive features
- Group-by-group performance metrics
- Visual comparisons across demographic groups

### Fairness Metrics
- **Demographic Parity**: Equal positive prediction rates across groups
- **Equalized Odds**: Equal true positive and false positive rates
- **Calibration Parity**: Equal predictive values across groups  
- **Disparate Impact**: 80% rule compliance testing

### Interactive Visualizations
- Group distribution comparisons
- Classification metrics by group
- Fairness dashboard with data labels
- Embedded charts (no external files)

### Technical Analysis
- Heatmap showing fairness scores across features and metrics
- Color-coded from red (0.8 - poor) to green (1.0 - perfect)
- Professional gradient colors for better contrast

### Recommendations
- Actionable improvement suggestions
- Metric-specific recommendations
- Technical and policy-based solutions


## File Structure

```
fairness_toolkit/
├── metrics/
│   ├── __init__.py
│   └── fairness_metrics.py       # Core fairness calculations
├── reporting/
│   ├── __init__.py
│   ├── fairness_report.py        # Basic reporting functions
│   ├── interactive_fairness_report.py  # Interactive HTML reports
│   └── interactive_template.html # HTML template
├── visualization/
│   ├── __init__.py
│   └── fairness_plots.py         # All visualization functions
├── utils/
│   └── __init__.py
├── README.md
├── setup.py
└── requirements.txt
```

## Requirements

- Python 3.7+
- NumPy
- Pandas  
- Scikit-learn
- Matplotlib
- Seaborn
- Jinja2

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request


## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{fairness_toolkit,
  title={Fairness Toolkit: Interactive ML Fairness Analysis},
  author={Jonathan Tsai},
  year={2025},
  url={https://github.com/cctsairush/fairness_toolkit}
}
```