# Fairness Toolkit

A comprehensive Python package for evaluating and improving fairness in binary classification models.

## Features

- **Multiple Fairness Metrics**: Calculate demographic parity, equal opportunity, equalized odds, calibration parity, disparate impact, and more
- **Visualization Tools**: Generate clear, informative plots to understand fairness issues
- **Automated Reporting**: Create detailed HTML reports with interpretations and actionable recommendations
- **Interactive Reports**: Analyze multiple sensitive features with interactive HTML reports
- **Easy Integration**: Simple API that works with numpy arrays and scikit-learn models

## Installation

```bash
pip install fairness-toolkit
```

Or install from source:

```bash
git clone https://github.com/cctsairush/fairness-toolkit.git
cd fairness-toolkit
pip install -e .
```

## Quick Start

```python
import numpy as np
from fairness_toolkit import calculate_all_metrics, create_fairness_dashboard, generate_fairness_report

# Example data
y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1])
y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
sensitive_features = np.array(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])  # e.g., gender, race

# Calculate all fairness metrics
metrics = calculate_all_metrics(y_true, y_pred, sensitive_features)

# Create visualizations
create_fairness_dashboard(y_true, y_pred, sensitive_features, save_path='fairness_dashboard.png')

# Generate comprehensive report
report = generate_fairness_report(
    y_true, y_pred, sensitive_features,
    output_path='fairness_report.html'
)
```

## Detailed Usage

### 1. Individual Metrics

```python
from fairness_toolkit import demographic_parity, equal_opportunity, calibration_parity

# Demographic Parity: Equal positive prediction rates
dp = demographic_parity(y_true, y_pred, sensitive_features)
print(f"Demographic Parity: {dp}")

# Equal Opportunity: Equal true positive rates
eo = equal_opportunity(y_true, y_pred, sensitive_features)
print(f"Equal Opportunity: {eo}")

# Calibration Parity: Equal PPV and NPV
cp = calibration_parity(y_true, y_pred, sensitive_features)
print(f"Calibration Parity: {cp}")
```

### 2. Visualization Options

```python
from fairness_toolkit import plot_fairness_metrics, plot_group_distributions

# Plot specific metrics
plot_fairness_metrics(
    metrics.demographic_parity, 
    metric_name="Demographic Parity",
    save_path='demographic_parity.png'
)

# Plot distributions
plot_group_distributions(
    y_true, y_pred, sensitive_features,
    y_prob=y_prob,  # Optional: include if you have probabilities
    save_path='distributions.png'
)
```

### 3. Working with Scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Train a model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Get predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate fairness
metrics = calculate_all_metrics(y_test, y_pred, sensitive_features_test, y_prob)
```

### 4. Interactive Reports with Multiple Sensitive Features

The toolkit now supports analyzing fairness across multiple sensitive features simultaneously with interactive HTML reports.

```python
from fairness_toolkit import generate_interactive_fairness_report

# Define multiple sensitive features
sensitive_features_dict = {
    'race': race_array,        # e.g., ['White', 'Black', 'Asian', ...]
    'sex': sex_array,          # e.g., ['Male', 'Female', ...]
    'age_group': age_array,    # e.g., ['<25', '25-40', '40-60', '>60']
    'ethnicity': ethnicity_array
}

# Generate interactive report
report = generate_interactive_fairness_report(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features_dict=sensitive_features_dict,
    y_prob=y_prob,  # Optional: include for probability-based metrics
    output_path='interactive_fairness_report.html',
    privileged_groups={  # Optional: specify privileged groups
        'race': 'White',
        'sex': 'Male',
        'age_group': '25-40'
    }
)

# The report includes:
# - Feature selector dropdown to switch between sensitive features
# - Per-feature analysis with metrics, visualizations, and recommendations
# - Interactive tabs for different aspects of the analysis
# - Comparative fairness scores across all features
```

#### Features of Interactive Reports:

1. **Feature Switching**: Dropdown menu to seamlessly switch between different sensitive features
2. **Comprehensive Analysis**: Each feature gets its own:
   - Model performance metrics by group
   - Fairness metrics (demographic parity, equal opportunity, etc.)
   - Visualizations (confusion matrices, distributions, dashboards)
   - Tailored recommendations
3. **Interactive Navigation**: Tabbed interface for easy navigation between:
   - Model Performance
   - Fairness Metrics
   - Visualizations
   - Recommendations
4. **No Page Reloads**: JavaScript-based switching for smooth user experience

#### Example with Multiple Features:

```python
import numpy as np
from fairness_toolkit import calculate_metrics_for_multiple_features, generate_interactive_fairness_report

# Calculate metrics for multiple features at once
multi_metrics = calculate_metrics_for_multiple_features(
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features_dict={
        'department': dept_array,
        'location': location_array,
        'tenure': tenure_array
    },
    y_prob=y_prob
)

# Access metrics for specific features
print("Demographic parity by department:")
for group, value in multi_metrics.metrics_by_feature['department'].demographic_parity.items():
    print(f"  {group}: {value:.3f}")

# Generate interactive report
report = generate_interactive_fairness_report(
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features_dict=sensitive_features_dict,
    y_prob=y_prob,
    output_path='fairness_analysis.html'
)

print(f"Report saved to: {report['report_path']}")
print("Fairness scores by feature:")
for feature, stats in report['feature_stats'].items():
    print(f"  {feature}: {stats['fairness_score']}%")
```

## Metrics Explained

### Demographic Parity
- **Definition**: P(Ŷ=1|A=a) = P(Ŷ=1|A=b) for all groups a, b
- **Interpretation**: All groups should receive positive predictions at the same rate
- **Use when**: You want equal treatment regardless of qualifications

### Equal Opportunity
- **Definition**: P(Ŷ=1|Y=1, A=a) = P(Ŷ=1|Y=1, A=b) for all groups a, b
- **Interpretation**: All groups should have equal true positive rates
- **Use when**: You want equal opportunity for qualified individuals

### Equalized Odds
- **Definition**: Equal TPR and FPR across groups
- **Interpretation**: Similar error rates for all groups
- **Use when**: You want consistent model performance across groups

### Calibration Parity
- **Definition**: P(Y=1|Ŷ=1, A=a) = P(Y=1|Ŷ=1, A=b) for all groups
- **Interpretation**: Predictions should be equally reliable across groups
- **Use when**: You need consistent prediction confidence

### Disparate Impact
- **Definition**: Ratio of positive rates between groups
- **Interpretation**: Values between 0.8-1.25 indicate fairness (4/5 rule)
- **Use when**: Legal compliance is important

## Report Interpretation

The generated HTML report includes:

1. **Executive Summary**: Overall fairness assessment
2. **Detailed Metrics**: Tables and interpretations for each metric
3. **Visualizations**: Charts showing disparities
4. **Recommendations**: Specific techniques to improve fairness
5. **Technical Details**: Metric definitions and calculations

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{fairness_toolkit,
  title = {Fairness Toolkit: A Python Package for Evaluating ML Fairness},
  author = {Jonathan Tsai},
  year = {2024},
  url = {https://github.com/cctsairush/fairness-toolkit}
}
```

## Support

- Issues: [GitHub Issues](https://github.com/cctsairush/fairness-toolkit/issues)
- Discussions: [GitHub Discussions](https://github.com/cctsairush/fairness-toolkit/discussions)