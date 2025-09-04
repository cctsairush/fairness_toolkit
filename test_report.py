import numpy as np
import sys
sys.path.insert(0, '..')
from fairness_toolkit.reporting.interactive_fairness_report import generate_interactive_fairness_report

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_samples = 1000

# Generate true labels (30% positive cases)
y_true = np.random.binomial(1, 0.3, n_samples)

# Generate predicted probabilities with some bias
base_prob = 0.3
y_prob = np.random.beta(2, 5, n_samples)

# Create predictions from probabilities
y_pred = (y_prob > 0.5).astype(int)

# Generate sensitive features
# Race: 60% White, 25% Black, 15% Other
race = np.random.choice(['White', 'Black', 'Other'], n_samples, p=[0.6, 0.25, 0.15])

# Gender: 52% Female, 48% Male
gender = np.random.choice(['Female', 'Male'], n_samples, p=[0.52, 0.48])

# Age groups: 40% Young, 35% Middle, 25% Senior
age_group = np.random.choice(['Young', 'Middle', 'Senior'], n_samples, p=[0.4, 0.35, 0.25])

# Introduce some bias in predictions
# Make model slightly less accurate for certain groups
for i in range(n_samples):
    if race[i] == 'Black':
        # Add some bias - slightly lower accuracy for Black individuals
        if np.random.random() < 0.1:
            y_pred[i] = 1 - y_pred[i]  # Flip some predictions
    
    if gender[i] == 'Female' and age_group[i] == 'Senior':
        # Add intersectional bias
        if np.random.random() < 0.08:
            y_pred[i] = 1 - y_pred[i]

# Create sensitive features dictionary
sensitive_features_dict = {
    'race': race,
    'gender': gender,
    'age_group': age_group
}

# Specify privileged groups
privileged_groups = {
    'race': 'White',
    'gender': 'Male',
    'age_group': 'Young'
}

print("Generating updated interactive fairness report...")
print(f"Sample size: {n_samples}")
print(f"Positive cases: {np.sum(y_true)} ({np.mean(y_true):.1%})")
print(f"Sensitive features: {list(sensitive_features_dict.keys())}")

# Generate the report
report_data = generate_interactive_fairness_report(
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features_dict=sensitive_features_dict,
    y_prob=y_prob,
    output_path="sample_updated_fairness_report.html",
    privileged_groups=privileged_groups
)

print(f"\nReport generated successfully!")
print(f"Report saved to: {report_data['report_path']}")
print(f"Features analyzed: {len(report_data['feature_stats'])}")

# Print some summary statistics
print("\nOverall Performance Summary:")
print(f"- Total samples: {n_samples}")
print(f"- Case count (y=1): {np.sum(y_true)}")
print(f"- Overall accuracy: {np.mean(y_pred == y_true):.3f}")

print("\nFeature Analysis Summary:")
for feature_name, stats in report_data['feature_stats'].items():
    print(f"- {feature_name}: {stats['num_groups']} groups, fairness score: {stats['fairness_score']}%")