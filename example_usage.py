#!/usr/bin/env python3
"""
Example usage of the Fairness Toolkit package.

This script demonstrates how to use the fairness toolkit to evaluate
and visualize fairness metrics for a binary classification model.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Import fairness toolkit
from fairness_toolkit import (
    calculate_all_metrics,
    create_fairness_dashboard,
    generate_fairness_report,
    plot_fairness_metrics,
    plot_group_distributions
)


def create_synthetic_data(n_samples=1000, random_state=42):
    """Create synthetic data with a sensitive attribute."""
    np.random.seed(random_state)
    
    # Create features and target
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=2,
        flip_y=0.1,
        random_state=random_state
    )
    
    # Create sensitive attribute (e.g., simulating two demographic groups)
    # Make it somewhat correlated with features to create potential bias
    sensitive_features = np.where(
        X[:, 0] + np.random.normal(0, 0.5, n_samples) > 0,
        'Group A',
        'Group B'
    )
    
    return X, y, sensitive_features


def main():
    print("Fairness Toolkit Example Usage")
    print("=" * 50)
    
    # 1. Create synthetic data
    print("\n1. Creating synthetic dataset...")
    X, y, sensitive_features = create_synthetic_data(n_samples=2000)
    
    # Split data
    X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
        X, y, sensitive_features, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    print(f"   - Groups: {np.unique(sensitive_features)}")
    
    # 2. Train a model
    print("\n2. Training Random Forest classifier...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = np.mean(y_test == y_pred)
    print(f"   - Overall accuracy: {accuracy:.3f}")
    
    # 3. Calculate fairness metrics
    print("\n3. Calculating fairness metrics...")
    metrics = calculate_all_metrics(y_test, y_pred, sf_test, y_prob)
    
    # Display key metrics
    print("\n   Demographic Parity (positive prediction rates):")
    for group, rate in metrics.demographic_parity.items():
        print(f"   - {group}: {rate:.3f}")
    
    print("\n   Equal Opportunity (true positive rates):")
    for group, rate in metrics.equal_opportunity.items():
        if not np.isnan(rate):
            print(f"   - {group}: {rate:.3f}")
    
    print("\n   Disparate Impact ratios:")
    for group, ratio in metrics.disparate_impact.items():
        if not np.isnan(ratio):
            status = "Fair" if 0.8 <= ratio <= 1.25 else "Potential Bias"
            print(f"   - {group}: {ratio:.3f} ({status})")
    
    # 4. Create visualizations
    print("\n4. Creating visualizations...")
    
    # Create fairness dashboard
    dashboard_fig = create_fairness_dashboard(
        y_test, y_pred, sf_test, y_prob,
        metrics=metrics,
        save_path='example_fairness_dashboard.png'
    )
    print("   - Saved dashboard to: example_fairness_dashboard.png")
    
    # Create individual plots
    plot_fairness_metrics(
        metrics.demographic_parity,
        metric_name="Demographic Parity",
        save_path='example_demographic_parity.png'
    )
    print("   - Saved demographic parity plot to: example_demographic_parity.png")
    
    plot_group_distributions(
        y_test, y_pred, sf_test, y_prob,
        save_path='example_distributions.png'
    )
    print("   - Saved distributions plot to: example_distributions.png")
    
    # 5. Generate comprehensive report
    print("\n5. Generating fairness report...")
    report_data = generate_fairness_report(
        y_test, y_pred, sf_test, y_prob,
        output_path='example_fairness_report.html',
        include_plots=True
    )
    
    print("   - Saved HTML report to: example_fairness_report.html")
    print(f"   - Overall fairness score: {report_data['fairness_score']}%")
    
    # 6. Display recommendations
    print("\n6. Fairness Improvement Recommendations:")
    for rec in report_data['recommendations']:
        print(f"\n   {rec['title']}:")
        print(f"   {rec['description']}")
        for action in rec['actions'][:3]:  # Show first 3 actions
            print(f"   • {action}")
    
    print("\n" + "=" * 50)
    print("Example completed! Check the generated files for detailed results.")
    

def demonstrate_different_scenarios():
    """Demonstrate the toolkit with different bias scenarios."""
    print("\n\nDemonstrating Different Bias Scenarios")
    print("=" * 50)
    
    # Scenario 1: High bias
    print("\n[Scenario 1] Model with high bias:")
    np.random.seed(123)
    n = 500
    y_true = np.random.binomial(1, 0.5, n)
    sensitive = np.random.choice(['A', 'B'], n)
    
    # Create biased predictions
    y_pred = y_true.copy()
    # Flip more predictions for group B
    group_b_mask = sensitive == 'B'
    flip_mask = np.random.random(n) < 0.3
    y_pred[group_b_mask & flip_mask] = 1 - y_pred[group_b_mask & flip_mask]
    
    metrics = calculate_all_metrics(y_true, y_pred, sensitive)
    print(f"  Demographic Parity - Group A: {metrics.demographic_parity['A']:.3f}")
    print(f"  Demographic Parity - Group B: {metrics.demographic_parity['B']:.3f}")
    print(f"  Disparate Impact - Group B: {metrics.disparate_impact['B']:.3f}")
    
    # Scenario 2: Fair model
    print("\n[Scenario 2] Fair model:")
    np.random.seed(456)
    y_true = np.random.binomial(1, 0.5, n)
    sensitive = np.random.choice(['A', 'B'], n)
    
    # Create fair predictions with similar error rates
    y_pred = y_true.copy()
    flip_mask = np.random.random(n) < 0.1  # 10% error rate for all
    y_pred[flip_mask] = 1 - y_pred[flip_mask]
    
    metrics = calculate_all_metrics(y_true, y_pred, sensitive)
    print(f"  Demographic Parity - Group A: {metrics.demographic_parity['A']:.3f}")
    print(f"  Demographic Parity - Group B: {metrics.demographic_parity['B']:.3f}")
    print(f"  Disparate Impact - Group B: {metrics.disparate_impact['B']:.3f}")


if __name__ == "__main__":
    main()
    demonstrate_different_scenarios()