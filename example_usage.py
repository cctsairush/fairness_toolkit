#!/usr/bin/env python3
"""
Example usage of the Fairness Toolkit package.

This script demonstrates how to use the fairness toolkit to evaluate
and visualize fairness metrics for a binary classification model with
interactive HTML reports and embedded visualizations.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Import fairness toolkit components
from fairness_toolkit.metrics import calculate_all_metrics, calculate_metrics_for_multiple_features
from fairness_toolkit.reporting.interactive_fairness_report import generate_interactive_fairness_report
from fairness_toolkit.visualization.fairness_plots import (
    plot_group_distributions,
    create_fairness_dashboard,
    plot_heatmap
)


def create_synthetic_data(n_samples=1000, random_state=42):
    """Create synthetic data with multiple sensitive attributes."""
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
    
    # Create multiple sensitive attributes
    # Race (correlated with feature 0)
    race = np.where(
        X[:, 0] + np.random.normal(0, 0.5, n_samples) > 0.5,
        'White',
        np.where(
            X[:, 0] + np.random.normal(0, 0.5, n_samples) > -0.5,
            'Black',
            'Hispanic'
        )
    )
    
    # Sex (correlated with feature 1)
    sex = np.where(
        X[:, 1] + np.random.normal(0, 0.5, n_samples) > 0,
        'Male',
        'Female'
    )
    
    # Age group (based on feature 2)
    age_values = X[:, 2] + np.random.normal(0, 0.5, n_samples)
    age = np.where(
        age_values > 0.5,
        'Senior',
        np.where(
            age_values > -0.5,
            'Middle',
            'Young'
        )
    )
    
    return X, y, race, sex, age


def main():
    print("🚀 Fairness Toolkit Example Usage (Latest Version)")
    print("=" * 60)
    print("Demonstrating interactive reports with embedded visualizations")
    print("=" * 60)
    
    # 1. Create synthetic data
    print("\n📊 STEP 1: Creating synthetic dataset...")
    X, y, race, sex, age = create_synthetic_data(n_samples=2000)
    
    # Split data
    (X_train, X_test, y_train, y_test, 
     race_train, race_test, 
     sex_train, sex_test,
     age_train, age_test) = train_test_split(
        X, y, race, sex, age, 
        test_size=0.3, 
        random_state=42, 
        stratify=y
    )
    
    print(f"   • Training samples: {len(X_train)}")
    print(f"   • Test samples: {len(X_test)}")
    print(f"   • Race groups: {np.unique(race)}")
    print(f"   • Sex groups: {np.unique(sex)}")
    print(f"   • Age groups: {np.unique(age)}")
    
    # 2. Train a model
    print("\n🤖 STEP 2: Training Random Forest classifier...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = np.mean(y_test == y_pred)
    print(f"   • Overall accuracy: {accuracy:.3f}")
    
    # 3. Calculate fairness metrics for individual features
    print("\n📈 STEP 3: Calculating fairness metrics...")
    
    # Single feature analysis
    race_metrics = calculate_all_metrics(y_test, y_pred, race_test, y_prob)
    
    print("\n   Race - Demographic Parity:")
    for group, rate in race_metrics.demographic_parity.items():
        print(f"      • {group}: {rate:.3f}")
    
    print("\n   Race - Disparate Impact:")
    for group, ratio in race_metrics.disparate_impact.items():
        if not np.isnan(ratio):
            status = "✅ Fair" if 0.8 <= ratio <= 1.25 else "⚠️ Potential Bias"
            print(f"      • {group}: {ratio:.3f} ({status})")
    
    # 4. Generate individual visualizations (returns base64 embedded images)
    print("\n🎨 STEP 4: Creating embedded visualizations...")
    
    # Create group distributions plot (returns base64 string)
    distributions_plot = plot_group_distributions(
        y_test, y_pred, race_test, y_prob, 
        feature_name='Race'
    )
    print(f"   • Group distributions plot generated ({len(distributions_plot):,} chars)")
    
    # Create fairness dashboard (returns base64 string)
    dashboard_plot = create_fairness_dashboard(
        y_test, y_pred, race_test, y_prob,
        metrics=race_metrics,
        feature_name='Race'
    )
    print(f"   • Fairness dashboard generated ({len(dashboard_plot):,} chars)")
    
    # 5. Generate comprehensive interactive report
    print("\n📝 STEP 5: Generating interactive fairness report...")
    
    # Prepare sensitive features dictionary
    sensitive_features_dict = {
        'race': race_test,
        'sex': sex_test,
        'age': age_test
    }
    
    # Define privileged groups (optional)
    privileged_groups = {
        'race': 'White',
        'sex': 'Male',
        'age': 'Middle'
    }
    
    # Generate the interactive report
    report_data = generate_interactive_fairness_report(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features_dict=sensitive_features_dict,
        y_prob=y_prob,
        output_path='example_interactive_report.html',
        privileged_groups=privileged_groups
    )
    
    print("   ✅ Interactive HTML report generated!")
    print(f"   • Report saved to: example_interactive_report.html")
    print(f"   • Features analyzed: {len(report_data['feature_stats'])}")
    
    # Display fairness scores
    print("\n   📊 Fairness Scores by Feature:")
    for feature, stats in report_data['feature_stats'].items():
        print(f"      • {feature}: {stats['fairness_score']}% ({stats['num_groups']} groups)")
    
    # 6. Create a multi-feature heatmap
    print("\n🗺️ STEP 6: Creating fairness heatmap...")
    
    # Calculate metrics for all features
    multi_metrics = calculate_metrics_for_multiple_features(
        y_test, y_pred, sensitive_features_dict, y_prob, privileged_groups
    )
    
    # Prepare data for heatmap
    features = list(sensitive_features_dict.keys())
    metric_names = ['demographic parity', 'equalized odds', 'calibration parity', 'disparate impact']
    
    # Calculate fairness scores for heatmap
    fairness_scores = {}
    for feature in features:
        metrics = multi_metrics.metrics_by_feature[feature]
        # Simple scoring: closer to perfect fairness = higher score
        scores = []
        
        # Demographic parity score
        dp_values = list(metrics.demographic_parity.values())
        dp_score = 1 - (max(dp_values) - min(dp_values)) if len(dp_values) > 1 else 1.0
        scores.append(max(0, min(1, dp_score)))
        
        # Equalized odds score (average of TPR and FPR fairness)
        tpr_values = [v['TPR'] for v in metrics.equalized_odds.values() if not np.isnan(v['TPR'])]
        fpr_values = [v['FPR'] for v in metrics.equalized_odds.values() if not np.isnan(v['FPR'])]
        tpr_score = 1 - (max(tpr_values) - min(tpr_values)) if len(tpr_values) > 1 else 1.0
        fpr_score = 1 - (max(fpr_values) - min(fpr_values)) if len(fpr_values) > 1 else 1.0
        scores.append(max(0, min(1, (tpr_score + fpr_score) / 2)))
        
        # Calibration parity score
        ppv_values = [v['PPV'] for v in metrics.calibration_parity.values() if not np.isnan(v['PPV'])]
        calib_score = 1 - (max(ppv_values) - min(ppv_values)) if len(ppv_values) > 1 else 1.0
        scores.append(max(0, min(1, calib_score)))
        
        # Disparate impact score
        di_values = [v for v in metrics.disparate_impact.values() if not np.isnan(v)]
        di_score = min([min(v, 1/v) for v in di_values]) if di_values else 1.0
        scores.append(max(0, min(1, di_score)))
        
        fairness_scores[feature] = scores
    
    # Create heatmap (returns base64 string)
    heatmap_plot = plot_heatmap(
        features=features,
        metrics=metric_names,
        fairness_scores=fairness_scores
    )
    print(f"   • Heatmap generated ({len(heatmap_plot):,} chars)")
    
    # 7. Create a simple HTML showcase
    print("\n💾 STEP 7: Creating showcase HTML...")
    
    showcase_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Fairness Toolkit Showcase</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #5A6C7D; border-bottom: 3px solid #5A6C7D; padding-bottom: 10px; }}
        h2 {{ color: #6C757D; margin-top: 30px; }}
        .plot {{ margin: 20px 0; text-align: center; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #5A6C7D; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Fairness Toolkit - Embedded Visualizations Showcase</h1>
        
        <div class="metric">
            <strong>Dataset:</strong> {len(y_test)} samples<br>
            <strong>Accuracy:</strong> {accuracy:.3f}<br>
            <strong>Features Analyzed:</strong> Race, Sex, Age
        </div>
        
        <div class="plot">
            <h2>Group Distributions by Race</h2>
            <img src="{distributions_plot}" alt="Group Distributions">
        </div>
        
        <div class="plot">
            <h2>Fairness Dashboard for Race</h2>
            <img src="{dashboard_plot}" alt="Fairness Dashboard">
        </div>
        
        <div class="plot">
            <h2>Multi-Feature Fairness Heatmap</h2>
            <img src="{heatmap_plot}" alt="Fairness Heatmap">
        </div>
        
        <div class="metric">
            <h2>Key Features</h2>
            ✅ All visualizations embedded as base64 data<br>
            ✅ Single self-contained HTML file<br>
            ✅ Professional color schemes<br>
            ✅ Data labels on all charts<br>
            ✅ No external dependencies
        </div>
    </div>
</body>
</html>
    """
    
    with open("example_showcase.html", "w") as f:
        f.write(showcase_html)
    
    print("   • Showcase saved to: example_showcase.html")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📁 Generated Files:")
    print("   1. example_interactive_report.html - Full interactive report")
    print("   2. example_showcase.html - Showcase of embedded visualizations")
    print("\n🌟 Key Features Demonstrated:")
    print("   • Interactive reports with multiple sensitive features")
    print("   • Embedded visualizations (no external files)")
    print("   • Professional color schemes and data labels")
    print("   • Self-contained, shareable HTML files")
    print("\n💡 Open the HTML files in your browser to explore the results!")
    print("=" * 60)


def demonstrate_api_usage():
    """Demonstrate programmatic API usage."""
    print("\n\n📚 API Usage Examples")
    print("=" * 60)
    
    # Quick example data
    np.random.seed(42)
    n = 100
    y_true = np.random.binomial(1, 0.5, n)
    y_pred = np.random.binomial(1, 0.5, n)
    y_prob = np.random.uniform(0, 1, n)
    groups = np.random.choice(['A', 'B', 'C'], n)
    
    print("\n1️⃣ Single Feature Analysis:")
    print("```python")
    print("from fairness_toolkit.metrics import calculate_all_metrics")
    print("")
    print("metrics = calculate_all_metrics(y_true, y_pred, groups, y_prob)")
    print("print(metrics.demographic_parity)")
    print("print(metrics.disparate_impact)")
    print("```")
    
    metrics = calculate_all_metrics(y_true, y_pred, groups, y_prob)
    print(f"\nResult: {metrics.demographic_parity}")
    
    print("\n2️⃣ Multiple Features Analysis:")
    print("```python")
    print("from fairness_toolkit.reporting.interactive_fairness_report import (")
    print("    generate_interactive_fairness_report")
    print(")")
    print("")
    print("sensitive_features_dict = {")
    print("    'feature1': array1,")
    print("    'feature2': array2")
    print("}")
    print("")
    print("report = generate_interactive_fairness_report(")
    print("    y_true, y_pred, sensitive_features_dict, y_prob")
    print(")")
    print("```")
    
    print("\n3️⃣ Individual Visualizations (Base64):")
    print("```python")
    print("from fairness_toolkit.visualization.fairness_plots import (")
    print("    plot_group_distributions,")
    print("    create_fairness_dashboard")
    print(")")
    print("")
    print("# Returns base64 encoded image string")
    print("plot_base64 = plot_group_distributions(")
    print("    y_true, y_pred, groups, y_prob")
    print(")")
    print("")
    print("# Embed in HTML")
    print('html = f\'<img src="{plot_base64}">\'')
    print("```")


if __name__ == "__main__":
    main()
    demonstrate_api_usage()