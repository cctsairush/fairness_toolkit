import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import io
import base64
from sklearn.metrics import confusion_matrix

# Set aesthetic style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Define modern color palettes
PALETTE_MAIN = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6C5B7B', '#355C7D']
PALETTE_PASTEL = ['#A8E6CF', '#DCEDC1', '#FFD3B6', '#FFAAA5', '#FF8B94', '#C8B6E2']
PALETTE_GRADIENT = ['#667BC6', '#7A86CC', '#8E9AD2', '#A2AED8', '#B6C2DE', '#CAD6E4']
PALETTE_CONTRAST = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']


def figure_to_base64(fig):
    """Convert a matplotlib figure to base64 string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def add_value_labels(ax, bars, format_str='{:.2f}', offset=0.01, fontsize=9):
    """Add value labels on top of bars."""
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height) and height != 0:
            label = format_str.format(height)
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                   label, ha='center', va='bottom', fontsize=fontsize, fontweight='bold')


def plot_fairness_metrics(metrics_dict: Dict[str, float], 
                         metric_name: str = "Fairness Metric",
                         figsize: Tuple[int, int] = (10, 6)) -> str:
    """Plot fairness metrics with enhanced visuals."""
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    groups = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    # Create bar plot with modern colors
    bars = plt.bar(groups, values, color=PALETTE_MAIN[0], alpha=0.85, edgecolor='white', linewidth=2)
    
    # Add value labels
    add_value_labels(plt.gca(), bars, format_str='{:.3f}', offset=0.01)
    
    plt.xlabel('Group', fontweight='bold')
    plt.ylabel('Metric Value', fontweight='bold')
    plt.title(metric_name, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.2, linestyle='--')
    
    # Add horizontal line at 1 for reference (perfect fairness for ratios)
    if 'disparate_impact' in metric_name.lower():
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Perfect Fairness')
        plt.legend()
    
    # Rotate x-axis labels if needed
    if len(groups) > 5:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    return figure_to_base64(fig)


def plot_heatmap(features: list, metrics: list,
                fairness_scores: Dict,
                figsize: Tuple[int, int] = (15, 5)) -> str:
    """Plot heatmap with enhanced colors - red at 0.8, green at 1.0."""
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Convert scores to numpy array
    scores = np.array([fairness_scores[feature] for feature in fairness_scores.keys()])
    
    # Create custom colormap: Green (1.0) -> Red (0.8) for professional contrast
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#C62828', '#FF9800', '#FFEB3B', '#4CAF50', '#1B5E20'] 
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('fairness', colors, N=n_bins)
    
    # Create heatmap with adjusted scale (0.8 to 1.0)
    ax = sns.heatmap(scores.T, vmin=0.8, vmax=1.0, annot=True, cmap=cmap, 
                     fmt='.2f', xticklabels=features, yticklabels=metrics,
                     cbar_kws={'label': 'Fairness Score', 'ticks': [0.8, 0.85, 0.9, 0.95, 1.0]}, 
                     linewidths=2, linecolor='white',
                     annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    
    # Set title and labels
    plt.title('Fairness Scores Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Metrics', fontweight='bold', fontsize=12)
    plt.xlabel('Features', fontweight='bold', fontsize=12)
    
    # Add note about scale
    plt.text(0.5, -0.15, 'Scale: 0.8 (Poor) to 1.0 (Perfect)', 
             ha='center', va='top', transform=ax.transAxes, 
             fontsize=9, style='italic', color='#666')
    
    plt.tight_layout()
    
    return figure_to_base64(fig)


def plot_group_distributions(y_true: np.ndarray, y_pred: np.ndarray,
                           sensitive_features: np.ndarray,
                           y_prob: Optional[np.ndarray] = None,
                           figsize: Tuple[int, int] = (16, 5),
                           feature_name: Optional[str] = None) -> str:
    """Plot group distributions with enhanced visuals and data labels."""
    groups = np.unique(sensitive_features)
    n_groups = len(groups)
    
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Create subplots
    if y_prob is not None:
        gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
    else:
        gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Actual vs Predicted by Group with data labels
    ax1 = fig.add_subplot(gs[0])
    
    data_actual = []
    data_pred = []
    group_labels = []
    
    for group in groups:
        mask = sensitive_features == group
        data_actual.append(np.mean(y_true[mask]))
        data_pred.append(np.mean(y_pred[mask]))
        group_labels.append(str(group))
    
    x = np.arange(len(group_labels))
    width = 0.35
    
    # Use modern colors
    bars1 = ax1.bar(x - width/2, data_actual, width, label='Actual', 
                    color=PALETTE_MAIN[0], alpha=0.85, edgecolor='white', linewidth=2)
    bars2 = ax1.bar(x + width/2, data_pred, width, label='Predicted', 
                    color=PALETTE_MAIN[1], alpha=0.85, edgecolor='white', linewidth=2)
    
    # Add value labels
    add_value_labels(ax1, bars1, format_str='{:.1%}', offset=0.01)
    add_value_labels(ax1, bars2, format_str='{:.1%}', offset=0.01)
    
    ax1.set_xlabel(f'{feature_name if feature_name else "Group"}', fontweight='bold')
    ax1.set_ylabel('Positive Rate', fontweight='bold')
    title = 'Actual vs Predicted Positive Rates by Group'
    if feature_name:
        title = f'Actual vs Predicted Positive Rates by {feature_name}'
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))
    ax1.set_title(title, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(group_labels)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_ylim([0, max(max(data_actual), max(data_pred)) * 1.15])
    
    # Plot 2: Classification Metrics by Group with data labels
    ax2 = fig.add_subplot(gs[1])
    
    metrics_data = {'TPR': [], 'FPR': [], 'PPV': [], 'NPV': []}
    
    for group in groups:
        mask = sensitive_features == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        if len(np.unique(y_true_group)) > 1 and len(np.unique(y_pred_group)) > 1:
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
        else:
            # Handle edge cases where confusion matrix might not be 2x2
            tn = fp = fn = tp = 0
            if len(y_true_group) > 0:
                if np.all(y_true_group == 0) and np.all(y_pred_group == 0):
                    tn = len(y_true_group)
                elif np.all(y_true_group == 1) and np.all(y_pred_group == 1):
                    tp = len(y_true_group)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        metrics_data['TPR'].append(tpr)
        metrics_data['FPR'].append(fpr)
        metrics_data['PPV'].append(ppv)
        metrics_data['NPV'].append(npv)
    
    # Create grouped bar chart with custom colors
    metrics_df = pd.DataFrame(metrics_data, index=group_labels)
    bar_width = 0.8 / len(metrics_data)
    x_pos = np.arange(len(group_labels))
    
    colors = PALETTE_CONTRAST[:4]
    for i, (metric, values) in enumerate(metrics_data.items()):
        offset = bar_width * (i - len(metrics_data)/2 + 0.5)
        bars = ax2.bar(x_pos + offset, values, bar_width, label=metric, 
                      color=colors[i], alpha=0.85, edgecolor='white', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if not np.isnan(val) and val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                        f'{val:.1%}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel(f'{feature_name if feature_name else "Group"}', fontweight='bold')
    ax2.set_ylabel('Rate', fontweight='bold')
    title = 'Classification Metrics by Group'
    if feature_name:
        title = f'Classification Metrics by {feature_name}'
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))
    ax2.set_title(title, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(group_labels, rotation=45 if len(group_labels) > 5 else 0)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.set_ylim([0, 1.15])
    
    # Plot 3: Probability Distributions (if available)
    if y_prob is not None:
        ax3 = fig.add_subplot(gs[2])
        
        # Use different colors for each group
        colors_pos = PALETTE_MAIN[:n_groups]
        colors_neg = PALETTE_PASTEL[:n_groups]
        
        for i, group in enumerate(groups):
            mask = sensitive_features == group
            
            # Plot positive class probabilities
            pos_mask = y_true[mask] == 1
            neg_mask = y_true[mask] == 0
            
            if np.sum(pos_mask) > 0:
                ax3.hist(y_prob[mask][pos_mask], bins=20, alpha=0.6, 
                        label=f'{group} (Positive)', density=True, 
                        color=colors_pos[i % len(colors_pos)], edgecolor='white', linewidth=1)
            if np.sum(neg_mask) > 0:
                ax3.hist(y_prob[mask][neg_mask], bins=20, alpha=0.4, 
                        label=f'{group} (Negative)', density=True,
                        color=colors_neg[i % len(colors_neg)], edgecolor='gray', linewidth=0.5)
        
        ax3.set_xlabel('Predicted Probability', fontweight='bold')
        ax3.set_ylabel('Density', fontweight='bold')
        title = 'Probability Distributions by Group and Class'
        if feature_name:
            title = f'Probability Distributions by {feature_name} and Class'
        ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))
        ax3.set_title(title, fontweight='bold', pad=15)
        ax3.legend(frameon=True, fancybox=True, shadow=True)
        ax3.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    return figure_to_base64(fig)


def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray,
                          sensitive_features: np.ndarray,
                          figsize: Tuple[int, int] = (15, 5),
                          feature_name: Optional[str] = None) -> str:
    """Plot confusion matrices with enhanced visuals."""
    groups = np.unique(sensitive_features)
    n_groups = len(groups)
    
    # Create subplots layout
    cols = min(n_groups, 4)
    rows = (n_groups + cols - 1) // cols
    
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    for idx, group in enumerate(groups):
        mask = sensitive_features == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        ax = fig.add_subplot(rows, cols, idx + 1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_group, y_pred_group)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar=False, square=True,
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['Actual 0', 'Actual 1'])
        
        ax.set_title(f'{group}\n(n={len(y_true_group)})', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide extra subplots
    for idx in range(n_groups, rows * cols):
        fig.add_subplot(rows, cols, idx + 1).set_visible(False)
    
    title = 'Confusion Matrices by Group'
    if feature_name:
        title = f'Confusion Matrices by {feature_name}'
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return figure_to_base64(fig)


def plot_calibration_curves(y_true: np.ndarray, y_prob: np.ndarray,
                          sensitive_features: np.ndarray,
                          n_bins: int = 10,
                          figsize: Tuple[int, int] = (12, 5),
                          feature_name: Optional[str] = None) -> str:
    """Plot calibration curves with enhanced visuals."""
    groups = np.unique(sensitive_features)
    
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    for i, group in enumerate(groups):
        mask = sensitive_features == group
        y_true_group = y_true[mask]
        y_prob_group = y_prob[mask]
        
        if len(y_true_group) == 0:
            continue
            
        # Calculate calibration curve
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob_group > bin_lower) & (y_prob_group <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true_group[in_bin].mean()
                avg_confidence_in_bin = y_prob_group[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Plot calibration curve
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration' if i == 0 else '')
        
        # Create bins and plot
        bin_centers = (bin_lowers + bin_uppers) / 2
        accuracies = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob_group > bin_lower) & (y_prob_group <= bin_upper)
            if in_bin.sum() > 0:
                accuracies.append(y_true_group[in_bin].mean())
            else:
                accuracies.append(np.nan)
        
        color = PALETTE_MAIN[i % len(PALETTE_MAIN)]
        plt.plot(bin_centers, accuracies, 'o-', color=color, 
                label=f'{group} (ECE: {ece:.3f})', linewidth=2, markersize=6)
    
    plt.xlabel('Mean Predicted Probability', fontweight='bold')
    plt.ylabel('Fraction of Positives', fontweight='bold')
    title = 'Calibration Curves by Group'
    if feature_name:
        title = f'Calibration Curves by {feature_name}'
    plt.title(title, fontweight='bold', pad=15)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    return figure_to_base64(fig)


def plot_analysis_curves(y_true: np.ndarray, y_prob: np.ndarray,
                       sensitive_features: np.ndarray,
                       figsize: Tuple[int, int] = (20, 5),
                       feature_name: Optional[str] = None) -> str:
    """Plot analysis curves with enhanced visuals."""
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
    
    groups = np.unique(sensitive_features)
    n_groups = len(groups)
    
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # ROC Curves
    ax1 = fig.add_subplot(1, 3, 1)
    for i, group in enumerate(groups):
        mask = sensitive_features == group
        y_true_group = y_true[mask]
        y_prob_group = y_prob[mask]
        
        if len(np.unique(y_true_group)) > 1:
            fpr, tpr, _ = roc_curve(y_true_group, y_prob_group)
            auc_score = roc_auc_score(y_true_group, y_prob_group)
            color = PALETTE_MAIN[i % len(PALETTE_MAIN)]
            ax1.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{group} (AUC: {auc_score:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontweight='bold')
    ax1.set_title('ROC Curves', fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    # Precision-Recall Curves
    ax2 = fig.add_subplot(1, 3, 2)
    for i, group in enumerate(groups):
        mask = sensitive_features == group
        y_true_group = y_true[mask]
        y_prob_group = y_prob[mask]
        
        if len(np.unique(y_true_group)) > 1:
            precision, recall, _ = precision_recall_curve(y_true_group, y_prob_group)
            color = PALETTE_MAIN[i % len(PALETTE_MAIN)]
            ax2.plot(recall, precision, color=color, linewidth=2, label=f'{group}')
    
    ax2.set_xlabel('Recall', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision-Recall Curves', fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.2, linestyle='--')
    
    # Score Distributions
    ax3 = fig.add_subplot(1, 3, 3)
    for i, group in enumerate(groups):
        mask = sensitive_features == group
        y_prob_group = y_prob[mask]
        color = PALETTE_MAIN[i % len(PALETTE_MAIN)]
        ax3.hist(y_prob_group, bins=20, alpha=0.6, label=f'{group}', 
                color=color, density=True, edgecolor='white', linewidth=1)
    
    ax3.set_xlabel('Predicted Probability', fontweight='bold')
    ax3.set_ylabel('Density', fontweight='bold')
    ax3.set_title('Score Distributions', fontweight='bold')
    ax3.legend(frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.2, linestyle='--')
    
    # Main title
    title = 'Analysis Curves by Group'
    if feature_name:
        title = f'Analysis Curves by {feature_name}'
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return figure_to_base64(fig)


def create_fairness_dashboard(y_true: np.ndarray, y_pred: np.ndarray,
                            sensitive_features: np.ndarray,
                            y_prob: Optional[np.ndarray] = None,
                            metrics: Optional[Dict] = None,
                            figsize: Tuple[int, int] = (20, 15),
                            feature_name: Optional[str] = None) -> str:
    """Create enhanced fairness dashboard with data labels and better colors."""
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.35)
    
    # If metrics not provided, calculate them
    if metrics is None:
        from ..metrics import calculate_all_metrics
        metrics = calculate_all_metrics(y_true, y_pred, sensitive_features, y_prob)
    
    # 1. Demographic Parity with data labels
    ax1 = fig.add_subplot(gs[0, 0])
    demo_values = list(metrics.demographic_parity.values())
    demo_labels = list(metrics.demographic_parity.keys())
    bars1 = ax1.bar(demo_labels, demo_values, color=PALETTE_GRADIENT[2], 
                    alpha=0.85, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars1, demo_values):
        ax1.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_title('Demographic Parity', fontweight='bold', fontsize=14, pad=15)
    ax1.set_ylabel('Positive Prediction Rate', fontweight='bold')
    ax1.set_xlabel(f'{feature_name if feature_name else "Group"}', fontweight='bold')
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_ylim([0, max(demo_values) * 1.15 if demo_values else 1])
    
    # 2. Disparate Impact with data labels
    ax2 = fig.add_subplot(gs[0, 1])
    di_values = list(metrics.disparate_impact.values())
    di_labels = list(metrics.disparate_impact.keys())
    
    # Color bars based on whether they meet 80% rule
    colors = [PALETTE_MAIN[2] if 0.8 <= v <= 1.25 else PALETTE_MAIN[3] for v in di_values]
    bars2 = ax2.bar(di_labels, di_values, color=colors, alpha=0.85, 
                    edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars2, di_values):
        ax2.text(bar.get_x() + bar.get_width()/2., val + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Perfect Parity')
    ax2.axhline(y=0.8, color='red', linestyle=':', alpha=0.5, linewidth=1.5, label='80% Rule')
    ax2.set_title('Disparate Impact', fontweight='bold', fontsize=14, pad=15)
    ax2.set_ylabel('Ratio', fontweight='bold')
    ax2.set_xlabel(f'{feature_name if feature_name else "Group"}', fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.2, linestyle='--')
    
    # 3. Equalized Odds with data labels
    ax3 = fig.add_subplot(gs[1, 0])
    eq_odds_data = pd.DataFrame(metrics.equalized_odds).T
    
    # Create grouped bars
    x_pos = np.arange(len(eq_odds_data.index))
    width = 0.35
    
    bars_tpr = ax3.bar(x_pos - width/2, eq_odds_data['TPR'], width, 
                       label='TPR', color=PALETTE_MAIN[0], alpha=0.85,
                       edgecolor='white', linewidth=2)
    bars_fpr = ax3.bar(x_pos + width/2, eq_odds_data['FPR'], width,
                       label='FPR', color=PALETTE_MAIN[1], alpha=0.85,
                       edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars_tpr, eq_odds_data['TPR']):
        if not np.isnan(val):
            ax3.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars_fpr, eq_odds_data['FPR']):
        if not np.isnan(val):
            ax3.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax3.set_title('Equalized Odds (TPR and FPR)', fontweight='bold', fontsize=14, pad=15)
    ax3.set_ylabel('Rate', fontweight='bold')
    ax3.set_xlabel(f'{feature_name if feature_name else "Group"}', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(eq_odds_data.index)
    ax3.legend(frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.2, linestyle='--')
    
    # 4. Calibration Parity with data labels
    ax4 = fig.add_subplot(gs[1, 1])
    calib_data = pd.DataFrame(metrics.calibration_parity).T
    
    # Create grouped bars
    x_pos = np.arange(len(calib_data.index))
    width = 0.35
    
    bars_ppv = ax4.bar(x_pos - width/2, calib_data['PPV'], width,
                       label='PPV', color=PALETTE_CONTRAST[1], alpha=0.85,
                       edgecolor='white', linewidth=2)
    bars_npv = ax4.bar(x_pos + width/2, calib_data['NPV'], width,
                       label='NPV', color=PALETTE_CONTRAST[2], alpha=0.85,
                       edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars_ppv, calib_data['PPV']):
        if not np.isnan(val):
            ax4.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars_npv, calib_data['NPV']):
        if not np.isnan(val):
            ax4.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax4.set_title('Calibration Parity', fontweight='bold', fontsize=14, pad=15)
    ax4.set_ylabel('Value', fontweight='bold')
    ax4.set_xlabel(f'{feature_name if feature_name else "Group"}', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(calib_data.index)
    ax4.legend(frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.2, linestyle='--')
    
    # 5. Enhanced Summary Statistics Table
    ax5 = fig.add_subplot(gs[2, :])
    
    # Create summary table
    summary_data = []
    groups = list(metrics.demographic_parity.keys())
    
    for group in groups:
        row = {
            'Group': group,
            'N': np.sum(sensitive_features == group),
            'Actual +Rate': f"{np.mean(y_true[sensitive_features == group]):.1%}",
            'Pred +Rate': f"{metrics.demographic_parity[group]:.1%}",
            'TPR': f"{metrics.equalized_odds[group]['TPR']:.1%}",
            'FPR': f"{metrics.equalized_odds[group]['FPR']:.1%}",
            'PPV': f"{metrics.calibration_parity[group]['PPV']:.1%}",
            'DI': f"{metrics.disparate_impact[group]:.2f}"
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create styled table
    table = ax5.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.1, 0.08, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    # Style the header with gradient color
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor(PALETTE_MAIN[0])
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors for better readability
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
            else:
                table[(i, j)].set_facecolor('white')
    
    ax5.axis('off')
    ax5.set_title('Summary Statistics by Group', fontweight='bold', fontsize=14, pad=20)
    
    # Main title
    title = 'Fairness Metrics Dashboard'
    if feature_name:
        title = f'Fairness Metrics Dashboard - {feature_name}'
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return figure_to_base64(fig)