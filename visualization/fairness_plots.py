from matplotlib import ticker
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from sklearn.calibration import LabelEncoder, calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.gridspec as gridspec


def plot_fairness_metrics(metrics_dict: Dict[str, Dict[str, float]], 
                         metric_name: str = "Fairness Metrics",
                         figsize: Tuple[int, int] = (12, 6),
                         save_path: Optional[str] = None,
                         feature_name: Optional[str] = None) -> plt.Figure:
    """
    Plot fairness metrics as grouped bar charts.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of metrics with groups as keys
    metric_name : str
        Title for the plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(metrics_dict).T
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    df.plot(kind='bar', ax=ax, width=0.8)
    
    title = f'{metric_name} by Group'
    if feature_name:
        title = f'{metric_name} by {feature_name}'
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(f'{feature_name if feature_name else "Group"}', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at 1 for reference (perfect fairness for ratios)
    if 'disparate_impact' in metric_name.lower():
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Perfect Fairness')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45 if len(metrics_dict) > 5 else 0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_heatmap(features: list[str], metrics: list[str], 
                 fairness_scores: Dict[str, list[float]],
                 figsize: Tuple[int, int] = (15, 5),
                 save_path: Optional[str] = None)  -> plt.Figure:
    """
    Plot heatmap of fairness metrics.
    
    Parameters:
    -----------
    features : list
        List of feature names
    metrics : list
        List of fairness metrics
    fairness_scores : dict
        Dictionary of fairness scores with groups as keys
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    # Convert scores to numpy array
    scores = np.array([fairness_scores[feature] for feature in fairness_scores.keys()])

    # Create heatmap
    fig = plt.figure(figsize=figsize)
    sns.heatmap(scores.T, vmin=0, vmax=1, annot=True, cmap='RdYlGn', fmt='.2f', xticklabels=features, yticklabels=metrics)
    
    # Set title and labels
    plt.title('Fairness Scores Heatmap')
    plt.ylabel('Metrics')
    plt.xlabel('Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_group_distributions(y_true: np.ndarray, y_pred: np.ndarray, 
                           sensitive_features: np.ndarray,
                           y_prob: Optional[np.ndarray] = None,
                           figsize: Tuple[int, int] = (15, 5),
                           save_path: Optional[str] = None,
                           feature_name: Optional[str] = None) -> plt.Figure:
    """
    Plot distributions of predictions and outcomes by group.
    
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
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    groups = np.unique(sensitive_features)
    n_groups = len(groups)
    
    fig = plt.figure(figsize=figsize)
    
    # Create subplots
    if y_prob is not None:
        gs = gridspec.GridSpec(1, 3, figure=fig)
    else:
        gs = gridspec.GridSpec(1, 2, figure=fig)
    
    # Plot 1: Actual vs Predicted by Group
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
    
    ax1.bar(x - width/2, data_actual, width, label='Actual', alpha=0.8)
    ax1.bar(x + width/2, data_pred, width, label='Predicted', alpha=0.8)
    
    ax1.set_xlabel(f'{feature_name if feature_name else "Group"}')
    ax1.set_ylabel('Positive Rate')
    title = 'Actual vs Predicted Positive Rates by Group'
    if feature_name:
        title = f'Actual vs Predicted Positive Rates by {feature_name}'
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))  # Format x-axis tick labels as percentages    
    ax1.set_title(title)
    ax1.set_xticks(x)
    ax1.set_xticklabels(group_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Confusion Matrix Metrics by Group
    ax2 = fig.add_subplot(gs[1])
    
    metrics_data = {'TPR': [], 'FPR': [], 'PPV': [], 'NPV': []}
    
    for group in groups:
        mask = sensitive_features == group
        tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask]).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        metrics_data['TPR'].append(tpr)
        metrics_data['FPR'].append(fpr)
        metrics_data['PPV'].append(ppv)
        metrics_data['NPV'].append(npv)
    
    metrics_df = pd.DataFrame(metrics_data, index=group_labels)
    metrics_df.plot(kind='bar', ax=ax2, width=0.8)
    
    ax2.set_xlabel(f'{feature_name if feature_name else "Group"}')
    ax2.set_ylabel('Rate')
    title = 'Classification Metrics by Group'
    if feature_name:
        title = f'Classification Metrics by {feature_name}'
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))  # Format x-axis tick labels as percentages    
    ax2.set_title(title)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticklabels(group_labels, rotation=45 if len(group_labels) > 5 else 0)
    
    # Plot 3: Probability Distributions (if available)
    if y_prob is not None:
        ax3 = fig.add_subplot(gs[2])
        
        for group in groups:
            mask = sensitive_features == group
            
            # Plot positive class probabilities
            pos_mask = y_true[mask] == 1
            neg_mask = y_true[mask] == 0
            
            if np.sum(pos_mask) > 0:
                ax3.hist(y_prob[mask][pos_mask], bins=20, alpha=0.5, 
                        label=f'{group} (Positive)', density=True)
            if np.sum(neg_mask) > 0:
                ax3.hist(y_prob[mask][neg_mask], bins=20, alpha=0.5, 
                        label=f'{group} (Negative)', density=True)
        
        ax3.set_xlabel('Predicted Probability')
        ax3.set_ylabel('Density')
        title = 'Probability Distributions by Group and Class'
        if feature_name:
            title = f'Probability Distributions by {feature_name} and Class'
        ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))  # Format x-axis tick labels as percentages    
        ax3.set_title(title)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_calibration_curves(y_true: np.ndarray, y_prob: np.ndarray,
                          sensitive_features: np.ndarray,
                          n_bins: int = 10,
                          figsize: Tuple[int, int] = (12, 6),
                          save_path: Optional[str] = None,
                          feature_name: Optional[str] = None) -> plt.Figure:
    """
    Plot calibration curves for each group.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    sensitive_features : array-like
        Sensitive attribute values
    n_bins : int
        Number of bins for calibration curve
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    groups = np.unique(sensitive_features)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot calibration curves
    for group in groups:
        mask = sensitive_features == group
        
        if np.sum(mask) > n_bins:  # Need enough samples
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true[mask], y_prob[mask], n_bins=n_bins
            )
            
            ax1.plot(mean_predicted_value, fraction_of_positives, 
                    marker='o', label=f'Group {group}', linewidth=2)
    
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k:', label='Perfect Calibration')
    
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    title = 'Calibration Curves by Group'
    if feature_name:
        title = f'Calibration Curves by {feature_name}'
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))  # Format x-axis tick labels as percentages    
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Plot calibration error bars
    calibration_errors = []
    group_names = []
    
    for group in groups:
        mask = sensitive_features == group
        
        if np.sum(mask) > n_bins:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true[mask], y_prob[mask], n_bins=n_bins
            )
            
            # Calculate expected calibration error (ECE)
            bin_counts, _ = np.histogram(y_prob[mask], bins=n_bins, range=(0, 1))
            ece = np.sum(np.abs(fraction_of_positives - mean_predicted_value) * 
                        bin_counts) / np.sum(bin_counts)
            
            calibration_errors.append(ece)
            group_names.append(str(group))
    
    ax2.bar(group_names, calibration_errors, alpha=0.7)
    ax2.set_xlabel(f'{feature_name if feature_name else "Group"}')
    ax2.set_ylabel('Expected Calibration Error')
    title = 'Calibration Error by Group'
    if feature_name:
        title = f'Calibration Error by {feature_name}'
    ax2.set_title(title)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_analysis_curves(y_true: np.ndarray, y_prob: np.ndarray,
                     sensitive_features: np.ndarray,
                     n_bins: int = 10,
                     figsize: Tuple[int, int] = (12, 6),
                     save_path: Optional[str] = None,
                     feature_name: Optional[str] = None) -> plt.Figure:
    """    Plot ROC AUC scores for each group.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    sensitive_features : array-like
        Sensitive attribute values
    n_bins : int
        Number of bins for ROC curve
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    groups = np.unique(sensitive_features)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot roc curves
    for group in groups:
        mask = sensitive_features == group
        
        fpr, tpr, thresholds = roc_curve(
            y_true[mask], y_prob[mask]
        )
            
        ax1.plot(fpr, tpr, label=f'Group {group}', linewidth=2)
    

    
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    title = 'ROC Curves by Group'
    if feature_name:
        title = f'ROC Curves by {feature_name}'
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))  # Format x-axis tick labels as percentages    
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))  # Format x-axis tick labels as percentages    
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])


    # Plot net benefit
    thresholds = np.linspace(0, .5, n_bins)
    for group in groups:
        mask = sensitive_features == group
        net_benefits = np.zeros(n_bins)
        for i, threshold in enumerate(thresholds):
            y_pred_group = (y_prob[mask] >= threshold).astype(int)
            tp = np.sum(y_true[mask] * y_pred_group)
            fp = np.sum((1 - y_true[mask]) * y_pred_group)
            n = len(y_true[mask])

            net_benefit = (tp - fp * (threshold / (1 - threshold))) / n
            net_benefits[i] = net_benefit
        ax2.plot(thresholds, net_benefits, label=f'Group {group}', linewidth=2)

    # Add the "Treat All" and "Treat None" lines
    prevalence = np.mean(y_true)
    net_benefit = prevalence - (1 - prevalence) * thresholds / (1 -thresholds)
    ax2.plot(thresholds, net_benefit, label='Treat All')
    ax2.axhline(y=0, linestyle=':', label='Treat None')


    ax2.set_xlim([0, .35])  # Set x-axis limits
    ax2.set_ylim([-0.1, .25])  # Set y-axis limits
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))  # Format x-axis tick labels as percentages
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Net Benefit')
    title = 'DCA Curves by Group'
    if feature_name:
        title = f'DCA Curves by {feature_name}'
    ax2.set_title(title)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
    
def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray,
                          sensitive_features: np.ndarray,
                          figsize: Optional[Tuple[int, int]] = None,
                          save_path: Optional[str] = None,
                          feature_name: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrices for each group.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    sensitive_features : array-like
        Sensitive attribute values
    figsize : tuple, optional
        Figure size (auto-calculated if None)
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    groups = np.unique(sensitive_features)
    n_groups = len(groups)
    
    # Calculate figure size if not provided
    if figsize is None:
        cols = min(3, n_groups)
        rows = (n_groups + cols - 1) // cols
        figsize = (5 * cols, 5 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if n_groups == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, group in enumerate(groups):
        mask = sensitive_features == group
        cm = confusion_matrix(y_true[mask], y_pred[mask])
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                   ax=axes[idx], cbar=True,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        
        axes[idx].set_title(f'Group {group}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    # Hide extra subplots
    for idx in range(n_groups, len(axes)):
        axes[idx].set_visible(False)
    
    title = 'Confusion Matrices by Group'
    if feature_name:
        title = f'Confusion Matrices by {feature_name}'
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_fairness_dashboard(y_true: np.ndarray, y_pred: np.ndarray,
                            sensitive_features: np.ndarray,
                            y_prob: Optional[np.ndarray] = None,
                            metrics: Optional[Dict] = None,
                            figsize: Tuple[int, int] = (20, 15),
                            save_path: Optional[str] = None,
                            feature_name: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive fairness dashboard with multiple visualizations.
    
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
    metrics : dict, optional
        Pre-calculated fairness metrics
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # If metrics not provided, calculate them
    if metrics is None:
        from ..metrics import calculate_all_metrics
        metrics = calculate_all_metrics(y_true, y_pred, sensitive_features, y_prob)
    
    # 1. Demographic Parity
    ax1 = fig.add_subplot(gs[0, 0])
    df_demo = pd.DataFrame({'Demographic Parity': metrics.demographic_parity})
    df_demo.plot(kind='bar', ax=ax1, legend=False, color='skyblue')
    ax1.set_title('Demographic Parity', fontweight='bold')
    ax1.set_ylabel('Positive Prediction Rate')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Disparate Impact
    ax2 = fig.add_subplot(gs[0, 1])
    df_di = pd.DataFrame({'Disparate Impact': metrics.disparate_impact})
    df_di.plot(kind='bar', ax=ax2, legend=False, color='lightgreen')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('Disparate Impact', fontweight='bold')
    ax2.set_ylabel('Ratio')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Equalized Odds
    ax3 = fig.add_subplot(gs[1, 0])
    eq_odds_data = pd.DataFrame(metrics.equalized_odds).T
    eq_odds_data.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_title('Equalized Odds (TPR and FPR)', fontweight='bold')
    ax3.set_ylabel('Rate')
    ax3.legend(title='Metrics')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Calibration Parity
    ax4 = fig.add_subplot(gs[1, 1])
    calib_data = pd.DataFrame(metrics.calibration_parity).T
    calib_data.plot(kind='bar', ax=ax4, width=0.8)
    ax4.set_title('Calibration Parity', fontweight='bold')
    ax4.set_ylabel('Value')
    ax4.legend(title='Metrics')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Summary Statistics
    ax5 = fig.add_subplot(gs[2, :])
    
    # Create summary table
    summary_data = []
    groups = list(metrics.demographic_parity.keys())
    
    for group in groups:
        row = {
            'Group': group,
            'Sample Size': np.sum(sensitive_features == group),
            'Actual Positive Rate': np.mean(y_true[sensitive_features == group]),
            'Predicted Positive Rate': metrics.demographic_parity[group],
            'TPR': metrics.equalized_odds[group]['TPR'],
            'FPR': metrics.equalized_odds[group]['FPR'],
            'PPV': metrics.calibration_parity[group]['PPV'],
            'Disparate Impact': metrics.disparate_impact[group]
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create table
    table = ax5.table(cellText=summary_df.round(3).values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.axis('off')
    ax5.set_title('Summary Statistics by Group', fontweight='bold', pad=20)
    
    plt.suptitle('Fairness Metrics Dashboard', fontsize=20, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig