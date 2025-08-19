import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional
from sklearn.metrics import confusion_matrix
from dataclasses import dataclass


@dataclass
class FairnessMetrics:
    """Container for fairness metrics results."""
    demographic_parity: Dict[str, float]
    equalized_odds: Dict[str, Dict[str, float]]
    calibration_parity: Dict[str, Dict[str, float]]
    disparate_impact: Dict[str, float]
    statistical_parity_difference: Dict[str, float]


@dataclass
class MultipleFairnessMetrics:
    """Container for fairness metrics results across multiple sensitive features."""
    metrics_by_feature: Dict[str, FairnessMetrics]
    

def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: np.ndarray, y_prob: Optional[np.ndarray] = None) -> None:
    """Validate input arrays."""
    if len(y_true) != len(y_pred) or len(y_true) != len(sensitive_features) or (y_prob is not None and len(y_true) != len(y_prob)):
        raise ValueError("All input arrays must have the same length")
    
    if not np.array_equal(np.unique(y_true), [0, 1]) or not np.array_equal(np.unique(y_pred), [0, 1]):
        raise ValueError("y_true and y_pred must be binary (0 or 1)")

    if y_prob is not None and np.any(y_prob < 0) or np.any(y_prob > 1):
        raise ValueError("y_prob must be between 0 and 1")

def demographic_parity(y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: np.ndarray) -> Dict[str, float]:
    """
    Calculate demographic parity metric.
    
    Demographic parity requires that the prediction rates (disregard true label) are equal across all groups.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    sensitive_features : array-like
        Sensitive attribute values (e.g., race, gender, age group)
        
    Returns:
    --------
    dict : Dictionary mapping each group to its positive prediction rate
    """
    groups = np.unique(sensitive_features)
    results = {}
    
    for group in groups:
        mask = sensitive_features == group
        positive_rate = np.mean(y_pred[mask])
        results[str(group)] = positive_rate
    
    return results


def equal_opportunity(y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: np.ndarray) -> Dict[str, float]:
    """
    Calculate equal opportunity metric (True Positive Rate).
    
    Equal opportunity requires that the true positive rates are equal across all groups.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    sensitive_features : array-like
        Sensitive attribute values
        
    Returns:
    --------
    dict : Dictionary mapping each group to its true positive rate
    """
    
    groups = np.unique(sensitive_features)
    results = {}
    
    for group in groups:
        mask = sensitive_features == group
        group_true = y_true[mask]
        group_pred = y_pred[mask]
        
        # Calculate TPR for positive class
        positive_mask = group_true == 1
        if np.sum(positive_mask) > 0:
            tpr = np.mean(group_pred[positive_mask])
        else:
            tpr = np.nan
            
        results[str(group)] = tpr
    
    return results


def equalized_odds(y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate equalized odds metric (TPR and FPR).
    
    Equalized odds requires that TPR and FPR are equal across all groups (stricter verson of equal opportunity).
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    sensitive_features : array-like
        Sensitive attribute values
        
    Returns:
    --------
    dict : Dictionary mapping each group to its TPR and FPR
    """
    
    groups = np.unique(sensitive_features)
    results = {}
    
    for group in groups:
        mask = sensitive_features == group
        group_true = y_true[mask]
        group_pred = y_pred[mask]
        
        # Calculate TPR
        positive_mask = group_true == 1
        if np.sum(positive_mask) > 0:
            tpr = np.mean(group_pred[positive_mask])
        else:
            tpr = np.nan
            
        # Calculate FPR
        negative_mask = group_true == 0
        if np.sum(negative_mask) > 0:
            fpr = np.mean(group_pred[negative_mask])
        else:
            fpr = np.nan
            
        results[str(group)] = {'TPR': tpr, 'FPR': fpr}
    
    return results


def calibration_parity(y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate calibration parity metrics (PPV and NPV).
    
    Calibration parity requires that positive predictive value (PPV) and 
    negative predictive value (NPV) are equal across all groups.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    sensitive_features : array-like
        Sensitive attribute values
    y_prob : array-like, optional
        Predicted probabilities (for calibration curve if needed)
        
    Returns:
    --------
    dict : Dictionary mapping each group to its PPV and NPV
    """
    
    groups = np.unique(sensitive_features)
    results = {}
    
    for group in groups:
        mask = sensitive_features == group
        group_true = y_true[mask]
        group_pred = y_pred[mask]
        
        # Calculate PPV (Positive Predictive Value)
        positive_pred_mask = group_pred == 1
        if np.sum(positive_pred_mask) > 0:
            ppv = np.mean(group_true[positive_pred_mask])
        else:
            ppv = np.nan
            
        # Calculate NPV (Negative Predictive Value)
        negative_pred_mask = group_pred == 0
        if np.sum(negative_pred_mask) > 0:
            npv = 1 - np.mean(group_true[negative_pred_mask])
        else:
            npv = np.nan
            
        results[str(group)] = {'PPV': ppv, 'NPV': npv}
    
    return results


def disparate_impact(y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: np.ndarray,
                    privileged_group: Optional[str] = None) -> Dict[str, float]:
    """
    Calculate disparate impact ratio.
    
    Disparate impact is the ratio of positive prediction rates between groups.
    A value close to 1 indicates fairness.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    sensitive_features : array-like
        Sensitive attribute values
    privileged_group : str, optional
        The privileged group to use as reference
        
    Returns:
    --------
    dict : Dictionary mapping each group to its disparate impact ratio
    """
    demographic_parity_rates = demographic_parity(y_true, y_pred, sensitive_features)
    
    if privileged_group is None:
        # Use the group with highest positive rate as privileged
        privileged_group = max(demographic_parity_rates, key=demographic_parity_rates.get)
    
    privileged_rate = demographic_parity_rates[str(privileged_group)]
    results = {}
    
    for group, rate in demographic_parity_rates.items():
        if privileged_rate > 0:
            results[group] = rate / privileged_rate
        else:
            results[group] = np.nan
    
    return results


def statistical_parity_difference(y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: np.ndarray,
                                 privileged_group: Optional[str] = None) -> Dict[str, float]:
    """
    Calculate statistical parity difference.
    
    Statistical parity difference is the difference in positive prediction rates between groups.
    A value close to 0 indicates fairness.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    sensitive_features : array-like
        Sensitive attribute values
    privileged_group : str, optional
        The privileged group to use as reference
        
    Returns:
    --------
    dict : Dictionary mapping each group to its statistical parity difference
    """
    demographic_parity_rates = demographic_parity(y_true, y_pred, sensitive_features)
    
    if privileged_group is None:
        # Use the group with highest positive rate as privileged
        privileged_group = max(demographic_parity_rates, key=demographic_parity_rates.get)
    
    privileged_rate = demographic_parity_rates[str(privileged_group)]
    results = {}
    
    for group, rate in demographic_parity_rates.items():
        results[group] = rate - privileged_rate
    
    return results


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: np.ndarray,
                         y_prob: Optional[np.ndarray] = None,
                         privileged_group: Optional[str] = None) -> FairnessMetrics:
    """
    Calculate all fairness metrics.
    
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
    privileged_group : str, optional
        The privileged group to use as reference
        
    Returns:
    --------
    FairnessMetrics : Object containing all calculated fairness metrics
    """

    _validate_inputs(y_true, y_pred, sensitive_features, y_prob)
    y_pred = y_pred if y_prob is None else (y_prob >= .4).astype(int)

    return FairnessMetrics(
        demographic_parity=demographic_parity(y_true, y_pred, sensitive_features),
        equalized_odds=equalized_odds(y_true, y_pred, sensitive_features),
        calibration_parity=calibration_parity(y_true, y_pred, sensitive_features, y_prob),
        disparate_impact=disparate_impact(y_true, y_pred, sensitive_features, privileged_group),
        statistical_parity_difference=statistical_parity_difference(y_true, y_pred, sensitive_features, privileged_group)
    )


def calculate_metrics_for_multiple_features(y_true: np.ndarray, y_pred: np.ndarray, 
                                           sensitive_features_dict: Dict[str, np.ndarray],
                                           y_prob: Optional[np.ndarray] = None,
                                           privileged_groups: Optional[Dict[str, str]] = None) -> MultipleFairnessMetrics:
    """
    Calculate fairness metrics for multiple sensitive features.
    
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
    privileged_groups : dict, optional
        Dictionary mapping feature names to their privileged group values
        
    Returns:
    --------
    MultipleFairnessMetrics : Object containing metrics for all sensitive features
    """
    if privileged_groups is None:
        privileged_groups = {}
    
    metrics_by_feature = {}
    
    for feature_name, feature_values in sensitive_features_dict.items():
        privileged_group = privileged_groups.get(feature_name, None)
        metrics = calculate_all_metrics(y_true, y_pred, feature_values, y_prob, privileged_group)
        metrics_by_feature[feature_name] = metrics
    
    return MultipleFairnessMetrics(metrics_by_feature=metrics_by_feature)