"""
Fairness Toolkit - A Python package for evaluating and improving fairness in binary classification models.
"""

__version__ = "0.1.0"

from .metrics import (
    demographic_parity,
    equal_opportunity,
    equalized_odds,
    calibration_parity,
    calculate_all_metrics
)

from .visualization import (
    plot_fairness_metrics,
    plot_group_distributions,
    create_fairness_dashboard
)

from .reporting import (
    generate_fairness_report,
    generate_interactive_fairness_report,
    FairnessReporter
)

__all__ = [
    'demographic_parity',
    'equal_opportunity',
    'equalized_odds',
    'calibration_parity',
    'calculate_all_metrics',
    'plot_fairness_metrics',
    'plot_group_distributions',
    'create_fairness_dashboard',
    'generate_fairness_report',
    'FairnessReporter'
]