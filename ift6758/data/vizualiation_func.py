import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.calibration import CalibrationDisplay

import matplotlib.pyplot as plt

def get_plot_style(index):
    """
    Cycles through color/linestyle/linewidth options for better plotting.
    
    Args:
        - index (int): The index of a given model for plotting.
    """
    colors = sns.color_palette("tab10", 10)
    linestyles = ['-', '--', '-.', ':']
    linewidths = [2.5, 2, 1.5, 2.5]
    return colors[index % len(colors)], linestyles[index % len(linestyles)], linewidths[index % len(linewidths)]


def plot_multi_roc_curve(y_val_list, y_proba_list, model_names, save_file="image.png"):
    """
    Plots the ROC curve for multiple models, including a Random Baseline.

    Args:
        - y_true (array-like): True binary labels.
        - y_proba_list (list of array-like): List of predicted probabilities.
        - model_names (list of str): List of names for the models.
    """

    y_proba_random = np.random.uniform(0, 1, size=len(np.ravel(y_val_list[0])))
    fpr_rand, tpr_rand, _ = roc_curve(np.ravel(y_val_list[0]), y_proba_random)
    roc_auc_rand = auc(fpr_rand, tpr_rand)
    
    plt.plot(fpr_rand, tpr_rand, color='gray', linestyle='--', linewidth=1.5,
             label=f'Random Baseline (AUC = {roc_auc_rand:.2f})', alpha=0.6)

    for i, (y_val, y_proba, model_name) in enumerate(zip(y_val_list, y_proba_list, model_names)):
        y_val = np.ravel(y_val)
        color, linestyle, lw = get_plot_style(i)
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})',
                 color=color, linestyle=linestyle, linewidth=lw, alpha=0.9)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks(np.arange(0, 1.01, 0.1))
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    
def plot_multi_goal_rate_by_percentile(y_val_list, y_proba_list, model_names, n_bins=10, save_file="image.png"):
    """
    Plots the Observed Goal Rate vs. Predicted Probability Percentile for multiple models.

    Args:
        - y_val_list (list of array-like): List of true binary labels for each model.
        - y_proba_list (list of array-like): List of predicted probability arrays.
        - model_names (list of str): List of names for the models.
        - n_bins (int): Number of quantiles (deciles).
    """
    plt.figure(figsize=(8, 6))
    plt.plot([0, 100], [0, 100], color='gray', linestyle='--', linewidth=1.5,
             label='Random Baseline', alpha=0.6)

    for i, (y_val, y_proba, model_name) in enumerate(zip(y_val_list, y_proba_list, model_names)):
        y_val = np.ravel(np.array(y_val))
        y_proba = np.ravel(np.array(y_proba))
        color, linestyle, lw = get_plot_style(i)

        # Make DataFrame
        df = pd.DataFrame({'predicted_prob': y_proba, 'is_goal': y_val})

        # Skip if not enough variation to create bins
        if df['predicted_prob'].nunique() < 2:
            print(f"Skipping {model_name}: not enough probability variation to bin.")
            continue

        df['percentile_bin'] = pd.qcut(
            df['predicted_prob'],
            q=n_bins,
            labels=False,
            duplicates='drop'
        )

        grouped = (
            df.groupby('percentile_bin', observed=True)['is_goal']
            .agg(goal_rate='mean')
            .reset_index()
        )

        actual_bins = len(grouped)
        if actual_bins == 0:
            print(f"Skipping {model_name}: no valid bins after qcut.")
            continue

        grouped['percentile'] = (grouped['percentile_bin'] + 0.5) * (100 / actual_bins)
        grouped['goal_rate_percent'] = grouped['goal_rate'] * 100

        plt.plot(grouped['percentile'], grouped['goal_rate_percent'],
                color=color, linestyle=linestyle, linewidth=lw, alpha=0.9,
                label=f'{model_name}')


    plt.ylim([0, 105])
    plt.yticks(np.arange(0, 101, 10))
    plt.xticks(np.arange(0, 101, 10))
    plt.xlabel('Shots Ranked by Model Probability (%)')
    plt.ylabel('Observed Goal Rate (%)')
    plt.gca().invert_xaxis()
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

    

def plot_multi_cumulative_goal_proportion(y_val_list, y_proba_list, model_names, save_file="image.png"):
    """
    Plots the Cumulative Proportion of Goals vs. Probability Percentile for multiple models.

    Args:
        - y_val_list (list of array-like): List of true binary labels for each model.
        - y_proba_list (list of array-like): List of predicted probability arrays.
        - model_names (list of str): List of names for the models.
    """
    plt.figure(figsize=(8, 6))
    plt.plot([0, 100], [0, 100], color='gray', linestyle='--', linewidth=1.5,
             label='Random Baseline', alpha=0.6)

    for i, (y_val, y_proba, model_name) in enumerate(zip(y_val_list, y_proba_list, model_names)):
        y_val = np.ravel(np.array(y_val))
        color, linestyle, lw = get_plot_style(i)
        total_goals = y_val.sum()
        if total_goals == 0:
            print(f"Warning: No positive outcomes (goals) found for {model_name}. Skipping.")
            continue

        df = pd.DataFrame({'predicted_prob': y_proba, 'is_goal': y_val})
        sorted_df = df.sort_values('predicted_prob', ascending=False).reset_index(drop=True)
        sorted_df['cumulative_goals'] = sorted_df['is_goal'].cumsum()
        sorted_df['cumulative_goal_proportion'] = sorted_df['cumulative_goals'] / total_goals
        sorted_df['percentile'] = np.linspace(0, 100, len(sorted_df), endpoint=False)

        n_bins = 100
        binned = (sorted_df.groupby(pd.cut(sorted_df['percentile'], bins=n_bins), observed=False)
                    .agg({'cumulative_goal_proportion': 'last'}).reset_index())
        binned['percentile_mid'] = binned['percentile'].apply(lambda x: x.mid)
        binned['cumulative_goal_percent'] = binned['cumulative_goal_proportion'] * 100

        plt.plot(binned['percentile_mid'], binned['cumulative_goal_percent'],
                 color=color, linestyle=linestyle, linewidth=lw, alpha=0.9,
                 label=f'{model_name}')

    plt.xlabel('Shots Ranked by Model Probability (%)')
    plt.ylabel('Cumulative Goal Percentage Captured (%)')
    plt.ylim([0, 105])
    plt.yticks(np.arange(0, 101, 10))
    plt.xticks(np.arange(0, 101, 10))
    plt.gca().invert_xaxis()
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()

    
def plot_multi_reliability_diagram(estimators, X_val_list, y_val_list, model_names, n_bins=20, strategy='uniform', save_file="image.png"):
    """
    Plots the Reliability Diagram (Calibration Curve) for multiple models.

    Args:
        - estimators (list of sklearn-like estimator): List of fitted models.
        - X_val_list (list of array-like): List of validation features for each model.
        - y_val_list (list of array-like): List of true validation labels for each model.
        - model_names (list of str): List of names for the models.
        - n_bins (int): The number of bins to use for grouping predictions.
        - strategy (str): 'uniform' (equal-width bins) or 'quantile' (equal sample size bins).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (estimator, X_val, y_val, name) in enumerate(zip(estimators, X_val_list, y_val_list, model_names)):
        color, linestyle, lw = get_plot_style(i)
        CalibrationDisplay.from_estimator(
            estimator,
            X_val,
            y_val,
            n_bins=n_bins,
            strategy=strategy,
            name=name,
            ax=ax,
            color=color,
            linewidth=lw,
            linestyle=linestyle
        )

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5,
            label="Random Baseline", alpha=0.6, zorder=0)

    ax.set_xlabel("Mean Predicted Probability", fontdict={"fontsize":12})
    ax.set_ylabel("Fraction of Positives (Observed Goal Rate)",  fontdict={"fontsize":12})
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()
