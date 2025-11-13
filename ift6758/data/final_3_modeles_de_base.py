import os
import joblib
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibrationDisplay

from sklearn.metrics import (accuracy_score, roc_curve, auc, roc_auc_score, 
                             precision_score, recall_score, f1_score, 
                             precision_recall_curve)


from vizualiation_func import (plot_multi_roc_curve, 
                               plot_multi_goal_rate_by_percentile, 
                               plot_multi_cumulative_goal_proportion, 
                               plot_multi_reliability_diagram)

df = pd.read_csv("train_processed_df.csv", index_col=False)
X = df[['distance_from_net_ft']].values
y = df['is_goal'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
print(f"y pred: {np.asarray((pred_unique, pred_counts)).T}")

val_unique, val_counts = np.unique(y_val, return_counts=True)
print(f"y val: {np.asarray((val_unique, val_counts)).T}")

print(f"{val_counts[0] / pred_counts[0]}")

X_full = df[['distance_from_net_ft', 'shot_angle_deg']].values
y = df['is_goal'].values

X_train_full, X_val, y_train, y_val = train_test_split(X_full, y, test_size=0.20)

# Distance
X_train_dist = X_train_full[:, [0]]
X_val_dist = X_val[:, [0]]
clf_dist = LogisticRegression().fit(X_train_dist, y_train)
y_proba_dist = clf_dist.predict_proba(X_val_dist)[:, 1]

# Angle
X_train_angle = X_train_full[:, [1]]
X_val_angle = X_val[:, [1]]
clf_angle = LogisticRegression().fit(X_train_angle, y_train)
y_proba_angle = clf_angle.predict_proba(X_val_angle)[:, 1]

# Distance and Angle
clf_dist_angle = LogisticRegression().fit(X_train_full, y_train)
y_proba_dist_angle = clf_dist_angle.predict_proba(X_val)[:, 1]

# Accumulate data for the vizualiation functions
y_val_list = [y_val, y_val, y_val, y_val]
X_val_list = [X_val_dist, X_val_angle, X_val]
proba_list = [y_proba_dist, y_proba_angle, y_proba_dist_angle]
estimator_list = [clf_dist, clf_angle, clf_dist_angle]
name_list = ['Logistic Regression (Distance)', 'Logistic Regression (Angle)', 'Logistic Regression (Distance+Angle)']

# For y_val list
y_val_arr = np.empty(len(y_val_list), dtype=object)
y_val_arr[:] = y_val_list

# For X_val_list
X_val_arr = np.empty(len(X_val_list), dtype=object)
X_val_arr[:] = X_val_list

# For proba_list
proba_arr = np.empty(len(proba_list), dtype=object)
proba_arr[:] = proba_list

# For estimator_list
estimator_arr = np.empty(len(estimator_list), dtype=object)
estimator_arr[:] = estimator_list

# For name_list
name_arr = np.empty(len(name_list), dtype=object)
name_arr[:] = name_list

np.savez("clf_models.npz",
         y_val_list=y_val_list,
         X_val_list=X_val_arr,
         proba_list=proba_arr,
         estimator_list=estimator_arr,
         name_list=name_arr)

for X_val_model, clf, name in zip(X_val_list, estimator_list, name_list):
    

    y_proba_full = clf.predict_proba(X_val_model)
    y_proba = y_proba_full[:, 1]
    y_pred = clf.predict(X_val_model)
    
    precision, recall, _ = precision_recall_curve(y_val, y_proba)
    pr_auc = auc(recall, precision)
    
    metrics = {"val_auc": roc_auc_score(y_val, y_proba),
               "pr_auc": pr_auc,
               "accuracy": accuracy_score(y_val, y_pred),
               "precision": precision_score(y_val, y_pred),
               "recall": recall_score(y_val, y_pred),
               "f1_score": f1_score(y_val, y_pred),
               "roc_curve": wandb.plot.roc_curve(y_val, 
                                                 y_proba_full, 
                                                 labels=["No Goal", "Goal"]),
               "pr_curve": wandb.plot.pr_curve(y_val, 
                                               y_proba_full, 
                                               labels=["No Goal", "Goal"]),
               "confusion_matrix": wandb.plot.confusion_matrix(y_true=y_val,
                                                               preds=y_pred,
                                                               class_names=["No Goal", "Goal"])}

    # model_path = f"{name.replace(' & ', '_')}_logreg_model.pkl"
    # joblib.dump(clf, model_path)

plot_multi_roc_curve(y_val, proba_list, name_list)
plot_multi_goal_rate_by_percentile(y_val, proba_list, name_list)
plot_multi_cumulative_goal_proportion(y_val, proba_list, name_list)
plot_multi_reliability_diagram(estimator_list, X_val_list, y_val, name_list, strategy='uniform')
