import wandb
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.calibration import CalibrationDisplay

from sklearn.metrics import (accuracy_score, roc_curve, auc, roc_auc_score, 
                             precision_score, recall_score, f1_score, 
                             precision_recall_curve)

from vizualiation_func import (plot_multi_roc_curve, 
                               plot_multi_goal_rate_by_percentile, 
                               plot_multi_cumulative_goal_proportion, 
                               plot_multi_reliability_diagram)

train_engineered_df = pd.read_csv("train_engineered_df.csv" , low_memory=False)
df = train_engineered_df.copy()

EVENT_TYPE = {"shot-on-goal", "goal", "missed-shot", "blocked-shot"}
df = df[df['typeDescKey'].isin(EVENT_TYPE)]
df = df[['distance_from_net_ft', 'shot_angle_deg', 'is_goal']]
df = df.dropna()

X = df[['distance_from_net_ft', 'shot_angle_deg']].values
y = df['is_goal'].values
X_train_base, X_val_base, y_train_base, y_val_base = train_test_split(X, y, test_size=0.20)

xgb_base = xgb.XGBClassifier() #eval_metric='logloss')
xgb_base.fit(X_train_base, y_train_base)
y_pred_base = xgb_base.predict(X_val_base)
y_pred_proba_base = xgb_base.predict_proba(X_val_base)[:, 1]
accuracy = accuracy_score(y_val_base, y_pred_base)
# run = wandb.init(project="IFT6758.2025-A01",
#                  entity="IFT67582025A1",
#                  name=f"XGBoost_Distance+Angle",
#                  tags=["XGBoost", "Distance+Angle"])

y_proba_full = xgb_base.predict_proba(X_val_base)
y_proba = y_proba_full[:, 1]
y_pred = xgb_base.predict(X_val_base)

precision, recall, _ = precision_recall_curve(y_val_base, y_proba)
pr_auc = auc(recall, precision)

# metrics = {"val_auc": roc_auc_score(y_val_base, y_proba),
#             "pr_auc": pr_auc,
#             "accuracy": accuracy_score(y_val_base, y_pred),
#             "precision": precision_score(y_val_base, y_pred),
#             "recall": recall_score(y_val_base, y_pred),
#             "f1_score": f1_score(y_val_base, y_pred),
#             "roc_curve": wandb.plot.roc_curve(y_val_base,
#                                                 y_proba_full,
#                                                 labels=["No Goal", "Goal"]),
#             "pr_curve": wandb.plot.pr_curve(y_val_base,
#                                             y_proba_full,
#                                             labels=["No Goal", "Goal"]),
#             "confusion_matrix": wandb.plot.confusion_matrix(y_true=y_val_base,
#                                                             preds=y_pred,
#                                                             class_names=["No Goal", "Goal"])}
# wandb.log(metrics)

# run.finish()

y_pred_base = xgb_base.predict(X_val_base)
y_pred_proba_base = xgb_base.predict_proba(X_val_base)[:, 1]
accuracy = accuracy_score(y_val_base, y_pred_base)
y_val_list = [y_val_base]
X_val_list = [X_val_base]
proba_list = [y_pred_proba_base]
estimator_list = [xgb_base]
name_list = ['XGBoost (Distance+Angle)']

# plot_multi_roc_curve(y_val_list, proba_list, name_list, save_file="figures/milestone2-5-1.png")
# plot_multi_goal_rate_by_percentile(y_val_list, proba_list, name_list, save_file="figures/milestone2-5-2.png")
# plot_multi_cumulative_goal_proportion(y_val_list, proba_list, name_list, save_file="figures/milestone2-5-3.png")
# plot_multi_reliability_diagram(estimator_list, X_val_list, y_val_list, name_list, strategy='uniform', save_file="figures/milestone2-5-4.png")

# model_path = f"5_XGBoost_Distance+Angle.pkl"
# joblib.dump(xgb_base, model_path)

EVENT_TYPE = {"shot-on-goal", "goal", "blocked-shot", "missed-shot"}

df = train_engineered_df.copy()
df = df[df['typeDescKey'].isin(EVENT_TYPE)] 
df = df.dropna(subset=['distance_from_net_ft', 'shot_angle_deg'])

df = df[['is_goal', 'game_seconds', 'periodDescriptor.number','x','y', 'distance_from_net_ft', 'shot_angle_deg','details.shotType', 'prev_event', 'prev_x', 'prev_y', 'seconds_since_prev', 'distance_to_prev', 'rebound', 'shot_angle_change', 'speed_from_prev']].copy()
df['rebound'] = df['rebound'].astype(int)
df['speed_from_prev'] = df['speed_from_prev'].replace([np.inf, -np.inf], 0).fillna(0)

categorical_cols = ['periodDescriptor.number', 'details.shotType', 'prev_event']

df = pd.get_dummies(df, 
                    columns=categorical_cols, 
                    prefix=categorical_cols, 
                    drop_first=False,
                    dtype=int)
X = df.drop(columns=['is_goal']).apply(pd.to_numeric).values
y = df['is_goal'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=(np.sum(y_train == 0) / np.sum(y_train == 1)),
    eval_metric='aucpr')

search_spaces = {
    'max_depth': Integer(5, 25),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'n_estimators': Integer(100, 600),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'min_child_weight': Integer(1, 10),
    'gamma': Real(0.0, 5.0),
    'reg_lambda': Real(1e-3, 10.0, prior='log-uniform')}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

opt = BayesSearchCV(
    estimator=xgb_model,
    search_spaces=search_spaces,
    n_iter=50,                 
    scoring='average_precision',         
    cv=cv,
    n_jobs=-1,
    verbose=0)

opt.fit(X_train, y_train)
best_model = opt.best_estimator_
y_pred_optimized = best_model.predict(X_val)
y_pred_proba_optimized = best_model.predict_proba(X_val)[:, 1]
accuracy = accuracy_score(y_val, y_pred_optimized)
y_val_list = [y_val]
X_val_list = [X_val]
proba_list = [y_pred_proba_optimized]
estimator_list = [best_model]
name_list = ['XGBoost (Optimized)']

# plot_multi_roc_curve(y_val_list, proba_list, name_list, save_file="figures/milestone2-5-5.png")
# plot_multi_goal_rate_by_percentile(y_val_list, proba_list, name_list, save_file="figures/milestone2-5-6.png")
# plot_multi_cumulative_goal_proportion(y_val_list, proba_list, name_list, save_file="figures/milestone2-5-7.png")
# plot_multi_reliability_diagram(estimator_list, X_val_list, y_val_list, name_list, strategy='uniform', save_file="figures/milestone2-5-8.png")
# model_path = f"5_XGBoost_Optimized.pkl"
# joblib.dump(best_model, model_path)
y_pred_optimized = best_model.predict(X_val)
y_pred_proba_optimized = best_model.predict_proba(X_val)[:, 1]
accuracy = accuracy_score(y_val, y_pred_optimized)

# run = wandb.init(project="IFT6758.2025-A01",
#                  entity="IFT67582025A1",
#                  name=f"XGBoost_Optimized",
#                  tags=["XGBoost", "Optimized", "Engineered"])
    
y_proba_full = best_model.predict_proba(X_val)
y_proba = y_proba_full[:, 1]
y_pred = best_model.predict(X_val)

precision, recall, _ = precision_recall_curve(y_val, y_proba)
pr_auc = auc(recall, precision)

# metrics = {"val_auc": roc_auc_score(y_val, y_proba),
#             "pr_auc": pr_auc,
#             "accuracy": accuracy_score(y_val, y_pred),
#             "precision": precision_score(y_val, y_pred),
#             "recall": recall_score(y_val, y_pred),
#             "f1_score": f1_score(y_val, y_pred),
#             "roc_curve": wandb.plot.roc_curve(y_val,
#                                                 y_proba_full,
#                                                 labels=["No Goal", "Goal"]),
#             "pr_curve": wandb.plot.pr_curve(y_val,
#                                             y_proba_full,
#                                             labels=["No Goal", "Goal"]),
#             "confusion_matrix": wandb.plot.confusion_matrix(y_true=y_val,
#                                                             preds=y_pred,
#                                                             class_names=["No Goal", "Goal"])}
# wandb.log(metrics)

# run.finish()
scores = np.array(opt.cv_results_['mean_test_score'])
best_scores_so_far = np.maximum.accumulate(scores)

# plt.plot(range(1, len(best_scores_so_far)+1), best_scores_so_far, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Best Cross-Validation Score (Until Current Iteration)')
plt.grid(True)
# plt.savefig("figures/milestone2-5-9.png", dpi=300, bbox_inches='tight')
plt.show()

results = pd.DataFrame(opt.cv_results_)
param_names = [key for key in results.columns if key.startswith('param_')]

plt.figure(figsize=(len(param_names)*4,4))
for i, param in enumerate(param_names):
    plt.subplot(1, len(param_names), i+1)
    plt.scatter(results[param], results['mean_test_score'])
    plt.xlabel(param)
    plt.ylabel('Mean Cross-Validation Score')
    plt.title(param)
plt.tight_layout()
# plt.savefig("figures/milestone2-5-10.png", dpi=300, bbox_inches='tight')
plt.show()
#%%
y_val_arr = np.empty(len(y_val_list), dtype=object)
y_val_arr[:] = y_val_list

X_val_arr = np.empty(len(X_val_list), dtype=object)
X_val_arr[:] = X_val_list

proba_arr = np.empty(len(proba_list), dtype=object)
proba_arr[:] = proba_list

estimator_arr = np.empty(len(estimator_list), dtype=object)
estimator_arr[:] = estimator_list

name_arr = np.empty(len(name_list), dtype=object)
name_arr[:] = name_list

# np.savez("xgb_models.npz",
#          y_val_list=y_val_list,
#          X_val_list=X_val_arr,
#          proba_list=proba_arr,
#          estimator_list=estimator_arr,
#          name_list=name_arr)

corr_matrix = df.corr().abs()
threshold = 0.5

upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
stacked_corr = upper_tri.stack().reset_index()
stacked_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']
stacked_corr = stacked_corr.sort_values('Correlation', ascending=False)

highly_correlated = stacked_corr[stacked_corr['Correlation'] >= threshold]
highly_correlated = highly_correlated.sort_values(by='Correlation', ascending=False)

feature_importances = best_model.feature_importances_
features = df.drop(columns=['is_goal']).columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance Scores')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
# plt.savefig("figures/milestone2-5-FeatImportance")
plt.show()