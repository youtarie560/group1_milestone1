# gather the 3 Logistic Regressions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_curve, auc, roc_auc_score,
                             precision_score, recall_score, f1_score,
                             precision_recall_curve)
from sklearn.calibration import CalibrationDisplay
import xgboost
import vizualiation_func


def get_logreg_dist_results(test_data: pd.DataFrame) -> dict:

    X = test_data[['distance_from_net_ft']].values
    y = test_data['is_goal'].values

    model_name = 'clf_dist.pkl'
    model = joblib.load(model_name)
    y_proba_dist = model.predict_proba(X)[:, 1]

    return {
        'X_test': X,
        'y_test': y,
        'y_pred_proba': y_proba_dist,
        'model': model,
        'name': 'Logistic_Regression_Distance'
    }

def get_logreg_angle_results(test_data: pd.DataFrame) -> dict:

    X = test_data[['shot_angle_deg']].values
    y = test_data['is_goal'].values

    model_name = 'clf_angle.pkl'
    model = joblib.load(model_name)
    y_proba_dist = model.predict_proba(X)[:, 1]

    return {
        'X_test': X,
        'y_test': y,
        'y_pred_proba': y_proba_dist,
        'model': model,
        'name': 'Logistic_Regression_Angle'
    }


def get_logreg_dist_ang_results(test_data: pd.DataFrame) -> dict:

    X = test_data[['distance_from_net_ft', 'shot_angle_deg']].values
    y = test_data['is_goal'].values

    model_name = 'clf_dist_angle.pkl'
    model = joblib.load(model_name)
    y_proba_dist = model.predict_proba(X)[:, 1]

    return {
        'X_test': X,
        'y_test': y,
        'y_pred_proba': y_proba_dist,
        'model': model,
        'name': 'Logistic_Regression_Distance+Angle'
    }

def get_xgb_optimized_results(test_data: pd.DataFrame) -> dict:

    X = test_data.drop(columns=['is_goal']).apply(pd.to_numeric).values
    y = test_data['is_goal'].values

    model_name = '5_XGBoost_Optimized.pkl'
    model = joblib.load(model_name)
    y_proba_dist = model.predict_proba(X)[:, 1]

    return {
        'X_test': X,
        'y_test': y,
        'y_pred_proba': y_proba_dist,
        'model': model,
        'name': 'XGBoost_Optimized'
    }

def get_best_model(test_data: pd.DataFrame) -> dict:

    X = test_data.drop(columns=['is_goal']).values
    y = test_data['is_goal'].values

    model_name = 'XGBoost_20251112_054334.pkl'
    model = joblib.load(model_name)
    y_proba_dist = model.predict_proba(X)[:, 1]

    return {
        'X_test': X,
        'y_test': y,
        'y_pred_proba': y_proba_dist,
        'model': model,
        'name': 'XGBoost_Best'
    }



def main() -> None:

    data1 = pd.read_csv('5_test_playoff_engineered_df.csv') # Models 1-4
    data2 = pd.read_csv('6_test_playoff_engineered_df.csv') # Model 5

    model1 = get_logreg_dist_results(data1)
    model2 = get_logreg_angle_results(data1)
    model3 = get_logreg_dist_ang_results(data1)
    model4 = get_xgb_optimized_results(data1)
    model5 = get_best_model(data2)

    X_val_list = [model1['X_test'], model2['X_test'], model3['X_test'], model4['X_test'], model5['X_test']]
    y_val_arr = [model1['y_test'], model2['y_test'], model3['y_test'], model4['y_test'], model5['y_test']]
    y_proba_arr = [model1['y_pred_proba'], model2['y_pred_proba'], model3['y_pred_proba'], model4['y_pred_proba'], model5['y_pred_proba']]
    models_arr = [model1['model'], model2['model'], model3['model'], model4['model'], model5['model']]
    names_arr = [model1['name'], model2['name'], model3['name'], model4['name'], model5['name']]

    vizualiation_func.plot_multi_roc_curve(
        y_val_arr,
        y_proba_arr,
        names_arr,  # Explicitly name this argument
        save_file='roc_curve_playoff.png'  # Explicitly name this argument
    )

    vizualiation_func.plot_multi_cumulative_goal_proportion(
        y_val_arr,
        y_proba_arr,
        names_arr,
        save_file='cumulative_goal_proportion_playoff.png'
    )

    vizualiation_func.plot_multi_goal_rate_by_percentile(
        y_val_arr,
        y_proba_arr,
        names_arr,
        save_file='goal_rate_by_percentile_playoff.png'
    )

    vizualiation_func.plot_multi_reliability_diagram(
        models_arr,
        X_val_list,
        y_val_arr,
        names_arr,
        strategy='uniform',
        save_file='reliability_diagram_playoff.png'
    )




if __name__ == '__main__':
    main()