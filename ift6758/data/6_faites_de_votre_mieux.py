import os
import json
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import BayesianOptimization
from kerastuner import Objective

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, 
                             classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score, 
                             balanced_accuracy_score, average_precision_score, 
                             make_scorer)
from sklearn.utils import class_weight

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Coordinates for determining the "Slot"
X_FAR_LIMIT = 59 
X_NEAR_LIMIT = 89 
Y_LIMIT = 4      

# Plays of interest
EVENT_TYPES = {'goal', 'shot-on-goal', 'blocked-shot', 'missed-shot'}

def time_to_seconds(time_str):
    """
    Converts a time format to total seconds.
    Args:
        - time_str (str): The time in MM:SS for the Period time in a hockey game.
    """
    if pd.isna(time_str):
        return np.nan
    try:
        m, s = map(float, time_str.split(':'))
        return m * 60 + s
    except ValueError:
        return np.nan
    

def save_model(model, model_name, hyperparams=None):
    """
    Saves the model locally with a timestamp and optionally saves hyperparameters.
    
    Args:
        - model: Trained model object
        - model_name: Name of the model (string)
        - hyperparams: dict of hyperparameters (optional)
        
    Returns:
        - path: Path to the saved model file
    """
    
    os.makedirs("saved_models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"
    
    path = os.path.join("saved_models", filename)
    joblib.dump(model, path)
    print(f"Saved {model_name} model as {filename}")
    
    if hyperparams is not None:
        hp_filename = f"{model_name}_{timestamp}_hyperparams.json"
        hp_path = os.path.join("saved_models", hp_filename)
        with open(hp_path, "w") as f:
            json.dump(hyperparams, f, indent=4)
        print(f"Saved {model_name} hyperparameters as {hp_filename}")
    
    return path


print("Loading data...")
df = pd.read_csv("train_engineered_df.csv", low_memory=False)

rename_dict = {
    'timeInPeriod': 'time_in_period',
    'timeRemaining': 'time_remaining_period',
    'periodDescriptor.number': 'period_number',
    'periodDescriptor.periodType': 'period_type',
    'periodDescriptor.maxRegulationPeriods': 'max_reg_periods',
    'game_seconds': 'game_time_sec',
    'typeCode': 'event_type_code',
    'typeDescKey': 'event_type_desc',
    'situationCode': 'game_situation_code',
    'details.zoneCode': 'zone_code',
    'x': 'shot_x',
    'y': 'shot_y',
    'distance_from_net_ft': 'shot_distance',
    'shot_angle_deg': 'shot_angle',
    'details.eventOwnerTeamId': 'owner_team_id',
    'details.losingPlayerId': 'losing_player_id',
    'details.winningPlayerId': 'winning_player_id',
    'details.playerId': 'player_id',
    'details.blockingPlayerId': 'blocker_id',
    'details.shootingPlayerId': 'shooter_id',
    'details.goalieInNetId': 'goalie_id',
    'details.hittingPlayerId': 'hitter_id',
    'details.hitteePlayerId': 'hittee_id',
    'details.scoringPlayerId': 'scorer_id',
    'details.assist1PlayerId': 'assist1_id',
    'details.assist2PlayerId': 'assist2_id',
    'details.committedByPlayerId': 'committed_by_id',
    'details.drawnByPlayerId': 'drawn_by_id',
    'details.servedByPlayerId': 'served_by_id',
    'details.shotType': 'shot_type',
    'details.awaySOG': 'away_sog',
    'details.homeSOG': 'home_sog',
    'details.awayScore': 'score_away',
    'details.homeScore': 'score_home',
    'prev_event': 'prev_event_type',
    'seconds_since_prev': 'time_since_prev_event',
    'distance_to_prev': 'dist_from_prev_event',
    'prev_distance_from_net': 'prev_shot_distance',
    'rebound': 'is_rebound',
    'shot_angle_change': 'angle_change',
    'speed_from_prev': 'speed'
}
df.rename(columns=rename_dict, inplace=True)

columns_to_drop = [
    'eventId', 'event_type_code', 'sortOrder', 'period_type',
    'details.xCoord', 'details.yCoord', 'details.reason',
    'details.secondaryReason', 'details.discreteClip',
    'homeTeamDefendingSide', 'details.descKey', 'details.duration',
    'prev_x', 'prev_y', 'prev_shot_angle',
    'details.scoringPlayerTotal', 'details.assist1PlayerTotal', 
    'details.assist2PlayerTotal'
]
df.drop(columns=columns_to_drop, errors='ignore', inplace=True)


print("Performing additional engineering features to add to the features engineered in Part 4...")

df['situation_code_str'] = df['game_situation_code'].astype(str).str.zfill(4)
df['away_goalie_on_ice'] = df['situation_code_str'].str[0] == "1"
df['home_goalie_on_ice'] = df['situation_code_str'].str[3] == "1"

df = df.assign(away_skaters=pd.to_numeric(df['situation_code_str'].str[1], errors='coerce').fillna(5).astype(int),
               home_skaters=pd.to_numeric(df['situation_code_str'].str[2], errors='coerce').fillna(5).astype(int))

df['is_empty_net'] = np.where(df['owner_team_id'] == df['home_id'],
                                  ~df['away_goalie_on_ice'],
                                  ~df['home_goalie_on_ice']).astype(int)

df['shooter_skater_diff'] = np.where(df['owner_team_id'] == df['home_id'],
                                     df['home_skaters'] - df['away_skaters'],
                                     df['away_skaters'] - df['home_skaters'])

df['is_even_strength'] = (df['shooter_skater_diff'] == 0).astype(int)
df['is_power_play']    = (df['shooter_skater_diff'] > 0).astype(int)
df['is_shorthanded']   = (df['shooter_skater_diff'] < 0).astype(int)

df.drop(columns=['situation_code_str', 'away_goalie_on_ice', 'home_goalie_on_ice',
                 'away_skaters', 'home_skaters', 'shooter_skater_diff'], inplace=True)

cols_to_fill = ['away_sog', 'home_sog', 'score_away', 'score_home']
df[cols_to_fill] = df.groupby('game_id')[cols_to_fill].ffill().fillna(0)

df['speed'].replace([np.inf, -np.inf], 0, inplace=True)

df['is_in_slot'] = (((df['shot_x'].between(X_FAR_LIMIT, X_NEAR_LIMIT)) & (df['shot_y'].abs() <= Y_LIMIT)) |
                    ((df['shot_x'].between(-X_NEAR_LIMIT, -X_FAR_LIMIT)) & (df['shot_y'].abs() <= Y_LIMIT))).astype(int)

df['shot_distance_binned'] = pd.cut(df['shot_distance'],
                                    bins=[0, 10, 25, 45, df['shot_distance'].max() + 1],
                                    labels=['Very_Close', 'Medium_Close', 'Medium_Far', 'Long_Shot'],
                                    right=False)

df['is_quick_release'] = (df['time_since_prev_event'] <= 3).astype(int)

df['is_prime_rebound'] = ((df['is_rebound'] == 1) &
                          (df['shot_distance'] <= 15) &
                          (df['time_since_prev_event'] <= 3)).astype(int)

df['speed_binned'] = pd.cut(df['speed'],
                            bins=[-1, 5, 10, 20, df['speed'].max() + 1],
                            labels=['Slow', 'Medium', 'Fast', 'Very_Fast'],
                            right=False)

df['angle_change'] = df['angle_change'].fillna(0)

drop_ids = ['player_id', 'hitter_id', 'hittee_id', 'shooter_id',
            'goalie_id', 'owner_team_id', 'home_id', 'away_id',
            'losing_player_id', 'winning_player_id', 'details.typeCode',
            'committed_by_id', 'drawn_by_id', 'served_by_id',
            'blocker_id', 'scorer_id', 'assist1_id', 'assist2_id',
            'attackingDirection', 'game_id']
df.drop(columns=drop_ids, inplace=True, errors='ignore')


ohe_cols = ['shot_type', 'zone_code', 'prev_event_type', 
            'shot_distance_binned', 'speed_binned']
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dummy_na=False)

df['time_in_period_sec'] = df['time_in_period'].apply(time_to_seconds)
df['time_remaining_period_sec'] = df['time_remaining_period'].apply(time_to_seconds)
df.drop(columns=['time_in_period', 'time_remaining_period'], inplace=True)

scaling_cols = ['time_in_period_sec', 'time_remaining_period_sec', 
                'shot_distance', 'shot_angle', 'game_time_sec', 
                'prev_shot_distance', 'time_since_prev_event', 
                'dist_from_prev_event','angle_change', 'speed']
scaler = StandardScaler()
df[scaling_cols] = scaler.fit_transform(df[scaling_cols])


df = df[df['event_type_desc'].isin(EVENT_TYPES)]
df['is_goal'] = (df['event_type_desc'] == 'goal').astype(int)
df.drop(columns=['game_situation_code', 'event_type_desc'], inplace=True, errors='ignore')

bool_cols = df.select_dtypes(include='bool').columns
if len(bool_cols) > 0:
    df[bool_cols] = df[bool_cols].astype(int)

# Drop missing values
df = df.dropna(subset=['shot_distance'])
df.dropna(inplace=True)


# Data preparation for the ML models
X = df.drop(columns=['is_goal']).values
y = df['is_goal'].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
print(f"Target balance: {y.mean():.3%} goals")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                  test_size=0.2, 
                                                  random_state=42, 
                                                  stratify=y)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Training goal rate: {y_train.mean():.3%}")
print(f"Validation goal rate: {y_val.mean():.3%}")


# ============================================================================
# Random Forest
# ============================================================================

print("\n" + "="*70)
print("TRAINING RANDOM FOREST MODEL")
print("="*70)

rf_model = RandomForestClassifier(random_state=42,
                                  n_jobs=-1,
                                  class_weight='balanced',
                                  warm_start=False)

rf_search_spaces = {
    'n_estimators': Integer(100, 600),
    'max_depth': Integer(5, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Categorical(['sqrt', 'log2', None]),
    'bootstrap': Categorical([True]),
    'criterion': Categorical(['gini', 'log_loss', 'entropy']),
    'max_samples': Real(0.5, 1.0)
}

cv = StratifiedKFold(n_splits=3, 
                     shuffle=True, 
                     random_state=42)

pr_auc_scorer = make_scorer(average_precision_score)

rf_opt = BayesSearchCV(
    estimator=rf_model,
    search_spaces=rf_search_spaces,
    n_iter=40,
    scoring=pr_auc_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

rf_opt.fit(X_train, y_train)
rf_best_model = rf_opt.best_estimator_
rf_hps = rf_opt.best_params_
save_model(rf_best_model, "Random_Forest", hyperparams=rf_hps)


# ============================================================================
# Neural Network
# ============================================================================

print("\n" + "="*70)
print("TRAINING NEURAL NETWORK MODEL")
print("="*70)

def build_model(hp):
    model = keras.Sequential([
        layers.Dense(
            128,
            activation='relu',
            input_shape=(X_train.shape[1],),
            kernel_regularizer=keras.regularizers.l2(hp.Float('l2_reg_1', 1e-6, 1e-2, sampling='log'))
        ),
        layers.Dropout(rate=hp.Float('dropout_1', 0.1, 0.5, step=0.05)),
        layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(hp.Float('l2_reg_2', 1e-6, 1e-2, sampling='log'))
        ),
        layers.Dropout(rate=hp.Float('dropout_2', 0.1, 0.5, step=0.05)),
        layers.Dense(1, activation='sigmoid')
    ])

    optimizer = keras.optimizers.Adam(
        learning_rate=hp.Float('learning_rate', 1e-5, 5e-2, sampling='log')
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='AUC')]
    )

    return model

X_train_np = np.array(X_train)
y_train_np = np.array(y_train)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

max_trials = 25
best_score = 0
best_hps = None

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_np, y_train_np), 1):
    print(f"\n=== Fold {fold} ===")

    X_tr, X_val_fold = X_train_np[train_idx], X_train_np[val_idx]
    y_tr, y_val_fold = y_train_np[train_idx], y_train_np[val_idx]

    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    cw_dict = dict(enumerate(cw))

    tuner = BayesianOptimization(
        build_model,
        objective=Objective('val_AUC', direction='max'),
        max_trials=max_trials,
        directory='keras_tuner_cv',
        overwrite=True,
        seed=42
    )

    stop_early = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    tuner.search(
        X_tr, y_tr,
        epochs=75,
        batch_size=64,
        validation_data=(X_val_fold, y_val_fold),
        class_weight=cw_dict,
        callbacks=[stop_early],
        verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.hypermodel.build(best_hp)

    best_model.fit(
        X_tr, y_tr,
        epochs=50,
        batch_size=64,
        validation_data=(X_val_fold, y_val_fold),
        class_weight=cw_dict,
        verbose=0,
        callbacks=[stop_early]
    )

    y_val_pred = best_model.predict(X_val_fold).ravel()
    fold_score = average_precision_score(y_val_fold, y_val_pred)
    print(f"Fold {fold} PR AUC: {fold_score:.4f}")

    if fold_score > best_score:
        best_score = fold_score
        best_hps = best_hp
        mlp_best_model = best_model
        
    mlp_hps = best_hps.values
    save_model(mlp_best_model, "Neural_Network_MLP", hyperparams=mlp_hps)

        
print("\n=============================================")
print("Best Cross-Validation PR AUC:", best_score)
print("Best Hyperparameters:")
for param, value in best_hps.values.items():
    print(f"  {param}: {value}")

save_model(mlp_best_model, "Neural_Network_MLP")

# ============================================================================
# LightGBM
# ============================================================================

print("\n" + "="*70)
print("TRAINING LIGHTGBM MODEL")
print("="*70)

lgb_model = lgb.LGBMClassifier(
    objective='binary',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

lgbm_search_spaces = {
    'num_leaves': Integer(30, 200),   
    'max_depth': Integer(5, 50),
    'learning_rate': Real(0.001, 0.3, prior='log-uniform'),
    'n_estimators': Integer(300, 1200),    
    'min_child_samples': Integer(5, 50),
    'subsample': Real(0.3, 1.0),             
    'colsample_bytree': Real(0.4, 1.0),      
    'reg_lambda': Real(1e-3, 20, prior='log-uniform')
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
pr_auc_scorer = make_scorer(average_precision_score)

lgb_opt = BayesSearchCV(
    estimator=lgb_model,
    search_spaces=lgbm_search_spaces,
    n_iter=50, 
    scoring=pr_auc_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

lgb_opt.fit(X_train, y_train)
lgb_best_model = lgb_opt.best_estimator_
lgb_hps = lgb_opt.best_params_
save_model(lgb_best_model, "LightGBM", hyperparams=lgb_hps)


# ============================================================================
# XGBoost (v3)
# ============================================================================

print("\n" + "="*70)
print("TRAINING XGBOOST MODEL")
print("="*70)

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss'
)

search_spaces = {
    'max_depth': Integer(3, 15),
    'learning_rate': Real(0.005, 0.5, prior='log-uniform'),
    'n_estimators': Integer(200, 800),
    'subsample': Real(0.4, 1.0),
    'colsample_bytree': Real(0.4, 1.0),
    'min_child_weight': Integer(1, 20),
    'gamma': Real(0.0, 10.0),
    'reg_lambda': Real(1e-3, 20.0, prior='log-uniform'),
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

xgb_opt = BayesSearchCV(
    estimator=xgb_model,
    search_spaces=search_spaces,
    n_iter=75,                 
    scoring='roc_auc',         
    cv=cv,
    n_jobs=-1,
    verbose=0
)

xgb_opt.fit(X_train, y_train)
xgb_best_model = xgb_opt.best_estimator_
xgb_hps = xgb_opt.best_params_
save_model(xgb_best_model, "XGBoost", hyperparams=xgb_hps)

# ============================================================================
# MODEL EVALUATION
# ============================================================================

print("\n" + "="*70)
print("EVALUATING MODELS ON VALIDATION SET")
print("="*70)

models = {
    'Random Forest': rf_best_model,
    'Neural Network (MLP)': mlp_best_model,
    'LightGBM': lgb_best_model,
    'XGBoost': xgb_best_model,
}

print(f"Class distribution: {np.bincount(y_val)}")
print(f"Positive class (goals) percentage: {y_val.mean():.2%}\n")

results = []
all_predictions = {}
all_probabilities = {}

for name, model in models.items():
    if isinstance(model, tf.keras.Model):
        y_pred_proba = model.predict(X_val).ravel()
        y_pred = (y_pred_proba >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    all_predictions[name] = y_pred
    all_probabilities[name] = y_pred_proba
    
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    pr_auc = average_precision_score(y_val, y_pred_proba)
    accuracy = accuracy_score(y_val, y_pred)
    balanced_acc = balanced_accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    f1_weighted = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    
    results.append({
        'Model': name,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'Accuracy': accuracy,
        'Balanced Accuracy': balanced_acc,
        'Precision (Goals)': precision,
        'Recall (Goals)': recall,
        'F1-Score': f1,
        'F1-Weighted': f1_weighted
    })
    
    print(f"\n{name}:")
    print(f"  ROC-AUC:          {roc_auc:.4f}")
    print(f"  PR-AUC:           {pr_auc:.4f}  ⭐ (Better for imbalanced data)")
    print(f"  Accuracy:         {accuracy:.4f}")
    print(f"  Balanced Accuracy:{balanced_acc:.4f}")
    print(f"  Precision (Goals):{precision:.4f}")
    print(f"  Recall (Goals):   {recall:.4f}")
    print(f"  F1-Score:         {f1:.4f}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('PR-AUC', ascending=False)

print("\n" + "="*70)
print("MODEL COMPARISON (sorted by PR-AUC)")
print("="*70)
print(results_df.to_string(index=False))