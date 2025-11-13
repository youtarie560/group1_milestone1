import os
import wandb
import pandas as pd
import numpy as np

def combine_all_years_to_df(data_dir, years=range(2016, 2024)):
    '''
    Function to combine data for multiple years into a single DataFrame.
    
    Args: 
        - years (list of int): List of years to combine.
    Returns:
        - combined_df (pd.DataFrame): A DataFrame containing all combined data for the given years.
    '''
    combined = []
    for year in years:
        path = os.path.join(data_dir, f"playbyplay-{str(year)}.json")
        df = pd.read_json(path)
        combined.append(df)
    
    combined_df = pd.concat(combined, ignore_index=True)
    return combined_df

df = combine_all_years_to_df(data_dir='data', years=range(2016,2021))
df = df[df['gameType'] == 2]
train_df = df[df['season'].isin([20162017, 20172018, 20182019, 20192020])]
test_df = df[df['season'] == 20202021]

def normalize_game_plays(game_row: pd.Series) -> pd.DataFrame:

    plays = game_row.get("plays", [])
    
    if not isinstance(plays, list) or len(plays) == 0:
        return pd.DataFrame()

    return pd.json_normalize(plays)


def infer_attacking_direction(plays: pd.DataFrame) -> pd.Series:

    tmp = plays.dropna(subset=["details.xCoord"]).copy()
    if tmp.empty:
        return pd.Series([None] * len(plays), index=plays.index)

    tmp["teamId"] = tmp["details.eventOwnerTeamId"]
    tmp["period"] = tmp["periodDescriptor.number"]
    mean_x = tmp.groupby(["teamId", "period"])["details.xCoord"].mean()

    cache = {}
    for (team_id, period), mx in mean_x.items():
        cache.setdefault(period, {})[team_id] = "left" if mx < 0 else "right"

    return plays.apply(lambda r: cache.get(r.get("periodDescriptor.number"), {}).get(r.get("details.eventOwnerTeamId")),
                       axis=1)


def compute_empty_net(plays: pd.DataFrame, home_team_id: int, away_team_id: int) -> pd.Series:

    if "situationCode" not in plays.columns:
        return pd.Series(0, index=plays.index, dtype=int)

    situation = plays["situationCode"].astype(str)
    away_goalie_in = situation.str[0] == "1"
    home_goalie_in = situation.str[3] == "1"

    shooter_team = plays["details.eventOwnerTeamId"]
    result = pd.Series(0, index=plays.index, dtype=int)
    home_shooting = shooter_team == home_team_id
    away_shooting = shooter_team == away_team_id

    result[home_shooting] = (~away_goalie_in[home_shooting]).astype(int)
    result[away_shooting] = (~home_goalie_in[away_shooting]).astype(int)
    return result


def calculate_net_properties(row):

    left_net_x = -89.0
    right_net_x = 89.0

    x = row.get("x")
    y = row.get("y")
    direction = row.get("attackingDirection")

    if pd.isna(x) or pd.isna(y):
        return np.nan, np.nan

    if direction == "right":
        net_x = right_net_x
    elif direction == "left":
        net_x = left_net_x
    else:
        net_x = right_net_x if x >= 0 else left_net_x

    net_y = 0
    dx =  abs(net_x - x)
    dy = net_y - y
    distance = np.sqrt(dx**2 + dy**2)

    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    
    return distance, angle_deg


def extract_shot_features(df: pd.DataFrame) -> pd.DataFrame:

    all_plays = []

    for _, game in df.iterrows():
        plays = normalize_game_plays(game)
        if plays.empty:
            continue
        
        
        plays["x"] = pd.to_numeric(plays["details.xCoord"], errors="coerce")
        plays["y"] = pd.to_numeric(plays["details.yCoord"], errors="coerce")

        plays["attackingDirection"] = infer_attacking_direction(plays)

        game_id = game["id"]
        home_id = game["homeTeam"]["id"]
        away_id = game["awayTeam"]["id"]
        plays["empty_net"] = compute_empty_net(plays, home_id, away_id)

        plays[["distance_from_net_ft", "shot_angle_deg"]] = plays.apply(
            calculate_net_properties, axis=1, result_type="expand")
        
        plays['distance_from_net_ft'] = plays['distance_from_net_ft'].round(2)
        plays['shot_angle_deg'] = plays['shot_angle_deg'].round(2)
        plays["is_goal"] = (plays["typeDescKey"] == "goal").astype(int)

        plays["game_id"] = game_id
        plays["home_id"] = home_id
        plays["away_id"] = away_id
        
        all_plays.append(plays)

    if not all_plays:
        return pd.DataFrame(columns=plays.columns)

    final = pd.concat(all_plays, ignore_index=True)
    return final

train_processed_df = extract_shot_features(train_df)

def mmss_to_seconds(mmss):
    minutes, seconds = map(int, mmss.split(':'))
    return (minutes * 60) + seconds

train_processed_df['game_seconds'] = train_processed_df['timeInPeriod'].apply(mmss_to_seconds)

#%%
train_processed_df['prev_event'] = train_processed_df['typeDescKey'].shift(1)
train_processed_df['prev_x'] = train_processed_df['details.xCoord'].shift(1)
train_processed_df['prev_y'] = train_processed_df['details.yCoord'].shift(1)
train_processed_df['prev_shot_angle'] = train_processed_df['shot_angle_deg'].shift(1)
train_processed_df['seconds_since_prev'] = train_processed_df['game_seconds'] - train_processed_df['game_seconds'].shift(1)
train_processed_df['distance_to_prev'] = (train_processed_df['distance_from_net_ft'] - train_processed_df['distance_from_net_ft'].shift(1)).abs().round(2)

SHOT_EVENTS = ['shot-on-goal', 'blocked-shot', 'missed-shot']
train_processed_df['rebound'] = train_processed_df['prev_event'].isin(SHOT_EVENTS)

def calculate_prev_net_properties(row):

    left_net_x = -89.0
    right_net_x = 89.0

    x = row.get("prev_x")
    y = row.get("prev_y")
    direction = row.get("attackingDirection")

    if pd.isna(x) or pd.isna(y):
        return np.nan, np.nan

    if direction == "right":
        net_x = right_net_x
    elif direction == "left":
        net_x = left_net_x
    else:
        net_x = right_net_x if x >= 0 else left_net_x

    net_y = 0
    dx = abs(net_x - x)
    dy = net_y - y
    distance = np.sqrt(dx**2 + dy**2)

    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    return distance, angle_deg


train_processed_df[["prev_distance_from_net", "prev_shot_angle"]] = train_processed_df.apply(calculate_prev_net_properties,
                                                                                             axis=1, 
                                                                                             result_type="expand").round(2)

train_processed_df['shot_angle_change'] = np.where(train_processed_df['rebound'],
                                                   np.abs(train_processed_df['shot_angle_deg'] - train_processed_df['prev_shot_angle']),
                                                   np.nan).round(2)                                                   

train_processed_df['speed_from_prev'] = (train_processed_df['distance_to_prev'] / train_processed_df['seconds_since_prev']).round(2)
df_wpg_wsh = train_processed_df[train_processed_df['game_id'] == 2017021065].copy()

train_processed_df.to_csv("train_engineered_df.csv", index=False)