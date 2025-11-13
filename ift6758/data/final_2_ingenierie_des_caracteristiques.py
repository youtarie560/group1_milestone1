import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
#test_df.to_csv("test_data.csv", index=False)

def normalize_game_plays(game_row: pd.Series) -> pd.DataFrame:

    plays = game_row.get("plays", [])
    
    if not isinstance(plays, list) or len(plays) == 0:
        return pd.DataFrame()

    return pd.json_normalize(plays)


def filter_shot_plays(plays: pd.DataFrame) -> pd.DataFrame:

    if plays.empty:
        return pd.DataFrame()
    
    SHOT_TYPES = {"shot-on-goal", "missed-shot", "blocked-shot", "goal"}
    return plays[plays["typeDescKey"].isin(SHOT_TYPES)].copy()


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

        plays = filter_shot_plays(plays)
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
        
        plays['game_id'] = game_id
        plays['home_id'] = home_id
        plays['away_id'] = away_id
        
        all_plays.append(plays[["distance_from_net_ft", "shot_angle_deg", "is_goal", "empty_net"]])

    if not all_plays:
        return pd.DataFrame(columns=["distance_from_net_ft", "shot_angle_deg", "is_goal", "empty_net"])

    final = pd.concat(all_plays, ignore_index=True)
    final.dropna(inplace=True)
    return final

train_processed_df = extract_shot_features(train_df)
train_processed_df.to_csv("train_processed_df.csv", index=False)
viz_df = train_processed_df.copy()
viz_df['Shot Result'] = train_processed_df['is_goal'].replace({1:'Goal', 0:'No Goal'})


sns.set_style("whitegrid")
plt.figure(figsize=(14, 6))

custom_palette = {'Goal': '#1a80bb', 
                  'No Goal': '#f1A226'} 

sns.histplot(data=viz_df,
             x='distance_from_net_ft',
             hue='Shot Result',
             multiple='stack',
             bins=np.arange(0, 201, 5),
             palette=custom_palette) 

plt.xlabel('Distance from the Net (ft)', fontsize=14)
plt.ylabel('Shot Count', fontsize=14)
plt.xticks(range(0, 201, 10))

ax = plt.gca() 
#ax.legend_.set_title('Shot Result')
# plt.savefig('milestone2-figure1.png', dpi=300, bbox_inches='tight')
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(14, 6))

custom_palette = {'Goal': '#1a80bb', 
                  'No Goal': '#f1A226'} 

sns.histplot(data=viz_df,
             x='shot_angle_deg',
             hue='Shot Result',
             multiple='stack',
             bins=np.arange(-90, 91, 5),
             palette=custom_palette)

plt.xlabel('Shot Angle (°)', fontsize=14)
plt.ylabel('Shot Count', fontsize=14)
plt.xticks(range(-90, 91, 15))
plt.grid(axis='y', alpha=0.5)

ax = plt.gca() 
# plt.savefig('figures/milestone2-figure2.png', dpi=300, bbox_inches='tight')
plt.show()



sns.set_theme(style="whitegrid")

x_bins = np.arange(0, 201, 5)     
y_bins = np.arange(-90, 91, 5)

g = sns.jointplot(data=viz_df,
                  x='distance_from_net_ft',
                  y='shot_angle_deg',
                  kind='hist',
                  bins=(x_bins, y_bins),
                  height=8,
                  cmap="BuPu")

ax = g.ax_joint

ax.set_xlabel('Distance from net (ft)')
ax.set_ylabel('Shot Angle (°)')

ax.set_xticks(range(0, 201, 10))
ax.set_yticks(range(-90, 91, 15))

ax.grid(True) 
g.ax_marg_x.grid(False)
g.ax_marg_y.grid(False)

plt.show()

BIN_SIZE = 3
distance_bins = np.arange(0, viz_df['distance_from_net_ft'].max() + BIN_SIZE, BIN_SIZE) 
viz_df['distance_bin'] = pd.cut(viz_df['distance_from_net_ft'], bins=distance_bins, include_lowest=True)

distance_summary = viz_df.groupby('distance_bin', observed=True)['is_goal'].agg(
    total_shots='count',
    goals='sum'
).reset_index()

distance_summary['goal_rate'] = distance_summary['goals'] / distance_summary['total_shots']

plt.figure(figsize=(14, 7))

distance_summary['mid_distance'] = distance_summary['distance_bin'].apply(lambda x: x.mid)

sns.lineplot(
    data=distance_summary, 
    x='mid_distance', 
    y='goal_rate', 
    marker='o', 
    linestyle='-', 
    color='darkblue'
)
plt.xlabel('Distance to the Net (ft)')
plt.ylabel('Goal Conversion Rate\n (Number of Goals / Number of Total Shots)')
plt.ylim(0, distance_summary['goal_rate'].max() * 1.1)
plt.grid(axis='y', alpha=0.5)
#plt.savefig('figures/milestone2-figure4.png', dpi=300, bbox_inches='tight')

BIN_SIZE = 3
angle_bins = np.arange(-90, 95, BIN_SIZE)
viz_df['angle_bin'] = pd.cut(viz_df['shot_angle_deg'], bins=angle_bins, include_lowest=True)

angle_summary = viz_df.groupby('angle_bin', observed=True)['is_goal'].agg(
    total_shots='count',
    goals='sum'
).reset_index()

angle_summary['goal_rate'] = angle_summary['goals'] / angle_summary['total_shots']
plt.figure(figsize=(14, 7))

angle_summary['mid_angle'] = angle_summary['angle_bin'].apply(lambda x: x.mid)

sns.lineplot(
    data=angle_summary, 
    x='mid_angle', 
    y='goal_rate', 
    marker='o', 
    linestyle='-', 
    color='darkgreen')

plt.xlabel('Shot Angle (°)')
plt.ylabel('Goal Conversion Rate\n (Number of Goals / Number of Total Shots)')
plt.xticks(range(-90, 91, 15))
plt.grid(axis='y', alpha=0.5)
#plt.savefig('figures/milestone2-figure5.png', dpi=300, bbox_inches='tight')

viz_df['Net_Status'] = viz_df['empty_net'].replace({1: 'Empty Net', 0: 'Non-Empty Net'})
goals_df = viz_df[viz_df['is_goal'] == 1].copy() 

plt.figure(figsize=(10, 6))
sns.histplot(
    data=goals_df, 
    x='distance_from_net_ft', 
    hue='Net_Status', 
    multiple='stack', 
    bins=np.arange(0, 201, 5),
)


plt.title('Distribution of Empty/Filled Net Goals by Distance')
plt.xlabel('Distance to the Net (ft)')
plt.ylabel('Number of Goals')
plt.xticks(range(0, 201, 10), fontsize=8)
plt.grid(axis='y', alpha=0.5)
plt.show()

sns.set_style("whitegrid")

viz_df['Net_Status'] = viz_df['empty_net'].replace({1: 'Empty Net', 0: 'Non-Empty Net'})
goals_df = viz_df[viz_df['is_goal'] == 1].copy()

custom_net_palette = {
    'Non-Empty Net': '#1a80bb', 
    'Empty Net': '#f1A226'
} 

plt.figure(figsize=(14, 6))

sns.histplot(data=goals_df, 
             x='distance_from_net_ft', 
             hue='Net_Status', 
             multiple='stack', 
             bins=np.arange(0, 201, 5),
             palette=custom_net_palette)
plt.yscale('log')
plt.xlabel('Distance to the Net (ft)', fontsize=14)
plt.ylabel('Number of Goals', fontsize=14)
plt.xticks(range(0, 201, 10))


ax = plt.gca() 
ax.legend_.set_title('Net Status')

# plt.savefig('figures/milestone2-figure6.png', dpi=300, bbox_inches='tight')

HALF_RINK = 100
NEUTRAL_ZONE = 50
NET_TO_FRIENDLY_BLUE_LINE = 64
NET_TO_END = 11

NET_TO_OPPONENT_BLUE_LINE = HALF_RINK - NET_TO_END + (NEUTRAL_ZONE / 2)
NET_TO_HALF_RINK = 100 - NET_TO_END

