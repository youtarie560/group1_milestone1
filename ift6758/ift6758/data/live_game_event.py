import requests
import pandas as pd
import numpy as np
import time
import logging
from typing import Optional, Tuple, Dict, Any


# perhaps replace prints with logs
# We don't know who scored the goal
# we don't know who one
# game_client.py
logger = logging.getLogger(__name__)
class LiveGameClient:

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.base_url = f"https://api-web.nhle.com/v1/gamecenter/{self.game_id}/play-by-play"

        self.last_event_id = -1

    def ping_game(self) -> pd.DataFrame:
        """
        Downloads the game data, processes it, and returns ONLY the events
        that have not been seen yet.
        """
        try:
            response = requests.get(self.base_url, timeout=10)
            if response.status_code != 200:
                logger.error(f"Game {self.game_id} not found.")
                return pd.DataFrame()

            data = response.json()

            full_df = self._process_events(data)

            if full_df.empty:
                return pd.DataFrame()

            new_events = full_df[full_df['eventId'] > self.last_event_id].copy()

            if not new_events.empty:
                self.last_event_id = new_events['eventId'].max()
                logger.info(f"New events found! processed up to ID: {self.last_event_id}")

            return new_events

        except requests.RequestException as e:
            logger.error(f"Error downloading data for game {self.game_id}: {e}")
            return pd.DataFrame()

    def _process_events(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Internal pipeline to clean and feature-engineer the events.
        """
        if 'plays' not in raw_data:
            return pd.DataFrame()

        plays = pd.json_normalize(raw_data['plays'])

        # keep 'missed-shots' temporarily to help calculate attacking direction
        valid_events = ['shot-on-goal', 'goal', 'missed-shot']
        plays = plays[plays['typeDescKey'].isin(valid_events)].copy()

        if plays.empty:
            return pd.DataFrame()

        plays['gameId'] = self.game_id
        plays['period'] = plays['periodDescriptor.number']
        plays['teamId'] = plays['details.eventOwnerTeamId']
        plays['x'] = plays['details.xCoord']
        plays['y'] = plays['details.yCoord']

        plays = plays.dropna(subset=['x', 'y'])

        direction_map = (
            plays.groupby(['period', 'teamId'])['x']
            .mean()
            .apply(lambda x: 'left' if x < 0 else 'right')
            .to_dict()
        )

        plays['attackingDirection'] = plays.apply(
            lambda row: direction_map.get((row['period'], row['teamId']), 'right'), axis=1
        )

        # Distance + Angle Calculation
        net_props = plays.apply(self._calculate_net_properties, axis=1, result_type='expand')
        plays[['distanceToNet', 'shotAngle']] = net_props

        # Empty Net Logic
        plays['is_empty_net'] = self._determine_empty_net(plays, raw_data)

        #is goal logic
        plays['is_goal'] = (plays['typeDescKey'] == 'goal').astype(int)

        final_cols = [
            'gameId', 'eventId', 'period', 'timeInPeriod',
            'distanceToNet', 'shotAngle', 'is_goal', 'is_empty_net'
        ]

        return plays[plays['typeDescKey'].isin(['shot-on-goal', 'goal'])][final_cols]

    @staticmethod
    def _calculate_net_properties(row) -> Tuple[float, float]:
        """
        Calculates distance and angle using user-specified geometry logic.
        """
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
        dx = abs(net_x - x)
        dy = net_y - y
        distance = np.sqrt(dx ** 2 + dy ** 2)

        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        return distance, angle_deg

    @staticmethod
    def _determine_empty_net(plays_df: pd.DataFrame, game_dict: Dict[str, Any]) -> np.ndarray:
        """
        Vectorized empty net calculation using situationCode
        """
        # Get Team IDs
        home_team_id = game_dict.get('homeTeam', {}).get('id')
        away_team_id = game_dict.get('awayTeam', {}).get('id')

        # "1540" -> AwayGoalie(idx 0), HomeGoalie(idx 3)
        sit_code = plays_df['situationCode'].astype(str)
        away_goalie_in = sit_code.str[0].astype(int)
        home_goalie_in = sit_code.str[3].astype(int)

        cond_home_shooter = (plays_df['teamId'] == home_team_id) & (away_goalie_in == 0)
        cond_away_shooter = (plays_df['teamId'] == away_team_id) & (home_goalie_in == 0)

        return np.where(cond_home_shooter | cond_away_shooter, 1, 0)



if __name__ == '__main__':
    client = LiveGameClient("2016020411")
    client2 = LiveGameClient("2016030411")
    df_for_model = client.ping_game()
    df_for_model2 = client2.ping_game()
    # df_for_model.to_csv("df_for_model.csv", index=False)

    print(df_for_model.head())
    print(df_for_model2.head())
    del df_for_model
    del df_for_model2

    game_id = "2022030411"
    client = LiveGameClient(game_id)
    print("--- 1st Ping (Should get all events) ---")
    df_batch_1 = client.ping_game()
    print(f"Batch 1 shape: {df_batch_1.shape}")

    if not df_batch_1.empty:
        print(f"Max Event ID seen: {client.last_event_id}")

    print("\n--- 2nd Ping (Should be empty) ---")
    df_batch_2 = client.ping_game()
    print(f"Batch 2 shape: {df_batch_2.shape}")

    print("\n--- Resetting Tracker manually ---")
    client.last_event_id = 0
    df_batch_3 = client.ping_game()
    print(f"Batch 3 shape (should be full again): {df_batch_3.shape}")
