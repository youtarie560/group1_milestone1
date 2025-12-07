import logging
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

class FrontEndData:
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.base_url = f"https://api-web.nhle.com/v1/gamecenter/{self.game_id}/play-by-play"

        self.last_event_id = -1

    def ping_game(self) -> pd.DataFrame:
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


        plays['home'] = raw_data['homeTeam']['commonName']['default']
        plays['away'] = raw_data['awayTeam']['commonName']['default']

        plays['homeGoals'] = raw_data['homeTeam']['score']
        plays['awayGoals'] = raw_data['awayTeam']['score']

        id_home = raw_data['homeTeam']['id']
        id_away = raw_data['awayTeam']['id']

        plays['team'] = plays['teamId'].apply(
            lambda id: "away" if id == id_away else "home" if id == id_home else None)

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

        # is goal logic
        plays['is_goal'] = (plays['typeDescKey'] == 'goal').astype(int)

        final_cols = [
            'gameId', 'eventId','teamId', 'period', 'timeInPeriod',
            'distanceToNet', 'shotAngle', 'is_goal', 'is_empty_net','team','home','away','homeGoals','awayGoals'
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

