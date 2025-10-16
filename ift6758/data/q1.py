import os
import json
import time
import requests
from typing import Optional, Iterator


class DataFetcher:
    """
    A class for acessing and downloading NHL play-by-play data from the official 
    NHL Stats API for one or multiple seasons. Play-by-play data is retrieved for 
    each game per-season and cached locally as JSON files. Subsequent calls reuse 
    cached data when possible.
    """
    
    def __init__(self):
        """
        Initialize the play-by-play data fetcher. Creates a local 'data/' directory 
        for storing downloaded json files and sets up the base URL for API requests. 
        """
        
        # Create a 'data' folder in the current working dir
        self.data_path = os.path.join(os.getcwd(), 'data')
        os.makedirs(self.data_path, exist_ok=True)
        
        self.season = None
        self.base_url = "https://api-web.nhle.com"
        
        

    def load_season_data(self, season: int) -> list[dict]:
        """
        Downloads every play-by-play data for each game in a given season and saves 
        it locally to the data/ directory. If a cached JSON file already exists for 
        the season, it is loaded instead of re-downloading.

        Args:
            - season (int): The starting year of the NHL season (e.g., 2022 for the 2022 - 2023 season)

        Returns:
            - list[dict]: A list of play-by-play game data dictionaries, one per game.
        """
        self.season = season 
        print(f"Downloading data for the {self.season}-{self.season + 1} season...")        

        file_path = os.path.join(self.data_path, f"playbyplay-{self.season}.json" )
        if not os.path.exists(file_path):
            
            season_data = []

            for game_id in self._generate_game_ids():
                data = self._download_game_data(game_id)
                if data:
                    season_data.append(data)

            with open(file_path, 'w') as f:
                json.dump(season_data, f, indent=2)
            print(f"Downloaded data for {self.season}-{self.season + 1}.")
        
        else:
            with open(file_path, 'r') as f:
                season_data = json.load(f)
                print(f"Loaded cached data for {self.season}-{self.season + 1}.")
             
        return season_data
    

    def _generate_game_ids(self) -> Iterator[str]:
        """
        Generates all expected game IDs for the NHL season (regular season and playoffs)

        The NHL uses a fixed game ID structure: `YYYYTTGGGG`, where:
            - `YYYY` is the season start year (e.g., 2022 for the 2022-2023 season)
            - `TT` is the game type (02 = regular season, 03 = playoffs)
            - `GGGG` is the sequential game number (i.e., 0001 -> 9999)

        Returns :
            - Iterator[str]: Game IDs for regular season and playoff games in a generator.
        """

        for i in range(1, 1350):
            yield f"{self.season}02{i:04d}"

        if self.season == 2019:
            print("Adding qualification matches for the 2019-2020 season...")
            for i in range(1, 70):
                 yield f"{self.season}0300{i:02d}"

        for matchup in range(1, 8 + 1):
            for game in range(1, 7 + 1):
                yield f"{self.season}0301{matchup}{game}"

        for matchup in range(1, 4 + 1):
            for game in range(1, 7 + 1):
                yield f"{self.season}0302{matchup}{game}"

        for matchup in range(1, 2 + 1):
            for game in range(1, 7 + 1):
                yield f"{self.season}0303{matchup}{game}"

        for game in range(1, 7 + 1):
            yield f"{self.season}03041{game}"


    def _download_game_data(self, game_id: str) -> Optional[dict]:
        """
        Downloads play-by-play data for a single game from the NHL API.

        This method attempts to download the data up to 3 times to handle
        temporary network issues or API errors. A short delay is
        added between requests to be respectful of the server.

        Args:
            - game_id (str): The unique 10-digit identifier of the game to download.

        Returns:
            - Optional[dict]: A dictionary containing the raw JSON data for the game if
              the download is successful. Returns None if the request
              fails after all attempts or if the game ID is invalid.

        """
        
        url = f"{self.base_url}/v1/gamecenter/{game_id}/play-by-play"
        
        for attempt in range(3):
            try:
                time.sleep(0.2)
                print(f"Downloading data for game ID: {game_id}...")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            
            except requests.RequestException:
                if attempt < 2:
                    time.sleep(1)
                else:
                    print(f"Error downloading data for game ID: {game_id}...")
                    print(f"Skipping {game_id}...")
                    return None
    

if __name__ == '__main__':
    
    fetcher = DataFetcher()
    for year in range(2016, 2024):
        season_data = fetcher.load_season_data(year)
        print("-" * 25)
