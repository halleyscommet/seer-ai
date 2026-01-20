"""
The Blue Alliance API Downloader
Downloads event data including matches and team information.
"""
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MatchAlliance:
    """Represents one alliance (red or blue) in a match."""
    teams: List[int]  # Team numbers
    
    @classmethod
    def from_api_data(cls, api_data: Dict[str, Any]) -> 'MatchAlliance':
        """Parse alliance data from TBA API response."""
        team_keys = api_data.get('team_keys', [])
        # Convert 'frc3707' to 3707
        teams = [int(key.replace('frc', '')) for key in team_keys]
        return cls(teams=teams)


@dataclass
class Match:
    """Represents a qualification match."""
    key: str  # e.g., '2024week0_qm1'
    match_number: int
    red_alliance: MatchAlliance
    blue_alliance: MatchAlliance
    
    def __str__(self):
        return f"Qual {self.match_number}: Red {self.red_alliance.teams} vs Blue {self.blue_alliance.teams}"


class TBADownloader:
    """Downloads event data from The Blue Alliance API."""
    
    BASE_URL = "https://www.thebluealliance.com/api/v3"
    
    def __init__(self, api_key: str):
        """
        Initialize the TBA downloader.
        
        Args:
            api_key: The Blue Alliance API key
        """
        self.api_key = api_key
        self.headers = {
            'X-TBA-Auth-Key': api_key,
            'Accept': 'application/json'
        }
    
    def _get(self, endpoint: str) -> Any:
        """
        Make a GET request to the TBA API.
        
        Args:
            endpoint: API endpoint (without base URL)
            
        Returns:
            Parsed JSON response
            
        Raises:
            requests.RequestException: If the request fails
        """
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_event_teams(self, event_key: str) -> List[int]:
        """
        Get all team numbers at an event.
        
        Args:
            event_key: Event key (e.g., '2024week0')
            
        Returns:
            List of team numbers sorted numerically
        """
        endpoint = f"/event/{event_key}/teams/simple"
        teams_data = self._get(endpoint)
        
        # Extract team numbers from team keys
        team_numbers = [int(team['key'].replace('frc', '')) for team in teams_data]
        return sorted(team_numbers)
    
    def get_qual_matches(self, event_key: str) -> List[Match]:
        """
        Get all qualification matches for an event.
        
        Args:
            event_key: Event key (e.g., '2024week0')
            
        Returns:
            List of Match objects sorted by match number
        """
        endpoint = f"/event/{event_key}/matches/simple"
        matches_data = self._get(endpoint)
        
        # Filter for qualification matches only
        qual_matches = []
        for match_data in matches_data:
            if match_data['comp_level'] != 'qm':
                continue
            
            # Parse alliances
            alliances = match_data.get('alliances', {})
            red_alliance = MatchAlliance.from_api_data(alliances.get('red', {}))
            blue_alliance = MatchAlliance.from_api_data(alliances.get('blue', {}))
            
            match = Match(
                key=match_data['key'],
                match_number=match_data['match_number'],
                red_alliance=red_alliance,
                blue_alliance=blue_alliance
            )
            qual_matches.append(match)
        
        # Sort by match number
        qual_matches.sort(key=lambda m: m.match_number)
        return qual_matches
    
    def get_event_data(self, event_key: str) -> Dict[str, Any]:
        """
        Get comprehensive event data including teams and matches.
        
        Args:
            event_key: Event key (e.g., '2024week0')
            
        Returns:
            Dictionary with 'teams' (list of team numbers) and 'matches' (list of Match objects)
        """
        teams = self.get_event_teams(event_key)
        matches = self.get_qual_matches(event_key)
        
        return {
            'event_key': event_key,
            'teams': teams,
            'matches': matches,
            'num_teams': len(teams),
            'num_qual_matches': len(matches)
        }
    
    def print_event_summary(self, event_key: str) -> None:
        """Print a summary of event data (useful for testing)."""
        data = self.get_event_data(event_key)
        
        print(f"\n=== Event: {event_key} ===")
        print(f"Teams ({data['num_teams']}): {data['teams']}")
        print(f"\nQualification Matches ({data['num_qual_matches']}):")
        for match in data['matches']:
            print(f"  {match}")
