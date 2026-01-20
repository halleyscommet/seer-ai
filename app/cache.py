"""
Data caching module for persisting TBA event data to JSON files.
This allows the app to recover from crashes without re-downloading data.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import asdict, dataclass
from .tba_downloader import Match, MatchAlliance


class CacheManager:
    """Manages caching of TBA event data to JSON files."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files. Defaults to ./cache/
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, event_key: str) -> Path:
        """Get the cache file path for an event."""
        # Sanitize event key to be filename-safe
        safe_key = event_key.replace('/', '_').replace('\\', '_')
        return self.cache_dir / f"{safe_key}.json"
    
    def save_event_data(self, event_key: str, event_data: Dict[str, Any]) -> None:
        """
        Save event data to cache file.
        
        Args:
            event_key: The event key (e.g., '2024week0')
            event_data: Dictionary containing teams and matches data
        """
        cache_path = self._get_cache_path(event_key)
        
        # Convert Match objects to serializable format
        serializable_data = {
            'event_key': event_data['event_key'],
            'teams': event_data['teams'],
            'num_teams': event_data['num_teams'],
            'num_qual_matches': event_data['num_qual_matches'],
            'cached_at': datetime.now().isoformat(),
            'matches': [
                {
                    'key': match.key,
                    'match_number': match.match_number,
                    'red_alliance': {
                        'teams': match.red_alliance.teams
                    },
                    'blue_alliance': {
                        'teams': match.blue_alliance.teams
                    }
                }
                for match in event_data['matches']
            ]
        }
        
        with open(cache_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    def load_event_data(self, event_key: str) -> Optional[Dict[str, Any]]:
        """
        Load event data from cache file if it exists.
        
        Args:
            event_key: The event key (e.g., '2024week0')
            
        Returns:
            Dictionary containing event data, or None if not cached
        """
        cache_path = self._get_cache_path(event_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct Match objects from cached data
            matches = []
            for match_data in data.get('matches', []):
                red_alliance = MatchAlliance(
                    teams=match_data['red_alliance']['teams']
                )
                blue_alliance = MatchAlliance(
                    teams=match_data['blue_alliance']['teams']
                )
                match = Match(
                    key=match_data['key'],
                    match_number=match_data['match_number'],
                    red_alliance=red_alliance,
                    blue_alliance=blue_alliance
                )
                matches.append(match)
            
            return {
                'event_key': data['event_key'],
                'teams': data['teams'],
                'matches': matches,
                'num_teams': data['num_teams'],
                'num_qual_matches': data['num_qual_matches'],
                'cached_at': data.get('cached_at')
            }
        except Exception as e:
            print(f"Error loading cache for {event_key}: {e}")
            return None
    
    def cache_exists(self, event_key: str) -> bool:
        """Check if cache exists for an event."""
        return self._get_cache_path(event_key).exists()
    
    def get_cache_timestamp(self, event_key: str) -> Optional[str]:
        """Get the timestamp when data was cached."""
        cache_path = self._get_cache_path(event_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            return data.get('cached_at')
        except Exception:
            return None
    
    def clear_cache(self, event_key: Optional[str] = None) -> None:
        """
        Clear cache for specific event or all events.
        
        Args:
            event_key: Specific event to clear, or None to clear all
        """
        if event_key:
            cache_path = self._get_cache_path(event_key)
            if cache_path.exists():
                cache_path.unlink()
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
