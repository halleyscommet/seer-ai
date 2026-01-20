import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    """Configuration class for accessing environment variables."""
    
    TBA_API_KEY = os.getenv('TBA_API_KEY')
    TBA_EVENT_KEY = os.getenv('TBA_EVENT_KEY', '2026mimil')
    
    @classmethod
    def validate(cls):
        """Validate that all required environment variables are set."""
        if not cls.TBA_API_KEY or cls.TBA_API_KEY == 'your_tba_api_key_here':
            raise ValueError(
                "TBA_API_KEY is not set or is using the default placeholder value. "
                "Please set your The Blue Alliance API key in the .env file."
            )
    
    @classmethod
    def get_tba_key(cls):
        """Get the TBA API key, with validation."""
        if not cls.TBA_API_KEY or cls.TBA_API_KEY == 'your_tba_api_key_here':
            raise ValueError("TBA_API_KEY not configured. Please update your .env file.")
        return cls.TBA_API_KEY
    
    @classmethod
    def get_event_key(cls):
        """Get the event key."""
        return cls.TBA_EVENT_KEY

# Optionally auto-validate on import (comment out if you want manual validation)
Config.validate()
