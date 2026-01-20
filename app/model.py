from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TeamStats:
    team: int
    fuel_made: int = 0
    fuel_missed: int = 0
    climb_low: int = 0
    climb_mid: int = 0
    climb_high: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)

    def log_event(self, event_type: str, meta: Optional[dict] = None) -> None:
        self.events.append(
            {"ts": time.time(), "event": event_type, "meta": meta or {}}
        )


class SessionStore:
    """
    Session-scoped state. Your vision system will call `record_event(team, event_type, meta)`.
    """
    def __init__(self) -> None:
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.match_label = "UNSET"
        self.started_at: Optional[float] = None
        self.teams: Dict[int, TeamStats] = {}

    def reset(self) -> None:
        self.__init__()

    def set_match_label(self, label: str) -> None:
        label = label.strip()
        self.match_label = label if label else "UNSET"

    def start(self) -> None:
        if self.started_at is None:
            self.started_at = time.time()

    def ensure_team(self, team: int) -> TeamStats:
        if team not in self.teams:
            self.teams[team] = TeamStats(team=team)
        return self.teams[team]

    def record_event(self, team: int, event_type: str, meta: Optional[dict] = None) -> None:
        ts = self.ensure_team(team)

        if event_type == "fuel_made":
            ts.fuel_made += 1
        elif event_type == "fuel_missed":
            ts.fuel_missed += 1
        elif event_type == "climb_low":
            ts.climb_low += 1
        elif event_type == "climb_mid":
            ts.climb_mid += 1
        elif event_type == "climb_high":
            ts.climb_high += 1

        ts.log_event(event_type, meta=meta)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "match_label": self.match_label,
            "started_at": self.started_at,
            "exported_at": time.time(),
            "teams": {str(k): asdict(v) for k, v in self.teams.items()},
        }

    def export_json(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def export_csv_summary(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["team", "fuel_made", "fuel_missed", "climb_low", "climb_mid", "climb_high"])
            for team in sorted(self.teams.keys()):
                s = self.teams[team]
                w.writerow([s.team, s.fuel_made, s.fuel_missed, s.climb_low, s.climb_mid, s.climb_high])

