import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from traffic_generator import Scenario


ORDERED_TYPES: List[str] = ["Low", "Medium", "High", "Wave", "Unbalanced"]

SCENARIO_BY_LABEL: Dict[str, Scenario] = {
    "Low": Scenario.LOW,
    "Medium": Scenario.MEDIUM,
    "High": Scenario.HIGH,
    "Wave": Scenario.WAVE,
    "Unbalanced": Scenario.UNBALANCED,
}

SCENARIO_COUNT_BOUNDS: Dict[Scenario, Tuple[int, int]] = {
    Scenario.LOW: (250, 1000),
    Scenario.MEDIUM: (1000, 1700),
    Scenario.HIGH: (1700, 2400),
    Scenario.WAVE: (1600, 2250),
    Scenario.UNBALANCED: (1600, 2250),
}

DEFAULT_START_SEEDS: Dict[str, int] = {
    "Low": 10000,
    "Medium": 20000,
    "High": 30000,
    "Wave": 40000,
    "Unbalanced": 50000,
}

def equidistributed_vehicle_counts(scenario, n):
    lo, hi = SCENARIO_COUNT_BOUNDS[scenario]
    if n <= 0:
        return []
    if n == 1:
        return [int((lo + hi) // 2)]

    if n > (hi - lo + 1):
        raise ValueError(
            f"Cannot generate {n} unique integers in [{lo}, {hi}] "
        )

    vals = np.rint(np.linspace(lo, hi, n)).astype(int).tolist() # generate n equidistributed values between lo and hi, inclusive
    return vals


@dataclass
class EpisodeBank:
    episodes_by_type: Dict[str, List[int]]
    metadata: Dict[int, Tuple[str, int]] # episode_id -> (scenario_label, vehicle_count)
    bank_config: Dict[str, object] 

    @property
    def all_episodes(self):
        """ Return all episode IDs ordered by type according to ORDERED_TYPES. """
        ordered: List[int] = []
        for t in ORDERED_TYPES:
            ordered.extend(self.episodes_by_type.get(t, []))
        return ordered

    def to_jsonable(self):
        """ Convert to a JSON-serializable structure. """
        return {
            "episodes_by_type": self.episodes_by_type,
            "metadata": {str(k): [v[0], int(v[1])] for k, v in self.metadata.items()},
            "bank_config": self.bank_config,
        }

    @classmethod
    def from_jsonable(cls, payload: Dict):
        """ Create an EpisodeBank instance from a JSON-deserialized structure. """
        return cls(
            episodes_by_type={k: list(v) for k, v in payload["episodes_by_type"].items()},
            metadata={int(k): (v[0], int(v[1])) for k, v in payload["metadata"].items()},
            bank_config=dict(payload.get("bank_config", {})),
        )


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_episode_bank(episodes_per_type, start_seeds = None):
    """Create a deterministic bank with explicit scenario and equidistributed vehicle counts"""
    start_seeds = start_seeds or DEFAULT_START_SEEDS

    episodes_by_type: Dict[str, List[int]] = {}
    metadata: Dict[int, Tuple[str, int]] = {}

    for label in ORDERED_TYPES:
        scenario = SCENARIO_BY_LABEL[label]
        start = int(start_seeds[label])
        stop = start + int(episodes_per_type)
        seeds = list(range(start, stop))
        counts = equidistributed_vehicle_counts(scenario, int(episodes_per_type))

        episodes_by_type[label] = seeds
        for ep_id, n_veh in zip(seeds, counts):
            metadata[int(ep_id)] = (label, int(n_veh))

    return EpisodeBank(
        episodes_by_type=episodes_by_type,
        metadata=metadata,
        bank_config={
            "episodes_per_type": int(episodes_per_type),
            "start_seeds": dict(start_seeds),
        },
    )


def ensure_episode_bank(path, episodes_per_type, start_seeds = None):
    """Load bank from disk if compatible, otherwise rebuild it."""
    start_seeds = start_seeds or DEFAULT_START_SEEDS

    if os.path.exists(path):
        bank = EpisodeBank.from_jsonable(load_json(path))
        cfg = bank.bank_config or {}
        if (
            int(cfg.get("episodes_per_type", -1)) == int(episodes_per_type)
            and dict(cfg.get("start_seeds", {})) == dict(start_seeds)
            and all(len(bank.episodes_by_type.get(t, [])) == int(episodes_per_type) for t in ORDERED_TYPES)
        ):
            return bank

    bank = create_episode_bank(episodes_per_type=episodes_per_type, start_seeds=start_seeds)
    save_json(path, bank.to_jsonable())
    return bank
