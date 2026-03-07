from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import optuna
from experiment_common import ControllerSpec


DEFAULT_STL_PARAMS = {"J": 100.0, "K": 1.0, "Ke": 5.0}

# Improvements lists
STL2_IMPROVEMENTS = [2]
STL12_CAND_IMPROVEMENTS = [1, 2]


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    baseline: ControllerSpec
    candidate: ControllerSpec
    baseline_module_defaults: Dict[str, float]
    candidate_module_defaults: Dict[str, float]

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, float]:
        raise NotImplementedError


@dataclass(frozen=True)
class STL2Spec(AlgoSpec):
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, float]:
        return {
            "J": float(trial.suggest_float("J", 10.0, 500.0, log=True)),
            "K": float(trial.suggest_float("K", 0.2, 5.0, log=True)),
        }


@dataclass(frozen=True)
class STL12Spec(AlgoSpec):
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, float]:
        return {
            "J": float(trial.suggest_float("J", 10.0, 500.0, log=True)),
            "K": float(trial.suggest_float("K", 0.2, 5.0, log=True)),
            "Ke": float(trial.suggest_float("Ke", 1.0, 12.0, log=True)),
        }


PHASE1_RANGES: Dict[str, Tuple[float, float]] = {
    "W_IDLE": (80.0, 180.0),
    "W_BRAKE": (0.8, 5.0),
    "W_DRAG": (0.015, 0.06),
    "RED_PRESSURE_GAIN": (4.0, 18.0),
    "W_PLATOON_APPROACH": (20.0, 70.0),
    "PLATOON_BONUS_DECAY": (2.0, 8.0),
    "PLATOON_TTI": (5.0, 9.0),
}

PHASE2_RANGES: Dict[str, Tuple[float, float]] = {
    "WAVE_LOOKAHEAD_DIST": (100.0, 150.0),
    "WAVE_HOLD_TTI": (9.0, 13.0),
    "WAVE_HOLD_MIN_VEH": (3.0, 6.0),
    "WAVE_HOLD_SOFT_MAX": (16.0, 24.0),
    "WAVE_HOLD_RED_TIME_LIMIT": (20.0, 45.0),
    "PLATOON_BONUS_FLOOR": (0.2, 0.5),
}


@dataclass(frozen=True)
class ImprovedSTLSpec(AlgoSpec):
    phase: int = 1

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, float]:
        ranges = PHASE1_RANGES if int(self.phase) == 1 else PHASE2_RANGES
        out = {}
        for key, (lo, hi) in ranges.items():
            if key == "WAVE_HOLD_MIN_VEH":
                out[key] = trial.suggest_int(key, int(lo), int(hi))
            else:
                out[key] = trial.suggest_float(key, lo, hi)
        return out


def build_stl2_spec() -> STL2Spec:
    baseline = ControllerSpec(module="smart_traffic_light", cls="SmartTrafficLight", improvements=list(STL2_IMPROVEMENTS))
    candidate = ControllerSpec(module="smart_traffic_light", cls="SmartTrafficLight", improvements=list(STL2_IMPROVEMENTS))
    return STL2Spec(
        name="stl2",
        baseline=baseline,
        candidate=candidate,
        baseline_module_defaults=dict(DEFAULT_STL_PARAMS),
        candidate_module_defaults=dict(DEFAULT_STL_PARAMS),
    )


def build_stl12_spec() -> STL12Spec:
    baseline = ControllerSpec(module="smart_traffic_light", cls="SmartTrafficLight", improvements=list(STL2_IMPROVEMENTS))
    candidate = ControllerSpec(module="smart_traffic_light", cls="SmartTrafficLight", improvements=list(STL12_CAND_IMPROVEMENTS))
    return STL12Spec(
        name="stl12",
        baseline=baseline,
        candidate=candidate,
        baseline_module_defaults=dict(DEFAULT_STL_PARAMS),
        candidate_module_defaults=dict(DEFAULT_STL_PARAMS),
    )


def _explicit_improved_stl_defaults() -> Dict[str, float]:
    from improved_smart_traffic_light import (
        # Tunable weights (Phase 1)
        W_IDLE,
        W_BRAKE,
        W_DRAG,
        RED_PRESSURE_GAIN,
        W_PLATOON_APPROACH,
        PLATOON_BONUS_DECAY,
        PLATOON_TTI,
        # Wave handling (Phase 2)
        WAVE_LOOKAHEAD_DIST,
        WAVE_HOLD_TTI,
        WAVE_HOLD_MIN_VEH,
        WAVE_HOLD_SOFT_MAX,
        WAVE_HOLD_RED_TIME_LIMIT,
        PLATOON_BONUS_FLOOR,
    )

    return {
        "W_IDLE": float(W_IDLE),
        "W_BRAKE": float(W_BRAKE),
        "W_DRAG": float(W_DRAG),
        "RED_PRESSURE_GAIN": float(RED_PRESSURE_GAIN),
        "W_PLATOON_APPROACH": float(W_PLATOON_APPROACH),
        "PLATOON_BONUS_DECAY": float(PLATOON_BONUS_DECAY),
        "PLATOON_TTI": float(PLATOON_TTI),
        "WAVE_LOOKAHEAD_DIST": float(WAVE_LOOKAHEAD_DIST),
        "WAVE_HOLD_TTI": float(WAVE_HOLD_TTI),
        "WAVE_HOLD_MIN_VEH": float(WAVE_HOLD_MIN_VEH),
        "WAVE_HOLD_SOFT_MAX": float(WAVE_HOLD_SOFT_MAX),
        "WAVE_HOLD_RED_TIME_LIMIT": float(WAVE_HOLD_RED_TIME_LIMIT),
        "PLATOON_BONUS_FLOOR": float(PLATOON_BONUS_FLOOR),
    }


def build_improved_stl_spec(*, phase: int, base_params: Optional[Dict[str, float]] = None) -> ImprovedSTLSpec:
    baseline = ControllerSpec(module="smart_traffic_light", cls="SmartTrafficLight", improvements=list(STL2_IMPROVEMENTS))
    candidate = ControllerSpec(module="improved_smart_traffic_light", cls="ImprovedSmartTrafficLight", improvements=list(STL2_IMPROVEMENTS))

    defaults = _explicit_improved_stl_defaults()
    if base_params: # for phase 2, we want to keep phase 1 params fixed and only tune the wave-related ones
        defaults.update({k: float(v) for k, v in base_params.items()})

    return ImprovedSTLSpec(
        name=f"improved_stl_p{int(phase)}",
        phase=int(phase),
        baseline=baseline,
        candidate=candidate,
        baseline_module_defaults=dict(DEFAULT_STL_PARAMS),
        candidate_module_defaults=defaults,
    )
