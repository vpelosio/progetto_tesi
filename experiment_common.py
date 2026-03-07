import concurrent.futures
import importlib
import multiprocessing
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from episode_bank import ORDERED_TYPES, SCENARIO_BY_LABEL
from sim_config import CONFIG_4WAY_160M
from sumo_sim import SumoSimulation


@dataclass(frozen=True)
class ControllerSpec:
    module: str  # e.g. "smart_traffic_light" or "improved_smart_traffic_light"
    cls: str  # e.g. "SmartTrafficLight" or "ImprovedSmartTrafficLight"
    improvements: List[int]


def improvement_pct(baseline_value, candidate_value):
    return ((baseline_value - candidate_value) / baseline_value) * 100.0 if baseline_value > 0 else 0.0


def _apply_module_params(module_name, default_params, overrides):
    """Apply parameters as module-level constants for the given module. Overrides take precedence over defaults."""
    mod = importlib.import_module(module_name)
    for k, v in (default_params or {}).items():
        if hasattr(mod, k):
            setattr(mod, k, float(v))
    if overrides:
        for k, v in overrides.items():
            if hasattr(mod, k):
                setattr(mod, k, float(v))


def _run_episode_worker(args):
    """Run a single episode and return metrics.

    args:
      (ep_id, label, controller_module, controller_cls, improvements,
       module_defaults, module_overrides,
       scenario_label, n_vehicles, log_dir)
    """

    (
        ep_id,
        label,
        controller_module,
        controller_cls,
        improvements,
        module_defaults,
        module_overrides,
        scenario_label,
        n_vehicles,
        log_dir,
    ) = args

    import libsumo

    sim = None
    try:
        # Apply parameters (module-level constants) before controller creation.
        _apply_module_params(controller_module, module_defaults, module_overrides)

        sim = SumoSimulation(
            libsumo=libsumo,
            sim_config=CONFIG_4WAY_160M,
            sim_step=0.5,
            episode_duration=3600,
            log_folder=log_dir,
            rank=f"{label}_{ep_id}",
            enable_measure=True,
        )

        scenario = SCENARIO_BY_LABEL[scenario_label]
        sim.initialize_episode(ep_id, scenario=scenario, n_vehicles=int(n_vehicles))

        mod = importlib.import_module(controller_module)
        tl_cls = getattr(mod, controller_cls)
        tl = tl_cls(sim.sim_config.tl_id, list(improvements))

        while (
            libsumo.simulation.getMinExpectedNumber() > 0
            and libsumo.simulation.getTime() < sim.episode_duration
        ):
            sim.simulation_step()
            tl.performStep()

        measures = sim.get_measures()
        vehicle_count = len(measures)
        total_co2 = sum(m["totalCO2Emissions"] for m in measures)

        return {
            "episode": int(ep_id),
            "co2": float(total_co2),
            "co2_avg": float(total_co2 / vehicle_count) if vehicle_count else 0.0,
            "travel_time": float(sum(m["totalTravelTime"] for m in measures)),
            "waiting_time": float(sum(m["totalWaitingTime"] for m in measures)),
        }

    except Exception as e:
        print(f"[{label}] error in episode {ep_id}: {e}")
        return None
    finally:
        if sim is not None:
            sim.close()


def evaluate_episodes(*, episodes, metadata, label, controller, default_params, params, log_dir, n_jobs = None):
    """Evaluate a controller on a list of episodes

    metadata must contain: episode -> (type_label, n_vehicles)
    """
    os.makedirs(log_dir, exist_ok=True)
    n_jobs = int(n_jobs or min(15, multiprocessing.cpu_count()))

    worker_args = []
    for ep_id in episodes:
        ep_type, n_vehs = metadata[ep_id]
        worker_args.append(
            (
                int(ep_id),
                str(label),
                controller.module,
                controller.cls,
                list(controller.improvements),
                dict(default_params or {}),
                dict(params) if params else None,
                str(ep_type),
                int(n_vehs),
                log_dir,
            )
        )

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as ex:
        results = list(ex.map(_run_episode_worker, worker_args))

    results = [r for r in results if r is not None]
    by_episode = {r["episode"]: r for r in results}
    return [by_episode[ep] for ep in episodes if ep in by_episode]


def summarize_results(results):
    return {
        "co2": float(sum(r["co2"] for r in results)),
        "co2_avg": float(np.mean([r["co2_avg"] for r in results])) if results else 0.0,
        "travel_time": float(sum(r["travel_time"] for r in results)),
        "waiting_time": float(sum(r["waiting_time"] for r in results)),
    }


def compute_category_gains(baseline_results, candidate_results, metadata):
    stats = defaultdict(lambda: {"baseline": [], "candidate": [], "diff": []}) #default struct for each episode type

    cand_by_ep = {r["episode"]: r for r in candidate_results}
    for base in baseline_results:
        ep = base["episode"]
        if ep not in cand_by_ep:
            continue
        cand = cand_by_ep[ep]
        ep_type, _ = metadata.get(ep, ("Unknown", 0)) # get episode type from metadata, default to "Unknown" if not found
        diff = improvement_pct(float(base["co2_avg"]), float(cand["co2_avg"]))
        stats[ep_type]["baseline"].append(float(base["co2_avg"]))
        stats[ep_type]["candidate"].append(float(cand["co2_avg"]))
        stats[ep_type]["diff"].append(float(diff))

    out = {}
    for ep_type, d in stats.items():
        out[ep_type] = {
            "baseline_mean": float(np.mean(d["baseline"])) if d["baseline"] else 0.0,
            "baseline_std": float(np.std(d["baseline"], ddof=0)) if d["baseline"] else 0.0,
            "candidate_mean": float(np.mean(d["candidate"])) if d["candidate"] else 0.0,
            "candidate_std": float(np.std(d["candidate"], ddof=0)) if d["candidate"] else 0.0,
            "gain_mean": float(np.mean(d["diff"])) if d["diff"] else 0.0,
            "gain_std": float(np.std(d["diff"], ddof=0)) if d["diff"] else 0.0,
        }
    return out


def compute_score(*, baseline_results, candidate_results, metadata, wave_neg_penalty = 1.0, wave_std_penalty = 0.2, unb_neg_penalty = 0.5, unb_std_penalty = 0.1):
    gains = compute_category_gains(baseline_results, candidate_results, metadata)

    base_mean = float(np.mean([r["co2_avg"] for r in baseline_results])) if baseline_results else 0.0
    cand_mean = float(np.mean([r["co2_avg"] for r in candidate_results])) if candidate_results else 0.0
    gain_overall = improvement_pct(base_mean, cand_mean)

    wave = gains.get("Wave", {"gain_mean": 0.0, "gain_std": 0.0})
    unb = gains.get("Unbalanced", {"gain_mean": 0.0, "gain_std": 0.0})

    score = float(gain_overall)
    if float(wave["gain_mean"]) < 0:
        score -= float(wave_neg_penalty) * abs(float(wave["gain_mean"]))
    score -= float(wave_std_penalty) * float(wave["gain_std"])

    if float(unb["gain_mean"]) < 0:
        score -= float(unb_neg_penalty) * abs(float(unb["gain_mean"]))
    score -= float(unb_std_penalty) * float(unb["gain_std"])

    return {
        "gain_overall": float(gain_overall),
        "gain_wave": float(wave["gain_mean"]),
        "std_wave": float(wave["gain_std"]),
        "gain_unbalanced": float(unb["gain_mean"]),
        "std_unbalanced": float(unb["gain_std"]),
        "score": float(score),
    }


def print_summary_compare(baseline_summary, candidate_summary, candidate_label):
    rows = [
        ("Total CO2 (g)", baseline_summary["co2"], candidate_summary["co2"]),
        ("Avg CO2/veh (g)", baseline_summary["co2_avg"], candidate_summary["co2_avg"]),
        ("Total Travel Time", baseline_summary["travel_time"], candidate_summary["travel_time"]),
        ("Total Waiting Time", baseline_summary["waiting_time"], candidate_summary["waiting_time"]),
    ]

    print("\n=======================================================")
    print("                 BENCHMARK RESULTS")
    print("=======================================================")
    print(f"{'Metric':<20} | {'Baseline':<18} | {candidate_label:<18} | {'Improvement %'}")
    print("-" * 75)

    for label, base_value, cand_value in rows:
        print(
            f"{label:<20} | {base_value:<18.2f} | {cand_value:<18.2f} | {improvement_pct(base_value, cand_value):+.2f}%"
        )


def print_type_statistics_compare(baseline_results, candidate_results, metadata, candidate_label):
    stats = defaultdict(lambda: {"baseline": [], "candidate": [], "diff": []})
    cand_by_ep = {r["episode"]: r for r in candidate_results}

    for base in baseline_results:
        ep = base["episode"]
        if ep not in cand_by_ep:
            continue
        cand = cand_by_ep[ep]
        episode_type, _ = metadata.get(ep, ("Unknown", 0)) # get episode type from metadata, default to "Unknown" if not found
        diff_pct = improvement_pct(float(base["co2_avg"]), float(cand["co2_avg"]))
        stats[episode_type]["baseline"].append(float(base["co2_avg"]))
        stats[episode_type]["candidate"].append(float(cand["co2_avg"]))
        stats[episode_type]["diff"].append(float(diff_pct))

    print("\n=======================================================")
    print("                 STATISTICS BY TYPE")
    print("=======================================================")
    print(f"{'Type':<12} | {'Baseline (g/v)':<18} | {candidate_label:<18} | {'Gain (%)'}")
    print("-" * 75)

    for ep_type in ORDERED_TYPES:
        d = stats.get(ep_type)
        if not d:
            continue
        base_mean = float(np.mean(d["baseline"]))
        base_std = float(np.std(d["baseline"], ddof=0))
        cand_mean = float(np.mean(d["candidate"]))
        cand_std = float(np.std(d["candidate"], ddof=0))
        gain_mean = float(np.mean(d["diff"]))
        gain_std = float(np.std(d["diff"], ddof=0))

        print(
            f"{ep_type:<12} | {base_mean:6.2f} ± {base_std:<6.2f} | {cand_mean:6.2f} ± {cand_std:<6.2f} | {gain_mean:+.2f}% ± {gain_std:.2f}  %"
        )


def print_episode_breakdown_compare(baseline_results, candidate_results, metadata, candidate_label):
    cand_by_ep = {r["episode"]: r for r in candidate_results}
    print("\nPer-Episode Breakdown (Avg CO2/veh in g):")
    print(
        f"{'Episode':<8} | {'Type':<10} | {'Vehs':<5} | {'Baseline (g/v)':<15} | {candidate_label:<15} | {'Diff (%)'}"
    )
    print("-" * 100)

    for base in baseline_results:
        ep = base["episode"]
        if ep not in cand_by_ep:
            continue
        cand = cand_by_ep[ep]
        ep_type, vehicles = metadata.get(ep, ("Unknown", 0))
        base_co2_avg = float(base["co2_avg"])
        cand_co2_avg = float(cand["co2_avg"])
        diff_pct = improvement_pct(base_co2_avg, cand_co2_avg)

        print(
            f"{ep:<8} | {ep_type:<10} | {vehicles:<5} | {base_co2_avg:<15.2f} | {cand_co2_avg:<15.2f} | {diff_pct:+.2f}%"
        )
