"""Benchmark runner for STL2, STL[1,2], and Improved STL.

Examples:
  # STL2 tuned vs default
  python run_benchmark.py --candidate stl2 --params logs/optimization_stl2/best_stl2_params.json

  # STL[1,2] tuned vs STL2 default
  python run_benchmark.py --candidate stl12 --params logs/optimization_stl12/best_stl12_params.json

  # Improved STL tuned (json) vs STL2 default
  python run_benchmark.py --candidate improved_stl --params logs/optimization_improved_stl/phase2/best_phase2.json
"""
import argparse
import os
from episode_bank import ensure_episode_bank, load_json
from algo_registry import build_stl2_spec, build_stl12_spec, build_improved_stl_spec
from experiment_common import (
    evaluate_episodes,
    summarize_results,
    print_summary_compare,
    print_type_statistics_compare,
    print_episode_breakdown_compare,
)


def _load_params(path):
    payload = load_json(path)
    if isinstance(payload, dict) and "params" in payload and isinstance(payload["params"], dict):
        return {k: float(v) for k, v in payload["params"].items()}
    return {k: float(v) for k, v in payload.items()} if isinstance(payload, dict) else {}


def main():
    parser = argparse.ArgumentParser(description="Unified benchmark on a deterministic 5xN episode bank.")
    parser.add_argument("--candidate", choices=["stl2", "stl12", "improved_stl"], required=True)
    parser.add_argument("--params", required=True, help="JSON with tuned params (best_*.json)")
    parser.add_argument("--episodes-per-type", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--print-breakdown", action="store_true")
    args = parser.parse_args()

    if args.candidate == "stl2":
        spec = build_stl2_spec()
        candidate_label = "STL2 Tuned"
        log_root = os.path.join("logs", "benchmark_stl2_tuned")
    elif args.candidate == "stl12":
        spec = build_stl12_spec()
        candidate_label = "STL [1,2] Tuned"
        log_root = os.path.join("logs", "benchmark_stl12_tuned")
    else:
        spec = build_improved_stl_spec(phase=2)
        candidate_label = "Improved STL (JSON)"
        log_root = os.path.join("logs", "benchmark_improved_stl_tuned")

    os.makedirs(log_root, exist_ok=True)

    bank_path = os.path.join(log_root, f"episode_bank_{args.episodes_per_type}x5.json")
    bank = ensure_episode_bank(bank_path, episodes_per_type=args.episodes_per_type)
    episodes = bank.all_episodes
    metadata = bank.metadata

    tuned_params = _load_params(args.params)

    print(f"Using {args.episodes_per_type} episodes per scenario ({len(episodes)} total).")
    print(f"Candidate: {args.candidate}")
    print(f"Params loaded from: {args.params}")

    baseline_results = evaluate_episodes(
        episodes=episodes,
        metadata=metadata,
        label="baseline",
        controller=spec.baseline,
        default_params=spec.baseline_module_defaults,
        params=None,
        log_dir=os.path.join(log_root, "baseline"),
        n_jobs=args.n_jobs,
    )

    candidate_results = evaluate_episodes(
        episodes=episodes,
        metadata=metadata,
        label="candidate",
        controller=spec.candidate,
        default_params=spec.candidate_module_defaults,
        params=tuned_params,
        log_dir=os.path.join(log_root, "candidate"),
        n_jobs=args.n_jobs,
    )

    base_summary = summarize_results(baseline_results)
    cand_summary = summarize_results(candidate_results)

    print_summary_compare(base_summary, cand_summary, candidate_label)
    print_type_statistics_compare(baseline_results, candidate_results, metadata, candidate_label + " (g/v)")
    if args.print_breakdown:
        print_episode_breakdown_compare(baseline_results, candidate_results, metadata, candidate_label + " (g/v)")


if __name__ == "__main__":
    main()
