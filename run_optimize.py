"""Optimizer for STL2, STL[1,2], and Improved STL.

Examples:
  # STL2 (J,K)
  python run_optimize.py --algo stl2 --trials 100 --episodes-per-type 10 --n-jobs 16

  # STL[1,2] (J,K,Ke)
  python run_optimize.py --algo stl12 --trials 100 --episodes-per-type 10 --n-jobs 16

  # Improved STL phase1
  python run_optimize.py --algo improved_stl --phase 1 --trials 100 --episodes-per-type 10 --n-jobs 16

  # Improved STL phase2 (starting from phase1 json)
  python run_optimize.py --algo improved_stl --phase 2 --phase1-json logs/optimization_improved_stl/phase1/best_phase1.json \
      --trials 100 --episodes-per-type 10 --n-jobs 16
"""
import argparse
import os
import optuna
from episode_bank import ensure_episode_bank, load_json, save_json
from algo_registry import build_stl2_spec, build_stl12_spec, build_improved_stl_spec
from experiment_common import evaluate_episodes, compute_score


def _load_params(path):
    payload = load_json(path)
    if isinstance(payload, dict) and "params" in payload and isinstance(payload["params"], dict):
        return {k: float(v) for k, v in payload["params"].items()}
    return {k: float(v) for k, v in payload.items()} if isinstance(payload, dict) else {}


def main():
    parser = argparse.ArgumentParser(description="Parameters tuning for Optuna.")
    parser.add_argument("--algo", choices=["stl2", "stl12", "improved_stl"], required=True)
    parser.add_argument("--phase", type=int, default=1, help="Improved STL only: 1 or 2")
    parser.add_argument("--phase1-json", default=None, help="Improved STL phase2: path to best_phase1.json")

    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--episodes-per-type", type=int, default=10)

    # objective shaping
    # Weight for wave gain is bigger
    parser.add_argument("--wave-neg-penalty", type=float, default=1.0)
    parser.add_argument("--wave-std-penalty", type=float, default=0.2)
    parser.add_argument("--unb-neg-penalty", type=float, default=0.5)
    parser.add_argument("--unb-std-penalty", type=float, default=0.1)

    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.algo == "stl2":
        spec = build_stl2_spec()
        log_root = os.path.join("logs", "optimization_stl2")
        best_path = os.path.join(log_root, "best_stl2_params.json")
    elif args.algo == "stl12":
        spec = build_stl12_spec()
        log_root = os.path.join("logs", "optimization_stl12")
        best_path = os.path.join(log_root, "best_stl12_params.json")
    else:
        phase = int(args.phase)
        if phase not in (1, 2):
            raise ValueError("--phase must be 1 or 2")
        base_params = None
        if phase == 2:
            if not args.phase1_json:
                raise ValueError("Improved STL phase2 requires --phase1-json")
            base_params = _load_params(args.phase1_json)
        spec = build_improved_stl_spec(phase=phase, base_params=base_params)
        log_root = os.path.join("logs", "optimization_improved_stl", f"phase{phase}")
        best_path = os.path.join(log_root, f"best_phase{phase}.json")

    os.makedirs(log_root, exist_ok=True)

    bank_path = os.path.join(log_root, f"validation_bank_{args.episodes_per_type*5}.json")
    bank = ensure_episode_bank(bank_path, episodes_per_type=args.episodes_per_type)
    episodes = bank.all_episodes
    metadata = bank.metadata

    baseline_cache = os.path.join(log_root, "baseline_validation.json")
    if os.path.exists(baseline_cache):
        baseline_payload = load_json(baseline_cache)
        baseline_results = baseline_payload["results"]
    else:
        baseline_results = evaluate_episodes(
            episodes=episodes,
            metadata=metadata,
            label="baseline",
            controller=spec.baseline,
            default_params=spec.baseline_module_defaults,
            params=None,
            log_dir=os.path.join(log_root, "baseline_logs"),
            n_jobs=args.n_jobs,
        )
        save_json(baseline_cache, {"episodes": episodes, "results": baseline_results})

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name or f"{args.algo}_tuning",
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> float:
        params = spec.suggest_params(trial)

        trial_log_dir = os.path.join(log_root, "trials", f"trial_{trial.number:04d}")
        cand_results = evaluate_episodes(
            episodes=episodes,
            metadata=metadata,
            label=f"trial_{trial.number}",
            controller=spec.candidate,
            default_params=spec.candidate_module_defaults,
            params=params,
            log_dir=trial_log_dir,
            n_jobs=args.n_jobs,
        )

        metrics = compute_score(
            baseline_results=baseline_results,
            candidate_results=cand_results,
            metadata=metadata,
            wave_neg_penalty=args.wave_neg_penalty,
            wave_std_penalty=args.wave_std_penalty,
            unb_neg_penalty=args.unb_neg_penalty,
            unb_std_penalty=args.unb_std_penalty,
        )

        for k, v in metrics.items():
            trial.set_user_attr(k, float(v))

        return float(metrics["score"])

    study.optimize(objective, n_trials=int(args.trials))

    best = study.best_trial
    payload = {
        "params": best.params,
        "score": float(best.value),
        "metrics": {k: best.user_attrs.get(k) for k in ["gain_overall", "gain_wave", "std_wave", "gain_unbalanced", "std_unbalanced"]},
        "validation_bank": bank_path,
        "baseline_cache": baseline_cache,
        "episodes": episodes,
        "algo": args.algo,
    }
    if args.algo == "improved_stl":
        payload["phase"] = int(args.phase)
        if args.phase == 2:
            payload["phase1_json"] = args.phase1_json

    save_json(best_path, payload)

    print("\nBest params found:")
    print(payload)


if __name__ == "__main__":
    main()
