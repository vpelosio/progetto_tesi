# Progetto Tesi di Laurea

Implementazione e tuning di un algoritmo per la riduzione della CO2 negli incroci semaforici.
---

## Setup dell'ambiente
```bash
python3 -m venv .venv
source .venv/bin/activate (su linux)
pip3 install -r requirements.txt
```

## Ottimizzazione e benchmark (scegli N per numero core cpu):
```bash
python run_optimize.py --algo stl2 --trials 100 --episodes-per-type 10 --n-jobs N
python run_benchmark.py --candidate stl2 --params logs/optimization_stl2/best_stl2_params.json --n-jobs N

python run_optimize.py --algo stl12 --trials 100 --episodes-per-type 10 --n-jobs N
python run_benchmark.py --candidate stl12 --params logs/optimization_stl12/best_stl12_params.json --n-jobs N

python run_optimize.py --algo improved_stl --phase 1 --trials 100 --episodes-per-type 10 --n-jobs N
python run_optimize.py --algo improved_stl --phase 2 --phase1-json logs/optimization_improved_stl/phase1/best_phase1.json --trials 100 --episodes-per-type 10 --n-jobs N
python run_benchmark.py --candidate improved_stl --params logs/optimization_improved_stl/phase2/best_phase2.json --n-jobs N
```