"""
Decode best solution từ mỗi thuật toán — lấy n_stations + coverage
Seed tốt nhất của mỗi thuật toán được đọc tự động từ results/raw_fitness_runs.csv
(sinh ra sau khi chạy mpa_ooa_evclp.py), nên không cần sửa tay khi thêm thuật toán mới.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from mpa_ooa_evclp import (
    EVCLP, GeneticAlgorithm, ParticleSwarmOptimization, MPA_OOA,
    GreedyConstructive, ClusteringHeuristic, HillClimbing,
    POP_SIZE_GA, POP_SIZE_PSO, POP_SIZE_MPA, MAX_ITER, OUT_DIR
)

ALGO_PARAMS = {
    'Greedy':       (GreedyConstructive,        {'max_iter': MAX_ITER}),
    'Clustering':   (ClusteringHeuristic,       {'max_iter': MAX_ITER}),
    'HillClimbing': (HillClimbing,              {'max_iter': MAX_ITER}),
    'GA':           (GeneticAlgorithm,          {'pop_size': POP_SIZE_GA,  'max_iter': MAX_ITER}),
    'PSO':          (ParticleSwarmOptimization, {'pop_size': POP_SIZE_PSO, 'max_iter': MAX_ITER}),
    'MPA-OOA':      (MPA_OOA,                   {'pop_size': POP_SIZE_MPA, 'max_iter': MAX_ITER}),
}

def load_best_seeds():
    raw_path = os.path.join(OUT_DIR, "raw_fitness_runs.csv")
    df_raw = pd.read_csv(raw_path)
    best = {}
    for algo in ALGO_PARAMS:
        sub = df_raw[df_raw['Algorithm'] == algo]
        row = sub.loc[sub['Fitness'].idxmin()]
        best[algo] = {'seed': int(row['Seed']), 'fitness': float(row['Fitness'])}
    return best

def main():
    print("Loading EVCLP problem (data_seed=42)...")
    problem = EVCLP()          # default n_locations=200, n_demand_points=100, data_seed=42
    print(f"  {problem.n_locations} candidate locations, {problem.n_demand_points} demand points\n")

    best_config = load_best_seeds()
    rows = []
    for name, cfg in best_config.items():
        seed = cfg['seed']
        AlgoClass, params = ALGO_PARAMS[name]
        print(f"[{name}] seed={seed} ...")
        np.random.seed(seed)
        alg = AlgoClass(problem, **params)
        best_sol, best_fit = alg.run()
        info = problem.get_metrics(best_sol)
        rows.append({
            'Algorithm':        name,
            'Best_Seed':        seed,
            'Best_Fitness':     best_fit,
            'N_Stations':       info['n_stations'],
            'Coverage_%':       round(info['coverage'], 2),
            'Avg_Distance_m':   round(info['avg_distance'], 1),
        })
        print(f"  → fitness={best_fit:.6f}  stations={info['n_stations']}  "
              f"coverage={info['coverage']:.1f}%  avg_dist={info['avg_distance']:.0f}m\n")

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "best_solution_decode.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(df.to_string(index=False))
    print(f"\n→ Đã lưu: {out_path}")

    # Tính % giảm số trạm so với GA (baseline)
    ga_n = df.loc[df['Algorithm']=='GA', 'N_Stations'].values[0]
    for _, row in df.iterrows():
        if row['Algorithm'] != 'GA':
            red = (ga_n - row['N_Stations']) / ga_n * 100
            print(f"  {row['Algorithm']} giảm {red:.1f}% số trạm so với GA")

if __name__ == '__main__':
    main()
