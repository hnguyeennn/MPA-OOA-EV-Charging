"""
Phụ lục — Phân tích độ nhạy theo ngân sách (trả lời phản biện PB1).
Không thay đổi thực nghiệm chính (main_stats.csv, ngân sách 100 tỷ, Bảng 7 gốc).
Ở đây siết ngân sách xuống 10 tỷ VNĐ để tạo áp lực đánh đổi thật giữa chi phí và
độ phủ, làm rõ khoảng cách chất lượng nghiệm giữa Greedy (tham lam một tiêu chí,
không hoạch định toàn cục) và MPA-OOA (tìm kiếm toàn cục + tái cấu trúc quần thể).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from mpa_ooa_evclp import (
    EVCLP, GreedyConstructive, ClusteringHeuristic, HillClimbing,
    GeneticAlgorithm, ParticleSwarmOptimization, MPA_OOA,
    POP_SIZE_GA, POP_SIZE_PSO, POP_SIZE_MPA, MAX_ITER, SEEDS, N_RUNS, OUT_DIR
)

TIGHT_BUDGET = 10_000   # 10 tỷ VNĐ (so với 100 tỷ VNĐ trong thực nghiệm chính)

ALGO_PARAMS = {
    'Greedy':       (GreedyConstructive,        {'max_iter': MAX_ITER}),
    'Clustering':   (ClusteringHeuristic,       {'max_iter': MAX_ITER}),
    'HillClimbing': (HillClimbing,              {'max_iter': MAX_ITER}),
    'GA':           (GeneticAlgorithm,          {'pop_size': POP_SIZE_GA,  'max_iter': MAX_ITER}),
    'PSO':          (ParticleSwarmOptimization, {'pop_size': POP_SIZE_PSO, 'max_iter': MAX_ITER}),
    'MPA-OOA':      (MPA_OOA,                   {'pop_size': POP_SIZE_MPA, 'max_iter': MAX_ITER}),
}


def main():
    print("Khởi tạo bài toán EVCLP với ngân sách siết chặt (10 tỷ VNĐ)...")
    problem = EVCLP(n_locations=200, n_demand_points=100, data_seed=42)
    problem.budget = TIGHT_BUDGET
    print(f"  Ngân sách: {problem.budget/1000:.0f} tỷ  (thực nghiệm chính dùng 100 tỷ)")

    results = {}
    rows = []
    for name, (cls, params) in ALGO_PARAMS.items():
        print(f"\n  Thuật toán: {name}")
        fit_list = []
        for run_idx, seed in enumerate(SEEDS[:N_RUNS]):
            np.random.seed(seed)
            alg = cls(problem, **params)
            _, bf = alg.run()
            fit_list.append(bf)
            print(f"    run {run_idx+1}/{N_RUNS} seed={seed}: fitness={bf:.6f}")
        results[name] = fit_list
        rows.append({
            'Algorithm': name,
            'Mean_Fitness': np.mean(fit_list), 'Std_Fitness': np.std(fit_list),
            'Best_Fitness': np.min(fit_list), 'Worst_Fitness': np.max(fit_list),
        })

    df_stats = pd.DataFrame(rows)

    ref = np.array(results['MPA-OOA'])
    wilcoxon_rows = []
    for comp in ['Greedy', 'Clustering', 'HillClimbing', 'GA', 'PSO']:
        comp_vals = np.array(results[comp])
        try:
            stat, p = wilcoxon(ref, comp_vals)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        except Exception:
            stat, p, sig = np.nan, np.nan, 'N/A'
        wilcoxon_rows.append({'Comparison': f'MPA-OOA vs {comp}', 'W_stat': stat,
                               'p_value': p, 'Significance': sig})
    df_wilcoxon = pd.DataFrame(wilcoxon_rows)

    out_stats = os.path.join(OUT_DIR, "sensitivity_budget_stats.csv")
    out_wil   = os.path.join(OUT_DIR, "sensitivity_budget_wilcoxon.csv")
    df_stats.to_csv(out_stats, index=False, encoding='utf-8-sig')
    df_wilcoxon.to_csv(out_wil, index=False, encoding='utf-8-sig')

    print("\n" + "="*70)
    print("KẾT QUẢ — Ngân sách siết chặt (10 tỷ VNĐ)")
    print("="*70)
    print(df_stats.to_string(index=False))
    print("\nWilcoxon:")
    print(df_wilcoxon.to_string(index=False))
    print(f"\n→ Đã lưu: {out_stats}")
    print(f"→ Đã lưu: {out_wil}")


if __name__ == '__main__':
    main()
