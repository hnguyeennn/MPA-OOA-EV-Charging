"""
Phụ lục — Kiểm định Wilcoxon cho Ablation Study (Bảng 9), đối chiếu với bộ số liệu
EVCLP mới (thay thế mốc 0,04964 không tái lập được của bài báo R1).

Lý do cần script riêng: run_ablation() trong mpa_ooa_evclp.py chỉ lưu
Mean/Std/Best/Median vào ablation_evclp.csv, không lưu raw fitness từng lần chạy,
nên không tính được kiểm định thống kê cho từng cặp "w/o X" so với "Full". Script
này chạy lại đúng 5 cấu hình ablation (ABLATION_CONFIGS, import từ mpa_ooa_evclp.py,
không sửa file gốc), lưu raw fitness, và tính Wilcoxon so với "Full (MPA-OOA)".

Không đụng main_stats.csv/ablation_evclp.csv — ghi ra file riêng.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from mpa_ooa_evclp import (
    EVCLP, MPA_OOA, ABLATION_CONFIGS,
    POP_SIZE_MPA, MAX_ITER, SEEDS, N_RUNS, OUT_DIR
)


def main():
    print("Khởi tạo bài toán EVCLP (ngân sách 100 tỷ, cùng cấu hình mpa_ooa_evclp.py)...")
    problem = EVCLP(n_locations=200, n_demand_points=100, data_seed=42)

    results = {}
    rows = []
    raw_rows = []
    for cfg_name, flags in ABLATION_CONFIGS.items():
        print(f"\n  Config: {cfg_name}")
        fit_list = []
        for run_idx, seed in enumerate(SEEDS[:N_RUNS]):
            np.random.seed(seed)
            alg = MPA_OOA(problem, pop_size=POP_SIZE_MPA, max_iter=MAX_ITER, **flags)
            _, bf = alg.run()
            fit_list.append(bf)
            raw_rows.append({'Config': cfg_name, 'Run': run_idx + 1, 'Seed': seed, 'Fitness': bf})
            print(f"    run {run_idx+1}/{N_RUNS} seed={seed}: fitness={bf:.6f}")
        results[cfg_name] = fit_list
        rows.append({
            'Config': cfg_name,
            'Mean_Fitness': np.mean(fit_list), 'Std_Fitness': np.std(fit_list),
            'Best_Fitness': np.min(fit_list), 'Median_Fitness': np.median(fit_list),
        })

    ref = np.array(results['Full (MPA-OOA)'])
    wilcoxon_rows = []
    for cfg_name in ABLATION_CONFIGS:
        if cfg_name == 'Full (MPA-OOA)':
            continue
        comp_vals = np.array(results[cfg_name])
        try:
            stat, p = wilcoxon(ref, comp_vals)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        except Exception:
            stat, p, sig = np.nan, np.nan, 'N/A'
        delta_pct = (np.mean(comp_vals) - np.mean(ref)) / np.mean(ref) * 100
        direction = 'xấu đi (cơ chế có đóng góp tích cực)' if delta_pct > 0 else 'tốt hơn (loại bỏ không làm xấu, thậm chí cải thiện)'
        wilcoxon_rows.append({
            'Comparison': f'Full vs {cfg_name}', 'Delta_%': round(delta_pct, 2),
            'Direction': direction, 'W_stat': stat, 'p_value': p, 'Significance': sig,
        })

    df_stats = pd.DataFrame(rows)
    df_wil = pd.DataFrame(wilcoxon_rows)
    df_raw = pd.DataFrame(raw_rows)

    out1 = os.path.join(OUT_DIR, "ablation_wilcoxon_stats.csv")
    out2 = os.path.join(OUT_DIR, "ablation_wilcoxon.csv")
    out3 = os.path.join(OUT_DIR, "ablation_raw_fitness.csv")
    df_stats.to_csv(out1, index=False, encoding='utf-8-sig')
    df_wil.to_csv(out2, index=False, encoding='utf-8-sig')
    df_raw.to_csv(out3, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 70)
    print("KẾT QUẢ — Ablation Study, kiểm định Wilcoxon (bộ số liệu mới)")
    print("=" * 70)
    print(df_stats.to_string(index=False))
    print("\nWilcoxon (vs Full):")
    print(df_wil.to_string(index=False))
    print(f"\n→ Đã lưu: {out1}")
    print(f"→ Đã lưu: {out2}")
    print(f"→ Đã lưu: {out3}")


if __name__ == '__main__':
    main()
