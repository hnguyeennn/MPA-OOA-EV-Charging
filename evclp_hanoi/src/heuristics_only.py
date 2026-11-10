"""
[KHÔNG PHẢI NGUỒN SỐ LIỆU CHÍNH — xem README mục 2] Script lịch sử/tùy chọn: chỉ
chạy 3 heuristic đơn giản (Greedy, Clustering, Hill Climbing), so với một bộ giá trị
MPA-OOA hard-code cứng bên dưới.

Bối cảnh: script này từng được viết để tránh đụng vào số liệu R1 "đã chốt" (khi đó
tưởng không tái lập được). Sau đó phát hiện bộ MPA_OOA_RAW_REF hard-code dưới đây
thực ra LẤY NHẦM từ một lần chạy lại (không phải R1 thật — R1 không công bố raw
fitness từng lần chạy) nên Wilcoxon tính ra không đáng tin. Quyết định cuối: chạy
lại toàn bộ 6 thuật toán trong CÙNG MỘT LẦN bằng `mpa_ooa_evclp.py` (đã có sẵn cả
Greedy/Clustering/HillClimbing/GA/PSO/MPA-OOA trong `configs`), dùng bộ số liệu đó
(`results/main_stats.csv`, `wilcoxon_evclp.csv`) làm NGUỒN CHÍNH DUY NHẤT.

File này giữ lại chỉ để tiện chạy nhanh 3 heuristic đơn giản khi cần thử nghiệm
riêng lẻ (không cần chờ GA/PSO/MPA-OOA chạy lại, vốn tốn ~90 phút). Nếu dùng để
tính Wilcoxon so với MPA-OOA, PHẢI cập nhật `MPA_OOA_RAW_REF` bằng đúng 10 giá trị
`raw_fitness_runs.csv` mới nhất (đọc trực tiếp, không hard-code lại lần nữa).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from mpa_ooa_evclp import EVCLP, GreedyConstructive, ClusteringHeuristic, HillClimbing, SEEDS, N_RUNS, MAX_ITER, OUT_DIR

def load_mpa_ooa_raw_fitness():
    """Đọc 10 giá trị fitness thô của MPA-OOA từ raw_fitness_runs.csv (nguồn chính,
    sinh bởi mpa_ooa_evclp.py) — KHÔNG hard-code, tránh lặp lại lỗi dùng nhầm dữ
    liệu cũ như trước đây."""
    raw_path = os.path.join(OUT_DIR, "raw_fitness_runs.csv")
    df_raw = pd.read_csv(raw_path)
    vals = df_raw[df_raw['Algorithm'] == 'MPA-OOA'].sort_values('Run')['Fitness'].values
    if len(vals) != N_RUNS:
        raise RuntimeError(
            f"Kỳ vọng {N_RUNS} giá trị MPA-OOA trong raw_fitness_runs.csv, "
            f"tìm thấy {len(vals)}. Chạy lại mpa_ooa_evclp.py trước.")
    return vals


def main():
    print("Khởi tạo bài toán EVCLP (ngân sách 100 tỷ, cùng cấu hình mpa_ooa_evclp.py)...")
    problem = EVCLP(n_locations=200, n_demand_points=100, data_seed=42)

    configs = {
        'Greedy':       (GreedyConstructive, {'max_iter': MAX_ITER}),
        'Clustering':   (ClusteringHeuristic, {'max_iter': MAX_ITER}),
        'HillClimbing': (HillClimbing, {'max_iter': MAX_ITER}),
    }

    results = {}
    rows = []
    for name, (cls, params) in configs.items():
        print(f"\n  Thuật toán: {name}")
        fit_list = []
        for run_idx, seed in enumerate(SEEDS[:N_RUNS]):
            np.random.seed(seed)
            alg = cls(problem, **params)
            _, bf = alg.run()
            fit_list.append(bf)
            print(f"    run {run_idx+1}/{N_RUNS} seed={seed}: fitness={bf:.6f}")
        results[name] = fit_list
        info = problem.get_metrics(alg.best_solution)
        best_idx = int(np.argmin(fit_list))
        rows.append({
            'Algorithm': name,
            'Mean_Fitness': np.mean(fit_list), 'Std_Fitness': np.std(fit_list),
            'Best_Fitness': np.min(fit_list), 'Worst_Fitness': np.max(fit_list),
        })

    # Wilcoxon: so 3 heuristic mới với MPA-OOA (đọc raw fitness thật từ file chính,
    # KHÔNG hard-code — nếu file chưa có, chạy mpa_ooa_evclp.py trước)
    ref = load_mpa_ooa_raw_fitness()
    wilcoxon_rows = []
    for name in configs:
        comp_vals = np.array(results[name])
        try:
            stat, p = wilcoxon(ref, comp_vals)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        except Exception:
            stat, p, sig = np.nan, np.nan, 'N/A'
        wilcoxon_rows.append({'Comparison': f'MPA-OOA vs {name}', 'W_stat': stat,
                               'p_value': p, 'Significance': sig})

    df_stats = pd.DataFrame(rows)
    df_wil = pd.DataFrame(wilcoxon_rows)

    out1 = os.path.join(OUT_DIR, "heuristics_only_stats.csv")
    out2 = os.path.join(OUT_DIR, "heuristics_only_wilcoxon.csv")
    df_stats.to_csv(out1, index=False, encoding='utf-8-sig')
    df_wil.to_csv(out2, index=False, encoding='utf-8-sig')

    print("\n" + "="*70)
    print("KẾT QUẢ — 3 heuristic đơn giản (so với MPA-OOA, raw fitness đọc từ file chính)")
    print("="*70)
    print(df_stats.to_string(index=False))
    print("\nWilcoxon (vs MPA-OOA):")
    print(df_wil.to_string(index=False))
    print(f"\n→ Đã lưu: {out1}")
    print(f"→ Đã lưu: {out2}")
    print("\nLưu ý: kết quả này CHỈ mang tính tham khảo nhanh — nguồn số liệu chính")
    print("thức là main_stats.csv/wilcoxon_evclp.csv sinh bởi mpa_ooa_evclp.py.")


if __name__ == '__main__':
    main()
