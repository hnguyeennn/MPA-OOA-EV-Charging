"""
Phụ lục — Greedy-seeded MPA-OOA (trả lời phản biện PB1, phần 2).
Không sửa mpa_ooa_evclp.py. Kỹ thuật: subclass MPA_OOA, chỉ thay khâu khởi tạo quần
thể (thêm nghiệm Greedy làm 1 cá thể mồi) rồi tái sử dụng nguyên vòng lặp gốc bằng
cách gọi lại logic run() của lớp cha thông qua khởi tạo quần thể trước khi super().run().

Câu hỏi cần trả lời: dùng nghiệm Greedy làm điểm khởi tạo cho MPA-OOA có giúp
(1) rút ngắn thời gian hội tụ / thời gian chạy, và (2) không làm xấu đi chất lượng
nghiệm (fitness), so với MPA-OOA khởi tạo ngẫu nhiên hoàn toàn (kịch bản chính)?
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from mpa_ooa_evclp import EVCLP, GreedyConstructive, MPA_OOA, POP_SIZE_MPA, MAX_ITER, SEEDS, N_RUNS, OUT_DIR


class SeededMPA_OOA(MPA_OOA):
    """MPA-OOA với 1 cá thể khởi tạo được thay bằng nghiệm Greedy (seeding),
    còn lại giữ nguyên 100% vòng lặp tối ưu hóa của lớp cha (MPA_OOA.run)."""

    def __init__(self, problem: EVCLP, greedy_solution: np.ndarray, **kwargs):
        super().__init__(problem, **kwargs)
        self._greedy_solution = greedy_solution

    def run(self):
        # Chạy đúng logic khởi tạo + vòng lặp của lớp cha, nhưng chèn nghiệm Greedy
        # vào quần thể ngay sau bước khởi tạo ngẫu nhiên gốc bằng cách tạm ghi đè
        # np.random.rand cho lần gọi khởi tạo đầu tiên là không cần thiết — thay vào
        # đó ta gọi lại chính xác đoạn khởi tạo của MPA_OOA.run rồi thay 1 cá thể.
        lb, ub = self.lb, self.ub
        n_rand = int(0.7 * self.pop_size)
        self.population = np.vstack([
            np.random.rand(n_rand, self.dim),
            np.array([lb + ub - np.random.rand(self.dim)
                      for _ in range(self.pop_size - n_rand)])
        ])
        self.population = np.clip(self.population, lb, ub)
        # Seeding: thay cá thể đầu tiên bằng nghiệm Greedy
        self.population[0] = np.clip(self._greedy_solution.astype(float), lb, ub)

        self.fitness = np.array([self.problem.objective_function(s) for s in self.population])

        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

        elite_idx = np.argsort(self.fitness)[:self.n_elite]
        self.elite_solutions = self.population[elite_idx].copy()
        self.elite_fitness = self.fitness[elite_idx].copy()
        self.convergence_history = [self.best_fitness]

        for t in range(self.max_iter):
            E_f = self._explore_factor(t)
            old_best = self.best_fitness
            n1 = int(0.4 * self.pop_size)
            n2 = int(0.8 * self.pop_size)

            for i in range(self.pop_size):
                if E_f > 0.5:
                    if i < n1:
                        elite = self._get_elite()
                        new = np.clip(self.population[i] +
                                      0.5 * self._levy_flight() * (elite - self.population[i]),
                                      lb, ub)
                    elif i < n2:
                        fish = (self.population[i]
                                if self.fitness[i] < self.best_fitness
                                else self.best_solution)
                        IF = np.random.randint(1, 3)
                        noise = np.random.randn(self.dim) * 0.01
                        new = np.clip(self.population[i] +
                                      np.random.rand(self.dim) * (fish - IF * self.population[i]) + noise,
                                      lb, ub)
                    else:
                        elite = self._get_elite()
                        obl = self._obl(self.population[i], E_f)
                        lev = self._levy_flight()
                        new = np.clip(0.6 * elite +
                                      0.3 * (self.population[i] + 0.3 * lev) + 0.1 * obl,
                                      lb, ub)
                else:
                    if i < self.pop_size // 2:
                        ref = self.population[
                            np.argmin(self.fitness[
                                np.random.choice(self.pop_size, min(3, self.pop_size), replace=False)])]
                        new = np.clip(self.population[i] +
                                      np.random.rand(self.dim) * (ref - self.population[i]),
                                      lb, ub)
                    else:
                        r = np.random.rand(self.dim)
                        new = np.clip(self.population[i] +
                                      self.P * r * (self.best_solution - self.population[i]),
                                      lb, ub)

                new = self._dynamic_fads(new, t, E_f)
                new_fit = self.problem.objective_function(new)
                if new_fit < self.fitness[i]:
                    self.population[i] = new
                    self.fitness[i] = new_fit
                    if new_fit < self.best_fitness:
                        self.best_fitness = new_fit
                        self.best_solution = new.copy()
                        self.best_improvement_iter = t

            self._update_elite()

            if abs(old_best - self.best_fitness) < 1e-8:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            if self.stagnation_counter > 20:
                centroid = np.mean(self.population, axis=0)
                sigma_c = np.mean([
                    np.linalg.norm(s - self.best_solution) /
                    (np.linalg.norm(self.best_solution) + 1e-10)
                    for s in self.population])
                if sigma_c < 0.01:
                    worst = np.argsort(self.fitness)[-int(0.2 * self.pop_size):]
                    for w in worst:
                        self.population[w] = np.random.rand(self.dim)
                        self.fitness[w] = self.problem.objective_function(self.population[w])
                    self.stagnation_counter = 0

            self.convergence_history.append(self.best_fitness)

        return self.best_solution, self.best_fitness


def main():
    print("Khởi tạo bài toán EVCLP (ngân sách gốc 100 tỷ, giống kịch bản chính)...")
    problem = EVCLP(n_locations=200, n_demand_points=100, data_seed=42)

    # Nghiệm Greedy dùng để seeding — tất định, tính 1 lần
    np.random.seed(SEEDS[0])
    g = GreedyConstructive(problem, max_iter=MAX_ITER)
    greedy_solution, greedy_fitness = g.run()
    print(f"  Nghiệm Greedy (seed nguồn): fitness={greedy_fitness:.6f}")

    rows_fit, rows_time = [], []
    fit_seeded, fit_baseline, time_seeded, time_baseline = [], [], [], []

    for run_idx, seed in enumerate(SEEDS[:N_RUNS]):
        # Baseline: MPA-OOA khởi tạo ngẫu nhiên (kịch bản chính, không seeding)
        np.random.seed(seed)
        t0 = time.time()
        base = MPA_OOA(problem, pop_size=POP_SIZE_MPA, max_iter=MAX_ITER)
        _, bf_base = base.run()
        t_base = time.time() - t0
        fit_baseline.append(bf_base)
        time_baseline.append(t_base)

        # Seeded: MPA-OOA với 1 cá thể = nghiệm Greedy
        np.random.seed(seed)
        t0 = time.time()
        seeded = SeededMPA_OOA(problem, greedy_solution, pop_size=POP_SIZE_MPA, max_iter=MAX_ITER)
        _, bf_seed = seeded.run()
        t_seed = time.time() - t0
        fit_seeded.append(bf_seed)
        time_seeded.append(t_seed)

        print(f"  run {run_idx+1}/{N_RUNS} seed={seed}: "
              f"baseline fit={bf_base:.6f} time={t_base:.1f}s | "
              f"seeded fit={bf_seed:.6f} time={t_seed:.1f}s")

    stat, p = wilcoxon(fit_seeded, fit_baseline)
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))

    df = pd.DataFrame({
        'Variant': ['MPA-OOA (baseline, random init)', 'MPA-OOA (Greedy-seeded)'],
        'Mean_Fitness': [np.mean(fit_baseline), np.mean(fit_seeded)],
        'Std_Fitness': [np.std(fit_baseline), np.std(fit_seeded)],
        'Best_Fitness': [np.min(fit_baseline), np.min(fit_seeded)],
        'Mean_Time_s': [np.mean(time_baseline), np.mean(time_seeded)],
    })
    df_wil = pd.DataFrame([{
        'Comparison': 'Seeded vs Baseline (fitness)', 'W_stat': stat,
        'p_value': p, 'Significance': sig
    }])

    out1 = os.path.join(OUT_DIR, "seeding_stats.csv")
    out2 = os.path.join(OUT_DIR, "seeding_wilcoxon.csv")
    df.to_csv(out1, index=False, encoding='utf-8-sig')
    df_wil.to_csv(out2, index=False, encoding='utf-8-sig')

    print("\n" + "="*70)
    print("KẾT QUẢ — Greedy-seeded MPA-OOA vs baseline (random init)")
    print("="*70)
    print(df.to_string(index=False))
    print("\nWilcoxon (fitness, seeded vs baseline):")
    print(df_wil.to_string(index=False))
    time_reduction = (1 - np.mean(time_seeded) / np.mean(time_baseline)) * 100
    print(f"\nGiảm thời gian trung bình: {time_reduction:.1f}%")
    print(f"\n→ Đã lưu: {out1}")
    print(f"→ Đã lưu: {out2}")


if __name__ == '__main__':
    main()
