"""
MPA-OOA Benchmark Analysis - Full Version
Bổ sung: Seed cố định, Wilcoxon, Friedman+Nemenyi, Ablation study
Xuất tất cả kết quả ra CSV + hình PNG
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
from scipy.special import gamma
from scipy.stats import levy
import warnings
import os
import time
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. CẤU HÌNH CHUNG
# ─────────────────────────────────────────────
SEEDS = [42 + i * 7 for i in range(10)]   # 10 seeds cố định, tái lập được
POP_SIZE  = 50
MAX_ITER  = 500
N_RUNS    = 10
DIMS      = [30, 100]

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. HÀM BENCHMARK CEC2017 (9 hàm)
# ─────────────────────────────────────────────
class CEC2017Functions:
    def __init__(self, dim: int = 30):
        self.dim = dim
        self.functions = {
            'F1': self.sphere,
            'F2': self.zakharov,
            'F3': self.rosenbrock,
            'F4': self.rastrigin,
            'F5': self.lunacek_bi_rastrigin,
            'F6': self.expanded_scaffers_f6,
            'F7': self.non_continuous_rastrigin,
            'F8': self.levy_function,
            'F9': self.schwefel,
        }
        self.bounds = {
            'F1': (-100, 100), 'F2': (-100, 100), 'F3': (-30, 30),
            'F4': (-5.12, 5.12), 'F5': (-5.12, 5.12), 'F6': (-100, 100),
            'F7': (-5.12, 5.12), 'F8': (-10, 10), 'F9': (-500, 500),
        }

    def sphere(self, x): return np.sum(x**2)

    def zakharov(self, x):
        s1 = np.sum(x**2)
        s2 = np.sum(0.5 * np.arange(1, len(x)+1) * x)
        return s1 + s2**2 + s2**4

    def rosenbrock(self, x):
        return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def rastrigin(self, x):
        return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))

    def lunacek_bi_rastrigin(self, x):
        mu1 = 2.5
        mu2 = -np.sqrt(abs((mu1**2 - 1)/1))
        s = 1 - 1/(2*np.sqrt(len(x)+20) - 8.2)
        d = 1
        s1 = np.sum((x - mu1)**2)
        s2 = np.sum((x - mu2)**2)
        s3 = np.sum(1 - np.cos(2*np.pi*(x - mu1)))
        return min(s1, d*len(x) + s*s2) + 10*s3

    def expanded_scaffers_f6(self, x):
        def g(a, b):
            num = np.sin(np.sqrt(a**2 + b**2))**2 - 0.5
            den = (1 + 0.001*(a**2 + b**2))**2
            return 0.5 + num/den
        return sum(g(x[i], x[(i+1) % len(x)]) for i in range(len(x)))

    def non_continuous_rastrigin(self, x):
        y = np.where(np.abs(x) < 0.5, x, np.round(2*x)/2)
        return 10*len(y) + np.sum(y**2 - 10*np.cos(2*np.pi*y))

    def levy_function(self, x):
        w = 1 + (x - 1)/4
        t1 = np.sin(np.pi*w[0])**2
        t3 = (w[-1]-1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
        s  = np.sum((w[:-1]-1)**2 * (1 + 10*np.sin(np.pi*w[:-1]+1)**2))
        return t1 + s + t3

    def schwefel(self, x):
        return 418.9829*len(x) - np.sum(x*np.sin(np.sqrt(np.abs(x))))


# ─────────────────────────────────────────────
# 2. THUẬT TOÁN MPA-OOA ĐẦY ĐỦ (từ ver3_dim30)
# ─────────────────────────────────────────────
class MPA_OOA_Hybrid:
    """MPA-OOA lai ghép đầy đủ — tất cả cơ chế BẬT"""

    def __init__(self, pop_size, max_iter, dim, lb, ub,
                 use_levy_chaos=True, use_obl=True,
                 use_dynamic_fads=True, use_elite_pool=True):
        self.pop_size   = pop_size
        self.max_iter   = max_iter
        self.dim        = dim
        self.lb         = lb
        self.ub         = ub
        # ablation flags
        self.use_levy_chaos    = use_levy_chaos
        self.use_obl           = use_obl
        self.use_dynamic_fads  = use_dynamic_fads
        self.use_elite_pool    = use_elite_pool

        self.stagnation_counter = 0
        self.chaos_r = np.random.rand()
        self.P = 0.1

        self.elite_size = max(3, int(0.1 * pop_size))
        self.elite_solutions = []
        self.elite_fitness   = []

        self.convergence_history = []

    # ── helpers ──────────────────────────────
    def _levy_flight(self):
        beta = 1.5
        sigma = (gamma(1+beta)*np.sin(np.pi*beta/2) /
                 (gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = 0.05 * u / (np.abs(v)**(1/beta))
        if self.use_levy_chaos:
            self.chaos_r = 4*self.chaos_r*(1 - self.chaos_r)
            step = step * self.chaos_r
        return step

    def _obl(self, x, E_f):
        if not self.use_obl:
            return np.random.uniform(self.lb, self.ub, self.dim)
        alpha = 0.1 * E_f
        noise = np.random.normal(0, 1, self.dim)
        opp = self.lb + self.ub - x + alpha * noise
        return np.clip(opp, self.lb, self.ub)

    def _explore_factor(self, t):
        E_base = 1 - t/self.max_iter
        E_stag = min(0.4, self.stagnation_counter * 0.05)
        if len(self.population) > 0:
            centroid = np.mean(self.population, axis=0)
            div = np.mean([np.linalg.norm(x - centroid) for x in self.population])
            E_div = min(0.3, div / (self.ub - self.lb))
        else:
            E_div = 0.0
        return min(0.95, E_base + E_stag + E_div)

    def _dynamic_fads(self, sol, t, E_f):
        if self.use_dynamic_fads:
            P_fads = 0.15 + 0.1*(1 - E_f)
        else:
            P_fads = 0.2                          # FADs cố định như MPA gốc
        if np.random.rand() >= P_fads:
            return sol
        r = np.random.rand()
        if r < 0.3:
            return np.random.uniform(self.lb, self.ub, self.dim)
        elif r < 0.6 and len(self.elite_solutions) > 0:
            idx = np.random.randint(len(self.elite_solutions))
            return np.clip(self.elite_solutions[idx] +
                           np.random.randn(self.dim)*0.1, self.lb, self.ub)
        elif r < 0.8:
            return np.clip(self.best_solution + self._levy_flight(), self.lb, self.ub)
        else:
            return self._obl(sol, E_f)

    def _update_elite(self, fitness):
        if not self.use_elite_pool:
            return
        idx = np.argsort(fitness)[:self.elite_size]
        self.elite_solutions = list(self.population[idx])
        self.elite_fitness   = list(fitness[idx])

    def _get_elite_ref(self):
        if self.use_elite_pool and len(self.elite_solutions) > 0:
            return self.elite_solutions[np.random.randint(len(self.elite_solutions))]
        return self.best_solution

    # ── vòng lặp chính ───────────────────────
    def optimize(self, fitness_func):
        lb, ub = self.lb, self.ub
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([fitness_func(x) for x in self.population])

        best_idx = np.argmin(fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness  = fitness[best_idx]
        self._update_elite(fitness)
        self.convergence_history = [self.best_fitness]
        prev_best = self.best_fitness

        for t in range(self.max_iter):
            E_f = self._explore_factor(t)
            new_pop = self.population.copy()

            n1 = int(0.4 * self.pop_size)
            n2 = int(0.8 * self.pop_size)

            if E_f > 0.5:   # ── Giai đoạn khám phá ──
                for i in range(n1):                            # 40% MPA Phase 1
                    elite = self._get_elite_ref()
                    levy  = self._levy_flight()
                    new_pop[i] = np.clip(
                        self.population[i] + 0.5*levy*(elite - self.population[i]),
                        lb, ub)
                for i in range(n1, n2):                        # 40% OOA hunting
                    PF = (self.population[i]
                          if fitness[i] < self.best_fitness
                          else self.best_solution)
                    IF = np.random.randint(1, 3)
                    noise = np.random.randn(self.dim)*0.01
                    new_pop[i] = np.clip(
                        self.population[i] + np.random.rand(self.dim)*(PF - IF*self.population[i]) + noise,
                        lb, ub)
                for i in range(n2, self.pop_size):             # 20% Hybrid + OBL
                    elite = self._get_elite_ref()
                    obl   = self._obl(self.population[i], E_f)
                    levy  = self._levy_flight()
                    new_pop[i] = np.clip(
                        0.6*elite + 0.3*(self.population[i] + 0.3*levy) + 0.1*obl,
                        lb, ub)
            else:            # ── Giai đoạn khai thác ──
                n_half = self.pop_size // 2
                for i in range(n_half):                        # 50% OOA carrying
                    k = max(2, int(0.08*self.pop_size))
                    idx = np.random.choice(self.pop_size, k, replace=False)
                    ref = self.population[idx[np.argmin(fitness[idx])]]
                    r   = np.random.rand(self.dim)
                    new_pop[i] = np.clip(self.population[i] + r*(ref - self.population[i]), lb, ub)
                for i in range(n_half, self.pop_size):         # 50% MPA Phase 3
                    r = np.random.rand(self.dim)
                    new_pop[i] = np.clip(
                        self.population[i] + self.P*r*(self.best_solution - self.population[i]),
                        lb, ub)

            # Dynamic FADs
            for i in range(self.pop_size):
                new_pop[i] = self._dynamic_fads(new_pop[i], t, E_f)

            # Đánh giá + chọn lọc
            new_fit = np.array([fitness_func(x) for x in new_pop])
            improved = new_fit < fitness
            self.population[improved] = new_pop[improved]
            fitness[improved] = new_fit[improved]

            cur_best_idx = np.argmin(fitness)
            if fitness[cur_best_idx] < self.best_fitness:
                self.best_fitness  = fitness[cur_best_idx]
                self.best_solution = self.population[cur_best_idx].copy()
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            self._update_elite(fitness)

            # Tái tạo khi hội tụ sớm
            if self.stagnation_counter > 20:
                centroid = np.mean(self.population, axis=0)
                sigma_c = np.mean([
                    np.linalg.norm(x - self.best_solution) /
                    (np.linalg.norm(self.best_solution) + 1e-10)
                    for x in self.population])
                if sigma_c < 0.01:
                    worst = np.argsort(fitness)[-int(0.2*self.pop_size):]
                    for w in worst:
                        self.population[w] = np.random.uniform(lb, ub, self.dim)
                        fitness[w] = fitness_func(self.population[w])
                    self.stagnation_counter = 0

            self.convergence_history.append(self.best_fitness)

        return self.best_solution, self.best_fitness, self.convergence_history


# ─────────────────────────────────────────────
# 3. CÁC THUẬT TOÁN ĐỐI CHỨNG (GA, PSO, MPA gốc, OOA gốc)
# ─────────────────────────────────────────────
def run_pso(fitness_func, dim, lb, ub, pop_size=50, max_iter=500):
    particles  = np.random.uniform(lb, ub, (pop_size, dim))
    velocities = np.random.uniform(-1, 1, (pop_size, dim)) * 0.2*(ub-lb)
    pbest      = particles.copy()
    pbest_fit  = np.array([fitness_func(p) for p in particles])
    gbest_idx  = np.argmin(pbest_fit)
    gbest      = pbest[gbest_idx].copy()
    gbest_fit  = pbest_fit[gbest_idx]
    conv = [gbest_fit]
    stag, prev = 0, gbest_fit
    for t in range(max_iter):
        w  = 0.9 - 0.5*t/max_iter
        c1 = 2.0 - t/max_iter
        c2 = 2.0*t/max_iter
        for i in range(pop_size):
            r1, r2 = np.random.rand(2)
            velocities[i] = (w*velocities[i]
                             + c1*r1*(pbest[i]-particles[i])
                             + c2*r2*(gbest-particles[i]))
            velocities[i] = np.clip(velocities[i], -0.2*(ub-lb), 0.2*(ub-lb))
            particles[i]  = np.clip(particles[i] + velocities[i], lb, ub)
            f = fitness_func(particles[i])
            if f < pbest_fit[i]:
                pbest[i] = particles[i].copy(); pbest_fit[i] = f
            if f < gbest_fit:
                gbest = particles[i].copy(); gbest_fit = f
        if abs(gbest_fit - prev) < 1e-12: stag += 1
        else: stag = 0; prev = gbest_fit
        if stag > 30:
            idx = np.random.choice(pop_size, int(0.2*pop_size), replace=False)
            for k in idx:
                particles[k] = np.random.uniform(lb, ub, dim)
                pbest[k] = particles[k].copy()
                pbest_fit[k] = fitness_func(particles[k])
            stag = 0
        conv.append(gbest_fit)
    return gbest, gbest_fit, conv


def run_ga(fitness_func, dim, lb, ub, pop_size=50, max_iter=500):
    pop  = np.random.uniform(lb, ub, (pop_size, dim))
    fit  = np.array([fitness_func(x) for x in pop])
    best = pop[np.argmin(fit)].copy()
    best_fit = fit.min()
    conv = [best_fit]
    for _ in range(max_iter):
        new_pop = [best.copy()]
        while len(new_pop) < pop_size:
            p1 = pop[np.random.choice(pop_size, 3)][np.argmin(fit[np.random.choice(pop_size, 3)])]
            p2 = pop[np.random.choice(pop_size, 3)][np.argmin(fit[np.random.choice(pop_size, 3)])]
            eta = 20
            if np.random.rand() < 0.9:
                beta = np.array([(2*u)**(1/(eta+1)) if u<=0.5 else (2*(1-u))**(-1/(eta+1))
                                 for u in np.random.rand(dim)])
                c1 = np.clip(0.5*((1+beta)*p1+(1-beta)*p2), lb, ub)
                c2 = np.clip(0.5*((1-beta)*p1+(1+beta)*p2), lb, ub)
            else:
                c1, c2 = p1.copy(), p2.copy()
            for c in [c1, c2]:
                mask = np.random.rand(dim) < 0.1/dim
                for j in np.where(mask)[0]:
                    u = np.random.rand()
                    dj = (2*u)**(1/(eta+1))-1 if u<=0.5 else 1-(2*(1-u))**(1/(eta+1))
                    c[j] = np.clip(c[j]+dj*(ub-lb)/2, lb, ub)
                new_pop.append(c)
                if len(new_pop) >= pop_size: break
        pop = np.array(new_pop[:pop_size])
        fit = np.array([fitness_func(x) for x in pop])
        if fit.min() < best_fit:
            best_fit = fit.min(); best = pop[np.argmin(fit)].copy()
        conv.append(best_fit)
    return best, best_fit, conv


def run_mpa(fitness_func, dim, lb, ub, pop_size=50, max_iter=500):
    pop  = np.random.uniform(lb, ub, (pop_size, dim))
    fit  = np.array([fitness_func(x) for x in pop])
    best = pop[np.argmin(fit)].copy(); best_fit = fit.min()
    conv = [best_fit]; P = 0.5; FADs = 0.2
    for t in range(max_iter):
        CF = (1 - t/max_iter)**(2*t/max_iter)
        for i in range(pop_size):
            if t < max_iter/3:
                step = P * np.random.normal(0,1,dim)
                pop[i] = np.clip(pop[i] + step*(best-pop[i]), lb, ub)
            elif t < 2*max_iter/3:
                if i < pop_size//2:
                    pop[i] = np.clip(pop[i] + P*np.random.normal(0,1,dim)*(best-pop[i]), lb, ub)
                else:
                    pop[i] = np.clip(pop[i] + P*levy.rvs(size=dim)*(best-pop[i]), lb, ub)
            else:
                pop[i] = np.clip(best + P*levy.rvs(size=dim)*(best-pop[i]), lb, ub)
            if np.random.rand() < FADs:
                r = np.random.rand()
                u = np.random.uniform(-1,1,dim)
                pop[i] = np.clip(pop[i] + CF*(lb + r*(ub-lb)*u), lb, ub)
        fit = np.array([fitness_func(x) for x in pop])
        if fit.min() < best_fit:
            best_fit = fit.min(); best = pop[np.argmin(fit)].copy()
        conv.append(best_fit)
    return best, best_fit, conv


def run_ooa(fitness_func, dim, lb, ub, pop_size=50, max_iter=500):
    pop  = np.random.uniform(lb, ub, (pop_size, dim))
    fit  = np.array([fitness_func(x) for x in pop])
    best = pop[np.argmin(fit)].copy(); best_fit = fit.min()
    conv = [best_fit]
    for t in range(max_iter):
        for i in range(pop_size):
            PF = best + np.random.uniform(-0.5, 0.5, dim)
            IF = np.random.rand()
            if t < max_iter/2:
                pop[i] = np.clip(pop[i] + np.random.rand(dim)*(PF - IF*pop[i]), lb, ub)
            else:
                pop[i] = np.clip(pop[i] + np.random.rand(dim)*(best - pop[i]), lb, ub)
        fit = np.array([fitness_func(x) for x in pop])
        if fit.min() < best_fit:
            best_fit = fit.min(); best = pop[np.argmin(fit)].copy()
        conv.append(best_fit)
    return best, best_fit, conv


# ─────────────────────────────────────────────
# 4. HÀM CHẠY 1 CẤU HÌNH (seed cố định)
# ─────────────────────────────────────────────
def run_single(algo_fn, seed):
    np.random.seed(seed)
    return algo_fn()


# ─────────────────────────────────────────────
# 5. THỰC NGHIỆM CHÍNH (so sánh 5 thuật toán)
# ─────────────────────────────────────────────
def run_main_experiments(dim=30):
    print(f"\n{'='*70}")
    print(f"THỰC NGHIỆM CHÍNH — dim={dim}, pop={POP_SIZE}, iter={MAX_ITER}, runs={N_RUNS}")
    print(f"{'='*70}")
    cec = CEC2017Functions(dim=dim)

    algo_names = ['MPA-OOA', 'MPA', 'OOA', 'PSO', 'GA']
    all_rows   = []
    conv_data  = {fn: {a: [] for a in algo_names} for fn in cec.functions}
    raw_fitness = {fn: {a: [] for a in algo_names} for fn in cec.functions}

    for fn_name, fn in cec.functions.items():
        lb, ub = cec.bounds[fn_name]
        print(f"\n  Hàm {fn_name} ...", end='', flush=True)

        for run_idx, seed in enumerate(SEEDS):
            # MPA-OOA
            np.random.seed(seed)
            alg = MPA_OOA_Hybrid(POP_SIZE, MAX_ITER, dim, lb, ub)
            _, bf, cv = alg.optimize(fn)
            raw_fitness[fn_name]['MPA-OOA'].append(bf)
            conv_data[fn_name]['MPA-OOA'].append(cv)

            # MPA
            np.random.seed(seed)
            _, bf, cv = run_mpa(fn, dim, lb, ub, POP_SIZE, MAX_ITER)
            raw_fitness[fn_name]['MPA'].append(bf)
            conv_data[fn_name]['MPA'].append(cv)

            # OOA
            np.random.seed(seed)
            _, bf, cv = run_ooa(fn, dim, lb, ub, POP_SIZE, MAX_ITER)
            raw_fitness[fn_name]['OOA'].append(bf)
            conv_data[fn_name]['OOA'].append(cv)

            # PSO
            np.random.seed(seed)
            _, bf, cv = run_pso(fn, dim, lb, ub, POP_SIZE, MAX_ITER)
            raw_fitness[fn_name]['PSO'].append(bf)
            conv_data[fn_name]['PSO'].append(cv)

            # GA
            np.random.seed(seed)
            _, bf, cv = run_ga(fn, dim, lb, ub, POP_SIZE, MAX_ITER)
            raw_fitness[fn_name]['GA'].append(bf)
            conv_data[fn_name]['GA'].append(cv)

            print('.', end='', flush=True)
        print()

        # Thống kê + Wilcoxon so với MPA-OOA
        for alg in algo_names:
            vals = np.array(raw_fitness[fn_name][alg])
            row = {
                'Dim': dim, 'Function': fn_name, 'Algorithm': alg,
                'Mean': np.mean(vals), 'Std': np.std(vals),
                'Best': np.min(vals), 'Median': np.median(vals), 'Worst': np.max(vals),
                'Wilcoxon_p': np.nan, 'Wilcoxon_sig': ''
            }
            if alg != 'MPA-OOA':
                try:
                    ref = np.array(raw_fitness[fn_name]['MPA-OOA'])
                    _, p = wilcoxon(ref, vals)
                    row['Wilcoxon_p'] = p
                    row['Wilcoxon_sig'] = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
                except Exception:
                    pass
            all_rows.append(row)

    df_stats = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT_DIR, f"benchmark_stats_dim{dim}.csv")
    df_stats.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n  → Đã xuất: {out_csv}")
    return df_stats, raw_fitness, conv_data, cec


# ─────────────────────────────────────────────
# 6. FRIEDMAN + NEMENYI
# ─────────────────────────────────────────────
def run_friedman_nemenyi(raw_fitness, dim, algo_names, func_names):
    """Friedman test trên 9 hàm, Nemenyi post-hoc nếu có gói scikit_posthocs."""
    print(f"\n{'='*70}")
    print(f"FRIEDMAN TEST — dim={dim}")
    print(f"{'='*70}")

    # Ma trận rank: hàng = hàm, cột = thuật toán
    rank_matrix = []
    for fn in func_names:
        means = [np.mean(raw_fitness[fn][a]) for a in algo_names]
        ranks = pd.Series(means).rank().values
        rank_matrix.append(ranks)
    rank_matrix = np.array(rank_matrix)

    avg_ranks = rank_matrix.mean(axis=0)
    groups = [rank_matrix[:, j] for j in range(len(algo_names))]
    stat, p = friedmanchisquare(*groups)

    print(f"  Friedman χ²={stat:.4f}, p={p:.6f}")
    rows = [{'Algorithm': a, 'Avg_Rank': r} for a, r in zip(algo_names, avg_ranks)]

    # Nemenyi (tuỳ chọn)
    nemenyi_df = None
    try:
        import scikit_posthocs as sp
        data_2d = pd.DataFrame(rank_matrix, columns=algo_names)
        nemenyi_df = sp.posthoc_nemenyi_friedman(data_2d)
        out = os.path.join(OUT_DIR, f"nemenyi_dim{dim}.csv")
        nemenyi_df.to_csv(out, encoding='utf-8-sig')
        print(f"  → Nemenyi: {out}")
    except ImportError:
        print("  (scikit_posthocs chưa cài — bỏ qua Nemenyi)")

    df_fr = pd.DataFrame(rows)
    df_fr['Friedman_stat'] = stat
    df_fr['Friedman_p']    = p
    out = os.path.join(OUT_DIR, f"friedman_dim{dim}.csv")
    df_fr.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"  → Friedman ranks: {out}")
    return df_fr, nemenyi_df


# ─────────────────────────────────────────────
# 7. ABLATION STUDY (4 cơ chế, 5 hàm đại diện)
# ─────────────────────────────────────────────
ABLATION_FUNCS = ['F1', 'F3', 'F4', 'F8', 'F9']   # đại diện unimodal + multimodal

ABLATION_CONFIGS = {
    'Full (MPA-OOA)':     dict(use_levy_chaos=True,  use_obl=True,  use_dynamic_fads=True,  use_elite_pool=True),
    'w/o Lévy Chaos':     dict(use_levy_chaos=False, use_obl=True,  use_dynamic_fads=True,  use_elite_pool=True),
    'w/o OBL':            dict(use_levy_chaos=True,  use_obl=False, use_dynamic_fads=True,  use_elite_pool=True),
    'w/o Dynamic FADs':   dict(use_levy_chaos=True,  use_obl=True,  use_dynamic_fads=False, use_elite_pool=True),
    'w/o Elite Pool':     dict(use_levy_chaos=True,  use_obl=True,  use_dynamic_fads=True,  use_elite_pool=False),
}

def run_ablation(dim=30):
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY — dim={dim}, runs={N_RUNS}")
    print(f"{'='*70}")
    cec = CEC2017Functions(dim=dim)
    rows = []
    for fn_name in ABLATION_FUNCS:
        fn   = cec.functions[fn_name]
        lb, ub = cec.bounds[fn_name]
        print(f"\n  Hàm {fn_name}:")
        for cfg_name, flags in ABLATION_CONFIGS.items():
            vals = []
            for seed in SEEDS:
                np.random.seed(seed)
                alg = MPA_OOA_Hybrid(POP_SIZE, MAX_ITER, dim, lb, ub, **flags)
                _, bf, _ = alg.optimize(fn)
                vals.append(bf)
            vals = np.array(vals)
            rows.append({
                'Dim': dim, 'Function': fn_name, 'Config': cfg_name,
                'Mean': np.mean(vals), 'Std': np.std(vals),
                'Best': np.min(vals), 'Median': np.median(vals),
            })
            print(f"    {cfg_name:25s} mean={np.mean(vals):.4e}  std={np.std(vals):.2e}")

    df_abl = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, f"ablation_dim{dim}.csv")
    df_abl.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n  → Ablation: {out}")
    return df_abl


# ─────────────────────────────────────────────
# 8. VẼ HÌNH
# ─────────────────────────────────────────────
COLORS = {
    'MPA-OOA': '#1f77b4',
    'MPA':     '#ff7f0e',
    'OOA':     '#2ca02c',
    'PSO':     '#d62728',
    'GA':      '#9467bd',
}

def plot_convergence(conv_data, dim, cec):
    plt.rcParams.update({'font.size': 9, 'figure.dpi': 300})
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    axes = axes.flatten()
    fn_labels = {
        'F1':'Sphere','F2':'Zakharov','F3':'Rosenbrock',
        'F4':'Rastrigin','F5':'Lunacek bi-Rastrigin','F6':"Expanded Scaffer's F6",
        'F7':'Non-continuous Rastrigin','F8':'Levy','F9':'Schwefel',
    }
    algo_names = list(COLORS.keys())
    for idx, fn_name in enumerate(cec.functions.keys()):
        ax = axes[idx]
        for alg in algo_names:
            curves = conv_data[fn_name][alg]
            if not curves: continue
            min_len = min(len(c) for c in curves)
            mat = np.array([c[:min_len] for c in curves])
            avg = np.mean(mat, axis=0)
            clean = np.where(avg > 0, avg, 1e-300)
            ax.semilogy(clean, color=COLORS[alg], linewidth=1.4,
                        label=alg if idx == 0 else '_')
        ax.set_title(f"{fn_labels[fn_name]} ({fn_name})", fontsize=8, pad=4)
        ax.set_xlabel('Iteration' if idx >= 6 else '')
        ax.set_ylabel('Fitness' if idx % 3 == 0 else '')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(direction='in')
    handles = [plt.Line2D([0],[0], color=COLORS[a], linewidth=1.5, label=a)
               for a in algo_names]
    fig.legend(handles=handles, loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 1.01), fontsize=8, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT_DIR, f"convergence_dim{dim}.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  → Hội tụ: {out}")


def plot_boxplot(raw_fitness, dim, cec):
    algo_names = list(COLORS.keys())
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    axes = axes.flatten()
    for idx, fn_name in enumerate(cec.functions.keys()):
        ax = axes[idx]
        data   = [np.log10(np.array(raw_fitness[fn_name][a]) + 1e-300) for a in algo_names]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', linewidth=1.5))
        for patch, alg in zip(bp['boxes'], algo_names):
            patch.set_facecolor(COLORS[alg])
            patch.set_alpha(0.7)
        ax.set_title(fn_name, fontsize=9)
        ax.set_xticklabels(algo_names, fontsize=7, rotation=15)
        ax.set_ylabel('log₁₀(Fitness)' if idx % 3 == 0 else '')
        ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"boxplot_dim{dim}.png")
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  → Boxplot: {out}")


def plot_ablation(df_abl, dim):
    configs = list(ABLATION_CONFIGS.keys())
    fns     = ABLATION_FUNCS
    fig, axes = plt.subplots(1, len(fns), figsize=(14, 4), sharey=False)
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(configs)))
    for j, fn in enumerate(fns):
        ax = axes[j]
        sub = df_abl[df_abl['Function'] == fn]
        means = [sub[sub['Config']==c]['Mean'].values[0] for c in configs]
        bars = ax.bar(range(len(configs)), means, color=colors)
        ax.set_title(fn, fontsize=9)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([c.replace('w/o ','−') for c in configs],
                           fontsize=7, rotation=30, ha='right')
        ax.set_ylabel('Mean Fitness' if j == 0 else '')
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle(f'Ablation Study — dim={dim}', fontsize=10, y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"ablation_dim{dim}.png")
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  → Ablation plot: {out}")


def plot_avg_rank(df_fr, dim):
    df = df_fr[['Algorithm','Avg_Rank']].drop_duplicates().sort_values('Avg_Rank')
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = [COLORS.get(a, '#888888') for a in df['Algorithm']]
    bars = ax.barh(df['Algorithm'], df['Avg_Rank'], color=colors, alpha=0.8)
    ax.set_xlabel('Average Friedman Rank')
    ax.set_title(f'Average Ranking (Friedman) — dim={dim}')
    for bar, val in zip(bars, df['Avg_Rank']):
        ax.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=8)
    ax.set_xlim(0, len(df)+0.5)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"avg_rank_dim{dim}.png")
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  → Avg rank: {out}")


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    algo_names = ['MPA-OOA', 'MPA', 'OOA', 'PSO', 'GA']

    for dim in DIMS:
        # 9.1 Thực nghiệm chính
        df_stats, raw_fitness, conv_data, cec = run_main_experiments(dim=dim)

        # 9.2 Friedman + Nemenyi
        df_fr, _ = run_friedman_nemenyi(
            raw_fitness, dim, algo_names, list(cec.functions.keys()))

        # 9.3 Ablation
        df_abl = run_ablation(dim=dim)

        # 9.4 Hình
        print(f"\nVẽ hình dim={dim} ...")
        plot_convergence(conv_data, dim, cec)
        plot_boxplot(raw_fitness, dim, cec)
        plot_ablation(df_abl, dim)
        plot_avg_rank(df_fr, dim)

    print(f"\n{'='*70}")
    print(f"HOÀN TẤT — Tổng thời gian: {(time.time()-t0)/60:.1f} phút")
    print(f"Tất cả kết quả trong: {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
