"""
MPA-OOA EVCLP Analysis - Full Version (dựa trên ver7)
Bổ sung: Seed cố định, Wilcoxon (so 10 runs fitness), Ablation study (4 cơ chế)
Xuất tất cả kết quả ra CSV + hình PNG
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon
from scipy.special import gamma
from shapely.geometry import Point
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. CẤU HÌNH CHUNG
# ─────────────────────────────────────────────
SEEDS   = [42 + i * 7 for i in range(10)]   # 10 seeds cố định
N_RUNS  = 10
POP_SIZE_GA  = 20   # ver7: GA dùng 20 (như bài báo gốc)
POP_SIZE_PSO = 50
POP_SIZE_MPA = 70   # ver7: MPA-OOA dùng 70
MAX_ITER     = 1000

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. LẤY DỮ LIỆU GIS (giữ nguyên từ ver7)
# ─────────────────────────────────────────────
def get_hanoi_ring1_data(n_locations: int = 200,
                          n_demand_points: int = 100,
                          data_seed: int = 42):
    # Hộp giới hạn Vành đai 1 Hà Nội — tọa độ UTM32648 (mét) tương đương
    # lat 20.9–21.1, lon 105.7–105.9
    # Dùng pyproj để chuyển bbox sang UTM thay vì osmnx (tránh lỗi API version)
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    lon_min, lat_min = 105.7, 20.9
    lon_max, lat_max = 105.9, 21.1
    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)

    poi_utm, centroid_utm = np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)

    # Cache file — đảm bảo dữ liệu GIS không thay đổi giữa các lần chạy
    _cache_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(_cache_dir, exist_ok=True)
    _cache_file = os.path.join(_cache_dir, "hanoi_ring1_cache.npz")

    if os.path.exists(_cache_file):
        cached = np.load(_cache_file)
        poi_utm      = cached['poi_utm']
        centroid_utm = cached['centroid_utm']
        print(f"Cache GIS: {len(poi_utm)} POI, {len(centroid_utm)} way centroids")
    else:
        # Lấy lần đầu từ Overpass rồi lưu cache
        try:
            import requests
            headers = {'User-Agent': 'MPA-OOA-EVCLP-Research/1.0 (academic)'}
            query = """
            [out:json][timeout:60];
            (
              node["amenity"~"^(fuel|charging_station|parking)$"](20.9,105.7,21.1,105.9);
              way["highway"]["highway"!="footway"]["highway"!="path"](20.9,105.7,21.1,105.9);
            );
            out center;
            """
            resp = requests.post(
                'https://overpass-api.de/api/interpreter',
                data={'data': query}, headers=headers, timeout=90)
            resp.raise_for_status()
            data = resp.json()

            poi_coords, way_centroids = [], []
            for el in data.get('elements', []):
                if el['type'] == 'node':
                    lat, lon = el['lat'], el['lon']
                    poi_coords.append(transformer.transform(lon, lat))
                elif el['type'] == 'way' and 'center' in el:
                    lat, lon = el['center']['lat'], el['center']['lon']
                    way_centroids.append(transformer.transform(lon, lat))

            if poi_coords:
                poi_utm = np.array(poi_coords)
            if way_centroids:
                centroid_utm = np.array(way_centroids)
            np.savez(_cache_file, poi_utm=poi_utm, centroid_utm=centroid_utm)
            print(f"Overpass OK + cached: {len(poi_utm)} POI, {len(centroid_utm)} way centroids")
        except Exception as e:
            print(f"Overpass API không khả dụng ({e}). Dùng tọa độ UTM ngẫu nhiên.")

    # Dùng RandomState riêng để sampling không ảnh hưởng global seed tối ưu
    rng_data = np.random.RandomState(data_seed)

    def _sample(arr, n):
        arr = np.array(arr) if not isinstance(arr, np.ndarray) else arr
        m = len(arr)
        if m >= n:
            return arr[rng_data.choice(m, n, replace=False)]
        missing = n - m
        extra = np.column_stack([
            rng_data.uniform(x_min, x_max, missing),
            rng_data.uniform(y_min, y_max, missing),
        ])
        return np.vstack([arr, extra]) if m > 0 else extra

    return _sample(poi_utm, n_locations), _sample(centroid_utm, n_demand_points)


# ─────────────────────────────────────────────
# 2. BÀI TOÁN EVCLP (giữ nguyên ver7)
# ─────────────────────────────────────────────
class EVCLP:
    def __init__(self, n_locations=200, n_demand_points=100, data_seed=42):
        self.n_locations     = n_locations
        self.n_demand_points = n_demand_points

        self.locations, self.demand_points = get_hanoi_ring1_data(
            n_locations, n_demand_points, data_seed)

        minx = np.min(self.locations[:, 0]); maxx = np.max(self.locations[:, 0])
        miny = np.min(self.locations[:, 1]); maxy = np.max(self.locations[:, 1])
        self.area_size = max(maxx-minx, maxy-miny)

        # Seed riêng cho dữ liệu (độc lập với seed tối ưu)
        rng = np.random.RandomState(data_seed)
        self.construction_costs  = rng.uniform(500, 1000, n_locations)
        self.operation_costs     = self.construction_costs * rng.uniform(0.1, 0.2, n_locations)
        self.demands             = rng.uniform(10, 50, n_demand_points)
        self.station_capacities  = rng.uniform(50, 120, n_locations)
        self.grid_capacities     = rng.uniform(60, 150, n_locations)

        self.coverage_radius    = 5000
        self.budget             = 100_000
        self.total_grid_capacity = 10_000
        self.w1, self.w2, self.w3, self.w4 = 0.3, 0.3, 0.2, 0.2

        # Cache ma trận khoảng cách (tính 1 lần)
        self._dist_matrix = self._calc_dist()

    def _calc_dist(self):
        dm = np.zeros((self.n_locations, self.n_demand_points))
        for i in range(self.n_locations):
            for j in range(self.n_demand_points):
                dm[i, j] = np.linalg.norm(self.locations[i] - self.demand_points[j])
        return dm

    def calculate_distance_matrix(self):
        return self._dist_matrix

    def objective_function(self, X):
        X = np.clip(np.round(X), 0, 1).astype(int)
        dm = self._dist_matrix

        C_cost = (np.sum(self.construction_costs*X) +
                  np.sum(self.operation_costs*X)) / self.budget

        cov_mat = (dm <= self.coverage_radius).astype(int)
        covered = np.array([np.any(cov_mat[:, j]*X) for j in range(self.n_demand_points)])
        C_cov   = 1.0 - np.mean(covered)

        active = np.where(X == 1)[0]
        if len(active) > 0:
            C_acc = np.sum([np.min(dm[active, j])*self.demands[j]
                            for j in range(self.n_demand_points)])
            C_acc /= (self.area_size * np.sum(self.demands))
        else:
            C_acc = 1.0

        C_grid = np.sum([max(0, self.station_capacities[i]-self.grid_capacities[i])
                         for i in range(self.n_locations) if X[i] == 1]) / self.total_grid_capacity

        F = self.w1*C_cost + self.w2*C_cov + self.w3*C_acc + self.w4*C_grid

        pen = 0.0
        if np.sum(self.construction_costs*X) > self.budget:
            pen += 10*(np.sum(self.construction_costs*X)-self.budget)/self.budget
        if np.sum(self.station_capacities*X) > self.total_grid_capacity:
            pen += 10*(np.sum(self.station_capacities*X)-self.total_grid_capacity)/self.total_grid_capacity
        return F + pen

    def get_metrics(self, X):
        X = np.clip(np.round(X), 0, 1).astype(int)
        sel = np.where(X == 1)[0]
        n   = len(sel)
        dm  = self._dist_matrix
        cov_mat = (dm <= self.coverage_radius).astype(int)
        covered = np.mean([np.any(cov_mat[sel, j]) for j in range(self.n_demand_points)]) * 100 if n > 0 else 0
        avg_d   = np.mean([np.min(dm[sel, j]) for j in range(self.n_demand_points)]) if n > 0 else float('inf')
        return {'n_stations': n, 'coverage': covered, 'avg_distance': avg_d}


# ─────────────────────────────────────────────
# 3. MPA-OOA VỚI FLAG ABLATION (dựa trên ver7)
# ─────────────────────────────────────────────
class MPA_OOA:
    def __init__(self, problem: EVCLP, pop_size=50, max_iter=1000,
                 use_levy_chaos=True, use_obl=True,
                 use_dynamic_fads=True, use_elite_pool=True):
        self.problem = problem
        self.pop_size   = pop_size
        self.max_iter   = max_iter
        self.dim        = problem.n_locations
        self.lb         = np.zeros(self.dim)
        self.ub         = np.ones(self.dim)
        self.P          = 0.1
        self.use_levy_chaos   = use_levy_chaos
        self.use_obl          = use_obl
        self.use_dynamic_fads = use_dynamic_fads
        self.use_elite_pool   = use_elite_pool

        self.n_elite = max(3, int(0.1*pop_size))
        self.stagnation_counter = 0
        self.best_improvement_iter = 0
        self.convergence_history = []

    def _levy_flight(self):
        beta  = 1.5
        sigma = (gamma(1+beta)*np.sin(np.pi*beta/2) /
                 (gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        u = np.random.randn(self.dim)*sigma
        v = np.random.randn(self.dim)
        step = 0.05*u/(np.abs(v)**(1/beta))
        if self.use_levy_chaos:
            r = np.random.rand()
            step = step * 4*r*(1-r)
        return step

    def _obl(self, sol, E_f):
        if not self.use_obl:
            return np.random.rand(self.dim)
        alpha = 0.1*E_f
        return np.clip(self.lb + self.ub - sol + alpha*np.random.randn(self.dim),
                       self.lb, self.ub)

    def _explore_factor(self, t):
        E_base = 1 - t/self.max_iter
        E_stag = min(0.4, self.stagnation_counter*0.05)
        centroid = np.mean(self.population, axis=0)
        div  = np.mean([np.linalg.norm(s - centroid) for s in self.population])
        E_div = min(0.3, div/np.linalg.norm(self.ub - self.lb))
        return min(0.95, E_base + E_stag + E_div)

    def _dynamic_fads(self, sol, t, E_f):
        P_fads = (0.15 + 0.1*(1-E_f)) if self.use_dynamic_fads else 0.2
        if np.random.rand() >= P_fads:
            return sol
        r = np.random.rand()
        if r < 0.3:
            return np.random.rand(self.dim)
        elif r < 0.6 and self.use_elite_pool and len(self.elite_solutions) > 0:
            idx = np.random.randint(self.n_elite)
            return np.clip(self.elite_solutions[idx]+np.random.randn(self.dim)*0.1,
                           self.lb, self.ub)
        elif r < 0.8:
            return np.clip(self.best_solution + self._levy_flight(), self.lb, self.ub)
        else:
            return self._obl(sol, E_f)

    def _get_elite(self):
        if self.use_elite_pool and self.elite_solutions is not None:
            return self.elite_solutions[np.random.randint(self.n_elite)]
        return self.best_solution

    def _update_elite(self):
        if not self.use_elite_pool: return
        combined_pop = np.vstack([self.population, self.elite_solutions])
        combined_fit = np.concatenate([self.fitness, self.elite_fitness])
        idx = np.argsort(combined_fit)[:self.n_elite]
        self.elite_solutions = combined_pop[idx].copy()
        self.elite_fitness   = combined_fit[idx].copy()

    def run(self):
        lb, ub = self.lb, self.ub
        # Khởi tạo: 70% random + 30% OBL
        n_rand = int(0.7*self.pop_size)
        self.population = np.vstack([
            np.random.rand(n_rand, self.dim),
            np.array([lb + ub - np.random.rand(self.dim)
                      for _ in range(self.pop_size - n_rand)])
        ])
        self.population = np.clip(self.population, lb, ub)
        self.fitness = np.array([self.problem.objective_function(s) for s in self.population])

        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness  = self.fitness[best_idx]

        elite_idx = np.argsort(self.fitness)[:self.n_elite]
        self.elite_solutions = self.population[elite_idx].copy()
        self.elite_fitness   = self.fitness[elite_idx].copy()
        self.convergence_history = [self.best_fitness]

        for t in range(self.max_iter):
            E_f = self._explore_factor(t)
            old_best = self.best_fitness
            n1 = int(0.4*self.pop_size)
            n2 = int(0.8*self.pop_size)

            for i in range(self.pop_size):
                if E_f > 0.5:
                    if i < n1:
                        elite = self._get_elite()
                        new = np.clip(self.population[i] +
                                      0.5*self._levy_flight()*(elite - self.population[i]),
                                      lb, ub)
                    elif i < n2:
                        fish = (self.population[i]
                                if self.fitness[i] < self.best_fitness
                                else self.best_solution)
                        IF   = np.random.randint(1, 3)
                        noise = np.random.randn(self.dim)*0.01
                        new  = np.clip(self.population[i] +
                                       np.random.rand(self.dim)*(fish - IF*self.population[i]) + noise,
                                       lb, ub)
                    else:
                        elite = self._get_elite()
                        obl   = self._obl(self.population[i], E_f)
                        lev   = self._levy_flight()
                        new   = np.clip(0.6*elite +
                                        0.3*(self.population[i]+0.3*lev) + 0.1*obl,
                                        lb, ub)
                else:
                    if i < self.pop_size//2:
                        ref = self.population[
                            np.argmin(self.fitness[
                                np.random.choice(self.pop_size, min(3,self.pop_size), replace=False)])]
                        new = np.clip(self.population[i] +
                                      np.random.rand(self.dim)*(ref - self.population[i]),
                                      lb, ub)
                    else:
                        r = np.random.rand(self.dim)
                        new = np.clip(self.population[i] +
                                      self.P*r*(self.best_solution - self.population[i]),
                                      lb, ub)

                new = self._dynamic_fads(new, t, E_f)
                new_fit = self.problem.objective_function(new)
                if new_fit < self.fitness[i]:
                    self.population[i] = new
                    self.fitness[i]    = new_fit
                    if new_fit < self.best_fitness:
                        self.best_fitness  = new_fit
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
                    (np.linalg.norm(self.best_solution)+1e-10)
                    for s in self.population])
                if sigma_c < 0.01:
                    worst = np.argsort(self.fitness)[-int(0.2*self.pop_size):]
                    for w in worst:
                        self.population[w] = np.random.rand(self.dim)
                        self.fitness[w]    = self.problem.objective_function(self.population[w])
                    self.stagnation_counter = 0

            self.convergence_history.append(self.best_fitness)
            if (t+1) % 200 == 0:
                print(f"      iter {t+1}/{self.max_iter}: fitness={self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness


# ─────────────────────────────────────────────
# 4. GA — copy nguyên xi từ ver7 (pop_size=20)
# ─────────────────────────────────────────────
class GeneticAlgorithm:
    def __init__(self, problem: EVCLP, pop_size: int = 20, max_iter: int = 1000):
        self.problem = problem
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = problem.n_locations
        self.pc = 0.8
        self.pm = 0.1
        self.elite_size = max(1, int(0.1 * pop_size))
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_history = []

    def initialize_population(self):
        self.population = np.random.rand(self.pop_size, self.dim)
        self.fitness = np.array([self.problem.objective_function(sol) for sol in self.population])
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

    def tournament_selection(self, k=3):
        parents = []
        n_parents_needed = self.pop_size - self.elite_size
        for _ in range(n_parents_needed):
            k_actual = min(k, len(self.fitness))
            candidates = np.random.choice(len(self.fitness), k_actual, replace=False)
            winner_idx = candidates[np.argmin(self.fitness[candidates])]
            parents.append(self.population[winner_idx].copy())
        return np.array(parents)

    def uniform_crossover(self, parent1, parent2):
        if np.random.rand() > self.pc:
            return parent1.copy(), parent2.copy()
        mask = np.random.rand(self.dim) < 0.5
        child1 = parent1.copy(); child2 = parent2.copy()
        child1[mask] = parent2[mask]; child2[mask] = parent1[mask]
        return child1, child2

    def gaussian_mutation(self, individual):
        mutant = individual.copy()
        for i in range(self.dim):
            if np.random.rand() < self.pm:
                mutant[i] += np.random.normal(0, 0.1)
                mutant[i] = np.clip(mutant[i], 0, 1)
        return mutant

    def run(self):
        self.initialize_population()
        self.convergence_history = [self.best_fitness]
        for t in range(self.max_iter):
            elite_idx = np.argsort(self.fitness)[:self.elite_size]
            elite_pop = self.population[elite_idx].copy()
            elite_fitness = self.fitness[elite_idx].copy()
            parents = self.tournament_selection()
            offspring = []; offspring_fitness = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self.uniform_crossover(parents[i], parents[i+1])
                    child1 = self.gaussian_mutation(child1)
                    child2 = self.gaussian_mutation(child2)
                    offspring.extend([child1, child2])
                    offspring_fitness.extend([
                        self.problem.objective_function(child1),
                        self.problem.objective_function(child2)
                    ])
                else:
                    child = self.gaussian_mutation(parents[i])
                    offspring.append(child)
                    offspring_fitness.append(self.problem.objective_function(child))
            n_offspring_needed = self.pop_size - self.elite_size
            while len(offspring) < n_offspring_needed:
                new_individual = np.random.rand(self.dim)
                offspring.append(new_individual)
                offspring_fitness.append(self.problem.objective_function(new_individual))
            offspring = offspring[:n_offspring_needed]
            offspring_fitness = offspring_fitness[:n_offspring_needed]
            new_population = np.vstack([elite_pop, offspring])
            new_fitness = np.concatenate([elite_fitness, offspring_fitness])
            self.population = new_population
            self.fitness = new_fitness
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_solution = self.population[current_best_idx].copy()
                self.best_fitness = self.fitness[current_best_idx]
            self.convergence_history.append(self.best_fitness)
            if (t + 1) % 200 == 0:
                print(f"      iter {t+1}/{self.max_iter}: fitness={self.best_fitness:.6f}")
        return self.best_solution, self.best_fitness


# ─────────────────────────────────────────────
# 5. PSO — copy nguyên xi từ ver7 (pop_size=50)
# ─────────────────────────────────────────────
class ParticleSwarmOptimization:
    def __init__(self, problem: EVCLP, pop_size: int = 50, max_iter: int = 1000):
        self.problem = problem
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = problem.n_locations
        self.w_max = 0.9
        self.w_min = 0.2
        self.c1 = 2.0
        self.c2 = 2.0
        self.v_max = 0.5
        self.positions = None; self.velocities = None
        self.fitness = None
        self.pbest_positions = None; self.pbest_fitness = None
        self.gbest_position = None; self.gbest_fitness = float('inf')
        self.convergence_history = []

    def initialize_swarm(self):
        self.positions = np.random.rand(self.pop_size, self.dim)
        self.velocities = np.random.uniform(-self.v_max, self.v_max, (self.pop_size, self.dim))
        self.fitness = np.array([self.problem.objective_function(pos) for pos in self.positions])
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = self.fitness.copy()
        best_idx = np.argmin(self.fitness)
        self.gbest_position = self.positions[best_idx].copy()
        self.gbest_fitness = self.fitness[best_idx]

    def update_inertia_weight(self, t):
        return self.w_max - (self.w_max - self.w_min) * t / self.max_iter

    def run(self):
        self.initialize_swarm()
        self.convergence_history = [self.gbest_fitness]
        for t in range(self.max_iter):
            w = self.update_inertia_weight(t)
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2, self.dim)
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social    = self.c2 * r2 * (self.gbest_position - self.positions[i])
                self.velocities[i] = w * self.velocities[i] + cognitive + social
                self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], 0, 1)
                new_fitness = self.problem.objective_function(self.positions[i])
                self.fitness[i] = new_fitness
                if new_fitness < self.pbest_fitness[i]:
                    self.pbest_positions[i] = self.positions[i].copy()
                    self.pbest_fitness[i] = new_fitness
                if new_fitness < self.gbest_fitness:
                    self.gbest_position = self.positions[i].copy()
                    self.gbest_fitness = new_fitness
            self.convergence_history.append(self.gbest_fitness)
            if (t + 1) % 200 == 0:
                print(f"      iter {t+1}/{self.max_iter}: fitness={self.gbest_fitness:.6f}")
        self.best_solution = self.gbest_position
        self.best_fitness  = self.gbest_fitness
        return self.gbest_position, self.gbest_fitness


# ─────────────────────────────────────────────
# 5b. GREEDY (set-cover cổ điển) — heuristic đơn giản đối chứng
# ─────────────────────────────────────────────
class GreedyConstructive:
    """Tham lam kinh điển kiểu set-cover: ở mỗi bước chọn vị trí có tỷ số
    (số điểm nhu cầu MỚI được phủ trong bán kính coverage_radius) / (chi phí xây dựng)
    lớn nhất, thêm vào tập chọn, dừng khi hết ngân sách hoặc đã phủ hết. Thuật toán
    CHỈ nhìn tiêu chí phủ sóng/chi phí đơn lẻ — không đánh giá lại toàn bộ hàm mục
    tiêu đa thành phần (cost, coverage, access, grid) ở từng bước như MPA-OOA, nên
    dễ cho nghiệm mất cân đối giữa các tiêu chí phụ (accessibility, tải lưới điện)
    dù chi phí/độ phủ có vẻ tốt — đại diện nhóm heuristic tham lam đơn giản, tất định,
    một tiêu chí duy nhất, thường dùng làm baseline cho bài toán facility location."""

    def __init__(self, problem: EVCLP, max_iter: int = 1000):
        self.problem = problem
        self.max_iter = max_iter          # chỉ dùng để đệm convergence_history cho đồng bộ đồ thị
        self.dim = problem.n_locations
        self.best_solution = None
        self.best_fitness  = float('inf')
        self.convergence_history = []

    def run(self):
        problem = self.problem
        dm = problem._dist_matrix
        cov_mat = (dm <= problem.coverage_radius)   # (n_locations, n_demand_points) bool

        X = np.zeros(self.dim)
        covered = np.zeros(problem.n_demand_points, dtype=bool)
        remaining = set(range(self.dim))
        spent = 0.0

        history = [problem.objective_function(X)]

        while remaining:
            best_ratio, best_idx, best_new_cov = -1.0, -1, None
            for idx in remaining:
                cost = problem.construction_costs[idx]
                if spent + cost > problem.budget:
                    continue
                newly = cov_mat[idx] & (~covered)
                n_new = np.count_nonzero(newly)
                if n_new == 0:
                    continue
                ratio = n_new / cost          # điểm nhu cầu mới phủ được / đơn vị chi phí
                if ratio > best_ratio:
                    best_ratio, best_idx, best_new_cov = ratio, idx, newly

            if best_idx < 0:
                break   # không còn vị trí nào vừa ngân sách vừa phủ thêm được điểm mới

            X[best_idx] = 1
            covered |= best_new_cov
            spent += problem.construction_costs[best_idx]
            remaining.discard(best_idx)
            history.append(problem.objective_function(X))

            if covered.all():
                break

        current_fit = problem.objective_function(X)
        while len(history) < self.max_iter + 1:
            history.append(current_fit)

        self.best_solution = X.copy()
        self.best_fitness  = current_fit
        self.convergence_history = history
        return self.best_solution, self.best_fitness


# ─────────────────────────────────────────────
# 5c. HILL CLIMBING — heuristic đơn giản đối chứng
# ─────────────────────────────────────────────
class HillClimbing:
    """Leo đồi ngẫu nhiên bit-flip: mỗi vòng lặp thử lật 1 bit ngẫu nhiên, chấp nhận
    nếu cải thiện fitness (first-improvement), không có cơ chế thoát cực trị địa phương
    (không Lévy flight, không FADs, không elite pool) — đại diện nhóm heuristic
    tìm kiếm địa phương thuần túy để đối chứng với metaheuristic lai MPA-OOA."""

    def __init__(self, problem: EVCLP, max_iter: int = 1000):
        self.problem = problem
        self.max_iter = max_iter
        self.dim = problem.n_locations
        self.best_solution = None
        self.best_fitness  = float('inf')
        self.convergence_history = []

    def run(self):
        X = (np.random.rand(self.dim) < 0.5).astype(float)
        current_fit = self.problem.objective_function(X)
        history = [current_fit]

        for t in range(self.max_iter):
            idx = np.random.randint(self.dim)
            X[idx] = 1.0 - X[idx]
            f = self.problem.objective_function(X)
            if f < current_fit:
                current_fit = f
            else:
                X[idx] = 1.0 - X[idx]   # revert nếu không cải thiện
            history.append(current_fit)

        self.best_solution = X.copy()
        self.best_fitness  = current_fit
        self.convergence_history = history
        return self.best_solution, self.best_fitness


# ─────────────────────────────────────────────
# 5d. CLUSTERING (K-means) — heuristic đơn giản đối chứng
# ─────────────────────────────────────────────
class ClusteringHeuristic:
    """Phân cụm K-means các điểm nhu cầu thành K cụm (K ước lượng theo tỷ lệ diện
    tích phủ được / tổng diện tích), rồi với mỗi cụm chọn vị trí ứng viên gần
    trọng tâm cụm nhất còn nằm trong ngân sách làm trạm sạc. Không tối ưu hóa lặp,
    không xét tương tác giữa các trạm (overlap coverage, tải lưới điện dồn) — đại
    diện nhóm heuristic phân cụm không gian thường dùng để khởi tạo vị trí cơ sở hạ
    tầng, cài đặt K-means thuần numpy (không phụ thuộc scikit-learn)."""

    def __init__(self, problem: EVCLP, max_iter: int = 1000, kmeans_iter: int = 50):
        self.problem = problem
        self.max_iter = max_iter
        self.kmeans_iter = kmeans_iter
        self.dim = problem.n_locations
        self.best_solution = None
        self.best_fitness  = float('inf')
        self.convergence_history = []

    def _kmeans(self, points, k):
        n = len(points)
        k = min(k, n)
        idx0 = np.random.choice(n, k, replace=False)
        centroids = points[idx0].copy()
        for _ in range(self.kmeans_iter):
            d = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(d, axis=1)
            new_centroids = centroids.copy()
            for c in range(k):
                members = points[labels == c]
                if len(members) > 0:
                    new_centroids[c] = members.mean(axis=0)
            if np.allclose(new_centroids, centroids):
                centroids = new_centroids
                break
            centroids = new_centroids
        return centroids

    def run(self):
        problem = self.problem
        # số cụm ước lượng: diện tích phủ 1 trạm (πr²) so với tổng diện tích khu vực
        area_total = problem.area_size ** 2
        area_per_station = np.pi * problem.coverage_radius ** 2
        k = max(1, min(self.dim, int(np.ceil(area_total / area_per_station))))

        centroids = self._kmeans(problem.demand_points, k)

        X = np.zeros(self.dim)
        spent = 0.0
        used = set()
        for c in centroids:
            d_to_c = np.linalg.norm(problem.locations - c[None, :], axis=1)
            order = np.argsort(d_to_c)
            for idx in order:
                if idx in used:
                    continue
                cost = problem.construction_costs[idx]
                if spent + cost > problem.budget:
                    continue
                X[idx] = 1
                used.add(idx)
                spent += cost
                break

        current_fit = problem.objective_function(X)
        history = [current_fit] * (self.max_iter + 1)

        self.best_solution = X.copy()
        self.best_fitness  = current_fit
        self.convergence_history = history
        return self.best_solution, self.best_fitness


# ─────────────────────────────────────────────
# 6. HÀM CHẠY N LẦN VỚI SEED CỐ ĐỊNH
# ─────────────────────────────────────────────
def run_n_times(AlgoClass, problem, algo_params, n_runs=N_RUNS):
    fitness_list, conv_list, time_list = [], [], []
    for run_idx, seed in enumerate(SEEDS[:n_runs]):
        np.random.seed(seed)
        t0  = time.time()
        alg = AlgoClass(problem, **algo_params)
        _, bf = alg.run()
        elapsed = time.time() - t0
        fitness_list.append(bf)
        conv_list.append(alg.convergence_history)
        time_list.append(elapsed)
        print(f"    run {run_idx+1}/{n_runs} seed={seed}: "
              f"fitness={bf:.6f}  time={elapsed:.1f}s")
    return fitness_list, conv_list, time_list


# ─────────────────────────────────────────────
# 7. THỰC NGHIỆM CHÍNH (GA, PSO, MPA-OOA)
# ─────────────────────────────────────────────
def run_main_experiments(problem: EVCLP):
    print(f"\n{'='*70}")
    print("THỰC NGHIỆM CHÍNH — EVCLP Vành đai 1 Hà Nội")
    print(f"{'='*70}")

    configs = {
        'Greedy':       (GreedyConstructive,        {'max_iter': MAX_ITER}),
        'Clustering':   (ClusteringHeuristic,        {'max_iter': MAX_ITER}),
        'HillClimbing': (HillClimbing,               {'max_iter': MAX_ITER}),
        'GA':           (GeneticAlgorithm,           {'pop_size': POP_SIZE_GA,  'max_iter': MAX_ITER}),
        'PSO':          (ParticleSwarmOptimization,  {'pop_size': POP_SIZE_PSO, 'max_iter': MAX_ITER}),
        'MPA-OOA':      (MPA_OOA,                    {'pop_size': POP_SIZE_MPA, 'max_iter': MAX_ITER}),
    }

    all_results = {}
    stat_rows   = []

    for name, (cls, params) in configs.items():
        print(f"\n  Thuật toán: {name}")
        fit_list, conv_list, time_list = run_n_times(cls, problem, params)
        all_results[name] = {'fitness': fit_list, 'conv': conv_list, 'time': time_list}
        stat_rows.append({
            'Algorithm': name,
            'Mean_Fitness': np.mean(fit_list), 'Std_Fitness': np.std(fit_list),
            'Best_Fitness': np.min(fit_list),  'Worst_Fitness': np.max(fit_list),
            'Mean_Time_s':  np.mean(time_list),
        })

    # Wilcoxon: MPA-OOA vs từng thuật toán đối chứng (heuristic đơn giản + metaheuristic)
    ref = np.array(all_results['MPA-OOA']['fitness'])
    wilcoxon_rows = []
    for comp in ['Greedy', 'Clustering', 'HillClimbing', 'GA', 'PSO']:
        comp_vals = np.array(all_results[comp]['fitness'])
        try:
            stat, p = wilcoxon(ref, comp_vals)
            sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
        except Exception:
            stat, p, sig = np.nan, np.nan, 'N/A'
        wilcoxon_rows.append({
            'Comparison': f'MPA-OOA vs {comp}',
            'W_stat': stat, 'p_value': p, 'Significance': sig
        })

    df_stats    = pd.DataFrame(stat_rows)
    df_wilcoxon = pd.DataFrame(wilcoxon_rows)

    df_stats.to_csv(os.path.join(OUT_DIR, "main_stats.csv"), index=False, encoding='utf-8-sig')
    df_wilcoxon.to_csv(os.path.join(OUT_DIR, "wilcoxon_evclp.csv"), index=False, encoding='utf-8-sig')

    # Raw fitness mỗi run
    raw_rows = []
    for name, res in all_results.items():
        for i, (f, t) in enumerate(zip(res['fitness'], res['time'])):
            raw_rows.append({'Algorithm': name, 'Run': i+1, 'Seed': SEEDS[i],
                             'Fitness': f, 'Time_s': t})
    pd.DataFrame(raw_rows).to_csv(
        os.path.join(OUT_DIR, "raw_fitness_runs.csv"), index=False, encoding='utf-8-sig')

    print(f"\n  → Kết quả chính: {OUT_DIR}/main_stats.csv")
    print(f"  → Wilcoxon: {OUT_DIR}/wilcoxon_evclp.csv")
    return all_results, df_stats, df_wilcoxon


# ─────────────────────────────────────────────
# 8. ABLATION STUDY (4 cơ chế, MPA-OOA)
# ─────────────────────────────────────────────
ABLATION_CONFIGS = {
    'Full (MPA-OOA)':   dict(use_levy_chaos=True,  use_obl=True,  use_dynamic_fads=True,  use_elite_pool=True),
    'w/o Lévy Chaos':   dict(use_levy_chaos=False, use_obl=True,  use_dynamic_fads=True,  use_elite_pool=True),
    'w/o OBL':          dict(use_levy_chaos=True,  use_obl=False, use_dynamic_fads=True,  use_elite_pool=True),
    'w/o Dynamic FADs': dict(use_levy_chaos=True,  use_obl=True,  use_dynamic_fads=False, use_elite_pool=True),
    'w/o Elite Pool':   dict(use_levy_chaos=True,  use_obl=True,  use_dynamic_fads=True,  use_elite_pool=False),
}

def run_ablation(problem: EVCLP):
    print(f"\n{'='*70}")
    print("ABLATION STUDY — EVCLP")
    print(f"{'='*70}")
    rows = []
    for cfg_name, flags in ABLATION_CONFIGS.items():
        print(f"\n  Config: {cfg_name}")
        fit_list, _, time_list = run_n_times(
            MPA_OOA, problem,
            {'pop_size': POP_SIZE_MPA, 'max_iter': MAX_ITER, **flags})
        rows.append({
            'Config': cfg_name,
            'Mean_Fitness': np.mean(fit_list), 'Std_Fitness': np.std(fit_list),
            'Best_Fitness': np.min(fit_list),  'Median_Fitness': np.median(fit_list),
            'Mean_Time_s':  np.mean(time_list),
        })
    df_abl = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "ablation_evclp.csv")
    df_abl.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n  → Ablation: {out}")
    return df_abl


# ─────────────────────────────────────────────
# 9. VẼ HÌNH
# ─────────────────────────────────────────────
COLORS = {
    'Greedy': '#8c564b', 'Clustering': '#7f7f7f', 'HillClimbing': '#e377c2',
    'GA': '#1f77b4', 'PSO': '#d62728', 'MPA-OOA': '#2ca02c',
}

def plot_convergence(all_results):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, res in all_results.items():
        curves  = res['conv']
        min_len = min(len(c) for c in curves)
        mat     = np.array([c[:min_len] for c in curves])
        avg     = np.mean(mat, axis=0)
        ax.plot(avg, color=COLORS[name], linewidth=2, label=name)
    ax.set_xlabel('Số lần lặp')
    ax.set_ylabel('Giá trị hàm mục tiêu')
    ax.set_title('Đường hội tụ các thuật toán — EVCLP')
    ax.legend(); ax.grid(True, alpha=0.3)
    out = os.path.join(OUT_DIR, "convergence_evclp.png")
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  → Hội tụ: {out}")


def plot_fitness_bar(df_stats):
    fig, ax = plt.subplots(figsize=(5, 4))
    algos = df_stats['Algorithm'].tolist()
    vals  = df_stats['Mean_Fitness'].tolist()
    bars  = ax.bar(algos, vals, color=[COLORS[a] for a in algos], alpha=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, f'{v:.5f}',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Mean Fitness')
    ax.set_title('So sánh fitness trung bình')
    ax.grid(axis='y', alpha=0.3)
    out = os.path.join(OUT_DIR, "fitness_bar_evclp.png")
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  → Fitness bar: {out}")


def plot_time_bar(df_stats):
    fig, ax = plt.subplots(figsize=(5, 4))
    algos = df_stats['Algorithm'].tolist()
    vals  = df_stats['Mean_Time_s'].tolist()
    bars  = ax.bar(algos, vals, color=[COLORS[a] for a in algos], alpha=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, f'{v:.1f}s',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Thời gian TB (giây)')
    ax.set_title('Thời gian chạy trung bình')
    ax.grid(axis='y', alpha=0.3)
    out = os.path.join(OUT_DIR, "time_bar_evclp.png")
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  → Time bar: {out}")


def plot_ablation(df_abl):
    fig, ax = plt.subplots(figsize=(7, 4))
    configs = df_abl['Config'].tolist()
    means   = df_abl['Mean_Fitness'].tolist()
    stds    = df_abl['Std_Fitness'].tolist()
    colors  = plt.cm.Set2(np.linspace(0, 0.8, len(configs)))
    bars = ax.bar(range(len(configs)), means, yerr=stds, color=colors,
                  capsize=4, alpha=0.85, error_kw={'linewidth':1.2})
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([c.replace('w/o ','−') for c in configs],
                       rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('Mean Fitness')
    ax.set_title('Ablation Study — EVCLP')
    ax.grid(axis='y', alpha=0.3)
    # đánh dấu Full là baseline
    ax.axhline(means[0], color='red', linestyle='--', linewidth=1, alpha=0.6, label='Full baseline')
    ax.legend(fontsize=8)
    out = os.path.join(OUT_DIR, "ablation_evclp.png")
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  → Ablation: {out}")


def plot_station_map(problem: EVCLP, best_solution):
    X = np.clip(np.round(best_solution), 0, 1).astype(int)
    sel = np.where(X == 1)[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(problem.demand_points[:,0], problem.demand_points[:,1],
               c='#1f77b4', s=20, alpha=0.5, label='Điểm nhu cầu', zorder=2)
    if len(sel) > 0:
        ax.scatter(problem.locations[sel,0], problem.locations[sel,1],
                   c='red', s=100, marker='s', label='Trạm sạc', zorder=3)
        for idx in sel:
            circle = plt.Circle(problem.locations[idx], problem.coverage_radius,
                                fill=False, edgecolor='green', alpha=0.25, linewidth=0.8)
            ax.add_patch(circle)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(f'Vị trí trạm sạc tốt nhất (MPA-OOA) — {len(sel)} trạm')
    ax.legend(); ax.set_aspect('equal')
    out = os.path.join(OUT_DIR, "station_map_evclp.png")
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  → Station map: {out}")


# ─────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    print("="*70)
    print("EVCLP — Vành đai 1 Hà Nội | seed cố định + Wilcoxon + Ablation")
    print("="*70)

    # Khởi tạo bài toán (seed data cố định = 42)
    print("\nKhởi tạo bài toán EVCLP ...")
    problem = EVCLP(n_locations=200, n_demand_points=100, data_seed=42)
    print(f"  Locations: {problem.n_locations}, Demand: {problem.n_demand_points}")
    print(f"  Area: {problem.area_size/1000:.1f} km, Budget: {problem.budget/1000:.0f} tỷ")

    # 10.1 Thực nghiệm chính
    all_results, df_stats, df_wilcoxon = run_main_experiments(problem)

    # 10.2 Ablation
    df_abl = run_ablation(problem)

    # 10.3 Hình
    print("\nVẽ hình ...")
    plot_convergence(all_results)
    plot_fitness_bar(df_stats)
    plot_time_bar(df_stats)
    plot_ablation(df_abl)

    # Tìm nghiệm tốt nhất MPA-OOA để vẽ bản đồ
    best_idx = int(np.argmin(all_results['MPA-OOA']['fitness']))
    np.random.seed(SEEDS[best_idx])
    alg_best = MPA_OOA(problem, pop_size=POP_SIZE_MPA, max_iter=MAX_ITER)
    alg_best.run()
    plot_station_map(problem, alg_best.best_solution)

    # 10.4 In bảng tóm tắt
    print("\n" + "="*70)
    print("TÓM TẮT KẾT QUẢ")
    print("="*70)
    print(df_stats.to_string(index=False))
    print("\nWilcoxon:")
    print(df_wilcoxon.to_string(index=False))
    print("\nAblation:")
    print(df_abl.to_string(index=False))

    print(f"\n{'='*70}")
    print(f"HOÀN TẤT — {(time.time()-t0)/60:.1f} phút")
    print(f"Tất cả kết quả trong: {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
