"""
VRPTW - ALNS Solver (Adaptive Weights, NO 2-opt)
Adaptive Large Neighbourhood Search with adaptive operator weight selection
for the Vehicle Routing Problem with Time Windows. 2-opt is intentionally
left unused in this ablation variant.

Pipeline:
1. Parse Solomon benchmark (C101)
2. Nearest Neighbour initial solution
3. ALNS main loop (adaptive Random/Worst Removal + Greedy Insertion + SA acceptance)
4. Visualisation & comparison
"""

# ── 1. Imports ───────────────────────────────────────────────────────────────
import math
import copy
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ── 2. Data Parser ────────────────────────────────────────────────────────────
def parse_solomon(filepath):
    """Parse a Solomon VRPTW benchmark file (e.g. C101.txt)."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if 'NUMBER' in line:
            parts = lines[i + 1].split()
            num_vehicles = int(parts[0])
            capacity = int(parts[1])
            break

    customer_start = None
    for i, line in enumerate(lines):
        if 'CUST' in line and 'NO' in line:
            customer_start = i + 1
            break

    depot = None
    customers = []
    for line in lines[customer_start:]:
        parts = line.split()
        if len(parts) == 0:
            continue
        node = {
            'id':           int(parts[0]),
            'x':            float(parts[1]),
            'y':            float(parts[2]),
            'demand':       float(parts[3]),
            'ready_time':   float(parts[4]),
            'due_date':     float(parts[5]),
            'service_time': float(parts[6]),
        }
        if node['id'] == 0:
            depot = node
        else:
            customers.append(node)

    return num_vehicles, capacity, depot, customers


def compute_distance_matrix(depot, customers):
    """Euclidean distance matrix for depot + all customers."""
    nodes = [depot] + customers
    n = len(nodes)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = nodes[i]['x'] - nodes[j]['x']
                dy = nodes[i]['y'] - nodes[j]['y']
                dist[i][j] = math.sqrt(dx * dx + dy * dy)
    return dist


# ── 3. Solution Class ─────────────────────────────────────────────────────────
class Solution:
    """
    Represents a complete VRPTW solution.

    routes: list of routes, each a list of customer indices (1-based, no depot).
            e.g. [[1, 3, 5], [2, 4, 6]]
    Depot is always index 0.
    """
    def __init__(self, routes, nodes, dist_matrix, capacity):
        self.routes = routes
        self.nodes = nodes
        self.dist_matrix = dist_matrix
        self.capacity = capacity

    # ── Cost ──
    def total_distance(self):
        """Total travel distance across all routes."""
        total = 0.0
        for route in self.routes:
            if not route:
                continue
            prev = 0
            for cust in route:
                total += self.dist_matrix[prev][cust]
                prev = cust
            total += self.dist_matrix[prev][0]
        return total

    # ── Feasibility ──
    def is_route_feasible(self, route):
        """Check capacity and time window feasibility for a single route."""
        if not route:
            return True
        load = 0.0
        time = 0.0
        prev = 0
        for cust in route:
            node = self.nodes[cust]
            load += node['demand']
            if load > self.capacity:
                return False
            time += self.dist_matrix[prev][cust]
            time = max(time, node['ready_time'])
            if time > node['due_date']:
                return False
            time += node['service_time']
            prev = cust
        time += self.dist_matrix[prev][0]
        if time > self.nodes[0]['due_date']:
            return False
        return True

    def is_feasible(self):
        return all(self.is_route_feasible(r) for r in self.routes)

    # ── Utility ──
    def copy(self):
        """Deep copy of routes only (nodes/dist_matrix are shared)."""
        return Solution(
            routes=copy.deepcopy(self.routes),
            nodes=self.nodes,
            dist_matrix=self.dist_matrix,
            capacity=self.capacity,
        )

    def num_vehicles(self):
        return sum(1 for r in self.routes if r)

    def __repr__(self):
        return (f"Solution(vehicles={self.num_vehicles()}, "
                f"distance={self.total_distance():.2f}, "
                f"feasible={self.is_feasible()})")


# ── 4. Nearest Neighbour Initial Solution ─────────────────────────────────────
def nearest_neighbor(nodes, dist_matrix, capacity):
    """
    Greedy Nearest Neighbour heuristic for VRPTW.
    Dispatches new vehicles whenever no feasible customer is reachable.
    """
    n = len(nodes) - 1
    unvisited = set(range(1, n + 1))
    routes = []

    while unvisited:
        route = []
        current = 0
        current_time = 0.0
        current_load = 0.0

        while True:
            best = None
            best_dist = float('inf')

            for j in unvisited:
                node = nodes[j]
                d = dist_matrix[current][j]
                arrival = max(current_time + d, node['ready_time'])
                if arrival <= node['due_date'] and current_load + node['demand'] <= capacity:
                    if d < best_dist:
                        best_dist = d
                        best = j

            if best is None:
                break

            node = nodes[best]
            current_time = max(current_time + dist_matrix[current][best], node['ready_time'])
            current_time += node['service_time']
            current_load += node['demand']
            route.append(best)
            unvisited.remove(best)
            current = best

        routes.append(route)

    sol = Solution(routes, nodes, dist_matrix, capacity)
    print(f"[NN] Vehicles: {sol.num_vehicles()},  Distance: {sol.total_distance():.2f}")
    return sol


# ── 5. Destroy Operators ──────────────────────────────────────────────────────
def random_removal(solution, n_remove):
    """Randomly remove n_remove customers from the solution."""
    sol = solution.copy()
    all_customers = [c for route in sol.routes for c in route]
    n_remove = min(n_remove, len(all_customers))
    removed = random.sample(all_customers, n_remove)
    removed_set = set(removed)
    sol.routes = [[c for c in route if c not in removed_set] for route in sol.routes]
    sol.routes = [r for r in sol.routes if r]
    return sol, removed


def worst_removal(solution, n_remove):
    """
    Remove the n_remove customers with the greatest individual distance saving.
    saving(c) = dist(prev, c) + dist(c, next) - dist(prev, next)
    Greedy: remove one at a time, recompute after each removal.
    """
    sol = solution.copy()
    removed = []
    dist = sol.dist_matrix

    for _ in range(n_remove):
        best_saving = -float('inf')
        best_cust = None
        best_route_idx = None
        best_pos = None

        for r_idx, route in enumerate(sol.routes):
            for pos, cust in enumerate(route):
                prev = route[pos - 1] if pos > 0 else 0
                nxt  = route[pos + 1] if pos < len(route) - 1 else 0
                saving = dist[prev][cust] + dist[cust][nxt] - dist[prev][nxt]
                if saving > best_saving:
                    best_saving = saving
                    best_cust = cust
                    best_route_idx = r_idx
                    best_pos = pos

        if best_cust is None:
            break

        sol.routes[best_route_idx].pop(best_pos)
        if not sol.routes[best_route_idx]:
            sol.routes.pop(best_route_idx)
        removed.append(best_cust)

    return sol, removed


# ── 6. Repair Operator — Greedy Insertion ─────────────────────────────────────
def greedy_insertion(solution, removed):
    """
    Re-insert removed customers using cheapest feasible insertion.
    Sorted by due_date ascending (tighter deadlines first).
    Falls back to opening a new route if no feasible position exists.
    """
    sol = solution.copy()
    dist = sol.dist_matrix
    nodes = sol.nodes

    to_insert = sorted(removed, key=lambda c: nodes[c]['due_date'])

    for cust in to_insert:
        best_cost = float('inf')
        best_route_idx = None
        best_pos = None

        for r_idx, route in enumerate(sol.routes):
            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt  = route[pos]     if pos < len(route) else 0
                delta = dist[prev][cust] + dist[cust][nxt] - dist[prev][nxt]
                candidate = route[:pos] + [cust] + route[pos:]
                if sol.is_route_feasible(candidate) and delta < best_cost:
                    best_cost = delta
                    best_route_idx = r_idx
                    best_pos = pos

        if best_route_idx is not None:
            sol.routes[best_route_idx].insert(best_pos, cust)
        else:
            sol.routes.append([cust])

    return sol


# ── 7. Intra-route 2-opt (defined but NOT used in this variant) ──────────────
# Kept here only so the ablation comparison shares the same codebase.
def two_opt_route(route, solution):
    """
    Apply 2-opt to a single route until no improving swap is found.
    Reverses segment [i..j] if it reduces distance and stays feasible.
    """
    dist = solution.dist_matrix
    best = route[:]

    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                a = best[i - 1] if i > 0 else 0
                b = best[i]
                c = best[j]
                d = best[j + 1] if j < len(best) - 1 else 0

                before = dist[a][b] + dist[c][d]
                after  = dist[a][c] + dist[b][d]

                if after < before - 1e-6:
                    candidate = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if solution.is_route_feasible(candidate):
                        best = candidate
                        improved = True

    return best


def two_opt_solution(solution):
    """Apply 2-opt to every route in the solution. Returns a new Solution."""
    sol = solution.copy()
    sol.routes = [
        two_opt_route(route, sol) if len(route) >= 3 else route
        for route in sol.routes
    ]
    return sol


# ── 8. Adaptive Weight Selection ──────────────────────────────────────────────
# Each destroy operator maintains a weight w. Every iteration:
#   1. Select operator by weighted random choice (roulette wheel)
#   2. Score the outcome: new global best / better than current / SA-accepted
#      worse solution / rejected
#   3. Update weight with exponential smoothing: w <- (1-lambda)*w + lambda*sigma
SCORE_GLOBAL_BEST = 10   # candidate is new global best
SCORE_BETTER      = 6    # candidate improves on current
SCORE_ACCEPTED    = 3    # SA accepted a worse candidate
SCORE_REJECTED    = 0    # candidate rejected


def weighted_choice(operators, weights):
    """
    Roulette-wheel selection.
    operators: list of callables
    weights:   list of floats (same length)
    Returns the selected operator and its index.
    """
    total = sum(weights)
    r = random.random() * total

    cumulative = 0.0
    for i, (op, w) in enumerate(zip(operators, weights)):
        cumulative += w
        if r <= cumulative:
            return op, i
    return operators[-1], len(operators) - 1


def update_weight(weight, score, lam):
    """Exponential smoothing weight update."""
    return (1 - lam) * weight + lam * score


# ── 9. ALNS Main Loop (Adaptive Weights + SA, NO 2-opt) ──────────────────────
def alns(
    initial_solution,
    n_iterations=1000,
    n_remove=5,
    sa_initial_temp=100.0,
    sa_cooling=0.995,
    weight_lambda=0.1,
    random_seed=42,
):
    """
    Adaptive LNS for VRPTW — NO 2-opt variant (ablation).

    Each iteration:
        1. Select destroy operator by adaptive weighted roulette wheel
        2. Greedy Insertion repair
        3. SA acceptance               (2-opt step removed for this ablation)
        4. Update operator weight based on outcome

    Args:
        weight_lambda: learning rate for exponential smoothing (0 < lambda < 1)
    """
    random.seed(random_seed)

    destroy_ops     = [random_removal, worst_removal]
    destroy_names   = ['RandomRemoval', 'WorstRemoval']
    weights         = [1.0, 1.0]   # initialise equally

    current = initial_solution.copy()
    best    = initial_solution.copy()
    temp    = sa_initial_temp

    history         = [(0, best.total_distance())]
    weight_history  = {name: [(0, 1.0)] for name in destroy_names}

    for iteration in range(1, n_iterations + 1):

        # ── Select destroy operator ────────────────────────────────────────────
        op, op_idx = weighted_choice(destroy_ops, weights)

        # ── Destroy ───────────────────────────────────────────────────────────
        destroyed, removed = op(current, n_remove)

        # ── Repair ────────────────────────────────────────────────────────────
        repaired = greedy_insertion(destroyed, removed)

        # ── (2-opt intentionally skipped in this ablation variant) ─────────────
        candidate = repaired

        # ── SA acceptance + scoring ───────────────────────────────────────────
        delta = candidate.total_distance() - current.total_distance()
        score = SCORE_REJECTED

        if candidate.total_distance() < best.total_distance():
            # New global best
            current = candidate
            best    = candidate.copy()
            score   = SCORE_GLOBAL_BEST

        elif delta < 0:
            # Better than current but not global best
            current = candidate
            score   = SCORE_BETTER

        elif random.random() < math.exp(-delta / temp):
            # SA accepted a worse solution
            current = candidate
            score   = SCORE_ACCEPTED

        # ── Update weight ─────────────────────────────────────────────────────
        weights[op_idx] = update_weight(weights[op_idx], score, weight_lambda)

        temp *= sa_cooling

        # ── Logging ───────────────────────────────────────────────────────────
        if iteration % 100 == 0:
            history.append((iteration, best.total_distance()))
            for i, name in enumerate(destroy_names):
                weight_history[name].append((iteration, weights[i]))
            print(f"  Iter {iteration:>5} | best={best.total_distance():.2f} "
                  f"| current={current.total_distance():.2f} "
                  f"| temp={temp:.4f} "
                  f"| w={[f'{w:.3f}' for w in weights]}")

    print(f"\n[ALNS] Vehicles: {best.num_vehicles()},  Distance: {best.total_distance():.2f}")
    return best, history, weight_history, destroy_names


# ── 10. Run ────────────────────────────────────────────────────────────────────
def main():
    num_vehicles, capacity, depot, customers = parse_solomon('c101.txt')
    nodes = [depot] + customers
    dist_matrix = compute_distance_matrix(depot, customers)

    print(f"Instance: C101  |  Customers: {len(customers)}  |  Capacity: {capacity}")

    print("\n[1] Nearest Neighbour")
    nn_sol = nearest_neighbor(nodes, dist_matrix, capacity)

    print("\n[2] ALNS (Adaptive Weights, NO 2-opt)")
    alns_sol, history, weight_history, destroy_names = alns(
        initial_solution=nn_sol,
        n_iterations=1000,
        n_remove=5,
        sa_initial_temp=100.0,
        sa_cooling=0.995,
        weight_lambda=0.1,
    )

    plot_routes(nodes, nn_sol.routes,   title='Nearest Neighbour — c101')
    plot_routes(nodes, alns_sol.routes, title='ALNS (Adaptive, no 2-opt) — c101')
    plot_comparison(nn_sol, alns_sol, history, weight_history, destroy_names)


# ── 11. Visualisation ──────────────────────────────────────────────────────────
def plot_routes(nodes, routes, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cm.tab20(np.linspace(0, 1, max(len(routes), 1)))
    for route, color in zip(routes, colors):
        path = [0] + route + [0]
        xs = [nodes[n]['x'] for n in path]
        ys = [nodes[n]['y'] for n in path]
        ax.plot(xs, ys, color=color, linewidth=1.5)
    for node in nodes[1:]:
        ax.scatter(node['x'], node['y'], color='steelblue', s=30, zorder=3)
    depot = nodes[0]
    ax.scatter(depot['x'], depot['y'], color='red', s=120, marker='*', zorder=4, label='Depot')
    ax.set_title(title, fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_comparison(nn_sol, alns_sol, history, weight_history, destroy_names):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = ['#e07b54', '#5b8db8']

    # Vehicles
    methods  = ['Nearest\nNeighbour', 'ALNS']
    vehicles = [nn_sol.num_vehicles(), alns_sol.num_vehicles()]
    axes[0].bar(methods, vehicles, color=colors)
    axes[0].set_title('Vehicles Used')
    for i, v in enumerate(vehicles):
        axes[0].text(i, v + 0.1, str(v), ha='center', fontweight='bold')

    # Distance
    distances = [nn_sol.total_distance(), alns_sol.total_distance()]
    axes[1].bar(methods, distances, color=colors)
    axes[1].set_title('Total Distance')
    for i, v in enumerate(distances):
        axes[1].text(i, v + 10, f'{v:.0f}', ha='center', fontweight='bold')

    # ALNS convergence
    iters, dists = zip(*history)
    axes[2].plot(iters, dists, color='#5b8db8', linewidth=2)
    axes[2].set_title('ALNS Convergence')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Best Distance')

    # Operator weight evolution
    op_colors = ['#e07b54', '#5b8db8']
    for name, color in zip(destroy_names, op_colors):
        iters_w, ws = zip(*weight_history[name])
        axes[3].plot(iters_w, ws, label=name, color=color, linewidth=2)
    axes[3].set_title('Operator Weight Evolution')
    axes[3].set_xlabel('Iteration')
    axes[3].set_ylabel('Weight')
    axes[3].legend()

    plt.suptitle('VRPTW c101: Nearest Neighbour vs ALNS (Adaptive, no 2-opt)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparison_adaptive_no2opt-c101.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved comparison_adaptive_no2opt.png")


if __name__ == '__main__':
    main()
