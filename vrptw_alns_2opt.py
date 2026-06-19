"""
VRPTW - ALNS + 2-opt Solver
Adaptive Large Neighbourhood Search with intra-route 2-opt local search for VRPTW.

Pipeline:
1. Parse Solomon benchmark (C101)
2. Nearest Neighbour initial solution
3. ALNS main loop (Random/Worst Removal + Greedy Insertion + 2-opt + SA acceptance)
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

    routes: list of routes, each route is a list of customer indices (1-based, no depot).
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


# ── 7. Intra-route 2-opt ───────────────────────────────────────────────────────
def two_opt_route(route, solution):
    """
    Apply 2-opt to a single route until no improving swap is found.

    2-opt swap: reverse the segment between index i+1 and j (inclusive).
    Only accept the swap if:
      (a) total distance decreases
      (b) the resulting route is still feasible (time windows + capacity)

    Returns the improved route.
    """
    dist = solution.dist_matrix
    best = route[:]

    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                # Nodes around the reversed segment
                a = best[i - 1] if i > 0 else 0
                b = best[i]
                c = best[j]
                d = best[j + 1] if j < len(best) - 1 else 0

                # Distance gain from reversing segment [i..j]
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


# ── 8. ALNS Main Loop (Destroy → Repair → 2-opt → SA) ─────────────────────────
def alns(
    initial_solution,
    n_iterations=1000,
    n_remove=5,
    sa_initial_temp=100.0,
    sa_cooling=0.995,
    random_seed=42,
):
    """
    ALNS for VRPTW with intra-route 2-opt after every repair step.

    Each iteration:
        1. 50% Random Removal / 50% Worst Removal
        2. Greedy Insertion repair
        3. 2-opt on every route of the repaired solution
        4. SA acceptance criterion
    """
    random.seed(random_seed)

    current = initial_solution.copy()
    best    = initial_solution.copy()
    temp    = sa_initial_temp
    history = [(0, best.total_distance())]

    for iteration in range(1, n_iterations + 1):

        # Destroy
        if random.random() < 0.5:
            destroyed, removed = random_removal(current, n_remove)
        else:
            destroyed, removed = worst_removal(current, n_remove)

        # Repair
        repaired = greedy_insertion(destroyed, removed)

        # 2-opt (intra-route improvement after every repair)
        candidate = two_opt_solution(repaired)

        # SA acceptance
        delta = candidate.total_distance() - current.total_distance()
        if delta < 0:
            current = candidate
        elif random.random() < math.exp(-delta / temp):
            current = candidate

        # Update global best
        if current.total_distance() < best.total_distance():
            best = current.copy()

        temp *= sa_cooling

        if iteration % 100 == 0:
            history.append((iteration, best.total_distance()))
            print(f"  Iter {iteration:>5} | best={best.total_distance():.2f} "
                  f"| current={current.total_distance():.2f} | temp={temp:.4f}")

    print(f"\n[ALNS+2opt] Vehicles: {best.num_vehicles()},  Distance: {best.total_distance():.2f}")
    return best, history


# ── 9. Run ─────────────────────────────────────────────────────────────────────
def main():
    num_vehicles, capacity, depot, customers = parse_solomon('c101.txt')
    nodes = [depot] + customers
    dist_matrix = compute_distance_matrix(depot, customers)

    print(f"Instance: C101  |  Customers: {len(customers)}  |  Capacity: {capacity}")

    print("\n[1] Nearest Neighbour")
    nn_sol = nearest_neighbor(nodes, dist_matrix, capacity)

    print("\n[2] ALNS + 2-opt")
    alns_sol, history = alns(
        initial_solution=nn_sol,
        n_iterations=1000,
        n_remove=5,
        sa_initial_temp=100.0,
        sa_cooling=0.995,
    )

    plot_routes(nodes, nn_sol.routes,   title='Nearest Neighbour — c101')
    plot_routes(nodes, alns_sol.routes, title='ALNS+2opt — c101')
    plot_comparison(nn_sol, alns_sol, history)


# ── 10. Visualisation ──────────────────────────────────────────────────────────
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


def plot_comparison(nn_sol, alns_sol, history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    methods = ['Nearest\nNeighbour', 'ALNS+2opt']
    colors  = ['#e07b54', '#5b8db8']

    vehicles = [nn_sol.num_vehicles(), alns_sol.num_vehicles()]
    axes[0].bar(methods, vehicles, color=colors)
    axes[0].set_title('Vehicles Used')
    for i, v in enumerate(vehicles):
        axes[0].text(i, v + 0.1, str(v), ha='center', fontweight='bold')

    distances = [nn_sol.total_distance(), alns_sol.total_distance()]
    axes[1].bar(methods, distances, color=colors)
    axes[1].set_title('Total Distance')
    for i, v in enumerate(distances):
        axes[1].text(i, v + 10, f'{v:.0f}', ha='center', fontweight='bold')

    iters, dists = zip(*history)
    axes[2].plot(iters, dists, color='#5b8db8', linewidth=2)
    axes[2].set_title('ALNS Convergence')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Best Distance')

    plt.suptitle('VRPTW c101: Nearest Neighbour vs ALNS+2opt(no adaptive)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparison-noadaptive-with2opt-c101.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved comparison.png")


if __name__ == '__main__':
    main()
