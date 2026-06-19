# VRPTW Solver: Nearest Neighbour + Adaptive Large Neighbourhood Search (ALNS)

A from-scratch implementation of a metaheuristic solver for the **Vehicle Routing
Problem with Time Windows (VRPTW)**, benchmarked on the Solomon 100-customer
instance set. The solver is built in stages — Nearest Neighbour construction,
ALNS with destroy/repair operators, intra-route 2-opt local search, and
adaptive operator weighting — with each stage isolated as a separate
experiment to measure its individual contribution.

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Algorithm Design](#algorithm-design)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Ablation Analysis](#ablation-analysis)
- [Comparison to Best Known Solutions](#comparison-to-best-known-solutions)
- [Discussion & Limitations](#discussion--limitations)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)

---

## Overview

VRPTW asks: given a depot, a fleet of capacity-limited vehicles, and a set of
customers each with a demand, a service time, and a delivery time window —
find a set of feasible routes that serves every customer while minimizing
(primarily) the number of vehicles used and (secondarily) total distance
travelled.

This project implements four versions of an ALNS-based solver, each adding
one more component on top of the last, so that the effect of each design
choice can be measured independently rather than just reported as a single
black-box result.

| Version | Adaptive operator selection | Intra-route 2-opt |
|---|---|---|
| **v1** | ❌ (uniform random choice) | ❌ |
| **v2** | ❌ (uniform random choice) | ✅ |
| **v3** | ✅ (roulette-wheel weighted) | ❌ |
| **v4** | ✅ (roulette-wheel weighted) | ✅ |

All four versions share the same initial solution (Nearest Neighbour) and the
same destroy/repair operator pool, so any difference in outcome is
attributable to the adaptive-weighting and 2-opt components specifically.

---

## Pipeline

```
Solomon instance (.txt)
        │
        ▼
 Nearest Neighbour construction  ──────────────►  initial feasible solution
        │
        ▼
 ALNS main loop  (1000 iterations)
   1. Select a destroy operator
        - v1 / v2 : uniform random (50/50)
        - v3 / v4 : adaptive roulette-wheel (weighted)
   2. Destroy   → remove n_remove customers
   3. Repair    → Greedy Insertion (cheapest feasible position)
   4. (v2 / v4 only) Intra-route 2-opt local search on the repaired solution
   5. Simulated Annealing acceptance test
   6. Update operator weight based on outcome (v3 / v4 only)
        │
        ▼
 Best solution found
```

---

## Algorithm Design

### Initial solution — Nearest Neighbour
Greedily builds one route at a time, always moving to the nearest feasible
unvisited customer (respecting time window and capacity), opening a new
vehicle when no feasible next customer exists.

### Destroy operators
| Operator | Logic |
|---|---|
| **Random Removal** | Removes `n_remove` customers chosen uniformly at random. |
| **Worst Removal** | Greedily removes the customer whose removal yields the largest distance saving, repeated `n_remove` times. |

### Repair operator
| Operator | Logic |
|---|---|
| **Greedy Insertion** | Re-inserts removed customers (sorted by tightest due-date first) into the cheapest feasible position across all routes; opens a new route only if no feasible position exists. |

### Local search — Intra-route 2-opt (v2 / v4 only)
For each route independently, repeatedly reverses any segment that reduces
total distance, **discarding the move if it violates time-window or capacity
feasibility**. Runs to a 2-opt local optimum before returning to the ALNS loop.

### Acceptance criterion — Simulated Annealing
```
accept if new solution is better than current
accept with probability exp(-Δ / T) otherwise
T *= cooling_rate   (every iteration)
```

### Adaptive operator weighting (v3 / v4 only)
Each destroy operator carries a weight, updated by exponential smoothing
after every iteration:

w ← (1 - λ) · w + λ · σ

| Outcome | Score σ |
|---|---|
| New global best | 10 |
| Improves current solution | 6 |
| Accepted by SA (worse solution) | 3 |
| Rejected | 0 |

Operator selection uses roulette-wheel sampling proportional to weight.

### Parameters used in all experiments
| Parameter | Value |
|---|---|
| Iterations | 1000 |
| `n_remove` | 5 |
| SA initial temperature | 100.0 |
| SA cooling rate | 0.995 |
| Weight learning rate (λ) | 0.1 |
| Random seed | 42 |

---

## Experimental Setup

All four versions were run on six Solomon benchmark instances spanning the
three structural categories:

| Category | Instances | Characteristics |
|---|---|---|
| **C** (clustered) | C101, C201 | Customers geographically clustered; wide time windows |
| **R** (random) | R101, R201 | Customers randomly scattered; tight time windows |
| **RC** (mixed) | RC101, RC201 | Mix of clustered and random; tight time windows |

`_1xx` instances have short scheduling horizons (more, smaller routes);
`_2xx` instances have long horizons (fewer, longer routes) — this is why
best-known solutions for `_201` instances typically use far fewer vehicles
than `_101` instances of the same category.

---

## Results

### Nearest Neighbour baseline (before any optimization)

| Instance | Vehicles | Distance |
|---|---|---|
| C101 | 21 | 1871.0 |
| C201 | 15 | 1880.0 |
| R101 | 37 | 2623.0 |
| R201 | 15 | 1985.0 |
| RC101 | 27 | 2711.0 |
| RC201 | 15 | 2468.0 |

### All four ALNS versions

| Instance | v1: base ALNS | v2: + 2-opt | v3: + adaptive | v4: + adaptive + 2-opt |
|---|---|---|---|---|
| C101  | 10 / 829  | 10 / 829  | 10 / 829  | 10 / 829  |
| C201  | 5 / 676   | 5 / 676   | 6 / 734   | 6 / 734   |
| R101  | 20 / 1728 | 20 / 1728 | 22 / 1753 | 22 / 1753 |
| R201  | 12 / 1323 | 12 / 1269 | 11 / 1276 | 12 / 1270 |
| RC101 | 19 / 1810 | 18 / 1807 | 19 / 1868 | 19 / 1903 |
| RC201 | 10 / 1372 | 11 / 1476 | 11 / 1479 | 10 / 1484 |

*(format: vehicles / total distance)*

### Improvement over Nearest Neighbour (best ALNS version per instance)

| Instance | NN Distance | Best ALNS Distance | Improvement |
|---|---|---|---|
| C101  | 1871 | 829  | **−55.7%** |
| C201  | 1880 | 676  | **−64.0%** |
| R101  | 2623 | 1728 | **−34.1%** |
| R201  | 1985 | 1269 | **−36.1%** |
| RC101 | 2711 | 1807 | **−33.4%** |
| RC201 | 2468 | 1372 | **−44.4%** |

Across all six instances, every ALNS variant cuts total distance by
**33% to 64%** relative to the Nearest Neighbour construction, and reduces
vehicle count by roughly half. This confirms the core ALNS loop is working
correctly and is the dominant source of improvement — far larger than the
incremental effect of 2-opt or adaptive weighting on top of it.

---

## Ablation Analysis

The four-version design isolates two questions: *does 2-opt help?* and
*does adaptive weighting help?*

### Effect of 2-opt (v1 → v2, v3 → v4)

| Instance | v1 dist | v2 dist | Δ (2-opt alone) | v3 dist | v4 dist | Δ (2-opt + adaptive) |
|---|---|---|---|---|---|---|
| C101  | 829  | 829  | 0      | 829  | 829  | 0      |
| C201  | 676  | 676  | 0      | 734  | 734  | 0      |
| R101  | 1728 | 1728 | 0      | 1753 | 1753 | 0      |
| R201  | 1323 | 1269 | **−4.1%** | 1276 | 1270 | −0.5%  |
| RC101 | 1810 | 1807 | −0.2%  | 1868 | 1903 | **+1.9%** |
| RC201 | 1372 | 1476 | **+7.6%** | 1479 | 1484 | +0.3%  |

2-opt's effect is small and **inconsistent in sign** — it helps on R201,
is essentially neutral on the C instances and R101, and actively hurts on
RC201. This matches the theoretical expectation discussed during design:
in tightly time-windowed instances, most 2-opt reversals fail the
feasibility check, leaving very little room for the operator to act. Where
it does find feasible moves (R201), the gain is real but modest (~4%).

### Effect of adaptive weighting (v1 → v3, v2 → v4)

| Instance | v1 dist | v3 dist | Δ (adaptive alone) |
|---|---|---|---|
| C101  | 829  | 829  | 0 |
| C201  | 676  | 734  | **+8.6%** |
| R101  | 1728 | 1753 | +1.4% |
| R201  | 1323 | 1276 | **−3.6%** |
| RC101 | 1810 | 1868 | +3.2% |
| RC201 | 1372 | 1479 | **+7.8%** |

This is the most interesting (and counter-intuitive) finding of the project:
**adaptive weighting did not consistently improve solution quality** in this
setup, and on three of six instances (C201, RC101, RC201) it produced a
*worse* result than uniform-random operator selection.

A plausible explanation, visible directly in the operator-weight plots: with
only two destroy operators and λ = 0.1, the weights converge to roughly
equal values (~3.0 each) within the first 200–300 iterations and stay there.
With only two candidates, "adaptive" selection collapses quickly to
something close to uniform selection anyway — so any deviation in outcome
is closer to random seed noise from the slightly different search trajectory
than a genuine learned preference. The mechanism would likely show clearer
benefit with **three or more destroy operators** (e.g. adding Shaw Removal),
where there is real heterogeneity in operator usefulness for the weighting
to discover.

### Convergence behaviour

All six instances converge well before 1000 iterations — most flatten out
between iteration 200 and 700 (see convergence plots). This suggests 1000
iterations is already more than sufficient for this problem size, and the
experiment could be run with fewer iterations (e.g. 500) without meaningful
quality loss, trading runtime for the same result.

---

## Comparison to Best Known Solutions

Best-known values from the Solomon benchmark reference table (SINTEF):

| Instance | Best Known (veh / dist) | Best of v1–v4 (veh / dist) | Vehicle Gap | Distance Gap |
|---|---|---|---|---|
| C101  | 10 / 828.94  | 10 / 829  | +0 | **+0.01%** |
| C201  | 3 / 591.56   | 5 / 676   | +2 | +14.3% |
| R101  | 19 / 1650.80 | 20 / 1728 | +1 | +4.7% |
| R201  | 4 / 1252.37  | 11 / 1269 | +7 | +1.3% |
| RC101 | 14 / 1696.95 | 18 / 1807 | +4 | +6.5% |
| RC201 | 4 / 1406.94  | 10 / 1372 | +6 | **−2.5%** |

Two clear patterns stand out:

1. **C101 is essentially solved exactly** (gap ≈ 0.01%) — a clustered,
   wide-time-window instance is the easiest case for this solver design.

2. **Vehicle count gap is much larger than distance gap, and it is largest
   on `_201` instances.** Best-known solutions for these use very few
   vehicles (3–4) running long single routes, whereas this solver
   consistently lands on 10+ vehicles. RC201 is a striking case: the
   solver's *total distance is actually 2.5% better* than best-known, but
   it uses 6 more vehicles to achieve it — i.e. many short cheap routes
   instead of few long ones. Since vehicle count is the primary VRPTW
   objective (ahead of distance), this means the solver is optimizing the
   *secondary* objective well while under-optimizing the *primary* one.

---

## Discussion & Limitations

- **`n_remove = 5` is likely too small to trigger route consolidation.**
  Removing 5 customers per iteration is enough to locally rearrange a route
  but rarely enough to empty out an entire route and let Greedy Insertion
  fold it into another — which is exactly the move needed to close the
  vehicle-count gap on `_201` instances.
- **Only two destroy operators limits what adaptive weighting can learn.**
  Adding a third operator with genuinely different behaviour (e.g. Shaw
  Removal, which removes *spatially/temporally similar* customers rather
  than random or worst-cost ones) would give the weighting mechanism more
  signal to act on, and is the most natural next experiment.
- **2-opt's value is bounded by time-window tightness**, as anticipated:
  it helps on the loosest instances (R201) and is neutral-to-harmful on
  tighter ones, because most candidate reversals fail feasibility checks
  before they can be evaluated for improvement.
- **A dedicated "reduce vehicle count" mechanism is the most promising next
  step** — e.g. an explicit "try to empty the shortest route" operator, run
  periodically regardless of distance cost, since the current operators are
  all implicitly distance-driven and have no direct incentive to consolidate
  routes.

---

## Repository Structure

```
.
├── vrptw_alns_base.py                  # v1: ALNS, no adaptive, no 2-opt
├── vrptw_alns_2opt.py                  # v2: ALNS + 2-opt, no adaptive
├── vrptw_alns_adaptive_no2opt.py       # v3: ALNS + adaptive, no 2-opt
├── vrptw_alns_adaptive.py              # v4: ALNS + adaptive + 2-opt (full)
├── data/
│   ├── C101.txt  C201.txt
│   ├── R101.txt  R201.txt
│   └── RC101.txt RC201.txt
├── results/
│   └── comparison_*.png                   # per-instance comparison plots
└── README.md
```

## How to Run

Each script is self-contained. To run a given version against a Solomon
instance, update the filepath in the `parse_solomon(...)` call and run the script directly:

```python
num_vehicles, capacity, depot, customers = parse_solomon('data/c101.txt')
```

Each script prints Nearest Neighbour and final ALNS results to stdout and
saves a comparison figure (`Vehicles Used` / `Total Distance` /
`ALNS Convergence` [/ `Operator Weight Evolution` for v3 and v4]) to the
working directory.

---

*Best-known solution values sourced from the SINTEF Solomon VRPTW benchmark
reference table.*
