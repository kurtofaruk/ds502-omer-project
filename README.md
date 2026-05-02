# DS502 Project

## Ömer Faruk Kurt 

# 📊 Project Development Pipeline

This repository contains the end-to-end workflow for the project, spanning from initial data acquisition to final stakeholder delivery.

## 🧑🏻‍💻 Main Script: Project can be run by `src/main_2/main.py`

```bash
pip install -r requirements.txt   # install dependencies
cd src/main_2
python main.py                     # MILP solver + reports
python mdp.py                      # greedy & exact DP heuristics
python genetic_algorithm.py        # genetic algorithm
```

---

## 🚀 Project Milestones

### 1. Data Ingestion & Preparation
* **Data Collection:** 31 TSPLIB benchmark instances (att48 – pr439) sourced from [mastqe/tsplib](https://github.com/mastqe/tsplib).
* **Preprocessing:** Coordinate extraction and K-means clustering (random_state=502) with predefined C values per instance.
* **Script:** `src/main_2/data_prep.py`

### 2. MILP Model
* **Formulation:** Binary MILP for the Generalized TSP — select one representative node per cluster and find the minimum-cost Hamiltonian cycle.
* **Subtour elimination:** Lazy constraint callback (`subtourelim`) added via Gurobi's `LazyConstraints` parameter.
* **Parameters:** 120 s time limit, 8 threads per instance.
* **Script:** `src/main_2/model.py`

### 3. MDP Reformulation & Heuristics
* **MDP:** CTSP recast as a finite-horizon, deterministic, fully observable Markov Decision Process (state = position + visited-cluster set).
* **Exact DP:** Held-Karp-style memoised recursion — optimal, feasible for C ≤ 15.
* **Greedy heuristic:** Nearest-unvisited-cluster policy, O(C² · n_max²), runs on all instances.
* **Scripts:** `src/main_2/mdp.py`, `src/main_2/mdp_notes.md`

### 4. Genetic Algorithm
* **Design:** Memetic GA — chromosome encodes one node selected per cluster; Gurobi solves the resulting C-node TSP at every fitness evaluation to find the optimal cluster visit order.
* **Operators:** Uniform crossover on node selections, per-cluster random mutation, node optimisation local search using the Gurobi-returned sequence.
* **Script:** `src/main_2/genetic_algorithm.py`

### 5. Reports & Visualisation
* **Metrics:** Objective value, runtime, MIP gap %, absolute gap, greedy/GA gap % vs MILP.
* **Plots:** Per-method bar charts, gap % analysis, runtime scatter, objective vs C.
* **Script:** `src/main_2/report.py`

### 6. Final Delivery
* **Materials:** Comprehensive report (`Deliverable5_MDP.md`) and presentation deck.

---

## 🛠 Progress Tracker

| Phase | Task | Script | Status |
| :--- | :--- | :--- | :--- |
| **01** | Data Collection & Preparation | `data_prep.py` | ✅ Done |
| **02** | MILP Model | `model.py` | ✅ Done |
| **03** | MDP Reformulation | `mdp_notes.md`, `mdp.py` | ✅ Done |
| **04** | Genetic Algorithm | `genetic_algorithm.py` | ✅ Done |
| **05** | Result Reporting & Plots | `report.py` | ✅ Done |
| **06** | Final Report & Presentation | pdf and pptx | 🟨 In Progress |

> Code quality is monitored progressively. All third-party dependencies are pinned in `requirements.txt`.

---

## 🧠 MDP Reformulation (Deliverable D5)

The CTSP is reinterpreted as a finite-horizon, deterministic, fully observable Markov Decision Process.

**State:** `s_t = ( (i,j), V )` — current cluster-node position and set of visited clusters  
**Action:** `a = (k, l)` — next unvisited cluster k and node l within it  
**Transition:** Deterministic — `s_{t+1} = ( (k,l), V ∪ {k} )`  
**Cost:** Euclidean distance between consecutive selected nodes  
**Horizon:** Finite, T = C stages (one per cluster), then return to start  

**Bellman Equation:**
```
V_t( (i,j), V ) = min_{(k,l): k ∉ V} { d((i,j),(k,l)) + V_{t+1}((k,l), V ∪ {k}) }
V_C( (i,j), M ) = d( (i,j), start_node )
```

The subtour elimination constraints are replaced entirely by the visited-set V — a cluster cannot be revisited because actions are restricted to `M \ V`.

**Key files:** `src/main_2/mdp.py` · `src/main_2/mdp_notes.md`

---

## 🧬 Genetic Algorithm

A memetic GA that separates the two sub-problems of the CTSP:

| Sub-problem | Solved by |
|---|---|
| Which node to visit in each cluster | GA (evolves chromosome = `node_sel` dict) |
| Optimal cluster visit order | Gurobi TSP solver (`tour_cost` in `gurobi_solver_route_parallel.py`) |

**Chromosome:** `{ cluster_id: node_id }` — one entry per cluster.  
**Fitness:** Gurobi solves the C-node TSP on the selected nodes, returning the optimal tour distance and the optimal cluster sequence.  
**Local search:** After each Gurobi evaluation, node selections are refined using the returned cluster sequence (each cluster picks the node minimising the sum of its two incident edge distances).

**Key file:** `src/main_2/genetic_algorithm.py`
