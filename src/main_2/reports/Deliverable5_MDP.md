# DS 502 – Semester Project  
## Deliverable D5: MDP Reformulation & Experiment Plan  
**Ömer Faruk Kurt**  
Week 9 & 10

---

## 1. Original Model Summary

### Problem
The **Generalized Traveling Salesman Problem (G-TSP)**, also known as the Clustered TSP (CTSP).  
Given N nodes pre-assigned to C clusters, find a minimum-cost Hamiltonian cycle that visits **exactly one node per cluster**.

### Instances
31 TSPLIB benchmark instances (att48 to pr439). N ranges from 48 to 439 nodes; C ranges from 10 to 88 clusters (C = ⌈N/5⌉). Clusters are formed via K-means (k=C, random_state=502).

### Objective
Minimize the total Euclidean travel distance of the tour:

$$\min \sum_{(i,j,k,l) \in X} c_{ijkl} \cdot x_{ijkl}$$

where $c_{ijkl} = \sqrt{(x_i^j - x_k^l)^2 + (y_i^j - y_k^l)^2}$.

### Decision Variables
- $y_{ij} \in \{0,1\}$ — 1 if node $j$ in cluster $i$ is the selected representative  
- $x_{ijkl} \in \{0,1\}$ — 1 if there is a directed arc from node $j$ (cluster $i$) to node $l$ (cluster $k$), $i \neq k$

### Constraints
| Label | Description |
|-------|-------------|
| C1 | $\sum_j y_{ij} = 1$ for all $i$ — exactly one node selected per cluster |
| C2 | $\sum_{k \neq i, l} x_{ijkl} = y_{ij}$ — selected node sends exactly one arc out |
| C3 | $\sum_{i \neq k, j} x_{ijkl} = y_{kl}$ — selected node receives exactly one arc in |
| SEC | Subtour elimination — added lazily via Gurobi callback |

---

## 2. Sequential Decision Interpretation

The G-TSP is naturally sequential: the traveler visits one cluster at a time, choosing which cluster to visit next and which node within it to use.

| Stage | What Happens |
|-------|-------------|
| **Stage 0** | Choose a starting cluster $i_0$ and its node $j_0$ — the origin of the tour. |
| **Stages 1 to C−1** | From the current node, choose the next **unvisited** cluster $k$ and one node $l$ within it. |
| **Stage C (terminal)** | All C clusters have been visited. Return to the starting node $(i_0, j_0)$ to close the tour. |

At each stage the traveler knows exactly where they are and which clusters have been visited. There is no uncertainty — every cost is a deterministic Euclidean distance. This makes the problem a **fully observable, deterministic, finite-horizon MDP**.

---

## 3. MDP Definition

### 3.1 State Space $\mathcal{S}$

$$s_t = \bigl( (i,\, j),\; V \bigr)$$

- $(i, j)$: current position — cluster $i$, node $j$ within cluster $i$  
- $V \subseteq M = \{1,\ldots,C\}$: frozenset of already-visited cluster indices, $|V| = t$

**Initial state (stage 0):** $s_0 = \bigl((i_0, j_0),\; \{i_0\}\bigr)$ for a chosen starting node.  
**Terminal state:** $V = M$ (all C clusters visited).  
**State space size:** $O(C \cdot n_{\max} \cdot 2^C)$ — exponential in C.

### 3.2 Action Space $\mathcal{A}(s)$

From state $s = \bigl((i,j), V\bigr)$, feasible actions are:

$$\mathcal{A}(s) = \bigl\{(k,\, l) \;:\; k \in M \setminus V,\; l \in \{1,\ldots,n_k\} \bigr\}$$

An action $(k, l)$ means "travel to node $l$ in the next cluster $k$, where $k$ has not yet been visited."

### 3.3 Transition Function

The transition is **deterministic** ($P(s' \mid s, a) = 1$ for the unique $s'$, 0 otherwise):

$$\mathcal{T}\bigl(s,\, a\bigr) = \bigl((k,\, l),\; V \cup \{k\}\bigr)$$

After action $a = (k, l)$, the new position is $(k, l)$ and the visited set grows by cluster $k$.

### 3.4 Reward / Cost Function

The immediate cost of action $a = (k, l)$ from state $s = \bigl((i,j), V\bigr)$:

$$c(s,\, a) = d\bigl((i,j),\,(k,l)\bigr) = \sqrt{(x_i^j - x_k^l)^2 + (y_i^j - y_k^l)^2}$$

At the terminal stage (all clusters visited), a closing cost is incurred:

$$c_{\text{terminal}}(s_C) = d\bigl((i_C, j_C),\,(i_0, j_0)\bigr)$$

The total tour cost equals the sum of all stage costs:

$$\text{Total cost} = \sum_{t=0}^{C} c(s_t, a_t)$$

This matches the original MILP objective exactly.

### 3.5 Policy

A **policy** $\pi$ is a mapping from states to actions:

$$\pi : \mathcal{S} \rightarrow \mathcal{A}(s)$$

In the G-TSP context, a policy is a decision rule that, given the current location and visited-cluster set, selects the next cluster-node pair to visit. Examples:

- **Greedy policy:** always travel to the nearest unvisited cluster-node (nearest-neighbor heuristic).
- **Optimal policy $\pi^*$:** minimizes total tour cost; found by backward DP or MILP.

### 3.6 Horizon and Terminal Condition

- **Horizon:** Finite, $T = C$ stages (one per cluster).
- **Terminal condition:** When $V = M$, close the tour by returning to the start node. No further actions.

### 3.7 MDP Classification

| Property | Value |
|----------|-------|
| Deterministic or stochastic | **Deterministic** |
| Fully or partially observable | **Fully observable** |
| Discounted or undiscounted | **Undiscounted** (no discount factor $\gamma$) |
| Horizon | **Finite**, $T = C$ stages |

---

## 4. Bellman Equation

Let $V_t\bigl((i,j), V\bigr)$ denote the **minimum remaining tour cost** — visiting all clusters in $M \setminus V$ and returning to start — when at node $(i,j)$ at stage $t$ having visited set $V$.

**Recursive equation (stages $t = 0, 1, \ldots, C-1$):**

$$V_t\!\bigl((i,j),\, V\bigr) = \min_{\substack{(k,l):\; k \notin V \\ l \in \{1,\ldots,n_k\}}} \Bigl[\; d\!\bigl((i,j),\,(k,l)\bigr) \;+\; V_{t+1}\!\bigl((k,l),\; V \cup \{k\}\bigr) \;\Bigr]$$

**Terminal condition (stage $C$, $V = M$):**

$$V_C\!\bigl((i,j),\, M\bigr) = d\!\bigl((i,j),\,(i_0, j_0)\bigr)$$

**Meaning of each term:**

| Term | Meaning |
|------|---------|
| $V_t((i,j), V)$ | Min remaining cost from $(i,j)$ with visited set $V$ |
| $d((i,j),(k,l))$ | Immediate arc cost — maps to $c_{ijkl} \cdot x_{ijkl}$ in the MILP |
| $V_{t+1}((k,l), V \cup \{k\})$ | Optimal future cost after committing to node $(k,l)$ |
| $\min$ over $(k,l): k \notin V$ | Decision: which unvisited cluster and node to visit next |
| $V_C((i,j), M)$ | Terminal cost: closing the tour back to start |

The optimal tour starting at $(i_0, j_0)$ has cost $V_0\!\bigl((i_0,j_0), \{i_0\}\bigr)$. The global optimum is found by trying all possible starting nodes and taking the minimum.

Since the MDP is deterministic ($P(s' \mid s, a) = 1$), the probabilistic sum $\sum_{s'} P(s' \mid s, a) V_{t+1}(s')$ collapses to a single term, yielding the cleaner recursion above.

---

## 5. Connection to the Original MILP

| MILP Element | MDP Element |
|---|---|
| Decision variable $x_{ijkl}$ | Action $a = (k,l)$ taken from state $s = ((i,j), V)$ |
| Decision variable $y_{ij}$ | Implicit: $y_{ij}=1$ iff $(i,j)$ appears as an action in the trajectory |
| Constraint C1 (one node per cluster) | Embedded in action feasibility: each stage selects exactly one $(k,l)$ per new cluster |
| Constraint C2 (out-flow = 1) | Embedded in sequential structure: each state produces exactly one action |
| Constraint C3 (in-flow = 1) | Embedded in sequential structure: each state reached via exactly one prior action |
| Subtour elimination constraints (lazy) | **Replaced entirely** by visited-set $V$: a cluster $k$ cannot be revisited because $k \notin V$ is required |
| Objective $\sum c_{ijkl} \cdot x_{ijkl}$ | Total cost = $\sum_{t=0}^{C} c(s_t, a_t)$ — sum of stage costs along trajectory |

**Key insight:** The subtour elimination constraints — the computationally hardest part of the MILP — are completely replaced by the visited-set component $V$ of the MDP state. No explicit SEC is needed because the action space already forbids revisiting any cluster.

---

## 6. Illustrative Example

Consider a small instance with $C = 3$ clusters, 2 nodes each. Start at node $(1, 1)$.

**Coordinates (illustrative):**  
Cluster 1: node 1 = (0, 0), node 2 = (1, 0)  
Cluster 2: node 1 = (3, 1), node 2 = (4, 2)  
Cluster 3: node 1 = (1, 4), node 2 = (2, 3)  

**Initial state:** $s_0 = \bigl((1,1),\; \{1\}\bigr)$

**Feasible actions from $s_0$:** $(2,1),\ (2,2),\ (3,1),\ (3,2)$

| Action | Cost $d((1,1), \cdot)$ |
|--------|------------------------|
| $(2,1)$ | $d((0,0),(3,1)) = 3.16$ |
| $(2,2)$ | $d((0,0),(4,2)) = 4.47$ |
| $(3,1)$ | $d((0,0),(1,4)) = 4.12$ |
| $(3,2)$ | $d((0,0),(2,3)) = 3.61$ |

**Suppose action $(2,1)$ is chosen (cost 3.16).**  
New state: $s_1 = \bigl((2,1),\; \{1,2\}\bigr)$

**Feasible actions from $s_1$:** $(3,1),\ (3,2)$

| Action | Cost $d((3,1), \cdot)$ | + Return cost to $(1,1)$ | Total remaining |
|--------|------------------------|--------------------------|-----------------|
| $(3,1)$ | $d((3,1),(1,4)) = 3.61$ | $d((1,4),(0,0)) = 4.12$ | $7.73$ |
| $(3,2)$ | $d((3,1),(2,3)) = 2.24$ | $d((2,3),(0,0)) = 3.61$ | $5.85$ |

**Optimal continuation:** action $(3,2)$. Total tour = $3.16 + 2.24 + 3.61 = 9.01$.

The Bellman recursion evaluates all such trajectories and selects the globally minimum-cost tour. Starting with action $(3,2)$ instead of $(2,1)$ might yield a shorter tour — the DP explores all options.

---

## 7. Discussion

### Benefits of the MDP View

1. **Subtour elimination is automatic.** The visited-set $V$ in the state makes subtours infeasible by construction — no explicit lazy constraints are needed.
2. **Supports approximate methods.** The Bellman equation motivates reinforcement learning approaches (Q-learning, policy gradients) that can scale to large instances where exact MILP is too slow.
3. **Interpretable sequential structure.** The policy perspective makes it natural to design and compare heuristics (greedy nearest-unvisited-cluster) against the optimal solution.
4. **Uncertainty extension.** If travel costs become stochastic (e.g., dynamic traffic), the MDP framework extends naturally; the static MILP would require scenario-based stochastic programming.

### Limitations

1. **Exponential state space.** The subset $V$ has $2^C$ possible values. For $C = 88$ (pr439 instance), exact DP is completely intractable.
2. **Artificial sequentiality.** The G-TSP is a static problem; the sequential structure is a reinterpretation, not a physical time process. The optimal tour does not inherently have a preferred direction.
3. **Starting node dependency.** The terminal cost depends on the chosen start $(i_0, j_0)$, requiring the DP to be run from all possible starts — $O(C)$ initializations.

### Assumptions Introduced

- Tour traversal is directed (one fixed direction); reversing gives the same cost due to symmetric distances.
- The starting node $(i_0, j_0)$ is fixed before running the recursion; the optimal solution tries all starts.
- Costs are deterministic Euclidean distances — no stochasticity in the base model.
- Exact DP is only run when $C \leq 11$ (DP threshold); larger instances use the Gurobi MILP.

---

## 8. Experiment Plan

### Goal
Compare MILP, exact DP, greedy MDP heuristic, and Genetic Algorithm across all 31 TSPLIB instances. Quantify how solution quality and runtime vary with problem size.

### What Will Be Tested

| Experiment | Description |
|------------|-------------|
| **MILP scalability** | How does runtime and MIP gap grow as $N$ and $C$ increase? |
| **Time limit sensitivity** | How does solution quality change at 60s vs 120s vs 300s limits? |
| **Heuristic vs optimal** | Compare greedy MDP policy against MILP best-found solution. |
| **DP vs MILP (small instances)** | Verify exact DP matches MILP optimum for $C \leq 15$. |
| **GA vs baselines** | Compare GA tour quality and runtime against MILP, greedy, and exact DP. |

### Varying Parameters

| Parameter | Range |
|-----------|-------|
| Instance size $N$ | 48 to 439 |
| Number of clusters $C$ | 10 to 88 |
| MILP time limit | 60s, 120s, 300s |
| GA time budget | 120s total per instance |

### Performance Measures

| Measure | Definition |
|---------|------------|
| **Objective value** | Total tour distance (minimize) |
| **Runtime** | Wall-clock seconds to solve |
| **MIP gap (%)** | $\frac{|\text{ObjVal} - \text{ObjBound}|}{\text{ObjVal}} \times 100$ |
| **Absolute gap** | $|\text{ObjVal} - \text{ObjBound}|$ |
| **Greedy gap (%)** | $\frac{\text{greedy} - \text{MILP}}{\text{MILP}} \times 100$ |
| **GA gap (%)** | $\frac{\text{GA} - \text{MILP}}{\text{MILP}} \times 100$ |

### Baselines

- **MILP optimal (or best-found):** Gurobi with lazy subtour elimination, 120s limit, 8 threads
- **Exact DP:** Held-Karp style memoized recursion, only for $C \leq 15$
- **Greedy heuristic:** Nearest-unvisited-cluster policy, tries one start per cluster
- **Genetic Algorithm:** Memetic GA with Gurobi-based fitness evaluation (see Section 9)

### Expected Findings

- MILP solves to optimality for small instances ($C \leq 20$) within 60s
- MIP gap increases significantly for $C > 40$ under 120s limit
- Greedy heuristic produces tours within ~15–30% of MILP optimal, orders of magnitude faster
- Exact DP confirms MILP optimality on small instances
- GA expected to improve on greedy by ~5–15% through structured node-selection search

---

## 9. Genetic Algorithm Approach

### 9.1 Motivation

The MDP view exposes a natural two-level decomposition of the CTSP:

1. **Node selection:** which representative node to visit in each cluster
2. **Cluster ordering:** the sequence in which clusters are visited

The greedy heuristic and exact DP address both levels simultaneously in a single forward pass. The Genetic Algorithm addresses level 1 explicitly (evolving node selections) while delegating level 2 to an exact TSP solver (Gurobi), yielding a principled search over the exponentially large node-selection space.

### 9.2 Chromosome Representation

Each individual in the population is an **`indexed_route` list** of length $C$:

$$\text{chromosome} = [\,n_1,\; n_2,\; \ldots,\; n_C\,]$$

where $n_i$ is the **within-cluster indexed node** (1-based) chosen for cluster $i$. Position $i-1$ (0-based) holds the indexed node for cluster $i$. Node indices are local to each cluster — cluster $i$ contains nodes $\{1, \ldots, |\text{cluster}_i|\}$.

There is no explicit cluster-sequence gene. Given a chromosome, the optimal cluster visit order is computed exactly by Gurobi (Section 9.3).

### 9.3 Fitness Evaluation

$$\text{fitness}(\text{chromosome}) = \min_{\text{cluster ordering}} \text{TSP}([\,\text{coord}(1, n_1),\; \ldots,\; \text{coord}(C, n_C)\,])$$

Fitness is evaluated in **parallel batches** via `call_mp_gurobi_route_level` (implemented in `gurobi_solver_route_parallel.py`):

1. Extract the $C$ selected node coordinates from the chromosome.
2. Batch all unevaluated individuals and solve each as a $C$-node TSP with Gurobi (subtour-elimination callback, `time_limit_eval` seconds per call).
3. Return the **optimal tour distance** (fitness) and the **optimal cluster visit sequence**.

A **chromosome-keyed cache** (`_fitness_cache`) stores results for every evaluated `indexed_route` tuple, avoiding redundant Gurobi calls for identical chromosomes that reappear across generations.

### 9.4 Genetic Operators

| Operator | Description |
|----------|-------------|
| **Initialisation** | Purely random — each cluster independently draws a uniformly random valid indexed node |
| **Selection** | Roulette wheel (minimisation-adapted): selection probability $\propto \max(f) - f_i + \varepsilon$; two distinct parents selected per pair |
| **Crossover** | Randomly chosen per offspring: two-point (30%), single-point (30%), or uniform (40%); followed by `validate_route` to repair any out-of-range indexed nodes |
| **Mutation** | Point mutation (60%): reassign one cluster's node to a different valid node; or swap mutation (40%): exchange two positions in the chromosome; followed by `validate_route` |
| **Survivor selection** | Truncation — parents and offspring are merged; top `pop_size` by fitness survive to the next generation (implicit elitism) |
| **Elite history** | All-time top `elite_size = 5` individuals tracked for reporting; does not re-inject into population |

### 9.5 Algorithm Parameters

| Parameter | Default Value |
|-----------|---------------|
| Population size | 100 |
| Max generations | 100 |
| Crossover pairs per generation (`ga_counter`) | `max(pop_size, 30)` |
| Crossover-then-mutate offspring per gen (`cx_mut_count`) | `ga_counter // 2` |
| Mutation-only offspring per gen (`mut_count`) | `ga_counter // 3` |
| Gurobi time limit per fitness eval | 20 s |
| Total GA wall-clock budget | 120 s |
| Stall limit (early stop) | 20 generations without improvement |
| Elite history size | 5 |
| Random seed | 42 |

Early stopping triggers on whichever comes first: time budget exceeded, stall limit reached, or full population convergence (all chromosomes identical).

### 9.6 Connection to the MDP

The GA fitness evaluation is equivalent to solving:

$$\min_{\text{cluster ordering}} V_0\!\bigl((i_0,j_0),\{i_0\}\bigr) \quad \text{given fixed node\_sel}$$

where the Bellman recursion collapses to a standard TSP on $C$ fixed points. The GA outer loop then searches over node selections to minimise this already-optimised inner cost — a Benders-style decomposition embedded in an evolutionary framework.

### 9.7 Output

Results are saved to `reports/results_ga.xlsx`. Comparison plots (GA vs MILP vs Greedy) are generated by `report.plot_ga_report()` and saved to the `figures/` directory:

| Figure | Content |
|--------|---------|
| `ga_vs_milp_objective.png` | Grouped bar chart: MILP vs GA vs Greedy tour distance |
| `ga_gap_comparison.png` | GA gap % vs Greedy gap % above MILP optimal |
| `ga_runtime_scatter.png` | MILP runtime vs GA runtime scatter |
| `ga_obj_vs_C.png` | Tour distance vs $C$ — one series per method |

---

*Report prepared for DS 502 – Semester Project, Deliverable D5*  
*Implementation files: `src/main_2/mdp.py`, `src/main_2/mdp_notes.md`, `src/main_2/genetic_algorithm.py`*
