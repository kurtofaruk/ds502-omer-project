# Deliverable D5: MDP Reformulation of the Clustered TSP

## 1. Original Model Summary

**Problem:** Clustered Traveling Salesman Problem (CTSP) — also known as the Generalized TSP (GTSP).

**Instances:** 31 TSPLIB benchmark instances (att48 to pr439), each with N nodes pre-assigned to C clusters via K-means clustering.

**Objective:** Minimize total Euclidean travel distance of a Hamiltonian cycle that visits **exactly one node from each cluster**.

**Decision Variables:**
- `y[i,j] ∈ {0,1}` — 1 if node j in cluster i is selected as the cluster representative
- `x[i,j,k,l] ∈ {0,1}` — 1 if there is a directed arc from node j in cluster i to node l in cluster k (i ≠ k)

**Constraints:**
- **C1:** Exactly one node selected per cluster: `Σ_j y[i,j] = 1` for all i
- **C2:** Out-flow: each selected node sends exactly one arc out: `Σ_{k≠i,l} x[i,j,k,l] = y[i,j]`
- **C3:** In-flow: each selected node receives exactly one arc in: `Σ_{i≠k,j} x[i,j,k,l] = y[k,l]`
- **Subtour Elimination (lazy):** Added dynamically via Gurobi callback — prevents partial cycles that don't span all C clusters

---

## 2. Sequential Decision Interpretation

The CTSP is naturally a sequential routing problem. The decision stages unfold as follows:

- **Stage 0:** Choose a starting cluster and node (i₀, j₀) to begin the tour.
- **Stage 1 to C−1:** From the current node, choose the next unvisited cluster k and which node l within that cluster to visit.
- **Stage C:** After all C clusters are visited, return to the starting node (i₀, j₀) to close the tour.

At each stage, the decision-maker knows exactly where they are and which clusters have been visited — this is a fully observable problem. The decision is purely sequential: no batch decisions are made.

---

## 3. MDP Definition

### State Space S

The state at stage t is:

```
s_t = ( (i, j),  V )
```

where:
- `(i, j)` is the current position: cluster i, node j within cluster i
- `V ⊆ M` is the set of cluster indices already visited, with `|V| = t`
- `M = {1, 2, ..., C}` is the full set of clusters

The initial state (stage 0, before any visit) is: `s_0 = ( (i₀, j₀), {i₀} )` for some chosen starting node.

The terminal state is reached at stage C when `V = M` (all clusters visited).

**State space size:** O(C · n_max · 2^C) — exponential in C due to the subset V. For the 31 instances (C ranges from 10 to 88), exact DP is feasible only for small instances; this motivates approximate methods.

### Action Space A(s)

From state `s = ( (i,j), V )`, the feasible actions are:

```
A(s) = { (k, l) : k ∈ M \ V,  l ∈ {1, ..., n_k} }
```

An action `a = (k, l)` means: travel to node l in the next cluster k, where k has not yet been visited.

Constraints C1, C2, and C3 from the original MILP are now implicitly enforced through the definition of A(s): only one node can be chosen per cluster (the action selects it), and the tour structure (in/out degree = 1) is guaranteed by the sequential traversal.

### Transition Function

The transition is **deterministic**:

```
T( s, a ) = ( (k, l),  V ∪ {k} )
```

After taking action `a = (k, l)` from state `s = ( (i,j), V )`:
- The new current position becomes `(k, l)`
- The visited set expands to `V ∪ {k}`

In probabilistic notation: `P( s' | s, a ) = 1` for the unique resulting s', and 0 for all other s'.

### Reward / Cost Function

The immediate cost of taking action `a = (k, l)` from state `s = ( (i,j), V )` is the Euclidean distance:

```
c( s, a ) = d( (i,j), (k,l) ) = sqrt( (x_i_j - x_k_l)² + (y_i_j - y_k_l)² )
```

This matches the original objective directly: the total tour cost equals the sum of arc distances, which is the sum of stage-wise costs.

At the terminal stage (all clusters visited), an additional return cost is incurred:

```
c_terminal( s_C ) = d( (i_C, j_C), (i₀, j₀) )
```

### Policy

A **policy** π is a mapping from states to actions:

```
π : S → A(s)
```

In the CTSP context, a policy is a decision rule that, given the current location and set of visited clusters, selects the next cluster-node pair to visit. An optimal policy π* minimizes the total expected tour cost.

Example of a greedy policy: always travel to the nearest unvisited cluster (nearest-neighbor heuristic). An optimal policy found by DP or MILP always outperforms this.

### Horizon and Terminal Condition

- **Horizon:** Finite, T = C stages (one per cluster).
- **Terminal condition:** When all C clusters have been visited (`V = M`), close the tour by returning to the start node. No further decisions are made after stage C.

### MDP Classification

| Property | Value |
|---|---|
| Deterministic or stochastic | **Deterministic** — costs and transitions are fully determined by (s, a) |
| Fully or partially observable | **Fully observable** — the state (position + visited set) is known exactly |
| Discounted or undiscounted | **Undiscounted** — minimize total tour distance, no discount factor γ |
| Horizon | **Finite**, T = C stages |

---

## 4. Bellman Equation

Let `V_t( (i,j), V )` denote the **minimum total cost to complete the remaining tour** — visiting all clusters in `M \ V` and returning to start — when at node `(i,j)` at stage t, having already visited set V.

**Recursive equation (stages t = 0, 1, ..., C−1):**

```
V_t( (i,j), V ) =  min        { d( (i,j), (k,l) )  +  V_{t+1}( (k,l), V ∪ {k} ) }
                (k,l): k ∉ V
                  l ∈ {1,...,n_k}
```

**Terminal condition (stage C, all clusters visited):**

```
V_C( (i,j), M ) = d( (i,j), (i₀, j₀) )
```

where `(i₀, j₀)` is the starting node of the tour.

**Meaning of each term:**

| Term | Meaning |
|---|---|
| `V_t( (i,j), V )` | Minimum remaining tour cost from position (i,j) with visited set V at stage t |
| `d( (i,j), (k,l) )` | Immediate travel cost from current node to next chosen node — maps to arc cost `c[i,j,k,l] · x[i,j,k,l]` in the MILP |
| `V_{t+1}( (k,l), V∪{k} )` | Optimal future cost after committing to node (k,l) next |
| `min over (k,l): k∉V` | Decision: which unvisited cluster and which node within it to visit next |
| `V_C( (i,j), M )` | Terminal cost: closing the tour by returning to the start node |

The value of the optimal tour starting at node `(i₀, j₀)` is `d( (i₀,j₀), (i₀,j₀) ) + V_0( (i₀,j₀), {i₀} )` — the minimum tour cost is found by trying all possible starting nodes.

---

## 5. Mapping: Original MILP ↔ MDP

| MILP Element | MDP Element |
|---|---|
| Decision variable `x[i,j,k,l]` | Action `a = (k,l)` taken from state `s = ((i,j), V)` |
| Decision variable `y[i,j]` | Implicitly encoded: y[i,j]=1 iff (i,j) appears as an action in the trajectory |
| Constraint C1 (one node per cluster) | Embedded in action feasibility: each stage selects exactly one (k,l) per new cluster |
| Constraint C2 (out-flow = 1) | Embedded in sequential structure: each state has exactly one action taken |
| Constraint C3 (in-flow = 1) | Embedded in sequential structure: each state is reached via exactly one prior action |
| Subtour elimination constraints (lazy) | Embedded in state structure: V tracks visited clusters, preventing revisits |
| Objective: `Σ c[i,j,k,l] · x[i,j,k,l]` | Total cost = `Σ_{t=0}^{C} c(s_t, a_t)` — sum of stage costs along trajectory |

**Key insight:** The subtour elimination constraints — the hardest part of the MILP — are completely replaced by the visited-set component V of the state. The state itself guarantees no subtours, since an action is only feasible if the target cluster has not been visited.

---

## 6. Illustrative Example

Consider a small instance with C = 3 clusters, 2 nodes each. Start at node (1, 1) (cluster 1, node 1).

**Initial state:** s₀ = ( (1,1), {1} )

**Feasible actions from s₀:**
- (2,1): travel to node 1 of cluster 2, cost d₁
- (2,2): travel to node 2 of cluster 2, cost d₂
- (3,1): travel to node 1 of cluster 3, cost d₃
- (3,2): travel to node 2 of cluster 3, cost d₄

**Suppose action (2,1) is taken (d₁ = 10.5):**

New state: s₁ = ( (2,1), {1,2} )

**Feasible actions from s₁:**
- (3,1): travel to node 1 of cluster 3, cost d₅ = 8.2
- (3,2): travel to node 2 of cluster 3, cost d₆ = 12.7

**Suppose action (3,1) is taken (cost 8.2):**

New state: s₂ = ( (3,1), {1,2,3} ) — all clusters visited

**Terminal cost:** return to start (1,1): d₇ = 6.4

**Total tour cost for this trajectory:** 10.5 + 8.2 + 6.4 = 25.1

The Bellman equation evaluates all such trajectories and selects the minimum. An alternative starting action (2,2) might yield a shorter total tour.

---

## 7. Discussion

### Benefits of the MDP View

1. **Subtour elimination is automatic:** The visited-set V in the state makes subtours infeasible by construction, without any explicit constraint.
2. **Supports approximate methods:** The Bellman equation motivates reinforcement learning approaches (Q-learning, policy gradients, actor-critic) that can scale to large instances where exact MILP is too slow.
3. **Interpretable sequential structure:** The policy perspective makes it natural to design heuristics (e.g., greedy nearest-unvisited-cluster policies) and compare them to optimal.
4. **Uncertainty extension:** If travel costs become stochastic (e.g., due to traffic or dynamic edge weights), the MDP framework extends naturally by adding probability distributions to transition costs, whereas the static MILP would require scenario-based stochastic programming.

### Limitations

1. **Exponential state space:** The subset V has 2^C possible values. For C = 88 clusters (e.g., pr439 instance), exact DP is completely intractable — the MILP with lazy constraints is the practical solver.
2. **Artificial sequentiality:** The original CTSP is a static problem; the sequential structure is a useful reinterpretation, but the optimal tour does not inherently have a preferred starting direction.
3. **Starting node dependency:** The terminal condition `V_C = return-to-start` depends on the chosen starting node (i₀, j₀), requiring the DP to be initialized from all possible start states and taking the minimum.

### Assumptions Introduced

- Tour traversal is directed (one fixed traversal direction); reversing the tour gives the same cost due to symmetric distances.
- The starting node `(i₀, j₀)` is fixed before running the recursion; the full optimal solution requires trying all possible starting nodes.
- Costs are deterministic Euclidean distances — no stochasticity is modeled in the base MDP.

---

## 8. Experiment Plan

### Goal
Evaluate the CTSP MILP solver across instance sizes and compare to MDP-inspired heuristics.

### What Will Be Tested
1. **Scalability:** How does solver runtime and MIP gap grow as N (nodes) and C (clusters) increase?
2. **Time limit sensitivity:** How does solution quality (gap) change at 60s vs 120s vs 300s time limits?
3. **Heuristic vs optimal:** Compare the greedy nearest-unvisited-cluster policy (MDP-inspired) against the MILP optimal solution.

### Varying Parameters
| Parameter | Values |
|---|---|
| Instance size N | 48 to 262 (small-medium range where MILP solves well) |
| Number of clusters C | 10 to 53 |
| Time limit | 60s, 120s, 300s |

### Performance Measures
- **Objective value:** Total tour distance (minimize)
- **Runtime:** Wall-clock seconds to solve
- **MIP gap (%):** Relative optimality gap at termination
- **Absolute gap:** `|ObjVal - ObjBound|`
- **Greedy gap:** `(greedy_cost - milp_cost) / milp_cost × 100%`

### Baseline
- **MILP optimal (or best-found):** Gurobi with lazy subtour elimination
- **Greedy heuristic:** Nearest-unvisited-cluster policy (select nearest node in nearest unvisited cluster)
- **Random policy:** Random cluster ordering, random node within cluster

### Expected Findings
- MILP solves to optimality for small instances (C ≤ 20) within 60s
- Gap increases significantly for C > 40 under 120s limit
- Greedy heuristic produces tours within ~15–30% of optimal, much faster than MILP
