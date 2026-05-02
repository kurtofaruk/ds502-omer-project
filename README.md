# DS502 Project

## Ömer Faruk Kurt 

# 📊 Project Development Pipeline

This repository contains the end-to-end workflow for the project, spanning from initial data acquisition to final stakeholder delivery.

## 🧑🏻‍💻 Main Script: Project can be run by src/main_2/main.py ⩤
---

## 🚀 Project Milestones

### 1. Data Ingestion & Preparation
* **Data Collection:** Gathering and centralizing all necessary raw data sources from TSBLIB.
* ***Github Repo:** https://github.com/mastqe/tsplib

* **Preprocessing:** Cleaning and structuring data to use in the Gurobi model.
* **Collected Data:** 31 instances were collected.
* **Clustering:** Applied K-means clustering with given C values for each instance.
* **Data Prep. Script:** These steps prepared in src/main/data_prep.py


### 2. Research & Modeling
* **Model Development:** Created MILP model and leveraged custom subtour elimination constraints to reduce runtime.
* **Extension:** Applied a smart subtour elimination constraint with LazyConstraint function of Gurobi.
* **Model Parameters:** Runtime was limited as 60 seconds for each instance and 8 threads were utilized.
* **Model Script:** These steps modeled in src/main/model.py

### 3. Reports
* **Result Reporting:** Analyzing performance metrics and documenting outcomes. Objective, optimality gap (nominal and percentage values), and runtime metrics gathered for each instance.
* **Report Script:** Report generation coded in src/main/report.py

### 3. Visualization 
* **Custom Plotting:** Developed rich plot functions for result interpretation.
* **Vis. Script:** Plot functions created in src/main/report.py

### 4. Final Delivery
* **Progressive QA:** Continuous code quality checks and refactoring throughout the lifecycle.
* **Materials:** Preparation of the final comprehensive report and presentation deck.

---

## 🛠 Progress Tracker

| Phase | Task | Script | Status |
| :--- | :--- | :--- | :--- |
| **01** | Data Collection | data_prep.py | ✅ Done |
| **02** | Model Creation | model.py | ✅ Done |
| **03** | Result Reporting | report.py | ✅ Done |
| **04** | Plotting Functions | report.py | ✅ Done |
| **05** | MDP Reformulation | mdp_notes.md, mdp.py | ✅ Done |
| **06** | Final Report & Presentation | pdf and pptx | 🟨 In Progress |

>Code quality is monitored progressively.

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

The subtour elimination constraints from the MILP are replaced entirely by the visited-set V in the state — a cluster cannot be revisited because actions are restricted to `M \ V`.

**Key Files:**
- `src/main_2/mdp_notes.md` — full MDP formulation, Bellman equation, mapping table, illustrative example, and experiment plan
- `src/main_2/mdp.py` — exact DP implementation (feasible for small instances, C ≤ ~15) and greedy nearest-cluster heuristic baseline

**New Assumptions Introduced:**
- Tour traversal direction is fixed (reversing gives same cost due to symmetric distances)
- Costs are deterministic Euclidean distances (no stochastic extension in base model)
- Exact DP is used only for small instances; large instances still use the Gurobi MILP

**Planned Experiments (Week 9/10):**
- Compare MILP objective vs greedy heuristic across all 31 instances
- Measure how MIP gap scales with N and C under 60s / 120s / 300s time limits
- Report: objective value, runtime, MIP gap %, absolute gap, greedy gap %
