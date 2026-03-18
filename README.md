# DS502 Project

## Ömer Faruk Kurt 

# 📊 Project Development Pipeline

This repository contains the end-to-end workflow for the project, spanning from initial data acquisition to final stakeholder delivery.

* **Main Script:** These steps prepared in src/main/main.py
---

## 🚀 Project Milestones

### 1. Data Ingestion & Preparation
* **Data Collection:** Gathering and centralizing all necessary raw data sources from TSBLIB.
* ***Github Repo:** https://github.com/mastqe/tsplib

* **Preprocessing:** Cleaning and structuring data for optimal model performance.
* **Collected Data:** 31 instances were collected.
* **Clustering:** Applied K-means clustering with given C values for each instance.
* ***Data Prep. Script:** These steps prepared in src/main/data_prep.py


### 2. Research & Modeling
* **Model Development:** Engineering and training the MILP architecture.
* **Extension:** Applied a smart subtour elimination constraint with LazyConstraint function of Gurobi.
* ***Model Script:** These steps modeled in src/main/model.py

### 3. Reports
* **Result Reporting:** Analyzing performance metrics and documenting outcomes. Objective, optimality gap(nominal and percentage values), runtime metrics gathered for each instance.
* ***Report Script:** These steps modeled in src/main/report.py

### 3. Visualization 
* **Custom Plotting:** Developed rich plot functions for result interpretation.
* ***Vis. Script:** These steps modeled in src/main/report.py

### 4. Final Delivery
* **Progressive QA:** Continuous code quality checks and refactoring throughout the lifecycle.
* **Materials:** Preparation of the final comprehensive report and presentation deck.

---

## 🛠 Progress Tracker

| Phase | Task | Status |
| :--- | :--- | :--- |
| **01** | Data Collection | ✅ Done |
| **02** | Model Creation | ✅ Done |
| **03** | Result Reporting | ✅ Done |
| **04** | Plotting Functions | ✅ Done |
| **05** | Final Report & Presentation | 🟨 In Progress |

>Code quality is monitored progressively. 