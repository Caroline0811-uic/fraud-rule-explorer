
# Multi-Objective Evolutionary Optimization for Interpretable Fraud Detection Rule Sets

*A Comparative Study of NSGA-II, SPEA2, and MOEA/D*

**University of Illinois Chicago | IDS 560: Analytics Strategy and Practice | Fall 2026**

**Industry Partner:** TransUnion

## Project Overview

Fraud detection systems must identify fraudulent transactions while minimizing false alerts and maintaining interpretable, manageable decision rules.

This research compares three multi-objective evolutionary algorithms—**NSGA-II, SPEA2, and MOEA/D**—to generate optimized fraud-detection rule sets that balance three competing objectives:

- **Precision:** Maximize the proportion of flagged transactions that are fraudulent.
- **Recall:** Maximize the proportion of fraudulent transactions successfully detected.
- **Complexity:** Minimize the number of conditions required by a rule set.

Rather than producing one universally optimal solution, the algorithms generate multiple rule-set options with different performance trade-offs, allowing organizations to select solutions that align with their operational priorities.

## Dataset

The research uses a synthetic fraud-detection dataset generated with Tonic Fabricate, with TransUnion data available to the project team.

| Dataset characteristic | Value |
|---|---|
| Total transactions | 500,000 |
| Fraudulent transactions | Approximately 2,500 (0.5%) |
| Features | 40 |
| Candidate conditions | 837 |
| Training set | 70% |
| Validation set | 15% |
| Test set | 15% |

Transactions were split chronologically, allowing evaluation on newer, previously unseen observations.

## Methodology

### Fraud Rule Representation

Each candidate solution represents a complete fraud-detection rule set.

- **Conditions** compare feature values against thresholds.
- **Rules** combine conditions using AND.
- **Rule sets** combine individual rules using OR.

The evolutionary algorithms search for combinations of rules that maximize precision and recall while minimizing total conditions.

### Algorithms

| Algorithm | Optimization approach |
|---|---|
| NSGA-II | Ranks competing solutions and preserves diversity across the Pareto front |
| SPEA2 | Maintains an archive of strong trade-off solutions during evolution |
| MOEA/D | Decomposes the multi-objective problem into smaller optimization subproblems |

Six experiments examined feature coverage, search configurations, mutation settings, and condition limits. Hyperparameter tuning used Scikit-learn and Optuna, with validation hypervolume and F1 scores supporting model selection.

Final performance was evaluated using held-out test data.

## Key Findings

- **Feature coverage:** Expanding from six selected features to all 40 features improved F1 scores by approximately 13–15 percentage points.
- **Search configuration:** Larger evolutionary searches improved F1 scores by approximately 16–28 percentage points compared with smaller searches.
- **Rule complexity:** Limiting solutions to 50 conditions retained F1 scores between 76% and 82%, while producing substantially simpler rule sets.
- **Algorithm comparison:** No algorithm consistently outperformed the others across all experiments.
- **Performance trade-offs:** In Experiment 4, SPEA2 achieved the highest precision (91%), while MOEA/D achieved the highest recall (82%) among the selected representative solutions.

These results illustrate the value of comparing multiple optimization methods and presenting a range of interpretable fraud-detection solutions.

## Interactive Streamlit Dashboard

The **TransUnion Fraud Rule Explorer** is an interactive prototype for exploring precomputed optimized rule sets.

Users can:

- Upload optimization results from NSGA-II, SPEA2, and MOEA/D.
- Filter rule sets by minimum precision, minimum recall, and maximum conditions.
- Compare solutions using an interactive precision–recall scatterplot.
- Select individual rule sets to inspect their performance.
- Examine the conditions and feature categories within each selected rule set.

The dashboard supports interactive exploration of research results; it does not perform live optimization or represent a deployed fraud-detection system.

**Live dashboard:** [Add Streamlit application URL]

## Repository Structure

```text
fraud-rule-explorer/
├── .devcontainer/
├── app.py
├── requirements.txt
├── README.md
├── algorithms/        # Algorithm implementations
├── results/           # Precomputed experiment outputs
└── docs/              # Research poster and documentation
```

*The algorithms, results, and documentation folders are proposed additions and will be populated as files are prepared for sharing.*

## Research Team

**University of Illinois Chicago — College of Business Administration**

- Surekha Janapareddy
- Ananya Gannamaneni
- Wenyao Mo
- Caroline Edmunds-Diez

**Faculty Advisor:** Professor Fatemeh Sarayloo

**Industry Partner:** TransUnion

**Industry Sponsors:** Paul Williams and Jonah Henry

## Conclusions and Future Work

The study demonstrates that multi-objective evolutionary optimization can generate interpretable fraud-detection rule sets with meaningful trade-offs among precision, recall, and complexity.

The results also suggest that broader feature coverage, larger searches, and carefully selected complexity constraints can improve the range and usefulness of candidate solutions.

Because the experiments used synthetic data and evolutionary algorithms involve randomness, results should not be interpreted as evidence of production fraud-detection performance.

Future research could investigate additional experimental repetitions, alternative complexity constraints, and validation in operational environments.

## Tools and Technologies

Python · Pandas · NumPy · DEAP · Scikit-learn · Optuna · Matplotlib · Seaborn · Streamlit · GitHub · VS Code · Jupyter

---

*Academic research prototype developed in collaboration with TransUnion as part of the University of Illinois Chicago MSBA program.*
