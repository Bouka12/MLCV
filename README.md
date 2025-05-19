
# Multi-Level Cross-Validation for Imbalanced Medical Datasets

🔒 This repository is part of a submission under review. The authors have anonymized the repository to comply with the double-blind review process.

---

## ⚠️ Disclaimer

> This code is a research prototype created for internal experimentation. It uses **absolute paths** and **custom folder structures**, so you may need to adapt parts of the code to run it on your machine.


---

## 📁 Project Structure

IHA-CODE/
├── All_Experimental_Results/ # Output results for all experiments
├── data_complexity_plots/ # Plots for data complexity analysis
├── hardness_shift_across_classes... # Analysis of hardness shifts across train/test
├── Instance_hardness_dist_violin_plots # Violin plots of hardness distributions
├── inputs/ # Input datasets and metadata
├── paper_figs/ # Final figures used in the paper
├── data_info.py # Loads dataset metadata
├── experiments_medical.py # Main experimental pipeline
├── hard_metrics_med.py # Computes instance hardness metrics
├── plot_distribution_shift.py # Visualizes distribution shifts
├── plot_hardness_distribution.py # Plots instance hardness distributions
├── statistical_tests.py # Performs statistical significance tests
└── README.md # This file

## 🛠 Requirements

This project assumes that the standard scientific Python stack is already installed, including:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `scipy`

### 📦 Project-Specific Dependencies

Install the additional dependencies:

```bash
pip install pyhard==2.2.4 problexity==0.5.9 imbalanced-learn

