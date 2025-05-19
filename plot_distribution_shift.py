# In the following: we plot the hardness distribution of train and test sets per class and for each fold
# The kilmogorov Smirnov statistical test is used to check the statistical similarity between each pair of train test sets from each class and for each fold

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import  LabelEncoder
from scipy.stats import ks_2samp
import random
from tqdm import tqdm
from pyhard.measures import ClassificationMeasures
from hard_metrics_med import merge_groups

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
# Use the classic scientific style
# sns.set(style="whitegrid")
# Set random seed
np.random.seed(42)
random.seed(42)

# Hyperparameters
N_ITERATIONS = 30
N_SPLITS = 3
N_BINS = 4  # Number of bins for quantile binning
# HARDNESS_PERCENTILE = 70  # Top % hardest instances
RESULTS_FILE = "experiment_results_MLCV_medical_plotting.csv"  # Define results file
RESULTS_FILE_STAT = "experiment_results_MLCV_stats_medical_plotting.csv"  # Define results file for stats
# Generate random seeds
tqdm_seeds = [random.randint(0, 1_000_000) for _ in range(N_ITERATIONS)]


# Initialize CSV if it doesn't exist
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w") as f:
        f.write("Dataset,Experiment,Iteration,Recall,F1,Precision,Mean_Recall,Std_Recall,Mean_F1,Std_F1,Mean_Precision,Std_Precision\n")

if not os.path.exists(RESULTS_FILE_STAT):
    with open(RESULTS_FILE_STAT, "w") as f:
        f.write("Dataset,Experiment,Mean_Recall,Std_Recall,Mean_F1,Std_F1,Mean_Precision,Std_Precision\n") 



def evaluate_model_baseline(dataset_name, X, y, hm_lsc_scores,strata_labels, random_seed, plot_seed, n_splits=N_SPLITS):
    """
    Performs Stratified K-Fold evaluation on given dataset.
    Basic cross validation without undersampling and without hardness bins.
    Plot the train vs. test sets distribution for each class across folds
    Perform the Kolmogorov Smirnov Test and show the results (statistic, p_value) down the plot
    Save the plot in the directory
    """

    # model = RandomForestClassifier(random_state=random_seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    # recall_scores, f1_scores, precision_scores = [], [], []
    for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        if plot_seed:
            print(f"Fold {idx+1}/{n_splits} - Train indices: {train_idx}, Test indices: {test_idx}")
            # Plot the hardness distribution of train and test sets for each fold :violin plot
            # 1 - get the hardness distribution of train and test sets from hm_lsc_scores
            ## get min / maj index of the train and test sets maj = 0, min = 1
            min_idx = np.where(y[train_idx] == 1)[0]
            maj_idx = np.where(y[train_idx] == 0)[0]
            train_min_idx = np.where(y[train_idx] == 1)[0]
            train_maj_idx = np.where(y[train_idx] == 0)[0]
            test_min_idx = np.where(y[test_idx] == 1)[0]
            test_maj_idx = np.where(y[test_idx] == 0)[0]
            # get the hardness scores of train and test sets
            train_min_hardness = hm_lsc_scores[train_min_idx]
            test_min_hardness = hm_lsc_scores[test_min_idx]
            train_maj_hardness = hm_lsc_scores[train_maj_idx]
            test_maj_hardness = hm_lsc_scores[test_maj_idx]

            train_hardness = hm_lsc_scores[train_idx]
            test_hardness = hm_lsc_scores[test_idx]

            # hardness strata distribution
            # train_min_hardness = strata_labels[train_min_idx]
            # train_maj_hardness = strata_labels[train_maj_idx]
            # test_min_hardness = strata_labels[test_min_idx]
            # test_maj_hardness = strata_labels[test_maj_idx]

            # train_hardness = strata_labels[train_idx]
            # test_hardness= strata_labels[test_idx]

            # Kilmogorv smirnov test
            ks_stat_min, p_value_min = ks_2samp(train_min_hardness, test_min_hardness)
            print(f"KS Statistic (Minority): {ks_stat_min}, p-value: {p_value_min}")
            ks_stat_maj, p_value_maj = ks_2samp(train_maj_hardness, test_maj_hardness)
            print(f"KS Statistic (Majority): {ks_stat_maj}, p-value: {p_value_maj}")

            # Ks test: Train vs Test
            # ks_stat, ks_p_value = ks_2samp(train_hardness, test_hardness)
            # print(f"KS Statistic: {ks_stat}, p-value: {ks_p_value}")

            # put all in one dataframe to plot it
            data = {
                'Hardness': list(train_min_hardness) + list(train_maj_hardness) + list(test_min_hardness) + list(test_maj_hardness),
                'Split_Class': ['Train Min'] * len(train_min_hardness) + ['Train Maj'] * len(train_maj_hardness) + ['Test Min'] * len(test_min_hardness) + ['Test Maj'] * len(test_maj_hardness)
            }
            df = pd.DataFrame(data)

            # 2 - plot the distribution of train and test sets: min and maj
            # plt.figure(figsize=(10, 6))
            # light_green = '#3CB371'
            # light_violet = '#8A2BE2'
            # sns.set(style="whitegrid")
            light_blue =  'blue' #tab:blue' #'#1f77b4' #'#069AF3' #'cornflowerblue'  #'#4a90e2' #'#1f77b4'
            light_orange = 'forestgreen'#'tab:orange' #'#ff7f0e' #'#FF6347' # '#f5a623'  #
            palette_ = {'Train Min': light_blue, 'Train Maj':light_orange, 'Test Min': light_blue, 'Test Maj': light_orange}           
            # caption_ks = (
            #     fr"\shortstack{{\textit{{KS Statistic (Minority): {ks_stat_min:.3f}, p-value: {p_value_min:.3f}}}\\"
            #     fr"\textit{{KS Statistic (Majority): {ks_stat_maj:.3f}, p-value: {p_value_maj:.3f}}}}}"
            # )
            caption_ks = (
                    fr"\begin{{flushleft}}\textit{{KS Statistic (Minority): {ks_stat_min:.3f}, p-value: {p_value_min:.3f}}} \\ "
                    fr"\textit{{KS Statistic (Majority): {ks_stat_maj:.3f}, p-value: {p_value_maj:.3f}}}\end{{flushleft}}"
                )

            # Create figure and axes with a larger size
            fig, ax = plt.subplots(figsize=(12, 8))  # Increased figure size

            sns.violinplot(x='Split_Class', y='Hardness', palette=palette_, data=df, ax=ax)

            # Title
            # ax.set_title(f'Hardness Distribution - Fold {idx + 1}', fontsize=16)

            # Add caption using `fig.text` — this is more reliable than `figtext`
            fig.text(
                0.5, -0.02,  # x=0.5 center, y=-0.12 (further down for more space)
                caption_ks,
                # caption_cramer,
                ha='center',
                fontsize=12,
                style='italic',
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 5}
            )

            # Adjust layout to make space at the bottom for the caption
            fig.subplots_adjust(bottom=0.1)  # Decrease bottom margin for more space
            path =r"..." # change the path
            # 3 - save the plot fo each fold "baseline_fold_{idx+1}_hardness_distribution.png"
            plt.savefig(f"{path}/{dataset_name}_baseline_fold_{idx+1}_hmlsc_dist_KSTest.png", bbox_inches='tight')
            # plt.show()

    return 0, 0, 0 



def evaluate_model_MLCV(dataset_name, X, y, strata_labels, hm_lsc_scores, random_seed, plot_seed, n_splits=N_SPLITS):
    """
    Performs Stratified K-Fold evaluation on given dataset.
    Plot the train vs. test sets distribution for each class across folds
    Perform the Kolmogorov Smirnov Test and show the results (statistic, p_value) down the plot
    Save the plot in the directory
    """
    # Initiate the model
    # model = RandomForestClassifier(random_state=random_seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    # strata_labels = merge_groups(y, hardness_bins, threshold=4)

    # recall_scores, f1_scores, precision_scores = [], [], []
    for idx, (train_idx, test_idx) in enumerate(skf.split(X, strata_labels)):
        
        
        X_train, y_train = X[train_idx], y[train_idx]
        if plot_seed:
            print(f"Fold {idx+1}/{n_splits} - Train indices: {train_idx}, Test indices: {test_idx}")
            # Plot the hardness distribution of train and test sets for each fold :violin plot
            # 1 - get the hardness distribution of train and test sets from hm_lsc_scores
            ## get min / maj index of the train and test sets maj = 0, min = 1
            min_idx = np.where(y[train_idx] == 1)[0]
            maj_idx = np.where(y[train_idx] == 0)[0]
            train_min_idx = np.where(y[train_idx] == 1)[0]
            train_maj_idx = np.where(y[train_idx] == 0)[0]
            test_min_idx = np.where(y[test_idx] == 1)[0]
            test_maj_idx = np.where(y[test_idx] == 0)[0]
            # get the hardness scores of train and test sets
            train_min_hardness = hm_lsc_scores[train_min_idx]
            test_min_hardness = hm_lsc_scores[test_min_idx]
            train_maj_hardness = hm_lsc_scores[train_maj_idx]
            test_maj_hardness = hm_lsc_scores[test_maj_idx]
            
            train_hardness = hm_lsc_scores[train_idx]
            test_hardness = hm_lsc_scores[test_idx]
            # train_min_hardness = strata_labels[train_min_idx]
            # train_maj_hardness = strata_labels[train_maj_idx]
            # test_min_hardness = strata_labels[test_min_idx]
            # test_maj_hardness = strata_labels[test_maj_idx]

            # train_hardness = strata_labels[train_idx]
            # test_hardness = strata_labels[test_idx]


            # # Kilmogorv smirnov test: Train vs Test
            # ks_stat, ks_p_value = ks_2samp(train_hardness, test_hardness)
            # print(f"KS Statistic: {ks_stat}, p-value: {ks_p_value}")
            # # Kilmogorv smirnov test
            ks_stat_min, p_value_min = ks_2samp(train_min_hardness, test_min_hardness)
            print(f"KS Statistic (Minority): {ks_stat_min}, p-value: {p_value_min}")
            ks_stat_maj, p_value_maj = ks_2samp(train_maj_hardness, test_maj_hardness)
            print(f"KS Statistic (Majority): {ks_stat_maj}, p-value: {p_value_maj}")

            # put all in one dataframe to plot it
            data = {
                'Hardness': list(train_min_hardness) + list(train_maj_hardness) + list(test_min_hardness) + list(test_maj_hardness),
                'Split_Class': ['Train Min'] * len(train_min_hardness) + ['Train Maj'] * len(train_maj_hardness) + ['Test Min'] * len(test_min_hardness) + ['Test Maj'] * len(test_maj_hardness)
            }
            df = pd.DataFrame(data)
            # sns.set(style="whitegrid")

            # 2 - plot the distribution of train and test sets: min and maj
            light_blue =  'blue'#1f77b4' #'#069AF3' #'cornflowerblue'  #'#4a90e2' #'#1f77b4'
            light_orange = 'forestgreen'#'#ff7f0e' #'#FF6347' # '#f5a623'  #
            palette_ = {'Train Min': light_blue, 'Train Maj':light_orange, 'Test Min': light_blue, 'Test Maj': light_orange}           

            caption_ks = (
                    fr"\begin{{flushleft}}\textit{{KS Statistic (Minority): {ks_stat_min:.3f}, p-value: {p_value_min:.3f}}} \\ "
                    fr"\textit{{KS Statistic (Majority): {ks_stat_maj:.3f}, p-value: {p_value_maj:.3f}}}\end{{flushleft}}"
                )


            # Create figure and axes with a larger size
            fig, ax = plt.subplots(figsize=(12, 8))  # Increased figure size

            sns.violinplot(x='Split_Class', y='Hardness', palette=palette_, data=df, ax=ax)

            # Title
            # ax.set_title(f'Hardness Distribution - Fold {idx + 1}', fontsize=16)

            # Add caption using `fig.text` — this is more reliable than `figtext`
            fig.text(
                0.5, -0.02,  # x=0.5 center, y=-0.12 (further down for more space)
                caption_ks,
                # caption_cramer,
                ha='center',
                fontsize=12,
                style='italic',
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 5}
            )

            # Adjust layout to make space at the bottom for the caption
            fig.subplots_adjust(bottom=0.1)  # Decrease bottom margin for more space
            path =r"..." # change the path
            plt.savefig(f"{path}/{dataset_name}_MLCV_fold_{idx + 1}_hmlsc_dist_KSTest.png", bbox_inches='tight')
            # plt.show()

    
    return 0, 0, 0 



def perform_experiment(dataset_name, X, y):
    """
    Runs Baseline, MLCV experiments.
    
    Each experiment is run for N_ITERATIONS with different random seeds.
    """
    
    df = pd.DataFrame(np.column_stack([X, y]))
    # df.columns = [f"feature_{i}" for i in range(df.shape[1] - 1)] + ["target"]
    n_features = X.shape[1]
    column_names = [f"feature_{i}" for i in range(n_features)] + ["target"]
    df.columns = column_names

    # Calculate instance hardness measures
    HM = ClassificationMeasures(df)
    data_HM = HM.calculate_all()
    hm_lsc = np.array(data_HM["feature_LSC"])
    # norm_hm_lsc = scaler.fit_transform(hm_lsc.reshape(-1, 1)).flatten()
    # Use quantile binning to create hardness bins
    bin_labels =pd.qcut(hm_lsc, q=N_BINS, labels=False, duplicates="drop")
    print(f"unique bin_labels: {np.unique(bin_labels)}")
    strata_labels = merge_groups(y, bin_labels, threshold=4)
    
    results = {}
    for exp_name in ["Baseline",'MLCV']: # "RUS_0", "RUS_1", "MLCV", "TomekLinks_0", "TomekLinks_1", "ENN_0", "ENN_1", "NearMiss_0", "NearMiss_1", "AllKNN_0", "AllKNN_1", "IHT_0", "IHT_1", "CNN_0", "CNN_1", "NCR_0", "NCR_1", "REEN_0", "REEN_1", "OSS_0", "OSS_1", "CC_0", "CC_1", "ROS_0", "ROS_1", "SMOTE_0", "SMOTE_1", "BSMOTE_0", "BSMOTE_1", "ADASYN_0", "ADASYN_1", "SVM_0", "SVM_1", "SMOTEN_0", "SMOTEN_1", "SMOTEENN_0", "SMOTEENN_1", "SMOTETomek_0", "SMOTETomek_1"]: # "KMeans_0", "KMeans_1",
        results[exp_name] = {"per_seed": [], "mean":None, "std":None}
    seed_0 = True
    for seed in tqdm_seeds:

        # Baseline
        base_recall, base_f1, base_precision = evaluate_model_baseline(dataset_name, X, y, hm_lsc_scores=hm_lsc,strata_labels=strata_labels, plot_seed=seed_0, random_seed=seed)
        results["Baseline"]["per_seed"].append((base_recall, base_f1, base_precision))    

        # MLCV
        mlcv_recall, mlcv_f1, mlcv_precision = evaluate_model_MLCV(dataset_name, X, y,strata_labels=strata_labels, hm_lsc_scores=hm_lsc, plot_seed=seed_0, random_seed=seed)
        results["MLCV"]["per_seed"].append((mlcv_recall, mlcv_f1, mlcv_precision))

        if seed_0:
            seed_0= False

    for exp_name in results:
        per_seed = np.array(results[exp_name]["per_seed"])
        results[exp_name]["mean"] = np.mean(per_seed, axis=0)
        results[exp_name]["std"] = np.std(per_seed, axis=0) 
    return results

# Medical datasets (12 datasets):
# df_names =  [
#     'BCWDD', 'Haberman', 'HeartCleveland', 'Hepatitis',
#     'Hypothyroid', 'ILPD', 'KidneyDisease', 'NewThyroid1', 'NewThyroid2',
#     'Pima', 'Thoracic', 'Vertebral'
# ]

# The example showed in the DSAA paper:
df_names = ['BCWDD', 'Hypothyroid', 'ILPD']


# Create an empty list to store all results
all_results = []

# Process each dataset
for dataset_name in tqdm(df_names, desc="Processing Datasets"):
    print(f"Processing dataset: {dataset_name}")  # Track progress

    df_path = rf"...\inputs\{dataset_name}_processed.csv"

    # medical datasets:
    data = pd.read_csv(df_path)
    X, y = np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1])


    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Run experiment
    results = perform_experiment(dataset_name, X, y)
    dataset_results = []
    dataset_res_stats = []
    # # Convert dictionary to structured DataFrame
    for exp_type, stats in results.items():
        mean_rec, mean_f1, mean_prec = stats["mean"]
        std_rec, std_f1, std_prec = stats["std"]
        for i, (rec, f1, prec) in enumerate(stats["per_seed"]):
            dataset_results.append([dataset_name, exp_type, i, rec, f1, prec])
        dataset_res_stats.append([dataset_name, exp_type, mean_rec, std_rec, mean_f1, std_f1, mean_prec, std_prec])


