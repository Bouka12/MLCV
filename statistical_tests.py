import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# All_Experimetnal_Results/
# ├── KDN-K3/
# │   ├── KNN/experiments_results_MLCV_stats_medical_knn_kdn.csv
# │   ├── RF/experiments_results_MLCV_stats_medical_rf_kdn.csv
# │   ├── LR/experiments_results_MLCV_stats_medical_lr_kdn.csv
# │   ├── SVC/experiments_results_MLCV_stats_medical_svc_kdn.csv
# ├── LSC-K3/
# │   ├── KNN/experiments_results_MLCV_stats_medical_knn_lsc.csv
# │   ├── RF/experiments_results_MLCV_stats_medical_rf_lsc.csv
# │   ├── LR/experiments_results_MLCV_stats_medical_lr_lsc.csv
# │   ├── SVC/experiments_results_MLCV_stats_medical_svc_lsc.csv

K=[10, 5, 3]
HM= ["KDN", "LSC"]
algorithms = ["KNN", "LR", "RF", "SVC"]
# path_results_to_read= fr"...\All_Experimental_Results\{hm}-K{K}\{algorithm}\experiment_results_MLCV_stats_medical_{str(algorithm)}_kDN_k{K}_.csv"#{str(hm)}_.csv"#_k{K}_.csv"
# sampling_results_path = fr"...\All_Experimental_Results\{hm}-K{K}\{algorithm}\sampling_methods_{algorithm}_{hm}_k{K}_.csv"

methods = ["RUS", "TomekLinks", "ENN", "NearMiss", "AllKNN", "IHT", "CNN", "NCR", "REEN", "OSS", "CC", "ROS", 
           "SMOTE", "BSMOTE", "SVM", "SMOTEENN", "SMOTETomek" ]


# Related to Table III and IV
for hm in HM:
    results_HM =  {method: {"Wins": None, "Ties": None, "Losses": None} for method in methods}
    sum_hm = {method: {"Wins": [], "Ties": [], "Losses": []} for method in methods}

    for k in K:
        results_K = {method: {"Wins": [], "Ties": [], "Losses": []} for method in methods}
        sum_K = {method: {"Wins": None, "Ties": None, "Losses": None} for method in methods}

        for algo in algorithms:
            sampling_results_path  = fr"...\All_Experimental_Results\{hm}-K{k}\{algo}\sampling_methods_{algo}_{hm}_k{k}_.csv" # -> change
            results = pd.read_csv(sampling_results_path)
            print(results.head())

            for method in methods:
                method_row = results[results['Method'] == method].iloc[0]
                # print(f"method_row: {method_row}")
                # Convert values to int explicitly
                wins = int(method_row['Wins'])
                # print(f"wins: {wins}")
                ties = int(method_row['Ties'])
                # print(f"ties: {ties}")
                losses = int(method_row['Losses'])
                # print(f"losses: {losses}")
                results_K[method]['Wins'].append(wins)
                results_K[method]['Ties'].append(ties)
                results_K[method]['Losses'].append(losses)
        # I want to sum the values in the lists in the dictionary results_K

        for method in methods:
            total_wins = np.sum(results_K[method]['Wins'])
            total_ties = np.sum(results_K[method]['Ties'])
            total_losses = np.sum(results_K[method]['Losses'])
            sum_K[method]['Wins'] = total_wins
            sum_K[method]['Ties'] = total_ties
            sum_K[method]['Losses'] = total_losses

            sum_hm[method]['Wins'].append(total_wins)
            sum_hm[method]['Ties'].append(total_ties)
            sum_hm[method]['Losses'].append(total_losses)

        print(f"Results for {hm}-{k}: {sum_K}")
        df_sum_K = pd.DataFrame.from_dict(sum_K, orient='index').reset_index()
        df_sum_K = df_sum_K.rename(columns={'index':'Method'})
        # Save to CSV:
        df_sum_K.to_csv(fr"...\All_Experimental_Results\{hm}-K{k}\{hm}_sum_{k}_Results.csv", index=False) # -> change the path

    for method in methods:
        total_wins_hm = np.sum(sum_hm[method]['Wins'])
        total_losses_hm = np.sum(sum_hm[method]['Losses'])
        total_ties_hm = np.sum(sum_hm[method]['Ties'])
        results_HM[method]['Wins'] = total_wins_hm
        results_HM[method]['Losses'] = total_losses_hm
        results_HM[method]['Ties'] = total_ties_hm

    print(f"Results for {hm}: {results_HM}")
    df_sum_HM = pd.DataFrame.from_dict(results_HM, orient='index').reset_index()
    df_sum_HM = df_sum_HM.rename(columns={'index':'Method'})
    # Save to CSV:
    df_sum_HM.to_csv(fr"...\All_Experimental_Results\DataLevel_{hm}_WinTieLoss_Results.csv", index=False) # -> change the path



# Table I in the paper:
## GET TABLE OF PERFORMANCE COMPARISON with tolerance for approximate equality epsilon = 1e-4
# baseline_results = results[results["Experiment"].isin(["Baseline", "MLCV"])]
# print(baseline_results.head())

# datasets = results['Dataset'].unique()
# summary = {"Recall": [0, 0, 0], "F1":[0, 0, 0], "Precision": [0, 0, 0]} # because we want to include in our table results like that:  [↑, ≈, ↓] --> high-level summary of results 
# epsilon = 1e-4

# for dataset in datasets:
#     df = results[results['Dataset']==dataset]
#     if len(df)<2:
#         continue # skip if we don't have both Baseline and MLCV
    
#     base = df[df['Experiment']=="Baseline"]
#     mlcv = df[df['Experiment']=="MLCV"]

#     if base.empty or mlcv.empty:
#         continue

#     # Extract metric values
#     for metric, mean_col in [("Recall", "Mean_Recall"), ("F1", "Mean_F1"), ("Precision", "Mean_Precision")]:
#         b = base[mean_col].values[0]
#         m = mlcv[mean_col].values[0]

#         if m - b > epsilon:
#             summary[metric][0] += 1 # ↑
#         elif abs(m-b) <= epsilon:
#             summary[metric][1] +=1 # ≈
#         else: 
#             summary[metric][2] +=1

# for metric in summary:
#     print(f"{metric}: ↑={summary[metric][0]}, ≈={summary[metric][1]}, ↓={summary[metric][2]}")




# Table III, IV 
## COMPARISON OF SAMPLING METHODS 
# for algorithm in algorithms:
#     results_sampling = {
#         'Method': [],
#         'Wins': [],
#         'Ties': [],
#         'Losses': []
#     }

#     methods = ["RUS", "TomekLinks", "ENN", "NearMiss", "AllKNN", "IHT", "CNN", "NCR", "REEN", "OSS", "CC", "ROS", 
#             "SMOTE", "BSMOTE", "SVM", "SMOTEENN", "SMOTETomek" ]
#     epsilon = 1e-4
#     for method in methods:
#         wins = 0
#         ties = 0
#         losses = 0

#         method_0 = results[results['Experiment'] ==f"{method}_0"]
#         method_1 = results[results['Experiment'] ==f"{method}_1"]

#         for dataset in datasets:
#             method_0_result = method_0[method_0['Dataset'] == dataset]
#             method_1_result = method_1[method_1['Dataset'] == dataset]

#             if not method_0_result.empty and not method_1_result.empty:
#                 metric_0 = method_0_result['Mean_F1'].values[0]
#                 metric_1 = method_1_result['Mean_F1'].values[0]

#                 if metric_1 - metric_0> epsilon :# metric_0:
#                     wins +=1
#                 elif abs(metric_1 - metric_0) <= epsilon:
#                     ties +=1
#                 else:
#                     losses +=1

#         results_sampling['Method'].append(method)
#         results_sampling['Wins'].append(wins)
#         results_sampling['Ties'].append(ties)
#         results_sampling['Losses'].append(losses)

#     summary_results_sampling = pd.DataFrame(results_sampling)
#     summary_results_sampling.to_csv(fr"...\All_Experimental_Results\{hm}-K{K}\{algorithm}\sampling_methods_{str(algorithm)}_{hm}_k{K}_.csv") # -> change the path
#     print(summary_results_sampling)



#------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------#

# Table  V, VI
# Perform Wilcoxon signed-rank test

# Get results for a certain algorithm, a hardness metric, a k folds from "All_Experimental_Results"
algorithm="KNN" # for example
path_results_to_read= fr"...\All_Experimental_Results\{hm}-K{K}\{algorithm}\experiment_results_MLCV_stats_medical_{str(algorithm)}_kDN_k{K}_.csv" # -> change the path
results =  pd.read_csv(path_results_to_read)

baseline_recalls = np.array(results[results["Experiment"] == "Baseline"]["Mean_Recall"])
mlcv_recalls = np.array(results[results["Experiment"] == "MLCV"]["Mean_Recall"])

rus0_recalls = np.array(results[results["Experiment"] == "RUS_0"]["Mean_Recall"])
rus1_recalls = np.array(results[results["Experiment"] == "RUS_1"]["Mean_Recall"])

tl0_recalls = np.array(results[results["Experiment"] == "TomekLinks_0"]["Mean_Recall"])
tl1_recalls = np.array(results[results["Experiment"] == "TomekLinks_1"]["Mean_Recall"])

enn0_recalls = np.array(results[results["Experiment"]== "ENN_0"]["Mean_Recall"])
enn1_recalls = np.array(results[results["Experiment"]== "ENN_1"]["Mean_Recall"])

nearm0_recalls = np.array(results[results["Experiment"]== "NearMiss_0"]["Mean_Recall"])
nearm1_recalls = np.array(results[results["Experiment"]== "NearMiss_1"]["Mean_Recall"])

allknn0_recalls = np.array(results[results["Experiment"]== "AllKNN_0"]["Mean_Recall"])
allknn1_recalls = np.array(results[results["Experiment"]== "AllKNN_1"]["Mean_Recall"])

iht0_recalls = np.array(results[results["Experiment"]== "IHT_0"]["Mean_Recall"])
iht1_recalls = np.array(results[results["Experiment"]== "IHT_1"]["Mean_Recall"])
cnn0_recalls = np.array(results[results["Experiment"]== "CNN_0"]["Mean_Recall"])
cnn1_recalls = np.array(results[results["Experiment"]== "CNN_1"]["Mean_Recall"])
ncr0_recalls = np.array(results[results["Experiment"]== "NCR_0"]["Mean_Recall"])
ncr1_recalls = np.array(results[results["Experiment"]== "NCR_1"]["Mean_Recall"])
reen0_recalls = np.array(results[results["Experiment"]== "REEN_0"]["Mean_Recall"])
reen1_recalls = np.array(results[results["Experiment"]== "REEN_1"]["Mean_Recall"])
oss0_recalls = np.array(results[results["Experiment"]== "OSS_0"]["Mean_Recall"])
oss1_recalls = np.array(results[results["Experiment"]== "OSS_1"]["Mean_Recall"])
cc0_recalls = np.array(results[results["Experiment"]== "CC_0"]["Mean_Recall"])
cc1_recalls = np.array(results[results["Experiment"]== "CC_1"]["Mean_Recall"])
ros0_recalls = np.array(results[results["Experiment"]== "ROS_0"]["Mean_Recall"])
ros1_recalls = np.array(results[results["Experiment"]== "ROS_1"]["Mean_Recall"])
smote0_recalls = np.array(results[results["Experiment"]== "SMOTE_0"]["Mean_Recall"])
smote1_recalls = np.array(results[results["Experiment"]== "SMOTE_1"]["Mean_Recall"])
bsmote0_recalls = np.array(results[results["Experiment"]== "BSMOTE_0"]["Mean_Recall"])
bsmote1_recalls = np.array(results[results["Experiment"]== "BSMOTE_1"]["Mean_Recall"])
svm0_recalls = np.array(results[results["Experiment"]== "SVM_0"]["Mean_Recall"])
svm1_recalls = np.array(results[results["Experiment"]== "SVM_1"]["Mean_Recall"])
smotenn0_recalls = np.array(results[results["Experiment"]== "SMOTEENN_0"]["Mean_Recall"])
smotenn1_recalls = np.array(results[results["Experiment"]== "SMOTEENN_1"]["Mean_Recall"])
smotetomek0_recalls = np.array(results[results["Experiment"]== "SMOTETomek_0"]["Mean_Recall"])
smotetomek1_recalls = np.array(results[results["Experiment"]== "SMOTETomek_1"]["Mean_Recall"])


baseline_f1s = np.array(results[results["Experiment"] == "Baseline"]["Mean_F1"])  
mlcv_f1s = np.array(results[results["Experiment"] == "MLCV"]["Mean_F1"])
# hard_f1s = np.array(results[results["Experiment"] == "Hard-Only"]["Mean_F1"])
rus0_f1s = np.array(results[results["Experiment"] == "RUS_0"]["Mean_F1"])
rus1_f1s = np.array(results[results["Experiment"] == "RUS_1"]["Mean_F1"])
tl0_f1s = np.array(results[results["Experiment"] == "TomekLinks_0"]["Mean_F1"])
tl1_f1s = np.array(results[results["Experiment"] == "TomekLinks_1"]["Mean_F1"])
enn0_f1s = np.array(results[results["Experiment"] == "ENN_0"]["Mean_F1"])
enn1_f1s = np.array(results[results["Experiment"] == "ENN_1"]["Mean_F1"])
nearm0_f1s = np.array(results[results["Experiment"] == "NearMiss_0"]["Mean_F1"])
nearm1_f1s = np.array(results[results["Experiment"] == "NearMiss_1"]["Mean_F1"])
allknn0_f1s = np.array(results[results["Experiment"] == "AllKNN_0"]["Mean_F1"])
allknn1_f1s = np.array(results[results["Experiment"] == "AllKNN_1"]["Mean_F1"])
iht0_f1s = np.array(results[results["Experiment"] == "IHT_0"]["Mean_F1"])
iht1_f1s = np.array(results[results["Experiment"] == "IHT_1"]["Mean_F1"])
cnn0_f1s = np.array(results[results["Experiment"] == "CNN_0"]["Mean_F1"])
cnn1_f1s = np.array(results[results["Experiment"] == "CNN_1"]["Mean_F1"])
ncr0_f1s = np.array(results[results["Experiment"] == "NCR_0"]["Mean_F1"])
ncr1_f1s = np.array(results[results["Experiment"] == "NCR_1"]["Mean_F1"])
reen0_f1s = np.array(results[results["Experiment"] == "REEN_0"]["Mean_F1"])
reen1_f1s = np.array(results[results["Experiment"] == "REEN_1"]["Mean_F1"])
oss0_f1s = np.array(results[results["Experiment"] == "OSS_0"]["Mean_F1"])
oss1_f1s = np.array(results[results["Experiment"] == "OSS_1"]["Mean_F1"])
cc0_f1s = np.array(results[results["Experiment"] == "CC_0"]["Mean_F1"])
cc1_f1s = np.array(results[results["Experiment"] == "CC_1"]["Mean_F1"])
ros0_f1s = np.array(results[results["Experiment"] == "ROS_0"]["Mean_F1"])
ros1_f1s = np.array(results[results["Experiment"] == "ROS_1"]["Mean_F1"])
smote0_f1s = np.array(results[results["Experiment"] == "SMOTE_0"]["Mean_F1"])
smote1_f1s = np.array(results[results["Experiment"] == "SMOTE_1"]["Mean_F1"])
bsmote0_f1s = np.array(results[results["Experiment"] == "BSMOTE_0"]["Mean_F1"])
bsmote1_f1s = np.array(results[results["Experiment"] == "BSMOTE_1"]["Mean_F1"])
svm0_f1s = np.array(results[results["Experiment"] == "SVM_0"]["Mean_F1"])
svm1_f1s = np.array(results[results["Experiment"] == "SVM_1"]["Mean_F1"])
smotenn0_f1s = np.array(results[results["Experiment"] == "SMOTEENN_0"]["Mean_F1"])
smotenn1_f1s = np.array(results[results["Experiment"] == "SMOTEENN_1"]["Mean_F1"])
smotetomek0_f1s = np.array(results[results["Experiment"] == "SMOTETomek_0"]["Mean_F1"])
smotetomek1_f1s = np.array(results[results["Experiment"] == "SMOTETomek_1"]["Mean_F1"])


baseline_prec = np.array(results[results["Experiment"] == "Baseline"]["Mean_Precision"])
mlcv_prec = np.array(results[results["Experiment"] == "MLCV"]["Mean_Precision"])
rus0_prec = np.array(results[results["Experiment"] == "RUS_0"]["Mean_Precision"])
rus1_prec = np.array(results[results["Experiment"] == "RUS_1"]["Mean_Precision"])
tl0_prec = np.array(results[results["Experiment"] == "TomekLinks_0"]["Mean_Precision"])  
tl1_prec = np.array(results[results["Experiment"] == "TomekLinks_1"]["Mean_Precision"])
enn0_prec = np.array(results[results["Experiment"] == "ENN_0"]["Mean_Precision"])
enn1_prec = np.array(results[results["Experiment"] == "ENN_1"]["Mean_Precision"])
nearm0_prec = np.array(results[results["Experiment"] == "NearMiss_0"]["Mean_Precision"])
nearm1_prec = np.array(results[results["Experiment"] == "NearMiss_1"]["Mean_Precision"])
allknn0_prec = np.array(results[results["Experiment"] == "AllKNN_0"]["Mean_Precision"])
allknn1_prec = np.array(results[results["Experiment"] == "AllKNN_1"]["Mean_Precision"])
iht0_prec = np.array(results[results["Experiment"] == "IHT_0"]["Mean_Precision"])
iht1_prec = np.array(results[results["Experiment"] == "IHT_1"]["Mean_Precision"])
cnn0_prec = np.array(results[results["Experiment"] == "CNN_0"]["Mean_Precision"])
cnn1_prec = np.array(results[results["Experiment"] == "CNN_1"]["Mean_Precision"])
ncr0_prec = np.array(results[results["Experiment"] == "NCR_0"]["Mean_Precision"])
ncr1_prec = np.array(results[results["Experiment"] == "NCR_1"]["Mean_Precision"])
reen0_prec = np.array(results[results["Experiment"] == "REEN_0"]["Mean_Precision"])
reen1_prec = np.array(results[results["Experiment"] == "REEN_1"]["Mean_Precision"])
oss0_prec = np.array(results[results["Experiment"] == "OSS_0"]["Mean_Precision"])
oss1_prec = np.array(results[results["Experiment"] == "OSS_1"]["Mean_Precision"])
cc0_prec = np.array(results[results["Experiment"] == "CC_0"]["Mean_Precision"])
cc1_prec = np.array(results[results["Experiment"] == "CC_1"]["Mean_Precision"])
ros0_prec = np.array(results[results["Experiment"] == "ROS_0"]["Mean_Precision"])
ros1_prec = np.array(results[results["Experiment"] == "ROS_1"]["Mean_Precision"])
smote0_prec = np.array(results[results["Experiment"] == "SMOTE_0"]["Mean_Precision"])
smote1_prec = np.array(results[results["Experiment"] == "SMOTE_1"]["Mean_Precision"])
bsmote0_prec = np.array(results[results["Experiment"] == "BSMOTE_0"]["Mean_Precision"])
bsmote1_prec = np.array(results[results["Experiment"] == "BSMOTE_1"]["Mean_Precision"])
svm0_prec = np.array(results[results["Experiment"] == "SVM_0"]["Mean_Precision"])
svm1_prec = np.array(results[results["Experiment"] == "SVM_1"]["Mean_Precision"])
smotenn0_prec = np.array(results[results["Experiment"] == "SMOTEENN_0"]["Mean_Precision"])
smotenn1_prec = np.array(results[results["Experiment"] == "SMOTEENN_1"]["Mean_Precision"])

smotetomek0_prec = np.array(results[results["Experiment"] == "SMOTETomek_0"]["Mean_Precision"])
smotetomek1_prec = np.array(results[results["Experiment"] == "SMOTETomek_1"]["Mean_Precision"])

from scipy.stats import wilcoxon


# List to store results for the CSV
results_stats = []

# Define experiment pairs for recall and F1 comparison
experiment_pairs = [
    ("Baseline", "MLCV", baseline_recalls, mlcv_recalls, baseline_f1s, mlcv_f1s, baseline_prec, mlcv_prec),
    ("RUS_0", "RUS_1", rus0_recalls, rus1_recalls, rus0_f1s, rus1_f1s, rus0_prec, rus1_prec),
    ("TomekLinks_0", "TomekLinks_1", tl0_recalls, tl1_recalls, tl0_f1s, tl1_f1s, tl0_prec, tl1_prec),
    ("ENN_0", "ENN_1", enn0_recalls, enn1_recalls, enn0_f1s, enn1_f1s, enn0_prec, enn1_prec),
    ("NearMiss_0", "NearMiss_1", nearm0_recalls, nearm1_recalls, nearm0_f1s, nearm1_f1s, nearm0_prec, nearm1_prec),
    ("AllKNN_0", "AllKNN_1", allknn0_recalls, allknn1_recalls, allknn0_f1s, allknn1_f1s, allknn0_prec, allknn1_prec),
    ("IHT_0", "IHT_1", iht0_recalls, iht1_recalls, iht0_f1s, iht1_f1s, iht0_prec, iht1_prec),
    ("CNN_0", "CNN_1", cnn0_recalls, cnn1_recalls, cnn0_f1s, cnn1_f1s, cnn0_prec, cnn1_prec),
    ("NCR_0", "NCR_1", ncr0_recalls, ncr1_recalls, ncr0_f1s, ncr1_f1s, ncr0_prec, ncr1_prec),
    ("Reen_0", "Reen_1", reen0_recalls, reen1_recalls, reen0_f1s, reen1_f1s, reen0_prec, reen1_prec),
    ("OSS_0", "OSS_1", oss0_recalls, oss1_recalls, oss0_f1s, oss1_f1s, oss0_prec, oss1_prec),
    ("CC_0", "CC_1", cc0_recalls, cc1_recalls, cc0_f1s, cc1_f1s, cc0_prec, cc1_prec),
    ("ROS_0", "ROS_1", ros0_recalls, ros1_recalls, ros0_f1s, ros1_f1s, ros0_prec, ros1_prec),
    ("SMOTE_0", "SMOTE_1", smote0_recalls, smote1_recalls, smote0_f1s, smote1_f1s, smote0_prec, smote1_prec),
    ("BSMOTE_0", "BSMOTE_1", bsmote0_recalls, bsmote1_recalls, bsmote0_f1s, bsmote1_f1s, bsmote0_prec, bsmote1_prec),
    ("SVM_0", "SVM_1", svm0_recalls, svm1_recalls, svm0_f1s, svm1_f1s, svm0_prec, svm1_prec),
    ("SMOTENN_0", "SMOTENN_1", smotenn0_recalls, smotenn1_recalls, smotenn0_f1s, smotenn1_f1s, smotenn0_prec, smotenn1_prec),
    ("SMOTETomek_0", "SMOTETomek_1", smotetomek0_recalls, smotetomek1_recalls, smotetomek0_f1s, smotetomek1_f1s, smotetomek0_prec, smotetomek1_prec),
]

# Perform the Wilcoxon test for each pair
# store the mean and std of the recalls, f1s and precisions of the exp1 and exp2 in a dictionary to add it to the resultd df
def interpret_result(pval, diff):
    if np.isnan(pval) or diff is None:
        return 'Insufficient Data'
    elif pval < 0.05:
        if np.median(diff) >= 0:
            return 'MLCV Significantly Better'
        elif np.median(diff) < 0:
            return 'CV Significantly Better'
        else:
            return 'No Difference (Significant)'
    else:
        return 'Not Significant'
mean_std_dict = []
for exp1, exp2, rec1, rec2, f1_1, f1_2, prec1, prec2 in experiment_pairs:
    # print the mean and std of the recalls of the exp1 and exp2
    print(f"Mean and std of recalls for {exp1}: {np.mean(rec1):.4f} ± {np.std(rec1):.4f}")
    print(f"Mean and std of recalls for {exp2}: {np.mean(rec2):.4f} ± {np.std(rec2):.4f}")
    mean_std_recalls={
        'Experiment 1': exp1,
        'Experiment 2': exp2,
        'Metric': 'Recall',
        'Mean 1': np.mean(rec1),
        'Mean 2': np.mean(rec2),
        'Std 1': np.std(rec1),
        'Std 2': np.std(rec2)
    }
    mean_std_dict.append(mean_std_recalls)
    # Wilcoxon test for recalls
    rec_diff = rec2-rec1
    stat_recalls, p_value_recalls = wilcoxon(rec_diff, zero_method='zsplit')
    result_recalls = {
        'Experiment 1': exp1,
        'Experiment 2': exp2,
        'Metric': 'Recall',
        'Median difference': np.median(rec_diff),
        'Statistic': stat_recalls,
        'P-value': p_value_recalls,
        'Interpretation': interpret_result(p_value_recalls, rec_diff)
    }
    results_stats.append(result_recalls)

    # print the mean and std of the f1s of the exp1 and exp2
    print(f"Mean and std of f1s for {exp1}: {np.mean(f1_1):.4f} ± {np.std(f1_1):.4f}")
    print(f"Mean and std of f1s for {exp2}: {np.mean(f1_2):.4f} ± {np.std(f1_2):.4f}")

    mean_std_f1 = {
        'Experiment 1': exp1,
        'Experiment 2': exp2,
        'Metric': 'F1',
        'Mean 1': np.mean(f1_1),
        'Mean 2': np.mean(f1_2),
        'Std 1': np.std(f1_1),
        'Std 2': np.std(f1_2)
    }
    mean_std_dict.append(mean_std_f1)

    # Wilcoxon test for F1 scores
    if np.array_equal(f1_1, f1_2):
        print(f"Warning: F1 values for {exp1} and {exp2} are equal.")
    else:
        print(f"F1 values for {exp1} and {exp2} are not equal.")
    f1_diff = f1_2-f1_1
    stat_f1, p_value_f1 = wilcoxon(f1_diff, zero_method='zsplit')
    # stat_f1, p_value_f1 = wilcoxon(f1_1, f1_2, zero_method='zsplit')
    result_f1 = {
        'Experiment 1': exp1,
        'Experiment 2': exp2,
        'Metric': 'F1',
        'Median difference': np.median(f1_diff),
        'Statistic': stat_f1,
        'P-value': p_value_f1,
        'Interpretation': interpret_result(p_value_f1, f1_diff)
    }
    results_stats.append(result_f1)

    # print the mean and std of the precisions of the exp1 and exp2
    print(f"Mean and std of precisions for {exp1}: {np.mean(prec1):.4f} ± {np.std(prec1):.4f}")
    print(f"Mean and std of precisions for {exp2}: {np.mean(prec2):.4f} ± {np.std(prec2):.4f}")
    mean_std_precisions = {
        'Experiment 1': exp1,
        'Experiment 2': exp2,
        'Metric': 'Precision',
        'Mean 1': np.mean(prec1),
        'Mean 2': np.mean(prec2),
        'Std 1': np.std(prec1),
        'Std 2': np.std(prec2)
    }
    mean_std_dict.append(mean_std_precisions)
    # Wilcoxon test for precision scores
    if np.array_equal(prec1, prec2):
        print(f"Warning: Precision values for {exp1} and {exp2} are equal.")
    else:
        print(f"Precision values for {exp1} and {exp2} are not equal.")
    prec_diff = prec2-prec1
    stat_prec, p_value_prec = wilcoxon(prec_diff, zero_method='zsplit')
    # stat_prec, p_value_prec = wilcoxon(prec1, prec2, zero_method='zsplit')
    result_prec = {
        'Experiment 1': exp1,
        'Experiment 2': exp2,
        'Metric': 'Precision',
        'Median difference': np.median(prec_diff),
        'Statistic': stat_prec,
        'P-value': p_value_prec,
        'Interpretation': interpret_result(p_value_prec, prec_diff)
    }
    results_stats.append(result_prec)


# Create DataFrame from results and save to CSV
mean_std_df = pd.DataFrame(mean_std_dict)
mean_std_df.to_csv(r"...\All_Experimental_Results\LSC-K10\SVC\experiment_results_MLCV_stats_medical_SVC_mean_std_.csv", index=False) # -> change the path
print(mean_std_df)
results_df = pd.DataFrame(results_stats)
print(f" Results of Wilcoxon tests for MLCV and Baseline on medical ({len(mlcv_f1s)} datasets):")
print(results_df)
# save to csv in the same directry as the results csv
results_df.to_csv(r"...\All_Experimental_Results\LSC-K10\SVC\experiment_results_MLCV_stats_medical_SVC_wilcoxon_.csv", index=False) # -> change the path

