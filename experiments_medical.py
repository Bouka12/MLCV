# In the following: we perform the baseline cross-validation and the multi-level cross-validation 
# Imply four different classifiers for classification: KNN, LR, RF, SVC
# Balancing data with undersampling, oversampling, and hybrid from imblearn
# Get the results on different set of the selected medical  datasets and save the results in csv files

import numpy as np
import os
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import  LabelEncoder
import random
from tqdm import tqdm
from pyhard.measures import ClassificationMeasures
from hard_metrics_med import merge_groups
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks, NearMiss, AllKNN, InstanceHardnessThreshold, CondensedNearestNeighbour, RepeatedEditedNearestNeighbours, OneSidedSelection, NeighbourhoodCleaningRule, ClusterCentroids
from imblearn.over_sampling import SMOTE, RandomOverSampler, BorderlineSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
# Set random seed
np.random.seed(42)
random.seed(42)

# Hyperparameters
N_ITERATIONS = 30
N_SPLITS = 5 # in the set k in {3, 5, 10}
HM_CODE = "kDN" # the hardness metric code in {"kDN", "LSC"}
HM_METRIC = f"feature_{HM_CODE}" # the hardness metric in the library is referred to as {"feature_kDN", "feature_LSC"}
N_BINS = 4  # Number of bins for quantile binning
ALGORITHM = "svc"
RESULTS_FILE = f"experiment_results_MLCV_medical_{ALGORITHM}_{HM_CODE}_k{N_SPLITS}.csv"  # Define results file
RESULTS_FILE_STAT = f"experiment_results_MLCV_stats_medical_{ALGORITHM}_{HM_CODE}_k{N_SPLITS}.csv"  # Define results file for stats
# Generate random seeds
tqdm_seeds = [random.randint(0, 1_000_000) for _ in range(N_ITERATIONS)]


# Initialize CSV if it doesn't exist
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w") as f:
        f.write("Dataset,Experiment,Iteration,Recall,F1,Precision,Mean_Recall,Std_Recall,Mean_F1,Std_F1,Mean_Precision,Std_Precision\n")

if not os.path.exists(RESULTS_FILE_STAT):
    with open(RESULTS_FILE_STAT, "w") as f:
        f.write("Dataset,Experiment,Mean_Recall,Std_Recall,Mean_F1,Std_F1,Mean_Precision,Std_Precision\n") 



def evaluate_model_baseline(X, y, random_seed, n_splits=N_SPLITS):
    """
    Performs Stratified K-Fold evaluation on given dataset.
    Basic cross validation without undersampling and without hardness bins.
    """

    # model = RandomForestClassifier(random_state=random_seed)
    model = SVC(random_state=random_seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    recall_scores, f1_scores, precision_scores = [], [], []
    for train_idx, test_idx in skf.split(X, y):
        # Apply hardness-preserving undersampling
        # X_train, y_train = hardness_preserving_undersampling(X[train_idx], y[train_idx], hardness_bins[train_idx], random_seed)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        # Evaluate performance on the test set
        recall_scores.append(recall_score(y[test_idx], y_pred))
        f1_scores.append(f1_score(y[test_idx], y_pred))
        precision_scores.append(precision_score(y[test_idx], y_pred))
    
    return np.mean(recall_scores), np.mean(f1_scores), np.mean(precision_scores)


def evaluate_model_undersampling(X, y, strata_labels, random_seed, undersampling_method='random', n_splits=N_SPLITS):
    """
    Performs Stratified K-Fold evaluation on given dataset with different undersampling methods.
    Supports 'random', 'nearmiss', and 'tomeklinks' undersampling methods.
    
    Parameters:
    - X: Features
    - y: Target labels
    - strata_labels: Labels for stratified splitting
    - random_seed: Random seed for reproducibility
    - undersampling_method: The undersampling technique ('random', 'nearmiss', or 'tomeklinks')
    - n_splits: Number of splits for StratifiedKFold
    
    Returns:
    - rec_0, f1_0, prec_0: Performance metrics without strata-based splitting
    - rec_1, f1_1, prec_1: Performance metrics with strata-based splitting
    """
    # Select the undersampling method
    if undersampling_method == 'random':
        sampler = RandomUnderSampler(random_state=random_seed)
    elif undersampling_method == 'nearmiss':
        sampler = NearMiss()
    elif undersampling_method == 'tomeklinks':
        sampler = TomekLinks()
    elif undersampling_method == 'allknn':
        sampler = AllKNN()
    elif undersampling_method == "cluster_centroids":
        sampler = ClusterCentroids(random_state=random_seed)
    elif undersampling_method == 'instance_hardness':
        sampler = InstanceHardnessThreshold(random_state=random_seed)
    elif undersampling_method == 'condensed_nearest':
        sampler = CondensedNearestNeighbour(random_state=random_seed)
    elif undersampling_method == 'repeated_edited':
        sampler = RepeatedEditedNearestNeighbours()
    elif undersampling_method == 'one_sided_selection':
        sampler = OneSidedSelection(random_state=random_seed)
    elif undersampling_method == 'neighbourhood_cleaning_rule':
        sampler = NeighbourhoodCleaningRule()
    elif undersampling_method == 'edited_nearest':
        sampler = EditedNearestNeighbours()
    else:
        raise ValueError("Unsupported undersampling method. Choose 'random', 'nearmiss', or 'tomeklinks'.")

    # model = RandomForestClassifier(random_state=random_seed)
    model = SVC(random_state=random_seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    # Store the evaluation metrics
    recall_scores, f1_scores, precision_scores = [], [], []
    
    # Without strata-based splitting
    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]

        # Apply the selected undersampling technique
        X_res, y_res = sampler.fit_resample(X_train, y_train)

        # Train the model on the resampled data
        model.fit(X_res, y_res)
        y_pred = model.predict(X[test_idx])

        # Evaluate performance on the test set
        recall_scores.append(recall_score(y[test_idx], y_pred))
        f1_scores.append(f1_score(y[test_idx], y_pred))
        precision_scores.append(precision_score(y[test_idx], y_pred))
    
    # Calculate mean performance without strata-based splitting
    rec_0, f1_0, prec_0 = np.mean(recall_scores), np.mean(f1_scores), np.mean(precision_scores)

    # With strata-based splitting
    recall_scores, f1_scores, precision_scores = [], [], []    
    for train_idx, test_idx in skf.split(X, strata_labels):
        X_train, y_train = X[train_idx], y[train_idx]

        # Apply the selected undersampling technique
        X_res, y_res = sampler.fit_resample(X_train, y_train)

        # Train the model on the resampled data
        model.fit(X_res, y_res)
        y_pred = model.predict(X[test_idx])

        # Evaluate performance on the test set
        recall_scores.append(recall_score(y[test_idx], y_pred))
        f1_scores.append(f1_score(y[test_idx], y_pred))
        precision_scores.append(precision_score(y[test_idx], y_pred))
    
    # Calculate mean performance with strata-based splitting
    rec_1, f1_1, prec_1 = np.mean(recall_scores), np.mean(f1_scores), np.mean(precision_scores)
    
    return rec_0, f1_0, prec_0, rec_1, f1_1, prec_1

def evaluate_model_MLCV(X, y, hardness_bins, random_seed, n_splits=N_SPLITS):
    """Performs Stratified K-Fold evaluation on given dataset."""
    # Initiate the model
    # model = RandomForestClassifier(random_state=random_seed)
    model = SVC(random_state=random_seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    strata_labels = merge_groups(y, hardness_bins, threshold=N_SPLITS)

    recall_scores, f1_scores, precision_scores = [], [], []
    for train_idx, test_idx in skf.split(X, strata_labels):
        # Apply hardness-preserving undersampling
        X_train, y_train = X[train_idx], y[train_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X[test_idx])

        # Evaluate performance on the test set
        recall_scores.append(recall_score(y[test_idx], y_pred))
        f1_scores.append(f1_score(y[test_idx], y_pred))
        precision_scores.append(precision_score(y[test_idx], y_pred))
    
    return np.mean(recall_scores), np.mean(f1_scores), np.mean(precision_scores)


def evaluate_model_oversampling(X, y, strata_labels, random_seed, oversampling_method='random', n_splits=N_SPLITS):
    # from sklearn.exceptions import NotFittedError
    import traceback

    def get_sampler(method, seed):
        if method == 'random':
            return RandomOverSampler(random_state=seed)
        elif method == 'smote':
            return SMOTE(random_state=seed)
        elif method == 'borderline_smote':
            return BorderlineSMOTE(random_state=seed)
        elif method == 'svm_smote':
            return SVMSMOTE(random_state=seed)
        elif method == 'smoteenn':
            return SMOTEENN(random_state=seed)
        elif method == 'smotetomek':
            return SMOTETomek(random_state=seed)
        else:
            raise ValueError("Unsupported oversampling method.")

    # model = RandomForestClassifier(random_state=random_seed)
    model = SVC(random_state=random_seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    recall_scores, f1_scores, precision_scores = [], [], []

    # ---------- Without strata-based splitting ----------
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        try:
            X_train, y_train = X[train_idx], y[train_idx]
            sampler = get_sampler(oversampling_method, random_seed)
            X_res, y_res = sampler.fit_resample(X_train, y_train)

            model.fit(X_res, y_res)
            y_pred = model.predict(X[test_idx])

            recall_scores.append(recall_score(y[test_idx], y_pred))
            f1_scores.append(f1_score(y[test_idx], y_pred))
            precision_scores.append(precision_score(y[test_idx], y_pred))
        except Exception as e:
            print(f"[Fold {fold_idx}] [ERROR - No Stratified] {oversampling_method} failed: {e}")
            traceback.print_exc()
            return None, None, None, None, None, None

    rec_0, f1_0, prec_0 = np.mean(recall_scores), np.mean(f1_scores), np.mean(precision_scores)

    # ---------- With strata-based splitting ----------
    recall_scores, f1_scores, precision_scores = [], [], []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, strata_labels)):
        try:
            X_train, y_train = X[train_idx], y[train_idx]
            sampler = get_sampler(oversampling_method, random_seed)
            X_res, y_res = sampler.fit_resample(X_train, y_train)

            model.fit(X_res, y_res)
            y_pred = model.predict(X[test_idx])

            recall_scores.append(recall_score(y[test_idx], y_pred))
            f1_scores.append(f1_score(y[test_idx], y_pred))
            precision_scores.append(precision_score(y[test_idx], y_pred))
        except Exception as e:
            print(f"[Fold {fold_idx}] [ERROR - Stratified] {oversampling_method} failed: {e}")
            traceback.print_exc()
            return None, None, None, None, None, None

    rec_1, f1_1, prec_1 = np.mean(recall_scores), np.mean(f1_scores), np.mean(precision_scores)
    return rec_0, f1_0, prec_0, rec_1, f1_1, prec_1


def perform_experiment(X, y):
    """
    Runs Baseline, MLCV, data-level intervention with baseline CV and MLCV experiments.
    
    Each experiment is run for N_ITERATIONS with different random seeds.
    """
    # scaler = MinMaxScaler()
    
    df = pd.DataFrame(np.column_stack([X, y]))
    # df.columns = [f"feature_{i}" for i in range(df.shape[1] - 1)] + ["target"]
    n_features = X.shape[1]
    column_names = [f"feature_{i}" for i in range(n_features)] + ["target"]
    df.columns = column_names

    # Calculate instance hardness measures
    HM = ClassificationMeasures(df)
    data_HM = HM.calculate_all()
    hm_lsc = np.array(data_HM[HM_METRIC])
    # norm_hm_lsc = scaler.fit_transform(hm_lsc.reshape(-1, 1)).flatten()
    # Use quantile binning to create hardness bins
    bin_labels =pd.qcut(hm_lsc, q=N_BINS, labels=False, duplicates="drop")
    print(f"unique bin_labels: {np.unique(bin_labels)}")
    strata_labels = merge_groups(y, bin_labels, threshold=N_SPLITS)
    
    results = {}
    for exp_name in ["Baseline", "RUS_0", "RUS_1", "MLCV", "TomekLinks_0", "TomekLinks_1", "ENN_0", "ENN_1", "NearMiss_0", "NearMiss_1", "AllKNN_0", "AllKNN_1", "IHT_0", "IHT_1", "CNN_0", "CNN_1", "NCR_0", "NCR_1", "REEN_0", "REEN_1", "OSS_0", "OSS_1", "CC_0", "CC_1", "ROS_0", "ROS_1", "SMOTE_0", "SMOTE_1", "BSMOTE_0", "BSMOTE_1", "ADASYN_0", "ADASYN_1", "SVM_0", "SVM_1", "SMOTEN_0", "SMOTEN_1", "SMOTEENN_0", "SMOTEENN_1", "SMOTETomek_0", "SMOTETomek_1", "KMeans_0", "KMeans_1"]: 
        results[exp_name] = {"per_seed": [], "mean":None, "std":None}
   
    for seed in tqdm_seeds:
        # Baseline
        base_recall, base_f1, base_precision = evaluate_model_baseline(X, y, random_seed=seed)
        results["Baseline"]["per_seed"].append((base_recall, base_f1, base_precision))

        # Undersampling methods
        ## 1. Random Undersampling
        rus_0_recall, rus_0_f1, rus_0_precision, rus_1_rec, rus_1_f1, rus_1_precision = evaluate_model_undersampling(X, y, strata_labels=strata_labels, random_seed=seed, undersampling_method='random')
        results["RUS_0"]["per_seed"].append((rus_0_recall, rus_0_f1, rus_0_precision))
        results["RUS_1"]["per_seed"].append((rus_1_rec, rus_1_f1, rus_1_precision))
        ## 2. Tomek Links
        tl_0_recall, tl_0_f1, tl_0_precision, tl_1_rec, tl_1_f1, tl_1_precision = evaluate_model_undersampling(X, y, strata_labels=strata_labels, random_seed=seed, undersampling_method='tomeklinks')
        results["TomekLinks_0"]["per_seed"].append((tl_0_recall, tl_0_f1, tl_0_precision))
        results["TomekLinks_1"]["per_seed"].append((tl_1_rec, tl_1_f1, tl_1_precision))
        ## 3. Edited Nearest Neighbors
        enn_0_recall, enn_0_f1, enn_0_precision, enn_1_rec, enn_1_f1, enn_1_precision = evaluate_model_undersampling(X, y, strata_labels=strata_labels, random_seed=seed, undersampling_method='edited_nearest')
        results["ENN_0"]["per_seed"].append((enn_0_recall, enn_0_f1, enn_0_precision))
        results["ENN_1"]["per_seed"].append((enn_1_rec, enn_1_f1, enn_1_precision))
        ## 4. NearMiss
        nm_0_recall, nm_0_f1, nm_0_precision, nm_1_rec, nm_1_f1, nm_1_precision = evaluate_model_undersampling(X, y, strata_labels=strata_labels, random_seed=seed, undersampling_method='nearmiss')
        results["NearMiss_0"]["per_seed"].append((nm_0_recall, nm_0_f1, nm_0_precision))
        results["NearMiss_1"]["per_seed"].append((nm_1_rec, nm_1_f1, nm_1_precision))
        ## 5. AllKNN
        aknn_0_recall, aknn_0_f1, aknn_0_precision, aknn_1_rec, aknn_1_f1, aknn_1_precision = evaluate_model_undersampling(X, y, strata_labels=strata_labels, random_seed=seed, undersampling_method='allknn')
        results["AllKNN_0"]["per_seed"].append((aknn_0_recall, aknn_0_f1, aknn_0_precision))
        results["AllKNN_1"]["per_seed"].append((aknn_1_rec, aknn_1_f1, aknn_1_precision))
        ## 6. Instance Hardness Threshold
        iht_0_recall, iht_0_f1, iht_0_precision, iht_1_rec, iht_1_f1, iht_1_precision = evaluate_model_undersampling(X, y, strata_labels=strata_labels, random_seed=seed, undersampling_method='instance_hardness')
        results["IHT_0"]["per_seed"].append((iht_0_recall, iht_0_f1, iht_0_precision))
        results["IHT_1"]["per_seed"].append((iht_1_rec, iht_1_f1, iht_1_precision))
        ## 7. Condensed Nearest Neighbor
        cnn_0_recall, cnn_0_f1, cnn_0_precision, cnn_1_rec, cnn_1_f1, cnn_1_precision = evaluate_model_undersampling(X, y, strata_labels=strata_labels, random_seed=seed, undersampling_method='condensed_nearest')
        results["CNN_0"]["per_seed"].append((cnn_0_recall, cnn_0_f1, cnn_0_precision))
        results["CNN_1"]["per_seed"].append((cnn_1_rec, cnn_1_f1, cnn_1_precision))
        ## 8. Neighbourhood Cleaning Rule
        ncr_0_recall, ncr_0_f1, ncr_0_precision, ncr_1_rec, ncr_1_f1, ncr_1_precision = evaluate_model_undersampling(X, y, strata_labels=strata_labels, random_seed=seed, undersampling_method='neighbourhood_cleaning_rule')
        results["NCR_0"]["per_seed"].append((ncr_0_recall, ncr_0_f1, ncr_0_precision))
        results["NCR_1"]["per_seed"].append((ncr_1_rec, ncr_1_f1, ncr_1_precision))
        ## 9. Repeated Edited Nearest Neighbors
        reen_0_recall, reen_0_f1, reen_0_precision, reen_1_rec, reen_1_f1, reen_1_precision = evaluate_model_undersampling(X, y, strata_labels=strata_labels, random_seed=seed, undersampling_method='repeated_edited')
        results["REEN_0"]["per_seed"].append((reen_0_recall, reen_0_f1, reen_0_precision))
        results["REEN_1"]["per_seed"].append((reen_1_rec, reen_1_f1, reen_1_precision))
        ## 10. One-Sided Selection
        oss_0_recall, oss_0_f1, oss_0_precision, oss_1_rec, oss_1_f1, oss_1_precision = evaluate_model_undersampling(X, y, strata_labels=strata_labels, random_seed=seed, undersampling_method='one_sided_selection')
        results["OSS_0"]["per_seed"].append((oss_0_recall, oss_0_f1, oss_0_precision))
        results["OSS_1"]["per_seed"].append((oss_1_rec, oss_1_f1, oss_1_precision))
        ## 11. Cluster Centroids
        cc_0_recall, cc_0_f1, cc_0_precision, cc_1_rec, cc_1_f1, cc_1_precision = evaluate_model_undersampling(X, y, strata_labels=strata_labels, random_seed=seed, undersampling_method='cluster_centroids')
        results["CC_0"]["per_seed"].append((cc_0_recall, cc_0_f1, cc_0_precision))
        results["CC_1"]["per_seed"].append((cc_1_rec, cc_1_f1, cc_1_precision))

     

        # Oversampling methods
        ## 1. Random Oversampling
        ros_0_recall, ros_0_f1, ros_0_precision, ros_1_rec, ros_1_f1, ros_1_precision = evaluate_model_oversampling(X, y, strata_labels=strata_labels, random_seed=seed, oversampling_method='random')
        results["ROS_0"]["per_seed"].append((ros_0_recall, ros_0_f1, ros_0_precision))
        results["ROS_1"]["per_seed"].append((ros_1_rec, ros_1_f1, ros_1_precision))
        ## 2. SMOTE
        smote_0_recall, smote_0_f1, smote_0_precision, smote_1_rec, smote_1_f1, smote_1_precision = evaluate_model_oversampling(X, y, strata_labels=strata_labels, random_seed=seed, oversampling_method='smote')
        results["SMOTE_0"]["per_seed"].append((smote_0_recall, smote_0_f1, smote_0_precision))
        results["SMOTE_1"]["per_seed"].append((smote_1_rec, smote_1_f1, smote_1_precision))
        ## 3. Borderline SMOTE
        bsmote_0_recall, bsmote_0_f1, bsmote_0_precision, bsmote_1_rec, bsmote_1_f1, bsmote_1_precision = evaluate_model_oversampling(X, y, strata_labels=strata_labels, random_seed=seed, oversampling_method='borderline_smote')
        results["BSMOTE_0"]["per_seed"].append((bsmote_0_recall, bsmote_0_f1, bsmote_0_precision))
        results["BSMOTE_1"]["per_seed"].append((bsmote_1_rec, bsmote_1_f1, bsmote_1_precision))

        # 6. SVM SMOTE
        svm_0_recall, svm_0_f1, svm_0_precision, svm_1_rec, svm_1_f1, svm_1_precision = evaluate_model_oversampling(X, y, strata_labels=strata_labels, random_seed=seed, oversampling_method='svm_smote')
        results["SVM_0"]["per_seed"].append((svm_0_recall, svm_0_f1, svm_0_precision))
        results["SVM_1"]["per_seed"].append((svm_1_rec, svm_1_f1, svm_1_precision))

        ## 9. SMOTEENN
        smoteenn_0_recall, smoteenn_0_f1, smoteenn_0_precision, smoteenn_1_rec, smoteenn_1_f1, smoteenn_1_precision = evaluate_model_oversampling(X, y, strata_labels=strata_labels, random_seed=seed, oversampling_method='smoteenn')
        results["SMOTEENN_0"]["per_seed"].append((smoteenn_0_recall, smoteenn_0_f1, smoteenn_0_precision))
        results["SMOTEENN_1"]["per_seed"].append((smoteenn_1_rec, smoteenn_1_f1, smoteenn_1_precision))
        ## 10. SMOTETomek
        smotetomek_0_recall, smotetomek_0_f1, smotetomek_0_precision, smotetomek_1_rec, smotetomek_1_f1, smotetomek_1_precision = evaluate_model_oversampling(X, y, strata_labels=strata_labels, random_seed=seed, oversampling_method='smotetomek')
        results["SMOTETomek_0"]["per_seed"].append((smotetomek_0_recall, smotetomek_0_f1, smotetomek_0_precision))
        results["SMOTETomek_1"]["per_seed"].append((smotetomek_1_rec, smotetomek_1_f1, smotetomek_1_precision))

        # MLCV
        mlcv_recall, mlcv_f1, mlcv_precision = evaluate_model_MLCV(X, y, bin_labels, seed)
        results["MLCV"]["per_seed"].append((mlcv_recall, mlcv_f1, mlcv_precision))

    for exp_name in results:
        per_seed_raw = np.array(results[exp_name]["per_seed"])
        # print(f" CHeck the per_seed_raw type: {type(per_seed_raw)}")
        # print(f" some values of per_seed_raw: {per_seed_raw[:5]}")
        # Replace None with np.nan BEFORE conversion
        per_seed_cleaned = [x if x is not None else np.nan for x in per_seed_raw]

        # Convert to float array
        per_seed = np.array(per_seed_cleaned, dtype=float)
        # print(f" CHeck the per_seed type: {type(per_seed)}")
        # print(f" some values of per_seed: {per_seed[:5]}")
        if np.all(np.isnan(per_seed)):
            # If all values are NaN, set mean and std to None    
            results[exp_name]["mean"] = np.array([np.nan, np.nan, np.nan])
            results[exp_name]["std"] = np.array([np.nan, np.nan, np.nan])
        else:
            results[exp_name]["mean"] = np.nanmean(per_seed, axis=0)
            results[exp_name]["std"] = np.nanstd(per_seed, axis=0) 
    return results


# TWELVE medical datasets:
df_names = [ 'BCWDD',  'Haberman', 'HeartCleveland', 'Hepatitis',
    'Hypothyroid', 'ILPD', 'KidneyDisease', 'NewThyroid1', 'NewThyroid2',
    'Pima', 'Thoracic', 'Vertebral'
]



# Create an empty list to store all results
all_results = []

# Process each dataset
for dataset_name in tqdm(df_names, desc="Processing Datasets"):
    print(f"Processing dataset: {dataset_name}")  # Track progress
    df_path = os.path.expanduser(f"~/.../inputs/{dataset_name}_processed.csv") # change the path


    # medical datasets:
    data = pd.read_csv(df_path)
    X, y = np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1])

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        

    # Run experiment
    results = perform_experiment(X, y)
    dataset_results = []
    dataset_res_stats = []
    # # Convert dictionary to structured DataFrame
    for exp_type, stats in results.items():
        mean_rec, mean_f1, mean_prec = stats["mean"]
        std_rec, std_f1, std_prec = stats["std"]
        for i, (rec, f1, prec) in enumerate(stats["per_seed"]):
            dataset_results.append([dataset_name, exp_type, i, rec, f1, prec])
        dataset_res_stats.append([dataset_name, exp_type, mean_rec, std_rec, mean_f1, std_f1, mean_prec, std_prec])

    # Convert to DataFrame
    df_dataset_results = pd.DataFrame(dataset_results, columns=["Dataset", "Experiment", "Iteration", "Recall", "F1", "Precision"])
    df_res_stats = pd.DataFrame(dataset_res_stats, columns=["Dataset", "Experiment","Mean_Recall", "Std_Recall", "Mean_F1", "Std_F1", "Mean_Precision", "Std_Precision"])   
    df_dataset_results.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
    df_res_stats.to_csv(RESULTS_FILE_STAT, mode='a', header=False, index=False)
    

print(f"Results saved to {RESULTS_FILE}")

