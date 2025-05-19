# Plot Hardness distribution for the selected datasets
# Used hardness metrics to measure instance hardness: LSC (Local Set Cardinality) and kDN (k-Disagreeing Neighbors)

#---------
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import random 
from pyhard.measures import ClassificationMeasures

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate random seeds dynamically
N_ITERATIONS = 10
RANDOM_SEEDS = [random.randint(0, 1_000_000) for _ in range(N_ITERATIONS)]
print(RANDOM_SEEDS)

# Paths and dataset names
DATA_DIR = './inputs'
DATASETS = ['BCWDD',  'Haberman', 'HeartCleveland', 'Hepatitis', 'Hypothyroid', 'ILPD', 'KidneyDisease', 'NewThyroid1', 'NewThyroid2', 'Pima', 'Thoracic', 'Vertebral']

results_df = pd.DataFrame()

df_names = ['BCWDD', 'Hypothyroid', 'ILPD']

for dataset_name in df_names:
    print(f"Processing dataset: {dataset_name}")  # Print message to track which dataset is being processed

    df_path = rf"...\inputs\{dataset_name}_processed.csv" # -> change the path

    data = pd.read_csv(df_path)
    X, y = np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1])
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    y_ = y.reshape(-1,1)
    data = np.concatenate([X, y_],axis=1)
    df = pd.DataFrame(data)
    print(f"df head: \n {df.head()}")
    n_features = X.shape[1]
    column_names = [f"feature_{i}" for i in range(n_features)] + ["target"]
    df.columns = column_names
    X_min = X[y==1]
    X_maj = X[y==0]

    print(f"X_min, X_maj : {X_min.shape}, {X_maj.shape}")

    # # Normalize to [0,1] using MinMaxScaler
    scaler = MinMaxScaler()
    HM = ClassificationMeasures(df)
    data_HM = HM.calculate_all()

    # HardnessMetrics = ['feature_kDN', 'feature_DS', 'feature_DCP', 'feature_TD_P',
    #    'feature_TD_U', 'feature_CL', 'feature_CLD', 'feature_MV', 'feature_CB',
    #    'feature_N1', 'feature_N2', 'feature_LSC', 'feature_LSR',
    #    'feature_Harmfulness', 'feature_Usefulness', 'feature_F1', 'feature_F2',
    #    'feature_F3', 'feature_F4']
    hm_lsc = np.array(data_HM["feature_LSC"]) # change here the hardness metrics kDN
    # print(f"hm_lsc: {hm_lsc[0:10]}")

    hm_kdn = np.array(data_HM['feature_kDN'])

    norm_hm_lsc = scaler.fit_transform(hm_lsc.reshape(-1,1)).flatten()
    hm_lsc_min = norm_hm_lsc[y==1]
    hm_lsc_maj = norm_hm_lsc[y==0]

    #
    norm_hm_kdn = scaler.fit_transform(hm_kdn.reshape(-1,1)).flatten()
    hm_kdn_min = norm_hm_kdn[y==1]
    hm_kdn_maj = norm_hm_kdn[y==0]

    # violin plot with sns
    # sns.set(style="whitegrid")
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'

    # Custom colors
    color_min = 'blue'  #"1f77b4"  # blue
    color_maj = 'darkorange'    #"ff7f0e"  # orange

    # Dictionary of the selected hardness metrics: "LSC" and "kDN"
    hardness_metrics = {
        "LSC": (hm_lsc_min, hm_lsc_maj),
        "kDN": (hm_kdn_min, hm_kdn_maj),
    }

    # Prepare subplot grid
    fig, axes = plt.subplots(2, 1, figsize=(8, 12), sharey=True)
    # fig.suptitle(f"Instance Hardness of {dataset_name}", fontsize=16, fontweight="bold")
    fig.tight_layout(pad=3.0, rect=[0, 0, 1, 0.97])  # Leave space on top for legend

    # Plot each hardness metric
    for ax, (name, (min_vals, maj_vals)) in zip(axes.flat, hardness_metrics.items()):
        # Combine into a single dataframe
        df = pd.DataFrame({
            "Hardness": np.concatenate([min_vals, maj_vals]),
            "Class": ["Minority"] * len(min_vals) + ["Majority"] * len(maj_vals)
        })
        sns.violinplot(data=df, x="Class", y="Hardness", ax=ax, hue="Class",
                    palette={"Minority": color_min, "Majority": color_maj},
                    inner="quartile")
        ax.set_title(name)
        ax.set_xlabel("")  # Hide x-labels per subplot
        ax.set_ylabel("")  # Hide y-labels per subplot

    # Save the figure
    plt.savefig(f"{dataset_name}_hardness_distribution.png")
    plt.show()
