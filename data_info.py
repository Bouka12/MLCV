import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import  LabelEncoder


df_names =  [
    'BCWDD', 'CervicalCancer', 'Haberman', 'HeartCleveland', 'Hepatitis',
    'Hypothyroid', 'ILPD', 'KidneyDisease', 'NewThyroid1', 'NewThyroid2',
    'Pima', 'Thoracic', 'Vertebral'
]


# Create an empty list to store all results
all_results = []

# Process each dataset
for dataset_name in tqdm(df_names, desc="Processing Datasets"):
    print(f"Processing dataset: {dataset_name}")  # Track progress
    df_path = rf"...\inputs\{dataset_name}_processed.csv" # -> change the path

    # medical datasets:
    data = pd.read_csv(df_path)
    X, y = np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1])


    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Count n_min and n_maj to calculate IR:
    n = Counter(y)
    print(f"count of y for {dataset_name} is {n}")
    # Get counts
    majority_class = n.most_common(1)[0]
    minority_class = n.most_common()[-1]

    majority_count = majority_class[1]
    minority_count = minority_class[1]

    # Imbalance ratio: majority / minority
    imbalance_ratio = round(minority_count / majority_count, 2)

    # Output
    print(f"Majority class ({majority_class[0]}): {majority_count}")
    print(f"Minority class ({minority_class[0]}): {minority_count}")
    print(f"Imbalance ratio (majority/minority): {imbalance_ratio}")


    print(f"|{dataset_name}|#Instances {X.shape[0]}| #Features {X.shape[1]}| IR(N_min/N_maj) {imbalance_ratio}")
