# https://ita-ml.gitlab.io/pyhard/pyhard.html#pyhard.measures.ClassificationMeasure
# All Classfication instance hardness in the above link with description
import os
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

#from pyhard.pyhard.measures import ClassificationMeasures

HardnessMetrics = ['feature_kDN', 'feature_DS', 'feature_DCP', 'feature_TD_P',
       'feature_TD_U', 'feature_CL', 'feature_CLD', 'feature_MV', 'feature_CB',
       'feature_N1', 'feature_N2', 'feature_LSC', 'feature_LSR',
       'feature_Harmfulness', 'feature_Usefulness', 'feature_F1', 'feature_F2',
       'feature_F3', 'feature_F4']

# Some classification hardness metrics:
    # "intra_extra_ratio":"feature_N2"
    # "k-Disagreeing Neighbors":"feature_kDN"
    # "ls_radius": "feature_LSR"
    # "tree_depth_pruned": "feature_TD"
    # "tree_depth_unpruned": "feature_TD_U"
    # "harmfulness": "feature_Harmfulness"
    # "borderline_points": "feature_N1"


def merge_groups(y, hardness, threshold=4):

    # print the y values and their count using counter
    print(f"Counter of y: {Counter(y)}") # >> Counter({0: 180, 1: 35})

    # print the hardness and their count using counter
    print(f"Counter of hardness: {Counter(hardness)}") # >> Counter({0: 57, 3: 54, 2: 52, 1: 52})

    # Create combined stratification label
    strata_labels = y + 2*hardness # >> Counter({4: 49, 2: 44, 6: 44, 0: 43, 1: 14, 7: 10, 3: 8, 5: 3})
    
    # print the strata labels and their count using counter
    print(f"Counter of strata_labels: {Counter(strata_labels)}") # the problem is we had a label of 5 and we don't accoutn for it in the merge_pairs.

    # Calculate the counts for each group
    unique_labels, counts = np.unique(strata_labels, return_counts=True)
    group_counts = dict(zip(unique_labels, counts))
    
    # Create a new label array for merged groups
    merged_labels = np.copy(strata_labels)
    
    # Define merge pairs based on your criteria
    merge_pairs = {0:2, 2:0, 4:6, 6:4, 1:3, 3:1, 5:7, 7:5}  #{0: 1, 1: 0, 2: 3, 3: 2}  # Each group points to the group it should merge with if below threshold
    merged_set = set()
    # Perform merging based on group size and predefined pairs
    for label, count in group_counts.items(): # take label 0 as example
        # Check if the group is already merged or if it meets the threshold
        if count >= threshold or label in merged_set:
            continue
        if label not in merge_pairs:
            print(f"Warning: No merge pair defined for label {label}.")
            continue

        pair = merge_pairs[label] # in our example: the pair of 0 is 2
        pair_count = group_counts.get(pair, 0) # in our example: we get the count of 2 in our dataset

        if count+pair_count >= threshold:

            if count < threshold and label not in merged_set: # in our example we check the pair count of 2 if it is larger than the threshold
                merged_labels[strata_labels == label] = pair # 
                merged_set.add (label)
            elif pair_count < threshold and pair not in merged_set:
                merged_labels[strata_labels == pair] = label
                merged_set.add(pair)
        elif count+pair_count < threshold:
            # This case is for when groups even merged together are still below threshold
            # we have a pattern that even labels like 0 <->2 and 4<->6 so if 0+2 are less than the threshold we can merge them together and then we merge them to the next 
            if label%2 == 0 : # >> we want 0 and 2 to be merged with 4 in case count of 1 and count of 2 are less than threshold , I imagine if 4 and 6 are less than threshold but this is impossible
                if pair - label  > 0:
                    dest_label = pair+2 if pair < 4 else pair - 4 #if pair>=4
                    merged_labels[strata_labels == label] = dest_label
                    merged_labels[strata_labels == pair] = dest_label
                    merged_set.add(label)
                    merged_set.add(pair)
                else:
                    dest_label = pair + 4 if pair < 4 else pair - 2 #if pair>=4
                    merged_labels[strata_labels == label] = dest_label
                    merged_labels[strata_labels == pair] = dest_label
                    merged_set.add(label)
                    merged_set.add(pair)
            elif label%2 == 1: # check the below
                if pair - label  > 0: # >> cases  3-1 and 7-5
                    dest_label = pair + 2 if pair < 4 else pair - 4 #if pair >=4 # >> 3-1 : 1, 3 -> 5 AND 7-5 : 5,7 ->3
                    merged_labels[strata_labels == label] = dest_label
                    merged_labels[strata_labels == pair] = dest_label
                    merged_set.add(label)
                    merged_set.add(pair)
                else: # >> cases 1-3 and 5-7
                    dest_label = pair + 4 if pair < 4 else pair - 2 #if pair >=4 # >> 1-3: 1, 3 -> 5 AND 5-7: 5, 7 -> 3
                    merged_labels[strata_labels == label] = dest_label
                    merged_labels[strata_labels == pair] = dest_label
                    merged_set.add(label)
                    merged_set.add(pair)

    return merged_labels



# Calculate and print the distribution ratios
def print_distribution(strata, total, title):
    unique, counts = np.unique(strata, return_counts=True)
    distribution = dict(zip(unique, counts / total))
    print(f"{title} distribution ratios:")
    for k, v in distribution.items():
        print(f"Strata {int(k)}: {v:.2f}")
        

from problexity import ComplexityCalculator
import matplotlib.pyplot as plt

def get_cmx(X,y, dataset_name):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    cmx = ComplexityCalculator().fit(X_s,y)
    score = cmx.score()
    #cmx_scores = cmx.complexity
    #metrics = cmx._metrics()
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(7,7))
    cmx.plot(fig, (1,1,1))
    fig.savefig(f"cmx_{dataset_name}.png")
    return score

df_names = ['BCWDD','Hypothyroid', 'ILPD']
for dataset_name in df_names:
    df_path = rf"C:\Users\BOUKA\Downloads\CBMs-Instance-Hard\IHA-Code\inputs\{dataset_name}_processed.csv"
    #data = fetch_datasets()[dataset_name]
    #X, y = data.data, data.target
    data = pd.read_csv(df_path)
    X, y = np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1])
    X.astype(np.float64)
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    #print(f"shape of y before reshape: {y.shape}")
    y_ = y.reshape(-1,1)
    score = get_cmx(X, y, dataset_name=dataset_name)
    print(f"cmx of {dataset_name} is {score}")


