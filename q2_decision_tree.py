import pandas as pd
import numpy as np
import os
from collections import Counter

# TO DO: Import any additional packages
 


# TO DO: implement the entropy calculation for a given Series.

def entropy(S):                 #  Entropy(S) = - Σ (p_i * log2(p_i)) 
    if len(S) == 0:             #S is like a subset in a tree so is use functions unique_values, counts
        return 0.0
    counter = Counter(S)

    # Extract unique values and their counts as separate lists
    unique_values = list(counter.keys())

    counts = list(counter.values())    
    counts = np.array(counts)

    probabilities = counts / len(S)                 #this is p_i 
    entropy_value = -np.sum(probabilities * np.log2(probabilities))    #- Σ (p_i * log2(p_i))                   

    return entropy_value


# To Do: implement miscalculation loss for a given Series.
def misclassification(S):    
    if len(S) == 0:
        return 0.0
    counter = Counter(S)

    # Extract unique values and their counts as separate lists
    unique_values = list(counter.keys())   #MisclassError = (# Misclassified Instances) / (Total #  Instances in  S)
    counts = list(counter.values())  
    max_class = max(counts)  

    misscalssificatio_error = 1.0 - (max_class/len(S))

    return misscalssificatio_error


# To Do: implement gini (impurity) loss for a given Series.
def gini(S):      #Gini(S) = 1 - Σ (p_i^2)

    if len(S) == 0:
        return 0.0
    counter = Counter(S)
    counts = list(counter.values())
    total_samp = len(S)
    gini =1.0 #i dont know if 1 is correct or it should be 0 ?

    for count in counts:
        probabilities = count / total_samp
        gini -= probabilities**2

    
    return gini


# To Do: Implement the information gain that takes a set of splits using
#        a single feature and given loss criterion to provide the 
#        information gain.
def calculate_gain(subset_splits, criterion="gini"):

    #Gain = Entropy(S) - Σ [(|S_v| / |S|) * Entropy(S_v)]

    total_samples = len(subset_splits[0][1])
    total_impurity = 0.0

    for split in subset_splits:
        subset = split[1]
        weight = len(subset) / total_samples

        if criterion == "gini":
            impurity = gini(subset)
        elif criterion == "entropy":        #This condition is needed becuse in finde best gain we change criteria
            impurity = entropy(subset)
        elif criterion == "misclassification":
            impurity = misclassification(subset)

        else:
            raise ValueError("Invalid criterion")
        
        total_impurity += weight * impurity

    org_impurity = gini(label)
    info_gain = org_impurity - total_impurity


    return info_gain


# To Do: Implement a function that splits a dataset by a specific
#        feature.
def dataset_split_by_feature(dataset, feature, label):
    splits = []

    unique_val = dataset[feature].unique()

    for val in unique_val:
        subset = dataset[dataset[feature] == val] #finde the subset of the data set where the fautre equals the value and append it to the split
        splits.append((val, subset, feature))

    return splits 


# To Do: Fill find_best_split(dataset, label, criterion) function
#        which will find the best split for a given dataset, label,
#        and criterion. It should out using the given string output
#        format.
def find_best_split(dataset, label, criterion="gini"):
    # for each split calculate gain, return best feature to split and the gain
    best_feature = best_gain = -1     #change to -1 to no not take 0 as feature

    for feature in dataset.columns:
        if feature  == label:
            continue
        #print(f"Checking feature: {feature}")

        subset_splits = dataset_split_by_feature(dataset, feature, label)
        info_gain = calculate_gain(subset_splits, criterion)

        #print(f"Info Gain for {feature}: {info_gain}")
        
        if info_gain > best_gain:
            best_gain = info_gain
            best_feature = feature

    #print(f"Best feature: {best_feature} with Gain: {best_gain:.2f}")
    return best_feature, best_gain


# If you used the starter code, you won't have to change any of the below
# code except for filling in your name and adjusting the criterion if you
# are in CS4361 and choose not to implement additional criterion.
if __name__ == '__main__':
    # read file
    name = "Doe_john"
    
    # Ensure the folder exists or create it if it doesn't
    if not os.path.exists(name):
        os.makedirs(name=name)
    
    data = pd.read_csv("agaricus-lepiota.csv")
    label = "class"
  
    # Find the best feature using suggested methods
    criteria = ["gini", "entropy", "misclassification"]
    f = open(f"{name}/mushrooms.txt", "w")
    for criterion in criteria:
        best_feature, best_gain = find_best_split(data, "class", criterion)
        f.write(f"Best feature: {best_feature} Using: {criterion}-Gain: {best_gain:.2f}\n")
    f.close()
