import numpy as np
from scipy.optimize import linear_sum_assignment

def evaluate_clustering(predicted_labels, true_labels, num_clusters):
    """
    Evaluate clustering accuracy using the Hungarian algorithm.
    
    Args:
        predicted_labels: List of predicted cluster IDs (0 to k-1)
        true_labels: List of true author IDs (0 to k-1)
        num_clusters: Number of clusters (k)
        
    Returns:
        accuracy: Fraction of correctly assigned chunks under optimal mapping
        mapping: Dictionary mapping predicted cluster ID to true author ID
    """
    # Create confusion matrix
    confusion_matrix = np.zeros((num_clusters, num_clusters), dtype=int)
    
    for p, t in zip(predicted_labels, true_labels):
        if 0 <= p < num_clusters and 0 <= t < num_clusters:
            confusion_matrix[p][t] += 1
            
    # We want to maximize correct assignments, which is equivalent to 
    # finding a permutation that maximizes sum of diagonal elements after permutation.
    # linear_sum_assignment minimizes cost, so we pass negative confusion matrix.
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    
    # Calculate accuracy
    total_correct = confusion_matrix[row_ind, col_ind].sum()
    total_samples = len(predicted_labels)
    accuracy = total_correct / total_samples
    
    # Mapping from Predicted (row) -> True (col)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    
    return accuracy, mapping