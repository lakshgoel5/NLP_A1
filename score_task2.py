import json
import argparse
from hungarian_eval import evaluate_clustering

def main():
    parser = argparse.ArgumentParser(description="Evaluate Task 2 Clustering Results")
    parser.add_argument("predictions_file", help="Path to predictions JSON file (list of cluster IDs)")
    parser.add_argument("ground_truth_file", help="Path to ground truth JSON file (list of author IDs)")
    args = parser.parse_args()

    # Load predictions
    print(f"Loading predictions from: {args.predictions_file}")
    with open(args.predictions_file, 'r') as f:
        predicted_labels = json.load(f)

    # Load ground truth
    print(f"Loading ground truth from: {args.ground_truth_file}")
    with open(args.ground_truth_file, 'r') as f:
        true_labels = json.load(f)

    # Verify lengths
    if len(predicted_labels) != len(true_labels):
        print(f"Error: Number of predictions ({len(predicted_labels)}) does not match number of ground truth labels ({len(true_labels)})")
        return

    # Determine number of clusters
    # Assuming labels are 0-indexed integers
    num_clusters = max(max(predicted_labels), max(true_labels)) + 1
    print(f"Detected {num_clusters} clusters/authors")

    # Evaluate
    print("Running Hungarian Algorithm evaluation...")
    accuracy, mapping = evaluate_clustering(predicted_labels, true_labels, num_clusters)

    print("-" * 30)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("-" * 30)
    print("Optimal Mapping (Predicted -> True):")
    for pred, true in sorted(mapping.items()):
        print(f"  Cluster {pred} -> Author {true}")
    print("-" * 30)

if __name__ == "__main__":
    main()
