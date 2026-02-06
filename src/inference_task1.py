import json
import torch
import numpy as np
import os
from model import Word2Vec
import argparse
from collections import Counter

def get_representation(text, model):
    # Read text and get tokens
    author_texts = {"temp_id": text}
    author_tokens_dict, _ = model.pre_process(author_texts)
    
    # Extract the actual tokens list from the dictionary
    tokens = author_tokens_dict["temp_id"]
    
    # Convert tokens to indices
    indices = [model.word2idx.get(t, model.word2idx.get("<UNK>", 0)) for t in tokens]
    
    if not indices:
        return np.zeros(model.embedding_dim)
        
    token_indices = torch.LongTensor(indices).to(model.device)
            
    # Get embeddings
    # For Skip-gram, using both input and output embeddings often improves quality.
    embeddings_in = model.W_in(token_indices)
    embeddings_out = model.W_out(token_indices)
    embeddings = 0.5 * (embeddings_in + embeddings_out)

    # # Implementing SIF

    # #Counter for words in the corpus
    # word_counts = Counter(tokens)
    # total_words = len(tokens)

    # # Weight(w) = a / (a + p(w))
    # a = 0
    # weights = []
    # for token in tokens:
    #     weights.append(a / (a + word_counts[token] / total_words))

    # # Normalise weights to 1 by summing and dividing
    # weights = torch.tensor(weights).to(model.device)
    # weights = weights / torch.sum(weights)

    # # embeddings * weights
    # weighted_embeddings = embeddings * weights.unsqueeze(1)
    
    # Average/Sum embeddings
    representation = torch.sum(embeddings, dim=0)
    
    return representation.detach().cpu().numpy()

def task1(query_text, candidates_dict, model):
    # Representation of query_text
    query_repr = get_representation(query_text, model)
    similarities = {}
    
    for author_id, cand_text in candidates_dict.items():
        cand_repr = get_representation(cand_text, model)

        # Cosine similarity
        norm_query = np.linalg.norm(query_repr)
        norm_cand = np.linalg.norm(cand_repr)
        
        if norm_query == 0 or norm_cand == 0:
            similarities[author_id] = 0.0
        else:
            similarity = np.dot(query_repr, cand_repr) / (norm_query * norm_cand)
            similarities[author_id] = similarity

    ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # print(ranked)
    return [author_id for author_id, _ in ranked]

def main():
    # Arguments test_file output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    
    # Load model
    model = Word2Vec(300, "sg")
    model.load("./models")
    model.eval()

    correct = 0
    reciprocal_rank_sum = 0.0
    incorrect = []

    # Read queries
    with open(args.test_file, "r") as f:
        data = json.load(f)

    results = []
    for item in data:
        query_id = item["query_id"]
        query_text = item["query_text"]
        candidates_dict = item["candidates"]
        correct_candidate = item["correct_candidate"]

        ranked_candidates = task1(query_text, candidates_dict, model)
        if(ranked_candidates[0] == correct_candidate):
            correct = correct+1
        else:
            incorrect.append(query_id)

        if correct_candidate in ranked_candidates:
            rank = ranked_candidates.index(correct_candidate) + 1
            reciprocal_rank_sum += 1.0 / rank
        
        results.append({
            "query_id": query_id, 
            "ranked_candidates": ranked_candidates
        })
    
    # We should save task1_predictions.json inside that directory
    if os.path.isdir(args.output_dir):
        output_path = os.path.join(args.output_dir, "task1_predictions.json")
    else:
        # If it's not a directory, treat as base path and create parent if needed
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        output_path = args.output_dir
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    total_queries = len(data)
    mrr = reciprocal_rank_sum / max(1, total_queries)

    print(f"Task 1 completed. Results saved to {output_path}")
    print(f"Top-1 Accuracy: {correct}/{total_queries}")
    print(f"MRR: {mrr:.4f}")
    print(incorrect)

if __name__ == "__main__":
    main()
