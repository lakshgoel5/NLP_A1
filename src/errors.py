# analyze_errors.py
import json
from sample import SimpleWord2Vec
import numpy as np
import torch

def get_representation(text, model):
    """Remove stopwords strategy"""
    author_texts = {"temp_id": text}
    author_tokens_dict, _ = model.pre_process(author_texts)
    tokens = author_tokens_dict["temp_id"]
    
    stop_tokens = {',', '.', 'the', 'and', 'of', 'to', 'a', 'in', 'i', 
                   'was', 'that', 'it', '"', '-', "'", 'he', 'you', 'she',
                   'his', 'had', 'her', 'with', 'for', 'as', 'at', 'is',
                   ';', 'on', 'but', 's'}
    
    tokens = [t for t in tokens if t not in stop_tokens]
    
    indices = [model.word2idx.get(t, model.word2idx.get("<UNK>", 0)) for t in tokens]
    if not indices:
        return np.zeros(model.embedding_dim), tokens
    
    token_indices = torch.LongTensor(indices).to(model.device)
    
    embeddings_in = model.W_in(token_indices)
    embeddings_out = model.W_out(token_indices)
    embeddings = 0.5 * (embeddings_in + embeddings_out)
    
    representation = torch.mean(embeddings, dim=0)
    return representation.detach().cpu().numpy(), tokens

def analyze_errors():
    model = SimpleWord2Vec(embedding_dim=100)
    model.load("../models_baseline")
    model.eval()
    
    with open("../sample_inputs/gen_task1.json", "r") as f:
        data = json.load(f)
    
    error_queries = ['q_9', 'q_15', 'q_21', 'q_51', 'q_58', 'q_61', 'q_65', 'q_66', 'q_79', 'q_87', 'q_91', 'q_93']
    
    print("🔍 ANALYZING THE 11 ERRORS")
    print("="*60)
    
    for item in data:
        if item["query_id"] not in error_queries:
            continue
        
        query_id = item["query_id"]
        query_text = item["query_text"]
        candidates_dict = item["candidates"]
        correct_candidate = item["correct_candidate"]
        
        print(f"\n❌ {query_id}: Correct = {correct_candidate}")
        print(f"   Query length: {len(query_text)} chars")
        
        query_repr, query_tokens = get_representation(query_text, model)
        print(f"   Query tokens (after stopword removal): {len(query_tokens)}")
        
        similarities = {}
        for author_id, cand_text in candidates_dict.items():
            cand_repr, cand_tokens = get_representation(cand_text, model)
            
            norm_query = np.linalg.norm(query_repr)
            norm_cand = np.linalg.norm(cand_repr)
            
            if norm_query == 0 or norm_cand == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query_repr, cand_repr) / (norm_query * norm_cand)
            
            similarities[author_id] = similarity
        
        ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        print(f"   Ranking:")
        for rank, (author, sim) in enumerate(ranked[:3], 1):
            marker = "✓" if author == correct_candidate else " "
            print(f"   {rank}. {author}: {sim:.4f} {marker}")

if __name__ == "__main__":
    analyze_errors()