import json
import torch
import torch.nn as nn
import numpy as np
import os
import math
from model import Word2Vec
import argparse
from collections import Counter
from k_means_constrained import KMeansConstrained
import time

def get_representation(text, model):
    # Read text and get tokens
    author_texts = {"temp_id": text}
    author_tokens_dict, _ = model.pre_process(author_texts)
    
    # Extract the actual tokens list from the dictionary
    tokens = author_tokens_dict["temp_id"]

    stop_tokens = {',', '.', 'the', 'and', 'of', 'to', 'a', 'in', 'i', 
                'was', 'that', 'it', '"', '-', "'", 'he', 'you', 'she',
                'his', 'had', 'her', 'with', 'for', 'as', 'at', 'is',
                ';', 'on', 'but', 's'}

    tokens = [t for t in tokens if t not in stop_tokens]
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

    representation = torch.mean(embeddings, dim=0)
    
    return representation.detach().cpu().numpy()

def expand_vocabulary(model, all_tokens, min_freq=0):
    print("EXPANDING VOCABULARY FOR NEW WORDS")
    word_counts = Counter(all_tokens)
    new_words = []
    
    for word, count in word_counts.items():
        if word not in model.word2idx and word not in ["<UNK>", "<PAD>"] and count > min_freq:
            new_words.append((word, count))
    
    if len(new_words) == 0:
        print("No new words to add (all words in vocab)")
        return model
    
    # Sort by frequency
    new_words.sort(key=lambda x: x[1], reverse=True)
    print(f"Found {len(new_words)} new words (freq > {min_freq})")
    print(f"Top 10: {[w for w, c in new_words[:10]]}")
    
    old_vocab_size = model.vocab_size
    
    # Add to vocabulary
    for word, count in new_words:
        model.word2idx[word] = model.vocab_size
        model.idx2word[model.vocab_size] = word
        model.vocab_size += 1
    
    new_vocab_size = model.vocab_size
    
    # Expand embedding matrices
    old_W_in = model.W_in
    old_W_out = model.W_out
    
    model.W_in = nn.Embedding(new_vocab_size, model.embedding_dim).to(model.device)
    model.W_out = nn.Embedding(new_vocab_size, model.embedding_dim).to(model.device)
    
    # Copy old embeddings
    with torch.no_grad():
        model.W_in.weight.data[:old_vocab_size] = old_W_in.weight.data
        model.W_out.weight.data[:old_vocab_size] = old_W_out.weight.data
        
        # Initialize new embeddings (same range as originals)
        init_range = 0.5 / math.sqrt(model.embedding_dim)
        model.W_in.weight.data[old_vocab_size:].uniform_(-init_range, init_range)
        model.W_out.weight.data[old_vocab_size:].uniform_(-init_range, init_range)
    
    # Expand vocab_counts
    old_counts = torch.tensor(model.vocab_counts, dtype=torch.long, device=model.device)
    model.vocab_counts = torch.zeros(new_vocab_size, dtype=torch.long, device=model.device)
    model.vocab_counts[:old_vocab_size] = old_counts
    
    for word, count in new_words:
        idx = model.word2idx[word]
        model.vocab_counts[idx] = count
    
    # Update unigram distribution
    model.unigram_dist = model.vocab_counts.pow(0.75)
    model.unigram_dist /= model.unigram_dist.sum()
    
    print(f"Vocab: {old_vocab_size} to {new_vocab_size} (+{len(new_words)})")
    
    return model

def fine_tune(model, chunks, max_epochs=50, max_time=25*60):
    print(f"Starting fine-tuning on {len(chunks)} chunks...")
    start_time = time.time()
    
    # Create a temporary author mapping for chunks
    chunk_texts = {f"chunk_{i}": chunk for i, chunk in enumerate(chunks)}
    
    # Preprocess and prepare for training
    model.author_tokens, all_tokens = model.pre_process(chunk_texts)
    
    model = expand_vocabulary(model, all_tokens, min_freq=0)
    
    model.epochs = max_epochs
    model.max_training_time = max_time
    
    print(f"Fine-tuning for max {max_epochs} epochs or {max_time/60:.1f} minutes...")
    model.train_skipgram()

    elapsed = time.time() - start_time
    print(f"Fine-tuning completed in {elapsed:.2f}s")
    
    return model

def cluster_chunks(chunk_embeddings, num_authors, min_chunks_per_author):
    # Normalize embeddings
    norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized_embeddings = chunk_embeddings / norms
    
    # Constrained K-means clustering
    clf = KMeansConstrained(n_clusters=num_authors, size_min=min_chunks_per_author, n_init=10, max_iter=300)

    labels = clf.fit_predict(normalized_embeddings)
    
    return labels

def main():
    # Arguments: test_file output_dir
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    
    total_start_time = time.time()
    
    # Load pre-trained model
    print("Loading pre-trained model...")
    model = Word2Vec(100)
    model.load("./models_bottom5")
    model.train()
    
    # Read queries
    with open(args.test_file, "r") as f:
        data = json.load(f)
    
    num_authors = data["num_authors"]
    min_chunks_per_author = data["min_chunks_per_author"]
    chunks = data["chunks"]
    
    model = fine_tune(model, chunks, max_epochs=40, max_time=25*60)

    model.eval()

    chunk_embeddings = []
    for i, chunk in enumerate(chunks):
        emb = get_representation(chunk, model)
        chunk_embeddings.append(emb)
    
    chunk_embeddings = np.array(chunk_embeddings)

    labels = cluster_chunks(chunk_embeddings, num_authors, min_chunks_per_author)

    results = []
    for label in labels:
        results.append(label)
    
    if args.output_dir.endswith("/") or os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "task2_predictions.json")
    else:
        parent = os.path.dirname(args.output_dir)
        if parent:
            os.makedirs(parent, exist_ok=True)
        output_path = args.output_dir
    
    with open(output_path, "w") as f:
        json.dump([int(x) for x in results], f)
    
    total_elapsed = time.time() - total_start_time
    print(f"\nTask 2 completed in {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print(f"Results saved to {output_path}")
    end_time = time.time()
    elapsed = (end_time - start_time)/60
    print(f"Total time taken: {elapsed:.2f} min")

if __name__ == "__main__":
    main()


