import os
import json
import random
import re

def clean_text(text):
    # Try to find the start and end of the Gutenberg text
    start_match = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
    if start_match:
        text = text[start_match.end():]
    
    end_match = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", text)
    if end_match:
        text = text[:end_match.start()]
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_chunks(text, min_words=50, max_words=500):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        # Pick a random size for variety between min and max
        size = random.randint(min_words, max_words)
        chunk_words = words[i : i + size]
        if len(chunk_words) >= min_words:
            chunks.append(" ".join(chunk_words))
        i += size
    return chunks

def main():
    novels_dir = "sample_inputs/web_data"
    files = [f for f in os.listdir(novels_dir) if f.endswith(".txt")]
    
    book_chunks = {}
    for filename in files:
        path = os.path.join(novels_dir, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            cleaned = clean_text(text)
            # Aim for chunks between 100-300 words for better balance
            chunks = get_chunks(cleaned, min_words=50, max_words=500)
            book_chunks[filename] = chunks
            print(f"Loaded {len(chunks)} chunks from {filename}")

    queries = []
    num_queries = 500
    
    # Authors list for easier indexing
    authors = list(book_chunks.keys())
    
    for i in range(num_queries):
        # Pick a target author
        target_author = random.choice(authors)
        while len(book_chunks[target_author]) < 5: 
            target_author = random.choice(authors)
        
        # Pick 2 chunks for query and correct candidate
        author_pool = book_chunks[target_author]
        idx1, idx2 = random.sample(range(len(author_pool)), 2)
        query_text = author_pool[idx1]
        correct_text = author_pool[idx2]
        
        # Pick 9 distractor authors/chunks
        distractors = []
        while len(distractors) < 9:
            other_author = random.choice(authors)
            # Try to pick from different authors mostly
            if other_author == target_author and random.random() > 0.05:
                continue 
            
            other_pool = book_chunks[other_author]
            dist_chunk = random.choice(other_pool)
            if dist_chunk not in [query_text, correct_text] and dist_chunk not in distractors:
                distractors.append(dist_chunk)
        
        # Assemble candidates
        candidates_list = distractors + [correct_text]
        random.shuffle(candidates_list)
        
        candidates_dict = {}
        correct_label = ""
        for j, cand in enumerate(candidates_list):
            label = f"cand_{j+1}"
            candidates_dict[label] = cand
            if cand == correct_text:
                correct_label = label
        
        queries.append({
            "query_id": f"gen_large_{i+1}",
            "query_text": query_text,
            "candidates": candidates_dict,
            "correct_candidate": correct_label
        })

    output_path = "sample_inputs/gen_task1_6.json"
    with open(output_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False makes the output readable (no \u characters)
        json.dump(queries, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully generated {len(queries)} test cases in {output_path}")

if __name__ == "__main__":
    main()
