# diagnostic.py
import os
import re
from collections import Counter
import json

def analyze_training_data(train_dir):
    """Analyze the training data to understand what we're dealing with"""
    
    author_texts = {}
    files = [f for f in os.listdir(train_dir) if f.endswith('.txt')]
    
    for filename in files:
        file_path = os.path.join(train_dir, filename)
        author_id = filename.split(".")[0]
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        author_texts[author_id] = text
    
    print(f"📊 TRAINING DATA ANALYSIS")
    print(f"=" * 60)
    print(f"Total files: {len(author_texts)}")
    print()
    
    # Analyze each file
    all_tokens = []
    for author_id, text in author_texts.items():
        # Basic tokenization
        text_lower = text.lower()
        tokens = re.findall(r'\w+|[^\w\s]', text_lower)
        
        print(f"Author: {author_id}")
        print(f"  - Characters: {len(text):,}")
        print(f"  - Tokens: {len(tokens):,}")
        print(f"  - Unique tokens: {len(set(tokens)):,}")
        print(f"  - First 100 chars: {text[:100]}")
        print()
        
        all_tokens.extend(tokens)
    
    # Overall stats
    word_counts = Counter(all_tokens)
    print(f"\n📈 OVERALL STATISTICS")
    print(f"=" * 60)
    print(f"Total tokens: {len(all_tokens):,}")
    print(f"Unique tokens: {len(word_counts):,}")
    print(f"Vocabulary size (min_freq=1): {len(word_counts):,}")
    print(f"Vocabulary size (min_freq=5): {len([w for w, c in word_counts.items() if c >= 5]):,}")
    print()
    
    # Top 30 most common words
    print(f"🔤 TOP 30 MOST COMMON TOKENS")
    print(f"=" * 60)
    for word, count in word_counts.most_common(30):
        freq = count / len(all_tokens)
        print(f"{word:20s} : {count:6,} ({freq:.4f})")
    
    # Check for noise indicators
    print(f"\n⚠️  NOISE INDICATORS")
    print(f"=" * 60)
    noise_words = ['price', 'postpaid', 'cents', 'dollar', 'publishers', 'broadway', 'catalogue']
    for word in noise_words:
        if word in word_counts:
            print(f"'{word}' appears {word_counts[word]:,} times - POSSIBLE CATALOG TEXT")
    
    return author_texts, all_tokens, word_counts

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python diagnostic.py <train_directory>")
        sys.exit(1)
    
    train_dir = sys.argv[1]
    analyze_training_data(train_dir)